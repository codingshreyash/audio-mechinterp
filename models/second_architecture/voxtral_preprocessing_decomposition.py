"""Fit-only causal preprocessing decomposition for exact Piper red/blue PCM.

Default mode is CPU-only preflight. --run explicitly loads the pinned local
Voxtral checkpoint onto MPS for six predeclared feature-input conditions.
Hybrid mel inputs are off-manifold; this is not a security finding.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voxtral_representation_spread import (
    CONNECTOR_ROWS, MEL_FRAMES,
    l2_summary, partition_indices, prepare_pair,
)
from voxtral_smoke import (
    HERE, MAX_NEW_TOKENS, MEMORY_LIMIT, MemorySampler, SNAPSHOT,
    sha256_file,
)


OUT_DIR = HERE / "runs" / "voxtral_preprocessing_decomposition_fit_v1"
SAMPLE_RATE = 16_000
PADDED_SAMPLES = 480_000
N_FFT = 400
HOP = 160
EXPECTED_SOURCE_SHA256 = {
    "modeling_voxtral": "2d3ec9d78c469b77c42a8d5ba258185fb1c58f4c2c7bf6007ec724436dfc7056",
    "processing_voxtral": "239e84657d1a0287ab27a540c946a787a69b4b0bf45727d6b24bee21ccec0296",
    "feature_extraction_whisper": "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb",
}
CONDITION_ORDER = (
    "word_red_outside_red",     # native red features
    "word_blue_outside_blue",   # native blue features
    "word_red_outside_blue",    # crossed feature hybrid
    "word_blue_outside_red",    # crossed feature hybrid
    "red_with_blue_padding",    # padding-only swap
    "blue_with_red_padding",    # padding-only swap
)


def source_hashes() -> dict:
    from transformers.models.voxtral import modeling_voxtral, processing_voxtral
    from transformers.models.whisper import feature_extraction_whisper

    modules = {
        "modeling_voxtral": modeling_voxtral,
        "processing_voxtral": processing_voxtral,
        "feature_extraction_whisper": feature_extraction_whisper,
    }
    hashes = {name: sha256_file(Path(inspect.getfile(module))) for name, module in modules.items()}
    if hashes != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Installed Transformers source differs from pinned tag: {hashes}")
    return hashes


def independent_raw_log_mel(wav: Path, mel_filters: np.ndarray) -> tuple[torch.Tensor, float]:
    """Recreate pinned Whisper torch frontend before its clip-max floor.

    This intentionally uses the documented STFT/math, not the processor's
    private `_torch_extract_fbank_features` function. Equality to its output
    is checked separately in preflight.
    """
    pcm, sample_rate = sf.read(wav, dtype="float32")
    if sample_rate != SAMPLE_RATE or pcm.ndim != 1 or len(pcm) > PADDED_SAMPLES:
        raise ValueError(f"Unexpected PCM for {wav}")
    padded = np.pad(pcm, (0, PADDED_SAMPLES - len(pcm)), mode="constant")
    waveform = torch.from_numpy(padded).unsqueeze(0)
    window = torch.hann_window(N_FFT)
    spectrum = torch.stft(
        waveform, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True,
    )
    power = (spectrum[..., :-1].abs() ** 2).contiguous()
    filters = torch.from_numpy(np.asarray(mel_filters)).to(dtype=torch.float32)
    raw = torch.clamp(filters.T @ power, min=1e-10).log10()[0]
    if tuple(raw.shape) != (128, MEL_FRAMES):
        raise ValueError(f"Unexpected independent mel shape: {tuple(raw.shape)}")
    return raw, float(raw.max())


def normalize_raw(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    return (torch.maximum(raw, torch.tensor(maximum - 8.0, dtype=raw.dtype)) + 4.0) / 4.0


def feature_frame_l2(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    if tuple(a.shape) != (1, 128, MEL_FRAMES) or a.shape != b.shape:
        raise ValueError("Expected paired [1,128,3000] features")
    return torch.linalg.vector_norm((b - a).float()[0], dim=0).tolist()


def feature_sha256(features: torch.Tensor) -> str:
    return hashlib.sha256(features.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_conditions(features: dict[str, torch.Tensor], word_frames: list[int], padding_from: int) -> dict:
    red = features["red"]
    blue = features["blue"]
    if red.shape != blue.shape or tuple(red.shape) != (1, 128, MEL_FRAMES):
        raise ValueError("Unexpected feature shape")
    if not word_frames or max(word_frames) >= padding_from:
        raise ValueError("Invalid word-window or padding boundary")
    masks = torch.zeros(MEL_FRAMES, dtype=torch.bool)
    masks[word_frames] = True
    conditions = {}
    for word in ("red", "blue"):
        for outside in ("red", "blue"):
            name = f"word_{word}_outside_{outside}"
            hybrid = features[outside].clone()
            hybrid[:, :, masks] = features[word][:, :, masks]
            if not torch.equal(hybrid[:, :, masks], features[word][:, :, masks]):
                raise AssertionError("Word factor not applied")
            if not torch.equal(hybrid[:, :, ~masks], features[outside][:, :, ~masks]):
                raise AssertionError("Outside factor not applied")
            conditions[name] = {
                "features": hybrid, "word_source": word, "outside_source": outside,
                "padding_source": outside, "off_manifold": word != outside,
            }
    for base, other in (("red", "blue"), ("blue", "red")):
        name = f"{base}_with_{other}_padding"
        swapped = features[base].clone()
        swapped[:, :, padding_from:] = features[other][:, :, padding_from:]
        if not torch.equal(swapped[:, :, :padding_from], features[base][:, :, :padding_from]):
            raise AssertionError("Padding swap altered real-audio frames")
        conditions[name] = {
            "features": swapped, "word_source": base, "outside_source": base,
            "padding_source": other, "off_manifold": True,
        }
    if set(conditions) != set(CONDITION_ORDER):
        raise AssertionError("Unexpected condition set")
    if not torch.equal(conditions["word_red_outside_red"]["features"], red):
        raise AssertionError("Native red endpoint changed")
    if not torch.equal(conditions["word_blue_outside_blue"]["features"], blue):
        raise AssertionError("Native blue endpoint changed")
    return conditions


def prepare() -> tuple[dict, object, dict, dict]:
    report, pair = prepare_pair()  # pinned snapshot/manifest, PCM and processor checks
    sources = source_hashes()
    red_processor, red_batch = pair["red"]
    _, blue_batch = pair["blue"]
    raw = {}
    maxima = {}
    independent_error = {}
    for word, (_, batch) in pair.items():
        wav = Path(report["stimuli"][word]["wav"])
        raw[word], maxima[word] = independent_raw_log_mel(wav, red_processor.feature_extractor.mel_filters)
        independently_normalized = normalize_raw(raw[word], maxima[word])
        independent_error[word] = float((independently_normalized - batch["input_features"][0]).abs().max())
        if independent_error[word] > 1e-6:
            raise ValueError(f"Independent Whisper mel mismatch for {word}: {independent_error[word]}")

    word_frames = report["shape_math"]["word_mel_frames"]
    padding_from = int(report["shape_math"]["nominal_feature_padding_from"])
    shared_max = max(maxima.values())
    shared_red = normalize_raw(raw["red"], shared_max).unsqueeze(0)
    shared_blue = normalize_raw(raw["blue"], shared_max).unsqueeze(0)
    raw_l2 = feature_frame_l2(raw["red"].unsqueeze(0), raw["blue"].unsqueeze(0))
    shared_l2 = feature_frame_l2(shared_red, shared_blue)
    outside = [i for i in range(MEL_FRAMES) if i not in set(word_frames)]
    max_outside_raw = max(raw_l2[i] for i in outside)
    max_outside_shared = max(shared_l2[i] for i in outside)
    if max_outside_raw > 1e-6:
        raise ValueError(f"Raw log-mel differs outside changed word: {max_outside_raw}")
    if max_outside_shared > 1e-6:
        raise ValueError(f"Shared-max normalization did not equalize off-word frames: {max_outside_shared}")

    features = {word: pair[word][1]["input_features"].clone() for word in ("red", "blue")}
    conditions = make_conditions(features, word_frames, padding_from)
    feature_groups = partition_indices(MEL_FRAMES, word_frames, padding_from)
    report.update({
        "assay": "causal_preprocessing_decomposition",
        "script_sha256": sha256_file(Path(__file__)),
        "installed_transformers_source_sha256": sources,
        "source_hash_reference": "Transformers v5.20.2 tag commit 18329f1065382e54db698e242cf98bd3f3e5a766",
        "preprocessing_decomposition": {
            "raw_log_mel_maximum": maxima,
            "processor_reproduction_max_abs_error": independent_error,
            "raw_off_word_max_frame_l2": max_outside_raw,
            "raw_frame_l2": raw_l2,
            "shared_raw_log_mel_maximum": shared_max,
            "shared_max_off_word_max_frame_l2": max_outside_shared,
            "shared_max_frame_l2": shared_l2,
            "native_frame_l2": feature_frame_l2(features["red"], features["blue"]),
            "shared_max_partition_summary": l2_summary(shared_l2, feature_groups),
            "condition_order": list(CONDITION_ORDER),
            "conditions": {name: {**{k: v for k, v in condition.items() if k != "features"},
                                  "feature_sha256": feature_sha256(condition["features"])}
                           for name, condition in conditions.items()},
            "off_manifold_note": "Crossed and padding-swap mel tensors may have discontinuities absent from natural audio.",
        },
    })
    return report, red_processor, red_batch, conditions


def generate_with_projector_hook(model, processor, batch: dict) -> tuple[dict, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().to(device="cpu", dtype=torch.float32))

    handle = model.model.multi_modal_projector.register_forward_hook(capture)
    try:
        eos = int(model.generation_config.eos_token_id)
        if eos != 2:
            raise ValueError(f"Unexpected EOS ID: {eos}")
        inputs = {key: value.to("mps") for key, value in batch.items()}
        t0 = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, use_cache=True, max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=eos, pad_token_id=int(model.generation_config.pad_token_id),
            )
        torch.mps.synchronize()
        seconds = time.monotonic() - t0
    finally:
        handle.remove()
    if len(captured) != 1 or tuple(captured[0].shape) != (CONNECTOR_ROWS, 3072):
        raise ValueError(f"Unexpected projector captures: {[tuple(x.shape) for x in captured]}")
    generated_ids = output[0, batch["input_ids"].shape[1]:].tolist()
    return {
        "generated_token_ids": generated_ids,
        "full_answer": processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip(),
        "stopped_on_eos": bool(generated_ids and generated_ids[-1] == eos),
        "generation_seconds": seconds,
    }, captured[0]


def teacher_forced_candidate_score(
    model, batch: dict, candidate_ids: list[int], eos: int, device: str = "mps",
) -> dict:
    """Score complete candidate reply, including EOS, under teacher forcing."""
    prompt_len = int(batch["input_ids"].shape[1])
    ids = torch.tensor([candidate_ids], dtype=batch["input_ids"].dtype)
    full_ids = torch.cat((batch["input_ids"], ids), dim=1).to(device)
    mask = torch.cat((batch["attention_mask"], torch.ones_like(ids)), dim=1).to(device)
    target_ids = candidate_ids + [eos]
    with torch.inference_mode():
        logits = model(
            input_ids=full_ids,
            attention_mask=mask,
            input_features=batch["input_features"].to(device),
            use_cache=False,
            logits_to_keep=0,
        ).logits[0]
        token_logprobs = []
        for offset, token_id in enumerate(target_ids):
            row = logits[prompt_len - 1 + offset].float()
            token_logprobs.append(float((row[token_id] - torch.logsumexp(row, dim=-1)).cpu()))
    return {
        "candidate_token_ids": candidate_ids,
        "eos_token_id": eos,
        "token_logprobs_including_eos": token_logprobs,
        "complete_reply_logprob": sum(token_logprobs),
    }


def run_model(report: dict, processor, native_batch, conditions: dict) -> dict:
    from transformers import VoxtralForConditionalGeneration

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no model inference started")
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("Cannot establish MPS memory capacity")
    fraction = min(1.0, MEMORY_LIMIT / recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    candidate_ids = {
        "Red": processor.tokenizer.encode("Red", add_special_tokens=False),
        "Blue": processor.tokenizer.encode("Blue", add_special_tokens=False),
    }
    if candidate_ids != {"Red": [12846], "Blue": [24851]}:
        raise ValueError(f"Candidate token IDs changed: {candidate_ids}")
    report["model_result"] = {
        "candidate_token_ids": candidate_ids,
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "dtype": "bfloat16", "attn_implementation": "sdpa"},
        "mps_recommended_max_bytes": recommended,
        "mps_fraction_limit": fraction,
        "conditions": {},
    }
    with MemorySampler(torch) as memory:
        t0 = time.monotonic()
        model = VoxtralForConditionalGeneration.from_pretrained(
            str(SNAPSHOT), local_files_only=True, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa",
        ).eval()
        report["model_result"]["load_seconds"] = time.monotonic() - t0
        if torch.mps.driver_allocated_memory() > MEMORY_LIMIT:
            raise MemoryError("MPS allocation exceeded 30 GiB after load")
        projected = {}
        for name in CONDITION_ORDER:
            condition = conditions[name]
            batch = {key: value.clone() for key, value in native_batch.items()}
            batch["input_features"] = condition["features"]
            answer, projected[name] = generate_with_projector_hook(model, processor, batch)
            scores = {
                label: teacher_forced_candidate_score(model, batch, ids, 2)
                for label, ids in candidate_ids.items()
            }
            report["model_result"]["conditions"][name] = {
                "answer": answer,
                "teacher_forced": scores,
                "blue_minus_red_complete_reply_logprob": (
                    scores["Blue"]["complete_reply_logprob"] - scores["Red"]["complete_reply_logprob"]
                ),
                "projector_output_shape": list(projected[name].shape),
            }
            if name == "word_blue_outside_blue":
                for expected_word in ("red", "blue"):
                    native = report["model_result"]["conditions"][f"word_{expected_word}_outside_{expected_word}"]["answer"]
                    if not native["stopped_on_eos"] or not re.fullmatch(
                        rf"{expected_word}[.!]?", native["full_answer"], flags=re.I,
                    ):
                        raise RuntimeError(f"Native {expected_word} full-answer gate failed: {native}")
            if torch.mps.driver_allocated_memory() > MEMORY_LIMIT:
                raise MemoryError(f"MPS allocation exceeded 30 GiB after {name}")
    report["model_result"]["mps_current_observed_max_bytes"] = memory.current_max
    report["model_result"]["mps_driver_observed_max_bytes"] = memory.driver_max
    word_rows = report["shape_math"]["word_connector_rows"]
    padding_from = report["shape_math"]["nominal_connector_padding_from"]
    groups = partition_indices(CONNECTOR_ROWS, word_rows, padding_from)
    red_reference = projected["word_red_outside_red"]
    blue_reference = projected["word_blue_outside_blue"]
    for name in CONDITION_ORDER:
        profiles = {}
        for ref_name, ref_tensor in (("native_red", red_reference), ("native_blue", blue_reference)):
            l2 = torch.linalg.vector_norm(projected[name] - ref_tensor, dim=-1).tolist()
            profiles[f"vs_{ref_name}"] = {
                "l2_by_row": l2,
                "partition_summary": l2_summary(l2, groups),
            }
        report["model_result"]["conditions"][name]["projector_profiles"] = profiles
    return report["model_result"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="load local weights and run six MPS conditions")
    parser.add_argument("--output", type=Path, help="new JSON output path; never overwrite")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report, processor, native_batch, conditions = prepare()
    if not args.run:
        concise = {
            "status": report["status"],
            "phase": report["phase"],
            "script_sha256": report["script_sha256"],
            "manifest_sha256": report["stimuli"]["red"]["manifest_sha256"],
            "word_interval_samples": report["word_interval_samples"],
            "source_sha256": report["installed_transformers_source_sha256"],
            "preprocessing": {
                k: v for k, v in report["preprocessing_decomposition"].items()
                if k not in ("raw_frame_l2", "shared_max_frame_l2", "native_frame_l2")
            },
        }
        print(json.dumps(concise, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or OUT_DIR / f"piper_red_blue_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    try:
        run_model(report, processor, native_batch, conditions)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        with output.open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"Assay failed; evidence saved to {output}", file=sys.stderr)
        raise
    with output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Assay saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
