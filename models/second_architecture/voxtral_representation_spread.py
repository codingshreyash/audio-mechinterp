"""Fit-only, passive Voxtral red/blue representation-spread assay.

Default is CPU-only feature/shape preflight. --run explicitly enables MPS
generation: baseline and passive-hook passes for each exact PCM Piper clip.
No activations are patched and no security claim follows from this assay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch

from voxtral_smoke import (
    HERE, MANIFEST, MAX_NEW_TOKENS, MEMORY_LIMIT, MemorySampler,
    REVISION, SNAPSHOT, TRANSFORMERS_COMMIT, package_versions,
    prepare_processor, select_clip, sha256_file, validate_snapshot,
)


OUT_DIR = HERE / "runs" / "voxtral_representation_spread_fit_v1"
MEL_HOP = 160          # samples: 10 ms at 16 kHz
MEL_HALF_WINDOW = 200  # torch.stft default centered 400-sample window
MEL_FRAMES = 3000
ENCODER_ROWS = 1500
CONNECTOR_ROWS = 375
CONNECTOR_GROUP = 4   # four stride-2 encoder rows per projector input


def nominal_word_mel_frames(start: int, end: int) -> list[int]:
    """Frames whose centered STFT window could overlap the PCM word slot."""
    return [i for i in range(MEL_FRAMES)
            if i * MEL_HOP - MEL_HALF_WINDOW < end
            and i * MEL_HOP + MEL_HALF_WINDOW > start]


def nominal_word_connector_rows(start: int, end: int) -> list[int]:
    """Conservative pre-attention receptive field; not a localization claim.

    Conv1: k=3,s=1,p=1; Conv2: k=3,s=2,p=1. Encoder row j sees
    mel frames 2j-2..2j+2. Projector row r groups j=4r..4r+3, so its
    direct pre-attention mel support is 8r-2..8r+8.
    """
    rows = []
    for r in range(CONNECTOR_ROWS):
        first_mel = 8 * r - 2
        last_mel = 8 * r + 8
        first_sample = first_mel * MEL_HOP - MEL_HALF_WINDOW
        last_sample = last_mel * MEL_HOP + MEL_HALF_WINDOW
        if first_sample < end and last_sample > start:
            rows.append(r)
    return rows


def partition_indices(n: int, word_indices: list[int], padded_from: int) -> dict[str, list[int]]:
    word = set(word_indices)
    if len(word) != len(word_indices) or not all(0 <= i < n for i in word):
        raise ValueError("Invalid word indices")
    if not 0 <= padded_from <= n:
        raise ValueError("Invalid padding boundary")
    if any(i >= padded_from for i in word):
        raise ValueError("Word window intersects nominal padding")
    return {
        "word_window": sorted(word),
        "outside_real_audio": [i for i in range(padded_from) if i not in word],
        "nominal_padding": list(range(padded_from, n)),
    }


def l2_summary(values: list[float], groups: dict[str, list[int]]) -> dict:
    def stats(indices: list[int]) -> dict:
        subset = [values[i] for i in indices]
        squared = sum(v * v for v in subset)
        return {
            "count": len(subset),
            "mean_l2": sum(subset) / len(subset) if subset else None,
            "max_l2": max(subset) if subset else None,
            "nonzero_gt_1e_6": sum(v > 1e-6 for v in subset),
            "sum_squared_l2": squared,
        }

    out = {name: stats(indices) for name, indices in groups.items()}
    total = sum(x["sum_squared_l2"] for x in out.values())
    for x in out.values():
        x["share_squared_l2"] = x["sum_squared_l2"] / total if total else 0.0
    return out


def prepare_pair() -> tuple[dict, dict[str, object]]:
    snapshot = validate_snapshot()
    versions = package_versions()
    red_path, red_stimulus = select_clip("piper", "red")
    blue_path, blue_stimulus = select_clip("piper", "blue")
    if red_stimulus["frames"] != blue_stimulus["frames"]:
        raise ValueError("Paired WAVs have different lengths")
    manifest = json.loads(MANIFEST.read_text())
    group = manifest["engines"]["piper"]
    if not group["prefix_and_suffix_pcm_identical"]:
        raise ValueError("Manifest does not assert identical outside-word PCM")
    word_start = int(group["word_slot_start_sample"])
    word_end = int(group["word_slot_end_sample_exclusive"])
    red_pcm, red_rate = sf.read(red_path, dtype="float32")
    blue_pcm, blue_rate = sf.read(blue_path, dtype="float32")
    if red_rate != 16000 or blue_rate != 16000 or red_pcm.ndim != 1 or blue_pcm.ndim != 1:
        raise ValueError("Expected 16 kHz mono PCM")
    if len(red_pcm) != len(blue_pcm) or len(red_pcm) != red_stimulus["frames"]:
        raise ValueError("PCM length disagrees with manifest")
    if not (0 <= word_start < word_end <= len(red_pcm)):
        raise ValueError("Invalid word span")
    outside_pcm_equal = bool((red_pcm[:word_start] == blue_pcm[:word_start]).all()
                             and (red_pcm[word_end:] == blue_pcm[word_end:]).all())
    if not outside_pcm_equal:
        raise ValueError("PCM differs outside the specified word slot")

    processor_red, red_batch, red_prep = prepare_processor(red_path)
    processor_blue, blue_batch, blue_prep = prepare_processor(blue_path)
    if red_prep["input_ids"] != blue_prep["input_ids"]:
        raise ValueError("Red/blue chat token IDs differ")
    if red_batch["input_features"].shape != blue_batch["input_features"].shape:
        raise ValueError("Red/blue mel shapes differ")
    diff = (blue_batch["input_features"] - red_batch["input_features"]).float()
    feature_l2 = torch.linalg.vector_norm(diff[0], dim=0).tolist()
    if len(feature_l2) != MEL_FRAMES:
        raise ValueError("Unexpected mel frame count")
    feature_word = nominal_word_mel_frames(word_start, word_end)
    connector_word = nominal_word_connector_rows(word_start, word_end)
    actual_samples = len(red_pcm)
    feature_padding_from = min(MEL_FRAMES, (actual_samples + MEL_HOP - 1) // MEL_HOP)
    connector_padding_from = min(CONNECTOR_ROWS, (actual_samples + 8 * MEL_HOP - 1) // (8 * MEL_HOP))
    feature_groups = partition_indices(MEL_FRAMES, feature_word, feature_padding_from)
    connector_groups = partition_indices(CONNECTOR_ROWS, connector_word, connector_padding_from)
    report = {
        "schema_version": 1,
        "status": "preflight",
        "phase": "fit-only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "voxtral_smoke_script_sha256": sha256_file(HERE / "voxtral_smoke.py"),
        "model_id": "mistralai/Voxtral-Mini-3B-2507",
        "model_revision": REVISION,
        "transformers_release_tag_commit": TRANSFORMERS_COMMIT,
        "transformers_source_verification": "release tag association; installed PyPI wheel checked by version only",
        "dependency_versions": versions,
        "requirements_freeze_sha256": sha256_file(HERE / "requirements.freeze.txt"),
        "snapshot": snapshot,
        "stimuli": {"red": red_stimulus, "blue": blue_stimulus},
        "processor": {"red": red_prep, "blue": blue_prep},
        "pcm_outside_word_identical": outside_pcm_equal,
        "word_interval_samples": [word_start, word_end],
        "shape_math": {
            "sample_rate": 16000,
            "mel_hop_samples": MEL_HOP,
            "mel_window_samples": 2 * MEL_HALF_WINDOW,
            "mel_centered": True,
            "mel_frames": MEL_FRAMES,
            "encoder_rows": ENCODER_ROWS,
            "connector_rows": CONNECTOR_ROWS,
            "connector_group_encoder_rows": CONNECTOR_GROUP,
            "nominal_feature_padding_from": feature_padding_from,
            "nominal_connector_padding_from": connector_padding_from,
            "word_mel_frames": feature_word,
            "word_connector_rows": connector_word,
            "feature_partition": {k: {"start": min(v) if v else None,
                                      "end_exclusive": max(v) + 1 if v else None,
                                      "count": len(v)} for k, v in feature_groups.items()},
            "connector_partition": {k: {"start": min(v) if v else None,
                                        "end_exclusive": max(v) + 1 if v else None,
                                        "count": len(v)} for k, v in connector_groups.items()},
        },
        "feature_difference": {
            "l2_by_mel_frame": feature_l2,
            "partition_summary": l2_summary(feature_l2, feature_groups),
            "note": "Whisper log-mel normalization uses a clip-wide maximum; off-word differences may start before the model.",
        },
    }
    return report, {"red": (processor_red, red_batch), "blue": (processor_blue, blue_batch)}


def generated_suffix(model, processor, batch, hook_capture: list[torch.Tensor] | None = None) -> dict:
    captured = hook_capture
    handle = None
    if captured is not None:
        def capture(_module, _inputs, output):
            captured.append(output.detach().to(device="cpu", dtype=torch.float32))

        handle = model.model.multi_modal_projector.register_forward_hook(capture)
    try:
        eos = int(model.generation_config.eos_token_id)
        if eos != 2:
            raise ValueError(f"Unexpected EOS ID: {eos}")
        inputs = batch.to("mps")
        t0 = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=eos,
                pad_token_id=int(model.generation_config.pad_token_id),
            )
        torch.mps.synchronize()
        seconds = time.monotonic() - t0
    finally:
        if handle is not None:
            handle.remove()
    ids = output[0, batch["input_ids"].shape[1]:].tolist()
    return {
        "generated_token_ids": ids,
        "full_answer": processor.batch_decode([ids], skip_special_tokens=True)[0].strip(),
        "stopped_on_eos": bool(ids and ids[-1] == eos),
        "generation_seconds": seconds,
    }


def run_model(report: dict, pair: dict[str, object]) -> dict:
    from transformers import VoxtralForConditionalGeneration

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no model inference started")
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("Cannot establish MPS recommended maximum memory")
    fraction = min(1.0, MEMORY_LIMIT / recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    with MemorySampler(torch) as memory:
        t0 = time.monotonic()
        model = VoxtralForConditionalGeneration.from_pretrained(
            str(SNAPSHOT), local_files_only=True, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa",
        ).eval()
        load_seconds = time.monotonic() - t0
        if torch.mps.driver_allocated_memory() > MEMORY_LIMIT:
            raise MemoryError("MPS allocation exceeded 30 GiB after load")
        results = {}
        projected = {}
        for word in ("red", "blue"):
            processor, batch = pair[word]
            baseline = generated_suffix(model, processor, batch)
            captures: list[torch.Tensor] = []
            hooked = generated_suffix(model, processor, batch, captures)
            if len(captures) != 1 or tuple(captures[0].shape) != (CONNECTOR_ROWS, 3072):
                raise ValueError(f"Unexpected projector hook calls/shapes: {[tuple(x.shape) for x in captures]}")
            projected[word] = captures[0]
            results[word] = {
                "unhooked": baseline,
                "passive_hook": hooked,
                "hook_preserved_generated_ids": baseline["generated_token_ids"] == hooked["generated_token_ids"],
                "projector_output_shape": list(captures[0].shape),
            }
            if torch.mps.driver_allocated_memory() > MEMORY_LIMIT:
                raise MemoryError(f"MPS allocation exceeded 30 GiB after {word} generation")
    row_l2 = torch.linalg.vector_norm(projected["blue"] - projected["red"], dim=-1).tolist()
    if len(row_l2) != CONNECTOR_ROWS:
        raise ValueError("Unexpected projector difference row count")
    pad_from = report["shape_math"]["nominal_connector_padding_from"]
    word_rows = report["shape_math"]["word_connector_rows"]
    groups = partition_indices(CONNECTOR_ROWS, word_rows, pad_from)
    return {
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "attn_implementation": "sdpa", "dtype": "bfloat16"},
        "load_seconds": load_seconds,
        "mps_recommended_max_bytes": recommended,
        "mps_fraction_limit": fraction,
        "mps_current_observed_max_bytes": memory.current_max,
        "mps_driver_observed_max_bytes": memory.driver_max,
        "generations": results,
        "both_hooks_preserved_generated_ids": all(x["hook_preserved_generated_ids"] for x in results.values()),
        "projector_difference": {
            "recipient": "red", "donor": "blue",
            "l2_by_row": row_l2,
            "partition_summary": l2_summary(row_l2, groups),
            "note": "Rows are post-bidirectional-encoder; nominal acoustic window does not bound encoded information.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="load local weights and run bounded MPS assay")
    parser.add_argument("--output", type=Path, help="new JSON artifact path; never overwrites")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report, pair = prepare_pair()
    if not args.run:
        concise = {k: report[k] for k in ("status", "model_revision", "word_interval_samples", "shape_math")}
        concise["feature_difference_summary"] = report["feature_difference"]["partition_summary"]
        print(json.dumps(concise, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or OUT_DIR / f"piper_red_blue_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    try:
        report["model_result"] = run_model(report, pair)
        report["status"] = ("complete" if report["model_result"]["both_hooks_preserved_generated_ids"]
                            else "hook_parity_failed")
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
    print("Passive hook ID parity:", report["model_result"]["both_hooks_preserved_generated_ids"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
