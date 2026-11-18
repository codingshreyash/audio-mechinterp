"""Bounded, FIT-only Whisper encoder word-window causal assay.

See WHISPER_FIT_WORD_WINDOW_CAUSAL_PROSPECTIVE_V1.md. The `preflight` command
does not load model weights. `run` requires a byte-identical preflight receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
WEIGHT_SHA = "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad"
EXPECTED_WINDOWS = {"piper": (90, 161), "kokoro": (101, 184)}
EXPECTED_SLOTS = {"piper": (30527, 49847), "kokoro": (33920, 57120)}
WORDS = ("silver", "teal")
SITES = ("post_conv_pos", "post_block1", "post_norm")
CAPTURE_SITES = ("post_conv_pos", "post_block0", "post_block1", "post_block2", "post_block3", "post_norm")
SEEDS = (20261118, 20261119)
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Case:
    engine: str
    word: str
    path: Path
    file_sha256: str
    pcm_sha256: str
    frames: int
    slot: tuple[int, int]
    window: tuple[int, int]
    prefix: tuple[int, int]

    @property
    def key(self) -> str:
        return f"{self.engine}_{self.word}"


def check_case_manifest() -> tuple[list[Case], dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    if file_sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("render manifest hash mismatch")
    manifest = json.loads(MANIFEST.read_text())
    selected = [p for p in manifest["pairs"] if p["pair_id"] == "color_05" and p["phrasing_id"] == "fit_chosen"]
    if len(selected) != 2 or {p["engine"] for p in selected} != {"piper", "kokoro"}:
        raise ValueError("expected exactly Piper and Kokoro FIT pairs")
    cases: list[Case] = []
    evidence: dict[str, Any] = {}
    for pair in selected:
        engine = pair["engine"]
        if pair["split"] != "fit_pool" or pair["map_fit"] is not True or pair["words"] != list(WORDS):
            raise ValueError("pair is not the frozen FIT word contrast")
        if pair["outside_slot_pcm_identical"] is not True or pair["inside_slot_pcm_differs"] is not True:
            raise ValueError("paired-PCM assertions absent")
        slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
        if slot != EXPECTED_SLOTS[engine]:
            raise ValueError("frozen sample slot changed")
        start, end = slot
        calculated = (math.floor(start / 16000 * 50) - 5, math.ceil(end / 16000 * 50) + 5)
        if calculated != EXPECTED_WINDOWS[engine]:
            raise ValueError("frozen encoder window changed")
        width = calculated[1] - calculated[0]
        prefix = (10, 10 + width)
        if prefix[1] >= calculated[0]:
            raise ValueError("prefix overlaps word window")
        pair_pcm = {}
        for word in WORDS:
            output = pair["outputs"][word]
            path = ROOT / output["path"]
            if not path.is_file() or file_sha(path) != output["file_sha256"]:
                raise ValueError(f"WAV file hash mismatch: {path}")
            pcm, rate = sf.read(path, dtype="int16", always_2d=False)
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != output["frames"]:
                raise ValueError(f"WAV format mismatch: {path}")
            pcm = np.asarray(pcm, dtype="<i2")
            pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
            if pcm_hash != output["pcm_int16_sha256"]:
                raise ValueError(f"WAV PCM hash mismatch: {path}")
            pair_pcm[word] = pcm
            cases.append(Case(engine, word, path, output["file_sha256"], pcm_hash, len(pcm), slot, calculated, prefix))
        a, b = (pair_pcm[word] for word in WORDS)
        if len(a) != len(b) or not np.array_equal(a[:start], b[:start]) or not np.array_equal(a[end:], b[end:]):
            raise ValueError(f"outside-slot PCM differs for {engine}")
        if np.array_equal(a[start:end], b[start:end]):
            raise ValueError(f"inside-slot PCM does not differ for {engine}")
        evidence[engine] = {"slot_samples": list(slot), "word_window": list(calculated), "prefix_window": list(prefix)}
    return sorted(cases, key=lambda c: (c.engine, c.word)), evidence


def preflight() -> dict[str, Any]:
    import soundfile
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers.models.whisper import feature_extraction_whisper, generation_whisper, modeling_whisper, processing_whisper, tokenization_whisper

    cases, windows = check_case_manifest()
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    weights = snapshot / "model.safetensors"
    if not weights.is_file() or file_sha(weights) != WEIGHT_SHA:
        raise ValueError("Whisper checkpoint hash mismatch")
    config = json.loads((snapshot / "config.json").read_text())
    shape = {k: config[k] for k in ("num_mel_bins", "max_source_positions", "d_model", "encoder_layers")}
    if shape != {"num_mel_bins": 80, "max_source_positions": 1500, "d_model": 384, "encoder_layers": 4}:
        raise ValueError("Whisper config shape mismatch")
    source_modules = [modeling_whisper, generation_whisper, feature_extraction_whisper, processing_whisper, tokenization_whisper]
    source_hashes = {module.__name__: file_sha(Path(inspect.getfile(module))) for module in source_modules}
    script_relative = str(Path(__file__).resolve().relative_to(ROOT))
    dirty = subprocess.run(["git", "diff", "--quiet", "--", script_relative], cwd=ROOT, check=False)
    staged_dirty = subprocess.run(["git", "diff", "--cached", "--quiet", "--", script_relative], cwd=ROOT, check=False)
    if dirty.returncode != 0 or staged_dirty.returncode != 0:
        raise ValueError("runner source has uncommitted changes")
    source_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", script_relative], cwd=ROOT, text=True
    ).strip()
    if len(source_commit) != 40:
        raise ValueError("runner has no committed source revision")
    asset_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "preprocessor_config.json")
    asset_hashes = {}
    for name in asset_names:
        asset = snapshot / name
        if not asset.is_file():
            raise ValueError(f"missing cached Whisper asset: {name}")
        asset_hashes[name] = file_sha(asset)
    snapshot_hashes = {path.name: file_sha(path) for path in sorted(snapshot.iterdir()) if path.is_file()}
    if "model.safetensors" not in snapshot_hashes or snapshot_hashes["model.safetensors"] != WEIGHT_SHA:
        raise ValueError("Whisper snapshot file inventory mismatch")
    return {
        "schema": "whisper_fit_word_window_preflight_v1",
        "script_sha256": file_sha(Path(__file__)),
        "script_source_git_commit": source_commit,
        "manifest_sha256": MANIFEST_SHA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": WEIGHT_SHA,
        "config_sha256": file_sha(snapshot / "config.json"),
        "asset_sha256": asset_hashes,
        "snapshot_files_sha256": snapshot_hashes,
        "source_hashes": source_hashes,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "soundfile_version": soundfile.__version__,
        "python_version": sys.version.split()[0],
        "device": "cpu",
        "threads": 4,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "output_scores": True, "return_dict_in_generate": True},
        "capture_sites": list(CAPTURE_SITES),
        "intervention_sites": list(SITES),
        "cases": [{"key": c.key, "file_sha256": c.file_sha256, "pcm_sha256": c.pcm_sha256, "frames": c.frames} for c in cases],
        "windows": windows,
    }


def validate_state(state: Any, expected: Any | None = None) -> None:
    import torch

    if not isinstance(state, torch.Tensor) or state.shape != (1, 1500, 384):
        raise ValueError(f"encoder state must be [1,1500,384], got {getattr(state, 'shape', None)}")
    if state.device.type != "cpu" or state.dtype != torch.float32 or not torch.isfinite(state).all():
        raise ValueError("encoder state is not finite CPU float32")
    if expected is not None and (state.device != expected.device or state.dtype != expected.dtype or state.shape != expected.shape):
        raise ValueError("donor/recipient state mismatch")


def patch_rows(recipient: Any, donor: Any, window: tuple[int, int], mode: str, seed: int | None = None, audit: dict[str, Any] | None = None) -> Any:
    import torch

    validate_state(recipient)
    validate_state(donor, recipient)
    start, end = window
    if not 0 <= start < end <= 1500:
        raise ValueError("invalid encoder window")
    result = recipient.clone()
    if mode in {"donor", "self", "full_donor"}:
        if mode == "full_donor":
            if window != (0, 1500):
                raise ValueError("full donor requires all rows")
            result[:] = donor
        else:
            result[:, start:end, :] = donor[:, start:end, :]
    elif mode == "random":
        if seed not in SEEDS:
            raise ValueError("unfrozen random seed")
        target = torch.linalg.vector_norm(donor[:, start:end, :] - recipient[:, start:end, :])
        if not torch.isfinite(target) or target <= 0:
            raise ValueError("random target Frobenius norm is zero or nonfinite")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, end - start, 384), generator=generator, dtype=torch.float32)
        norm = torch.linalg.vector_norm(noise)
        if norm <= 0:
            raise ValueError("zero random norm")
        noise *= target / norm
        result[:, start:end, :] += noise
        realized = torch.linalg.vector_norm(result[:, start:end, :] - recipient[:, start:end, :])
        if not torch.allclose(realized, target, rtol=1e-5, atol=1e-6):
            raise ValueError("random edit Frobenius norm mismatch")
        if audit is not None:
            audit.update({
                "random_edit_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                "random_edit_norm": float(realized),
                "random_target_norm": float(target),
            })
    else:
        raise ValueError("unknown intervention mode")
    validate_state(result)
    if start > 0 and not torch.equal(result[:, :start, :], recipient[:, :start, :]):
        raise ValueError("patch changed earlier rows")
    if end < 1500 and not torch.equal(result[:, end:, :], recipient[:, end:, :]):
        raise ValueError("patch changed later rows")
    return result


def site_module(model: Any, site: str) -> tuple[Any, bool]:
    encoder = model.model.encoder
    if site == "post_conv_pos":
        return encoder.layers[0], True
    if site == "post_block1":
        return encoder.layers[1], False
    if site == "post_norm":
        return encoder.layer_norm, False
    raise ValueError(f"unsupported intervention site: {site}")


def install_capture_hooks(model: Any, captures: dict[str, Any], counts: dict[str, int]) -> list[Any]:
    handles = []
    encoder = model.model.encoder

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        key = "post_conv_pos"
        counts[key] += 1
        if counts[key] != 1:
            raise ValueError("capture hook fired more than once")
        validate_state(args[0])
        captures[key] = args[0].detach().clone()

    handles.append(encoder.layers[0].register_forward_pre_hook(pre_hook))
    for index in range(4):
        key = f"post_block{index}"

        def post_hook(_module: Any, _args: tuple[Any, ...], output: Any, key: str = key) -> None:
            counts[key] += 1
            if counts[key] != 1:
                raise ValueError("capture hook fired more than once")
            validate_state(output)
            captures[key] = output.detach().clone()

        handles.append(encoder.layers[index].register_forward_hook(post_hook))

    def norm_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        counts["post_norm"] += 1
        if counts["post_norm"] != 1:
            raise ValueError("capture hook fired more than once")
        validate_state(output)
        captures["post_norm"] = output.detach().clone()

    handles.append(encoder.layer_norm.register_forward_hook(norm_hook))
    return handles


def install_patch_hook(model: Any, site: str, donor: Any, window: tuple[int, int], mode: str, seed: int | None, count: list[int], audit: dict[str, Any] | None = None) -> Any:
    module, is_pre = site_module(model, site)

    if is_pre:
        def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            count[0] += 1
            if count[0] != 1:
                raise ValueError("patch hook fired more than once")
            return (patch_rows(args[0], donor, window, mode, seed, audit), *args[1:])
        return module.register_forward_pre_hook(pre_hook)

    def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        count[0] += 1
        if count[0] != 1:
            raise ValueError("patch hook fired more than once")
        return patch_rows(output, donor, window, mode, seed, audit)
    return module.register_forward_hook(post_hook)


def normalized_words(text: str) -> list[str]:
    import re

    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())


def transcript_outcome(transcript: str, recipient: str, donor: str, baseline: str) -> dict[str, Any]:
    words = normalized_words(transcript)
    base = normalized_words(baseline)
    expected = [donor if word == recipient else word for word in base]

    def distance(a: list[str], b: list[str]) -> int:
        row = list(range(len(b) + 1))
        for index, token in enumerate(a, 1):
            next_row = [index]
            for column, other in enumerate(b, 1):
                next_row.append(min(next_row[-1] + 1, row[column] + 1, row[column - 1] + (token != other)))
            row = next_row
        return row[-1]

    return {
        "target_replacement": words == expected and base.count(recipient) == 1 and donor not in base,
        "donor_once_recipient_absent": words.count(donor) == 1 and words.count(recipient) == 0,
        "recipient_transcript_match": words == base,
        "wer_to_recipient": distance(words, base) / max(len(base), 1),
        "wer_to_single_word_donor_counterfactual": distance(words, expected) / max(len(expected), 1),
        "token_difference_count": None,  # filled from generated token IDs in the runner
        "normalized_words": words,
    }


def current_rss_bytes() -> int:
    # macOS ru_maxrss is bytes; this study is local to the Mac host.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_model(model: Any, processor: Any, batch: Any, *, hook: Any = None, hook_count: list[int] | None = None) -> dict[str, Any]:
    import torch

    start = time.monotonic()
    try:
        with torch.inference_mode():
            generation = model.generate(
                **batch, do_sample=False, num_beams=1, max_new_tokens=128,
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        if hook is not None:
            hook.remove()
    ids = generation.sequences[0].detach().cpu().tolist()
    transcript = processor.batch_decode(generation.sequences, skip_special_tokens=True)[0].strip()
    eos_id = model.generation_config.eos_token_id
    if isinstance(eos_id, list):
        eos_ids = set(eos_id)
    else:
        eos_ids = {eos_id}
    step_scores = []
    alias_tokens = {word: processor.tokenizer.encode(" " + word, add_special_tokens=False) for word in WORDS}

    def finite_or_none(value: Any) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    for step, score in enumerate(generation.scores):
        token = ids[len(ids) - len(generation.scores) + step]
        values = score[0].detach().cpu()
        top = torch.topk(values, k=2)
        margin = None
        margin_status = "multi_token_alias"
        if all(len(alias_tokens[word]) == 1 for word in WORDS):
            margin = finite_or_none(values[alias_tokens["silver"][0]] - values[alias_tokens["teal"][0]])
            margin_status = "finite" if margin is not None else "nonfinite_suppressed_alias"
        step_scores.append({
            "token_id": token, "selected_score": finite_or_none(values[token]),
            "top2_ids": top.indices.tolist(), "top2_scores": [finite_or_none(value) for value in top.values],
            "generation_processed_silver_minus_teal_first_token_score": margin,
            "alias_margin_status": margin_status,
        })
    return {
        "token_ids": ids,
        "transcript": transcript,
        "eos_seen": any(token in eos_ids for token in ids),
        "censored": not any(token in eos_ids for token in ids),
        "step_scores": step_scores,
        "elapsed_seconds": time.monotonic() - start,
        "rss_bytes": current_rss_bytes(),
        "hook_count": hook_count[0] if hook_count is not None else None,
    }


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute(expected_preflight: Path, output: Path) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    current = preflight()  # all provenance and input hashes before weight load
    expected = json.loads(expected_preflight.read_text())
    if canonical(expected) != canonical(current):
        raise ValueError("reviewed preflight receipt differs from current environment or source")
    if output.exists():
        raise FileExistsError("output directory already exists; preserve prior attempt")
    output.mkdir(parents=True)
    append_record(output / "records.jsonl", {"type": "preflight", "receipt": current, "expected_receipt_sha256": file_sha(expected_preflight)})
    start = time.monotonic()

    def check_budget() -> None:
        if time.monotonic() - start > MAX_SECONDS or current_rss_bytes() > MAX_RSS_BYTES:
            raise RuntimeError("time or resident-memory hard stop")

    try:
        torch.set_num_threads(4)
        processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).to("cpu", dtype=torch.float32).eval()
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("model is not entirely CPU float32")
        cases, _ = check_case_manifest()
        baselines: dict[str, dict[str, Any]] = {}
        states: dict[str, dict[str, Any]] = {}
        batches: dict[str, Any] = {}
        capture_hashes: dict[str, dict[str, str]] = {}
        capture_dir = output / "captures"
        capture_dir.mkdir()
        for case in cases:
            check_budget()
            pcm, rate = sf.read(case.path, dtype="int16")
            samples = np.asarray(pcm, dtype=np.float32) / 32768.0
            batch = processor(samples, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)
            if tuple(batch["input_features"].shape) != (1, 80, 3000):
                raise ValueError("processor mel tensor shape changed")
            batches[case.key] = batch
            captures: dict[str, Any] = {}
            counts = {site: 0 for site in CAPTURE_SITES}
            handles = install_capture_hooks(model, captures, counts)
            try:
                result = run_model(model, processor, batch)
            finally:
                for handle in handles:
                    handle.remove()
            capture_complete = all(count == 1 for count in counts.values()) and set(captures) == set(CAPTURE_SITES)
            other_word = WORDS[1 - WORDS.index(case.word)]
            eligible = (normalized_words(result["transcript"]).count(case.word) == 1
                        and other_word not in normalized_words(result["transcript"])
                        and not result["censored"])
            states[case.key] = captures
            baselines[case.key] = result
            hashes = {}
            mel_path = capture_dir / f"{case.key}_mel.npy"
            np.save(mel_path, batch["input_features"].numpy(), allow_pickle=False)
            hashes["mel"] = file_sha(mel_path)
            for site, tensor in captures.items():
                tensor_path = capture_dir / f"{case.key}_{site}.npy"
                np.save(tensor_path, tensor.numpy(), allow_pickle=False)
                hashes[site] = file_sha(tensor_path)
            capture_hashes[case.key] = hashes
            append_record(output / "records.jsonl", {"type": "baseline", "case": case.key, "result": result, "capture_sha256": hashes, "eligible": eligible, "capture_complete": capture_complete, "capture_hook_counts": counts})
            if not capture_complete:
                raise ValueError(f"capture hooks did not fire exactly once: {counts}")
            if not eligible:
                raise ValueError(f"baseline target-word eligibility failed: {case.key}")
            check_budget()

        for engine in ("piper", "kokoro"):
            silver = f"{engine}_silver"
            teal = f"{engine}_teal"
            case = next(c for c in cases if c.key == silver)
            traces = {}
            for site in ("mel", *CAPTURE_SITES):
                if site == "mel":
                    a = batches[silver]["input_features"]
                    b = batches[teal]["input_features"]
                    delta = torch.linalg.vector_norm(a - b, dim=1)[0].numpy()
                    maxabs = torch.amax(torch.abs(a - b), dim=1)[0].numpy()
                    word_window = (2 * case.window[0], 2 * case.window[1])
                    prefix_window = (2 * case.prefix[0], 2 * case.prefix[1])
                else:
                    difference = states[silver][site] - states[teal][site]
                    delta = torch.linalg.vector_norm(difference, dim=-1)[0].numpy()
                    maxabs = torch.amax(torch.abs(difference), dim=-1)[0].numpy()
                    word_window = case.window
                    prefix_window = case.prefix
                trace_path = capture_dir / f"{engine}_{site}_silver_teal_l2.npy"
                np.save(trace_path, delta, allow_pickle=False)
                maxabs_path = capture_dir / f"{engine}_{site}_silver_teal_maxabs.npy"
                np.save(maxabs_path, maxabs, allow_pickle=False)
                outside = np.ones(len(delta), dtype=bool)
                outside[slice(*word_window)] = False
                outside[slice(*prefix_window)] = False
                traces[site] = {
                    "path": str(trace_path.relative_to(output)), "sha256": file_sha(trace_path),
                    "maxabs_path": str(maxabs_path.relative_to(output)), "maxabs_sha256": file_sha(maxabs_path),
                    "shape": list(delta.shape), "word_window": list(word_window), "prefix_window": list(prefix_window),
                    "word_l2_sum": float(delta[slice(*word_window)].sum()),
                    "prefix_l2_sum": float(delta[slice(*prefix_window)].sum()),
                    "remaining_l2_sum": float(delta[outside].sum()),
                    "global_max_abs_difference": float(maxabs.max()),
                    "first_nonzero_frame": int(np.flatnonzero(delta > 0)[0]) if np.any(delta > 0) else None,
                }
            append_record(output / "records.jsonl", {"type": "trace", "engine": engine, "words": list(WORDS), "traces": traces})

        for engine in ("piper", "kokoro"):
            for recipient_word, donor_word in (("silver", "teal"), ("teal", "silver")):
                recipient = f"{engine}_{recipient_word}"
                donor = f"{engine}_{donor_word}"
                case = next(c for c in cases if c.key == recipient)
                # The case order, sites, conditions and seeds were fixed before inference.
                for site in SITES:
                    for condition, window in (("word", case.window), ("prefix", case.prefix), ("self", case.window)):
                        check_budget()
                        source = states[recipient][site] if condition == "self" else states[donor][site]
                        count = [0]
                        handle = install_patch_hook(model, site, source, window, "self" if condition == "self" else "donor", None, count)
                        result = run_model(model, processor, batches[recipient], hook=handle, hook_count=count)
                        control_pass = None if condition != "self" else count[0] == 1 and result["token_ids"] == baselines[recipient]["token_ids"]
                        outcome = transcript_outcome(result["transcript"], recipient_word, donor_word, baselines[recipient]["transcript"])
                        outcome["target_replacement"] = outcome["target_replacement"] and not result["censored"]
                        outcome["token_difference_count"] = sum(a != b for a, b in zip(result["token_ids"], baselines[recipient]["token_ids"])) + abs(len(result["token_ids"]) - len(baselines[recipient]["token_ids"]))
                        append_record(output / "records.jsonl", {
                            "type": "intervention", "recipient": recipient, "donor": donor, "site": site,
                            "condition": condition, "window": list(window), "result": result,
                            "hook_count": count[0],
                            "valid_hook": count[0] == 1,
                            "state_shape": [1, 1500, 384], "state_dtype": "float32", "finite_checked": True,
                            "recipient_state_sha256": capture_hashes[recipient][site],
                            "source_state_sha256": capture_hashes[recipient if condition == "self" else donor][site],
                            "control_pass": control_pass,
                            "outcome": outcome,
                        })
                        if count[0] != 1:
                            raise ValueError(f"patch hook fired {count[0]} times: {recipient}/{site}/{condition}")
                        if control_pass is False:
                            raise ValueError(f"self patch changed tokens: {recipient}/{site}")
                        check_budget()
                for condition, window, seed in (("full_donor", (0, 1500), None), *(("random", case.window, seed) for seed in SEEDS)):
                    check_budget()
                    count = [0]
                    patch_audit: dict[str, Any] = {}
                    handle = install_patch_hook(model, "post_norm", states[donor]["post_norm"], window, condition, seed, count, patch_audit)
                    result = run_model(model, processor, batches[recipient], hook=handle, hook_count=count)
                    control_pass = None if condition != "full_donor" else count[0] == 1 and result["token_ids"] == baselines[donor]["token_ids"]
                    outcome = transcript_outcome(result["transcript"], recipient_word, donor_word, baselines[recipient]["transcript"])
                    outcome["target_replacement"] = outcome["target_replacement"] and not result["censored"]
                    outcome["token_difference_count"] = sum(a != b for a, b in zip(result["token_ids"], baselines[recipient]["token_ids"])) + abs(len(result["token_ids"]) - len(baselines[recipient]["token_ids"]))
                    append_record(output / "records.jsonl", {
                        "type": "intervention", "recipient": recipient, "donor": donor, "site": "post_norm",
                        "condition": condition, "seed": seed, "window": list(window), "result": result,
                        "hook_count": count[0], "patch_audit": patch_audit,
                        "valid_hook": count[0] == 1,
                        "state_shape": [1, 1500, 384], "state_dtype": "float32", "finite_checked": True,
                        "recipient_state_sha256": capture_hashes[recipient]["post_norm"],
                        "source_state_sha256": capture_hashes[donor]["post_norm"],
                        "control_pass": control_pass,
                        "outcome": outcome,
                    })
                    if count[0] != 1:
                        raise ValueError(f"patch hook fired {count[0]} times: {recipient}/post_norm/{condition}")
                    if control_pass is False:
                        raise ValueError(f"full-donor implementation control failed: {recipient}")
                    check_budget()
        check_budget()
        append_record(output / "records.jsonl", {"type": "complete", "elapsed_seconds": time.monotonic() - start, "rss_bytes": current_rss_bytes()})
    except BaseException as exc:
        append_record(output / "records.jsonl", {"type": "failure", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.monotonic() - start})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--expected-preflight", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        receipt = preflight()
        if args.output.exists():
            raise FileExistsError("preflight receipt already exists")
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(args.output)
    else:
        execute(args.expected_preflight, args.output)


if __name__ == "__main__":
    main()
