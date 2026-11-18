"""Independent saved-data audit for the Wav2Vec2 FIT stage-baseline assay.

No model weights are loaded. The audit checks exact FIT audio identity, stock and
shared-statistic preprocessing, all captured stage tensors, complete greedy CTC
decoding, and the paired time traces. It cannot prove a model forward from files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
RUNNER = ROOT / "models/second_architecture/wav2vec2_fit_baseline_stage_capture_v1.py"
LOCALITY_PREFLIGHT = ROOT / "models/second_architecture/WAV2VEC2_INPUT_LOCALITY_PREFLIGHT_V1.json"
LOCALITY_PREFLIGHT_SHA = "15074ffee8bde339810b4be667e3ce40bd6a45e234739ea37c614497b6eba4a2"
MODEL_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
WEIGHTS_SHA = "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a"
CONFIG_SHA = "d3ec255c063d9f95057b553b19c20135b259875834a4fe9deb218a6be25b4cf3"
PREPROCESSOR_SHA = "b225d617c025463b9e157e06afea8b90dc7078fc70b013c533328423e0486b4a"
SPECIAL_TOKENS_SHA = "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675"
TOKENIZER_CONFIG_SHA = "dc790594f5bc351a4311c6624f40acd95850d4aaf2a5cb3c656c9b610720b608"
VOCAB_SHA = "19727f8944fe6459fc3f240ae2c198395b740f6a029bd23e06656266b83bcf64"
FEATURE_EXTRACTOR_SOURCE_SHA = "e5e9a0baf70716fee503f4f66a7a61312a132be989b2d7e2649e057ccbefa2cc"
MODEL_SOURCE_SHA = "c404982e3a063a5f68ee85809356c8a5ed29ad24c8441eea1d56346a57b5da0c"
PROCESSOR_SOURCE_SHA = "95adaf4e8398ca3d72570eba32925dcd964ed3b0027ef9abff1fd2454a96d559"
TOKENIZER_SOURCE_SHA = "b4f8f80838e462bc5d77b3998aa500879bdcce951779223c94ce126f7ffa68f2"
PAIRS = {"color_05": ("silver", "teal"), "object_02": ("plate", "cup")}
ENGINES = ("piper", "kokoro")
MODES = ("S", "C")
SLOTS = {
    ("color_05", "piper"): (30527, 49847),
    ("color_05", "kokoro"): (33920, 57120),
    ("object_02", "piper"): (30527, 42788),
    ("object_02", "kokoro"): (33920, 54320),
}
KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)
STAGES = (
    "input_values", "first_conv_raw", "first_conv_norm",
    "final_feature_conv", "feature_projection", "encoder_pre_block0",
    "post_block0", "post_block5", "post_block11", "logits",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_SECONDS = 1800
MAX_RSS = 4 * 1024**3
MAX_OUTPUT_BYTES = 2 * 1024**3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def normalized_words(value: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", value.casefold())


def conv_length(raw_length: int) -> int:
    length = raw_length
    for kernel, stride in zip(KERNELS, STRIDES, strict=True):
        length = (length - kernel) // stride + 1
    require(length > 0, "convolution length invalid")
    return length


def conv1_length(raw_length: int) -> int:
    return (raw_length - KERNELS[0]) // STRIDES[0] + 1


def nominal_window(slot: tuple[int, int], stage: str) -> tuple[int, int]:
    start, end = slot
    if stage == "input_values":
        return slot
    if stage in {"first_conv_raw", "first_conv_norm"}:
        receptive, stride = 10, 5
    else:
        receptive, stride = 400, 320
    return (math.ceil((start - receptive + 1) / stride),
            (end - 1) // stride + 1)


def case_key(pair_id: str, engine: str, word: str) -> str:
    return f"{pair_id}_{engine}_{word}"


def trusted_inputs() -> dict[str, dict[str, Any]]:
    require(sha(MANIFEST) == MANIFEST_SHA, "FIT manifest bytes changed")
    manifest = parse_json(MANIFEST.read_text())
    cases: dict[str, dict[str, Any]] = {}
    for pair_id, words in PAIRS.items():
        for engine in ENGINES:
            selected = [row for row in manifest["pairs"]
                        if row.get("pair_id") == pair_id
                        and row.get("engine") == engine
                        and row.get("phrasing_id") == "fit_chosen"]
            require(len(selected) == 1, f"FIT pair coverage differs: {pair_id}/{engine}")
            pair = selected[0]
            require(pair.get("split") == "fit_pool" and pair.get("map_fit") is True
                    and tuple(pair.get("words", ())) == words,
                    f"non-FIT or changed words: {pair_id}/{engine}")
            slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
            require(slot == SLOTS[(pair_id, engine)], "word slot changed")
            require(pair.get("outside_slot_pcm_identical") is True
                    and pair.get("inside_slot_pcm_differs") is True,
                    "manifest PCM assertion absent")
            pair_pcm: dict[str, np.ndarray] = {}
            for word in words:
                meta = pair["outputs"][word]
                path = ROOT / meta["path"]
                require(path.is_file() and not path.is_symlink()
                        and sha(path) == meta["file_sha256"],
                        f"WAV bytes changed: {path}")
                with wave.open(str(path), "rb") as wav:
                    require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                             wav.getnframes()) == (1, 2, 16000, meta["frames"]),
                            f"WAV format changed: {path}")
                    raw = wav.readframes(wav.getnframes())
                require(hashlib.sha256(raw).hexdigest() == meta["pcm_int16_sha256"],
                        f"PCM bytes changed: {path}")
                pcm = np.frombuffer(raw, dtype="<i2").copy()
                pair_pcm[word] = pcm
                cases[case_key(pair_id, engine, word)] = {
                    "pair_id": pair_id, "engine": engine, "word": word,
                    "wav_sha256": meta["file_sha256"],
                    "pcm_sha256": meta["pcm_int16_sha256"],
                    "frames": len(pcm), "slot": slot, "pcm": pcm,
                }
            a, b = (pair_pcm[word] for word in words)
            start, end = slot
            require(len(a) == len(b) and np.array_equal(a[:start], b[:start])
                    and np.array_equal(a[end:], b[end:])
                    and not np.array_equal(a[start:end], b[start:end]),
                    f"actual PCM contrast invalid: {pair_id}/{engine}")
    require(len(cases) == 8, "FIT case count changed")
    return cases


def expected_input(case: dict[str, Any], mode: str,
                   cases: dict[str, dict[str, Any]]) -> np.ndarray:
    require(mode in MODES, "input mode changed")
    pcm = case["pcm"]
    values = pcm.astype(np.float32) / np.float32(32768.0)
    if mode == "S":
        mean, variance = values.mean(), values.var()
    else:
        words = PAIRS[case["pair_id"]]
        peer = cases[case_key(case["pair_id"], case["engine"], words[0])]
        start, end = case["slot"]
        shared = np.concatenate((peer["pcm"][:start], peer["pcm"][end:]))
        shared_values = shared.astype(np.float32) / np.float32(32768.0)
        mean, variance = shared_values.mean(), shared_values.var()
    return ((values - mean) / np.sqrt(variance + np.float32(1e-7))).astype(np.float32)


def common_stats(case: dict[str, Any], cases: dict[str, dict[str, Any]]) -> tuple[float, float]:
    first = PAIRS[case["pair_id"]][0]
    peer = cases[case_key(case["pair_id"], case["engine"], first)]
    start, end = case["slot"]
    common = np.concatenate((peer["pcm"][:start], peer["pcm"][end:]))
    values = common.astype(np.float32) / np.float32(32768.0)
    return float(np.float32(values.mean())), float(np.float32(values.var()))


def read_records(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "records.jsonl absent or symlink")
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "records JSONL truncated")
    rows = []
    for line in raw.decode("utf-8").splitlines():
        row = parse_json(line)
        require(isinstance(row, dict), "non-object JSONL row")
        rows.append(row)
    return rows


def trusted_capture_path(output: Path, relative: str) -> Path:
    raw = output / relative
    path = raw.resolve()
    require(not raw.is_symlink() and path.is_relative_to(output.resolve()) and path.is_file(),
            f"capture path invalid: {relative}")
    return path


def checked_array(output: Path, relative: str, expected_sha: str,
                  shape: tuple[int, ...]) -> np.ndarray:
    require(isinstance(relative, str) and isinstance(expected_sha, str)
            and HEX64.fullmatch(expected_sha) is not None,
            "capture reference invalid")
    path = trusted_capture_path(output, relative)
    require(sha(path) == expected_sha, f"capture SHA mismatch: {relative}")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    require(arr.dtype == np.float32 and tuple(arr.shape) == shape
            and bool(np.isfinite(arr).all()), f"capture shape/dtype/finite invalid: {relative}")
    return arr


def stage_shape(stage: str, frames: int) -> tuple[int, ...]:
    t1, t7 = conv1_length(frames), conv_length(frames)
    if stage == "input_values":
        return (1, frames)
    if stage in {"first_conv_raw", "first_conv_norm"}:
        return (1, 512, t1)
    if stage == "final_feature_conv":
        return (1, 512, t7)
    if stage == "logits":
        return (1, t7, 32)
    require(stage in STAGES, "unknown stage")
    return (1, t7, 768)


def stage_time_axis(stage: str) -> int:
    return 2 if stage in {"first_conv_raw", "first_conv_norm", "final_feature_conv"} else 1


def validate_baseline(row: dict[str, Any], output: Path,
                      cases: dict[str, dict[str, Any]], processor: Any) -> dict[str, Any]:
    require(row.get("type") == "baseline", "not a baseline row")
    case_id = row.get("case_key")
    mode = row.get("mode")
    require(case_id in cases and mode in MODES, "unknown baseline case/mode")
    case = cases[case_id]
    require(row.get("schema") == "wav2vec2_fit_baseline_run_v1", "baseline schema differs")
    require(row.get("case") == {k: case[k] for k in ("pair_id", "engine", "word")},
            "baseline case metadata differs")
    require(all(row.get(key) == case[key] for key in
                ("pair_id", "engine", "word", "wav_sha256", "pcm_sha256")),
            "baseline source identity differs")
    if mode == "C":
        mean, variance = common_stats(case, cases)
        require(row.get("common_mean") == mean and row.get("common_variance") == variance,
                "Mode C shared statistics differ")
    else:
        require(row.get("common_mean") is None and row.get("common_variance") is None,
                "Mode S must not declare shared statistics")
    hashes = row.get("capture_sha256")
    shapes = row.get("capture_shapes")
    require(isinstance(hashes, dict) and set(hashes) == set(STAGES)
            and isinstance(shapes, dict) and set(shapes) == set(STAGES),
            "capture stage coverage differs")
    arrays = {}
    for stage in STAGES:
        expected_shape = stage_shape(stage, case["frames"])
        require(shapes[stage] == list(expected_shape), f"capture shape receipt differs: {stage}")
        relative = f"captures/{mode}_{case_id}_{stage}.npy"
        arrays[stage] = checked_array(output, relative, hashes[stage], expected_shape)
    require(row.get("input_sha256") == hashes["input_values"]
            and row.get("logits_sha256") == hashes["logits"]
            and row.get("logits_shape") == shapes["logits"],
            "input/logit aliases differ from capture map")
    wanted = expected_input(case, mode, cases)
    require(np.array_equal(np.asarray(arrays["input_values"])[0], wanted),
            f"preprocessed waveform differs: {mode}/{case_id}")
    logits = np.asarray(arrays["logits"])
    ids = logits.argmax(axis=-1)[0].tolist()
    require(row.get("argmax_ids") == ids, f"CTC argmax differs: {mode}/{case_id}")
    decoded = processor.batch_decode([ids])[0]
    require(row.get("transcript") == decoded,
            f"CTC transcript differs: {mode}/{case_id}")
    normalized = normalized_words(decoded)
    require(row.get("normalized_words") == normalized,
            "normalized transcript differs")
    alternative = next(word for word in PAIRS[case["pair_id"]] if word != case["word"])
    eligible = normalized.count(case["word"]) == 1 and alternative not in normalized
    require(row.get("eligible") is eligible, "baseline eligibility flag differs")
    hook_counts = row.get("hook_counts")
    require(isinstance(hook_counts, dict) and set(hook_counts) == set(STAGES)
            and all(type(value) is int and value == 1 for value in hook_counts.values()),
            "capture hook counts differ")
    require(isinstance(row.get("elapsed_seconds"), (int, float))
            and 0 <= row["elapsed_seconds"] <= MAX_SECONDS
            and type(row.get("rss_bytes")) is int and 0 < row["rss_bytes"] <= MAX_RSS,
            "baseline runtime/memory receipt invalid")
    return {"case_key": case_id, "mode": mode, "transcript": decoded,
            "eligible": eligible, "capture_sha256": hashes}


def trace_vectors(a: np.ndarray, b: np.ndarray, stage: str) -> tuple[np.ndarray, np.ndarray]:
    require(a.shape == b.shape, "paired trace shape differs")
    axis = stage_time_axis(stage)
    if stage == "input_values":
        diff = np.asarray(a[0], dtype=np.float64) - np.asarray(b[0], dtype=np.float64)
        return np.sqrt(diff * diff).astype(np.float32), np.abs(diff).astype(np.float32)
    if axis == 2:
        diff = np.asarray(a[0], dtype=np.float64) - np.asarray(b[0], dtype=np.float64)
        return np.sqrt(np.sum(diff * diff, axis=0)).astype(np.float32), np.max(np.abs(diff), axis=0).astype(np.float32)
    diff = np.asarray(a[0], dtype=np.float64) - np.asarray(b[0], dtype=np.float64)
    return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32), np.max(np.abs(diff), axis=1).astype(np.float32)


def validate_trace(row: dict[str, Any], output: Path,
                   baselines: dict[tuple[str, str], dict[str, Any]],
                   cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(row.get("type") == "trace", "not a trace row")
    pair_id, engine, mode = row.get("pair_id"), row.get("engine"), row.get("mode")
    require(pair_id in PAIRS and engine in ENGINES and mode in MODES,
            "trace identity invalid")
    left, right = PAIRS[pair_id]
    require(row.get("schema") == "wav2vec2_fit_baseline_run_v1"
            and row.get("words") == [left, right], "trace metadata differs")
    left_key, right_key = case_key(pair_id, engine, left), case_key(pair_id, engine, right)
    require((mode, left_key) in baselines and (mode, right_key) in baselines,
            "trace lacks paired baselines")
    stages = row.get("stages")
    require(isinstance(stages, dict) and set(stages) == set(STAGES),
            "trace stage coverage differs")
    results = {}
    case = cases[left_key]
    for stage in STAGES:
        item = stages[stage]
        require(isinstance(item, dict), "trace stage item malformed")
        shape = stage_shape(stage, case["frames"])
        a = checked_array(output, f"captures/{mode}_{left_key}_{stage}.npy",
                          baselines[(mode, left_key)]["capture_sha256"][stage], shape)
        b = checked_array(output, f"captures/{mode}_{right_key}_{stage}.npy",
                          baselines[(mode, right_key)]["capture_sha256"][stage], shape)
        l2, maxabs = trace_vectors(a, b, stage)
        for kind, expected in (("l2", l2), ("maxabs", maxabs)):
            relative = f"traces/{mode}_{pair_id}_{engine}_{stage}_{kind}.npy"
            observed = checked_array(output, relative, item.get(f"{kind}_sha256"), tuple(expected.shape))
            require(np.array_equal(observed, expected),
                    f"trace values differ: {mode}/{pair_id}/{engine}/{stage}/{kind}")
        window = nominal_window(case["slot"], stage)
        require(item.get("nominal_window") == list(window), "trace nominal window differs")
        nonzero = np.flatnonzero(maxabs > 0)
        first = int(nonzero[0]) if len(nonzero) else None
        require(item.get("first_nonzero_index") == first, "trace first-nonzero differs")
        outside = np.r_[0:window[0], window[1]:len(maxabs)]
        offslot = float(maxabs[outside].max()) if len(outside) else 0.0
        require(type(item.get("offslot_maxabs")) is float
                and item["offslot_maxabs"] == offslot,
                "trace offslot maximum differs")
        if mode == "C" and stage in {"input_values", "first_conv_raw"}:
            require(offslot == 0.0,
                    f"shared-stat input locality failed: {pair_id}/{engine}/{stage}")
        if mode == "S" and stage == "input_values":
            require(offslot > 0.0,
                    f"stock normalization offslot contrast absent: {pair_id}/{engine}")
        results[stage] = {"first_nonzero_index": first, "offslot_maxabs": offslot}
    return {"pair_id": pair_id, "engine": engine, "mode": mode,
            "stages": results}


def audit_preflight(record: dict[str, Any], expected_path: Path,
                    expected_sha: str, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(isinstance(expected_sha, str) and HEX64.fullmatch(expected_sha) is not None
            and sha(expected_path) == expected_sha,
            "external preflight SHA mismatch")
    expected = parse_json(expected_path.read_text())
    receipt = record.get("receipt")
    require(record.get("type") == "preflight"
            and record.get("schema") == "wav2vec2_fit_baseline_run_v1"
            and receipt == expected
            and record.get("expected_receipt_sha256") == expected_sha,
            "preflight record differs from external receipt")
    require(receipt.get("schema") == "wav2vec2_fit_baseline_preflight_v1"
            and receipt.get("manifest_sha256") == MANIFEST_SHA
            and receipt.get("locality_preflight_sha256") == LOCALITY_PREFLIGHT_SHA
            and receipt.get("model_id") == "facebook/wav2vec2-base-960h"
            and receipt.get("model_revision") == MODEL_REVISION
            and receipt.get("checkpoint_sha256") == WEIGHTS_SHA
            and receipt.get("device") == "cpu" and receipt.get("threads") == 4,
            "preflight model/execution identity changed")
    require(sha(LOCALITY_PREFLIGHT) == LOCALITY_PREFLIGHT_SHA,
            "prior no-model locality receipt changed")
    commit = receipt.get("runner_source_git_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            "source commit identity malformed")
    source = subprocess.run(["git", "show", f"{commit}:models/second_architecture/{RUNNER.name}"],
                            cwd=ROOT, check=False, capture_output=True)
    require(source.returncode == 0
            and hashlib.sha256(source.stdout).hexdigest() == receipt.get("runner_sha256"),
            "source commit does not bind frozen runner bytes")
    snapshot = (Path.home() / ".cache/huggingface/hub"
                / "models--facebook--wav2vec2-base-960h/snapshots" / MODEL_REVISION)
    assets = receipt.get("snapshot_files_sha256")
    require(isinstance(assets, dict) and set(assets) ==
            {"config.json", "model.safetensors", "preprocessor_config.json",
             "special_tokens_map.json", "tokenizer_config.json", "vocab.json"},
            "snapshot inventory differs")
    for name, digest in assets.items():
        require(HEX64.fullmatch(digest) is not None and sha(snapshot / name) == digest,
                f"snapshot file changed: {name}")
    require(assets["model.safetensors"] == WEIGHTS_SHA,
            "checkpoint differs from pinned weight hash")
    require(assets["config.json"] == CONFIG_SHA
            and assets["preprocessor_config.json"] == PREPROCESSOR_SHA,
            "pinned model or preprocessor config differs")
    require(assets["special_tokens_map.json"] == SPECIAL_TOKENS_SHA
            and assets["tokenizer_config.json"] == TOKENIZER_CONFIG_SHA
            and assets["vocab.json"] == VOCAB_SHA,
            "pinned tokenizer assets differ")
    source_hashes = receipt.get("source_hashes")
    require(isinstance(source_hashes, dict), "installed source hash inventory absent")
    require(source_hashes.get("transformers.models.wav2vec2.feature_extraction_wav2vec2")
            == FEATURE_EXTRACTOR_SOURCE_SHA
            and source_hashes.get("transformers.models.wav2vec2.modeling_wav2vec2")
            == MODEL_SOURCE_SHA
            and source_hashes.get("transformers.models.wav2vec2.processing_wav2vec2")
            == PROCESSOR_SOURCE_SHA
            and source_hashes.get("transformers.models.wav2vec2.tokenization_wav2vec2")
            == TOKENIZER_SOURCE_SHA,
            "pinned Wav2Vec2 implementation changed")
    import inspect
    from transformers.models.wav2vec2 import (
        feature_extraction_wav2vec2, modeling_wav2vec2,
        processing_wav2vec2, tokenization_wav2vec2,
    )
    modules = (feature_extraction_wav2vec2, modeling_wav2vec2,
               processing_wav2vec2, tokenization_wav2vec2)
    require(set(source_hashes) == {module.__name__ for module in modules},
            "installed Wav2Vec2 source module coverage differs")
    for module in modules:
        require(source_hashes[module.__name__] == sha(Path(inspect.getfile(module))),
                f"installed source changed: {module.__name__}")
    import soundfile
    import torch
    import transformers
    require(receipt.get("torch_version") == torch.__version__
            and receipt.get("transformers_version") == transformers.__version__
            and receipt.get("soundfile_version") == soundfile.__version__
            and receipt.get("python_version") == sys.version.split()[0],
            "installed runtime versions differ")
    expected_case_rows = [{"key": case_id, "pair_id": case["pair_id"],
                           "engine": case["engine"], "word": case["word"],
                           "frames": case["frames"], "slot_samples": list(case["slot"]),
                           "wav_sha256": case["wav_sha256"],
                           "pcm_sha256": case["pcm_sha256"]}
                          for case_id, case in cases.items()]
    require(receipt.get("cases") == expected_case_rows,
            "preflight FIT case coverage differs")
    require(receipt.get("modes") == list(MODES)
            and receipt.get("stage_names") == list(STAGES)
            and receipt.get("run_schema") == "wav2vec2_fit_baseline_run_v1"
            and receipt.get("expected_success_rows") == 26,
            "preflight matrix/stage declaration differs")
    normalization = receipt.get("normalization")
    require(isinstance(normalization, dict) and set(normalization) == {"S", "C", "common_stats"}
            and normalization.get("S") == "stock Wav2Vec2Processor; unpadded; per-utterance mean/variance; no attention mask"
            and normalization.get("C") == "float32 (pcm/32768 - shared outside-slot mean) / sqrt(shared outside-slot variance + 1e-7)",
            "preflight normalization declaration differs")
    expected_stats = {}
    for pair_id in PAIRS:
        for engine in ENGINES:
            first = cases[case_key(pair_id, engine, PAIRS[pair_id][0])]
            mean, variance = common_stats(first, cases)
            expected_stats[f"{pair_id}_{engine}"] = {"mean": mean, "variance": variance}
    require(normalization["common_stats"] == expected_stats,
            "preflight shared statistics differ")
    limits = receipt.get("limits")
    require(isinstance(limits, dict) and limits.get("max_seconds") == MAX_SECONDS
            and limits.get("max_rss_bytes") == MAX_RSS
            and limits.get("max_output_bytes") == MAX_OUTPUT_BYTES,
            "preflight resource limits differ")
    return receipt


def audit(output: Path, expected_preflight: Path,
          expected_preflight_sha256: str) -> dict[str, Any]:
    cases = trusted_inputs()
    rows = read_records(output / "records.jsonl")
    require(rows and rows[0].get("type") == "preflight", "preflight first row absent")
    receipt = audit_preflight(rows[0], expected_preflight,
                              expected_preflight_sha256, cases)
    if rows[-1].get("type") == "failure":
        failure = rows[-1]
        require(failure.get("schema") == "wav2vec2_fit_baseline_run_v1"
                and isinstance(failure.get("error_type"), str)
                and isinstance(failure.get("error"), str), "malformed failure record")
        provisional = failure.get("computed_output")
        if provisional is not None:
            require(isinstance(provisional, dict), "failure provisional output malformed")
            case_id, mode = provisional.get("case_key"), provisional.get("mode")
            require(case_id in cases and mode in MODES, "failure provisional identity differs")
            ids = provisional.get("argmax_ids")
            require(isinstance(ids, list) and len(ids) == conv_length(cases[case_id]["frames"])
                    and all(type(item) is int and 0 <= item < 32 for item in ids),
                    "failure provisional full CTC argmax invalid")
            from transformers import Wav2Vec2Processor
            processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h", revision=MODEL_REVISION,
                local_files_only=True)
            counts = provisional.get("hook_counts")
            require(isinstance(counts, dict) and set(counts) == set(STAGES)
                    and all(type(value) is int and value == 1 for value in counts.values()),
                    "failure provisional hook coverage differs")
            reported = {"transcript", "normalized_words", "eligible"} & set(provisional)
            require(not reported or reported == {"transcript", "normalized_words", "eligible"},
                    "failure provisional CTC fields partial")
            if reported:
                decoded = processor.batch_decode([ids])[0]
                tokens = normalized_words(decoded)
                alternative = next(word for word in PAIRS[cases[case_id]["pair_id"]]
                                   if word != cases[case_id]["word"])
                eligible = tokens.count(cases[case_id]["word"]) == 1 and alternative not in tokens
                require(provisional.get("transcript") == decoded
                        and provisional.get("normalized_words") == tokens
                        and provisional.get("eligible") is eligible,
                        "failure provisional CTC output differs")
        raise ValueError("failed attempt retained but cannot pass scientific audit")
    require(all(row.get("type") != "failure" for row in rows),
            "failure row within purported success")
    require(rows[-1].get("type") == "complete", "completion/failure row absent")
    require(len(rows) == 26, "planned 26-row matrix incomplete or extra")
    baseline_rows = [row for row in rows[1:-1] if row.get("type") == "baseline"]
    trace_rows = [row for row in rows[1:-1] if row.get("type") == "trace"]
    require(len(baseline_rows) == 16 and len(trace_rows) == 8,
            "baseline/trace matrix differs")
    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h", revision=MODEL_REVISION,
        local_files_only=True)
    checked_baselines = {}
    for row in baseline_rows:
        result = validate_baseline(row, output, cases, processor)
        key = (result["mode"], result["case_key"])
        require(key not in checked_baselines, "duplicate baseline")
        checked_baselines[key] = result
    require(set(checked_baselines) == {(mode, case_id) for mode in MODES for case_id in cases},
            "baseline case/mode coverage differs")
    checked_traces = {}
    for row in trace_rows:
        result = validate_trace(row, output, checked_baselines, cases)
        key = (result["pair_id"], result["engine"], result["mode"])
        require(key not in checked_traces, "duplicate trace")
        checked_traces[key] = result
    require(set(checked_traces) == {(pair_id, engine, mode) for pair_id in PAIRS
                                   for engine in ENGINES for mode in MODES},
            "trace coverage differs")
    complete = rows[-1]
    require(complete.get("schema") == "wav2vec2_fit_baseline_run_v1", "completion schema differs")
    require(isinstance(complete.get("elapsed_seconds"), (int, float))
            and 0 <= complete["elapsed_seconds"] <= MAX_SECONDS
            and type(complete.get("rss_bytes")) is int and 0 < complete["rss_bytes"] <= MAX_RSS,
            "completion time/memory receipt invalid")
    inventory = {path.relative_to(output).as_posix(): path.stat().st_size
                 for path in output.rglob("*") if path.is_file()}
    expected_inventory = {"records.jsonl"}
    expected_inventory.update(f"captures/{mode}_{case_id}_{stage}.npy"
                              for mode in MODES for case_id in cases for stage in STAGES)
    expected_inventory.update(f"traces/{mode}_{pair_id}_{engine}_{stage}_{kind}.npy"
                              for mode in MODES for pair_id in PAIRS for engine in ENGINES
                              for stage in STAGES for kind in ("l2", "maxabs"))
    require(set(inventory) == expected_inventory, "output file inventory differs")
    last_line_bytes = len((output / "records.jsonl").read_bytes().splitlines(keepends=True)[-1])
    require(type(complete.get("output_bytes")) is int
            and complete["output_bytes"] == sum(inventory.values()) - last_line_bytes
            and sum(inventory.values()) <= MAX_OUTPUT_BYTES,
            "completion output-byte receipt or cap differs")
    aggregate = Counter((mode, row["eligible"]) for (mode, _), row in checked_baselines.items())
    return {"status": "independent_saved_data_audit_pass",
            "records_sha256": sha(output / "records.jsonl"),
            "preflight_sha256": expected_preflight_sha256,
            "baseline_count": 16, "trace_count": 8,
            "eligibility_by_mode": {mode: {"eligible": aggregate[(mode, True)],
                                            "ineligible": aggregate[(mode, False)]}
                                    for mode in MODES},
            "baselines": [{"mode": mode, "case": case_id,
                           "transcript": row["transcript"], "eligible": row["eligible"]}
                          for (mode, case_id), row in sorted(checked_baselines.items())],
            "trace": list(checked_traces.values()),
            "limits": "Saved-data consistency only; model execution cannot be independently proven from files."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.audit_output.exists(), "audit output already exists")
    report = audit(args.output, args.expected_preflight,
                   args.expected_preflight_sha256)
    args.audit_output.write_text(json.dumps(report, sort_keys=True, indent=2,
                                           allow_nan=False) + "\n")
    print(args.audit_output)


if __name__ == "__main__":
    main()
