"""Bounded FIT-only Wav2Vec2 clean baseline and audio-stage capture.

See WAV2VEC2_FIT_BASELINE_STAGE_CAPTURE_PROSPECTIVE_V1.md. ``preflight``
does not load model weights; ``run`` requires an identical frozen receipt.
No causal intervention is performed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
LOCALITY = ROOT / "models/second_architecture/WAV2VEC2_INPUT_LOCALITY_PREFLIGHT_V1.json"
LOCALITY_SHA = "15074ffee8bde339810b4be667e3ce40bd6a45e234739ea37c614497b6eba4a2"
MODEL_ID = "facebook/wav2vec2-base-960h"
MODEL_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
WEIGHTS_SHA = "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub"
            / "models--facebook--wav2vec2-base-960h/snapshots" / MODEL_REVISION)
PAIR_WORDS = {"color_05": ("silver", "teal"), "object_02": ("plate", "cup")}
ENGINES = ("piper", "kokoro")
SLOTS = {
    ("color_05", "piper"): (30527, 49847),
    ("color_05", "kokoro"): (33920, 57120),
    ("object_02", "piper"): (30527, 42788),
    ("object_02", "kokoro"): (33920, 54320),
}
STAGES = (
    "input_values", "first_conv_raw", "first_conv_norm", "final_feature_conv",
    "feature_projection", "encoder_pre_block0", "post_block0", "post_block5",
    "post_block11", "logits",
)
MODES = ("S", "C")
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 2 * 1024**3
# The 4-GiB RSS cap is checked before/after a single stock forward. Reserve
# one GiB at admission; a CPU forward cannot be interrupted at every operator.
PRE_FORWARD_RSS_RESERVE = 1 * 1024**3
PRE_FORWARD_OUTPUT_RESERVE = 128 * 1024**2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def normalized_words(value: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", value.casefold())


@dataclass(frozen=True)
class Case:
    pair_id: str
    engine: str
    word: str
    path: Path
    wav_sha256: str
    pcm_sha256: str
    frames: int
    slot: tuple[int, int]

    @property
    def key(self) -> str:
        return f"{self.pair_id}_{self.engine}_{self.word}"


def load_pcm(case: Case) -> "np.ndarray":
    import numpy as np

    if not case.path.is_file() or case.path.is_symlink() or sha(case.path) != case.wav_sha256:
        raise ValueError(f"WAV bytes changed: {case.key}")
    with wave.open(str(case.path), "rb") as stream:
        actual = (stream.getnchannels(), stream.getsampwidth(),
                  stream.getframerate(), stream.getnframes())
        if actual != (1, 2, 16000, case.frames):
            raise ValueError(f"WAV format changed: {case.key}: {actual}")
        raw = stream.readframes(case.frames)
    if hashlib.sha256(raw).hexdigest() != case.pcm_sha256:
        raise ValueError(f"PCM bytes changed: {case.key}")
    return np.frombuffer(raw, dtype="<i2").copy()


def trusted_cases() -> tuple[list[Case], dict[tuple[str, str], dict[str, float]]]:
    import numpy as np

    if sha(MANIFEST) != MANIFEST_SHA or sha(LOCALITY) != LOCALITY_SHA:
        raise ValueError("FIT manifest or no-weight locality receipt changed")
    manifest = json.loads(MANIFEST.read_text())
    locality = json.loads(LOCALITY.read_text())
    if locality.get("schema") != "wav2vec2_input_locality_preflight_v1" or len(locality.get("records", [])) != 4:
        raise ValueError("locality receipt schema/coverage changed")
    cases: list[Case] = []
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for pair_id, words in PAIR_WORDS.items():
        for engine in ENGINES:
            selected = [p for p in manifest["pairs"] if p["pair_id"] == pair_id
                        and p["engine"] == engine and p["phrasing_id"] == "fit_chosen"]
            if len(selected) != 1:
                raise ValueError(f"FIT pair missing/duplicated: {pair_id}/{engine}")
            pair = selected[0]
            slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
            if pair["split"] != "fit_pool" or pair["map_fit"] is not True \
                    or tuple(pair["words"]) != words or slot != SLOTS[(pair_id, engine)]:
                raise ValueError(f"FIT pair identity changed: {pair_id}/{engine}")
            if pair["outside_slot_pcm_identical"] is not True or pair["inside_slot_pcm_differs"] is not True:
                raise ValueError("pair PCM assertions absent")
            selected_locality = [r for r in locality["records"] if r["pair_id"] == pair_id
                                 and r["engine"] == engine]
            if len(selected_locality) != 1 or selected_locality[0]["slot_samples"] != list(slot) \
                    or tuple(selected_locality[0]["words"]) != words:
                raise ValueError("locality receipt does not match pair")
            pair_cases = []
            for word in words:
                info = pair["outputs"][word]
                case = Case(pair_id, engine, word, ROOT / info["path"],
                            info["file_sha256"], info["pcm_int16_sha256"],
                            int(info["frames"]), slot)
                pair_cases.append(case)
                cases.append(case)
            a, b = [load_pcm(case) for case in pair_cases]
            s, e = slot
            if len(a) != len(b) or not np.array_equal(a[:s], b[:s]) \
                    or not np.array_equal(a[e:], b[e:]) or np.array_equal(a[s:e], b[s:e]):
                raise ValueError("actual exact-PCM contrast invalid")
            if len(a) != selected_locality[0]["frames"]:
                raise ValueError("locality receipt length mismatch")
            outside = np.concatenate((a[:s], a[e:])).astype(np.float32) / np.float32(32768.0)
            mu, var = np.float32(outside.mean()), np.float32(outside.var())
            if not np.isfinite(mu) or not np.isfinite(var) or var <= 0:
                raise ValueError("common normalization statistics invalid")
            stats[(pair_id, engine)] = {"mean": float(mu), "variance": float(var)}
    if len(cases) != 8 or len(stats) != 4:
        raise ValueError("FIT case coverage incomplete")
    return cases, stats


def normalized_input(pcm: "np.ndarray", *, mode: str, stats: dict[str, float],
                     processor: Any | None = None) -> "np.ndarray":
    import numpy as np

    x = pcm.astype(np.float32) / np.float32(32768.0)
    if mode == "C":
        mu, var = np.float32(stats["mean"]), np.float32(stats["variance"])
        return np.asarray((x - mu) / np.sqrt(var + np.float32(1e-7)), dtype=np.float32)
    if mode != "S" or processor is None:
        raise ValueError("unknown mode or stock processor absent")
    batch = processor(x, sampling_rate=16000, padding=False,
                      return_attention_mask=False, return_tensors="np")
    if set(batch.keys()) != {"input_values"}:
        raise ValueError("stock processor returned unexpected fields")
    values = np.asarray(batch["input_values"], dtype=np.float32)
    if values.shape != (1, len(x)) or not np.isfinite(values).all():
        raise ValueError("stock processor input shape/finite check failed")
    return values[0].copy()


def source_commit() -> str:
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    for args in (("git", "diff", "--quiet", "--", relative),
                 ("git", "diff", "--cached", "--quiet", "--", relative)):
        result = subprocess.run(args, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise ValueError("runner source has uncommitted changes")
    revision = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True
    ).strip()
    if len(revision) != 40:
        raise ValueError("runner has no committed source revision")
    return revision


def preflight() -> dict[str, Any]:
    import soundfile
    import torch
    import transformers
    from transformers.models.wav2vec2 import (feature_extraction_wav2vec2,
                                             modeling_wav2vec2,
                                             processing_wav2vec2,
                                             tokenization_wav2vec2)

    cases, stats = trusted_cases()
    snapshot = SNAPSHOT
    if not snapshot.is_dir():
        raise ValueError("pinned local Wav2Vec2 snapshot absent")
    weights = snapshot / "model.safetensors"
    if not weights.is_file() or sha(weights) != WEIGHTS_SHA:
        raise ValueError("Wav2Vec2 checkpoint bytes changed")
    config = json.loads((snapshot / "config.json").read_text())
    expected = {"num_hidden_layers": 12, "hidden_size": 768, "vocab_size": 32,
                "feat_extract_norm": "group", "conv_stride": [5, 2, 2, 2, 2, 2, 2],
                "conv_kernel": [10, 3, 3, 3, 3, 2, 2]}
    if {key: config[key] for key in expected} != expected:
        raise ValueError("pinned Wav2Vec2 architecture changed")
    assets = {path.name: sha(path) for path in sorted(snapshot.iterdir()) if path.is_file()}
    if len(assets) != 6 or assets.get("model.safetensors") != WEIGHTS_SHA:
        raise ValueError("pinned Wav2Vec2 snapshot inventory changed")
    modules = (feature_extraction_wav2vec2, modeling_wav2vec2,
               processing_wav2vec2, tokenization_wav2vec2)
    hashes = {module.__name__: sha(Path(inspect.getfile(module))) for module in modules}
    return {
        "schema": "wav2vec2_fit_baseline_preflight_v1",
        "runner_sha256": sha(Path(__file__)),
        "runner_source_git_commit": source_commit(),
        "manifest_sha256": MANIFEST_SHA,
        "locality_preflight_sha256": LOCALITY_SHA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": WEIGHTS_SHA,
        "snapshot_files_sha256": assets,
        "source_hashes": hashes,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "soundfile_version": soundfile.__version__,
        "python_version": sys.version.split()[0],
        "device": "cpu", "threads": 4,
        "modes": list(MODES), "stage_names": list(STAGES),
        "normalization": {
            "S": "stock Wav2Vec2Processor; unpadded; per-utterance mean/variance; no attention mask",
            "C": "float32 (pcm/32768 - shared outside-slot mean) / sqrt(shared outside-slot variance + 1e-7)",
            "common_stats": {f"{p}_{e}": stats[(p, e)] for p in PAIR_WORDS for e in ENGINES},
        },
        "limits": {"max_seconds": MAX_SECONDS, "max_rss_bytes": MAX_RSS_BYTES,
                   "max_output_bytes": MAX_OUTPUT_BYTES,
                   "preforward_rss_reserve": PRE_FORWARD_RSS_RESERVE,
                   "preforward_output_reserve": PRE_FORWARD_OUTPUT_RESERVE,
                   "budget_check_limit": "operator-level forward peak is verified after return, not interruptible within one forward"},
        "cases": [{"key": case.key, "pair_id": case.pair_id, "engine": case.engine,
                   "word": case.word, "frames": case.frames, "slot_samples": list(case.slot),
                   "wav_sha256": case.wav_sha256, "pcm_sha256": case.pcm_sha256}
                  for case in cases],
        "run_schema": "wav2vec2_fit_baseline_run_v1",
        "expected_success_rows": 26,
    }


def current_rss_bytes() -> int:
    # ru_maxrss is in bytes on macOS; this runner is pinned to a Mac host.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def output_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def check_budget(start: float, output: Path, *, before_forward: bool = False) -> None:
    if time.monotonic() - start > MAX_SECONDS:
        raise RuntimeError("30-minute time budget exhausted")
    rss = current_rss_bytes()
    disk = output_bytes(output)
    if rss > MAX_RSS_BYTES or disk > MAX_OUTPUT_BYTES:
        raise RuntimeError("RSS or output-directory budget exhausted")
    if before_forward and (rss + PRE_FORWARD_RSS_RESERVE > MAX_RSS_BYTES
                           or disk + PRE_FORWARD_OUTPUT_RESERVE > MAX_OUTPUT_BYTES):
        raise RuntimeError("insufficient reserved RSS/disk headroom before forward")


def save_array(path: Path, value: "np.ndarray", output: Path, start: float) -> str:
    import numpy as np

    if path.exists():
        raise FileExistsError(f"capture already exists: {path}")
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"capture not finite float32: {path}")
    # NPY header is small; reserve 1 MiB above exact payload before any write.
    if output_bytes(output) + value.nbytes + (1 << 20) > MAX_OUTPUT_BYTES:
        raise RuntimeError("capture would exceed output-directory cap")
    check_budget(start, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.exists():
        raise FileExistsError(f"capture temp already exists: {temp}")
    try:
        with temp.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if output_bytes(output) > MAX_OUTPUT_BYTES:
            raise RuntimeError("capture write exceeded output-directory cap")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    check_budget(start, output)
    return sha(path)


def capture_modules(model: Any) -> dict[str, tuple[Any, str]]:
    wav = model.wav2vec2
    conv0 = wav.feature_extractor.conv_layers[0]
    return {
        "first_conv_raw": (conv0.conv, "post"),
        "first_conv_norm": (conv0.layer_norm, "post"),
        "final_feature_conv": (wav.feature_extractor, "post"),
        "feature_projection": (wav.feature_projection, "post_first"),
        "encoder_pre_block0": (wav.encoder.layers[0], "pre"),
        "post_block0": (wav.encoder.layers[0], "post_first"),
        "post_block5": (wav.encoder.layers[5], "post_first"),
        "post_block11": (wav.encoder.layers[11], "post_first"),
        "logits": (model.lm_head, "post"),
    }


def validate_tensor(stage: str, tensor: Any, frames: int) -> "np.ndarray":
    import torch

    first_frames = (frames - 10) // 5 + 1
    final_frames = (frames - 400) // 320 + 1
    shapes = {
        "input_values": (1, frames),
        "first_conv_raw": (1, 512, first_frames),
        "first_conv_norm": (1, 512, first_frames),
        "final_feature_conv": (1, 512, final_frames),
        "feature_projection": (1, final_frames, 768),
        "encoder_pre_block0": (1, final_frames, 768),
        "post_block0": (1, final_frames, 768),
        "post_block5": (1, final_frames, 768),
        "post_block11": (1, final_frames, 768),
        "logits": (1, final_frames, 32),
    }
    if stage not in shapes or not torch.is_tensor(tensor) or tuple(tensor.shape) != shapes[stage]:
        raise ValueError(f"capture shape mismatch at {stage}: {getattr(tensor, 'shape', None)}")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"capture dtype/device/finite mismatch at {stage}")
    return tensor.detach().cpu().contiguous().numpy().copy()


def model_forward(model: Any, input_values: Any, frames: int) -> tuple[dict[str, "np.ndarray"], dict[str, int]]:
    import torch

    captures: dict[str, Any] = {"input_values": input_values.detach().clone()}
    counts = {stage: (1 if stage == "input_values" else 0) for stage in STAGES}
    handles = []
    for stage, (module, kind) in capture_modules(model).items():
        if kind == "pre":
            def hook(_module: Any, args: tuple[Any, ...], stage: str = stage) -> None:
                counts[stage] += 1
                if counts[stage] != 1:
                    raise ValueError(f"hook fired more than once at {stage}")
                captures[stage] = args[0].detach().clone()
            handles.append(module.register_forward_pre_hook(hook))
        else:
            def hook(_module: Any, _args: tuple[Any, ...], result: Any,
                     stage: str = stage, kind: str = kind) -> None:
                counts[stage] += 1
                if counts[stage] != 1:
                    raise ValueError(f"hook fired more than once at {stage}")
                tensor = result[0] if kind == "post_first" else result
                captures[stage] = tensor.detach().clone()
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            result = model(input_values=input_values, attention_mask=None,
                           output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(count != 1 for count in counts.values()) or set(captures) != set(STAGES):
        raise ValueError(f"stage hook coverage mismatch: {counts}")
    if not torch.equal(result.logits, captures["logits"]):
        raise ValueError("lm_head capture differs from returned logits")
    arrays = {stage: validate_tensor(stage, captures[stage], frames) for stage in STAGES}
    return arrays, counts


def stage_window(case: Case, stage: str, length: int) -> tuple[int, int]:
    s, e = case.slot
    if stage == "input_values":
        start, end = s, e
    elif stage in ("first_conv_raw", "first_conv_norm"):
        start, end = math.ceil((s - 9) / 5), math.floor((e - 1) / 5) + 1
    else:
        start, end = math.ceil((s - 399) / 320), math.floor((e - 1) / 320) + 1
    return max(0, start), min(length, end)


def trace_arrays(donor: "np.ndarray", recipient: "np.ndarray", stage: str,
                 case: Case) -> tuple["np.ndarray", "np.ndarray", dict[str, Any]]:
    import numpy as np

    if donor.shape != recipient.shape or donor.dtype != np.float32 or recipient.dtype != np.float32:
        raise ValueError("trace source shape/dtype mismatch")
    if stage == "input_values":
        positions = donor.shape[1]
        difference = (donor[0].astype(np.float64) - recipient[0].astype(np.float64))[:, None]
    elif stage in ("first_conv_raw", "first_conv_norm", "final_feature_conv"):
        positions = donor.shape[2]
        difference = donor[0].T.astype(np.float64) - recipient[0].T.astype(np.float64)
    else:
        positions = donor.shape[1]
        difference = donor[0].astype(np.float64) - recipient[0].astype(np.float64)
    l2 = np.sqrt(np.sum(difference * difference, axis=-1)).astype(np.float32)
    maxabs = np.max(np.abs(difference), axis=-1).astype(np.float32)
    window = stage_window(case, stage, positions)
    if not 0 <= window[0] < window[1] <= positions:
        raise ValueError("invalid nominal stage window")
    outside = np.r_[maxabs[:window[0]], maxabs[window[1]:]]
    first = np.flatnonzero(maxabs > 0)
    summary = {"nominal_window": list(window),
               "first_nonzero_index": int(first[0]) if len(first) else None,
               "offslot_maxabs": float(outside.max()) if len(outside) else 0.0}
    return l2, maxabs, summary


def append_record(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute(expected_preflight: Path, output: Path) -> None:
    import numpy as np
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    current = preflight()
    if canonical(json.loads(expected_preflight.read_text())) != canonical(current):
        raise ValueError("frozen preflight differs from current source/input/environment")
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.mkdir(parents=True)
    append_record(output / "records.jsonl", {
        "type": "preflight", "schema": "wav2vec2_fit_baseline_run_v1",
        "receipt": current, "expected_receipt_sha256": sha(expected_preflight),
    })
    start = time.monotonic()
    try:
        check_budget(start, output, before_forward=True)
        torch.set_num_threads(4)
        processor = Wav2Vec2Processor.from_pretrained(SNAPSHOT, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(SNAPSHOT, local_files_only=True).to(
            "cpu", dtype=torch.float32).eval()
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("model not entirely CPU float32")
        cases, stats = trusted_cases()
        in_progress: dict[str, Any] | None = None
        for case in cases:
            pcm = load_pcm(case)
            pair_stats = stats[(case.pair_id, case.engine)]
            for mode in MODES:
                check_budget(start, output, before_forward=True)
                values = normalized_input(pcm, mode=mode, stats=pair_stats, processor=processor)
                if len(values) != case.frames or not np.isfinite(values).all():
                    raise ValueError("normalized input invalid")
                if mode == "C":
                    s, e = case.slot
                    other = next(c for c in cases if c.pair_id == case.pair_id and c.engine == case.engine
                                 and c.word != case.word)
                    other_values = normalized_input(load_pcm(other), mode="C", stats=pair_stats)
                    if not np.array_equal(values[:s], other_values[:s]) \
                            or not np.array_equal(values[e:], other_values[e:]):
                        raise ValueError("common-normalized inputs differ outside slot")
                input_tensor = torch.from_numpy(values.copy())[None]
                forward_start = time.monotonic()
                arrays, counts = model_forward(model, input_tensor, case.frames)
                ids = arrays["logits"].argmax(axis=-1)[0].astype(np.int64).tolist()
                transcript = processor.batch_decode([ids])[0]
                target, other_word = case.word, next(w for w in PAIR_WORDS[case.pair_id] if w != target)
                tokens = normalized_words(transcript)
                eligible = tokens.count(target) == 1 and tokens.count(other_word) == 0
                in_progress = {"case_key": case.key, "mode": mode, "argmax_ids": ids,
                               "hook_counts": counts,
                               "transcript": transcript, "normalized_words": tokens,
                               "eligible": eligible}
                check_budget(start, output)
                hashes: dict[str, str] = {}
                shapes: dict[str, list[int]] = {}
                for stage in STAGES:
                    path = output / "captures" / f"{mode}_{case.key}_{stage}.npy"
                    hashes[stage] = save_array(path, arrays[stage], output, start)
                    shapes[stage] = list(arrays[stage].shape)
                append_record(output / "records.jsonl", {
                    "type": "baseline", "schema": "wav2vec2_fit_baseline_run_v1",
                    "case_key": case.key,
                    "case": {"pair_id": case.pair_id, "engine": case.engine,
                             "word": case.word},
                    "pair_id": case.pair_id, "engine": case.engine, "word": case.word,
                    "mode": mode,
                    "common_mean": pair_stats["mean"] if mode == "C" else None,
                    "common_variance": pair_stats["variance"] if mode == "C" else None,
                    "wav_sha256": case.wav_sha256, "pcm_sha256": case.pcm_sha256,
                    "input_sha256": hashes["input_values"],
                    "capture_sha256": hashes, "capture_shapes": shapes,
                    "logits_sha256": hashes["logits"], "logits_shape": shapes["logits"],
                    "hook_counts": counts, "argmax_ids": ids,
                    "transcript": transcript, "normalized_words": tokens,
                    "eligible": eligible, "elapsed_seconds": time.monotonic() - forward_start,
                    "rss_bytes": current_rss_bytes(),
                })
                in_progress = None
                del arrays, input_tensor, values
                check_budget(start, output)
        for pair_id, words in PAIR_WORDS.items():
            for engine in ENGINES:
                case = next(c for c in cases if c.pair_id == pair_id and c.engine == engine)
                for mode in MODES:
                    trace = {}
                    for stage in STAGES:
                        donor_path = output / "captures" / f"{mode}_{pair_id}_{engine}_{words[1]}_{stage}.npy"
                        recipient_path = output / "captures" / f"{mode}_{pair_id}_{engine}_{words[0]}_{stage}.npy"
                        donor = np.load(donor_path, mmap_mode="r", allow_pickle=False)
                        recipient = np.load(recipient_path, mmap_mode="r", allow_pickle=False)
                        l2, maxabs, summary = trace_arrays(donor, recipient, stage, case)
                        del donor, recipient
                        l2_path = output / "traces" / f"{mode}_{pair_id}_{engine}_{stage}_l2.npy"
                        max_path = output / "traces" / f"{mode}_{pair_id}_{engine}_{stage}_maxabs.npy"
                        summary["l2_sha256"] = save_array(l2_path, l2, output, start)
                        summary["maxabs_sha256"] = save_array(max_path, maxabs, output, start)
                        trace[stage] = summary
                    append_record(output / "records.jsonl", {
                        "type": "trace", "schema": "wav2vec2_fit_baseline_run_v1",
                        "pair_id": pair_id, "engine": engine, "mode": mode,
                        "words": list(words), "stages": trace,
                    })
                    check_budget(start, output)
        check_budget(start, output)
        append_record(output / "records.jsonl", {
            "type": "complete", "schema": "wav2vec2_fit_baseline_run_v1",
            "elapsed_seconds": time.monotonic() - start,
            "rss_bytes": current_rss_bytes(), "output_bytes": output_bytes(output),
        })
    except BaseException as exc:
        append_record(output / "records.jsonl", {
            "type": "failure", "schema": "wav2vec2_fit_baseline_run_v1",
            "error_type": type(exc).__name__, "error": str(exc),
            "computed_output": in_progress if "in_progress" in locals() else None,
            "elapsed_seconds": time.monotonic() - start,
            "rss_bytes": current_rss_bytes(), "output_bytes": output_bytes(output),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--expected-preflight", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        if args.output.exists():
            raise FileExistsError("preflight receipt already exists")
        receipt = preflight()
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n")
        print(args.output)
    else:
        execute(args.expected_preflight, args.output)


if __name__ == "__main__":
    main()
