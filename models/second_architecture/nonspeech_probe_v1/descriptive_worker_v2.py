"""V2 remaining-model descriptive runner. No causal edits.

The parent must validate an independent GO before invoking this CLI. Importing
this file never imports Transformers or loads weights. All asset access is local.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import resource
import subprocess
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any

from . import render
from . import descriptive_worker_v1 as base

HERE = Path(__file__).resolve().parent
PINS = HERE / "descriptive_pins_v1.json"
MANIFEST = HERE / "runs/render_20261119_0740Z/manifest.json"
MANIFEST_SHA256 = "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90"
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_EVIDENCE_BYTES = 128 * 1024**2
MAX_NEW_TOKENS = 128
MODEL_NAMES = ("whisper_english", "wav2vec2")
V1_RECEIPT = HERE / "runs/descriptive_source_preflight_v2/preflight.json"
V1_GO = HERE / "INDEPENDENT_DESCRIPTIVE_SOURCE_GO_V2.json"
MODEL_PATHS = {
    "whisper_english": {
        "attempt": HERE / "runs/descriptive_whisper_english_20261119_v2",
        "receipt": HERE / "runs/descriptive_remaining_repair_preflight_v2/whisper_english.json",
        "review": HERE / "INDEPENDENT_DESCRIPTIVE_WHISPER_ENGLISH_REPAIR_GO_V2.json",
        "prior": HERE / "runs/descriptive_whisper_english_20261119_v1",
        "prior_sha256": {"parent_result.json": "d43ae9bd4faa5dae40f78c6c8a72708d5bbb68ad767de1671ecc58fcd4b9e4d2",
                         "records.jsonl": "687d03e967f3fb6d12904a6a72e01ddb090f7a162d8a7c87f259087c9c17911b"},
    },
    "wav2vec2": {
        "attempt": HERE / "runs/descriptive_wav2vec2_20261119_v2",
        "receipt": HERE / "runs/descriptive_remaining_repair_preflight_v2/wav2vec2.json",
        "review": HERE / "INDEPENDENT_DESCRIPTIVE_WAV2VEC2_REPAIR_GO_V2.json",
        "prior": HERE / "runs/descriptive_wav2vec2_20261119_v1",
        "prior_sha256": {"parent_result.json": "dd201e2677b1ed36eaa772bf68b7db7990bae120e1070bae25c0cfc6ef48c7f6",
                         "records.jsonl": "dd35532374cacdacbf8faf9eec172374f86b0d7330e6f46b279786e1fd2fae0f"},
    },
}
V2_SOURCE_NAMES = (
    "descriptive_worker_v2.py", "descriptive_remaining_parent_v2.py",
    "descriptive_remaining_audit_v2.py", "test_descriptive_remaining_repair_v2.py",
    "DESCRIPTIVE_REMAINING_WORKER_REPAIR_PROSPECTIVE_V2.md",
)
WHISPER_STAGES = ("mel", "post_conv_pos", "post_block0", "final_encoder")
WAV_STAGES = ("input_values", "first_conv_raw", "first_conv_norm", "feature_projection",
              "post_block0", "final_encoder", "logits")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_identity() -> dict[str, str]:
    return {name: sha_file(HERE / name) for name in (
        *V2_SOURCE_NAMES, "descriptive_worker_v1.py", "descriptive_parent_v1.py",
        "descriptive_audit_v1.py", "descriptive_parent_v2.py", "descriptive_audit_v2.py",
        "descriptive_pins_v1.json",
        "whisper_tiny_metadata_pin.json", "spec.json", "render.py",
        "NON_SPEECH_MECHANISM_PROSPECTIVE_V1.md")}


def _snapshot(model_name: str, model_spec: dict) -> Path:
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    repo = model_spec["repo"]
    require("/" in repo and ".." not in repo, "invalid pinned repository")
    path = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / model_spec["revision"]
    require(path.is_dir(), f"pinned local snapshot unavailable: {model_name}")
    return path


def verify_local_assets(model_name: str) -> dict:
    """Read and hash pinned files without deserializing candidate weights."""
    require(model_name in MODEL_NAMES, "unknown model")
    pins = json.loads(PINS.read_text())
    spec = pins["models"][model_name]
    snapshot = _snapshot(model_name, spec)
    files = dict(spec.get("files_sha256", {}))
    if "metadata_file" in spec:
        metadata = json.loads((HERE / spec["metadata_file"]).read_text())
        require(metadata["repo"] == spec["repo"] and metadata["revision"] == spec["revision"],
                "multilingual metadata pin differs")
        files.update(metadata["files_sha256"])
    files["model.safetensors"] = spec["weight_sha256"]
    actual = {}
    for name, digest in sorted(files.items()):
        path = snapshot / name
        require(path.is_file(), f"pinned asset unavailable: {name}")
        observed = sha_file(path)
        require(observed == digest, f"pinned asset bytes changed: {name}")
        actual[name] = observed
    weight_bytes = (snapshot / "model.safetensors").stat().st_size
    if "weight_bytes" in spec:
        require(weight_bytes == spec["weight_bytes"], "pinned weight byte count changed")
    return {"repo": spec["repo"], "revision": spec["revision"], "snapshot": str(snapshot),
            "weight_bytes": weight_bytes, "files_sha256": actual}


def verify_installed_sources() -> dict:
    import numpy as np
    import torch
    import transformers
    pins = json.loads(PINS.read_text())
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__}
    require(versions == pins["versions"], "runtime package versions changed")
    actual = {}
    for name, digest in pins["source_sha256"].items():
        module = importlib.import_module(name)
        actual[name] = sha_file(Path(inspect.getfile(module)))
        require(actual[name] == digest, f"installed package source changed: {name}")
    return {"versions": versions, "source_sha256": actual}


def trusted_cases() -> list[dict]:
    require(MANIFEST.is_file() and sha_file(MANIFEST) == MANIFEST_SHA256,
            "audited source manifest bytes changed or unavailable")
    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("status") == "complete_source_only_no_model_inference"
            and manifest.get("spec_sha256") == render.SPEC_SHA256, "source manifest changed")
    rows = manifest.get("rows")
    require(isinstance(rows, list) and len(rows) == 20, "source case count changed")
    expected = {name: pcm for name, pcm in render.cases(render.read_spec())}
    cases = []
    for row in rows:
        name = row.get("case_id")
        require(name in expected and row.get("wav") == name + ".wav", "source case ID changed")
        path = MANIFEST.parent / row["wav"]
        require(path.is_file() and not path.is_symlink() and sha_file(path) == row["wav_sha256"],
                f"source WAV bytes changed: {name}")
        with wave.open(str(path), "rb") as wav:
            require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes())
                    == (1, 2, 16000, len(expected[name])), f"source WAV format changed: {name}")
            raw = wav.readframes(wav.getnframes())
        require(sha_bytes(raw) == row["pcm_sha256"] == sha_bytes(expected[name].astype("<i2").tobytes()),
                f"source PCM bytes changed: {name}")
        cases.append({"case_id": name, "frames": len(expected[name]),
                      "wav_sha256": row["wav_sha256"], "pcm_sha256": row["pcm_sha256"],
                      "pcm": expected[name]})
    require(set(expected) == {case["case_id"] for case in cases}, "source coverage changed")
    return sorted(cases, key=lambda row: row["case_id"])


def verify_prior_failure(model_name: str) -> dict:
    require(model_name in MODEL_NAMES, "unknown remaining model")
    spec = MODEL_PATHS[model_name]
    prior = spec["prior"]
    require(prior.is_dir() and not prior.is_symlink(), "frozen failed V1 attempt unavailable")
    for name, digest in spec["prior_sha256"].items():
        path = prior / name
        require(path.is_file() and not path.is_symlink() and sha_file(path) == digest,
                f"frozen failed V1 attempt bytes changed: {name}")
    return {"prior_attempt": str(prior.resolve()),
            "status": "invalid_preserved_by_parent_report",
            "files_sha256": spec["prior_sha256"]}


def preflight_receipt(model_name: str) -> dict:
    """Authenticate every local asset and source; this never loads candidate weights."""
    require(model_name in MODEL_NAMES, "unknown remaining model")
    base.require_independent_go(V1_RECEIPT, V1_GO)
    cases = trusted_cases()
    paths = MODEL_PATHS[model_name]
    return {"schema": "nonspeech_descriptive_remaining_repair_preflight_v2",
            "status": "source_only_no_model_load",
            "model": model_name,
            "attempt_path": str(paths["attempt"].resolve()),
            "review_path": str(paths["review"].resolve()),
            "prior_failed_attempt": verify_prior_failure(model_name),
            "v1_receipt_sha256": sha_file(V1_RECEIPT),
            "v1_go_sha256": sha_file(V1_GO),
            "manifest_sha256": MANIFEST_SHA256,
            "source_identity": source_identity(),
            "host_gate_source_sha256": sha_file(
                HERE.parent / "voxtral_fp32_nonbeaver_linearity_continuation_gate_v1.py"),
            "model_assets": verify_local_assets(model_name),
            "installed": verify_installed_sources(),
            "case_ids": [case["case_id"] for case in cases],
            "limits": {"seconds_per_model": MAX_SECONDS,
                       "rss_bytes": MAX_RSS_BYTES,
                       "evidence_bytes": MAX_EVIDENCE_BYTES,
                       "max_new_tokens": MAX_NEW_TOKENS}}


def _file_commit(path: Path) -> str:
    result = subprocess.run(("git", "log", "-1", "--format=%H", "--", str(path)),
                            cwd=HERE.parents[2], text=True, capture_output=True, check=True)
    require(bool(result.stdout.strip()), f"file is not committed: {path.name}")
    return result.stdout.strip()


def require_independent_go(model_name: str, receipt_path: Path,
                           review_path: Path) -> dict:
    require(model_name in MODEL_NAMES, "unknown remaining model")
    paths = MODEL_PATHS[model_name]
    require(receipt_path.resolve() == paths["receipt"].resolve()
            and review_path.resolve() == paths["review"].resolve(),
            "model-specific V2 receipt/review paths changed")
    require(receipt_path.is_file() and review_path.is_file()
            and not receipt_path.is_symlink() and not review_path.is_symlink(),
            "source preflight or independent review missing")
    receipt = json.loads(receipt_path.read_text())
    require(receipt == preflight_receipt(model_name), "source preflight receipt differs")
    review = json.loads(review_path.read_text())
    validate_review_fields(model_name, review, receipt, sha_file(receipt_path))
    review_commit = _file_commit(review_path)
    for name in V2_SOURCE_NAMES:
        source_commit = _file_commit(HERE / name)
        require(source_commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", source_commit, review_commit),
            cwd=HERE.parents[2], check=False).returncode == 0,
            f"independent review must follow frozen source: {name}")
    return review


def validate_review_fields(model_name: str, review: dict, receipt: dict,
                           receipt_sha256: str) -> None:
    require(model_name in MODEL_NAMES, "unknown remaining model")
    require(review.get("schema") == "nonspeech_descriptive_remaining_repair_review_v2"
            and review.get("status") == "GO"
            and review.get("model") == model_name
            and review.get("attempt_path") == receipt["attempt_path"]
            and review.get("preflight_sha256") == receipt_sha256
            and review.get("source_identity") == receipt["source_identity"]
            and review.get("prior_failed_attempt") == receipt["prior_failed_attempt"]
            and review.get("manifest_sha256") == MANIFEST_SHA256,
            "independent V2 model GO does not bind current source and assets")


def _array(tensor: Any) -> Any:
    import numpy as np
    import torch
    require(torch.is_tensor(tensor) and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
            "capture must be finite CPU FP32 tensor")
    return np.ascontiguousarray(tensor.detach().numpy().copy())


def _time_axis(stage: str, array: Any) -> int:
    if stage in ("mel", "first_conv_raw", "first_conv_norm"):
        return 2
    if stage == "input_values":
        return 1
    require(array.ndim == 3, "hidden capture must be rank 3")
    return 1


def trace_metric(stage: str, array: Any) -> dict:
    import numpy as np
    axis = _time_axis(stage, array)
    moved = np.moveaxis(array[0], axis - 1, 0)
    values = moved.astype(np.float64)
    norms = (np.abs(values) if values.ndim == 1 else
             np.linalg.norm(values, axis=tuple(range(1, values.ndim))))
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": sha_bytes(array.tobytes()), "time_axis": axis,
            "row_l2": norms.tolist()}


def trace_difference(stage: str, a: Any, silence: Any) -> dict:
    import numpy as np
    require(a.shape == silence.shape, "same-duration silence trace shape changed")
    delta = a.astype(np.float64) - silence.astype(np.float64)
    axis = _time_axis(stage, a)
    moved = np.moveaxis(delta[0], axis - 1, 0)
    norms = (np.abs(moved) if moved.ndim == 1 else
             np.linalg.norm(moved, axis=tuple(range(1, moved.ndim))))
    return {"whole_l2": float(np.linalg.norm(delta)), "row_l2": norms.tolist(),
            "nonzero_rows": int(np.count_nonzero(norms))}


def scripts(text: str) -> list[str]:
    result = set()
    for char in text:
        if unicodedata.category(char).startswith("L"):
            name = unicodedata.name(char, "UNKNOWN")
            result.add(name.split()[0])
    return sorted(result)


def lexical_nonempty(text: str) -> bool:
    return any(unicodedata.category(char).startswith(("L", "N")) for char in text)


def whisper_geometry(mask: Any, pcm_frames: int, mel_frames: int = 3000) -> dict:
    """Bind processor-emitted mask and the nominal 2:1 encoder time map."""
    import torch
    require(torch.is_tensor(mask) and tuple(mask.shape) == (1, mel_frames),
            "Whisper processor attention-mask geometry changed")
    values = [int(x) for x in mask[0].tolist()]
    require(set(values) <= {0, 1}, "Whisper mask is not binary")
    active = sum(values)
    expected = (pcm_frames + 159) // 160
    require(active == expected and values == [1] * active + [0] * (mel_frames - active),
            "Whisper mask no longer marks exact active mel prefix")
    return {"pcm_frames": pcm_frames, "mel_frames": mel_frames,
            "attention_mask": values, "active_mel_frames": active,
            "encoder_frames": mel_frames // 2,
            "nominal_active_encoder_rows": (active + 1) // 2,
            "encoder_time_map": "conv1(k3,s1,p1),conv2(k3,s2,p1); nominal row r uses mel centers 2r",
            "mask_sha256": sha_bytes(bytes(values))}


def wav_geometry(pcm_frames: int, ctc_frames: int) -> dict:
    length = pcm_frames
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2),
                              (5, 2, 2, 2, 2, 2, 2), strict=True):
        length = (length - kernel) // stride + 1
    require(length == ctc_frames, "Wav2Vec2 CTC frame geometry changed")
    return {"pcm_frames": pcm_frames, "attention_mask_kind": "implicit_all_active_unpadded",
            "active_input_samples": pcm_frames, "ctc_frames": ctc_frames,
            "active_ctc_frames": ctc_frames}


def score_whisper(result: Any, tokenizer: Any, detection: dict | None,
                  *, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Align one score to one actual generation step; never strip trailing pad=eos."""
    import torch
    seq = result.sequences
    scores = result.scores
    raw_logits = result.logits
    decoder_states = result.decoder_hidden_states
    require(torch.is_tensor(seq) and seq.ndim == 2 and seq.shape[0] == 1,
            "Whisper output sequence shape changed")
    require(isinstance(scores, (tuple, list)) and 1 <= len(scores) <= max_new_tokens,
            "Whisper score-step count changed")
    require(isinstance(raw_logits, (tuple, list)) and len(raw_logits) == len(scores)
            and isinstance(decoder_states, (tuple, list))
            and len(decoder_states) == len(scores),
            "Whisper raw-logit or decoder-state step count changed")
    ids = [int(v) for v in seq[0].tolist()]
    steps = len(scores)
    require(len(ids) > steps, "Whisper prefix or generated steps missing")
    prefix, generated = ids[:-steps], ids[-steps:]
    eos = int(tokenizer.eos_token_id)
    require(eos == int(tokenizer.pad_token_id), "Whisper pad/eos identity changed")
    score_rows = []
    for token, scores_t, raw_t in zip(generated, scores, raw_logits, strict=True):
        require(torch.is_tensor(scores_t) and scores_t.shape[0] == 1
                and scores_t.ndim == 2 and token < scores_t.shape[1],
                "Whisper score tensor changed")
        require(torch.is_tensor(raw_t) and raw_t.ndim == 2
                and raw_t.shape == scores_t.shape and raw_t.dtype == torch.float32,
                "Whisper raw logit tensor changed")
        vec = scores_t[0].detach().to("cpu", dtype=torch.float32)
        raw_vec = raw_t[0].detach().to("cpu", dtype=torch.float32)
        require(bool(torch.isfinite(vec).any()), "all Whisper logits nonfinite")
        require(bool(torch.isfinite(raw_vec).all()), "raw Whisper logits nonfinite")
        score_rows.append({"token_id": token, "selected_logit": float(vec[token]),
                           "selected_logprob": float(torch.log_softmax(vec, -1)[token]),
                           "top_id": int(torch.argmax(vec)),
                           "processed_logits_sha256": sha_bytes(vec.numpy().tobytes()),
                           "raw_selected_logit": float(raw_vec[token]),
                           "raw_top_id": int(torch.argmax(raw_vec)),
                           "raw_logits_sha256": sha_bytes(raw_vec.numpy().tobytes())})
    first_hidden = decoder_states[0][-1]
    require(torch.is_tensor(first_hidden) and first_hidden.ndim == 3
            and first_hidden.shape[0] == 1 and first_hidden.shape[1] == len(prefix)
            and first_hidden.shape[2] == 384
            and first_hidden.dtype == torch.float32 and bool(torch.isfinite(first_hidden).all()),
            "Whisper first decoder-step final-layer state changed")
    first_hidden_array = first_hidden.detach().to("cpu").numpy()
    first_raw_array = raw_logits[0][0].detach().to("cpu").numpy()
    eos_positions = [i for i, token in enumerate(generated) if token == eos]
    require(not eos_positions or eos_positions[0] == steps - 1,
            "Whisper sequence has tokens after first generated EOS")
    eos_seen = bool(eos_positions)
    termination = "eos" if eos_seen else ("length_censored" if steps == max_new_tokens
                                            else "incomplete_unknown")
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"sequence_ids": ids, "prefix_ids": prefix, "generated_ids": generated,
            "actual_generation_steps": steps, "score_rows": score_rows,
            "first_decoder_step": {
                "site": "decoder_hidden_states[0][-1] (final decoder layer, first generation step)",
                "shape": list(first_hidden_array.shape),
                "values": first_hidden_array.tolist(),
                "sha256": sha_bytes(first_hidden_array.tobytes())},
            "first_raw_logits": {"shape": list(first_raw_array.shape),
                                 "values": first_raw_array.tolist(),
                                 "sha256": sha_bytes(first_raw_array.tobytes())},
            "eos_seen": eos_seen, "censored": termination == "length_censored",
            "termination": termination,
            "text_with_special_tokens": raw, "transcript": text,
            "lexical_nonempty": lexical_nonempty(text), "unicode_scripts": scripts(text),
            "language_detection": detection}


def score_ctc(logits: Any, processor: Any) -> dict:
    import torch
    require(torch.is_tensor(logits) and logits.ndim == 3 and logits.shape[0] == 1,
            "CTC logits shape changed")
    require(bool(torch.isfinite(logits).all()), "CTC logits nonfinite")
    ids = [int(x) for x in logits.argmax(-1)[0].tolist()]
    blank = int(processor.tokenizer.pad_token_id)
    collapsed = []
    prior = None
    for token in ids:
        if token != prior and token != blank:
            collapsed.append(token)
        prior = token
    text = processor.batch_decode(torch.tensor([ids]))[0]
    arr = _array(logits)
    return {"frame_ids": ids, "collapsed_nonblank_ids": collapsed,
            "blank_id": blank, "logits_sha256": sha_bytes(arr.tobytes()),
            "transcript": text, "lexical_nonempty": lexical_nonempty(text),
            "unicode_scripts": scripts(text), "termination": "complete_ctc_frames"}


class Capture:
    def __init__(self, model: Any, kind: str):
        self.model, self.kind = model, kind
        self.phase = "generate" if kind.startswith("whisper") else "ctc"
        self.arrays: dict[str, dict[str, Any]] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.detection: dict | None = None
        self.handles = []
        self.original_detect = None

    def _record(self, stage: str, tensor: Any) -> None:
        phase = self.phase
        self.arrays.setdefault(phase, {})[stage] = _array(tensor)
        counts = self.counts.setdefault(phase, {})
        counts[stage] = counts.get(stage, 0) + 1
        require(counts[stage] == 1, f"capture hook called repeatedly: {phase}/{stage}")

    def add_post(self, stage: str, module: Any, *, first_tuple: bool = False) -> None:
        def hook(_module: Any, _args: Any, result: Any) -> None:
            value = result[0] if first_tuple else result
            self._record(stage, value)
        self.handles.append(module.register_forward_hook(hook))

    def add_pre(self, stage: str, module: Any) -> None:
        def hook(_module: Any, args: Any) -> None:
            self._record(stage, args[0])
        self.handles.append(module.register_forward_pre_hook(hook))

    def install(self) -> None:
        if self.kind.startswith("whisper"):
            enc = self.model.model.encoder
            self.add_pre("post_conv_pos", enc.layers[0])
            self.add_post("post_block0", enc.layers[0])
            self.add_post("final_encoder", enc.layer_norm)
            if self.kind == "whisper_multilingual":
                self.original_detect = self.model.detect_language

                def detect(*args: Any, **kwargs: Any) -> Any:
                    self.phase = "detect"
                    try:
                        lang_ids = self.original_detect(*args, **kwargs)
                        require(self.detection is not None, "detection logits hook absent")
                        self.detection["chosen_language_ids"] = [int(x) for x in lang_ids.tolist()]
                        return lang_ids
                    finally:
                        self.phase = "generate"

                self.model.detect_language = detect

                def language_hook(_module: Any, _args: Any, result: Any) -> None:
                    if self.phase != "detect":
                        return
                    import torch
                    vec = result.logits[0, -1].detach().to("cpu", dtype=torch.float32)
                    config = self.model.generation_config
                    lang_logits = {name: float(vec[token]) for name, token in config.lang_to_id.items()}
                    self.detection = {"raw_logits_sha256": sha_bytes(vec.numpy().tobytes()),
                                      "language_logits": lang_logits,
                                      "candidate_count": len(lang_logits)}
                self.handles.append(self.model.register_forward_hook(language_hook))
        else:
            wav = self.model.wav2vec2
            conv0 = wav.feature_extractor.conv_layers[0]
            self.add_post("first_conv_raw", conv0.conv)
            self.add_post("first_conv_norm", conv0.layer_norm)
            self.add_post("feature_projection", wav.feature_projection, first_tuple=True)
            self.add_post("post_block0", wav.encoder.layers[0])
            self.add_post("final_encoder", wav.encoder.layers[-1])
            self.add_post("logits", self.model.lm_head)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.original_detect is not None:
            self.model.detect_language = self.original_detect


def _snapshot_load(model_name: str, assets: dict) -> tuple[Any, Any]:
    """Only called after parent independent-GO gate and local asset checks."""
    import torch
    snapshot = assets["snapshot"]
    if model_name.startswith("whisper"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=True).to("cpu", dtype=torch.float32).eval()
    else:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(snapshot, local_files_only=True)
        model, loading = Wav2Vec2ForCTC.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=True, output_loading_info=True)
        require(set(loading["missing_keys"]) == {"wav2vec2.masked_spec_embed"}
                and not loading["unexpected_keys"] and not loading["mismatched_keys"]
                and not loading["error_msgs"], "Wav2Vec2 checkpoint load report changed")
        model = model.to("cpu", dtype=torch.float32).eval()
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
            "model not wholly CPU FP32")
    return processor, model


def english_generation_kwargs(features: Any, mask: Any) -> dict:
    """English-only Whisper must not receive task or language keywords."""
    return {"input_features": features, "attention_mask": mask,
            "return_timestamps": False, "force_unique_generate_call": True,
            "condition_on_prev_tokens": False, "do_sample": False, "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS, "output_scores": True,
            "output_logits": True, "output_hidden_states": True,
            "return_dict_in_generate": True}


def infer_one(model_name: str, processor: Any, model: Any, pcm: Any) -> tuple[dict, dict]:
    import torch
    import numpy as np
    audio = pcm.astype(np.float32) / np.float32(32768.0)
    capture = Capture(model, model_name)
    try:
        capture.install()
        if model_name == "whisper_english":
            processed = processor(audio, sampling_rate=16000, return_tensors="pt",
                                  return_attention_mask=True)
            features = processed.input_features
            require(features.dtype == torch.float32 and tuple(features.shape) == (1, 80, 3000),
                    "Whisper mel geometry changed")
            geometry = whisper_geometry(processed.attention_mask, len(pcm), features.shape[2])
            capture._record("mel", features)
            require(model.generation_config.is_multilingual is False,
                    "English-only generation config changed")
            with torch.inference_mode():
                result = model.generate(**english_generation_kwargs(
                    features, processed.attention_mask))
            output = score_whisper(result, processor.tokenizer, capture.detection)
            require(capture.counts.get("generate") == {
                "mel": 1, "post_conv_pos": 1, "post_block0": 1, "final_encoder": 1},
                "Whisper transcription capture topology changed")
            require("detect" not in capture.counts and capture.detection is None,
                    "English-only Whisper unexpectedly detected language")
        else:
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
            values = inputs.input_values
            require(values.dtype == torch.float32 and tuple(values.shape) == (1, len(pcm)),
                    "Wav2Vec2 processor geometry changed")
            capture._record("input_values", values)
            with torch.inference_mode():
                result = model(input_values=values, attention_mask=None, return_dict=True)
            output = score_ctc(result.logits, processor)
            geometry = wav_geometry(len(pcm), len(output["frame_ids"]))
            require(capture.counts.get("ctc") == {name: 1 for name in WAV_STAGES},
                    "Wav2Vec2 capture topology changed")
            require(sha_bytes(_array(result.logits).tobytes())
                    == sha_bytes(capture.arrays["ctc"]["logits"].tobytes()),
                    "Wav2Vec2 captured logits differ")
        metrics = {phase: {name: trace_metric(name, value) for name, value in stages.items()}
                   for phase, stages in capture.arrays.items()}
        return {"output": output, "input_geometry": geometry,
                "capture_counts": capture.counts,
                "traces": metrics}, capture.arrays
    finally:
        capture.close()


def _rss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _disk_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_model(model_name: str, output_dir: Path, receipt_path: Path,
              review_path: Path, *, load=_snapshot_load, infer=infer_one) -> None:
    """Worker entry point; independently committed review is mandatory."""
    require(model_name in MODEL_NAMES and output_dir.is_absolute()
            and output_dir.is_dir()
            and {p.name for p in output_dir.iterdir()} == {
                "parent_start.json", "host_before.json", "host_before_power_log.txt.gz",
                "worker_stdout.txt", "worker_stderr.txt"},
            "exclusive parent-created attempt with exact starting evidence required")
    import torch
    torch.set_num_threads(4)
    require_independent_go(model_name, receipt_path, review_path)
    cases = trusted_cases()
    assets = verify_local_assets(model_name)
    installed = verify_installed_sources()
    records = output_dir / "records.jsonl"
    started = time.monotonic()
    _append(records, {"type": "start", "schema": "nonspeech_descriptive_remaining_v2",
                      "model": model_name, "cases": [x["case_id"] for x in cases],
                      "assets": assets, "installed": installed,
                      "source_identity": source_identity(),
                      "manifest_sha256": MANIFEST_SHA256})
    done = set()
    seen_runs: set[str] = set()
    plan = cases + [x for x in cases if x["case_id"].endswith("_silence")]
    pending: dict[str, tuple[str, dict]] = {}
    silence: dict[str, dict] = {}

    def budget() -> None:
        require(time.monotonic() - started <= MAX_SECONDS, "worker time cap exceeded")
        require(_rss() <= MAX_RSS_BYTES, "worker RSS cap exceeded")
        require(_disk_bytes(output_dir) <= MAX_EVIDENCE_BYTES, "worker evidence cap exceeded")

    try:
        budget()
        processor, model = load(model_name, assets)
        for case in plan:
            name = case["case_id"]
            repeat = name in done
            run_id = name + ("__repeat" if repeat else "")
            budget()
            case_started = time.monotonic()
            try:
                evidence, arrays = infer(model_name, processor, model, case["pcm"])
                budget()
                _append(records, {"type": "case", "run_id": run_id, "case_id": name,
                                  "repeat": repeat, "status": "complete", "frames": case["frames"],
                                  "elapsed_seconds": time.monotonic() - case_started,
                                  "wav_sha256": case["wav_sha256"],
                                  "pcm_sha256": case["pcm_sha256"], **evidence})
                seen_runs.add(run_id)
                if not repeat:
                    done.add(name)
                    duration = name[:3]
                    if name.endswith("_silence"):
                        silence[duration] = arrays
                        for prior, (prior_duration, prior_arrays) in list(pending.items()):
                            if prior_duration == duration:
                                _write_diff(records, prior, prior_arrays, arrays)
                                del pending[prior]
                    elif duration in silence:
                        _write_diff(records, name, arrays, silence[duration])
                    else:
                        pending[name] = (duration, arrays)
            except Exception as error:
                _append(records, {"type": "case", "run_id": run_id, "case_id": name,
                                  "repeat": repeat, "status": "failed", "error": repr(error),
                                  "elapsed_seconds": time.monotonic() - case_started,
                                  "wav_sha256": case["wav_sha256"],
                                  "pcm_sha256": case["pcm_sha256"]})
                seen_runs.add(run_id)
                if not repeat:
                    done.add(name)
                raise
        require(not pending and len(done) == 20, "baseline coverage incomplete")
        _append(records, {"type": "complete", "status": "complete", "planned_cases": 20,
                          "planned_repeats": 2, "elapsed_seconds": time.monotonic() - started,
                          "peak_rss_bytes": _rss(), "evidence_bytes": _disk_bytes(output_dir)})
    except BaseException as error:
        for index, case in enumerate(plan):
            run_id = case["case_id"] + ("__repeat" if index >= len(cases) else "")
            if run_id not in seen_runs:
                _append(records, {"type": "case", "run_id": run_id,
                                  "case_id": case["case_id"],
                                  "repeat": index >= len(cases),
                                  "status": "not_run_after_failure",
                                  "wav_sha256": case["wav_sha256"],
                                  "pcm_sha256": case["pcm_sha256"]})
        _append(records, {"type": "complete", "status": "invalid_or_partial",
                          "error": repr(error), "planned_cases": 20,
                          "planned_repeats": 2, "elapsed_seconds": time.monotonic() - started,
                          "peak_rss_bytes": _rss(), "evidence_bytes": _disk_bytes(output_dir)})
        raise


def _write_diff(records: Path, name: str, arrays: dict, donor: dict) -> None:
    require(set(arrays) == set(donor), "silence comparison phase coverage changed")
    differences = {}
    for phase, stages in arrays.items():
        require(set(stages) == set(donor[phase]), "silence comparison stages changed")
        differences[phase] = {stage: trace_difference(stage, value, donor[phase][stage])
                              for stage, value in stages.items()}
    _append(records, {"type": "silence_difference", "case_id": name,
                      "donor_case_id": name[:3] + "_silence", "differences": differences})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    run_model(args.model, args.output_dir, args.preflight, args.review)


if __name__ == "__main__":
    main()
