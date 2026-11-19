"""Frozen plan, source admission, and one-checkpoint Whisper natural-tail worker.

Importing and admitting this module never loads candidate weights. Inference is
available only through a separately committed GO and the bounded parent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import stat
import subprocess
import threading
import time
import wave
from pathlib import Path

from . import descriptive_worker_v1 as old
from . import natural_tail_waveform_v1 as waveforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RENDER = HERE / "runs/natural_tail_waveform_v1/source_v1"
RENDER_SHA256 = "2ae99cfa279c3fd95e4552fd8f59ac0294afb445ddb92c2081b201661b9e0d11"
MODELS = ("whisper_multilingual", "whisper_english")
SOURCE_NAMES = ("natural_tail_runner_v4.py", "natural_tail_parent_v4.py",
                "natural_tail_saved_audit_v4.py", "test_natural_tail_runner_v4.py",
                "WHISPER_NATURAL_TAIL_FOLLOWUP_PROSPECTIVE_V4.md",
                "natural_tail_runner_v3.py", "natural_tail_parent_v3.py",
                "natural_tail_saved_audit_v3.py", "test_natural_tail_runner_v3.py",
                "WHISPER_NATURAL_TAIL_FOLLOWUP_PROSPECTIVE_V3.md",
                "WHISPER_NATURAL_TAIL_INDEPENDENT_EXECUTABLE_REVIEW_V2.md",
                "WHISPER_NATURAL_TAIL_FOLLOWUP_PROSPECTIVE_V2.md",
                "WHISPER_NATURAL_TAIL_FOLLOWUP_PROSPECTIVE_V1.md",
                "WHISPER_NATURAL_TAIL_SOURCE_V1_INDEPENDENT_BYTE_AUDIT.md",
                "natural_tail_waveform_v1.py", "descriptive_pins_v1.json",
                "descriptive_worker_v1.py", "causal_parent_v2.py",
                "wav2vec2_fit_noise_norm_causal_host_v1.py")
WALL_SECONDS = 1800
RSS_BYTES = 4 * 1024**3
EVIDENCE_BYTES = 1024**3
EVIDENCE_RESERVE = 32 * 1024**2
MAX_NEW_TOKENS = 128
ATTEMPTS = {name: HERE/"runs/natural_tail_waveform_v1"/f"{name}_attempt_v4"
            for name in MODELS}
HEARTBEAT_MAX_GAP = 2.0


class ParentGuard:
    """Fail-stop inherited pipe guardian; no disk token can authorize a worker."""

    def __init__(self, fd: int, parent_pid: int, deadline: float):
        self.fd, self.parent_pid, self.deadline = fd, parent_pid, deadline
        self.last_beat = time.monotonic()
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def check(self) -> None:
        require(not self.failed.is_set() and os.getppid() == self.parent_pid
                and time.monotonic() < self.deadline
                and time.monotonic()-self.last_beat < HEARTBEAT_MAX_GAP,
                "live bounded parent heartbeat or deadline lost")

    def _watch(self) -> None:
        try:
            pending = b""
            while True:
                self.check()
                ready, _, _ = select.select([self.fd], [], [], 0.25)
                if ready:
                    data = os.read(self.fd, 256)
                    if not data:
                        raise ValueError("parent heartbeat pipe closed or malformed")
                    pending += data
                    if len(pending) > 256:
                        raise ValueError("parent heartbeat overflow")
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if line != b"H":
                            raise ValueError("parent heartbeat malformed")
                        self.last_beat = time.monotonic()
        except BaseException:
            self.failed.set()
            os._exit(72)

    def start(self) -> None:
        self.thread.start()


def start_parent_guard(attempt: Path, fd: int) -> ParentGuard:
    """Authenticate the live inherited parent before package/model admission."""
    require(fd >= 3 and stat.S_ISFIFO(os.fstat(fd).st_mode),
            "inherited parent pipe required before model import")
    start_path = attempt/"launch_start.json"
    require(start_path.is_file() and not start_path.is_symlink(),
            "parent launch receipt required")
    start = json.loads(start_path.read_text())
    parent_pid = start.get("parent_pid")
    deadline = start.get("deadline_monotonic")
    nonce_hash = start.get("nonce_sha256")
    require(type(parent_pid) is int and parent_pid == os.getppid()
            and type(deadline) in (int, float) and time.monotonic() < deadline
            and isinstance(nonce_hash, str) and len(nonce_hash) == 64,
            "live parent identity or inherited deadline invalid")
    hello = b""
    until = min(float(deadline), time.monotonic()+3.0)
    while len(hello) < 39:
        remaining = until-time.monotonic()
        require(remaining > 0, "parent one-use nonce timed out")
        ready, _, _ = select.select([fd], [], [], remaining)
        require(bool(ready), "parent one-use nonce not received")
        chunk = os.read(fd, 39-len(hello))
        require(bool(chunk), "parent one-use nonce pipe closed")
        hello += chunk
    require(hello.startswith(b"HELLO ") and hello.endswith(b"\n")
            and hello.count(b"\n") == 1 and len(hello) == 39
            and hashlib.sha256(hello[6:-1]).hexdigest() == nonce_hash,
            "parent one-use nonce differs")
    guard = ParentGuard(fd, parent_pid, float(deadline))
    guard.check()
    guard.start()
    return guard


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: object) -> str:
    return old.canonical(value)


def source_identity() -> dict[str, str]:
    result = {}
    for name in SOURCE_NAMES:
        path = HERE.parent / name if name.startswith("wav2vec2_") else HERE / name
        result[name] = old.sha_file(path)
    return result


def _manifest() -> dict:
    require(RENDER.is_dir() and not RENDER.is_symlink(), "render directory missing")
    manifest = RENDER / "manifest.json"
    require(manifest.is_file() and not manifest.is_symlink()
            and old.sha_file(manifest) == RENDER_SHA256, "render manifest changed")
    data = json.loads(manifest.read_text())
    require(data.get("schema") == "whisper_natural_tail_waveform_v1"
            and data.get("status") == "complete_source_only_no_model_inference"
            and len(data.get("rows", [])) == 140, "render status or denominator changed")
    return data


def schedule(model: str) -> list[dict]:
    require(model in MODELS, "unfrozen checkpoint")
    rows = _manifest()["rows"]
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    require(len(by_case) == 20, "case denominator changed")
    plan = []
    for case, cases in by_case.items():
        require([r["condition"] for r in cases] == list(waveforms.CONDITIONS),
                "condition order changed")
        for row in cases:
            plan.append({"index": len(plan)+1, "model": model, "case_id": case,
                         "condition": row["condition"], "replay": False,
                         "wav": row["wav"], "wav_sha256": row["wav_sha256"],
                         "pcm_sha256": row["pcm_sha256"], "frames": row["frames"]})
    for case, cases in by_case.items():
        row = cases[0]
        plan.append({"index": len(plan)+1, "model": model, "case_id": case,
                     "condition": "base", "replay": True, "wav": row["wav"],
                     "wav_sha256": row["wav_sha256"], "pcm_sha256": row["pcm_sha256"],
                     "frames": row["frames"]})
    require(len(plan) == 160 and len({(p["case_id"], p["condition"])
                                       for p in plan if not p["replay"]}) == 140,
            "call schedule changed")
    return plan


def verify_wavs(plan: list[dict]) -> None:
    seen = set()
    for row in plan:
        name = row["wav"]
        if name in seen:
            continue
        seen.add(name)
        path = RENDER / name
        require(path.name == name and not path.is_symlink() and path.is_file()
                and old.sha_file(path) == row["wav_sha256"], f"WAV changed: {name}")
        with wave.open(str(path), "rb") as audio:
            require((audio.getnchannels(), audio.getsampwidth(),
                     audio.getframerate(), audio.getnframes())
                    == (1, 2, 16000, row["frames"]), f"WAV geometry changed: {name}")
            raw = audio.readframes(audio.getnframes())
        require(old.sha_bytes(raw) == row["pcm_sha256"], f"PCM changed: {name}")
    require(len(seen) == 140, "WAV census changed")


def generation_kwargs(model: str, features: object, mask: object) -> dict:
    require(model in MODELS, "unfrozen checkpoint")
    kwargs = {"input_features": features, "attention_mask": mask,
              "return_timestamps": False, "force_unique_generate_call": True,
              "condition_on_prev_tokens": False, "do_sample": False,
              "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS,
              "output_scores": True, "output_logits": True,
              "output_hidden_states": True, "return_dict_in_generate": True}
    if model == "whisper_multilingual":
        kwargs.update(task="transcribe", language=None)
    return kwargs


def admission(model: str) -> dict:
    plan = schedule(model)
    verify_wavs(plan)
    installed = old.verify_installed_sources()
    assets = old.verify_local_assets(model)  # hashes files; never deserializes weights
    return {"schema": "whisper_natural_tail_admission_v4", "status": "source_only",
            "model": model, "attempt_path": str(ATTEMPTS[model]),
            "render_manifest_sha256": RENDER_SHA256,
            "source_identity": source_identity(), "installed": installed,
            "assets": assets, "schedule": plan,
            "limits": {"wall_seconds": WALL_SECONDS, "tree_rss_bytes": RSS_BYTES,
                       "whole_attempt_evidence_bytes": EVIDENCE_BYTES}}


def write_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_commit(path: Path) -> str:
    return subprocess.run(("git", "log", "-1", "--format=%H", "--", str(path)),
                          cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _tracked_clean(path: Path) -> None:
    require(path.is_file() and not path.is_symlink(),
            f"missing committed artifact: {path.name}")
    relative = str(path.relative_to(ROOT))
    require(subprocess.run(("git", "ls-files", "--error-unmatch", "--", relative),
                           cwd=ROOT, capture_output=True).returncode == 0
            and all(subprocess.run(("git", "diff", *flag, "--quiet", "--", relative),
                                   cwd=ROOT, capture_output=True).returncode == 0
                    for flag in ((), ("--cached",))),
            f"artifact not tracked and clean: {relative}")


def validate_go(admission_path: Path, review_path: Path, model: str) -> dict:
    require(admission_path.is_file() and review_path.is_file()
            and not admission_path.is_symlink() and not review_path.is_symlink(),
            "admission or executable GO missing")
    for frozen in (admission_path, review_path, RENDER/"manifest.json"):
        _tracked_clean(frozen)
    for name in SOURCE_NAMES:
        _tracked_clean(HERE.parent/name if name.startswith("wav2vec2_") else HERE/name)
    expected = admission(model)
    require(json.loads(admission_path.read_text()) == expected, "admission changed")
    go = json.loads(review_path.read_text())
    require(go.get("schema") == "whisper_natural_tail_executable_review_v4"
            and go.get("status") == "GO" and go.get("model") == model
            and go.get("attempt_path") == expected["attempt_path"]
            and go.get("admission_sha256") == old.sha_file(admission_path)
            and go.get("source_identity") == expected["source_identity"],
            "independent executable GO invalid")
    review_commit = _file_commit(review_path)
    require(review_commit, "GO must be committed")
    for frozen in (admission_path, RENDER/"manifest.json"):
        commit = _file_commit(frozen)
        require(commit and commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, review_commit),
            cwd=ROOT, check=False).returncode == 0,
            f"GO must follow committed {frozen.name}")
    for name in SOURCE_NAMES:
        path = HERE.parent / name if name.startswith("wav2vec2_") else HERE / name
        commit = _file_commit(path)
        require(commit and commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, review_commit),
            cwd=ROOT, check=False).returncode == 0,
            f"GO must follow frozen source: {name}")
    return expected


def _npy_once(path: Path, array: object) -> dict:
    import numpy as np
    value = np.ascontiguousarray(array)
    require(value.dtype in (np.dtype("float32"), np.dtype("int64")), "sidecar dtype changed")
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": path.name, "sha256": old.sha_file(path),
            "dtype": str(value.dtype), "shape": list(value.shape),
            "values_sha256": old.sha_bytes(value.tobytes())}


def infer_one(model_name: str, processor: object, model: object, pcm: object,
              sidecars: Path, stem: str) -> dict:
    import numpy as np
    import torch
    audio = np.asarray(pcm, dtype=np.float32)/np.float32(32768)
    processed = processor(audio, sampling_rate=16000, return_tensors="pt",
                          return_attention_mask=True)
    features, mask = processed.input_features, processed.attention_mask
    require(features.dtype == torch.float32 and tuple(features.shape) == (1, 80, 3000),
            "Whisper mel geometry changed")
    geometry = old.whisper_geometry(mask, len(pcm), features.shape[2])
    capture = old.Capture(model, model_name)
    try:
        capture.install()
        capture._record("mel", features)
        multilingual = model_name == "whisper_multilingual"
        require(model.generation_config.is_multilingual is multilingual,
                "checkpoint language configuration changed")
        if multilingual:
            require(getattr(model.generation_config, "language", None) is None,
                    "multilingual auto-detection disabled")
        with torch.inference_mode():
            result = model.generate(**generation_kwargs(model_name, features, mask))
        outcome = old.score_whisper(result, processor.tokenizer, capture.detection)
        require(capture.counts.get("generate") == {name: 1 for name in
                ("mel", "post_conv_pos", "post_block0", "final_encoder")},
                "transcription encoder-pass count changed")
        if multilingual:
            require(capture.counts.get("detect") == {name: 1 for name in
                    ("post_conv_pos", "post_block0", "final_encoder")},
                    "language-detection pass count changed")
            chosen = capture.detection["chosen_language_ids"]
            require(len(chosen) == 1 and outcome["prefix_ids"] == [
                model.generation_config.decoder_start_token_id, chosen[0],
                model.generation_config.task_to_id["transcribe"],
                model.generation_config.no_timestamps_token_id],
                "multilingual resolved prefix changed")
        else:
            require("detect" not in capture.counts and capture.detection is None,
                    "English-only checkpoint detected language")
        require(outcome["termination"] in ("eos", "length_censored"),
                "Whisper completion state ambiguous")
        mel = features[0].detach().cpu().numpy().copy()
        mask_values = np.asarray(mask[0].tolist(), dtype=np.int64)
        raw = np.asarray(outcome.pop("first_raw_logits")["values"], dtype=np.float32)
        processed_vec = result.scores[0][0].detach().cpu().to(torch.float32).numpy().copy()
        hidden = np.asarray(outcome.pop("first_decoder_step")["values"], dtype=np.float32)
        references = {"mel": _npy_once(sidecars/f"{stem}_mel.npy", mel),
                      "mask": _npy_once(sidecars/f"{stem}_mask.npy", mask_values),
                      "first_raw_logits": _npy_once(sidecars/f"{stem}_raw.npy", raw),
                      "first_processed_scores": _npy_once(sidecars/f"{stem}_processed.npy", processed_vec),
                      "first_decoder_state": _npy_once(sidecars/f"{stem}_decoder.npy", hidden)}
        max_normalized = float(mel.max())
        return {"output": outcome, "geometry": geometry, "sidecars": references,
                "capture_counts": capture.counts,
                "processor_max_normalized": max_normalized,
                "derived_preclamp_log_spec_max": max_normalized*4.0-4.0,
                "derived_preclamp_floor": max_normalized*4.0-12.0,
                "generation_kwargs": {k: ("tensor" if k in ("input_features", "attention_mask") else v)
                                      for k, v in generation_kwargs(model_name, features, mask).items()}}
    finally:
        capture.close()


def run(model_name: str, attempt: Path, admission_path: Path, review_path: Path,
        heartbeat_fd: int) -> None:
    require(model_name in MODELS and attempt == ATTEMPTS[model_name]
            and attempt.is_absolute() and attempt.is_dir() and not attempt.is_symlink(),
            "fixed parent-created V4 attempt required")
    guard = start_parent_guard(attempt, heartbeat_fd)
    guard.check()
    receipt = validate_go(admission_path, review_path, model_name)
    guard.check()
    parent_start = attempt/"launch_start.json"
    pre_host = attempt/"pre_host.json"
    require(parent_start.is_file() and pre_host.is_file()
            and not parent_start.is_symlink() and not pre_host.is_symlink(),
            "bounded parent start/host evidence required")
    start = json.loads(parent_start.read_text())
    require(start.get("schema") == "whisper_natural_tail_parent_v4"
            and start.get("model") == model_name
            and start.get("attempt") == str(attempt)
            and start.get("parent_pid") == guard.parent_pid
            and start.get("deadline_monotonic") == guard.deadline
            and start.get("admission_sha256") == old.sha_file(admission_path)
            and start.get("go_sha256") == old.sha_file(review_path),
            "bounded parent start mismatch")
    from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
    pre = json.loads(pre_host.read_text())
    host_gate.validate_power(pre)
    host_gate.load_archive(attempt, pre, "pre")
    guard.check()
    import numpy as np
    import torch
    plan = receipt["schedule"]
    torch.set_num_threads(4)
    assets = old.verify_local_assets(model_name)
    guard.check()
    processor, model = old._snapshot_load(model_name, assets)
    guard.check()
    sidecars = attempt/"sidecars"
    sidecars.mkdir()
    progress = attempt/"progress.jsonl"
    records = attempt/"records.jsonl"
    for row in plan:
        guard.check()
        with wave.open(str(RENDER/row["wav"]), "rb") as audio:
            raw = audio.readframes(audio.getnframes())
        require(old.sha_bytes(raw) == row["pcm_sha256"], "WAV PCM changed at call")
        stem = f'{row["index"]:03d}'
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(canonical({"type":"attempt", "index":row["index"]})+"\n")
            stream.flush(); os.fsync(stream.fileno())
        result = infer_one(model_name, processor, model,
                           np.frombuffer(raw, dtype="<i2"), sidecars, stem)
        guard.check()
        with records.open("a", encoding="utf-8") as stream:
            stream.write(canonical({"type":"call", "plan":row, **result})+"\n")
            stream.flush(); os.fsync(stream.fileno())
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(canonical({"type":"complete", "index":row["index"]})+"\n")
            stream.flush(); os.fsync(stream.fileno())
    with records.open("a", encoding="utf-8") as stream:
        stream.write(canonical({"type":"complete", "calls":len(plan)})+"\n")
        stream.flush(); os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--heartbeat-fd", type=int)
    args = parser.parse_args()
    if args.command == "preflight":
        require(args.review is None and args.attempt is None and args.heartbeat_fd is None
                and not args.admission.exists(),
                "preflight is source-only and requires new receipt")
        write_once(args.admission, admission(args.model))
    else:
        require(args.review is not None and args.attempt is not None,
                "worker review and attempt required")
        require(args.heartbeat_fd is not None, "inherited parent pipe required")
        run(args.model, args.attempt, args.admission, args.review, args.heartbeat_fd)


if __name__ == "__main__":
    main()
