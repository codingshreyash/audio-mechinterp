"""Pinned Wav2Vec2 CTC screen on 16 human-speech WAVs; CPU FP32 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

from models.second_architecture.nonspeech_probe_v1 import descriptive_worker_v1 as pinned

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_v1"
MANIFEST = SOURCE / "manifest.json"
ADMISSION = HERE / "ASR_PROXY_ADMISSION_V1.json"
ATTEMPT = HERE / "runs/asr_proxy_v1/attempt_001"
EXPECTED_MANIFEST_SHA = "c0593a4631191b7643714651c751fde29ebc6323bf1d59680d3a4c1d1bd38051"
SOURCES = ("ASR_SCREEN_PROTOCOL_V1.md", "asr_proxy_worker_v1.py",
           "asr_proxy_parent_v1.py", "asr_proxy_saved_audit_v1.py",
           "test_source_only_v1.py",
           "source_v1/manifest.json", "SOURCE_AUDIT_V1.json",
           "SOURCE_INDEPENDENT_TECHNICAL_REVIEW_V1.md")


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan() -> list[dict]:
    need(sha(MANIFEST) == EXPECTED_MANIFEST_SHA, "source manifest changed")
    data = json.loads(MANIFEST.read_text())
    need(data.get("schema") == "human_audio_source_v1"
         and data.get("status") == "source_only_no_model_inference"
         and len(data["rows"]) == 4, "human source status changed")
    rows = []
    for source in data["rows"]:
        need([c["condition"] for c in source["conditions"]]
             == ["base", "z32000", "n32000_s0", "n32000_s1"],
             "human condition order changed")
        for condition in source["conditions"]:
            path = SOURCE / condition["wav"]
            need(path.name == condition["wav"] and path.is_file()
                 and not path.is_symlink() and sha(path) == condition["wav_sha256"],
                 "human WAV changed")
            rows.append({"index": len(rows) + 1, "utterance_id": source["utterance_id"],
                         "speaker_id": source["speaker_id"], "split": source["split"],
                         "condition": condition["condition"],
                         "wav": condition["wav"], "wav_sha256": condition["wav_sha256"],
                         "pcm_sha256": condition["pcm_sha256"],
                         "frames": condition["frames"],
                         "reference_transcript": source["reference_transcript"]})
    need(len(rows) == 16 and len({r["wav"] for r in rows}) == 16,
         "human ASR denominator changed")
    return rows


def source_identity() -> dict[str, str]:
    result = {}
    for name in SOURCES:
        path = HERE / name
        need(path.is_file() and not path.is_symlink(), f"source file missing: {name}")
        result[name] = sha(path)
    result["descriptive_pins_v1.json"] = sha(pinned.PINS)
    result["descriptive_worker_v1.py"] = sha(Path(pinned.__file__))
    return result


def admission() -> dict:
    source_receipt = json.loads((HERE / "SOURCE_AUDIT_V1.json").read_text())
    need(source_receipt.get("status") == "PASS"
         and source_receipt.get("manifest_sha256") == EXPECTED_MANIFEST_SHA,
         "human source technical audit missing")
    assets = pinned.verify_local_assets("wav2vec2")  # hashes weights; no deserialization
    installed = pinned.verify_installed_sources()  # imports source; no model weights
    return {"schema": "human_asr_proxy_admission_v1", "status": "source_only_no_model_inference",
            "model": "facebook/wav2vec2-base-960h", "model_revision": assets["revision"],
            "source_manifest_sha256": EXPECTED_MANIFEST_SHA,
            "source_identity": source_identity(), "assets": assets,
            "installed": installed, "plan": plan(),
            "limits": {"whole_wall_seconds": 600, "tree_rss_bytes": 4 * 1024**3,
                       "attempt_evidence_bytes": 256 * 1024**2}}


def load_pcm(row: dict) -> Any:
    import numpy as np
    path = SOURCE / row["wav"]
    need(sha(path) == row["wav_sha256"], "input WAV changed")
    with wave.open(str(path), "rb") as wav:
        need((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
              wav.getnframes()) == (1, 2, 16000, row["frames"]),
             "WAV geometry changed")
        raw = wav.readframes(wav.getnframes())
    need(hashlib.sha256(raw).hexdigest() == row["pcm_sha256"], "PCM hash changed")
    return np.frombuffer(raw, dtype="<i2")


def save_npy(path: Path, array: Any) -> dict:
    import numpy as np
    value = np.ascontiguousarray(array, dtype=np.float32)
    need(bool(np.isfinite(value).all()), "nonfinite ASR sidecar")
    with path.open("xb") as out:
        np.save(out, value, allow_pickle=False)
        out.flush()
        os.fsync(out.fileno())
    return {"file": path.name, "sha256": sha(path),
            "values_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "shape": list(value.shape), "dtype": "float32"}


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        out.flush()
        os.fsync(out.fileno())


def run(attempt: Path, admitted: Path) -> None:
    need(attempt == ATTEMPT and admitted == ADMISSION
         and attempt.is_dir() and not attempt.is_symlink(), "fixed parent attempt required")
    start = json.loads((attempt / "launch.json").read_text())
    need(start.get("schema") == "human_asr_proxy_parent_v1"
         and start.get("worker_parent_pid") == os.getppid()
         and start.get("admission_sha256") == sha(admitted),
         "bounded parent launch missing")
    expected = admission()
    need(json.loads(admitted.read_text()) == expected, "admission changed")
    import numpy as np
    import torch
    torch.set_num_threads(4)
    processor, model = pinned._snapshot_load("wav2vec2", expected["assets"])
    sidecars = attempt / "sidecars"
    sidecars.mkdir(exist_ok=False)
    for row in expected["plan"]:
        started = time.monotonic()
        pcm = load_pcm(row)
        audio = pcm.astype(np.float32) / np.float32(32768.0)
        processed = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
        values = processed.input_values
        need(values.dtype == torch.float32 and tuple(values.shape) == (1, len(pcm)),
             "processor values changed")
        with torch.inference_mode():
            logits = model(input_values=values, attention_mask=None,
                           return_dict=True).logits
        need(logits.dtype == torch.float32 and logits.ndim == 3
             and logits.shape[0] == 1 and bool(torch.isfinite(logits).all()),
             "CTC logits changed")
        ids = [int(x) for x in logits.argmax(-1)[0].tolist()]
        decoded = processor.batch_decode(torch.tensor([ids]))[0]
        stem = f"{row['index']:02d}"
        values_ref = save_npy(sidecars / f"{stem}_input_values.npy", values[0].cpu().numpy())
        logits_ref = save_npy(sidecars / f"{stem}_logits.npy", logits[0].cpu().numpy())
        append_jsonl(attempt / "records.jsonl",
                     {"type": "call", "index": row["index"], "plan": row,
                      "output": {"raw_transcript": decoded, "frame_ids": ids,
                                 "ctc_blank_id": int(processor.tokenizer.pad_token_id),
                                 "logit_frames": int(logits.shape[1]),
                                 "vocab_size": int(logits.shape[2]),
                                 "termination": "complete_ctc_frames"},
                      "sidecars": {"input_values": values_ref, "logits": logits_ref},
                      "elapsed_seconds": time.monotonic() - started})
    append_jsonl(attempt / "records.jsonl", {"type": "complete", "calls": 16})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--attempt", type=Path, default=ATTEMPT)
    parser.add_argument("--admission", type=Path, default=ADMISSION)
    args = parser.parse_args()
    if args.command == "preflight":
        result = admission()
        with args.admission.open("x") as out:
            out.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        print(json.dumps({"status": result["status"], "calls": len(result["plan"]),
                          "admission": str(args.admission)}))
    else:
        run(args.attempt, args.admission)


if __name__ == "__main__":
    main()
