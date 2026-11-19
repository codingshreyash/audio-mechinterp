"""Saved-only CTC audit; reconstruct outputs without target or proxy weights."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from models.second_architecture.nonspeech_probe_v1 import whisper_tail_causal_host_v1 as host
from . import asr_proxy_parent_v1 as parent
from . import asr_proxy_worker_v1 as worker

RECEIPT = worker.HERE / "runs/asr_proxy_v1/attempt_001_audit.json"


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()))


def load_ref(sidecars: Path, ref: dict, expected_shape: tuple[int, ...]) -> np.ndarray:
    name = ref.get("file")
    need(isinstance(name, str) and name == Path(name).name
         and name.endswith(".npy"), "sidecar filename invalid")
    path = sidecars / name
    need(path.is_file() and not path.is_symlink()
         and worker.sha(path) == ref["sha256"], "sidecar hash mismatch")
    value = np.load(path, allow_pickle=False)
    need(value.dtype == np.float32 and value.shape == expected_shape
         and ref["shape"] == list(expected_shape)
         and ref["dtype"] == "float32" and np.isfinite(value).all()
         and worker.sha_bytes(value.tobytes()) == ref["values_sha256"],
         "sidecar contents mismatch")
    return value


def audit() -> dict:
    admission = parent.validate_admission()
    attempt = worker.ATTEMPT
    need(attempt.is_dir() and not attempt.is_symlink(), "ASR attempt unavailable")
    result = json.loads((attempt / "parent_result.json").read_text())
    need(result.get("schema") == "human_asr_proxy_parent_result_v1"
         and result.get("status") == "completed_pending_saved_audit"
         and result.get("reason") == "normal_exit"
         and result.get("exit_code") == 0
         and result.get("admission_sha256") == worker.sha(worker.ADMISSION)
         and result.get("peak_tree_rss_bytes", 2**64) <= 4 * 1024**3
         and result.get("whole_elapsed_seconds", 10**9) <= 600,
         "parent result invalid")
    launch = json.loads((attempt / "launch.json").read_text())
    need(launch.get("schema") == "human_asr_proxy_parent_v1"
         and launch.get("admission_sha256") == worker.sha(worker.ADMISSION)
         and isinstance(launch.get("worker_parent_pid"), int), "launch binding invalid")
    before = json.loads((attempt / "host_before.json").read_text())
    after = json.loads((attempt / "host_after.json").read_text())
    for sample in (before, after):
        host.validate_saved_text(sample)
        need(sample["external_connected"] is True and sample["lid_open"] is True
             and sample["thermal_ok"] is True, "AC/lid/thermal gate invalid")
    samples = [json.loads(line) for line in (attempt / "resource_samples.jsonl").read_text().splitlines()]
    need(samples and len(samples) == result["resource_samples"]
         and all(s["tree_rss_bytes"] <= 4 * 1024**3
                 and s["evidence_bytes"] <= 256 * 1024**2 for s in samples)
         and max(s["tree_rss_bytes"] for s in samples) == result["peak_tree_rss_bytes"]
         and all(a["monotonic"] < b["monotonic"] for a, b in zip(samples, samples[1:]))
         and 0 <= samples[0]["monotonic"] - launch["started_monotonic"] <= 5
         and samples[-1]["monotonic"] <= launch["deadline_monotonic"],
         "resource monitoring evidence invalid")
    records = [json.loads(line) for line in (attempt / "records.jsonl").read_text().splitlines()]
    need(len(records) == 17 and records[-1] == {"type": "complete", "calls": 16},
         "ASR record census invalid")
    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(admission["assets"]["snapshot"],
                                                   local_files_only=True)
    outputs = []
    referenced = set()
    for index, (row, planned) in enumerate(zip(records[:-1], admission["plan"]), 1):
        need(row.get("type") == "call" and row.get("index") == index
             and row.get("plan") == planned, "ASR call plan changed")
        pcm = worker.load_pcm(planned)
        processed = processor(pcm.astype(np.float32) / np.float32(32768),
                              sampling_rate=16000, return_tensors="np", padding=False)
        values = np.asarray(processed.input_values, dtype=np.float32)
        sidecars = row["sidecars"]
        need(set(sidecars) == {"input_values", "logits"}, "ASR sidecar fields changed")
        input_values = load_ref(attempt / "sidecars", sidecars["input_values"], (len(pcm),))
        need(np.array_equal(input_values, values[0]), "processor values changed")
        logits_ref = sidecars["logits"]
        need(logits_ref["shape"][1] == len(processor.tokenizer), "CTC vocab changed")
        logits = load_ref(attempt / "sidecars", logits_ref,
                          (logits_ref["shape"][0], logits_ref["shape"][1]))
        ids = [int(x) for x in np.argmax(logits, axis=-1).tolist()]
        output = row["output"]
        text = processor.batch_decode(np.asarray([ids], dtype=np.int64))[0]
        need(output == {"raw_transcript": text, "frame_ids": ids,
                        "ctc_blank_id": int(processor.tokenizer.pad_token_id),
                        "logit_frames": len(ids), "vocab_size": logits.shape[1],
                        "termination": "complete_ctc_frames"},
             "saved CTC output differs from logits/tokenizer")
        need(isinstance(row.get("elapsed_seconds"), (float, int))
             and 0 <= row["elapsed_seconds"] <= 600, "call elapsed invalid")
        referenced.update((sidecars["input_values"]["file"], logits_ref["file"]))
        outputs.append({"index": index, "speaker_id": planned["speaker_id"],
                        "split": planned["split"], "condition": planned["condition"],
                        "reference_transcript": planned["reference_transcript"],
                        "raw_transcript": text,
                        "normalized_transcript": normalized(text),
                        "normalized_reference_match": normalized(text)
                            == normalized(planned["reference_transcript"])})
    need(len(referenced) == 32
         and {p.name for p in (attempt / "sidecars").iterdir()} == referenced,
         "ASR sidecar inventory invalid")
    by = {(r["speaker_id"], r["condition"]): r for r in outputs}
    comparisons = []
    for speaker in [r["speaker_id"] for r in json.loads(worker.MANIFEST.read_text())["rows"]]:
        zero = by[(speaker, "z32000")]
        for seed in (0, 1):
            noise = by[(speaker, f"n32000_s{seed}")]
            comparisons.append({"speaker_id": speaker, "split": zero["split"],
                                "seed": seed,
                                "proxy_zero_noise_equal": zero["normalized_transcript"]
                                    == noise["normalized_transcript"]})
    return {"schema": "human_asr_proxy_saved_audit_v1", "status": "PASS",
            "source_manifest_sha256": worker.EXPECTED_MANIFEST_SHA,
            "admission_sha256": worker.sha(worker.ADMISSION),
            "attempt": str(attempt), "calls": len(outputs),
            "sidecars": len(referenced), "outputs": outputs,
            "zero_noise_comparisons": comparisons,
            "reference_matches": sum(r["normalized_reference_match"] for r in outputs),
            "proxy_stable_pairs": sum(r["proxy_zero_noise_equal"] for r in comparisons),
            "human_listening_records": 0,
            "target_model_inference": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RECEIPT)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x") as out:
        out.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"], "calls": result["calls"],
                      "sidecars": result["sidecars"],
                      "proxy_stable_pairs": result["proxy_stable_pairs"]}))


if __name__ == "__main__":
    main()
