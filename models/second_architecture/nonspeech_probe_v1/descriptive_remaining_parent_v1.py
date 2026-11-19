"""Separate one-shot parents for English Whisper and Wav2Vec2 descriptive arms.

No candidate weights are loaded by import, preflight, or review admission.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import descriptive_parent_v1 as old_parent
from . import descriptive_parent_v2 as timing
from . import descriptive_worker_v1 as worker

HERE = Path(__file__).resolve().parent
V1_RECEIPT = HERE / "runs/descriptive_source_preflight_v2/preflight.json"
V1_GO = HERE / "INDEPENDENT_DESCRIPTIVE_SOURCE_GO_V2.json"
MODEL_PATHS = {
    "whisper_english": {
        "attempt": HERE / "runs/descriptive_whisper_english_20261119_v1",
        "receipt": HERE / "runs/descriptive_remaining_preflight_v1/whisper_english.json",
        "review": HERE / "INDEPENDENT_DESCRIPTIVE_WHISPER_ENGLISH_GO_V1.json",
    },
    "wav2vec2": {
        "attempt": HERE / "runs/descriptive_wav2vec2_20261119_v1",
        "receipt": HERE / "runs/descriptive_remaining_preflight_v1/wav2vec2.json",
        "review": HERE / "INDEPENDENT_DESCRIPTIVE_WAV2VEC2_GO_V1.json",
    },
}
SOURCE_NAMES = (
    "descriptive_remaining_parent_v1.py", "descriptive_remaining_audit_v1.py",
    "test_descriptive_remaining_v1.py", "DESCRIPTIVE_REMAINING_HOST_TIMING_PROSPECTIVE_V1.md",
    "descriptive_parent_v2.py", "descriptive_parent_v1.py", "descriptive_worker_v1.py",
)


def paths(model: str) -> dict[str, Path]:
    worker.require(model in MODEL_PATHS and model in worker.MODEL_NAMES,
                   "remaining descriptive model name changed")
    return MODEL_PATHS[model]


def source_identity() -> dict[str, str]:
    return {name: worker.sha_file(HERE / name) for name in SOURCE_NAMES}


def preflight_receipt(model: str) -> dict:
    model_paths = paths(model)
    worker.require_independent_go(V1_RECEIPT, V1_GO)
    old = worker.preflight_receipt()  # hashes assets and source; no weight load
    return {"schema": "nonspeech_descriptive_remaining_preflight_v1",
            "status": "source_only_no_model_load", "model": model,
            "attempt_path": str(model_paths["attempt"].resolve()),
            "review_path": str(model_paths["review"].resolve()),
            "source_identity": source_identity(),
            "v1_source_identity": old["source_identity"],
            "v1_receipt_sha256": worker.sha_file(V1_RECEIPT),
            "v1_go_sha256": worker.sha_file(V1_GO),
            "model_assets": old["models"][model],
            "manifest_sha256": worker.MANIFEST_SHA256,
            "case_ids": old["case_ids"], "planned_cases": 20,
            "planned_silence_repeats": 2,
            "max_post_probe_lag_seconds": timing.MAX_POST_PROBE_LAG_SECONDS,
            "limits": old["limits"]}


def preflight_only(model: str) -> dict:
    path = paths(model)["receipt"]
    worker.require(path.parent.is_dir() and not path.exists(),
                   "exclusive model-specific preflight receipt required")
    receipt = preflight_receipt(model)
    old_parent.write_once(path, receipt)
    return receipt


def validate_review_fields(model: str, review: dict, receipt: dict,
                           receipt_sha256: str) -> None:
    model_paths = paths(model)
    worker.require(review.get("schema") == "nonspeech_descriptive_remaining_model_review_v1"
                   and review.get("status") == "GO"
                   and review.get("model") == model
                   and review.get("attempt_path") == str(model_paths["attempt"].resolve())
                   and review.get("preflight_sha256") == receipt_sha256
                   and review.get("source_identity") == receipt["source_identity"],
                   "model-specific independent GO does not bind this arm")


def require_model_go(model: str) -> dict:
    model_paths = paths(model)
    receipt_path, review_path = model_paths["receipt"], model_paths["review"]
    worker.require(receipt_path.is_file() and review_path.is_file()
                   and not receipt_path.is_symlink() and not review_path.is_symlink(),
                   "model-specific receipt or independent review missing")
    receipt = json.loads(receipt_path.read_text())
    worker.require(receipt == preflight_receipt(model),
                   "model-specific source preflight differs")
    review = json.loads(review_path.read_text())
    validate_review_fields(model, review, receipt, worker.sha_file(receipt_path))
    review_commit = worker._file_commit(review_path)
    for name in SOURCE_NAMES:
        source_commit = worker._file_commit(HERE / name)
        worker.require(source_commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", source_commit, review_commit),
            cwd=worker.HERE.parents[2], check=False).returncode == 0,
            f"model-specific review must follow frozen source: {name}")
    return review


def run_once(model: str) -> dict:
    model_paths = paths(model)
    require_model_go(model)
    attempt = model_paths["attempt"]
    worker.require(attempt.parent.is_dir() and not attempt.exists(),
                   "exclusive model-specific attempt path unavailable")
    before = old_parent.host_probe()
    attempt.mkdir()
    started = datetime.now(timezone.utc)
    old_parent.write_once(attempt / "parent_start.json", {
        "schema": "nonspeech_descriptive_parent_start_v1", "model": model,
        "preflight_sha256": worker.sha_file(V1_RECEIPT),
        "review_sha256": worker.sha_file(V1_GO),
        "source_identity": worker.source_identity(),
        "host_before_sha256": worker.sha_bytes(before["outputs"]["power_log"].encode()),
        "remaining_preflight_sha256": worker.sha_file(model_paths["receipt"]),
        "remaining_review_sha256": worker.sha_file(model_paths["review"]),
        "utc": started.isoformat()})
    result = {"exit_code": None, "limit_reason": "launch_error",
              "elapsed_seconds": 0.0, "peak_monitored_rss_bytes": 0,
              "evidence_bytes": old_parent.attempt_bytes(attempt)}
    worker_exit_observed = None
    try:
        old_parent.save_host(attempt, "before", before)
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.descriptive_worker_v1",
                   "--model", model, "--output-dir", str(attempt),
                   "--preflight", str(V1_RECEIPT), "--review", str(V1_GO)]
        result = old_parent.run_bounded(command, attempt)
        worker_exit_observed = datetime.now(timezone.utc)
        result["worker_exit_observed_utc"] = worker_exit_observed.isoformat()
    except BaseException as error:
        result["launch_error"] = repr(error)
    try:
        after = old_parent.host_probe()
        log = after["outputs"]["power_log"]
        collected = timing.utc(after["collected_utc"])
        old_parent.save_host(attempt, "after", after)
        worker.require(worker_exit_observed is not None,
                       "worker exit observation unavailable")
        result.update(timing.validate_post_host(log, started, worker_exit_observed, collected))
        result["host_after_ok"] = True
    except Exception as error:
        result["host_after_ok"] = False
        result["host_after_error"] = repr(error)
    result.update({"schema": "nonspeech_descriptive_parent_result_v1",
                   "model": model,
                   "remaining_parent_source_sha256": worker.sha_file(Path(__file__)),
                   "remaining_preflight_sha256": worker.sha_file(model_paths["receipt"]),
                   "remaining_review_sha256": worker.sha_file(model_paths["review"]),
                   "status": "worker_exited_pending_audit" if result["exit_code"] == 0
                   and result["limit_reason"] is None and result["host_after_ok"]
                   else "invalid_or_partial"})
    if old_parent.attempt_bytes(attempt) > worker.MAX_EVIDENCE_BYTES:
        result["status"] = "invalid_or_partial"
        result["limit_reason"] = result["limit_reason"] or "evidence_cap_after_exit"
    old_parent.write_once(attempt / "parent_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(worker.canonical(preflight_only(args.model)))
    else:
        print(worker.canonical(run_once(args.model)))


if __name__ == "__main__":
    main()
