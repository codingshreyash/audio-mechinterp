"""One-shot V2 parent for repaired English Whisper and Wav2Vec2 workers."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import descriptive_parent_v1 as host_parent
from . import descriptive_parent_v2 as timing
from . import descriptive_worker_v2 as worker

MODEL_PATHS = worker.MODEL_PATHS


def preflight_only(model: str) -> dict:
    paths = MODEL_PATHS[model]
    receipt_path = paths["receipt"]
    worker.require(receipt_path.parent.is_dir() and not receipt_path.exists(),
                   "exclusive V2 model-specific preflight receipt required")
    receipt = worker.preflight_receipt(model)
    host_parent.write_once(receipt_path, receipt)
    return receipt


def require_model_go(model: str) -> dict:
    paths = MODEL_PATHS[model]
    return worker.require_independent_go(model, paths["receipt"], paths["review"])


def run_once(model: str) -> dict:
    worker.require(model in MODEL_PATHS, "unknown V2 remaining model")
    paths = MODEL_PATHS[model]
    require_model_go(model)
    attempt = paths["attempt"]
    worker.require(attempt.parent.is_dir() and not attempt.exists(),
                   "exclusive V2 model attempt unavailable")
    before = host_parent.host_probe()
    attempt.mkdir()
    started = datetime.now(timezone.utc)
    host_parent.write_once(attempt / "parent_start.json", {
        "schema": "nonspeech_descriptive_parent_start_v2", "model": model,
        "preflight_sha256": worker.sha_file(paths["receipt"]),
        "review_sha256": worker.sha_file(paths["review"]),
        "source_identity": worker.source_identity(),
        "host_before_sha256": worker.sha_bytes(before["outputs"]["power_log"].encode()),
        "utc": started.isoformat()})
    result = {"exit_code": None, "limit_reason": "launch_error",
              "elapsed_seconds": 0.0, "peak_monitored_rss_bytes": 0,
              "evidence_bytes": host_parent.attempt_bytes(attempt)}
    worker_exit_observed = None
    try:
        host_parent.save_host(attempt, "before", before)
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.descriptive_worker_v2",
                   "--model", model, "--output-dir", str(attempt),
                   "--preflight", str(paths["receipt"]),
                   "--review", str(paths["review"])]
        result = host_parent.run_bounded(command, attempt)
        worker_exit_observed = datetime.now(timezone.utc)
        result["worker_exit_observed_utc"] = worker_exit_observed.isoformat()
    except BaseException as error:
        result["launch_error"] = repr(error)
    try:
        after = host_parent.host_probe()
        log = after["outputs"]["power_log"]
        collected = timing.utc(after["collected_utc"])
        host_parent.save_host(attempt, "after", after)
        worker.require(worker_exit_observed is not None,
                       "worker exit observation unavailable")
        result.update(timing.validate_post_host(log, started, worker_exit_observed, collected))
        result["host_after_ok"] = True
    except Exception as error:
        result["host_after_ok"] = False
        result["host_after_error"] = repr(error)
    result.update({"schema": "nonspeech_descriptive_parent_result_v2",
                   "model": model,
                   "parent_source_sha256": worker.sha_file(Path(__file__)),
                   "preflight_sha256": worker.sha_file(paths["receipt"]),
                   "review_sha256": worker.sha_file(paths["review"]),
                   "prior_failed_sha256": paths["prior_sha256"],
                   "status": "worker_exited_pending_audit" if result["exit_code"] == 0
                   and result["limit_reason"] is None and result["host_after_ok"]
                   else "invalid_or_partial"})
    if host_parent.attempt_bytes(attempt) > worker.MAX_EVIDENCE_BYTES:
        result["status"] = "invalid_or_partial"
        result["limit_reason"] = result["limit_reason"] or "evidence_cap_after_exit"
    host_parent.write_once(attempt / "parent_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=worker.MODEL_NAMES, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(worker.canonical(preflight_only(args.model)))
    else:
        print(worker.canonical(run_once(args.model)))


if __name__ == "__main__":
    main()
