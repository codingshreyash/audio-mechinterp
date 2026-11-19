"""One-shot V2 host timing gate for the frozen non-speech descriptive worker.

Only the host-after timing fence changes. The V1 worker and failed V1 attempt
are immutable. Import and preflight never deserialize model weights.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import descriptive_parent_v1 as old_parent
from . import descriptive_worker_v1 as worker

HERE = Path(__file__).resolve().parent
V1_RECEIPT = HERE / "runs/descriptive_source_preflight_v2/preflight.json"
V1_GO = HERE / "INDEPENDENT_DESCRIPTIVE_SOURCE_GO_V2.json"
V1_ATTEMPT = HERE / "runs/descriptive_whisper_multilingual_20261119_v1"
V2_ATTEMPT = HERE / "runs/descriptive_whisper_multilingual_20261119_v2"
V2_RECEIPT = HERE / "runs/descriptive_host_timing_preflight_v2/preflight.json"
MAX_POST_PROBE_LAG_SECONDS = 120
PRIOR_SHA256 = {
    "parent_result.json": "806bfd0c84899e88cec47b24b0966274561cdd155908c52369d0699496c2404a",
    "records.jsonl": "984625850ac33cd4c5d039304162a9b4396bd2a2ad3d31513a231eb452e708a3",
}
V2_SOURCES = ("descriptive_parent_v2.py", "descriptive_audit_v2.py",
              "test_descriptive_host_timing_v2.py", "DESCRIPTIVE_HOST_TIMING_AMENDMENT_V2.md")


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    worker.require(parsed.tzinfo is not None, "host timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def verify_prior_failure() -> dict:
    """Hash prior bytes without opening or scoring prior model output records."""
    worker.require(V1_ATTEMPT.is_dir() and not V1_ATTEMPT.is_symlink(),
                   "immutable V1 failed attempt absent")
    for name, digest in PRIOR_SHA256.items():
        path = V1_ATTEMPT / name
        worker.require(path.is_file() and not path.is_symlink()
                       and worker.sha_file(path) == digest,
                       f"immutable V1 attempt bytes changed: {name}")
    result = json.loads((V1_ATTEMPT / "parent_result.json").read_text())
    worker.require(result.get("schema") == "nonspeech_descriptive_parent_result_v1"
                   and result.get("model") == "whisper_multilingual"
                   and result.get("status") == "invalid_or_partial"
                   and result.get("exit_code") == 0
                   and result.get("host_after_ok") is False
                   and result.get("host_after_error") ==
                   "ValueError('Host power-log coverage is stale or truncated')",
                   "V1 failed-parent disposition changed")
    return {"status": "prior_invalid_preserved", "files_sha256": PRIOR_SHA256}


def source_identity() -> dict[str, str]:
    result = {name: worker.sha_file(HERE / name) for name in V2_SOURCES}
    result["descriptive_parent_v1.py"] = worker.sha_file(HERE / "descriptive_parent_v1.py")
    result["descriptive_worker_v1.py"] = worker.sha_file(HERE / "descriptive_worker_v1.py")
    return result


def preflight_receipt() -> dict:
    worker.require_independent_go(V1_RECEIPT, V1_GO)
    old = worker.preflight_receipt()  # hashes pinned local assets, no model load
    return {"schema": "nonspeech_descriptive_host_timing_preflight_v2",
            "status": "source_only_no_model_load",
            "v1_preflight_sha256": worker.sha_file(V1_RECEIPT),
            "v1_go_sha256": worker.sha_file(V1_GO),
            "v1_source_identity": old["source_identity"],
            "manifest_sha256": worker.MANIFEST_SHA256,
            "prior_failed_attempt": verify_prior_failure(),
            "v2_source_identity": source_identity(),
            "model": "whisper_multilingual",
            "attempt_path": str(V2_ATTEMPT.resolve()),
            "max_post_probe_lag_seconds": MAX_POST_PROBE_LAG_SECONDS,
            "limits": old["limits"]}


def preflight_only() -> dict:
    worker.require(not V2_RECEIPT.exists() and V2_RECEIPT.parent.is_dir(),
                   "exclusive V2 preflight receipt path required")
    receipt = preflight_receipt()
    old_parent.write_once(V2_RECEIPT, receipt)
    return receipt


def require_v2_go(review_path: Path) -> dict:
    worker.require(V2_RECEIPT.is_file() and review_path.is_file()
                   and not V2_RECEIPT.is_symlink() and not review_path.is_symlink(),
                   "V2 preflight or independent review missing")
    receipt = json.loads(V2_RECEIPT.read_text())
    worker.require(receipt == preflight_receipt(), "V2 source preflight differs")
    review = json.loads(review_path.read_text())
    worker.require(review.get("schema") == "nonspeech_descriptive_host_timing_review_v2"
                   and review.get("status") == "GO"
                   and review.get("preflight_sha256") == worker.sha_file(V2_RECEIPT)
                   and review.get("v2_source_identity") == receipt["v2_source_identity"]
                   and review.get("prior_failed_attempt") == receipt["prior_failed_attempt"],
                   "independent V2 host timing GO does not bind current source")
    review_commit = worker._file_commit(review_path)
    for name in V2_SOURCES:
        source_commit = worker._file_commit(HERE / name)
        worker.require(source_commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", source_commit, review_commit),
            cwd=worker.HERE.parents[2], check=False).returncode == 0,
            f"V2 review must follow frozen source: {name}")
    return review


def validate_post_host(log: str, start: datetime, worker_exit: datetime,
                       collected: datetime) -> dict:
    """Fence to worker exit, never a timestamp captured after the probe."""
    worker.require(start.tzinfo is not None and worker_exit.tzinfo is not None
                   and collected.tzinfo is not None and start <= worker_exit <= collected,
                   "post-host timestamp order invalid")
    lag = (collected - worker_exit).total_seconds()
    worker.require(0 <= lag <= MAX_POST_PROBE_LAG_SECONDS,
                   "post-host probe lag exceeded")
    old_parent.host_gate._power_log_coverage(log, start, worker_exit, collected)
    worker.require(not old_parent.host_gate._power_events(log, start, collected),
                   "sleep, wake, or thermal event during worker or probe gap")
    return {"host_timing_schema": "nonspeech_host_after_exit_fence_v2",
            "host_fence_end_utc": worker_exit.isoformat(),
            "host_probe_collected_utc": collected.isoformat(),
            "host_probe_lag_seconds": lag,
            "max_post_probe_lag_seconds": MAX_POST_PROBE_LAG_SECONDS}


def run_once(review_path: Path) -> dict:
    require_v2_go(review_path)
    worker.require(V2_ATTEMPT.parent.is_dir() and not V2_ATTEMPT.exists(),
                   "reserved V2 attempt already exists or parent absent")
    before = old_parent.host_probe()
    V2_ATTEMPT.mkdir()
    start = datetime.now(timezone.utc)
    old_parent.write_once(V2_ATTEMPT / "parent_start.json", {
        "schema": "nonspeech_descriptive_parent_start_v1", "model": "whisper_multilingual",
        "preflight_sha256": worker.sha_file(V1_RECEIPT),
        "review_sha256": worker.sha_file(V1_GO),
        "source_identity": worker.source_identity(),
        "host_before_sha256": worker.sha_bytes(before["outputs"]["power_log"].encode()),
        "v2_preflight_sha256": worker.sha_file(V2_RECEIPT),
        "v2_review_sha256": worker.sha_file(review_path),
        "utc": start.isoformat()})
    result = {"exit_code": None, "limit_reason": "launch_error",
              "elapsed_seconds": 0.0, "peak_monitored_rss_bytes": 0,
              "evidence_bytes": old_parent.attempt_bytes(V2_ATTEMPT)}
    worker_exit_observed = None
    try:
        old_parent.save_host(V2_ATTEMPT, "before", before)
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.descriptive_worker_v1",
                   "--model", "whisper_multilingual", "--output-dir", str(V2_ATTEMPT),
                   "--preflight", str(V1_RECEIPT), "--review", str(V1_GO)]
        result = old_parent.run_bounded(command, V2_ATTEMPT)
        # This observation is made immediately after wait(), before any post-host
        # command. It is the latest possible end of worker activity.
        worker_exit_observed = datetime.now(timezone.utc)
        result["worker_exit_observed_utc"] = worker_exit_observed.isoformat()
    except BaseException as error:
        result["launch_error"] = repr(error)
    try:
        after = old_parent.host_probe()
        log = after["outputs"]["power_log"]
        collected = utc(after["collected_utc"])
        old_parent.save_host(V2_ATTEMPT, "after", after)
        worker.require(worker_exit_observed is not None,
                       "worker exit observation unavailable")
        result.update(validate_post_host(log, start, worker_exit_observed, collected))
        result["host_after_ok"] = True
    except Exception as error:
        result["host_after_ok"] = False
        result["host_after_error"] = repr(error)
    result.update({"schema": "nonspeech_descriptive_parent_result_v1",
                   "model": "whisper_multilingual",
                   "v2_parent_source_sha256": worker.sha_file(Path(__file__)),
                   "v2_preflight_sha256": worker.sha_file(V2_RECEIPT),
                   "v2_review_sha256": worker.sha_file(review_path),
                   "prior_failed_parent_sha256": PRIOR_SHA256["parent_result.json"],
                   "status": "worker_exited_pending_audit" if result["exit_code"] == 0
                   and result["limit_reason"] is None and result["host_after_ok"]
                   else "invalid_or_partial"})
    if old_parent.attempt_bytes(V2_ATTEMPT) > worker.MAX_EVIDENCE_BYTES:
        result["status"] = "invalid_or_partial"
        result["limit_reason"] = result["limit_reason"] or "evidence_cap_after_exit"
    old_parent.write_once(V2_ATTEMPT / "parent_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.preflight_only:
        worker.require(args.review is None, "preflight cannot include review")
        print(worker.canonical(preflight_only()))
    else:
        worker.require(args.review is not None, "V2 independent review required")
        print(worker.canonical(run_once(args.review)))


if __name__ == "__main__":
    main()
