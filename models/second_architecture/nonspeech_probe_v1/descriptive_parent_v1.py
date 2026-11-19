"""Local-only source preflight and one-shot bounded descriptive launch.

No inference is permitted without a later committed independent GO receipt.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import descriptive_worker_v1 as worker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as host_gate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(worker.canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def preflight_only(receipt_path: Path) -> dict:
    worker.require(receipt_path.is_absolute() and receipt_path.parent.is_dir()
                   and not receipt_path.exists(), "new absolute preflight receipt required")
    receipt = worker.preflight_receipt()
    write_once(receipt_path, receipt)
    return receipt


def host_probe() -> dict:
    """Use the already audited Mac power, lid, and recent-transition gate."""
    worker.require(worker.sha_file(Path(host_gate.__file__))
                   == "1fbc29e7851491d53c1c8245a7acd385563a1491435d21790a4370a7aa3492be",
                   "host gate source changed")
    return host_gate._host_probe(require_stable=True)


def save_host(attempt: Path, label: str, evidence: dict) -> None:
    raw = evidence["outputs"].pop("power_log").encode()
    packed = gzip.compress(raw, mtime=0)
    worker.require(len(packed) <= 8 * 1024**2 and len(raw) <= 128 * 1024**2,
                   "host power log exceeds evidence cap")
    archive = attempt / f"host_{label}_power_log.txt.gz"
    with archive.open("xb") as stream:
        stream.write(packed)
    evidence["power_log_archive"] = {"file": archive.name,
                                      "raw_sha256": worker.sha_bytes(raw),
                                      "compressed_sha256": worker.sha_file(archive)}
    write_once(attempt / f"host_{label}.json", evidence)


def tree_rss(process: subprocess.Popen) -> int:
    import psutil
    try:
        root = psutil.Process(process.pid)
        return sum(p.memory_info().rss for p in [root, *root.children(recursive=True)]
                   if p.is_running())
    except psutil.NoSuchProcess:
        return 0


def attempt_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_bounded(command: list[str], attempt: Path, *, seconds: int = worker.MAX_SECONDS,
                rss_bytes: int = worker.MAX_RSS_BYTES,
                disk_bytes: int = worker.MAX_EVIDENCE_BYTES) -> dict:
    """Kill the process group as soon as a monitored cap is exceeded."""
    started = time.monotonic()
    peak = 0
    reason = None
    with (attempt / "worker_stdout.txt").open("xb") as stdout, (
            attempt / "worker_stderr.txt").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            peak = max(peak, tree_rss(process))
            size = attempt_bytes(attempt)
            if elapsed > seconds:
                reason = "wall_cap"
            elif peak > rss_bytes:
                reason = "rss_cap"
            elif size > disk_bytes:
                reason = "evidence_cap"
            if reason:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.25)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=10)
            reason = reason or "kill_timeout"
    return {"exit_code": code, "limit_reason": reason,
            "elapsed_seconds": time.monotonic() - started,
            "peak_monitored_rss_bytes": peak,
            "evidence_bytes": attempt_bytes(attempt)}


def run_once(model_name: str, attempt: Path, receipt: Path, review: Path) -> dict:
    worker.require(model_name in worker.MODEL_NAMES and attempt.is_absolute()
                   and attempt.parent.is_dir() and not attempt.exists(),
                   "new absolute model attempt required")
    worker.require_independent_go(receipt, review)
    before = host_probe()
    attempt.mkdir()
    started_utc = datetime.now(timezone.utc)
    write_once(attempt / "parent_start.json", {
        "schema": "nonspeech_descriptive_parent_start_v1", "model": model_name,
        "preflight_sha256": worker.sha_file(receipt), "review_sha256": worker.sha_file(review),
        "source_identity": worker.source_identity(), "host_before_sha256":
        worker.sha_bytes(before["outputs"]["power_log"].encode()),
        "utc": started_utc.isoformat()})
    result = {"exit_code": None, "limit_reason": "launch_error",
              "elapsed_seconds": 0.0, "peak_monitored_rss_bytes": 0,
              "evidence_bytes": attempt_bytes(attempt)}
    try:
        save_host(attempt, "before", before)
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.descriptive_worker_v1",
                   "--model", model_name, "--output-dir", str(attempt),
                   "--preflight", str(receipt), "--review", str(review)]
        # Worker expects a parent-created attempt containing only parent receipts.
        result = run_bounded(command, attempt)
    except BaseException as error:
        result["launch_error"] = repr(error)
    try:
        after = host_probe()
        finished_utc = datetime.now(timezone.utc)
        log = after["outputs"]["power_log"]
        host_gate._power_log_coverage(log, started_utc, finished_utc,
                                      datetime.fromisoformat(after["collected_utc"]))
        worker.require(not host_gate._power_events(log, started_utc, finished_utc),
                       "sleep, wake, or thermal event during worker")
        save_host(attempt, "after", after)
        result["host_after_ok"] = True
    except Exception as error:
        result["host_after_ok"] = False
        result["host_after_error"] = repr(error)
    result.update({"schema": "nonspeech_descriptive_parent_result_v1",
                   "model": model_name,
                   "status": "worker_exited_pending_audit" if result["exit_code"] == 0
                   and result["limit_reason"] is None and result["host_after_ok"]
                   else "invalid_or_partial"})
    if attempt_bytes(attempt) > worker.MAX_EVIDENCE_BYTES:
        result["status"] = "invalid_or_partial"
        result["limit_reason"] = result["limit_reason"] or "evidence_cap_after_exit"
    write_once(attempt / "parent_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model", choices=worker.MODEL_NAMES)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.preflight_only:
        worker.require(args.model is None and args.attempt is None and args.review is None,
                       "preflight-only cannot launch model")
        print(worker.canonical(preflight_only(args.receipt)))
    else:
        worker.require(args.model is not None and args.attempt is not None
                       and args.review is not None, "model launch args missing")
        print(worker.canonical(run_once(args.model, args.attempt, args.receipt, args.review)))


if __name__ == "__main__":
    main()
