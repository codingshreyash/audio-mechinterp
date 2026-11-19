"""One-shot host and resource parent for Whisper causal attempt 002.

Import and preflight do not load a candidate checkpoint. The only live run
requires a separately committed admission and independent executable GO.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
from . import causal_plan_v1 as plan
from . import causal_worker_v2 as worker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTEMPT = HERE / "runs/causal_whisper_v1/attempt_002"
RSS_POLL_SECONDS = 0.10
MAX_POST_PROBE_LAG_SECONDS = 120


def expected_names() -> list[str]:
    return [f'{row["stage"]}_{row["case_id"]}_{row["arm"]}' for row in plan.schedule()]


def process_tree_rss_bytes(root_pid: int) -> int | None:
    sample = subprocess.run(("ps", "-axo", "pid=,ppid=,rss="),
                            capture_output=True, text=True, check=True, timeout=5)
    rows: dict[int, tuple[int, int]] = {}
    for line in sample.stdout.splitlines():
        values = line.split()
        if len(values) != 3:
            continue
        pid, ppid, rss_kib = map(int, values)
        rows[pid] = ppid, rss_kib
    if root_pid not in rows:
        return None
    included = {root_pid}
    while True:
        children = {pid for pid, (ppid, _rss) in rows.items() if ppid in included}
        grown = included | children
        if grown == included:
            break
        included = grown
    return sum(rows[pid][1] for pid in included) * 1024


def kill_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def output_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_bounded(argv: list[str], attempt: Path, *, seconds: int = worker.MAX_SECONDS,
                rss_bytes: int = worker.MAX_RSS_BYTES,
                evidence_bytes: int = worker.MAX_EVIDENCE_BYTES) -> dict:
    """Sample the process tree and evidence every 100 ms; kill on any limit."""
    started = time.monotonic()
    stdout, stderr = attempt / "worker_stdout.txt", attempt / "worker_stderr.txt"
    rss_log = attempt / "worker_rss_samples.jsonl"
    with stdout.open("xb") as out, stderr.open("xb") as err, \
            rss_log.open("x", encoding="utf-8") as samples_file:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err,
                                   start_new_session=True,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1",
                                        "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"})
        reason, monitor_error, peak, samples = "normal_exit", None, 0, 0
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= seconds:
                    reason = "wall_limit"
                    kill_group(process)
                    break
                try:
                    rss = process_tree_rss_bytes(process.pid)
                    size = output_bytes(attempt)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    reason, monitor_error = "monitor_error", repr(error)
                    kill_group(process)
                    break
                if rss is not None:
                    samples += 1
                    peak = max(peak, rss)
                    samples_file.write(worker.canonical({
                        "sample_index": samples, "elapsed_seconds": elapsed,
                        "worker_pid": process.pid, "tree_rss_bytes": rss,
                        "evidence_bytes": size}) + "\n")
                    samples_file.flush()
                    if rss >= rss_bytes:
                        reason = "rss_limit"
                        kill_group(process)
                        break
                if size >= evidence_bytes:
                    reason = "evidence_limit"
                    kill_group(process)
                    break
                time.sleep(RSS_POLL_SECONDS)
            process.wait(timeout=10)
        except BaseException:
            kill_group(process)
            raise
        samples_file.flush()
        os.fsync(samples_file.fileno())
    elapsed = time.monotonic() - started
    return {"worker_returncode": process.returncode,
            "worker_stop_reason": reason,
            "worker_timed_out": reason == "wall_limit" or not math.isfinite(elapsed)
            or elapsed > seconds,
            "worker_elapsed_seconds": elapsed,
            "worker_peak_tree_rss_bytes": peak,
            "worker_rss_limit_bytes": rss_bytes,
            "worker_rss_samples": samples,
            "worker_rss_sample_log_sha256": worker.sha_file(rss_log),
            "worker_rss_limit_hit": reason == "rss_limit",
            "worker_evidence_limit_hit": reason == "evidence_limit",
            "worker_monitor_error": monitor_error}


def progress_census(path: Path, names: list[str]) -> dict:
    """Classify every scheduled call even after a parent SIGKILL."""
    if not path.exists():
        return {"completed_call_names": [], "failed_attempted_call_name": None,
                "censored_call_names": names}
    worker.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1 << 20,
                   "call progress evidence invalid")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    position, completed, failed = 0, [], None
    for index, name in enumerate(names, start=1):
        if position == len(rows):
            break
        start = rows[position]
        worker.require(start == {"type": "attempt", "call_index": index,
                                 "call_name": name}, "call attempt order invalid")
        position += 1
        if position == len(rows):
            failed = name
            break
        finish = rows[position]
        worker.require(finish == {"type": "complete", "call_index": index,
                                  "call_name": name}, "call completion order invalid")
        completed.append(name)
        position += 1
    worker.require(position == len(rows), "extra call progress records")
    after = len(completed) + (1 if failed else 0)
    return {"completed_call_names": completed,
            "failed_attempted_call_name": failed,
            "censored_call_names": names[after:]}


def host_probe(attempt: Path, label: str, *, run_start: datetime | None = None) -> dict:
    evidence = host_gate.probe(run_start=run_start)
    host_gate.archive(attempt, evidence, label)
    worker.write_once(attempt / f"{label}_host.json", evidence)
    return evidence


def launch(admission: Path, go: Path) -> dict:
    receipt = worker.validate_go(admission, go)
    names = expected_names()
    worker.require(len(names) == len(plan.schedule()), "frozen call census changed")
    worker.require(not ATTEMPT.exists(), "frozen causal attempt already exists")
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPT.mkdir()
    start = datetime.now(timezone.utc)
    worker.write_once(ATTEMPT / "launch_start.json", {
        "schema": "nonspeech_whisper_causal_launch_v2", "status": "started",
        "start_utc": start.isoformat(), "attempt": str(ATTEMPT),
        "admission_sha256": worker.sha_file(admission),
        "go_sha256": worker.sha_file(go),
        "source_identity": receipt["sources"],
        "scheduled_call_names": names})
    result: dict = {"schema": "nonspeech_whisper_causal_launch_v2",
                    "status": "invalid_or_partial", "attempt": str(ATTEMPT),
                    "start_utc": start.isoformat(),
                    "admission_sha256": worker.sha_file(admission),
                    "go_sha256": worker.sha_file(go)}
    worker_exit: datetime | None = None
    try:
        pre = host_probe(ATTEMPT, "pre")
        worker.require(output_bytes(ATTEMPT) < worker.MAX_EVIDENCE_BYTES,
                       "pre-run evidence ceiling exceeded")
        cmd = [sys.executable, "-m",
               "models.second_architecture.nonspeech_probe_v1.causal_worker_v2",
               "run", "--output", str(ATTEMPT / "worker_output"),
               "--admission", str(admission.resolve()), "--go", str(go.resolve())]
        observation = run_bounded(cmd, ATTEMPT)
        worker_exit = datetime.now(timezone.utc)
        result.update(observation)
        result["worker_exit_observed_utc"] = worker_exit.isoformat()
        progress = ATTEMPT / "worker_output/call_progress.jsonl"
        result["worker_call_progress_sha256"] = (
            worker.sha_file(progress) if progress.is_file() else None)
        result["worker_call_census"] = progress_census(progress, names)
        post = host_probe(ATTEMPT, "post", run_start=start)
        collected = datetime.fromisoformat(post["collected_utc"])
        lag = (collected - worker_exit).total_seconds()
        worker.require(0 <= lag <= MAX_POST_PROBE_LAG_SECONDS,
                       "post-host probe lag exceeded")
        result["host_probe_lag_seconds"] = lag
        result["host_collected_utc"] = [pre["collected_utc"], post["collected_utc"]]
        result["pre_host_sha256"] = worker.sha_file(ATTEMPT / "pre_host.json")
        result["post_host_sha256"] = worker.sha_file(ATTEMPT / "post_host.json")
        records = ATTEMPT / "worker_output/records.jsonl"
        census = result["worker_call_census"]
        if (observation["worker_returncode"] == 0
                and observation["worker_stop_reason"] == "normal_exit"
                and not observation["worker_timed_out"]
                and 0 < observation["worker_peak_tree_rss_bytes"] < worker.MAX_RSS_BYTES
                and observation["worker_rss_samples"] > 0
                and len(census["completed_call_names"]) == len(names)
                and census["failed_attempted_call_name"] is None
                and not census["censored_call_names"] and records.is_file()):
            tail = records.read_text().splitlines()[-1:]
            if tail and json.loads(tail[0]).get("type") == "complete":
                result["status"] = "completed_pending_saved_audit"
        worker.require(output_bytes(ATTEMPT) <= worker.MAX_EVIDENCE_BYTES,
                       "parent evidence ceiling exceeded")
    except BaseException as error:
        result["launch_error"] = {"type": type(error).__name__, "message": str(error)}
    result["end_utc"] = datetime.now(timezone.utc).isoformat()
    result["evidence_bytes_before_result"] = output_bytes(ATTEMPT)
    worker.write_once(ATTEMPT / "launch_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.admission, args.go)
    print(worker.canonical({"status": result["status"],
                            "completed_calls": len(result.get("worker_call_census", {}).get(
                                "completed_call_names", [])),
                            "worker_stop_reason": result.get("worker_stop_reason"),
                            "attempt": result["attempt"]}))
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
