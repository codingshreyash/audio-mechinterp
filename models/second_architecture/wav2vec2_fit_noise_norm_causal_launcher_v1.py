"""One-shot Mac host and wall-clock gate for the FIT normalization worker."""
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

from models.second_architecture import wav2vec2_fit_noise_norm_causal_worker_v1 as worker
from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "runs/wav2vec2_fit_noise_norm_causal_v1/attempt_001"
RSS_POLL_SECONDS = 0.10


def process_tree_rss_bytes(root_pid: int) -> int | None:
    """Sum resident KiB for the worker and all currently parented descendants."""
    sample = subprocess.run(("ps", "-axo", "pid=,ppid=,rss="),
                            capture_output=True, text=True, check=True, timeout=5)
    rows: dict[int, tuple[int, int]] = {}
    for line in sample.stdout.splitlines():
        values = line.split()
        if len(values) != 3:
            continue
        pid, ppid, rss_kib = map(int, values)
        rows[pid] = (ppid, rss_kib)
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


def run_bounded(argv: list[str], stdout: Path, stderr: Path, rss_log: Path,
                *, seconds: int = worker.MAX_SECONDS) -> dict:
    start = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err, \
         rss_log.open("x", encoding="utf-8") as memory_log:
        process = subprocess.Popen(argv, cwd=worker.ROOT, stdout=out, stderr=err,
                                   start_new_session=True,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1",
                                        "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"})
        peak, samples, reason, monitor_error = 0, 0, "normal_exit", None
        try:
            while process.poll() is None:
                if time.monotonic() - start >= seconds:
                    reason = "timeout"
                    kill_group(process)
                    break
                try:
                    rss = process_tree_rss_bytes(process.pid)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    reason, monitor_error = "rss_monitor_error", str(error)
                    kill_group(process)
                    break
                if rss is not None:
                    samples += 1
                    peak = max(peak, rss)
                    memory_log.write(worker.canonical({
                        "sample_index": samples,
                        "elapsed_seconds": time.monotonic()-start,
                        "worker_pid": process.pid,
                        "tree_rss_bytes": rss}) + "\n")
                    memory_log.flush()
                    if rss >= worker.MAX_RSS:
                        reason = "rss_limit"
                        kill_group(process)
                        break
                time.sleep(RSS_POLL_SECONDS)
            process.wait(timeout=10)
        except BaseException:
            kill_group(process)
            raise
        memory_log.flush()
        os.fsync(memory_log.fileno())
    elapsed = time.monotonic() - start
    return {"worker_returncode": process.returncode,
            "worker_timed_out": reason == "timeout" or not math.isfinite(elapsed)
            or elapsed > seconds,
            "worker_elapsed_seconds": elapsed,
            "worker_peak_tree_rss_bytes": peak,
            "worker_rss_limit_bytes": worker.MAX_RSS,
            "worker_rss_samples": samples,
            "worker_rss_sample_log_sha256": worker.source.sha_file(rss_log),
            "worker_rss_limit_hit": reason == "rss_limit",
            "worker_stop_reason": reason,
            "worker_rss_monitor_error": monitor_error}


def probe_host(attempt: Path, name: str, *, run_start: datetime | None = None) -> dict:
    probe = host_gate.probe(run_start=run_start)
    host_gate.archive(attempt, probe, name)
    worker.write_once(attempt / f"{name}_host.json", probe)
    return probe


def progress_census(path: Path, expected_names: list[str]) -> dict:
    """Recover exact call status even if the worker was killed mid-forward."""
    if not path.exists():
        return {"completed_call_names": [], "failed_attempted_call_name": None,
                "censored_call_names": expected_names}
    worker.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1 << 20,
                   "call-progress evidence invalid")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    position, completed, failed = 0, [], None
    for index, name in enumerate(expected_names, start=1):
        if position == len(rows):
            break
        start = rows[position]
        worker.require(start == {"type": "attempt", "call_index": index,
                                 "call_name": name}, "call-progress attempt order invalid")
        position += 1
        if position == len(rows):
            failed = name
            break
        finish = rows[position]
        worker.require(finish == {"type": "complete", "call_index": index,
                                  "call_name": name}, "call-progress completion order invalid")
        completed.append(name)
        position += 1
    worker.require(position == len(rows), "extra call-progress rows")
    censored_start = len(completed) + (1 if failed is not None else 0)
    return {"completed_call_names": completed,
            "failed_attempted_call_name": failed,
            "censored_call_names": expected_names[censored_start:]}


def launch(admission_path: Path, go_path: Path) -> dict:
    admission = worker.validate_go(admission_path, go_path)
    expected_names = worker.ordered_forward_names(worker.execution_plan())
    worker.require(not ATTEMPT.exists(), "frozen one-shot attempt already exists")
    ATTEMPT.mkdir(parents=True)
    start_utc = datetime.now(timezone.utc)
    worker.write_once(ATTEMPT / "launch_start.json",
                      {"schema": "fit_norm_causal_launch_v1",
                       "status": "started", "start_utc": start_utc.isoformat(),
                       "admission_sha256": worker.source.sha_file(admission_path),
                       "go_sha256": worker.source.sha_file(go_path),
                       "source_binding": admission["source_binding"],
                       "attempt": str(ATTEMPT)})
    result = {"schema": "fit_norm_causal_launch_v1", "status": "invalid_or_partial",
              "start_utc": start_utc.isoformat(), "attempt": str(ATTEMPT),
              "admission_sha256": worker.source.sha_file(admission_path),
              "go_sha256": worker.source.sha_file(go_path)}
    try:
        pre = probe_host(ATTEMPT, "pre")
        worker.require(worker.output_bytes(ATTEMPT) < worker.MAX_OUTPUT,
                       "pre-run output ceiling exceeded")
        cmd = [sys.executable, "-m",
               "models.second_architecture.wav2vec2_fit_noise_norm_causal_worker_v1",
               "run", "--output", str(ATTEMPT / "worker_output"),
               "--admission", str(admission_path.resolve()),
               "--go", str(go_path.resolve())]
        observation = run_bounded(
            cmd, ATTEMPT / "worker_stdout.txt", ATTEMPT / "worker_stderr.txt",
            ATTEMPT / "worker_rss_samples.jsonl")
        result.update(observation)
        progress_path = ATTEMPT / "worker_output/call_progress.jsonl"
        result["worker_call_progress_sha256"] = (
            worker.source.sha_file(progress_path) if progress_path.is_file() else None)
        result["worker_call_census"] = progress_census(progress_path, expected_names)
        post = probe_host(ATTEMPT, "post", run_start=start_utc)
        result["pre_host_sha256"] = worker.source.sha_file(ATTEMPT / "pre_host.json")
        result["post_host_sha256"] = worker.source.sha_file(ATTEMPT / "post_host.json")
        result["host_collected_utc"] = [pre["collected_utc"], post["collected_utc"]]
        records = ATTEMPT / "worker_output/records.jsonl"
        if (observation["worker_returncode"] == 0
                and observation["worker_stop_reason"] == "normal_exit"
                and not observation["worker_timed_out"]
                and 0 < observation["worker_peak_tree_rss_bytes"] < worker.MAX_RSS
                and observation["worker_rss_samples"] > 0
                and len(result["worker_call_census"]["completed_call_names"]) == worker.MAX_FORWARDS
                and result["worker_call_census"]["failed_attempted_call_name"] is None
                and not result["worker_call_census"]["censored_call_names"]
                and records.is_file()):
            lines = records.read_text().splitlines()
            if lines and json.loads(lines[-1]).get("type") == "complete":
                result["status"] = "completed_pending_saved_audit"
        worker.require(worker.output_bytes(ATTEMPT) <= worker.MAX_OUTPUT,
                       "parent evidence ceiling exceeded")
    except BaseException as error:
        result["launch_error"] = {"type": type(error).__name__, "message": str(error)}
    result["end_utc"] = datetime.now(timezone.utc).isoformat()
    result["evidence_bytes_before_result"] = worker.output_bytes(ATTEMPT)
    worker.write_once(ATTEMPT / "launch_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.admission, args.go)
    print(worker.canonical(result))
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
