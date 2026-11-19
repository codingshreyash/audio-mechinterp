"""Bounded one-shot parent for the Whisper time-translation probe.

This module imports no model weights. It requires a separately committed GO,
then monitors one stage's child process, host state, memory and saved bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import psutil

from . import whisper_tail_causal_host_v1 as host
from . import whisper_time_translation_gate_v1 as gate

ROOT = gate.ROOT
LOCK = gate.RUNS / "active_model_lock"
POLL_SECONDS = 0.1
HOST_PROBE_SECONDS = 2.0
HOST_MAX_GAP_SECONDS = 15.0
POST_RESERVE_SECONDS = 180.0
EVIDENCE_RESERVE_BYTES = 32 * 1024**2
MAX_PROGRESS_BYTES = 128 * 1024
MAX_RECORD_BYTES = 4 * 1024**2


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def write_once(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def evidence_bytes(root: Path) -> int:
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            item = Path(base) / name
            need(not item.is_symlink(), "linked evidence entry")
            if item.is_file():
                total += item.stat().st_size
            else:
                need(item.is_dir(), "special evidence entry")
    return total


def tree_rss_bytes(pid: int) -> int | None:
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
        return sum(proc.memory_info().rss for proc in processes if proc.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _read_jsonl(path: Path, cap: int) -> list[dict]:
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size <= cap, "missing or oversized JSONL")
    data = path.read_bytes()
    need(data.endswith(b"\n"), "incomplete JSONL line")
    return [json.loads(line) for line in data.splitlines()]


def progress_census(attempt: Path, stage: str) -> dict:
    max_calls = 28 if stage == "original" else 62
    path = attempt / "progress.jsonl"
    if not path.exists():
        return {"complete": 0, "attempted_incomplete": None}
    rows = _read_jsonl(path, MAX_PROGRESS_BYTES)
    need(len(rows) <= 2 * max_calls, "progress exceeds stage schedule")
    complete = 0
    for index in range(1, max_calls + 1):
        at = 2 * (index - 1)
        if at >= len(rows):
            break
        need(rows[at] == {"type": "attempt", "index": index},
             "progress attempt order changed")
        if at + 1 >= len(rows):
            break
        need(rows[at + 1] == {"type": "complete", "index": index},
             "progress completion order changed")
        complete += 1
    need(len(rows) in (2 * complete, 2 * complete + 1), "progress extras")
    return {"complete": complete,
            "attempted_incomplete": complete + 1 if len(rows) % 2 else None}


def complete_worker_output(attempt: Path, stage: str) -> bool:
    """Check envelope only. The independent saved auditor decides result PASS."""
    try:
        max_calls = 28 if stage == "original" else 62
        census = progress_census(attempt, stage)
        rows = _read_jsonl(attempt / "records.jsonl", MAX_RECORD_BYTES)
        if not rows or len(rows) != census["complete"] + 1:
            return False
        end = rows[-1]
        if (not {"type", "stage", "executed_calls", "planned_max_calls",
                 "skipped_arms"}.issubset(end)
                or end["type"] != "complete" or end["stage"] != stage
                or type(end["executed_calls"]) is not int
                or end["executed_calls"] != census["complete"]
                or census["attempted_incomplete"] is not None
                or end["planned_max_calls"] != max_calls
                or not 2 <= end["executed_calls"] <= max_calls
                or type(end["skipped_arms"]) is not list
                or len(end["skipped_arms"]) != max_calls - end["executed_calls"]):
            return False
        if stage == "original" and end["executed_calls"] != max_calls:
            return False
        if stage != "original" and end["executed_calls"] < 4:
            return False
        skipped = end["skipped_arms"]
        arm_count = max_calls - (2 if stage == "original" else 4)
        if (any(type(ordinal) is not int or not 1 <= ordinal <= arm_count
                for ordinal in skipped)
                or skipped != sorted(set(skipped))):
            return False
        return all(row.get("type") == "call" and row.get("index") == index
                   for index, row in enumerate(rows[:-1], 1))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _write_host(out, samples: list[dict], sample: dict) -> None:
    if samples:
        need(0 < sample["monotonic_seconds"] - samples[-1]["monotonic_seconds"]
             <= HOST_MAX_GAP_SECONDS, "host probe cadence lost")
    samples.append(sample)
    out.write(canonical(sample))
    out.flush()
    os.fsync(out.fileno())


def run_bounded(command: list[str], attempt: Path, stage: str, *, nonce: str,
                deadline: float, samples: list[dict], host_out) -> dict:
    read_fd, write_fd = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    pending: Future | None = None
    probe_started = 0.0
    peak = beats = count = 0
    reason = "normal_exit"
    error_type: str | None = None
    rss_log = attempt / "worker_rss_samples.jsonl"
    probes = ThreadPoolExecutor(max_workers=1)
    try:
        with (attempt / "worker_stdout.txt").open("xb") as stdout, \
             (attempt / "worker_stderr.txt").open("xb") as stderr, \
             rss_log.open("xb") as rss_out:
            process = subprocess.Popen(command + ["--heartbeat-fd", str(read_fd)],
                cwd=ROOT, stdout=stdout, stderr=stderr, pass_fds=(read_fd,),
                start_new_session=True,
                env={**os.environ, "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"})
            os.close(read_fd)
            read_fd = -1
            os.write(write_fd, b"HELLO " + nonce.encode() + b"\n")
            next_beat = time.monotonic()
            next_host = samples[-1]["monotonic_seconds"] + HOST_PROBE_SECONDS
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline - POST_RESERVE_SECONDS:
                    reason = "wall_limit"
                    kill_group(process)
                    break
                if pending is None and now >= next_host:
                    probe_started = now
                    pending = probes.submit(host.quick_probe, initial=False)
                if pending is not None and pending.done():
                    try:
                        _write_host(host_out, samples, pending.result())
                    except BaseException as error:
                        reason, error_type = "host_probe_failed", type(error).__name__
                        kill_group(process)
                        break
                    next_host = samples[-1]["monotonic_seconds"] + HOST_PROBE_SECONDS
                    pending = None
                if pending is not None and now - probe_started >= HOST_MAX_GAP_SECONDS:
                    reason = "host_probe_timeout"
                    kill_group(process)
                    break
                try:
                    size = evidence_bytes(attempt)
                    rss = tree_rss_bytes(process.pid)
                except BaseException as error:
                    reason, error_type = "monitor_failed", type(error).__name__
                    kill_group(process)
                    break
                if rss is not None and rss > 0:
                    count += 1
                    peak = max(peak, rss)
                    rss_out.write(canonical({"sample_index": count,
                        "monotonic_seconds": now, "tree_rss_bytes": rss,
                        "evidence_bytes": size}))
                    rss_out.flush()
                    if rss >= gate.RSS_BYTES:
                        reason = "rss_limit"
                        kill_group(process)
                        break
                if size >= gate.EVIDENCE_BYTES - EVIDENCE_RESERVE_BYTES:
                    reason = "evidence_limit"
                    kill_group(process)
                    break
                if now >= next_beat:
                    try:
                        os.write(write_fd, b"H\n")
                        beats += 1
                    except BrokenPipeError:
                        # The worker can close the pipe just before exiting.
                        time.sleep(0.5)
                        if process.poll() is None:
                            reason = "heartbeat_pipe_lost"
                            kill_group(process)
                        break
                    next_beat = now + 0.25
                time.sleep(POLL_SECONDS)
            process.wait(timeout=10)
            if pending is not None:
                try:
                    _write_host(host_out, samples, pending.result(
                        timeout=max(0.0, HOST_MAX_GAP_SECONDS -
                                    (time.monotonic() - probe_started))))
                except BaseException as error:
                    reason, error_type = "host_probe_failed", type(error).__name__
            rss_out.flush()
            os.fsync(rss_out.fileno())
    except BaseException:
        if process is not None and process.poll() is None:
            kill_group(process)
        raise
    finally:
        probes.shutdown(wait=False, cancel_futures=True)
        if process is not None:
            # The leader can exit while a child remains in its session.
            kill_group(process)
        if read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)
    need(process is not None, "worker not started")
    return {"worker_returncode": process.returncode,
            "worker_stop_reason": reason, "worker_peak_tree_rss_bytes": peak,
            "worker_rss_samples": count, "worker_monitor_error_type": error_type,
            "worker_rss_sample_log_sha256": gate.sha(rss_log),
            "parent_heartbeat_count": beats}


def launch(stage: str, admission: Path, go: Path) -> dict:
    receipt = gate.validate_go(stage, admission, go)
    attempt = gate.ATTEMPTS[stage]
    need(attempt.is_absolute() and attempt.parent.is_dir() and not attempt.exists(),
         "fresh fixed time-translation attempt required")
    need(shutil.disk_usage(attempt.parent).free >= 2 * gate.EVIDENCE_BYTES,
         "disk headroom below 2 GiB")
    need(psutil.virtual_memory().available >= 5 * 1024**3,
         "memory headroom below 5 GiB")
    need(not LOCK.exists(), "another time-translation checkpoint active")
    started = time.monotonic()
    deadline = started + gate.WALL_SECONDS
    start_utc = datetime.now(timezone.utc)
    initial = host.quick_probe(initial=True)
    need(0 <= initial["monotonic_seconds"] - started <= HOST_MAX_GAP_SECONDS
         and time.monotonic() < deadline - POST_RESERVE_SECONDS,
         "initial host probe exhausted budget")
    nonce = secrets.token_hex(16)
    LOCK.mkdir()
    try:
        attempt.mkdir()
    except BaseException:
        LOCK.rmdir()
        raise
    result = {"schema": "whisper_time_translation_parent_v1",
              "status": "invalid_or_partial", "stage": stage,
              "attempt": str(attempt), "parent_pid": os.getpid(),
              "start_utc": start_utc.isoformat(),
              "start_monotonic": started, "deadline_monotonic": deadline,
              "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
              "admission_sha256": gate.sha(admission),
              "go_sha256": gate.sha(go),
              "source_identity": receipt["source_identity"]}
    try:
        write_once(attempt / "launch_start.json", {**result, "status": "started"})
        samples: list[dict] = []
        with (attempt / "host_samples.jsonl").open("xb") as host_out:
            _write_host(host_out, samples, initial)
            command = [sys.executable, "-B", "-m",
                "models.second_architecture.nonspeech_probe_v1.whisper_time_translation_worker_v1",
                "run", "--stage", stage, "--attempt", str(attempt),
                "--admission", str(admission), "--go", str(go)]
            monitor = run_bounded(command, attempt, stage, nonce=nonce,
                                  deadline=deadline, samples=samples,
                                  host_out=host_out)
            worker_exit_utc = datetime.now(timezone.utc)
            _write_host(host_out, samples, host.quick_probe(initial=False))
        monitor_end = time.monotonic()
        host.validate_cadence(samples, start=started, end=monitor_end)
        result.update({"host_monitor_end_monotonic": monitor_end,
                       "monitor": monitor,
                       "worker_exit_utc": worker_exit_utc.isoformat(),
                       "host_samples_sha256": gate.sha(attempt / "host_samples.jsonl"),
                       "host_samples_count": len(samples),
                       "progress": progress_census(attempt, stage)})
        need(time.monotonic() < deadline, "whole deadline before post log")
        power = host.power_log(start_utc, datetime.now(timezone.utc))
        with (attempt / "post_power_log.txt").open("x") as stream:
            stream.write(power)
            stream.flush()
            os.fsync(stream.fileno())
        result["post_power_log_sha256"] = gate.sha(attempt / "post_power_log.txt")
        result["post_power_log_collected_utc"] = datetime.now(timezone.utc).isoformat()
        if (monitor["worker_returncode"] == 0
                and monitor["worker_stop_reason"] == "normal_exit"
                and 0 < monitor["worker_peak_tree_rss_bytes"] < gate.RSS_BYTES
                and complete_worker_output(attempt, stage)
                and evidence_bytes(attempt) < gate.EVIDENCE_BYTES
                and time.monotonic() < deadline):
            result["status"] = "completed_pending_saved_audit"
    except BaseException as error:
        result["error"] = {"type": type(error).__name__}
    finally:
        result["end_utc"] = datetime.now(timezone.utc).isoformat()
        result["whole_elapsed_seconds"] = time.monotonic() - started
        try:
            result["evidence_bytes_before_result"] = evidence_bytes(attempt)
            if (result["whole_elapsed_seconds"] >= gate.WALL_SECONDS
                    or result["evidence_bytes_before_result"] >= gate.EVIDENCE_BYTES):
                result["status"] = "invalid_or_partial"
        except BaseException as error:
            result["status"] = "invalid_or_partial"
            result["evidence_error_type"] = type(error).__name__
        try:
            write_once(attempt / "parent_result.json", result)
        finally:
            LOCK.rmdir()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=gate.STAGES, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.stage, args.admission, args.go)
    print(canonical({"status": result["status"], "attempt": result["attempt"]}).decode(),
          end="")
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
