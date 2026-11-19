"""Bounded one-shot H100 parent. No model import in this process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from . import gate, plan

SCHEMA = "qwen_audio_resource_parent_v1"
RESERVE_SECONDS = 20


def output_bytes(folder) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def process_tree_rss_bytes(root_pid: int) -> int | None:
    sample = subprocess.run(("ps", "-axo", "pid=,ppid=,rss="), capture_output=True,
                            text=True, timeout=5, check=True)
    rows = {}
    for line in sample.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(x.isdecimal() for x in fields):
            pid, ppid, rss_kib = map(int, fields)
            rows[pid] = (ppid, rss_kib)
    if root_pid not in rows:
        return None
    included = {root_pid}
    while True:
        grown = included | {pid for pid, (ppid, _) in rows.items() if ppid in included}
        if grown == included:
            break
        included = grown
    return sum(rows[pid][1] for pid in included) * 1024


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(name: str, value: dict) -> None:
    gate.require(output_bytes(plan.ATTEMPT) + len(json.dumps(value))
                 < plan.EVIDENCE_BYTES, "whole-attempt evidence cap")
    gate.write_once(plan.ATTEMPT / name, value)


def gpu_used_bytes() -> int:
    rows = gate._smi("--query-gpu=memory.used")
    gate.require(len(rows) == 1 and rows[0].isdecimal(), "GPU monitor unavailable")
    return int(rows[0]) * 1024**2


def _probe() -> dict:
    gpu = gate.gpu_inventory()
    return {"utc": utc(), "runtime_identity": gate.runtime_identity(require_idle=False),
            "gpu_inventory": gpu, "parent_pid": os.getpid()}


def _monitor(deadline: float, nonce: bytes) -> dict:
    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    process = None
    reason, error = "normal_exit", None
    samples, beats, peak_rss, peak_gpu = 0, 0, 0, 0
    resource_path = plan.ATTEMPT / "worker_resource_samples.jsonl"
    try:
        with (plan.ATTEMPT / "worker_stdout.txt").open("xb") as out, \
             (plan.ATTEMPT / "worker_stderr.txt").open("xb") as err, \
             resource_path.open("xb") as log:
            cmd = [sys.executable, "-m",
                "models.second_architecture.qwen_action_method_v1.audio_resource_pilot_v1.worker",
                "--worker", "--attempt", str(plan.ATTEMPT)]
            process = subprocess.Popen(cmd, cwd=plan.ROOT, stdout=out, stderr=err,
                start_new_session=True, pass_fds=(read_fd,),
                env={**os.environ, "QWEN_AUDIO_RESOURCE_PARENT_FD": str(read_fd),
                     "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "0"})
            os.close(read_fd)
            read_fd = -1
            os.write(write_fd, nonce.hex().encode() + b"\n")
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline - RESERVE_SECONDS:
                    reason = "wall_limit"
                    kill_group(process)
                    break
                try:
                    rss = process_tree_rss_bytes(process.pid)
                    gpu = gpu_used_bytes()
                    evidence = output_bytes(plan.ATTEMPT)
                    gate.require(rss is not None, "worker RSS unavailable")
                except BaseException as exc:
                    reason, error = "monitor_error", repr(exc)
                    kill_group(process)
                    break
                samples += 1
                peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, gpu)
                row = {"sample_index": samples, "elapsed_seconds": now-started,
                       "tree_rss_bytes": rss, "gpu_used_bytes": gpu,
                       "evidence_bytes": evidence}
                log.write((json.dumps(row, sort_keys=True) + "\n").encode())
                log.flush()
                if rss >= plan.RSS_BYTES or gpu >= plan.GPU_BYTES:
                    reason = "rss_limit" if rss >= plan.RSS_BYTES else "gpu_limit"
                    kill_group(process)
                    break
                if evidence >= plan.EVIDENCE_BYTES - 2 * 1024**2:
                    reason = "evidence_limit"
                    kill_group(process)
                    break
                try:
                    os.write(write_fd, b".")
                    beats += 1
                except BrokenPipeError:
                    reason = "parent_pipe_closed"
                    kill_group(process)
                    break
                time.sleep(0.2)
            process.wait(timeout=10)
            for handle in (out, err, log):
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)
    return {"worker_returncode": process.returncode if process else None,
            "worker_stop_reason": reason, "worker_monitor_error": error,
            "worker_elapsed_seconds": time.monotonic()-started,
            "worker_resource_samples": samples, "worker_parent_heartbeats": beats,
            "worker_peak_tree_rss_bytes": peak_rss,
            "worker_peak_gpu_used_bytes": peak_gpu,
            "worker_resource_log_sha256": plan.sha(resource_path)}


def launch() -> dict:
    go = gate.validate_go(live=True)  # Before attempt creation.
    gate.require(not plan.ATTEMPT.exists(), "one-shot attempt already exists")
    plan.ATTEMPT.mkdir(parents=True)
    began = utc()
    started = time.monotonic()
    deadline = started + plan.MAX_SECONDS
    result = {"schema": SCHEMA, "status": "invalid_or_partial",
              "start_utc": began, "go_sha256": plan.sha(plan.GO)}
    complete = False
    try:
        pre = _probe()
        gate.require(pre["runtime_identity"] == go["runtime_identity"],
                     "pre-run runtime differs")
        _write("pre_host.json", pre)
        nonce = secrets.token_bytes(32)
        _write("start.json", {"schema": SCHEMA, "status": "worker_authorized",
            "attempt": str(plan.ATTEMPT.resolve()), "start_utc": began,
            "source_identity": go["source_identity"],
            "runtime_identity": go["runtime_identity"],
            "go_sha256": plan.sha(plan.GO), "pre_host_sha256": plan.sha(plan.ATTEMPT / "pre_host.json"),
            "parent_pid": os.getpid(), "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
            "deadline_monotonic": deadline})
        result["worker_seconds_granted"] = deadline-time.monotonic()-RESERVE_SECONDS
        gate.require(result["worker_seconds_granted"] > 0, "worker has no time")
        result.update(_monitor(deadline, nonce))
        result["worker_exit_utc"] = utc()
        gate.require(time.monotonic() < deadline-RESERVE_SECONDS,
                     "worker exhausted terminal reserve")
        post = _probe()
        gate.require(post["runtime_identity"] == go["runtime_identity"],
                     "post-run runtime differs")
        _write("post_host.json", post)
        result["pre_host_sha256"] = plan.sha(plan.ATTEMPT / "pre_host.json")
        result["post_host_sha256"] = plan.sha(plan.ATTEMPT / "post_host.json")
        complete = (result["worker_returncode"] == 0
            and result["worker_stop_reason"] == "normal_exit"
            and result["worker_resource_samples"] > 0
            and result["worker_parent_heartbeats"] > 0
            and 0 < result["worker_peak_tree_rss_bytes"] < plan.RSS_BYTES
            and 0 <= result["worker_peak_gpu_used_bytes"] < plan.GPU_BYTES
            and (plan.ATTEMPT / "worker_output/result.json").is_file()
            and output_bytes(plan.ATTEMPT) < plan.EVIDENCE_BYTES)
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["status"] = "candidate_saved_pending_audit" if complete else "invalid_or_partial"
    result["end_utc"] = utc()
    result["elapsed_seconds_before_result"] = time.monotonic()-started
    _write("result.json", result)
    if complete:
        _write("terminal.json", {"schema": "qwen_audio_resource_terminal_v1",
            "status": result["status"], "start_sha256": plan.sha(plan.ATTEMPT / "start.json"),
            "result_sha256": plan.sha(plan.ATTEMPT / "result.json"),
            "terminal_utc": utc(), "elapsed_seconds_after_result": time.monotonic()-started})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--source-preflight", action="store_true")
    choice.add_argument("--capture-admission", action="store_true")
    choice.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.source_preflight:
        print(json.dumps({"status": "independent_go_required",
            "wav": plan.verify_wav(), "go_exists": plan.GO.exists(),
            "attempt_exists": plan.ATTEMPT.exists()}, sort_keys=True))
    elif args.capture_admission:
        print(json.dumps(gate.capture_admission(), sort_keys=True))
    else:
        output = launch()
        print(json.dumps({"status": output["status"]}, sort_keys=True))
        if output["status"] != "candidate_saved_pending_audit":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
