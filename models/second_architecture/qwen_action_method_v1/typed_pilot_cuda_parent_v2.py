"""One-shot, bounded Linux/CUDA parent for load and typed F_L1 pilot stages.

Importing this module never loads weights. Both independent GO receipts are absent
from source; a fresh reviewed receipt is required for each fixed attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from models.second_architecture.nonspeech_probe_v1 import causal_parent_v2 as bounded
from . import typed_pilot_cuda_backend_v2 as backend
from . import typed_pilot_cuda_admission_v2 as admission

RSS_BYTES = 24 * 1024**3
EVIDENCE_BYTES = 64 * 1024**2
RESERVE_SECONDS = 15
WALL_SECONDS = {"load": 180, "pilot": 900}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(attempt: Path, name: str, value: dict) -> None:
    backend.require(bounded.output_bytes(attempt) + len(backend.canonical(value))
                    < EVIDENCE_BYTES, "whole-attempt evidence cap before write")
    backend.write_once(attempt / name, value)
    _fsync_path(attempt)


def _gpu_used_bytes() -> int:
    result = subprocess.run(("nvidia-smi", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"), capture_output=True,
                            text=True, timeout=10, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    backend.require(len(lines) == 1 and lines[0].isdigit(),
                    "GPU memory monitor inventory invalid")
    return int(lines[0]) * 1024**2


def _host_probe() -> dict:
    return {"collected_utc": _utc(), "runtime_identity": backend.runtime_identity(),
            "gpu_inventory": admission.gpu_inventory(), "parent_pid": os.getpid()}


def _run_worker(command: list[str], attempt: Path, *, stage: str,
                deadline: float, nonce: bytes) -> dict:
    read_fd, write_fd = os.pipe()
    stdout, stderr = attempt / "worker_stdout.txt", attempt / "worker_stderr.txt"
    monitor = attempt / "worker_resource_samples.jsonl"
    started = time.monotonic()
    process = None
    reason, error, peak_rss, peak_gpu, samples, beats = "normal_exit", None, 0, 0, 0, 0
    gpu_limit = backend.LOAD_GPU_BYTES if stage == "load" else backend.PILOT_GPU_BYTES
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err, monitor.open("xb") as log:
            process = subprocess.Popen(command, cwd=backend.ROOT, stdout=out,
                stderr=err, start_new_session=True, pass_fds=(read_fd,),
                env={**os.environ, "QWEN_TYPED_PILOT_PARENT_FD": str(read_fd),
                     "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "0"})
            os.close(read_fd)
            read_fd = -1
            os.write(write_fd, nonce.hex().encode() + b"\n")
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if now >= deadline - RESERVE_SECONDS:
                        reason = "wall_limit"
                        bounded.kill_group(process)
                        break
                    try:
                        rss = bounded.process_tree_rss_bytes(process.pid)
                        gpu = _gpu_used_bytes()
                        size = bounded.output_bytes(attempt)
                        backend.require(rss is not None, "worker RSS sample unavailable")
                    except BaseException as exc:
                        reason, error = "monitor_error", repr(exc)
                        bounded.kill_group(process)
                        break
                    samples += 1
                    peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, gpu)
                    log.write(backend.canonical({"sample_index": samples,
                        "elapsed_seconds": now - started, "worker_pid": process.pid,
                        "tree_rss_bytes": rss, "gpu_used_bytes": gpu,
                        "evidence_bytes": size}))
                    log.flush()
                    if rss >= RSS_BYTES or gpu >= gpu_limit:
                        reason = "rss_limit" if rss >= RSS_BYTES else "gpu_limit"
                        bounded.kill_group(process)
                        break
                    if size >= EVIDENCE_BYTES - 2 * 1024**2:
                        reason = "evidence_limit"
                        bounded.kill_group(process)
                        break
                    try:
                        os.write(write_fd, b".")
                        beats += 1
                    except BrokenPipeError:
                        reason = "parent_pipe_closed"
                        bounded.kill_group(process)
                        break
                    time.sleep(0.2)
                process.wait(timeout=10)
            except BaseException:
                bounded.kill_group(process)
                raise
            for handle in (log, out, err):
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)
    return {"worker_returncode": process.returncode,
            "worker_stop_reason": reason,
            "worker_elapsed_seconds": time.monotonic() - started,
            "worker_peak_tree_rss_bytes": peak_rss,
            "worker_peak_gpu_used_bytes": peak_gpu,
            "worker_resource_samples": samples,
            "worker_resource_sample_log_sha256": backend.sha(monitor),
            "worker_parent_heartbeats": beats,
            "worker_monitor_error": error}


def launch(stage: str) -> dict:
    attempt, go = backend.stage_paths(stage)
    receipt = backend.validate_go(stage)  # Must precede attempt creation.
    backend.require(not attempt.exists(), "fixed CUDA attempt already used")
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.mkdir()
    _fsync_path(attempt.parent)
    began, started = _utc(), time.monotonic()
    deadline = started + WALL_SECONDS[stage]
    result: dict = {"schema": "qwen_action_typed_pilot_cuda_parent_v2",
                    "status": "invalid_or_partial", "stage": stage,
                    "start_utc": began, "go_sha256": backend.sha(go)}
    complete = False
    try:
        pre = _host_probe()
        backend.require(pre["runtime_identity"] == receipt["runtime_identity"],
                        "pre-run GPU runtime identity changed")
        _write(attempt, "pre_host.json", pre)
        stage_start_gpu = admission.gpu_inventory()
        backend.require(all(stage_start_gpu[key] == receipt["gpu_baseline"][key]
                            for key in ("gpu_uuid", "gpu_name", "gpu_memory_mib",
                                        "mig_mode", "compute_processes",
                                        "idle_gpu_limit_bytes"))
                        and type(stage_start_gpu["gpu_used_bytes"]) is int
                        and 0 <= stage_start_gpu["gpu_used_bytes"]
                            < admission.IDLE_GPU_LIMIT_BYTES,
                        "stage-start GPU inventory changed")
        _write(attempt, "stage_start_gpu.json", stage_start_gpu)
        nonce = secrets.token_bytes(32)
        _write(attempt, "start.json", {"schema": "qwen_action_typed_pilot_cuda_parent_v2",
            "status": "worker_authorized", "stage": stage,
            "start_utc": began, "approved_attempt": str(attempt.resolve()),
            "go_sha256": backend.sha(go), "pre_host_sha256": backend.sha(attempt / "pre_host.json"),
            "stage_start_gpu_sha256": backend.sha(attempt / "stage_start_gpu.json"),
            "source_identity": receipt["source_identity"], "runtime_identity": receipt["runtime_identity"],
            "parent_pid": os.getpid(), "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
            "deadline_monotonic": deadline})
        result["worker_seconds_granted"] = deadline - time.monotonic() - RESERVE_SECONDS
        backend.require(result["worker_seconds_granted"] > 0,
                        "worker has no wall allowance")
        command = [sys.executable, "-m",
            "models.second_architecture.qwen_action_method_v1.typed_pilot_cuda_backend_v2",
            "--worker", "--stage", stage, "--attempt", str(attempt)]
        result.update(_run_worker(command, attempt, stage=stage,
                                  deadline=deadline, nonce=nonce))
        result["worker_exit_utc"] = _utc()
        backend.require(time.monotonic() < deadline - RESERVE_SECONDS,
                        "worker exhausted terminal reserve")
        post = _host_probe()
        backend.require(post["runtime_identity"] == receipt["runtime_identity"],
                        "post-run GPU runtime identity changed")
        _write(attempt, "post_host.json", post)
        result["pre_host_sha256"] = backend.sha(attempt / "pre_host.json")
        result["post_host_sha256"] = backend.sha(attempt / "post_host.json")
        complete = (result["worker_returncode"] == 0
            and result["worker_stop_reason"] == "normal_exit"
            and 0 < result["worker_elapsed_seconds"] <= result["worker_seconds_granted"]
            and result["worker_resource_samples"] > 0
            and result["worker_parent_heartbeats"] > 0
            and 0 < result["worker_peak_tree_rss_bytes"] < RSS_BYTES
            and 0 <= result["worker_peak_gpu_used_bytes"]
                < (backend.LOAD_GPU_BYTES if stage == "load" else backend.PILOT_GPU_BYTES)
            and (attempt / "worker_output/result.json").is_file()
            and bounded.output_bytes(attempt) < EVIDENCE_BYTES)
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["status"] = "candidate_saved_pending_audit" if complete else "invalid_or_partial"
    result["end_utc"] = _utc()
    result["elapsed_seconds_before_result"] = time.monotonic() - started
    _write(attempt, "result.json", result)
    if complete:
        terminal = attempt / "terminal.json"
        try:
            backend.require(time.monotonic() < deadline,
                            "terminal deadline exhausted")
            _fsync_path(attempt / "worker_output")
            _fsync_path(attempt)
            _write(attempt, "terminal.json", {
                "schema": "qwen_action_typed_pilot_cuda_terminal_v2",
                "status": "candidate_saved_pending_audit", "stage": stage,
                "result_sha256": backend.sha(attempt / "result.json"),
                "start_sha256": backend.sha(attempt / "start.json"),
                "elapsed_seconds_after_result": time.monotonic() - started,
                "terminal_utc": _utc()})
            backend.require(time.monotonic() < deadline,
                            "terminal write exceeded deadline")
        except BaseException:
            if terminal.exists():
                terminal.unlink()
                _fsync_path(attempt)
            raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--capture-admission", action="store_true")
    parser.add_argument("--run-stage", choices=("load", "pilot"))
    args = parser.parse_args()
    if args.source_preflight:
        print(json.dumps({"status": "independent_go_required",
            "load_attempt_exists": backend.LOAD_ATTEMPT.exists(),
            "pilot_attempt_exists": backend.PILOT_ATTEMPT.exists(),
            "load_go_exists": backend.LOAD_GO.exists(),
            "pilot_go_exists": backend.PILOT_GO.exists()}, sort_keys=True))
    elif args.capture_admission:
        print(json.dumps({"schema": "qwen_action_cuda_admission_candidate_v2",
            "source_identity": backend.source_identity(),
            "runtime_identity": backend.runtime_identity(),
            "gpu_baseline": admission.gpu_inventory()}, sort_keys=True))
    elif args.run_stage:
        result = launch(args.run_stage)
        print(json.dumps({"status": result["status"],
                          "stage": args.run_stage}, sort_keys=True))
        if result["status"] != "candidate_saved_pending_audit":
            raise SystemExit(2)
    else:
        parser.error("choose --source-preflight, --capture-admission, or --run-stage")


if __name__ == "__main__":
    main()
