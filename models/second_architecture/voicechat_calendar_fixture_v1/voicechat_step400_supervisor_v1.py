"""Source-only bounded parent for a future VoiceChat step-400 worker.

No CLI, NeMo import, model load, or executable GO lives here. A separately
reviewed launcher must bind the exact command, environment, and host freeze.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import selectors
import shutil
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .public_sample_smoke_v1 import _process_tree

MAX_GPU_MIB = 130_000
MAX_WALL_SECONDS = 1_800
MAX_EVIDENCE_BYTES = 64 * 1024**2
TERMINAL_RESERVE_BYTES = 65_536
MAX_LOG_BYTES = 8 * 1024**2
POLL_PERIOD_SECONDS = 0.1
MAX_SAMPLE_GAP_SECONDS = 0.2
HEARTBEAT_DEADLINE_SECONDS = 5.0
MAX_POLL_RECORDS = 36_000
MIN_POST_STAGING_FREE_BYTES = 30 * 1024**3


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _write_once(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _reserved_receipt_matches(path: Path, fd: int, expected_size: int = 0) -> bool:
    """Reject a sibling receipt replaced while the child held the same UID."""
    try:
        held, named = os.fstat(fd), path.lstat()
    except FileNotFoundError:
        return False
    return (stat.S_ISREG(named.st_mode) and held.st_dev == named.st_dev
            and held.st_ino == named.st_ino and held.st_nlink == 1
            and held.st_size == named.st_size == expected_size)


def _finish_reserved_receipt(path: Path, fd: int, data: bytes) -> bool:
    """Write only to the parent's reserved inode; never overwrite child text."""
    if not _reserved_receipt_matches(path, fd):
        return False
    expected_size = len(data)
    while data:
        written = os.write(fd, data)
        need(written > 0, "reserved receipt short write")
        data = data[written:]
    os.fsync(fd)
    return _reserved_receipt_matches(path, fd, expected_size)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _inventory(root: Path) -> tuple[dict[str, dict], int]:
    rows: dict[str, dict] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        need(not path.is_symlink(), "evidence symlink rejected")
        if path.is_dir():
            raise ValueError("nested evidence directory rejected")
        need(path.is_file(), "nonregular evidence entry")
        rel = str(path.relative_to(root))
        need(len(rel) <= 128, "evidence path too long")
        size = path.stat().st_size
        total += size
        rows[rel] = {"bytes": size}
        need(len(rows) <= 32, "evidence file census exceeded")
    return rows, total


def _evidence_priority(name: str) -> tuple[int, str]:
    return (0 if name == "gpu_poll.jsonl" else
            1 if name.endswith("checkpoints.jsonl") else
            2 if name in {"stdout.log", "stderr.log"} else 3, name)


def _name_hash(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8", "surrogateescape")).hexdigest()


def _retain_within_budget(root: Path, budget: int) -> list[dict[str, int | str]]:
    """After a killed child, preserve bounded prefixes with explicit truncations."""
    files, total = _inventory(root)
    if total <= budget:
        return []
    retained = 0
    truncations: list[dict[str, int | str]] = []
    for name in sorted(files, key=_evidence_priority):
        path = root / name
        before = files[name]["bytes"]
        allowed = min(before, max(0, budget - retained))
        if allowed != before:
            with path.open("r+b") as stream:
                stream.truncate(allowed)
                stream.flush()
                os.fsync(stream.fileno())
            truncations.append({"path_sha256": _name_hash(name), "original_bytes": before,
                                "retained_bytes": allowed})
        retained += allowed
    need(retained <= budget, "failed to retain within evidence budget")
    return truncations


def _hash_inventory(root: Path) -> dict[str, dict]:
    rows, _ = _inventory(root)
    hashed: dict[str, dict] = {}
    for name, info in rows.items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        hashed[_name_hash(name)] = {"bytes": info["bytes"], "sha256": digest.hexdigest()}
    return hashed


def _limit_file_size() -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    ceiling = MAX_EVIDENCE_BYTES - TERMINAL_RESERVE_BYTES
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (ceiling if hard == resource.RLIM_INFINITY else min(ceiling, hard), hard))


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise RuntimeError("process group did not stop")


@dataclass(frozen=True)
class GpuSample:
    """Conservative device usage for cap plus process usage for diagnosis."""

    device_mib: int
    process_mib: int | None
    model_pid_present: bool


class NvmlGpuSampler:
    """Keep direct NVML initialized across polls; a slow call fails coverage."""

    def __init__(self) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        try:
            need(pynvml.nvmlDeviceGetCount() == 1, "expected one GPU")
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def __call__(self, pid: int) -> GpuSample:
        pynvml = self.nvml
        handle = self.handle
        device_mib = (int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) + 1024**2 - 1) // 1024**2
        descendants = _process_tree(pid)
        running = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        process_bytes = 0
        present = False
        for row in running:
            if int(row.pid) in descendants:
                amount = int(row.usedGpuMemory)
                need(amount >= 0, "NVML process usage unavailable")
                process_bytes += amount
                present |= int(row.pid) == pid
        process_mib = ((process_bytes + 1024**2 - 1) // 1024**2) if present else None
        return GpuSample(device_mib=device_mib, process_mib=process_mib,
                         model_pid_present=present)

    def close(self) -> None:
        self.nvml.nvmlShutdown()


def _valid_sample(sample: GpuSample) -> None:
    need(type(sample) is GpuSample and type(sample.device_mib) is int
         and sample.device_mib >= 0 and type(sample.model_pid_present) is bool
         and (sample.process_mib is None or
              (type(sample.process_mib) is int and sample.process_mib >= 0)),
         "GPU telemetry invalid")
    need(sample.model_pid_present == (sample.process_mib is not None),
         "GPU PID and usage status disagree")
    need(sample.process_mib is None or sample.device_mib >= sample.process_mib,
         "GPU process usage exceeds device usage")


def _drain(selector: selectors.BaseSelector, log_sizes: dict[str, int],
           heartbeat: dict[str, float], now: float) -> str | None:
    for key, _ in selector.select(timeout=0):
        role, path = key.data
        fd = key.fileobj if isinstance(key.fileobj, int) else key.fileobj.fileno()
        block = os.read(fd, 65536)
        if not block:
            selector.unregister(key.fileobj)
            if isinstance(key.fileobj, int):
                os.close(key.fileobj)
            else:
                key.fileobj.close()
            continue
        if role == "heartbeat":
            need(all(byte == 1 for byte in block), "invalid heartbeat byte")
            heartbeat["last"] = now
            heartbeat["count"] += len(block)
            continue
        assert path is not None
        if log_sizes[role] + len(block) > MAX_LOG_BYTES:
            return f"{role}_log_cap"
        log_fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            need(stat.S_ISREG(os.fstat(log_fd).st_mode), "log path no longer regular")
            os.write(log_fd, block)
        finally:
            os.close(log_fd)
        log_sizes[role] += len(block)
    return None


def supervise_candidate(
    *, command: list[str], environment: dict[str, str], output_dir: Path,
    staged_checkpoint_dir: Path,
    gpu_sample: Callable[[int], GpuSample] | None = None,
    clock: Callable[[], float] = time.monotonic,
    disk_free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
) -> dict:
    """Run one externally admitted child; this function is not an admission.

    The future worker must write byte 1 to VOICECHAT_DIAG_HEARTBEAT_FD at least
    once per 5 seconds, beginning before model load. This parent samples GPU
    conservatively every 0.1 s, fails if an observed interval exceeds 0.2 s,
    and stops the entire child session on any resource/telemetry failure.
    """
    need(os.name == "posix" and command and all(type(x) is str and x for x in command)
         and type(environment) is dict and not output_dir.exists()
         and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}", output_dir.name),
         "invalid or consumed diagnostic attempt")
    external_receipt_path = output_dir.parent / f"{output_dir.name}.parent_failure.json"
    need(not external_receipt_path.exists() and not external_receipt_path.is_symlink(),
         "external failure receipt path consumed")
    need(staged_checkpoint_dir.is_dir() and not staged_checkpoint_dir.is_symlink(),
         "staged checkpoint directory missing or linked")
    need(output_dir.parent.is_dir() and not output_dir.parent.is_symlink()
         and output_dir.parent.stat().st_dev == staged_checkpoint_dir.stat().st_dev,
         "diagnostic output must share the staged checkpoint filesystem")
    need(disk_free_bytes(staged_checkpoint_dir) >= MIN_POST_STAGING_FREE_BYTES,
         "less than 30 GiB free after staging")
    need(not any(key.endswith("KEY_HEX") for key in environment),
         "review/controller secret in child environment")
    # Claim the parent-reserved failure path before creating the attempt. All
    # subsequent startup operations, including os.pipe(), are inside try.
    receipt_fd = os.open(external_receipt_path,
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    stdout_path, stderr_path = output_dir / "stdout.log", output_dir / "stderr.log"
    poll_path = output_dir / "gpu_poll.jsonl"
    read_fd = write_fd = -1
    selector: selectors.BaseSelector | None = None
    process: subprocess.Popen[bytes] | None = None
    started = 0.0
    peak_device = 0
    peak_process = 0
    model_pid_seen = False
    previous_sample: float | None = None
    sample_count = 0
    largest_gap = 0.0
    heartbeat = {"last": 0.0, "count": 0}
    log_sizes = {"stdout": 0, "stderr": 0}
    reason: str | None = None
    owned_sampler: NvmlGpuSampler | None = None
    poll_stream = None
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_once(stdout_path, b"")
        _write_once(stderr_path, b"")
        _write_once(poll_path, b"")
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        child_env = dict(environment)
        child_env["VOICECHAT_DIAG_HEARTBEAT_FD"] = str(write_fd)
        selector = selectors.DefaultSelector()
        poll_stream = poll_path.open("ab")
        started = clock()
        heartbeat["last"] = started
        if gpu_sample is None:
            owned_sampler = NvmlGpuSampler()
            gpu_sample = owned_sampler
        # NVML initialization is preflight, before the cap-monitor interval.
        started = clock()
        heartbeat["last"] = started
        process = subprocess.Popen(command, cwd=output_dir, env=child_env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   pass_fds=(write_fd,), start_new_session=True,
                                   preexec_fn=_limit_file_size)
        os.close(write_fd)
        write_fd = -1
        need(process.stdout is not None and process.stderr is not None,
             "child log pipes unavailable")
        selector.register(process.stdout, selectors.EVENT_READ, ("stdout", stdout_path))
        selector.register(process.stderr, selectors.EVENT_READ, ("stderr", stderr_path))
        selector.register(read_fd, selectors.EVENT_READ, ("heartbeat", None))
        while True:
            now = clock()
            if now - started > MAX_WALL_SECONDS:
                reason = "wall_cap"
                break
            if now - heartbeat["last"] > HEARTBEAT_DEADLINE_SECONDS:
                reason = "heartbeat_timeout"
                break
            sample_begin = clock()
            try:
                sample = gpu_sample(process.pid)
                _valid_sample(sample)
            except BaseException:
                reason = "gpu_telemetry_failure"
                break
            stamped = clock()
            observed_gap = stamped - (previous_sample if previous_sample is not None else started)
            largest_gap = max(largest_gap, observed_gap)
            previous_sample = stamped
            sample_count += 1
            peak_device = max(peak_device, sample.device_mib)
            if sample.process_mib is not None:
                peak_process = max(peak_process, sample.process_mib)
                model_pid_seen = True
            poll_stream.write(_json_bytes({"sample_begin_monotonic": sample_begin,
                                           "sample_end_monotonic": stamped,
                                           "device_mib": sample.device_mib,
                                           "process_mib": sample.process_mib,
                                           "model_pid_present": sample.model_pid_present,
                                           "completed_sample_gap_seconds": observed_gap}))
            poll_stream.flush()
            if largest_gap > MAX_SAMPLE_GAP_SECONDS:
                reason = "gpu_telemetry_gap"
                break
            if sample_count > MAX_POLL_RECORDS:
                reason = "poll_census_cap"
                break
            if peak_device > MAX_GPU_MIB or peak_process > MAX_GPU_MIB:
                reason = "gpu_memory_cap"
                break
            try:
                _files, retained = _inventory(output_dir)
                if retained > MAX_EVIDENCE_BYTES - TERMINAL_RESERVE_BYTES:
                    reason = "evidence_cap"
                    break
                drained = _drain(selector, log_sizes, heartbeat, clock())
                if drained is not None:
                    reason = drained
                    break
            except BaseException:
                reason = "evidence_or_heartbeat_failure"
                break
            if process.poll() is not None and not selector.get_map():
                break
            wait = min(POLL_PERIOD_SECONDS, max(0.0, POLL_PERIOD_SECONDS - (clock() - stamped)))
            selector.select(timeout=wait)
        if reason is not None:
            _kill_group(process)
        returncode = process.wait(timeout=5)
        poll_stream.flush()
        os.fsync(poll_stream.fileno())
        poll_stream.close()
        try:
            truncations = _retain_within_budget(
                output_dir, MAX_EVIDENCE_BYTES - TERMINAL_RESERVE_BYTES)
            _, final_bytes = _inventory(output_dir)
            hashed_inventory = _hash_inventory(output_dir)
            need(len(hashed_inventory) <= 31, "no terminal inventory slot")
        except (ValueError, OSError) as error:
            # Preserve the exact invalid child tree. Do not traverse links,
            # read arbitrary filenames, or delete evidence to make it sealable.
            failure = {"schema": "voicechat_step400_external_parent_failure_v1",
                       "status": "invalid_or_partial",
                       "reason": "unsealed_child_evidence",
                       "inventory_error_type": type(error).__name__,
                       "prior_stop_reason": reason,
                       "returncode": returncode,
                       "sample_count": sample_count,
                       "largest_sample_gap_seconds": largest_gap,
                       "peak_device_mib": peak_device,
                       "peak_process_mib": peak_process,
                       "heartbeat_count": heartbeat["count"],
                       "child_tree_preserved_unmodified": True,
                       "cap_policy_status": "failed_no_64mib_proof",
                       "last_valid_checkpoint_unverified": True,
                       "no_go": True,
                       "output_dir": str(output_dir)}
            failure["external_receipt_status"] = "parent_reserved_inode_verified"
            data = _json_bytes(failure)
            need(len(data) <= TERMINAL_RESERVE_BYTES, "external receipt size exceeded")
            if not _finish_reserved_receipt(external_receipt_path, receipt_fd, data):
                failure["external_receipt_status"] = "path_replaced_or_missing_no_authentic_receipt"
            return failure
        if truncations:
            reason = reason or "evidence_cap_after_stop"
        if not model_pid_seen:
            reason = reason or "model_pid_never_seen"
        if heartbeat["count"] == 0:
            reason = reason or "heartbeat_never_seen"
        if returncode != 0:
            reason = reason or "child_nonzero"
        result = {"schema": "voicechat_step400_parent_candidate_v1",
                  "status": "bounded_child_complete_pending_saved_audit" if reason is None
                            else "invalid_or_partial",
                  "reason": reason, "returncode": returncode,
                  "elapsed_monotonic_seconds": clock() - started,
                  "sample_count": sample_count, "largest_sample_gap_seconds": largest_gap,
                  "peak_device_mib": peak_device, "peak_process_mib": peak_process,
                  "model_pid_seen": model_pid_seen,
                  "heartbeat_count": heartbeat["count"],
                  "retained_evidence_bytes_before_parent_receipt": final_bytes,
                  "truncated_evidence": truncations,
                  "evidence_inventory_before_parent_receipt": hashed_inventory,
                  "output_dir": str(output_dir)}
        if not _reserved_receipt_matches(external_receipt_path, receipt_fd):
            result["status"] = "invalid_or_partial"
            result["reason"] = "external_receipt_reservation_lost"
            result["no_go"] = True
        data = _json_bytes(result)
        need(final_bytes + len(data) <= MAX_EVIDENCE_BYTES,
             "no room for parent terminal receipt")
        _write_once(output_dir / "parent_result.json", data)
        if _reserved_receipt_matches(external_receipt_path, receipt_fd):
            external_receipt_path.unlink()
        return result
    except Exception as error:
        # NVML initialization and process spawn occur after attempt creation.
        # They must leave a bounded no-go receipt rather than an unexplained
        # consumed directory. Do not serialize exception text or child names.
        stop_confirmed = process is None
        if process is not None:
            try:
                _kill_group(process)
                stop_confirmed = True
            except Exception:
                stop_confirmed = False
        failure = {"schema": "voicechat_step400_external_parent_failure_v1",
                   "status": "invalid_or_partial", "reason": "parent_exception",
                   "exception_type": type(error).__name__,
                   "child_group_stop_confirmed": stop_confirmed,
                   "cap_policy_status": "failed_no_64mib_proof",
                   "last_valid_checkpoint_unverified": True, "no_go": True,
                   "output_dir": str(output_dir),
                   "external_receipt_status": "parent_reserved_inode_verified"}
        data = _json_bytes(failure)
        receipt_written = False
        if len(data) <= TERMINAL_RESERVE_BYTES and receipt_fd >= 0:
            try:
                receipt_written = _finish_reserved_receipt(
                    external_receipt_path, receipt_fd, data)
            except OSError:
                receipt_written = False
        if receipt_written:
            failure["external_receipt_status"] = "parent_reserved_inode_verified"
        else:
            failure["external_receipt_status"] = "missing_or_unverified"
        return failure
    finally:
        os.close(receipt_fd)
        if poll_stream is not None and not poll_stream.closed:
            poll_stream.close()
        if selector is not None:
            selector.close()
        try:
            if process is not None:
                # Even if the leader exited, descendants can remain in its session.
                _kill_group(process)
        finally:
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
            if read_fd >= 0:
                try:
                    os.close(read_fd)
                except OSError:
                    pass
            if write_fd >= 0:
                os.close(write_fd)
            if owned_sampler is not None:
                owned_sampler.close()
