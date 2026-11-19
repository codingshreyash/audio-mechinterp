"""Source-only bounded parent for a future VoiceChat step-400 worker.

No CLI, NeMo import, model load, or executable GO lives here. A separately
reviewed launcher must bind the exact command, environment, and host freeze.
"""
from __future__ import annotations

import json
import os
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


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _inventory(root: Path) -> tuple[dict[str, dict], int]:
    rows: dict[str, dict] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        need(not path.is_symlink(), "evidence symlink rejected")
        if path.is_dir():
            continue
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


def _sanitize_evidence(root: Path) -> dict[str, object]:
    """After the child stops, remove unsafe entries without following links.

    The step-400 worker's evidence is flat. Child-created directories and
    special entries are out of scope and cannot be opened by the sealer.
    """
    import hashlib

    removed: list[dict[str, str]] = []
    removed_count = 0

    def note(name: str, kind: str) -> None:
        nonlocal removed_count
        removed_count += 1
        if len(removed) < 32:
            removed.append({"name_prefix": name[:128], "name_sha256":
                            hashlib.sha256(name.encode("utf-8", "surrogateescape")).hexdigest(),
                            "kind": kind})

    regular: list[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                os.unlink(entry.path)
                note(entry.name, "symlink_removed")
            elif stat.S_ISDIR(mode):
                # rmtree does not traverse directory symlinks; the child is
                # already stopped, so this tree cannot be mutated by it.
                shutil.rmtree(entry.path)
                note(entry.name, "directory_removed")
            elif not stat.S_ISREG(mode):
                os.unlink(entry.path)
                note(entry.name, "special_entry_removed")
            elif len(entry.name) > 128 or entry.name == "parent_result.json":
                os.unlink(entry.path)
                note(entry.name, "reserved_or_long_name_removed")
            else:
                regular.append(entry.name)
    # Leave one inventory slot for the parent's terminal receipt.
    for name in sorted(regular, key=_evidence_priority)[31:]:
        os.unlink(root / name)
        note(name, "excess_file_removed")
    return {"removed_count": removed_count, "removed_entries_first_32": removed}


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
            truncations.append({"path": name, "original_bytes": before,
                                "retained_bytes": allowed})
        retained += allowed
    need(retained <= budget, "failed to retain within evidence budget")
    return truncations


def _hash_inventory(root: Path) -> dict[str, dict]:
    import hashlib

    rows, _ = _inventory(root)
    for name in rows:
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        rows[name]["sha256"] = digest.hexdigest()
    return rows


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
         and type(environment) is dict and not output_dir.exists(),
         "invalid or consumed diagnostic attempt")
    need(staged_checkpoint_dir.is_dir() and not staged_checkpoint_dir.is_symlink(),
         "staged checkpoint directory missing or linked")
    need(output_dir.parent.is_dir() and not output_dir.parent.is_symlink()
         and output_dir.parent.stat().st_dev == staged_checkpoint_dir.stat().st_dev,
         "diagnostic output must share the staged checkpoint filesystem")
    need(disk_free_bytes(staged_checkpoint_dir) >= MIN_POST_STAGING_FREE_BYTES,
         "less than 30 GiB free after staging")
    need(not any(key.endswith("KEY_HEX") for key in environment),
         "review/controller secret in child environment")
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = output_dir / "stdout.log", output_dir / "stderr.log"
    _write_once(stdout_path, b"")
    _write_once(stderr_path, b"")
    poll_path = output_dir / "gpu_poll.jsonl"
    _write_once(poll_path, b"")
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    child_env = dict(environment)
    child_env["VOICECHAT_DIAG_HEARTBEAT_FD"] = str(write_fd)
    selector = selectors.DefaultSelector()
    process: subprocess.Popen[bytes] | None = None
    started = clock()
    peak_device = 0
    peak_process = 0
    model_pid_seen = False
    previous_sample: float | None = None
    sample_count = 0
    largest_gap = 0.0
    heartbeat = {"last": started, "count": 0}
    log_sizes = {"stdout": 0, "stderr": 0}
    reason: str | None = None
    owned_sampler: NvmlGpuSampler | None = None
    poll_stream = poll_path.open("ab")
    try:
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
        sanitization = _sanitize_evidence(output_dir)
        truncations = _retain_within_budget(
            output_dir, MAX_EVIDENCE_BYTES - TERMINAL_RESERVE_BYTES)
        _, final_bytes = _inventory(output_dir)
        if sanitization["removed_count"]:
            reason = reason or "unexpected_evidence_removed"
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
                  "sanitization": sanitization,
                  "truncated_evidence": truncations,
                  "evidence_inventory_before_parent_receipt": _hash_inventory(output_dir),
                  "output_dir": str(output_dir)}
        data = _json_bytes(result)
        need(final_bytes + len(data) <= MAX_EVIDENCE_BYTES,
             "no room for parent terminal receipt")
        _write_once(output_dir / "parent_result.json", data)
        return result
    finally:
        if not poll_stream.closed:
            poll_stream.close()
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
