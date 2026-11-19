"""Source-only VoiceChat step-400 memory probe core.

This module imports no NeMo code and has no executable model/instance path.
The reviewed V1/V2 diagnostic still needs a separate launcher, parent watchdog,
host freeze, independent executable review, and one-shot GO. This core only
instruments an already constructed model when a future approved caller uses it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

CHECKPOINTS = (1, 50, 100, 200, 300, 400)
EXPECTED_T = 859
MAX_DEPTH = 12
MAX_NODES = 8192
MAX_TENSOR_ROWS = 4096
MAX_RECORD_BYTES = 1 << 20
MAX_LEDGER_BYTES = 8 << 20
TTS_KWARGS = {
    "current_subword_id", "prev_subword_id", "current_subword_mask",
    "prev_audio_tokens", "past_key_values", "guidance_enabled",
    "generation_config", "ignore_eos_flag_stop",
}
CACHE_TYPES = {
    ("transformers.cache_utils", "DynamicCache"),
    ("transformers.cache_utils", "DynamicLayer"),
    ("transformers.cache_utils", "DynamicSlidingWindowLayer"),
}


class DiagnosticStop(BaseException):
    """Planned exit after the complete, fsynced step-400 checkpoint."""


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


class CheckpointLedger:
    """One new append-only checkpoint file, with a small independent size cap."""

    def __init__(self, path: Path) -> None:
        need(not path.exists() and not path.is_symlink(), "checkpoint path consumed")
        self.path = path
        self.handle = path.open("xb")
        self.bytes_written = 0
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        need(self.count < len(CHECKPOINTS), "too many checkpoints")
        need(record.get("t") == CHECKPOINTS[self.count], "checkpoint order changed")
        data = _canonical(record)
        need(len(data) <= MAX_RECORD_BYTES
             and self.bytes_written + len(data) <= MAX_LEDGER_BYTES,
             "checkpoint evidence cap exceeded")
        self.handle.write(data)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.bytes_written += len(data)
        self.count += 1

    def close(self) -> dict[str, Any]:
        if not self.handle.closed:
            self.handle.close()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return {"path": self.path.name, "sha256": digest,
                "bytes": self.bytes_written, "checkpoints": self.count}


def nvml_process_sample(pid: int) -> dict[str, Any]:
    """Checkpoint telemetry only; a future parent must poll independently."""
    from .public_sample_smoke_v1 import _process_tree

    started = time.monotonic()
    completed = subprocess.run(
        ("nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"), capture_output=True, text=True,
        timeout=5, check=True)
    need(len(completed.stdout) <= 65536, "NVML output cap exceeded")
    relevant = _process_tree(pid)
    need(pid in relevant, "NVML process tree missing model PID")
    total_mib = 0
    present = False
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        need(len(parts) == 2 and all(part.isdecimal() for part in parts),
             "NVML process record malformed")
        if int(parts[0]) in relevant:
            total_mib += int(parts[1])
            present |= int(parts[0]) == pid
    need(present, "NVML did not include model PID")
    ended = time.monotonic()
    return {"used_bytes": total_mib * 1024**2, "model_pid_present": True,
            "sample_monotonic": ended, "query_seconds": ended - started}


def _tensor_storage(tensor: Any) -> tuple[tuple[Any, ...], int]:
    """Return in-process identity and capacity; the identity is never saved."""
    storage = tensor.untyped_storage()
    nbytes = int(storage.nbytes())
    need(nbytes >= 0, "negative storage size")
    pointer = int(storage.data_ptr())
    # Zero-capacity storage contributes no bytes; avoid treating all null
    # pointers as one positive-capacity allocation.
    need(nbytes == 0 or pointer > 0, "unbacked storage")
    device = str(tensor.device)
    return (device, pointer, nbytes), nbytes


def inventory_carried_state(stt_state: Any, tts_cache: Any) -> dict[str, Any]:
    """Bounded, metadata-only walk with a cross-family storage union."""
    import torch

    rows: list[dict[str, Any]] = []
    unknown: list[dict[str, str]] = []
    family_storages: dict[str, dict[tuple[Any, ...], int]] = {"stt": {}, "tts": {}}
    family_logical = {"stt": 0, "tts": 0}
    family_tensors = {"stt": 0, "tts": 0}
    seen_containers: set[int] = set()
    nodes = 0

    def unknown_node(path: str, value: Any) -> None:
        unknown.append({"path": path[:256],
                        "type": f"{type(value).__module__}.{type(value).__name__}"[:160]})

    def walk(value: Any, family: str, path: str, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            unknown_node(path, value)
            return
        if torch.is_tensor(value):
            if len(rows) >= MAX_TENSOR_ROWS:
                unknown_node(path, value)
                return
            try:
                storage_id, capacity = _tensor_storage(value)
                shape = [int(dim) for dim in value.shape]
                logical = int(value.numel()) * int(value.element_size())
                need(logical >= 0, "negative logical tensor bytes")
            except (RuntimeError, TypeError, ValueError, NotImplementedError):
                unknown_node(path, value)
                return
            family_storages[family][storage_id] = capacity
            family_logical[family] += logical
            family_tensors[family] += 1
            rows.append({"family": family, "path": path[:256], "shape": shape,
                         "dtype": str(value.dtype), "device": str(value.device),
                         "logical_bytes": logical, "storage_bytes": capacity})
            return
        if value is None or type(value) in (bool, int, float, str, bytes):
            return
        if id(value) in seen_containers:
            # A repeated reference is already inventoried; cross-family
            # aliases are still visited because each family has a fresh seen set.
            return
        seen_containers.add(id(value))
        if type(value) is dict:
            if len(value) > MAX_NODES:
                unknown_node(path, value)
                return
            for key in sorted(value, key=lambda item: str(item)):
                if nodes >= MAX_NODES:
                    unknown_node(path, value)
                    break
                if not isinstance(key, (str, int)):
                    unknown_node(path, value)
                    continue
                walk(value[key], family, f"{path}.{key}", depth + 1)
        elif type(value) in (list, tuple):
            if len(value) > MAX_NODES:
                unknown_node(path, value)
                return
            for index, item in enumerate(value):
                if nodes >= MAX_NODES:
                    unknown_node(path, value)
                    break
                walk(item, family, f"{path}[{index}]", depth + 1)
        elif type(value) in (set, frozenset):
            if len(value) > MAX_NODES:
                unknown_node(path, value)
                return
            if any(type(item) not in (bool, int, float, str, bytes)
                   for item in value):
                unknown_node(path, value)
        else:
            identity = (type(value).__module__, type(value).__name__)
            if identity not in CACHE_TYPES:
                unknown_node(path, value)
            elif identity[1] == "DynamicCache":
                walk(value.layers, family, f"{path}.layers", depth + 1)
            else:
                walk(value.keys, family, f"{path}.keys", depth + 1)
                walk(value.values, family, f"{path}.values", depth + 1)

    for family, value in (("stt", stt_state), ("tts", tts_cache)):
        seen_containers = set()
        walk(value, family, family, 0)
    a, b = family_storages["stt"], family_storages["tts"]
    shared = set(a) & set(b)
    union = {**a, **b}
    need(all(a[key] == b[key] for key in shared), "shared storage size mismatch")
    per_family = {
        family: {"tensor_count": family_tensors[family],
                 "logical_bytes": family_logical[family],
                 "unique_storage_bytes": sum(items.values()),
                 "unique_cuda_storage_bytes": sum(size for (device, _ptr, _n), size
                                                   in items.items() if device.startswith("cuda"))}
        for family, items in family_storages.items()
    }
    return {"complete": not unknown, "unknown": unknown[:128],
            "unknown_count": len(unknown), "node_count": nodes,
            "tensor_rows": rows, "families": per_family,
            "shared_storage_bytes": sum(a[key] for key in shared),
            "global_unique_storage_bytes": sum(union.values()),
            "global_unique_cuda_storage_bytes": sum(
                size for (device, _ptr, _n), size in union.items()
                if device.startswith("cuda"))}


class Step400Probe:
    """Attach only to this model instance, then stop after step 400."""

    def __init__(self, model: Any, ledger: CheckpointLedger, cuda: Any,
                 nvml: Callable[[int], dict[str, Any]], *, device: Any = 0,
                 pid: int | None = None, clock: Callable[[], float] = time.monotonic,
                 inventory: Callable[[Any, Any], dict[str, Any]] = inventory_carried_state):
        self.model, self.ledger, self.cuda = model, ledger, cuda
        self.nvml, self.device, self.pid = nvml, device, os.getpid() if pid is None else pid
        self.clock, self.inventory = clock, inventory
        self.next_step = 1
        self.pending_state: Any = None
        self.pending_step: int | None = None
        self.baseline: dict[str, Any] | None = None
        self.interval_start: float | None = None
        self.installed = False
        self.finished = False
        self.inventory_complete = True

    def _cuda_record(self) -> dict[str, int]:
        self.cuda.synchronize(self.device)
        values = {"allocated_bytes": int(self.cuda.memory_allocated(self.device)),
                  "reserved_bytes": int(self.cuda.memory_reserved(self.device)),
                  "max_allocated_bytes": int(self.cuda.max_memory_allocated(self.device)),
                  "max_reserved_bytes": int(self.cuda.max_memory_reserved(self.device))}
        need(all(value >= 0 for value in values.values()), "negative CUDA counter")
        need(values["reserved_bytes"] >= values["allocated_bytes"]
             and values["max_reserved_bytes"] >= values["max_allocated_bytes"],
             "CUDA allocator counters inconsistent")
        return values

    def _nvml_record(self) -> dict[str, Any]:
        result = self.nvml(self.pid)
        need(type(result) is dict and result.get("model_pid_present") is True
             and type(result.get("used_bytes")) is int and result["used_bytes"] >= 0
             and isinstance(result.get("sample_monotonic"), (int, float))
             and math.isfinite(result["sample_monotonic"]),
             "NVML telemetry missing model process")
        return result

    @contextmanager
    def install(self) -> Iterator["Step400Probe"]:
        need(not self.installed and not self.finished, "probe reused")
        stt, tts = self.model.stt_model, self.model.tts_model
        original_stt, original_tts = stt._step_inference, tts.infer_codes_one_step
        owned = {(obj, name): (name in vars(obj), vars(obj).get(name))
                 for obj, name in ((stt, "_step_inference"),
                                   (tts, "infer_codes_one_step"))}
        self.installed = True

        def step(t: int, state: Any, *args: Any, **kwargs: Any) -> Any:
            need(type(t) is int and t == self.next_step and self.pending_step is None
                 and type(state) is dict and state.get("T") == EXPECTED_T,
                 "STT step order or T changed")
            if t == 1:
                before = self._cuda_record()
                baseline_time = self.clock()
                nvml = self._nvml_record()
                need(nvml["sample_monotonic"] >= baseline_time,
                     "NVML baseline timestamp precedes CUDA timestamp")
                self.baseline = {"snapshot_monotonic": baseline_time,
                                 "cuda": before, "nvml": nvml, "T": EXPECTED_T}
                self.cuda.reset_peak_memory_stats(self.device)
                self.interval_start = self.clock()
                self.baseline["reset_monotonic"] = self.interval_start
            self.pending_state, self.pending_step = state, t
            try:
                return original_stt(t, state, *args, **kwargs)
            except BaseException:
                self.pending_state = None
                self.pending_step = None
                raise

        def tts_step(*args: Any, **kwargs: Any) -> Any:
            need(not args and set(kwargs) == TTS_KWARGS
                 and kwargs["ignore_eos_flag_stop"] is True
                 and self.pending_step == self.next_step
                 and self.pending_state is not None,
                 "TTS call order or arguments changed")
            t, state = self.pending_step, self.pending_state
            try:
                result = original_tts(*args, **kwargs)
                need(type(result) is tuple and len(result) == 2,
                     "TTS cache return structure changed")
                if t in CHECKPOINTS:
                    cuda = self._cuda_record()
                    now = self.clock()
                    nvml = self._nvml_record()
                    need(nvml["sample_monotonic"] >= now,
                         "NVML sample timestamp precedes CUDA timestamp")
                    carried = self.inventory(state, result[1])
                    need(type(carried) is dict and type(carried.get("complete")) is bool,
                         "carried-state inventory malformed")
                    self.inventory_complete &= carried["complete"]
                    record = {"schema": "voicechat_step400_checkpoint_v1",
                              "t": t, "T": EXPECTED_T, "interval_start_monotonic": self.interval_start,
                              "snapshot_monotonic": now,
                              "nvml_lag_seconds": nvml["sample_monotonic"] - now,
                              "device_id": str(self.device), "cuda": cuda, "nvml": nvml,
                              "carried": carried}
                    if t == 1:
                        need(self.baseline is not None, "first interval baseline absent")
                        record["pre_step_1_baseline"] = self.baseline
                    self.ledger.write(record)
                    if t != CHECKPOINTS[-1]:
                        self.cuda.reset_peak_memory_stats(self.device)
                        self.interval_start = self.clock()
                self.next_step += 1
            finally:
                self.pending_state = None
                self.pending_step = None
            if t == CHECKPOINTS[-1]:
                self.finished = True
                raise DiagnosticStop("diagnostic_early_stop_complete")
            return result

        try:
            stt._step_inference = step
            tts.infer_codes_one_step = tts_step
            yield self
        finally:
            for (obj, name), (existed, prior) in owned.items():
                if existed:
                    setattr(obj, name, prior)
                else:
                    delattr(obj, name)
            self.pending_state = None
            self.pending_step = None
            self.installed = False

    def completion(self) -> dict[str, Any]:
        need(self.finished and self.next_step == 401
             and self.ledger.count == len(CHECKPOINTS),
             "diagnostic did not reach planned stop")
        return {"schema": "voicechat_step400_core_completion_v1",
                "status": ("diagnostic_early_stop_complete" if self.inventory_complete
                           else "diagnostic_early_stop_incomplete_inventory"),
                "inventory_complete": self.inventory_complete,
                "checkpoint_steps": list(CHECKPOINTS),
                "ledger": self.ledger.close(),
                "no_model_behavior_claim": True}


def run_pinned_pass1_only(*, model: Any, input_signal: Any,
                          input_signal_lens: Any, prompt_tokens: Any,
                          prompt_token_lens: Any, offline_inference: Callable[..., Any],
                          ledger: CheckpointLedger, cuda: Any,
                          nvml: Callable[[int], dict[str, Any]],
                          device: Any = 0, pid: int | None = None) -> dict[str, Any]:
    """Use the exact first-pass arguments from the pinned FC helper.

    The future launcher must inject the pinned official `run_offline_inference`
    function and preserve the official helper's input preparation. This core
    cannot perform or authorize that source/environment verification.
    """
    probe = Step400Probe(model, ledger, cuda, nvml, device=device, pid=pid)
    with probe.install():
        try:
            offline_inference(
                model,
                input_signal=input_signal,
                input_signal_lens=input_signal_lens,
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_token_lens,
                decode_audio=False,
            )
        except DiagnosticStop:
            pass
        else:
            raise ValueError("pass 1 returned before the diagnostic step-400 stop")
    return probe.completion()
