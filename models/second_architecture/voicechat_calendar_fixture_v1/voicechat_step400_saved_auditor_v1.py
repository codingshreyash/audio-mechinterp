"""Independent, no-model saved-evidence audit for the VoiceChat step-400 probe.

This does not admit or launch an attempt. A PASS means only that the saved
diagnostic trajectory is internally complete and matches an independently
supplied frozen identity. It cannot establish that the model would finish.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from pathlib import Path
from typing import Any

STEPS = (1, 50, 100, 200, 300, 400)
ALLOWED = frozenset({"parent_result.json", "gpu_poll.jsonl", "stdout.log",
                     "stderr.log", "checkpoints.jsonl", "worker_completion.json"})
MAX_BYTES = 64 * 1024**2
MAX_LEDGER_BYTES = 8 * 1024**2
MAX_RECORD_BYTES = 1 << 20
MAX_GAP = .2
MAX_GPU_MIB = 130_000
MAX_GPU_BYTES = MAX_GPU_MIB * 1024**2
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PATH = re.compile(r"(?:stt|tts)(?:\[entry:[0-9]+\]|\[[0-9]+\]|\.(?:layers|keys|values))*\Z")
TYPE = re.compile(r"[A-Za-z_][\w.]*(?:\.[A-Za-z_][\w]*)?\Z")
DEVICE = re.compile(r"(?:cuda(?::[0-9]+)?|cpu|mps(?::[0-9]+)?)\Z")
PINNED_SOURCES = {
    "nemo/collections/speechlm2/models/duplex_stt_model.py": "415e1e01584bab96d9c059d5b0fe9162adaf19677ef43e565c3afad6e073e74b",
    "nemo/collections/speechlm2/models/nemotron_voicechat.py": "fe11dcdde3c31d7144469dab0c6ec9629ff8f7e4dbf881d8d1db258243a67f80",
    "nemo/collections/speechlm2/inference/utils/offline_voicechat.py": "ebde701baec12a014443eac36262a6af7a7a62b3e052f1c4a387db6aaad8da44",
    "examples/speechlm2/function_calling/template.jinja": "8b4f88e9f0052c0dfc19896b9d6a84821f68f84a1049406908e1f664155edcc9",
    "examples/speechlm2/offline_voicechat_fc_infer.py": "797a80d9cef654550514e045a9388ca689a25dc9bbf70436eb117d249a17644f",
}
PINNED_SAMPLES = {
    "examples/speechlm2/sample_audio/sample_fc.wav": "265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136",
    "examples/speechlm2/function_calling/random_number_response.json": "e36ae164cdc14a0b9bbed7b4be2b37223dc3730b39de1edafb99e73638e2b60f",
}
PINNED_SAMPLE_BYTES = {
    "examples/speechlm2/sample_audio/sample_fc.wav": 384_652,
    "examples/speechlm2/function_calling/random_number_response.json": 82,
}


class AuditError(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AuditError(reason)


def keys(value: Any, required: set[str], reason: str) -> None:
    require(type(value) is dict and set(value) == required, reason)


def nonnegative(value: Any, reason: str) -> int:
    require(type(value) is int and value >= 0, reason)
    return value


def finite(value: Any, reason: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            reason)
    return float(value)


def read_regular(path: Path, limit: int) -> bytes:
    require(not path.is_symlink(), "linked evidence")
    try:
        mode = path.lstat().st_mode
        size = path.stat().st_size
    except OSError as error:
        raise AuditError("missing evidence") from error
    require(stat.S_ISREG(mode) and size <= limit, "unsafe or oversized evidence")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit and len(data) == size, "evidence changed during audit")
    return data


def json_object(data: bytes, reason: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(data, object_pairs_hook=unique_pairs,
                           parse_constant=lambda _x: (_ for _ in ()).throw(
                               ValueError("non-finite JSON constant")))
    except (UnicodeError, ValueError) as error:
        raise AuditError(reason) from error
    require(type(value) is dict, reason)
    return value


def jsonl(data: bytes, max_line: int, reason: str) -> list[dict[str, Any]]:
    require(data.endswith(b"\n"), reason)
    lines = data.splitlines()
    require(all(0 < len(line) <= max_line for line in lines), reason)
    return [json_object(line, reason) for line in lines]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inventory(root: Path) -> tuple[dict[str, bytes], int]:
    require(root.is_dir() and not root.is_symlink(), "missing or linked attempt")
    children = list(root.iterdir())
    require(len(children) == len(ALLOWED) and {p.name for p in children} == ALLOWED,
            "incomplete or extra evidence")
    result: dict[str, bytes] = {}
    total = 0
    for path in children:
        data = read_regular(path, MAX_BYTES)
        result[path.name] = data
        total += len(data)
        require(total <= MAX_BYTES, "retained evidence over cap")
    return result, total


def _check_parent(parent: dict[str, Any], files: dict[str, bytes], total: int) -> None:
    keys(parent, {"schema", "status", "reason", "returncode",
                  "elapsed_monotonic_seconds", "sample_count",
                  "largest_sample_gap_seconds", "peak_device_mib", "peak_process_mib",
                  "model_pid_seen", "heartbeat_count",
                  "retained_evidence_bytes_before_parent_receipt", "truncated_evidence",
                  "evidence_inventory_before_parent_receipt", "output_dir"},
         "parent receipt schema changed")
    require(parent.get("schema") == "voicechat_step400_parent_candidate_v1"
            and parent.get("status") == "bounded_child_complete_pending_saved_audit"
            and parent.get("reason") is None and parent.get("returncode") == 0,
            "parent did not complete cleanly")
    require(parent.get("model_pid_seen") is True
            and nonnegative(parent.get("heartbeat_count"), "missing heartbeat") > 0,
            "model PID or heartbeat absent")
    require(parent.get("truncated_evidence") == [], "evidence truncated")
    require(type(parent.get("output_dir")) is str, "parent output path invalid")
    require(0 < finite(parent.get("elapsed_monotonic_seconds"), "bad elapsed") <= 1800,
            "wall cap failed")
    before = sum(len(data) for name, data in files.items()
                 if name != "parent_result.json")
    require(parent.get("retained_evidence_bytes_before_parent_receipt") == before
            and total <= MAX_BYTES, "parent evidence byte count mismatch")
    recorded = parent.get("evidence_inventory_before_parent_receipt")
    require(type(recorded) is dict and len(recorded) == len(ALLOWED) - 1,
            "parent evidence inventory incomplete")
    actual = {digest(name.encode()): {"bytes": len(data), "sha256": digest(data)}
              for name, data in files.items() if name != "parent_result.json"}
    require(recorded == actual, "parent evidence hashes mismatch")
    require(nonnegative(parent.get("peak_device_mib"), "bad parent device peak") <= MAX_GPU_MIB
            and nonnegative(parent.get("peak_process_mib"), "bad parent process peak") <= MAX_GPU_MIB,
            "parent GPU cap exceeded")


def _check_poll(rows: list[dict[str, Any]], parent: dict[str, Any]) -> tuple[float, float, float]:
    require(len(rows) == nonnegative(parent.get("sample_count"), "bad poll count")
            and len(rows) > 0 and len(rows) <= 36_000, "poll count mismatch")
    starts: list[float] = []
    ends: list[float] = []
    gaps: list[float] = []
    device_peak = process_peak = 0
    seen = False
    for row in rows:
        keys(row, {"sample_begin_monotonic", "sample_end_monotonic", "device_mib",
                   "process_mib", "model_pid_present", "completed_sample_gap_seconds"},
             "poll schema changed")
        start = finite(row["sample_begin_monotonic"], "bad poll start")
        end = finite(row["sample_end_monotonic"], "bad poll end")
        gap = finite(row["completed_sample_gap_seconds"], "bad poll gap")
        require(start <= end and (not ends or start >= ends[-1]),
                "poll timestamps out of order")
        if ends:
            require(math.isclose(gap, end - ends[-1], abs_tol=1e-6),
                    "poll gap mismatch")
        require(gap <= MAX_GAP and (not ends or end - ends[-1] <= MAX_GAP),
                "poll coverage gap")
        dev = nonnegative(row["device_mib"], "bad device sample")
        proc = row["process_mib"]
        require(type(row["model_pid_present"]) is bool
                and row["model_pid_present"] == (proc is not None),
                "poll process presence mismatch")
        if proc is not None:
            proc = nonnegative(proc, "bad process sample")
            require(proc <= dev, "poll process exceeds device")
            process_peak = max(process_peak, proc)
            seen = True
        device_peak = max(device_peak, dev)
        starts.append(start)
        ends.append(end)
        gaps.append(gap)
    require(seen and device_peak == parent.get("peak_device_mib")
            and process_peak == parent.get("peak_process_mib"),
            "parent peak or PID status mismatch")
    require(math.isclose(max(gaps), finite(parent.get("largest_sample_gap_seconds"),
                                           "bad parent gap"), abs_tol=1e-6),
            "parent largest poll gap mismatch")
    require(finite(parent["elapsed_monotonic_seconds"], "bad parent elapsed")
            >= ends[-1] - starts[0], "parent elapsed shorter than GPU polling span")
    return starts[0], ends[-1], max(gaps)


def _check_carried(value: Any) -> dict[str, Any]:
    keys(value, {"complete", "unknown", "unknown_count", "node_count",
                 "tensor_rows", "families", "shared_storage_bytes",
                 "global_unique_storage_bytes", "global_unique_cuda_storage_bytes"},
         "carried schema changed")
    require(type(value) is dict and value.get("complete") is True
            and value.get("unknown") == [] and value.get("unknown_count") == 0,
            "carried inventory incomplete")
    rows = value.get("tensor_rows")
    require(type(rows) is list and len(rows) <= 4096, "invalid tensor census")
    require(nonnegative(value.get("node_count"), "bad node count") <= 8192,
            "node census exceeds bound")
    families = value.get("families")
    require(type(families) is dict and set(families) == {"stt", "tts"},
            "missing carried family")
    counts = {"stt": 0, "tts": 0}
    logical = {"stt": 0, "tts": 0}
    unique: dict[str, int] = {}
    for row in rows:
        keys(row, {"family", "path", "shape", "dtype", "device",
                   "logical_bytes", "storage_bytes"}, "tensor row schema changed")
        family = row["family"]
        require(family in counts and type(row["path"]) is str
                and len(row["path"]) <= 256 and PATH.fullmatch(row["path"]) is not None,
                "tensor path may contain content")
        require(type(row["shape"]) is list and len(row["shape"]) <= 32
                and all(type(x) is int and 0 <= x <= 1_000_000_000 for x in row["shape"]),
                "tensor shape invalid")
        require(type(row["dtype"]) is str and row["dtype"].startswith("torch.")
                and TYPE.fullmatch(row["dtype"]) is not None
                and type(row["device"]) is str
                and DEVICE.fullmatch(row["device"]) is not None,
                "tensor type/device invalid")
        logical[family] += nonnegative(row["logical_bytes"], "bad logical bytes")
        nonnegative(row["storage_bytes"], "bad storage bytes")
        counts[family] += 1
    for family in counts:
        info = families[family]
        keys(info, {"tensor_count", "logical_bytes", "unique_storage_bytes",
                    "unique_cuda_storage_bytes"}, "family schema changed")
        require(nonnegative(info["tensor_count"], "bad tensor count") == counts[family]
                and nonnegative(info["logical_bytes"], "bad family logical") == logical[family],
                "carried row totals mismatch")
        unique[family] = nonnegative(info["unique_storage_bytes"], "bad family unique")
        require(nonnegative(info["unique_cuda_storage_bytes"], "bad family CUDA")
                <= unique[family], "family CUDA exceeds all storage")
    shared = nonnegative(value.get("shared_storage_bytes"), "bad shared storage")
    global_bytes = nonnegative(value.get("global_unique_storage_bytes"), "bad union")
    require(shared <= min(unique.values())
            and global_bytes == unique["stt"] + unique["tts"] - shared,
            "shared-storage union arithmetic failed")
    require(nonnegative(value.get("global_unique_cuda_storage_bytes"), "bad CUDA union")
            <= global_bytes, "CUDA union exceeds all storage")
    return {"stt_unique_bytes": unique["stt"], "tts_unique_bytes": unique["tts"],
            "shared_bytes": shared, "global_unique_bytes": global_bytes}


def _check_cuda(cuda: Any) -> int:
    keys(cuda, {"allocated_bytes", "reserved_bytes", "max_allocated_bytes",
                "max_reserved_bytes"}, "CUDA counter schema changed")
    allocated, reserved, peak_allocated, peak_reserved = (
        nonnegative(cuda[k], "bad CUDA counter") for k in
        ("allocated_bytes", "reserved_bytes", "max_allocated_bytes",
         "max_reserved_bytes"))
    require(allocated <= reserved and allocated <= peak_allocated
            and reserved <= peak_reserved and peak_allocated <= peak_reserved,
            "CUDA counters inconsistent")
    require(peak_reserved <= MAX_GPU_BYTES,
            "observed CUDA peak exceeds GPU cap")
    return allocated


def _check_checkpoint_rows(rows: list[dict[str, Any]], first_poll: float,
                           last_poll: float) -> list[dict[str, Any]]:
    require(len(rows) == len(STEPS)
            and all(type(row.get("t")) is int and row["t"] == expected
                    for row, expected in zip(rows, STEPS)),
            "checkpoint order incomplete")
    summaries: list[dict[str, Any]] = []
    previous_nvml: float | None = None
    previous_device: str | None = None
    for index, row in enumerate(rows):
        expected = {"schema", "t", "T", "interval_start_monotonic",
                    "snapshot_monotonic", "nvml_lag_seconds", "device_id",
                    "cuda", "nvml", "carried"}
        if index == 0:
            expected.add("pre_step_1_baseline")
        keys(row, expected, "checkpoint schema changed")
        require(row["schema"] == "voicechat_step400_checkpoint_v1"
                and type(row["T"]) is int and row["T"] == 859,
                "checkpoint identity changed")
        require(type(row["device_id"]) is str and
                re.fullmatch(r"(?:[0-9]+|cuda:[0-9]+)", row["device_id"]) is not None,
                "checkpoint device invalid")
        if previous_device is not None:
            require(row["device_id"] == previous_device, "checkpoint device changed")
        previous_device = row["device_id"]
        start = finite(row["interval_start_monotonic"], "bad interval start")
        snap = finite(row["snapshot_monotonic"], "bad checkpoint time")
        lag = finite(row["nvml_lag_seconds"], "bad checkpoint NVML lag")
        nvml = row["nvml"]
        keys(nvml, {"used_bytes", "model_pid_present", "sample_monotonic",
                    "query_seconds"}, "checkpoint NVML schema changed")
        sample = finite(nvml["sample_monotonic"], "bad checkpoint NVML time")
        require(nvml["model_pid_present"] is True
                and nonnegative(nvml["used_bytes"], "bad checkpoint process bytes")
                <= MAX_GPU_MIB * 1024**2, "checkpoint PID absent or GPU cap exceeded")
        query = finite(nvml["query_seconds"], "bad checkpoint query time")
        require(first_poll <= start < snap <= sample <= last_poll
                and math.isclose(lag, sample - snap, abs_tol=1e-6)
                and query <= lag + 1e-6,
                "checkpoint interval or telemetry outside parent coverage")
        if previous_nvml is not None:
            require(previous_nvml <= start, "checkpoint intervals overlap")
        if index == 0:
            baseline = row["pre_step_1_baseline"]
            keys(baseline, {"snapshot_monotonic", "cuda", "nvml", "T",
                            "reset_monotonic"}, "baseline schema changed")
            base_snap = finite(baseline["snapshot_monotonic"], "bad baseline time")
            base_reset = finite(baseline["reset_monotonic"], "bad reset time")
            base_nvml = baseline["nvml"]
            keys(base_nvml, {"used_bytes", "model_pid_present", "sample_monotonic",
                             "query_seconds"}, "baseline NVML schema changed")
            require(type(baseline["T"]) is int and baseline["T"] == 859
                    and first_poll <= base_snap
                    <= finite(base_nvml["sample_monotonic"], "bad baseline NVML time")
                    <= base_reset == start and base_reset < snap
                    and base_nvml["model_pid_present"] is True,
                    "pre-step-1 baseline/reset absent or out of order")
            nonnegative(base_nvml["used_bytes"], "bad baseline NVML bytes")
            require(base_nvml["used_bytes"] <= MAX_GPU_MIB * 1024**2,
                    "baseline GPU cap exceeded")
            require(finite(base_nvml["query_seconds"], "bad baseline query")
                    <= base_nvml["sample_monotonic"] - base_snap + 1e-6,
                    "baseline NVML query exceeds elapsed time")
            _check_cuda(baseline["cuda"])
        allocated = _check_cuda(row["cuda"])
        carried = _check_carried(row["carried"])
        require(row["carried"]["global_unique_cuda_storage_bytes"] <= allocated,
                "enumerated CUDA storage exceeds PyTorch allocated bytes")
        summaries.append({"t": row["t"], "allocated_bytes": row["cuda"]["allocated_bytes"],
                          "max_allocated_bytes": row["cuda"]["max_allocated_bytes"],
                          "reserved_bytes": row["cuda"]["reserved_bytes"],
                          "checkpoint_process_bytes": nvml["used_bytes"],
                          "nvml_lag_seconds": lag,
                          **carried})
        previous_nvml = sample
    return summaries


def _check_worker(worker: dict[str, Any], ledger_data: bytes,
                  expected_identity: dict[str, Any]) -> None:
    keys(worker, {"schema", "status", "identity", "core", "no_second_pass"},
         "worker schema changed")
    require(worker["schema"] == "voicechat_step400_worker_completion_v1"
            and worker["status"] == "diagnostic_early_stop_complete"
            and worker["no_second_pass"] is True,
            "worker did not stop at step 400")
    identity = worker["identity"]
    identity_fields = {"nemo_revision", "voicechat_revision", "nano_revision",
                       "source_sha256", "sample_sha256", "sample_bytes",
                       "config_sha256", "weight_sha256", "weight_bytes",
                       "template_sha256", "system_message_sha256", "tools_sha256",
                       "prompt_sha256", "sample_id"}
    keys(identity, identity_fields, "worker identity schema changed")
    require(identity["nemo_revision"] == "097dfe9e2f55baf653b83035868bdc89849f1b47"
            and identity["voicechat_revision"] == "443794ea956ef0065f001967ffd00e77f519cb39"
            and identity["nano_revision"] == "6533e8de2c68e4536bf7c411d7a3ce5734111476"
            and identity["source_sha256"] == PINNED_SOURCES
            and identity["sample_sha256"] == PINNED_SAMPLES
            and identity["sample_bytes"] == PINNED_SAMPLE_BYTES
            and identity["config_sha256"] == "2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5"
            and identity["weight_sha256"] == "d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b"
            and identity["weight_bytes"] == 44_382_749_892
            and identity["template_sha256"] == PINNED_SOURCES[
                "examples/speechlm2/function_calling/template.jinja"]
            and identity["sample_id"] == "sample_fc"
            and all(type(identity[key]) is str and HEX64.fullmatch(identity[key])
                    for key in ("system_message_sha256", "tools_sha256", "prompt_sha256")),
            "pinned worker identity invalid")
    require(type(expected_identity) is dict and expected_identity
            and identity == expected_identity,
            "worker identity differs from independent freeze")
    core = worker["core"]
    keys(core, {"schema", "status", "inventory_complete", "checkpoint_steps",
                "ledger", "no_model_behavior_claim"}, "core completion schema changed")
    require(core["schema"] == "voicechat_step400_core_completion_v1"
            and core["status"] == "diagnostic_early_stop_complete"
            and core["inventory_complete"] is True
            and core["no_model_behavior_claim"] is True
            and core["checkpoint_steps"] == list(STEPS),
            "core diagnostic incomplete")
    keys(core["ledger"], {"path", "sha256", "bytes", "checkpoints"},
         "core ledger receipt changed")
    require(core["ledger"] == {"path": "checkpoints.jsonl", "sha256": digest(ledger_data),
                              "bytes": len(ledger_data), "checkpoints": len(STEPS)},
            "core ledger hash/count mismatch")


def audit_saved_attempt(root: Path, *, expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded PASS/NO_GO receipt; never load NeMo or model weights."""
    try:
        require(not (root.parent / f"{root.name}.parent_failure.json").exists(),
                "external parent failure receipt exists")
        files, total = _inventory(root)
        parent = json_object(files["parent_result.json"], "bad parent JSON")
        worker = json_object(files["worker_completion.json"], "bad worker JSON")
        _check_parent(parent, files, total)
        require(Path(parent["output_dir"]).resolve() == root.resolve(),
                "parent attempt path mismatch")
        first, last, max_gap = _check_poll(
            jsonl(files["gpu_poll.jsonl"], 1024, "bad GPU poll JSONL"), parent)
        ledger = files["checkpoints.jsonl"]
        require(len(ledger) <= MAX_LEDGER_BYTES, "checkpoint ledger over cap")
        rows = jsonl(ledger, MAX_RECORD_BYTES, "bad checkpoint JSONL")
        _check_worker(worker, ledger, expected_identity)
        summaries = _check_checkpoint_rows(rows, first, last)
        return {"schema": "voicechat_step400_saved_audit_v1", "status": "PASS",
                "diagnostic_only": True, "checkpoint_count": len(STEPS),
                "max_parent_poll_gap_seconds": max_gap,
                "sampled_peak_device_mib": parent["peak_device_mib"],
                "sampled_peak_process_mib": parent["peak_process_mib"],
                "retained_evidence_bytes": total, "checkpoints": summaries,
                "no_model_behavior_claim": True}
    except (AuditError, OSError, TypeError, KeyError) as error:
        return {"schema": "voicechat_step400_saved_audit_v1", "status": "NO_GO",
                "reason": str(error) if isinstance(error, AuditError) else "malformed evidence",
                "diagnostic_only": True, "no_model_behavior_claim": True}
