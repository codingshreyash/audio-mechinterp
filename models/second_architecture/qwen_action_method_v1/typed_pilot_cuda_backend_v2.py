"""Gated CUDA typed-only Qwen pilot. Importing this file loads no weights.

The fixed GO is deliberately absent until an independent executable review.
Audio, development, and locked cases are not accepted by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import select
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import endpoint_v2, readout_v2, runner_contract_v1, typed_pilot_cuda_source_audit_v2
from . import typed_pilot_cuda_admission_v2 as admission
from .endpoint import SPEC_PATH, load_spec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOAD_ATTEMPT = HERE / "runs/typed_pilot_cuda_v2/load_001"
PILOT_ATTEMPT = HERE / "runs/typed_pilot_cuda_v2/pilot_001"
LOAD_GO = HERE / "TYPED_PILOT_CUDA_LOAD_GO_V2.json"
PILOT_GO = HERE / "TYPED_PILOT_CUDA_PILOT_GO_V2.json"
LOAD_AUDIT_PATH = LOAD_ATTEMPT.parent / "load_001_audit.json"
REVIEW = HERE / "TYPED_PILOT_CUDA_INDEPENDENT_REVIEW_V2.md"
QA_AMENDMENT = HERE / "TYPED_PILOT_CUDA_QA_EXCEPTION_V2.md"
PARENT_AMENDMENT = HERE / "TYPED_PILOT_CUDA_PROTOCOL_V2.md"
SOURCES = (
    "models/second_architecture/qwen_action_method_v1/typed_pilot_cuda_backend_v2.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_cuda_parent_v2.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_cuda_audit_v2.py",
    "models/second_architecture/qwen_action_method_v1/test_typed_pilot_cuda_v2.py",
    "models/second_architecture/qwen_action_method_v1/runner_contract_v1.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_cuda_source_audit_v2.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_cuda_admission_v2.py",
    "models/second_architecture/qwen_action_method_v1/TYPED_PILOT_CUDA_HF_SNAPSHOT_MANIFEST_V2.json",
    "models/second_architecture/qwen_action_method_v1/endpoint.py",
    "models/second_architecture/qwen_action_method_v1/endpoint_v2.py",
    "models/second_architecture/qwen_action_method_v1/readout_v2.py",
    "models/second_architecture/qwen_action_method_v1/frozen_spec_v1.json",
    "models/second_architecture/qwen_action_method_v1/READOUT_FREEZE_V2.json",
    "models/second_architecture/qwen_action_method_v1/PROTOCOL_AMENDMENT_V2.md",
    "models/second_architecture/qwen_action_method_v1/PROTOCOL_AMENDMENT_V3.md",
    "models/second_architecture/qwen_action_method_v1/TYPED_PILOT_BACKEND_HANDOFF_V3.md",
    "models/second_architecture/qwen_action_method_v1/TYPED_PILOT_CUDA_QA_EXCEPTION_V2.md",
    "models/second_architecture/qwen_action_method_v1/TYPED_PILOT_CUDA_PROTOCOL_V2.md",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_audit_v3.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_backend_v3.py",
    "models/second_architecture/qwen_action_method_v1/source_audit_v2.py",
    "models/qwen_omni/adapter.py",
    "models/qwen_omni/pins.json",
    "models/second_architecture/nonspeech_probe_v1/causal_parent_v2.py",
)
PACKAGES = {"torch": "2.11.0+cu128", "transformers": "5.24.0.dev0",
            "accelerate": "1.16.0", "numpy": "2.5.5", "psutil": "7.2.2"}
LABELS = ("LIST", "ASK", "CANCEL")
SITES = runner_contract_v1.SITES
LOAD_GPU_BYTES = 56 * 1024**3
PILOT_GPU_BYTES = 60 * 1024**3
MAX_EVIDENCE_BYTES = 64 * 1024**2
CUDA_MAJOR_MINOR = "12.8"
MIN_GPU_BYTES = 80_000_000_000


def stage_paths(stage: str) -> tuple[Path, Path]:
    require(stage in ("load", "pilot"), "unknown CUDA stage")
    return (LOAD_ATTEMPT, LOAD_GO) if stage == "load" else (PILOT_ATTEMPT, PILOT_GO)


def runtime_identity() -> dict[str, Any]:
    """Read device identity without touching checkpoint weights."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    math = math_mode(torch)
    require_math_mode(math)
    require(platform.system() == "Linux" and torch.version.cuda == CUDA_MAJOR_MINOR
            and os.environ.get("CUDA_VISIBLE_DEVICES", "0") in ("", "0"),
            "pinned Linux/single-GPU/CUDA runtime required")
    query = subprocess.run(("nvidia-smi", "--query-gpu=uuid,name,memory.total,compute_cap",
                            "--format=csv,noheader,nounits"), capture_output=True,
                           text=True, timeout=10, check=True)
    lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    require(len(lines) == 1, "CUDA device inventory changed")
    uuid, name, mib, capability = [part.strip() for part in lines[0].split(",")]
    require(uuid.startswith("GPU-") and name.startswith("NVIDIA A100")
            and mib.isdecimal() and int(mib) * 1024**2 >= MIN_GPU_BYTES
            and capability == "8.0", "target A100 80GB identity changed")
    driver = subprocess.run(("nvidia-smi", "--query-gpu=driver_version",
                             "--format=csv,noheader"), capture_output=True,
                            text=True, timeout=10, check=True).stdout.strip()
    return {"system": "Linux", "cuda": torch.version.cuda,
            "torch": importlib.metadata.version("torch"), "gpu_uuid": uuid,
            "gpu_name": name, "gpu_memory_mib": int(mib),
            "compute_capability": [8, 0], "driver_version": driver,
            "math_mode": math}


def math_mode(torch: Any) -> dict[str, Any]:
    return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "default_dtype": str(torch.get_default_dtype()),
            "cudnn_version": torch.backends.cudnn.version()}


def require_math_mode(mode: dict[str, Any]) -> None:
    require(mode["matmul_allow_tf32"] is False
            and mode["cudnn_allow_tf32"] is False
            and mode["float32_matmul_precision"] == "highest"
            and mode["default_dtype"] == "torch.float32",
            "FP32/TF32 math mode changed")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def write_once(path: Path, value: Any) -> None:
    data = canonical(value)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def save_array(folder: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    require(array.dtype == np.float32 and np.isfinite(array).all(),
            f"{name} must be finite FP32")
    path = folder / f"{name}.npy"
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": path.name, "sha256": sha(path), "shape": list(array.shape),
            "dtype": str(array.dtype), "bytes": path.stat().st_size}


def source_identity() -> dict[str, Any]:
    """Require every reviewed source to be committed and unchanged."""
    hashes: dict[str, str] = {}
    for relative in SOURCES:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing source: {relative}")
        tracked = subprocess.run(("git", "ls-files", "--error-unmatch", "--", relative),
                                 cwd=ROOT, capture_output=True)
        require(tracked.returncode == 0, f"untracked source: {relative}")
        for flag in ((), ("--cached",)):
            clean = subprocess.run(("git", "diff", *flag, "--quiet", "--", relative),
                                   cwd=ROOT, capture_output=True)
            require(clean.returncode == 0, f"dirty source: {relative}")
        hashes[relative] = sha(path)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == PACKAGES, "pilot package versions changed")
    distribution = importlib.metadata.distribution("transformers")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(direct.get("vcs_info", {}).get("commit_id")
            == load_spec()["transformers_revision"],
            "installed Transformers revision changed")
    source_audit = typed_pilot_cuda_source_audit_v2.audit()
    snapshot = admission.verify_snapshot()
    binaries = admission.verify_binaries()
    return {"sources": hashes, "packages": versions,
            "transformers_commit": direct["vcs_info"]["commit_id"],
            "spec_sha256": sha(SPEC_PATH),
            "readout_sha256": sha(HERE / "READOUT_FREEZE_V2.json"),
            "installed_source_audit": source_audit,
            "active_snapshot": snapshot, "installed_binaries": binaries}


def validate_go(stage: str, *, verify_runtime: bool = True) -> dict[str, Any]:
    identity = source_identity()
    attempt, go = stage_paths(stage)
    require(go.is_file() and not go.is_symlink()
            and REVIEW.is_file() and not REVIEW.is_symlink(),
            "independent CUDA stage GO absent")
    receipt = json.loads(go.read_text())
    require(receipt.get("schema") == "qwen_action_typed_pilot_cuda_go_v2"
            and receipt.get("status") == "independent_executable_go"
            and receipt.get("stage") == stage
            and receipt.get("approved_attempt") == str(attempt.resolve())
            and receipt.get("scope") == "one typed F_L1 pilot only"
            and receipt.get("qa_policy") == "typed_F_L1_only_no_audio_QA_exception"
            and receipt.get("qa_amendment_sha256") == sha(QA_AMENDMENT)
            and receipt.get("parent_amendment_sha256") == sha(PARENT_AMENDMENT)
            and receipt.get("source_identity") == identity
            and isinstance(receipt.get("runtime_identity"), dict)
            and isinstance(receipt.get("gpu_baseline"), dict)
            and receipt.get("review_sha256") == sha(REVIEW),
            "typed-pilot CUDA GO binding invalid")
    baseline = receipt["gpu_baseline"]
    require(set(baseline) == {"gpu_uuid", "gpu_name", "gpu_memory_mib",
                            "mig_mode", "compute_processes", "gpu_used_bytes",
                            "idle_gpu_limit_bytes"}
            and baseline["gpu_uuid"] == receipt["runtime_identity"].get("gpu_uuid")
            and baseline["gpu_name"] == receipt["runtime_identity"].get("gpu_name")
            and baseline["gpu_memory_mib"] == receipt["runtime_identity"].get("gpu_memory_mib")
            and baseline["mig_mode"] == "Disabled"
            and baseline["compute_processes"] == []
            and baseline["idle_gpu_limit_bytes"] == admission.IDLE_GPU_LIMIT_BYTES
            and type(baseline["gpu_used_bytes"]) is int
            and 0 <= baseline["gpu_used_bytes"] < admission.IDLE_GPU_LIMIT_BYTES,
            "reviewed GPU baseline/isolation policy invalid")
    if verify_runtime:
        require(receipt["runtime_identity"] == runtime_identity(),
                "live CUDA runtime differs from GO")
        live = admission.gpu_inventory()
        require(all(live[key] == baseline[key] for key in
                    ("gpu_uuid", "gpu_name", "gpu_memory_mib", "mig_mode",
                     "compute_processes", "idle_gpu_limit_bytes")),
                "live GPU inventory differs from reviewed baseline")
        require(type(live["gpu_used_bytes"]) is int
                and 0 <= live["gpu_used_bytes"] < admission.IDLE_GPU_LIMIT_BYTES,
                "live GPU idle memory exceeds isolation limit")
    if stage == "pilot":
        from . import typed_pilot_cuda_audit_v2
        load_audit = typed_pilot_cuda_audit_v2.audit("load")
        require(json.loads(LOAD_AUDIT_PATH.read_text()) == load_audit
                and receipt.get("load_audit_sha256") == sha(LOAD_AUDIT_PATH)
                and receipt.get("load_go_sha256") == sha(LOAD_GO),
                "passing load stage must be bound before pilot")
    return receipt


def parent_admission(attempt: Path, stage: str) -> dict[str, Any]:
    """Prove live bounded-parent launch before any candidate import or load."""
    expected, go = stage_paths(stage)
    require(attempt.resolve() == expected.resolve() and attempt.is_dir()
            and not (attempt / "worker_output").exists(),
            "fixed new pilot output required")
    pre_path = attempt / "pre_host.json"
    pre = json.loads(pre_path.read_text())
    stage_start_path = attempt / "stage_start_gpu.json"
    stage_start = json.loads(stage_start_path.read_text())
    start = json.loads((attempt / "start.json").read_text())
    require(start.get("schema") == "qwen_action_typed_pilot_cuda_parent_v2"
            and start.get("status") == "worker_authorized"
            and start.get("stage") == stage
            and start.get("approved_attempt") == str(expected.resolve())
            and start.get("go_sha256") == sha(go)
            and start.get("pre_host_sha256") == sha(pre_path)
            and start.get("stage_start_gpu_sha256") == sha(stage_start_path)
            and stage_start.get("gpu_uuid") == pre["runtime_identity"]["gpu_uuid"]
            and stage_start.get("compute_processes") == []
            and stage_start.get("gpu_used_bytes", admission.IDLE_GPU_LIMIT_BYTES)
                < admission.IDLE_GPU_LIMIT_BYTES
            and start.get("parent_pid") == os.getppid()
            and isinstance(start.get("nonce_sha256"), str)
            and len(start["nonce_sha256"]) == 64
            and 0 < start.get("deadline_monotonic", 0) - time.monotonic()
            <= (180 if stage == "load" else 900),
            "bounded parent start receipt invalid")
    def alarm(_sig: int, _frame: Any) -> None:
        raise TimeoutError("pilot worker wall deadline")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, start["deadline_monotonic"]-time.monotonic())
    fd_text = os.environ.get("QWEN_TYPED_PILOT_PARENT_FD", "")
    require(fd_text.isdecimal(), "inherited parent pipe absent")
    fd = int(fd_text)
    require(fd > 2 and stat.S_ISFIFO(os.fstat(fd).st_mode),
            "inherited parent pipe invalid")
    ready, _, _ = select.select([fd], [], [], 2)
    require(bool(ready), "parent start nonce timed out")
    nonce = b""
    while not nonce.endswith(b"\n") and len(nonce) <= 65:
        nonce += os.read(fd, 1)
        require(bool(nonce), "parent pipe closed before nonce")
    try:
        nonce_bytes = bytes.fromhex(nonce[:-1].decode())
    except (ValueError, UnicodeDecodeError):
        nonce_bytes = b""
    require(len(nonce) == 65 and nonce.endswith(b"\n")
            and len(nonce_bytes) == 32
            and hashlib.sha256(nonce_bytes).hexdigest() == start["nonce_sha256"],
            "parent nonce invalid")
    def watch() -> None:
        while True:
            ready, _, _ = select.select([fd], [], [], 3)
            if not ready or os.read(fd, 4096) == b"":
                os._exit(88)
    threading.Thread(target=watch, name="bounded-parent-watch", daemon=True).start()
    receipt = validate_go(stage, verify_runtime=False)
    require(pre["runtime_identity"] == receipt["runtime_identity"]
            and start["runtime_identity"] == receipt["runtime_identity"]
            and start["source_identity"] == receipt["source_identity"]
            and all(stage_start[key] == receipt["gpu_baseline"][key] for key in
                    ("gpu_uuid", "gpu_name", "gpu_memory_mib", "mig_mode",
                     "compute_processes", "idle_gpu_limit_bytes")),
            "bounded parent source/runtime identity changed")
    return start


def fixed_case() -> dict[str, str]:
    row = next(row for row in load_spec()["scripts"] if row["id"] == "F_L1")
    require(row["split"] == "fit_text" and row["label"] == "LIST",
            "fixed typed pilot case changed")
    return row


def measure_cuda(torch: Any, phase: str, folder: Path, *, stage: str = "pilot") -> int:
    math = math_mode(torch)
    require_math_mode(math)
    torch.cuda.synchronize(0)
    allocated = int(torch.cuda.memory_allocated(0))
    reserved = int(torch.cuda.memory_reserved(0))
    peak = int(torch.cuda.max_memory_reserved(0))
    limit = LOAD_GPU_BYTES if stage == "load" else PILOT_GPU_BYTES
    data = {"phase": phase, "allocated_bytes": allocated,
            "reserved_bytes": reserved, "peak_reserved_bytes": peak,
            "monotonic_seconds": time.monotonic(), "math_mode": math}
    with (folder / "cuda_samples.jsonl").open("ab") as handle:
        handle.write(canonical(data))
        handle.flush()
        os.fsync(handle.fileno())
    require(max(allocated, reserved, peak) < limit,
            "CUDA allocation/reservation limit exceeded")
    return peak


def _first(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_first(output: Any, replacement: Any) -> Any:
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    return replacement


def unmodified_references(model: Any, extended: dict[str, Any],
                          modules: dict[str, Any], torch: Any,
                          prompt_last: int) -> dict[str, Any]:
    """Stock logits, then an unmodified read-only-hook state/logit replay."""
    with torch.no_grad():
        stock = model(**extended, use_cache=False, return_dict=True)
    native = {
        "first_logits": stock.logits[0, prompt_last].float().detach().cpu().numpy().astype(np.float32),
        "after_label_logits": stock.logits[0, prompt_last+1].float().detach().cpu().numpy().astype(np.float32),
    }
    del stock
    calls: dict[str, list[Any]] = {site: [] for site in SITES}
    handles = []
    for site in SITES:
        def observe(_module: Any, _inputs: Any, output: Any, *, key: str = site) -> None:
            state = _first(output)
            require(state.ndim == 3 and state.shape[0] == 1
                    and state.shape[-1] == runner_contract_v1.WIDTH,
                    "read-only reference state shape invalid")
            calls[key].append(state.detach())
            return None
        handles.append(modules[site].register_forward_hook(observe))
    try:
        with torch.no_grad():
            observed = model(**extended, use_cache=False, return_dict=True)
        require(all(len(calls[site]) == 1 for site in SITES),
                "read-only reference hook count invalid")
        read_only = {
            "first_logits": observed.logits[0, prompt_last].float().detach().cpu().numpy().astype(np.float32),
            "after_label_logits": observed.logits[0, prompt_last+1].float().detach().cpu().numpy().astype(np.float32),
            "states": {site: calls[site][0][0, prompt_last].float().cpu().numpy().astype(np.float32)
                       for site in SITES},
            "hook_calls": {site: len(calls[site]) for site in SITES},
        }
    finally:
        for handle in reversed(handles):
            handle.remove()
    return {"no_hook": native, "read_only": read_only}


def teacher_forced_triplet(model: Any, batch: dict[str, Any],
                           modules: dict[str, Any], torch: Any,
                           folder: Path) -> list[dict[str, Any]]:
    prompt = batch["input_ids"]
    require(prompt.ndim == 2 and tuple(prompt.shape[:1]) == (1,)
            and prompt.shape[1] > 0, "typed prompt tensor invalid")
    prompt_ids = [int(value) for value in prompt[0].tolist()]
    prompt_hash = hashlib.sha256(json.dumps(prompt_ids,
                             separators=(",", ":")).encode()).hexdigest()
    records = []
    for label in LABELS:
        label_id = readout_v2.LABEL_TOKEN_IDS[LABELS.index(label)]
        extended = dict(batch)
        suffix = torch.tensor([[label_id]], dtype=prompt.dtype, device=prompt.device)
        extended["input_ids"] = torch.cat((prompt, suffix), dim=1)
        if "attention_mask" in batch:
            extended["attention_mask"] = torch.cat((batch["attention_mask"],
                                      torch.ones_like(suffix, dtype=batch["attention_mask"].dtype)), dim=1)
        references = unmodified_references(model, extended, modules, torch,
                                           len(prompt_ids)-1)
        calls: dict[str, list[Any]] = {site: [] for site in SITES}
        handles = []
        for site in SITES:
            def capture(_module: Any, _inputs: Any, output: Any, *, key: str = site) -> Any:
                value = _first(output)
                require(value.ndim == 3 and value.shape[0] == 1
                        and value.shape[-1] == runner_contract_v1.WIDTH,
                        f"{key} hook state shape invalid")
                if key == SITES[0]:
                    # With frozen weights and integer IDs, no upstream tensor
                    # requires grad. Insert a numerically identical leaf at the
                    # first site; the later site remains connected to it.
                    value = value.detach().requires_grad_(True)
                    calls[key].append(value)
                    return _replace_first(output, value)
                calls[key].append(value)
                return None
            handles.append(modules[site].register_forward_hook(capture))
        try:
            with torch.enable_grad():
                output = model(**extended, use_cache=False, return_dict=True)
                require(all(len(calls[site]) == 1 for site in SITES),
                        "teacher-forced hook call count invalid")
                first = output.logits[0, len(prompt_ids)-1].float()
                after = output.logits[0, len(prompt_ids)].float()
                require(first.ndim == after.ndim == 1
                        and first.shape == after.shape
                        and first.numel() > readout_v2.EOS_TOKEN_ID,
                        "teacher-forced full-vocabulary logits invalid")
                # The model and saved logits remain FP32. Normalize the very
                # large vocabulary in FP64 so the scalar exactly matches the
                # independent NumPy log-sum-exp audit.
                score = (first.double().log_softmax(-1)[label_id]
                         + after.double().log_softmax(-1)[readout_v2.EOS_TOKEN_ID])
                gradients = torch.autograd.grad(score,
                    tuple(calls[site][0] for site in SITES), allow_unused=False)
            entry: dict[str, Any] = {"label": label,
                "target_ids": [label_id, readout_v2.EOS_TOKEN_ID],
                "prompt_ids": prompt_ids, "prompt_ids_sha256": prompt_hash,
                "teacher_forced_input_ids": prompt_ids + [label_id],
                "logit_positions": [len(prompt_ids)-1, len(prompt_ids)],
                "hook_calls": {site: len(calls[site]) for site in SITES},
                "claimed_score": readout_v2.complete_label_log_score(
                    first.detach().cpu().numpy().astype(np.float32),
                    after.detach().cpu().numpy().astype(np.float32), label_id),
                "autograd_score": float(score.detach().cpu())}
            require(abs(entry["claimed_score"]-entry["autograd_score"]) <= 1e-6,
                    "autograd and saved-logit scores disagree")
            parity: dict[str, float] = {}
            for kind, hooked in (("first_logits", first.detach().cpu().numpy()),
                                 ("after_label_logits", after.detach().cpu().numpy())):
                native = references["no_hook"][kind]
                read_only = references["read_only"][kind]
                parity[f"native_read_only_{kind}"] = float(np.max(np.abs(native-read_only)))
                parity[f"read_only_leaf_{kind}"] = float(np.max(np.abs(read_only-hooked)))
            for site in SITES:
                native_state = references["read_only"]["states"][site]
                leaf_state = calls[site][0][0, len(prompt_ids)-1].detach().cpu().numpy()
                parity[f"read_only_leaf_{site}"] = float(np.max(np.abs(native_state-leaf_state)))
            require(all(value <= 1e-5 for value in parity.values()),
                    "unmodified/read-only/leaf teacher-forced parity failed")
            entry["reference_parity_maxabs"] = parity
            entry["reference_hook_calls"] = references["read_only"]["hook_calls"]
            entry["sidecars"] = {
                "first_logits": save_array(folder, f"{label}_first_logits",
                                          first.detach().cpu().numpy().astype(np.float32)),
                "after_label_logits": save_array(folder, f"{label}_after_logits",
                                          after.detach().cpu().numpy().astype(np.float32)),
                "states": {}, "gradients": {},
                "reference": {"no_hook": {}, "read_only": {"states": {}}}}
            for kind in ("first_logits", "after_label_logits"):
                entry["sidecars"]["reference"]["no_hook"][kind] = save_array(
                    folder, f"{label}_native_{kind}", references["no_hook"][kind])
                entry["sidecars"]["reference"]["read_only"][kind] = save_array(
                    folder, f"{label}_readonly_{kind}", references["read_only"][kind])
            for site, grad in zip(SITES, gradients, strict=True):
                key = site.replace(".", "_")
                entry["sidecars"]["reference"]["read_only"]["states"][site] = save_array(
                    folder, f"{label}_{key}_readonly_state",
                    references["read_only"]["states"][site])
                entry["sidecars"]["states"][site] = save_array(folder,
                    f"{label}_{key}_state",
                    calls[site][0][0, len(prompt_ids)-1].detach().cpu().numpy().astype(np.float32))
                entry["sidecars"]["gradients"][site] = save_array(folder,
                    f"{label}_{key}_vjp",
                    grad[0, len(prompt_ids)-1].detach().cpu().numpy().astype(np.float32))
            records.append(entry)
            measure_cuda(torch, f"after_{label}_vjp", folder)
        finally:
            for handle in reversed(handles):
                handle.remove()
        require(sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
                < MAX_EVIDENCE_BYTES, "pilot evidence cap exceeded")
    return records


def generation_pair(model: Any, batch: dict[str, Any], module: Any,
                    processor: Any, torch: Any) -> dict[str, Any]:
    prompt_length = int(batch["input_ids"].shape[1])
    settings = load_spec()["generation"]
    def generate(*, read_only: bool) -> tuple[list[int], int]:
        calls = []
        handle = None
        if read_only:
            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                calls.append(tuple(_first(output).shape))
                return None
            handle = module.register_forward_hook(capture)
        try:
            with torch.no_grad():
                result = model.generate(**batch, **settings,
                                        eos_token_id=readout_v2.EOS_TOKEN_ID,
                                        return_dict_in_generate=True)
            return [int(x) for x in result.sequences[0, prompt_length:].tolist()], len(calls)
        finally:
            if handle is not None:
                handle.remove()
    first, _ = generate(read_only=False)
    second, hook_calls = generate(read_only=True)
    first_result = endpoint_v2.score_complete_answer(first, processor.tokenizer)
    second_result = endpoint_v2.score_complete_answer(second, processor.tokenizer)
    require(first == second, "read-only generation changes complete token path")
    return {"no_hook": first_result, "read_only": second_result,
            "read_only_hook_calls": hook_calls,
            "raw_no_hook_ids": first, "raw_read_only_ids": second}


def run_worker(attempt: Path, stage: str) -> None:
    start = parent_admission(attempt, stage)  # Before candidate imports or weights.
    _, go = stage_paths(stage)
    require(os.getppid() == start["parent_pid"]
            and time.monotonic() < start["deadline_monotonic"],
            "bounded parent lost before candidate import")
    import torch
    from models.qwen_omni.adapter import resolve_hook_points
    from transformers import (Qwen2_5OmniProcessor,
                              Qwen2_5OmniThinkerForConditionalGeneration)
    require(os.getppid() == start["parent_pid"]
            and time.monotonic() < start["deadline_monotonic"],
            "bounded parent lost before model load")
    require(runtime_identity() == start["runtime_identity"]
            and torch.cuda.is_available(), "CUDA runtime changed after admission")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    output = attempt / "worker_output"
    output.mkdir()
    write_once(output / "start.json", {"schema": "qwen_action_typed_pilot_cuda_worker_v2",
               "status": "started", "case_id": "F_L1__typed", "stage": stage,
               "go_sha256": sha(go), "pre_host_sha256": sha(attempt / "pre_host.json"),
               "stage_start_gpu_sha256": sha(attempt / "stage_start_gpu.json"),
               "parent_start_sha256": sha(attempt / "start.json"),
               "parent_pid": start["parent_pid"], "nonce_sha256": start["nonce_sha256"]})
    snapshot = Path(start["source_identity"]["active_snapshot"]["snapshot_path"])
    processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float32,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    resolve_hook_points(model)
    require(model.dtype == torch.float32
            and int(processor.tokenizer.eos_token_id) == readout_v2.EOS_TOKEN_ID,
            "pinned model dtype/tokenizer changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    measure_cuda(torch, "after_model_load", output, stage=stage)
    if stage == "load":
        write_once(output / "result.json", {"schema": "qwen_action_typed_pilot_cuda_worker_v2",
            "status": "candidate_saved_pending_parent_and_audit", "stage": "load",
            "case_id": "F_L1__typed", "model_revision": load_spec()["model_revision"],
            "cuda_log_sha256": sha(output / "cuda_samples.jsonl")})
        return
    row = fixed_case()
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": load_spec()["system_prompt"]}]},
        {"role": "user", "content": [{"type": "text", "text": row["text"]}]},
    ]
    prepared = processor.apply_chat_template(conversation, tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    require(set(prepared).issubset({"input_ids", "attention_mask"}),
            "typed processor unexpectedly added modality inputs")
    batch = {key: value.to("cuda:0") for key, value in prepared.items()}
    points = resolve_hook_points(model, audio_layers=(), text_layers=(17, 34))
    modules = {point.name: point.module for point in points if point.name in SITES}
    require(set(modules) == set(SITES), "pinned residual hook sites absent")
    triplet = teacher_forced_triplet(model, batch, modules, torch, output)
    generated = generation_pair(model, batch, modules["text.residual.34"],
                                processor, torch)
    measure_cuda(torch, "after_generation", output)
    # The read-only contract rechecks every saved FP32 array and score.
    write_once(output / "result.json", {"schema": "qwen_action_typed_pilot_cuda_worker_v2",
               "status": "candidate_saved_pending_parent_and_audit",
               "stage": "pilot", "case_id": "F_L1__typed", "model_revision": load_spec()["model_revision"],
               "prompt_text": row["text"], "triplet": triplet,
               "generation": generated,
               "generation_config": settings_snapshot(model),
               "cuda_log_sha256": sha(output / "cuda_samples.jsonl")})


def settings_snapshot(model: Any) -> dict[str, Any]:
    config = model.generation_config.to_dict()
    return {key: config.get(key) for key in
            ("eos_token_id", "pad_token_id", "bos_token_id", "max_new_tokens",
             "do_sample", "temperature", "top_p", "num_beams")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--stage", choices=("load", "pilot"))
    args = parser.parse_args()
    if args.source_preflight:
        print(json.dumps({"status": "go_required_before_weights",
                          "load_go_exists": LOAD_GO.is_file(),
                          "pilot_go_exists": PILOT_GO.is_file()}, sort_keys=True))
        return
    require(args.worker and args.attempt is not None and args.stage is not None,
            "only fixed typed worker or source preflight exists")
    run_worker(args.attempt, args.stage)


if __name__ == "__main__":
    main()
