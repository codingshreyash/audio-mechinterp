"""No-weight, one-shot H100 admission gate for the non-study audio pilot."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

from . import plan

ROOT = plan.ROOT
SOURCE_FILES = (
    "models/qwen_omni/adapter.py",
    "models/qwen_omni/pins.json",
    "models/qwen_omni/config.snapshot.json",
    "models/qwen_omni/sample_apple_5p5s.wav",
    "models/qwen_omni/time_map/index.py",
    "models/second_architecture/qwen_action_method_v1/audio_candidate_v1.py",
    "models/second_architecture/qwen_action_method_v1/human_qa_v2.py",
    "models/second_architecture/qwen_action_method_v1/runner_contract_v1.py",
    "models/second_architecture/qwen_action_method_v1/listening_bundle_v2.py",
    "models/second_architecture/qwen_action_method_v1/render_sources_v2.py",
    "models/second_architecture/qwen_action_method_v1/audit_sources_v2.py",
    "models/second_architecture/qwen_action_method_v1/gate_v3.py",
    "models/second_architecture/qwen_action_method_v1/readout_v2.py",
    "models/second_architecture/qwen_action_method_v1/endpoint_v2.py",
    "models/second_architecture/qwen_action_method_v1/plan.py",
    "models/second_architecture/qwen_action_method_v1/endpoint.py",
    "models/second_architecture/qwen_action_method_v1/frozen_spec_v1.json",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_cuda_admission_v2.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_cuda_admission_v3.py",
    "models/second_architecture/qwen_action_method_v1/TYPED_PILOT_CUDA_HF_SNAPSHOT_MANIFEST_V2.json",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/__init__.py",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/plan.py",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/gate.py",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/worker.py",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/parent.py",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/saved_audit.py",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/test_audio_resource_pilot_v1.py",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/PROTOCOL_V1.md",
    "models/second_architecture/qwen_action_method_v1/audio_resource_pilot_v1/nonstudy_apple_zero_pad_16s.wav",
)
PACKAGES = {"torch": "2.11.0+cu128", "transformers": "5.24.0.dev0",
            "accelerate": "1.16.0", "numpy": "2.5.5", "psutil": "7.2.2",
            "Pillow": "13.0.0", "torchvision": "0.26.0+cu128"}
IDLE_GPU_BYTES = 512 * 1024**2
ONE_GPU = re.compile(r"(?:[0-9]+|GPU-[0-9a-fA-F-]{36})\Z")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 4 * 1024**2,
            f"missing, linked or oversized receipt: {path}")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate receipt key")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    require(isinstance(value, dict), "receipt root is not object")
    return value


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_identity() -> dict[str, str]:
    plan.verify_wav()
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"source missing: {relative}")
        require(subprocess.run(("git", "ls-files", "--error-unmatch", "--", relative),
                               cwd=ROOT, capture_output=True).returncode == 0,
                f"source untracked: {relative}")
        for mode in ((), ("--cached",)):
            require(subprocess.run(("git", "diff", *mode, "--quiet", "--", relative),
                                   cwd=ROOT, capture_output=True).returncode == 0,
                    f"source dirty: {relative}")
        result[relative] = plan.sha(path)
    return result


def _smi(query: str) -> list[str]:
    response = subprocess.run(("nvidia-smi", query, "--format=csv,noheader,nounits"),
                              capture_output=True, text=True, timeout=10, check=True)
    return [x.strip() for x in response.stdout.splitlines() if x.strip()]


def gpu_inventory(*, require_idle: bool = True) -> dict[str, Any]:
    rows = _smi("--query-gpu=uuid,name,memory.total,memory.used,mig.mode.current")
    require(len(rows) == 1, "one visible physical GPU required")
    fields = [x.strip() for x in rows[0].split(",")]
    require(len(fields) == 5, "GPU inventory shape changed")
    uuid, name, total, used, mig = fields
    require(uuid.startswith("GPU-") and name.startswith("NVIDIA H100")
            and total.isdecimal() and int(total) * 1024**2 >= 80_000_000_000
            and used.isdecimal() and mig == "Disabled", "H100 80GB/MIG inventory invalid")
    if require_idle:
        processes = _smi("--query-compute-apps=gpu_uuid,pid,used_gpu_memory")
        require(not processes or processes == ["No running processes found"],
                "other GPU compute process present")
        require(int(used) * 1024**2 < IDLE_GPU_BYTES, "GPU not idle")
    return {"gpu_uuid": uuid, "gpu_name": name, "gpu_memory_mib": int(total),
            "gpu_used_bytes": int(used) * 1024**2, "mig_mode": mig,
            "compute_processes": [], "idle_gpu_limit_bytes": IDLE_GPU_BYTES}


def require_one_slurm_gpu(job_gpus: str, step_gpus: str,
                          gpus_on_node: str) -> dict[str, str]:
    """Accept a single step allocation only if this Slurm leaves job GPUs empty."""
    require(gpus_on_node == "1", "exactly one Slurm GPU on node required")
    if job_gpus:
        require(ONE_GPU.fullmatch(job_gpus) is not None,
                "SLURM_JOB_GPUS is not one GPU")
    if step_gpus:
        require(ONE_GPU.fullmatch(step_gpus) is not None,
                "SLURM_STEP_GPUS is not one GPU")
    if job_gpus and step_gpus:
        require(job_gpus == step_gpus,
                "Slurm job and step GPU allocation identifiers disagree")
    if job_gpus:
        return {"source": "SLURM_JOB_GPUS", "value": job_gpus}
    require(bool(step_gpus), "both Slurm GPU allocation fields empty")
    return {"source": "SLURM_STEP_GPUS", "value": step_gpus}


def require_torch_binary_preflight(admission: dict[str, Any], actual: dict[str, Any]) -> None:
    from .. import typed_pilot_cuda_admission_v2
    require(isinstance(admission.get("torch_binary_preflight"), dict)
            and actual.get("wheel_sha256") == typed_pilot_cuda_admission_v2.WHEEL_SHA256
            and actual == admission["torch_binary_preflight"],
            "pinned Torch wheel/installed CUDA binaries differ from admission")


def runtime_identity(*, require_idle: bool = True) -> dict[str, Any]:
    job = os.environ.get("SLURM_JOB_ID", "")
    nodes = os.environ.get("SLURM_JOB_NUM_NODES", "")
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
    job_gpus = os.environ.get("SLURM_JOB_GPUS", "")
    step_gpus = os.environ.get("SLURM_STEP_GPUS", "")
    gpus_on_node = os.environ.get("SLURM_GPUS_ON_NODE", "")
    hostname = socket.gethostname().split(".")[0]
    require(job.isdecimal() and nodes == "1" and bool(nodelist),
            "single-node Slurm GPU allocation required")
    selected_gpu = require_one_slurm_gpu(job_gpus, step_gpus, gpus_on_node)
    hosts = subprocess.run(("scontrol", "show", "hostnames", nodelist),
                           capture_output=True, text=True, timeout=10, check=True)
    listed = [x.strip().split(".")[0] for x in hosts.stdout.splitlines() if x.strip()]
    require(listed == [hostname], "current host differs from Slurm allocation")
    gpu = gpu_inventory(require_idle=require_idle)
    import torch  # No model weights.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    require(platform.system() == "Linux" and torch.version.cuda == "12.8"
            and torch.cuda.is_available()
            and os.environ.get("CUDA_VISIBLE_DEVICES", "0") in ("", "0"),
            "pinned single-GPU CUDA runtime required")
    actual = {package: importlib.metadata.version(package) for package in PACKAGES}
    require(actual == PACKAGES, "package versions differ")
    return {"system": "Linux", "cuda": torch.version.cuda,
            "packages": actual, "gpu_uuid": gpu["gpu_uuid"],
            "gpu_name": gpu["gpu_name"], "gpu_memory_mib": gpu["gpu_memory_mib"],
            "slurm_job_id": job, "slurm_node": hostname,
            "slurm_job_gpus": job_gpus, "slurm_step_gpus": step_gpus,
            "slurm_selected_gpu": selected_gpu,
            "slurm_gpus_on_node": gpus_on_node,
            "slurm_nodelist": nodelist,
            "math_mode": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                          "float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "default_dtype": str(torch.get_default_dtype())}}


def _commit(path: Path) -> str:
    result = subprocess.run(("git", "log", "-1", "--format=%H", "--", str(path)),
                            cwd=ROOT, capture_output=True, text=True, check=True)
    require(bool(result.stdout.strip()), "review/admission source not committed")
    return result.stdout.strip()


def _ancestor(a: str, b: str) -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", a, b),
                          cwd=ROOT, capture_output=True).returncode == 0


def validate_go(*, live: bool = True) -> dict[str, Any]:
    identity = source_identity()
    for path in (plan.ADMISSION, plan.REVIEW, plan.GO):
        relative = str(path.relative_to(ROOT))
        require(path.is_file() and not path.is_symlink()
                and subprocess.run(("git", "ls-files", "--error-unmatch", "--", relative),
                                   cwd=ROOT, capture_output=True).returncode == 0,
                f"missing committed gate artifact: {relative}")
        for mode in ((), ("--cached",)):
            require(subprocess.run(("git", "diff", *mode, "--quiet", "--", relative),
                                   cwd=ROOT, capture_output=True).returncode == 0,
                    f"dirty gate artifact: {relative}")
    source_commits = [_commit(ROOT / x) for x in SOURCE_FILES]
    admission_commit, review_commit, go_commit = map(_commit,
        (plan.ADMISSION, plan.REVIEW, plan.GO))
    require(all(x != review_commit and _ancestor(x, review_commit)
                for x in [*source_commits, admission_commit])
            and review_commit != go_commit and _ancestor(review_commit, go_commit),
            "source/admission, review, GO chronology invalid")
    admission = read(plan.ADMISSION)
    go = read(plan.GO)
    require(admission.get("schema") == "qwen_audio_resource_admission_v1"
            and admission.get("status") == "source_only_no_weights"
            and admission.get("source_identity") == identity
            and admission.get("approved_attempt") == str(plan.ATTEMPT.resolve())
            and go.get("schema") == "qwen_audio_resource_go_v1"
            and go.get("status") == "independent_executable_go"
            and go.get("scope") == "NON_STUDY_APPLE_ZERO_PAD_16S_resource_parity_only"
            and go.get("approved_attempt") == str(plan.ATTEMPT.resolve())
            and go.get("admission_sha256") == plan.sha(plan.ADMISSION)
            and go.get("review_sha256") == plan.sha(plan.REVIEW)
            and go.get("source_identity") == identity
            and go.get("runtime_identity") == admission.get("runtime_identity")
            and go.get("gpu_baseline") == admission.get("gpu_baseline")
            and go.get("wav_sha256") == plan.WAV_SHA256,
            "one-shot GO binding invalid")
    if live:
        observed = gpu_inventory()
        require(runtime_identity(require_idle=False) == go["runtime_identity"],
                "live runtime differs from GO")
        require(all(observed[k] == go["gpu_baseline"][k] for k in
                    ("gpu_uuid", "gpu_name", "gpu_memory_mib", "mig_mode",
                     "compute_processes", "idle_gpu_limit_bytes")),
                "live GPU differs from GO")
        from .. import typed_pilot_cuda_admission_v3
        snapshot = Path(admission["processor_preflight"]["snapshot"]["snapshot_path"])
        require(typed_pilot_cuda_admission_v3.preflight(snapshot) ==
                admission["processor_preflight"],
                "live cached snapshot/processor/wheel differs from admission")
        from .. import typed_pilot_cuda_admission_v2
        require_torch_binary_preflight(admission,
            typed_pilot_cuda_admission_v2.verify_binaries())
    return go


def capture_admission() -> dict[str, Any]:
    """No-weight admission capture for later independent review; never called here."""
    require(not plan.ADMISSION.exists() and not plan.ATTEMPT.exists(),
            "pilot already admitted or attempted")
    identity = source_identity()
    from .. import typed_pilot_cuda_admission_v2, typed_pilot_cuda_admission_v3
    processor = typed_pilot_cuda_admission_v3.preflight()
    torch_binaries = typed_pilot_cuda_admission_v2.verify_binaries()
    gpu = gpu_inventory()
    result = {"schema": "qwen_audio_resource_admission_v1",
              "status": "source_only_no_weights", "source_identity": identity,
              "runtime_identity": runtime_identity(require_idle=False),
              "gpu_baseline": gpu,
              "processor_preflight": processor,
              "torch_binary_preflight": torch_binaries,
              "approved_attempt": str(plan.ATTEMPT.resolve()),
              "wav_sha256": plan.WAV_SHA256}
    write_once(plan.ADMISSION, result)
    return result
