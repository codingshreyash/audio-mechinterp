"""No-weight CUDA V2 admission checks for exact checkpoint and installed binaries."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "TYPED_PILOT_CUDA_HF_SNAPSHOT_MANIFEST_V2.json"
WHEEL_NAME = "torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
WHEEL_SHA256 = "d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997"
IDLE_GPU_LIMIT_BYTES = 512 * 1024**2
BINARY_DISTRIBUTIONS = ("torch", "tokenizers", "safetensors", "triton",
    "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(folder: Path) -> dict[str, Path]:
    require(folder.is_dir() and not folder.is_symlink(), "snapshot folder invalid")
    found: dict[str, Path] = {}
    for path in folder.rglob("*"):
        require(path.is_file() and not path.is_dir(), "snapshot contains non-file")
        name = path.relative_to(folder).as_posix()
        require(name not in found, "snapshot path duplicated")
        found[name] = path
    return found


def verify_snapshot(snapshot: Path | None = None) -> dict[str, Any]:
    """Hash every file in the exact active HF snapshot, including all shards."""
    manifest = json.loads(MANIFEST.read_text())
    require(manifest["schema"] == "qwen_omni_3b_exact_snapshot_manifest_v2"
            and manifest["model_id"] == "Qwen/Qwen2.5-Omni-3B"
            and manifest["revision"] == "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
            and len(manifest["files"]) == 18,
            "expected HF manifest changed")
    require(len({row["path"] for row in manifest["files"]}) == 18
            and all(isinstance(row["size"], int) and row["size"] > 0
                    and isinstance(row["sha256"], str)
                    and len(row["sha256"]) == 64 for row in manifest["files"]),
            "expected HF manifest malformed")
    if snapshot is None:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(manifest["model_id"],
            revision=manifest["revision"], local_files_only=True))
    snapshot = Path(snapshot)
    require(snapshot.name == manifest["revision"]
            and snapshot.parent.name == "snapshots",
            "active cached snapshot revision/path changed")
    files = _files(snapshot)
    expected = {row["path"]: row for row in manifest["files"]}
    require(set(files) == set(expected), "active snapshot file inventory changed")
    results = {}
    for name, path in sorted(files.items()):
        row = expected[name]
        require(path.stat().st_size == row["size"]
                and sha(path) == row["sha256"],
                f"active snapshot file changed: {name}")
        results[name] = row["sha256"]
    return {"snapshot_path": str(snapshot.absolute()),
            "manifest_sha256": sha(MANIFEST), "revision": manifest["revision"],
            "file_count": len(results), "total_bytes": sum(row["size"] for row in expected.values()),
            "file_sha256": results}


def _wheel_record(wheel: Path) -> dict[str, tuple[str, int]]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        require(len(names) == 1, "torch wheel RECORD missing")
        records = {}
        for path, encoded, size in csv.reader(io.StringIO(
                archive.read(names[0]).decode("utf-8"))):
            if path.startswith("torch/") and ".so" in Path(path).name:
                require(encoded.startswith("sha256=") and size.isdecimal(),
                        "torch binary RECORD unpinned")
                raw = base64.urlsafe_b64decode(encoded[7:] + "===")
                records[path] = (raw.hex(), int(size))
        require(any(path.endswith("libtorch_cuda.so") for path in records),
                "torch CUDA binary absent from wheel")
        return records


def verify_binaries(wheel_path: Path | None = None) -> dict[str, Any]:
    """Verify the published wheel and installed torch binaries against its RECORD."""
    if wheel_path is None:
        wheel_text = os.environ.get("QWEN_TORCH_CUDA_WHEEL", "")
        require(bool(wheel_text), "pinned CUDA wheel path absent")
        wheel_path = Path(wheel_text)
    wheel = Path(wheel_path)
    require(wheel.is_file() and not wheel.is_symlink()
            and wheel.name == WHEEL_NAME and sha(wheel) == WHEEL_SHA256,
            "pinned CUDA wheel SHA-256 changed")
    record = _wheel_record(wheel)
    torch_dist = importlib.metadata.distribution("torch")
    installed_torch = {}
    for relative, (expected_digest, size) in sorted(record.items()):
        path = Path(torch_dist.locate_file(relative))
        require(path.is_file() and path.stat().st_size == size
                and sha(path) == expected_digest,
                f"installed torch binary differs from verified wheel: {relative}")
        installed_torch[relative] = expected_digest
    dependencies = {}
    names = set(BINARY_DISTRIBUTIONS[1:])
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if (name.startswith("nvidia-") and name.endswith("-cu12")
                and any(".so" in Path(str(entry)).name
                        for entry in distribution.files or ())):
            names.add(name)
    for name in sorted(names):
        distribution = importlib.metadata.distribution(name)
        binary_files = []
        for entry in distribution.files or ():
            if ".so" in Path(str(entry)).name:
                path = Path(distribution.locate_file(entry))
                require(path.is_file(), f"installed {name} binary missing")
                binary_files.append({"path": str(entry), "sha256": sha(path),
                                     "size": path.stat().st_size})
        require(binary_files, f"installed {name} binary inventory empty")
        dependencies[name] = {"version": distribution.version,
                              "binaries": sorted(binary_files, key=lambda row: row["path"])}
    return {"wheel_path": str(wheel.absolute()), "wheel_sha256": WHEEL_SHA256,
            "installed_torch_binary_sha256": installed_torch,
            "binary_dependencies": dependencies}


def _smi(args: tuple[str, ...]) -> list[str]:
    output = subprocess.run(("nvidia-smi", *args), capture_output=True, text=True,
                            timeout=10, check=True).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def gpu_inventory() -> dict[str, Any]:
    """Use physical nvidia-smi inventory; reject MIG and other compute users."""
    rows = _smi(("--query-gpu=uuid,name,memory.total,memory.used,mig.mode.current",
                 "--format=csv,noheader,nounits"))
    require(len(rows) == 1, "one physical GPU required")
    fields = [field.strip() for field in rows[0].split(",")]
    require(len(fields) == 5, "GPU inventory fields invalid")
    uuid, name, total, used, mig = fields
    require(uuid.startswith("GPU-") and name.startswith("NVIDIA A100")
            and total.isdecimal() and used.isdecimal() and mig == "Disabled",
            "physical A100/MIG inventory invalid")
    processes = _smi(("--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                      "--format=csv,noheader,nounits"))
    require(not processes or processes == ["No running processes found"],
            "competing GPU compute process present")
    memory = int(used) * 1024**2
    require(memory < IDLE_GPU_LIMIT_BYTES,
            "idle GPU memory baseline too high")
    return {"gpu_uuid": uuid, "gpu_name": name, "gpu_memory_mib": int(total),
            "mig_mode": mig, "compute_processes": [],
            "gpu_used_bytes": memory, "idle_gpu_limit_bytes": IDLE_GPU_LIMIT_BYTES}
