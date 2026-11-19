"""V3 exact image/video prerequisites and no-weight processor admission.

The V2 snapshot, torch binary, and GPU checks remain the base contract. This
module adds the two missing optional processor backends seen in failed load_001.
"""
from __future__ import annotations

import base64
import csv
import importlib.metadata
import io
import os
import zipfile
from pathlib import Path
from typing import Any

from . import typed_pilot_cuda_admission_v2 as base

IDLE_GPU_LIMIT_BYTES = base.IDLE_GPU_LIMIT_BYTES
verify_snapshot = base.verify_snapshot
verify_binaries = base.verify_binaries
gpu_inventory = base.gpu_inventory

PREREQUISITES = {
    "Pillow": {
        "version": "13.0.0",
        "env": "QWEN_PILLOW_WHEEL",
        "wheel": "pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "sha256": "42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca",
        "prefixes": ("PIL/", "pillow.libs/"),
    },
    "torchvision": {
        "version": "0.26.0+cu128",
        "env": "QWEN_TORCHVISION_WHEEL",
        "wheel": "torchvision-0.26.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256": "ccf26b4b659cfce6f2208cb8326071d51c70219a34856dfdf468d1e19af52c0d",
        "prefixes": ("torchvision/", "torchvision.libs/"),
    },
}
PROCESSOR_CLASSES = {
    "processor": "Qwen2_5OmniProcessor",
    "image_processor": "Qwen2VLImageProcessor",
    "video_processor": "Qwen2VLVideoProcessor",
    "feature_extractor": "WhisperFeatureExtractor",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify_prerequisites(*, wheel_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """Verify official wheel bytes and installed processor package file bytes."""
    records = {}
    for package, pin in PREREQUISITES.items():
        require(importlib.metadata.version(package) == pin["version"],
                f"{package} version changed")
        value = (wheel_paths or {}).get(package)
        if value is None:
            value = os.environ.get(pin["env"], "")
        wheel = Path(value)
        require(bool(str(value)) and wheel.is_file() and not wheel.is_symlink()
                and wheel.name == pin["wheel"] and base.sha(wheel) == pin["sha256"],
                f"{package} pinned wheel unavailable or changed")
        distribution = importlib.metadata.distribution(package)
        installed = {}
        with zipfile.ZipFile(wheel) as archive:
            record_names = [name for name in archive.namelist()
                            if name.endswith(".dist-info/RECORD")]
            require(len(record_names) == 1, f"{package} wheel RECORD missing")
            for path, encoded, size in csv.reader(io.StringIO(
                    archive.read(record_names[0]).decode("utf-8"))):
                if not path.startswith(pin["prefixes"]):
                    continue
                require(".." not in Path(path).parts and encoded.startswith("sha256=")
                        and size.isdecimal(), f"{package} RECORD entry unpinned")
                expected = base64.urlsafe_b64decode(encoded[7:] + "===").hex()
                local = Path(distribution.locate_file(path))
                require(local.is_file() and not local.is_symlink()
                        and local.stat().st_size == int(size)
                        and base.sha(local) == expected,
                        f"installed {package} file differs: {path}")
                installed[path] = expected
        require(installed and any(".so" in Path(name).name for name in installed),
                f"{package} binary inventory empty")
        records[package] = {"version": pin["version"], "wheel_path": str(wheel.resolve()),
                            "wheel_sha256": pin["sha256"],
                            "installed_file_sha256": installed}
    return records


def verify_processor(snapshot: Path) -> dict[str, Any]:
    """Instantiate only the processor from the exact local snapshot, no weights."""
    from transformers import Qwen2_5OmniProcessor

    require(snapshot.is_dir() and snapshot.name ==
            "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
            "processor snapshot revision changed")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(snapshot), local_files_only=True)
    classes = {name: type(processor if name == "processor" else getattr(processor,name)).__name__
               for name in PROCESSOR_CLASSES}
    require(classes == PROCESSOR_CLASSES, "image/video/audio processor classes changed")
    return {"snapshot": str(snapshot.resolve()), "classes": classes,
            "tokenizer_eos_id": int(processor.tokenizer.eos_token_id)}


def preflight(snapshot: Path | None = None,
              *, wheel_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """All optional-dependency and processor checks before weight import/load."""
    verified = verify_snapshot(snapshot)
    wheels = verify_prerequisites(wheel_paths=wheel_paths)
    processor = verify_processor(Path(verified["snapshot_path"]))
    require(processor["tokenizer_eos_id"] == 151645,
            "processor tokenizer EOS changed")
    return {"schema":"qwen_action_cuda_processor_preflight_v3",
            "status":"no_weights_loaded", "snapshot":verified,
            "prerequisites":wheels,"processor":processor}
