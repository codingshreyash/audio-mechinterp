"""Isolated, one-attempt NVIDIA VoiceChat public-sample environment smoke.

This module imports no NeMo/model code. Its CLI can start only the unchanged
pinned NVIDIA example after sample-only preflight and external freeze review.
It never admits a calendar fixture. Fake dependency injection is for CPU tests;
the CLI has no injection option.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import re
import resource
import selectors
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from models.second_architecture import voicechat_h100_local_preflight as pinned

HERE = Path(__file__).resolve().parent
SAMPLE_SCRIPT = "examples/speechlm2/offline_voicechat_fc_infer.py"
SAMPLE_WAV = "examples/speechlm2/sample_audio/sample_fc.wav"
SAMPLE_RESPONSE = "examples/speechlm2/function_calling/random_number_response.json"
SAMPLE_SHA256 = {
    SAMPLE_WAV: "265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136",
    SAMPLE_RESPONSE: "e36ae164cdc14a0b9bbed7b4be2b37223dc3730b39de1edafb99e73638e2b60f",
}
SAMPLE_BYTES = {SAMPLE_WAV: 384_652, SAMPLE_RESPONSE: 82}
AMENDMENT_SHA256 = "9e4fe95f0f3828c8bedfb6d36705a5824feffcb34854371ba6450e0ddb9b8be3"
MAX_WALL_SECONDS = 1_800
MAX_VRAM_MIB = 75_000
MAX_EVIDENCE_BYTES = 1 << 30
MIN_EVIDENCE_BYTES = 65_536
TERMINAL_RESERVE_BYTES = 16_384
MAX_FREEZE_BYTES = 65_536
RUNNER_PACKAGE_PINS = {"numpy": "1.26.4", "pyarrow": "25.0.1", "datasets": "5.0.1"}
MAMBA_DEPENDENCY_SPECS = {"tilelang": "==0.1.8", "apache-tvm-ffi": "<=0.1.9",
                          "quack-kernels": ">=0.3.4"}
FFMPEG_MAJOR = 7
FFMPEG_REQUIRED_SONAMES = {
    "libavutil": "libavutil.so.59", "libavcodec": "libavcodec.so.61",
    "libavformat": "libavformat.so.61", "libswresample": "libswresample.so.5",
    "libswscale": "libswscale.so.8",
}
FFMPEG_OPTIONAL_SONAMES = {
    "libavfilter": "libavfilter.so.10", "libavdevice": "libavdevice.so.61",
    "libpostproc": "libpostproc.so.58",
}
CHILD_ENV_KEYS = (
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
    "VIRTUAL_ENV", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHASHSEED",
    "HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
    "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "CUDA_LAUNCH_BLOCKING", "CUBLAS_WORKSPACE_CONFIG", "TORCH_DETERMINISTIC",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
)
SOURCE_FILES = {
    "runner": HERE / "public_sample_smoke_v1.py",
    "tests": HERE / "test_public_sample_smoke_v1.py",
    "handoff": HERE / "VOICECHAT_PUBLIC_SAMPLE_SMOKE_SOURCE_HANDOFF_V1.md",
    "preflight": HERE.parent / "voicechat_h100_local_preflight.py",
    "amendment": HERE / "VOICECHAT_PUBLIC_SAMPLE_SMOKE_AMENDMENT_V1.md",
    "concept_review": HERE / "VOICECHAT_PUBLIC_SAMPLE_SMOKE_AMENDMENT_INDEPENDENT_CONCEPTUAL_REVIEW_GO_V1.md",
    "native_amendment": HERE / "VOICECHAT_PUBLIC_SAMPLE_NATIVE_FFMPEG_AMENDMENT_V4.md",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def child_environment() -> dict[str, str]:
    """Freeze and pass only variables needed by the pinned offline example."""
    result = {key: os.environ[key] for key in CHILD_ENV_KEYS if key in os.environ}
    require("VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX" not in result
            and "VOICECHAT_CONTROLLER_KEY_HEX" not in result,
            "Review/controller key reached child environment")
    return result


def installed_environment_receipt() -> dict[str, Any]:
    """Bind every installed Python distribution and check sample-specific imports.

    This runs pip only; it does not import NeMo, a model, or third-party packages.
    The full sorted inventory is represented by its digest and line count.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=False, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("Full pip freeze unavailable") from error
    require(completed.returncode == 0 and len(completed.stdout) <= (1 << 20),
            "Full pip freeze failed or exceeded size cap")
    lines = [line for line in completed.stdout.splitlines() if line]
    require(0 < len(lines) <= 10_000 and all(line == line.strip() for line in lines),
            "Full pip freeze contains invalid lines")
    inventory = ("\n".join(sorted(lines)) + "\n").encode("utf-8")

    versions: dict[str, str | None] = {}
    for name in (*RUNNER_PACKAGE_PINS, *MAMBA_DEPENDENCY_SPECS):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    for name, expected in RUNNER_PACKAGE_PINS.items():
        require(versions[name] == expected,
                f"Sample runtime requires {name}=={expected}")
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError as error:
        raise ValueError("Version specifier checker unavailable") from error
    for name, spec in MAMBA_DEPENDENCY_SPECS.items():
        version = versions[name]
        require(version is None or version in SpecifierSet(spec),
                f"Installed Mamba dependency {name} violates {spec}")
    return {"schema": "voicechat_full_python_environment_v1",
            "python_executable": sys.executable,
            "pip_freeze_all_sorted_sha256": sha(inventory),
            "pip_freeze_all_line_count": len(lines),
            "runner_package_versions": {name: versions[name] for name in RUNNER_PACKAGE_PINS},
            "mamba_dependency_versions": {name: versions[name] for name in MAMBA_DEPENDENCY_SPECS},
            "mamba_dependency_specs": MAMBA_DEPENDENCY_SPECS}


def _os_release_identity() -> dict[str, str]:
    path = Path("/etc/os-release")
    require(path.is_file() and path.stat().st_size <= 4096, "OS release file unavailable")
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        name, value = line.split("=", 1)
        if name in {"ID", "VERSION_ID"}:
            values[name] = value.strip('"')
    require(values.get("ID") and values.get("VERSION_ID"), "OS release identity missing")
    return {"id": values["ID"], "version_id": values["VERSION_ID"]}


def _conda_source_receipt(root: Path) -> dict[str, Any]:
    manifest_path = root / "native-source-manifest.json"
    manifest, _ = _read_json(manifest_path)
    require(set(manifest) == {"schema", "kind", "ffmpeg_major", "explicit_spec",
                              "package_archives"}
            and manifest["schema"] == "voicechat_native_source_manifest_v1"
            and manifest["kind"] == "conda_explicit"
            and manifest["ffmpeg_major"] == FFMPEG_MAJOR,
            "Native source manifest scope differs")
    descriptor = manifest["explicit_spec"]
    require(type(descriptor) is dict
            and set(descriptor) == {"path", "sha256", "bytes"}
            and descriptor["path"] == "conda-explicit.txt"
            and isinstance(descriptor["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is not None
            and type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= (1 << 20),
            "Native conda explicit-spec descriptor invalid")
    spec_path = root / descriptor["path"]
    digest, size = hash_file(spec_path, cap=1 << 20)
    require(digest == descriptor["sha256"] and size == descriptor["bytes"],
            "Conda explicit spec changed")
    spec_lines = spec_path.read_text().splitlines()
    require(spec_lines.count("@EXPLICIT") == 1, "Conda explicit spec header missing")
    package_lines = [line for line in spec_lines if line and not line.startswith("#")
                     and line != "@EXPLICIT"]
    require(0 < len(package_lines) <= 512 and all(
        re.fullmatch(r"https://[^\s#]+(?:\.conda|\.tar\.bz2)#[0-9a-fA-F]{32}(?:[0-9a-fA-F]{32})?", line)
        is not None for line in package_lines),
        "Conda explicit spec lacks hashed package URLs")
    expected_archives = {line.rsplit("/", 1)[1].split("#", 1)[0]
                         for line in package_lines}
    require(len(expected_archives) == len(package_lines),
            "Conda explicit spec has duplicate archive names")
    metadata_dir = root / "conda-meta"
    require(metadata_dir.is_dir() and not metadata_dir.is_symlink(),
            "Conda package metadata directory missing")
    metadata_paths = sorted(metadata_dir.iterdir(), key=lambda item: item.name)
    require(0 < len(metadata_paths) <= 512
            and any(item.name.endswith(".json") for item in metadata_paths),
            "Conda package metadata count invalid")
    metadata: list[dict[str, Any]] = []
    for item in metadata_paths:
        require(item.is_file() and not item.is_symlink(),
                "Unexpected Conda package metadata entry")
        item_sha, item_size = hash_file(item, cap=1 << 20)
        metadata.append({"path": str(item.relative_to(root)),
                         "sha256": item_sha, "bytes": item_size})
    archives = manifest["package_archives"]
    require(type(archives) is list and len(archives) == len(package_lines),
            "Conda package archive count differs from explicit spec")
    archive_dir = root / "package-archives"
    require(archive_dir.is_dir() and not archive_dir.is_symlink()
            and archive_dir == archive_dir.resolve(),
            "Conda package archive directory is missing or linked")
    seen_archives: set[str] = set()
    archive_hashes: list[dict[str, Any]] = []
    for item in archives:
        require(type(item) is dict and set(item) == {"path", "sha256", "bytes"}
                and isinstance(item["path"], str)
                and re.fullmatch(r"package-archives/[A-Za-z0-9_+.-]+(?:\.conda|\.tar\.bz2)",
                                 item["path"]) is not None
                and isinstance(item["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None
                and type(item["bytes"]) is int and item["bytes"] > 0,
                "Conda package archive descriptor invalid")
        name = Path(item["path"]).name
        require(name in expected_archives and name not in seen_archives,
                "Conda archive differs from explicit spec or is duplicated")
        seen_archives.add(name)
        archive_path = root / item["path"]
        require(archive_path == archive_path.resolve(),
                "Conda package archive path is linked")
        archive_sha, archive_size = hash_file(archive_path)
        require(archive_sha == item["sha256"] and archive_size == item["bytes"],
                "Conda package archive bytes changed")
        metadata_name = name.removesuffix(".conda").removesuffix(".tar.bz2") + ".json"
        metadata_path = metadata_dir / metadata_name
        metadata_value, _ = _read_json(metadata_path, cap=1 << 20)
        require(metadata_value.get("sha256") == archive_sha,
                "Conda package archive does not match installed metadata SHA-256")
        archive_hashes.append({"path": item["path"], "sha256": archive_sha,
                               "bytes": archive_size})
    require(seen_archives == expected_archives, "Conda package archives incomplete")
    require({item.name for item in archive_dir.iterdir()} == seen_archives,
            "Conda package archive directory has unexpected files")
    manifest_sha, manifest_size = hash_file(manifest_path, cap=MAX_FREEZE_BYTES)
    return {"kind": "conda_explicit", "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha, "manifest_bytes": manifest_size,
            "explicit_spec": {"path": str(spec_path), "sha256": digest,
                              "bytes": size, "package_count": len(package_lines)},
            "conda_metadata": metadata,
            "package_archives_count": len(archive_hashes),
            "package_archives_sorted_sha256": sha(canonical(sorted(
                archive_hashes, key=lambda entry: entry["path"])))}


def _torchcodec_core_path() -> Path:
    try:
        distribution = importlib.metadata.distribution("torchcodec")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("TorchCodec distribution unavailable") from error
    path = Path(distribution.locate_file(f"torchcodec/libtorchcodec_core{FFMPEG_MAJOR}.so"))
    require(path.is_absolute() and path == path.resolve()
            and path.is_file() and not path.is_symlink()
            and path.is_relative_to(Path(sys.prefix).resolve()),
            "TorchCodec matching FFmpeg core missing or outside Python environment")
    return path


def native_ffmpeg_runtime(nemo_dir: Path) -> dict[str, Any]:
    """Bind a task-local Conda FFmpeg prefix, its source manifest, and native bytes."""
    value = os.environ.get("LD_LIBRARY_PATH", "")
    components = value.split(":")
    require(value and all(component and Path(component).is_absolute()
                          for component in components),
            "LD_LIBRARY_PATH must contain only absolute nonempty components")
    libdir = Path(components[0])
    require(libdir.name == "lib" and libdir == libdir.resolve() and libdir.is_dir(),
            "First LD_LIBRARY_PATH component is not an isolated FFmpeg lib directory")
    root = libdir.parent
    require(root.is_dir() and root == root.resolve()
            and root.name.startswith("voicechat-sample-"),
            "FFmpeg root is not a task-local prefix")
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "FFmpeg runtime requires Linux x86_64")
    release = _os_release_identity()
    source = _conda_source_receipt(root)

    entries = sorted((item for item in libdir.iterdir() if ".so" in item.name),
                     key=lambda item: item.name)
    require(5 <= len(entries) <= 2048, "FFmpeg shared-library count invalid")
    target_hashes: dict[str, dict[str, Any]] = {}
    aliases: dict[str, dict[str, str | None]] = {}
    present: set[str] = set()
    sonames = {**FFMPEG_REQUIRED_SONAMES, **FFMPEG_OPTIONAL_SONAMES}
    for entry in entries:
        require(entry.is_file() or entry.is_symlink(),
                "Unexpected FFmpeg library directory entry")
        require(re.fullmatch(r"[A-Za-z0-9_+.-]+\.so(?:\.[A-Za-z0-9_.+-]+)?", entry.name)
                is not None, "Unexpected FFmpeg shared-library filename")
        for family, soname in sonames.items():
            if entry.name.startswith(family + ".so."):
                require(entry.name == soname or entry.name.startswith(soname + "."),
                        "Unexpected FFmpeg library major version")
                present.add(entry.name)
        target = entry.resolve()
        require(target.is_relative_to(libdir) and target.is_file()
                and not target.is_symlink(),
                "FFmpeg library symlink escapes isolated directory")
        relative = str(target.relative_to(libdir))
        if relative not in target_hashes:
            digest, size = hash_file(target)
            target_hashes[relative] = {"sha256": digest, "bytes": size}
        aliases[entry.name] = {"resolved_target": relative,
                               "symlink_target": os.readlink(entry) if entry.is_symlink() else None}
    require(set(FFMPEG_REQUIRED_SONAMES.values()) <= present,
            "Required FFmpeg soname missing")
    core_path = _torchcodec_core_path()
    core_sha, core_size = hash_file(core_path)
    sample_hashes = check_sample(nemo_dir)
    return {"schema": "voicechat_native_ffmpeg_runtime_v1",
            "platform": {"system": platform.system(), "machine": platform.machine(),
                         "distribution": release},
            "ffmpeg_major": FFMPEG_MAJOR,
            "isolated_root": str(root), "isolated_libdir": str(libdir),
            "ld_library_path": value,
            "required_sonames": FFMPEG_REQUIRED_SONAMES,
            "source": source,
            "library_aliases": aliases, "library_targets": target_hashes,
            "torchcodec_core": {"path": str(core_path), "sha256": core_sha,
                                "bytes": core_size},
            "public_wav_sha256": sample_hashes[SAMPLE_WAV]}


def _utc_record(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def hash_file(path: Path, *, cap: int | None = None) -> tuple[str, int]:
    require(path.is_file() and not path.is_symlink(), f"Not a regular unlinked file: {path}")
    before = path.stat().st_size
    require(cap is None or before <= cap, f"File exceeds cap: {path}")
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as stream:
        while block := stream.read(65536):
            seen += len(block)
            require(cap is None or seen <= cap, f"File grew beyond cap: {path}")
            digest.update(block)
    require(seen == before and path.stat().st_size == before,
            f"File changed while hashing: {path}")
    return digest.hexdigest(), before


def _read_json(path: Path, cap: int = MAX_FREEZE_BYTES) -> tuple[dict[str, Any], bytes]:
    digest, size = hash_file(path, cap=cap)
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    require(len(data) == size and sha(data) == digest, "Review/freeze changed during read")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    require(type(value) is dict, "Review/freeze must be an object")
    return value, data


def check_sample(nemo_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, expected in SAMPLE_SHA256.items():
        path = nemo_dir / relative
        actual, size = hash_file(path, cap=SAMPLE_BYTES[relative])
        require(actual == expected and size == SAMPLE_BYTES[relative],
                f"Pinned public sample changed: {relative}")
        result[relative] = actual
    with (nemo_dir / SAMPLE_WAV).open("rb") as stream:
        require(stream.read(12)[:4] == b"RIFF", "Pinned sample is not RIFF/WAVE")
    return result


def check_exact_nano_ref(nano_home: Path) -> str:
    path = (nano_home / "hub" / pinned.NANO_CACHE_REPO / "refs" / "main")
    digest, size = hash_file(path, cap=41)
    expected = sha(pinned.NANO_COMMIT.encode("ascii"))
    require(size == 40 and digest == expected,
            "Nano refs/main exact bytes differ")
    return digest


def fixed_command(nemo_dir: Path, checkpoint_dir: Path, output_dir: Path) -> list[str]:
    return [sys.executable, str(nemo_dir / SAMPLE_SCRIPT),
            "--checkpoint", str(checkpoint_dir),
            "--wav", str(nemo_dir / SAMPLE_WAV),
            "--api-response-json", str(nemo_dir / SAMPLE_RESPONSE),
            "--output-dir", str(output_dir / "official_outputs")]


def expected_freeze_fields(nemo_dir: Path, checkpoint_dir: Path,
                           nano_home: Path, output_dir: Path,
                           gpu_policy: str) -> dict[str, Any]:
    paths = [nemo_dir, checkpoint_dir, nano_home, output_dir]
    require(all(path.is_absolute() and path == path.resolve() for path in paths),
            "Freeze paths must be absolute and resolved")
    require(gpu_policy in ("h100", "a100-80gb"), "GPU policy invalid")
    require(hash_file(SOURCE_FILES["amendment"])[0] == AMENDMENT_SHA256,
            "Sample-only amendment changed")
    require(hash_file(SOURCE_FILES["preflight"])[0] == "6233cc01b6bbd69b71aa978f3b2dabc0210bf6175faf4f34bf73eb44ead586a0",
            "Shared pinned preflight changed")
    source_hashes = {name: hash_file(path)[0] for name, path in SOURCE_FILES.items()}
    installed_environment = installed_environment_receipt()
    native_runtime = native_ffmpeg_runtime(nemo_dir)
    return {
        "paths": {"nemo_dir": str(nemo_dir), "checkpoint_dir": str(checkpoint_dir),
                  "nano_hf_home": str(nano_home), "output_dir": str(output_dir)},
        "gpu_policy": gpu_policy,
        "command": fixed_command(nemo_dir, checkpoint_dir, output_dir),
        "source_sha256": source_hashes,
        "identity": {
            "nemo_revision": pinned.NEMO_COMMIT,
            "nemo_source_sha256": pinned.SOURCE_SHA256,
            "sample_sha256": SAMPLE_SHA256,
            "sample_bytes": SAMPLE_BYTES,
            "voicechat_revision": pinned.VOICECHAT_COMMIT,
            "config_sha256": pinned.CONFIG_SHA256,
            "weight_sha256": pinned.WEIGHT_SHA256,
            "weight_bytes": pinned.WEIGHT_BYTES,
            "nano_revision": pinned.NANO_COMMIT,
            "nano_repo_id": pinned.NANO_REPO_ID,
            "nano_metadata_sha256": pinned.NANO_FILES_SHA256,
            "nano_ref_sha256": sha(pinned.NANO_COMMIT.encode("ascii")),
            "pinned_packages": pinned.PINNED_PACKAGES,
        },
        "child_environment": child_environment(),
        "child_environment_sha256": sha(canonical(child_environment())),
        "installed_environment": installed_environment,
        "native_ffmpeg_runtime": native_runtime,
    }


def validate_freeze(freeze_path: Path, review_path: Path,
                    expected: dict[str, Any], key: bytes) -> tuple[dict[str, Any], dict[str, Any], str]:
    require(len(key) >= 32, "External freeze-review key missing or short")
    freeze, raw = _read_json(freeze_path)
    require(set(freeze) == {"schema", "status", "run_id", "expected", "caps", "source_review"}
            and freeze["schema"] == "voicechat_public_sample_smoke_freeze_v1"
            and freeze["status"] == "externally_frozen_sample_only"
            and isinstance(freeze["run_id"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}", freeze["run_id"])
            and freeze["run_id"] == Path(expected["paths"]["output_dir"]).name
            and freeze["expected"] == expected, "Sample-only freeze scope differs")
    caps = freeze["caps"]
    require(type(caps) is dict and set(caps) == {"wall_seconds", "vram_mib", "total_evidence_bytes"}
            and all(type(value) is int and value > 0 for value in caps.values())
            and caps["wall_seconds"] <= MAX_WALL_SECONDS
            and caps["vram_mib"] <= MAX_VRAM_MIB
            and MIN_EVIDENCE_BYTES <= caps["total_evidence_bytes"] <= MAX_EVIDENCE_BYTES,
            "Sample-only caps invalid or relaxed")
    source_review = freeze["source_review"]
    require(type(source_review) is dict and set(source_review) == {"path", "sha256"},
            "Independent source review absent")
    source_review_path = Path(source_review["path"])
    require(source_review_path.is_absolute()
            and hash_file(source_review_path, cap=MAX_FREEZE_BYTES)[0] == source_review["sha256"],
            "Independent source review changed")
    source_review_record, _ = _read_json(source_review_path)
    require(set(source_review_record) == {"schema", "decision", "source_sha256",
                                          "reviewer_id", "reviewed_utc"}
            and source_review_record["schema"] == "voicechat_public_sample_source_review_v1"
            and source_review_record["decision"] == "GO"
            and source_review_record["source_sha256"] == expected["source_sha256"]
            and isinstance(source_review_record["reviewer_id"], str)
            and source_review_record["reviewer_id"].strip()
            and _utc_record(source_review_record["reviewed_utc"]),
            "Independent source review is not GO for exact source")
    review, _ = _read_json(review_path)
    require(set(review) == {"schema", "status", "freeze_sha256", "source_review_sha256",
                            "reviewer_id", "reviewed_utc", "hmac_sha256"}
            and review["schema"] == "voicechat_public_sample_freeze_review_v1"
            and review["status"] == "independently_reviewed_executable_freeze"
            and review["freeze_sha256"] == sha(raw)
            and review["source_review_sha256"] == source_review["sha256"]
            and isinstance(review["reviewer_id"], str) and review["reviewer_id"].strip()
            and _utc_record(review["reviewed_utc"]),
            "External freeze review scope differs")
    unsigned = {k: v for k, v in review.items() if k != "hmac_sha256"}
    expected_mac = hmac.new(key, canonical(unsigned), hashlib.sha256).hexdigest()
    require(hmac.compare_digest(review["hmac_sha256"], expected_mac),
            "External freeze review signature differs")
    return freeze, review, sha(raw)


def sample_ready_preload(nemo_dir: Path, checkpoint_dir: Path,
                         nano_home: Path, gpu_policy: str,
                         checks: Any = pinned) -> dict[str, Any]:
    """Shared pinned checks, omitting only calendar/QA/controller/anchor checks."""
    require(os.environ.get("PYTHONPATH") == str(nemo_dir),
            "PYTHONPATH must select only the pinned NeMo tree")
    receipts = []
    for phase in ("ready", "preload"):
        started = time.monotonic()
        source = checks.check_source(nemo_dir, checkpoint_dir)
        sample = check_sample(nemo_dir)
        nano = checks.check_nano_cache(nano_home)
        nano_ref = check_exact_nano_ref(nano_home)
        runtime = checks.check_runtime(checkpoint_dir, gpu_policy)
        if phase == "ready":
            checks.check_pristine_modules_cache(nano_home)
            offline = checks.verify_nano_offline(nano_home)
        else:
            offline = checks.check_preload_environment(nano_home)
        dynamic = checks.check_generated_modules_cache(nano_home)
        checks.check_nano_cache(nano_home)
        receipts.append({"phase": phase, "completed_utc": utc(),
                         "elapsed_monotonic_seconds": time.monotonic() - started,
                         "source": source, "sample_sha256": sample, "nano": nano,
                         "nano_ref_sha256": nano_ref,
                         "runtime": runtime, "offline": offline, "dynamic": dynamic})
    stable_runtime = ("weight_bytes", "weight_sha256", "gpu_policy", "gpu",
                      "gpu_memory_mib", "pinned_packages_verified")
    require(receipts[0]["sample_sha256"] == receipts[1]["sample_sha256"]
            and receipts[0]["source"] == receipts[1]["source"]
            and receipts[0]["nano"] == receipts[1]["nano"]
            and receipts[0]["nano_ref_sha256"] == receipts[1]["nano_ref_sha256"]
            and all(receipts[0]["runtime"].get(key) == receipts[1]["runtime"].get(key)
                    for key in stable_runtime),
            "Sample-only ready/preload identity changed")
    return {"status": "sample_only_ready_preload_passed_no_model_work", "receipts": receipts}


def evidence_inventory(root: Path, cap: int) -> tuple[dict[str, dict[str, Any]], int]:
    require(root.is_dir() and not root.is_symlink(), "Evidence root is not an unlinked directory")
    total = 0
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Evidence link rejected")
        if path.is_dir():
            continue
        require(path.is_file(), "Evidence path is not a file")
        size = path.stat().st_size
        total += size
        require(total <= cap, "Total sample evidence cap exceeded")
        digest, stable_size = hash_file(path, cap=cap)
        require(stable_size == size, "Sample evidence file changed")
        files[str(path.relative_to(root))] = {"sha256": digest, "bytes": size}
    return files, total


def evidence_bytes(root: Path, cap: int) -> int:
    """Stat active output without reading files while the worker writes them."""
    require(root.is_dir() and not root.is_symlink(), "Evidence root is not an unlinked directory")
    total = 0
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Evidence link rejected")
        if path.is_dir():
            continue
        require(path.is_file(), "Evidence path is not a file")
        total += path.stat().st_size
        require(total <= cap, "Total sample evidence cap exceeded")
    return total


def terminal_reserve(cap: int) -> int:
    return min(1 << 20, max(1024, cap // 4))


def _write_evidence(root: Path, name: str, data: bytes, cap: int) -> None:
    _, used = evidence_inventory(root, cap)
    require(used + len(data) <= cap, "Sample evidence cap exceeded before write")
    with (root / name).open("xb") as stream:
        stream.write(data)


def _root_digest(data: bytes) -> bytes:
    return f"{sha(data)}  final_manifest.json\n".encode("ascii")


def _final_manifest(status: str, entries: list[dict[str, Any]],
                    observed_bytes: int,
                    removed_nonfiles: list[dict[str, str]] | None = None) -> bytes:
    return canonical({"schema": "voicechat_public_sample_final_inventory_v1",
                      "status": status, "observed_bytes_before_seal": observed_bytes,
                      "entries": entries,
                      "removed_nonfiles": removed_nonfiles or [],
                      "self_reference_boundary": "final_manifest.sha256 hashes this manifest; it cannot hash itself"})


def _file_entries(root: Path, original: dict[str, dict[str, Any]] | None = None,
                  target: int | None = None) -> tuple[list[dict[str, Any]], int,
                                                      dict[str, dict[str, Any]],
                                                      list[dict[str, str]]]:
    """Snapshot original bytes, then retain bounded prefixes after a failure."""
    removed_nonfiles: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            removed_nonfiles.append({"path": str(path.relative_to(root)),
                                     "kind": "symlink_removed",
                                     "target_sha256": sha(os.readlink(path).encode())})
            path.unlink()
        elif not path.is_file() and not path.is_dir():
            removed_nonfiles.append({"path": str(path.relative_to(root)),
                                     "kind": "special_file_removed"})
            path.unlink()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    require(len(paths) <= 512, "Too many sample artifacts to seal")
    observed = sum(path.stat().st_size for path in paths)
    records = {} if original is None else original
    for path in paths:
        name = str(path.relative_to(root))
        if name not in records:
            size = path.stat().st_size
            digest = hash_file(path, cap=MAX_EVIDENCE_BYTES)[0] if size <= MAX_EVIDENCE_BYTES else None
            records[name] = {"original_bytes": size, "original_sha256": digest,
                             "original_hash_status": ("complete" if digest is not None
                                                      else "omitted_over_1GiB_read_ceiling")}
    if target is not None:
        protected = {"scope.json", "admission.json", "process_start.json", "result.json"}
        ordered = sorted(paths, key=lambda p: (str(p.relative_to(root)) not in protected,
                                             str(p.relative_to(root))))
        remaining = target
        for path in ordered:
            size = path.stat().st_size
            name = str(path.relative_to(root))
            keep = min(size, max(0, remaining))
            require(name not in protected or keep == size,
                    "Protected receipt consumed terminal reserve")
            if keep != size:
                with path.open("r+b") as stream:
                    stream.truncate(keep)
            remaining -= keep
    entries = []
    for path in paths:
        name = str(path.relative_to(root))
        digest, size = hash_file(path, cap=MAX_EVIDENCE_BYTES)
        entries.append({"path": name, "original_bytes": records[name]["original_bytes"],
                        "original_sha256": records[name]["original_sha256"],
                        "original_hash_status": records[name]["original_hash_status"],
                        "retained_bytes": size, "retained_sha256": digest,
                        "truncated": size != records[name]["original_bytes"]})
    return entries, observed, records, removed_nonfiles


def _seal_success(root: Path, cap: int, result: dict[str, Any]) -> None:
    """Write result plus terminal manifest/hash with a prospective size check."""
    entries, observed, _, removed = _file_entries(root)
    require(not removed, "Non-file appeared during success seal")
    result_bytes = canonical(result)
    entries.append({"path": "result.json", "original_bytes": len(result_bytes),
                    "original_sha256": sha(result_bytes),
                    "retained_bytes": len(result_bytes),
                    "retained_sha256": sha(result_bytes), "truncated": False})
    manifest = _final_manifest("environment_smoke_only", entries,
                               observed + len(result_bytes))
    digest_file = _root_digest(manifest)
    require(observed + len(result_bytes) + len(manifest) + len(digest_file) <= cap,
            "Terminal success evidence exceeds total cap")
    _write_evidence(root, "result.json", result_bytes, cap)
    _write_evidence(root, "final_manifest.json", manifest, cap)
    _write_evidence(root, "final_manifest.sha256", digest_file, cap)


def _seal_failure(root: Path, cap: int, failure: dict[str, Any]) -> None:
    """Retain hashed prefixes and original-size/hash receipts within the cap."""
    target = cap - terminal_reserve(cap)
    for name in ("final_manifest.json", "final_manifest.sha256"):
        path = root / name
        if path.exists():
            path.unlink()
    entries, observed, original, removed = _file_entries(root, target=target)
    failure["observed_bytes_before_preservation"] = observed
    failure["partial_preservation"] = bool(removed) or any(row["truncated"] for row in entries)
    for _ in range(3):
        failure_bytes = canonical(failure)
        manifest_entries = [*entries, {
            "path": "failure.json", "original_bytes": len(failure_bytes),
            "original_sha256": sha(failure_bytes), "retained_bytes": len(failure_bytes),
            "retained_sha256": sha(failure_bytes), "truncated": False}]
        manifest = _final_manifest("failed_or_incomplete_environment_smoke",
                                   manifest_entries, observed, removed)
        digest_file = _root_digest(manifest)
        current = sum(row["retained_bytes"] for row in entries)
        excess = current + len(failure_bytes) + len(manifest) + len(digest_file) - cap
        if excess <= 0:
            _write_evidence(root, "failure.json", failure_bytes, cap)
            _write_evidence(root, "final_manifest.json", manifest, cap)
            _write_evidence(root, "final_manifest.sha256", digest_file, cap)
            return
        entries, _, _, _ = _file_entries(root, original=original,
                                        target=max(0, current - excess))
        failure["partial_preservation"] = True
    raise ValueError("Failure evidence cannot be sealed within total cap")


def _process_tree(root_pid: int) -> set[int]:
    """Linux /proc parent chain; the sample is restricted to the single GPU host."""
    parents = {}
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            parts = entry.read_text().rsplit(") ", 1)[1].split()
            parents[int(entry.parent.name)] = int(parts[1])
        except (OSError, ValueError, IndexError):
            continue
    result = {root_pid}
    changed = True
    while changed:
        before = len(result)
        result.update(pid for pid, parent in parents.items() if parent in result)
        changed = len(result) != before
    return result


def gpu_memory_mib(pid: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=True)
    relevant = _process_tree(pid)
    total = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [piece.strip() for piece in line.split(",")]
        require(len(fields) == 2, "GPU telemetry malformed")
        if int(fields[0]) in relevant:
            require(fields[1].isdigit(), "GPU memory telemetry unavailable")
            total += int(fields[1])
    return total


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _limit_file_size(cap: int) -> None:
    """Child-only per-file ceiling; aggregate files remain monitor bounded."""
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    limit = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))


def execute_official(command: list[str], output_dir: Path, cap: dict[str, int],
                     environment: dict[str, str],
                     telemetry: Callable[[int], int] = gpu_memory_mib) -> dict[str, Any]:
    """Start one fresh official process, stream logs, and enforce finite caps."""
    require(not (output_dir / "official_outputs").exists(), "Official output already exists")
    out_path = output_dir / "stdout.log"
    err_path = output_dir / "stderr.log"
    selector = selectors.DefaultSelector()
    peak = 0
    reason = None
    start = time.monotonic()
    started_utc = utc()
    with out_path.open("xb") as stdout, err_path.open("xb") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment,
                                   start_new_session=True, cwd=str(output_dir),
                                   preexec_fn=lambda: _limit_file_size(
                                       cap["total_evidence_bytes"] -
                                       terminal_reserve(cap["total_evidence_bytes"])))
        assert process.stdout is not None and process.stderr is not None
        try:
            _write_evidence(output_dir, "process_start.json", canonical({
                "process_id": process.pid, "started_utc": started_utc,
                "command": command, "environment_sha256": sha(canonical(environment)),
            }), cap["total_evidence_bytes"] - terminal_reserve(cap["total_evidence_bytes"]))
            selector.register(process.stdout, selectors.EVENT_READ, stdout)
            selector.register(process.stderr, selectors.EVENT_READ, stderr)
            while selector.get_map() or process.poll() is None:
                if time.monotonic() - start > cap["wall_seconds"]:
                    reason = "wall_time_cap"
                    break
                peak = max(peak, telemetry(process.pid))
                if peak > cap["vram_mib"]:
                    reason = "gpu_memory_cap"
                    break
                effective_cap = cap["total_evidence_bytes"] - terminal_reserve(cap["total_evidence_bytes"])
                evidence_bytes(output_dir, effective_cap)
                for key, _ in selector.select(timeout=0.2):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    used = evidence_bytes(output_dir, effective_cap)
                    if used + len(data) > effective_cap:
                        reason = "log_evidence_cap"
                        break
                    key.data.write(data)
                    key.data.flush()
                if reason:
                    break
        except BaseException:
            _kill_tree(process)
            raise
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
        if reason:
            _kill_tree(process)
        return {"returncode": process.wait(timeout=10), "process_id": process.pid,
                "started_utc": started_utc, "finished_utc": utc(),
                "elapsed_monotonic_seconds": time.monotonic() - start,
                "peak_vram_mib": peak, "cap_reason": reason,
                "logs_complete": reason is None}


def run_sample(*, nemo_dir: Path, checkpoint_dir: Path, nano_home: Path,
               output_dir: Path, freeze_path: Path, review_path: Path,
               gpu_policy: str, review_key: bytes,
               checks: Any = pinned,
               executor: Callable[..., dict[str, Any]] = execute_official) -> dict[str, Any]:
    """One attempt; injected checks/executor are only used in no-model tests."""
    # The HMAC credential is parent-only, including for the offline metadata
    # resolver that shared preflight starts before model inference.
    os.environ.pop("VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX", None)
    os.environ.pop("VOICECHAT_CONTROLLER_KEY_HEX", None)
    expected = expected_freeze_fields(nemo_dir, checkpoint_dir, nano_home,
                                      output_dir, gpu_policy)
    freeze, _review, freeze_sha = validate_freeze(freeze_path, review_path,
                                                  expected, review_key)
    caps = dict(freeze["caps"])
    output_dir.mkdir(parents=True, exist_ok=False)
    run: dict[str, Any] | None = None
    try:
        _write_evidence(output_dir, "scope.json", canonical({
            "schema": "voicechat_public_sample_smoke_scope_v1", "status": "attempt_consumed",
            "run_id": freeze["run_id"],
            "freeze_sha256": freeze_sha, "review_sha256": hash_file(review_path)[0],
            "started_utc": utc(), "command": expected["command"],
            "child_environment": expected["child_environment"], "caps": caps,
            "native_ffmpeg_runtime": expected["native_ffmpeg_runtime"],
        }), caps["total_evidence_bytes"] - terminal_reserve(caps["total_evidence_bytes"]))
        admission = sample_ready_preload(nemo_dir, checkpoint_dir, nano_home,
                                         gpu_policy, checks)
        require(expected_freeze_fields(nemo_dir, checkpoint_dir, nano_home,
                                       output_dir, gpu_policy) == expected,
                "Sample-only bound source/environment changed after preload")
        require(check_sample(nemo_dir) == admission["receipts"][1]["sample_sha256"]
                and check_exact_nano_ref(nano_home) ==
                admission["receipts"][1]["nano_ref_sha256"],
                "Pinned sample or Nano ref changed before process start")
        _write_evidence(output_dir, "admission.json", canonical(admission),
                        caps["total_evidence_bytes"] - terminal_reserve(caps["total_evidence_bytes"]))
        command = fixed_command(nemo_dir, checkpoint_dir, output_dir)
        require(command == expected["command"], "Official sample command changed")
        environment = child_environment()
        require(environment == expected["child_environment"],
                "Frozen child environment changed before process start")
        run = executor(command, output_dir, caps, environment)
        files, total = evidence_inventory(
            output_dir, caps["total_evidence_bytes"] - terminal_reserve(caps["total_evidence_bytes"]))
        require(run.get("returncode") == 0 and run.get("cap_reason") is None
                and run.get("logs_complete") is True,
                "Official sample process failed or exceeded a cap")
        require(type(run.get("peak_vram_mib")) is int
                and 0 <= run["peak_vram_mib"] <= caps["vram_mib"]
                and isinstance(run.get("elapsed_monotonic_seconds"), (int, float))
                and run["elapsed_monotonic_seconds"] <= caps["wall_seconds"],
                "Official sample resource receipt invalid")
        require(any(name.startswith("official_outputs/") for name in files),
                "Official sample produced no transport artifact")
        result = {"status": "environment_smoke_only", "freeze_sha256": freeze_sha,
                  "returncode": 0, "run": run, "artifact_sha256": files,
                  "total_evidence_bytes_before_result": total,
                  "finished_utc": utc(), "no_calendar_or_parity_claim": True}
        _seal_success(output_dir, caps["total_evidence_bytes"], result)
        return result
    except BaseException as error:
        failure = {"status": "failed_or_incomplete_environment_smoke",
                   "freeze_sha256": freeze_sha, "error_type": type(error).__name__,
                   "error": str(error)[:512], "error_truncated": len(str(error)) > 512,
                   "finished_utc": utc(), "run": run}
        _seal_failure(output_dir, caps["total_evidence_bytes"], failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--nano-hf-home", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-review", type=Path, required=True)
    parser.add_argument("--gpu-policy", choices=("h100", "a100-80gb"), default="h100")
    args = parser.parse_args()
    key_hex = os.environ.get("VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX", "")
    try:
        key = bytes.fromhex(key_hex)
        result = run_sample(nemo_dir=args.nemo_dir, checkpoint_dir=args.checkpoint_dir,
                            nano_home=args.nano_hf_home, output_dir=args.output_dir,
                            freeze_path=args.freeze, review_path=args.freeze_review,
                            gpu_policy=args.gpu_policy, review_key=key)
    except BaseException as error:
        print(json.dumps({"status": "failed_closed_sample_only",
                          "error_type": type(error).__name__, "error": str(error)}),
              file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
