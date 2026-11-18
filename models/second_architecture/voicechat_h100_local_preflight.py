"""Offline-only source, checkpoint, fixture, and GPU readiness checks.

This script never downloads weights, imports NeMo, or calls model inference.
Run --self-test locally. Run --phase source after staging code/config, and
--phase ready only on a prepared H100 or explicitly selected A100 80GB host.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath


NEMO_COMMIT = "097dfe9e2f55baf653b83035868bdc89849f1b47"
VOICECHAT_COMMIT = "443794ea956ef0065f001967ffd00e77f519cb39"
NANO_COMMIT = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
NANO_REPO_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
NANO_CACHE_REPO = "models--nvidia--NVIDIA-Nemotron-Nano-9B-v2"
NANO_FILES_SHA256 = {
    "config.json": "642dc554a725ea216be55c3018d0a6f3dc9963b24779c3cc97f5a0ea21e9dcc3",
    "configuration_nemotron_h.py": "18186204622431c336a7c2813a80f0d37937dde962b8aebddf24caf5bffa03f7",
    "modeling_nemotron_h.py": "9c831864760ab5d62787040fbd05039e6dfec4b6239d1fcc96ec635efc114734",
    "tokenizer_config.json": "a3e4d48a0f8b4ce6fd746464199299cbdefe6ee202e34048940da2d9838a9aa6",
    "tokenizer.json": "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f",
    "special_tokens_map.json": "2a4d2e7403546286e5d75f5b6b3c197490be67fb1e2118e5c60ad5c26e6668b1",
}
SOURCE_SHA256 = {
    "nemo/collections/speechlm2/models/duplex_stt_model.py": "415e1e01584bab96d9c059d5b0fe9162adaf19677ef43e565c3afad6e073e74b",
    "nemo/collections/speechlm2/models/nemotron_voicechat.py": "fe11dcdde3c31d7144469dab0c6ec9629ff8f7e4dbf881d8d1db258243a67f80",
    "nemo/collections/speechlm2/inference/utils/offline_voicechat.py": "ebde701baec12a014443eac36262a6af7a7a62b3e052f1c4a387db6aaad8da44",
    "examples/speechlm2/function_calling/template.jinja": "8b4f88e9f0052c0dfc19896b9d6a84821f68f84a1049406908e1f664155edcc9",
    "examples/speechlm2/offline_voicechat_fc_infer.py": "797a80d9cef654550514e045a9388ca689a25dc9bbf70436eb117d249a17644f",
}
CONFIG_SHA256 = "2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5"
WEIGHT_SHA256 = "d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b"
WEIGHT_BYTES = 44_382_749_892
SCRIPT_IDS = ("A", "B", "C", "D", "A_literal", "B_literal")
PINNED_PACKAGES = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    "transformers": "4.56.0",
    "tokenizers": "0.22.0",
    "lhotse": "1.32.2",
    "huggingface-hub": "0.34.4",
    "hf-xet": "1.1.9",
    "torchcodec": "0.10.0",
    "causal-conv1d": "1.6.2.post1",
    "mamba-ssm": "2.3.2.post1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def check_file(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing file: {path}")
    require(sha256(path) == expected, f"SHA-256 mismatch: {path}")


def resolve_fixture_path(base: Path, raw: str, label: str) -> Path:
    """Accept only portable, normalized paths rooted inside the fixture tree."""
    require(isinstance(raw, str) and raw and "\\" not in raw,
            f"invalid relative fixture path: {label}")
    relative = PurePosixPath(raw)
    require(not relative.is_absolute() and not PureWindowsPath(raw).drive and
            raw == relative.as_posix() and
            all(segment not in (".", "..") for segment in raw.split("/")),
            f"unnormalized or absolute fixture path: {label}")
    local = (base / raw).resolve()
    require(local.is_relative_to(base), f"fixture escapes manifest directory: {label}")
    return local


def normalized_transcript(value: str) -> str:
    require(isinstance(value, str) and value.strip(), "missing transcript")
    normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))
    require(bool(normalized), "transcript has no words")
    return normalized


def check_qa_receipt(base: Path, descriptor: dict, *, kind: str, wav_sha256: str,
                     script_id: str, voice_id: str, transcript: str) -> tuple[Path, str]:
    require(isinstance(descriptor, dict), f"missing {kind} receipt descriptor")
    local = resolve_fixture_path(base, descriptor["path"], f"{kind} receipt")
    check_file(local, descriptor["sha256"])
    receipt = json.loads(local.read_text())
    require(receipt.get("schema") == "voicechat_wav_qa_receipt_v1" and
            receipt.get("kind") == kind and receipt.get("verdict") == "pass",
            f"invalid {kind} receipt verdict/schema")
    require(receipt.get("wav_sha256") == wav_sha256 and
            receipt.get("script_id") == script_id and receipt.get("voice_id") == voice_id,
            f"{kind} receipt not bound to WAV/case")
    require(normalized_transcript(receipt.get("reference_transcript")) ==
            normalized_transcript(transcript) and
            normalized_transcript(receipt.get("observed_transcript")) ==
            normalized_transcript(transcript),
            f"{kind} receipt transcript mismatch")
    require(isinstance(receipt.get("reviewed_utc"), str) and receipt["reviewed_utc"].strip(),
            f"missing {kind} review time")
    reviewed_at = datetime.fromisoformat(receipt["reviewed_utc"].replace("Z", "+00:00"))
    require(reviewed_at.utcoffset() == timedelta(0), f"{kind} review time must be UTC")
    if kind == "listening":
        require(isinstance(receipt.get("reviewer_id"), str) and receipt["reviewer_id"].strip(),
                "missing human listener identity")
    else:
        require(isinstance(receipt.get("model_id"), str) and receipt["model_id"].strip() and
                "voicechat" not in receipt["model_id"].casefold() and
                isinstance(receipt.get("model_revision"), str) and
                receipt["model_revision"].strip() and
                isinstance(receipt.get("settings"), dict),
                "independent ASR model/revision/settings missing")
    return local, descriptor["sha256"]


EGG_INFO_METADATA = frozenset({
    "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
    "namespace_packages.txt", "not-zip-safe", "requires.txt", "top_level.txt", "zip-safe",
})


def allowed_generated_metadata(relative: Path) -> bool:
    """Allow only setuptools metadata, never ignored importable code/cache."""
    parts = relative.parts
    return (len(parts) == 2 and parts[0].endswith(".egg-info") and
            parts[1] in EGG_INFO_METADATA)


def check_clean_worktree(source_root: Path) -> int:
    """Reject source drift; allow narrowly specified ignored editable-install metadata."""
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain=v1",
         "--untracked-files=all", "-z"],
        check=True, capture_output=True,
    ).stdout
    require(not status, f"NeMo worktree has tracked/untracked drift: {status[:160]!r}")
    ignored = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "--others", "--ignored",
         "--exclude-standard", "-z"],
        check=True, capture_output=True,
    ).stdout
    allowed_count = 0
    for encoded in ignored.split(b"\x00"):
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        local = source_root / relative
        require(allowed_generated_metadata(relative) and not local.is_symlink() and
                local.is_file(), f"NeMo ignored source/artifact not allowed: {relative}")
        allowed_count += 1
    return allowed_count


def check_source(nemo_dir: Path, checkpoint_dir: Path) -> dict:
    require(nemo_dir.is_dir() and checkpoint_dir.is_dir(), "source/checkpoint directory missing")
    source_root = nemo_dir.resolve()
    git_root = Path(subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()).resolve()
    require(git_root == source_root, "NeMo directory is not the Git worktree root")
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(head == NEMO_COMMIT, f"NeMo HEAD drift: {head}")
    generated_metadata_count = check_clean_worktree(source_root)
    for relative, expected in SOURCE_SHA256.items():
        check_file(source_root / relative, expected)
    config_path = checkpoint_dir / "config.json"
    check_file(config_path, CONFIG_SHA256)
    config = json.loads(config_path.read_text())
    stt = config["model"]["stt"]["model"]
    require(stt["use_function_head"] is True, "function head disabled")
    require(stt["duplex_function_channel_weight"] == 2.0, "function channel weight drift")
    require(stt["pretrained_llm"] == NANO_REPO_ID, "backbone drift")
    speech_generation = config["model"]["speech_generation"]["model"]
    require(speech_generation["pretrained_lm_name"] == NANO_REPO_ID and
            speech_generation["tts_config"]["cas_config"]["pretrained_tokenizer_name"] == NANO_REPO_ID,
            "speech-generation Nano repository ID drift")
    return {"nemo_commit": head, "nemo_worktree": "clean_except_setuptools_metadata",
            "nemo_generated_metadata_files_allowed": generated_metadata_count,
            "voicechat_commit": VOICECHAT_COMMIT,
            "source_files_verified": len(SOURCE_SHA256), "config_sha256": CONFIG_SHA256}


def check_nano_cache(hf_home: Path, *, expected_files: dict[str, str] | None = None,
                     revision: str = NANO_COMMIT) -> dict:
    """Verify a dedicated offline HF cache without importing or loading a model."""
    require(hf_home.is_absolute() and hf_home.is_dir(),
            "--nano-hf-home must be an existing absolute directory")
    home = hf_home.resolve()
    hub = home / "hub"
    repo = hub / NANO_CACHE_REPO
    snapshot = repo / "snapshots" / revision
    require(hub.is_dir() and repo.is_dir() and snapshot.is_dir(),
            "pinned Nano HF cache/snapshot missing")
    require({p.name for p in hub.iterdir()} <= {NANO_CACHE_REPO, ".locks"},
            "Nano HF_HOME hub is not dedicated to the pinned repository")
    require({p.name for p in repo.iterdir()} == {"refs", "snapshots", "blobs"},
            "Nano repository cache root has extra or missing entries")
    require({p.name for p in (repo / "snapshots").iterdir()} == {revision},
            "Nano cache has an unexpected snapshot revision")
    refs = repo / "refs"
    main_ref = refs / "main"
    require(refs.is_dir() and {p.name for p in refs.iterdir()} == {"main"} and
            main_ref.is_file() and not main_ref.is_symlink() and
            main_ref.read_text().strip() == revision,
            "Nano refs/main does not select the pinned revision")
    expected = NANO_FILES_SHA256 if expected_files is None else expected_files
    require({p.name for p in snapshot.iterdir()} == set(expected),
            "Nano snapshot file set differs from the six-file pin")
    referenced_blobs: set[Path] = set()
    for filename, digest in expected.items():
        path = snapshot / filename
        require(path.resolve().is_relative_to(repo),
                f"Nano cache file resolves outside its repository: {filename}")
        if path.is_symlink():
            referenced_blobs.add(path.resolve())
        check_file(path, digest)
    blobs = repo / "blobs"
    actual_blobs = {item.resolve() for item in blobs.iterdir()} if blobs.is_dir() else set()
    require(actual_blobs == referenced_blobs,
            "Nano cache contains an unreferenced or missing blob (including possible weights)")
    return {"nano_repo_id": NANO_REPO_ID, "nano_revision": revision,
            "nano_hf_home": str(home), "nano_refs_main": revision,
            "nano_snapshot": str(snapshot), "nano_files_verified": len(expected)}


def check_pristine_modules_cache(hf_home: Path) -> None:
    """Custom code must not be preseeded before Transformers copies pinned bytes."""
    modules = hf_home / "modules"
    require(not modules.is_symlink() and
            (not modules.exists() or (modules.is_dir() and not any(modules.iterdir()))),
            "Nano Transformers modules cache must be absent or empty before resolution")


def check_generated_modules_cache(hf_home: Path, *, expected_files: dict[str, str] | None = None,
                                  revision: str = NANO_COMMIT) -> dict:
    """Verify the exact executable custom Python copied by Transformers."""
    modules = hf_home / "modules"
    module_root = (modules / "transformers_modules" / "nvidia" /
                   "NVIDIA-Nemotron-Nano-9B-v2" / revision)
    expected = NANO_FILES_SHA256 if expected_files is None else expected_files
    dynamic_files = {module_root / name: expected[name] for name in
                     ("configuration_nemotron_h.py", "modeling_nemotron_h.py")}
    require(modules.is_dir() and not modules.is_symlink() and module_root.is_dir(),
            "pinned Nano dynamic-module directory missing or redirected")
    allowed_dirs = {modules}
    cursor = module_root
    while cursor != modules:
        allowed_dirs.add(cursor)
        cursor = cursor.parent
    for item in modules.rglob("*"):
        require(not item.is_symlink(), f"symlink in Nano modules cache: {item}")
        if item.is_dir():
            require(item in allowed_dirs or
                    (item.name == "__pycache__" and item.parent in allowed_dirs),
                    f"unexpected Nano modules directory: {item}")
        elif item.name == "__init__.py":
            require(item.parent in allowed_dirs and item.read_bytes() == b"",
                    f"unexpected executable Nano module initializer: {item}")
        elif item.suffix == ".pyc":
            require(item.parent.name == "__pycache__" and
                    item.parent.parent in allowed_dirs,
                    f"unexpected compiled Nano module: {item}")
        else:
            require(item in dynamic_files, f"unexpected Nano modules file: {item}")
            check_file(item, dynamic_files[item])
    for path, digest in dynamic_files.items():
        check_file(path, digest)
    return {"nano_dynamic_module_files_verified": len(dynamic_files),
            "nano_modules_cache": str(modules)}


def verify_nano_offline(hf_home: Path) -> dict:
    """Resolve the unchanged repo ID offline using pinned Transformers; no weights/model instance."""
    environment = os.environ.copy()
    environment.update(HF_HOME=str(hf_home), HF_HUB_CACHE=str(hf_home / "hub"),
                       HF_MODULES_CACHE=str(hf_home / "modules"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1")
    environment.pop("TRANSFORMERS_CACHE", None)
    program = f"""
import json
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
repo = {NANO_REPO_ID!r}
config = AutoConfig.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True, local_files_only=True)
model_class = get_class_from_dynamic_module(
    config.auto_map["AutoModelForCausalLM"], repo, local_files_only=True)
assert config.model_type == "nemotron_h"
assert config.hidden_size == 4480 and config.num_hidden_layers == 56
assert tokenizer.is_fast and len(tokenizer) == 131072 and tokenizer.bos_token_id == 1
print("NANO_OFFLINE_RECEIPT=" + json.dumps({{
    "config_class": config.__class__.__name__,
    "model_class": model_class.__name__,
    "tokenizer_class": tokenizer.__class__.__name__,
    "tokenizer_length": len(tokenizer),
    "bos_token_id": tokenizer.bos_token_id,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], env=environment, check=False,
        capture_output=True, text=True, timeout=180,
    )
    require(completed.returncode == 0,
            f"Nano offline metadata/class resolution failed: {completed.stderr[-2000:]}")
    receipts = [line.removeprefix("NANO_OFFLINE_RECEIPT=")
                for line in completed.stdout.splitlines()
                if line.startswith("NANO_OFFLINE_RECEIPT=")]
    require(len(receipts) == 1, "Nano offline resolver did not return one receipt")
    return {"nano_offline_resolver": json.loads(receipts[0]),
            "nano_offline_env": "dedicated_HF_HOME_HF_HUB_OFFLINE_TRANSFORMERS_OFFLINE"}


def check_fixtures(path: Path) -> dict:
    """Require all 12 TTS cases and exactly the declared optional human cases."""
    require(path.is_file(), f"fixture manifest missing: {path}")
    spec = json.loads(path.read_text())
    require(spec.get("status") == "frozen_voicechat_dual_head_fixture", "fixture not frozen")
    base = path.parent.resolve()
    for key in ("system_prompt", "tool_schema", "stub_calendar"):
        item = spec[key]
        local = resolve_fixture_path(base, item["path"], key)
        check_file(local, item["sha256"])
    voices = spec["tts_voice_ids"]
    require(isinstance(voices, list) and len(voices) == 2 and
            len(set(voices)) == 2 and all(isinstance(v, str) and v for v in voices),
            "expected exactly two distinct TTS voice IDs")
    human_scripts = spec["human_script_ids"]
    require(isinstance(human_scripts, list) and len(human_scripts) == len(set(human_scripts)) and
            set(human_scripts) <= set(SCRIPT_IDS), "invalid optional human script list")
    human_voice = spec.get("human_voice_id")
    require((not human_scripts and human_voice is None) or
            (bool(human_scripts) and isinstance(human_voice, str) and human_voice and
             human_voice not in voices), "human voice ID/list mismatch")
    expected = {(script, voice) for script in SCRIPT_IDS for voice in voices}
    expected.update((script, human_voice) for script in human_scripts)
    trailing = spec["trailing_silence_samples"]
    require(isinstance(trailing, int) and trailing >= 16_000,
            "freeze at least one second of 16-kHz trailing digital silence")
    wavs = spec["wavs"]
    require(len(wavs) == len(expected), "WAV count differs from declared cases")
    seen = set()
    wav_paths: set[Path] = set()
    wav_hashes: set[str] = set()
    qa_paths: set[Path] = set()
    qa_hashes: set[str] = set()
    for item in wavs:
        key = (item["script_id"], item["voice_id"])
        require(key not in seen, f"duplicate WAV case: {key}")
        seen.add(key)
        local = resolve_fixture_path(base, item["path"], f"WAV {key}")
        require(local not in wav_paths, f"WAV path reused: {local}")
        require(item["sha256"] not in wav_hashes, f"WAV bytes reused: {key}")
        wav_paths.add(local)
        wav_hashes.add(item["sha256"])
        check_file(local, item["sha256"])
        with wave.open(str(local), "rb") as wav:
            require((wav.getnchannels(), wav.getframerate(), wav.getsampwidth(),
                     wav.getcomptype()) == (1, 16000, 2, "NONE"),
                    f"WAV must be 16-kHz mono PCM16: {key}")
            require(wav.getnframes() > trailing, f"WAV lacks speech before trailing silence: {key}")
            wav.setpos(wav.getnframes() - trailing)
            require(wav.readframes(trailing) == b"\x00" * (2 * trailing),
                    f"frozen trailing silence is not exact digital zero: {key}")
        require(isinstance(item.get("transcript"), str) and item["transcript"].strip(),
                f"missing frozen transcript: {key}")
        for kind, field in (("listening", "listening_receipt"),
                            ("independent_asr", "asr_receipt")):
            qa_path, qa_hash = check_qa_receipt(
                base, item.get(field), kind=kind, wav_sha256=item["sha256"],
                script_id=key[0], voice_id=key[1], transcript=item["transcript"])
            require(qa_path not in qa_paths and qa_hash not in qa_hashes,
                    f"QA receipt reused: {key}, {kind}")
            qa_paths.add(qa_path)
            qa_hashes.add(qa_hash)
    require(seen == expected, f"WAV case matrix differs: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    return {"fixture_manifest_sha256": sha256(path), "wav_fixtures_verified": len(wavs),
            "tts_wavs": 12, "human_wavs": len(human_scripts),
            "qa_receipts_verified": len(qa_paths),
            "trailing_silence_samples": trailing}


def check_runtime(checkpoint_dir: Path, gpu_policy: str) -> dict:
    require(sys.version_info[:2] == (3, 12), "Python 3.12 required")
    versions = {}
    for package, expected in PINNED_PACKAGES.items():
        actual = importlib.metadata.version(package)
        require(actual == expected, f"{package} version {actual}, expected {expected}")
        versions[package] = actual
    weight = checkpoint_dir / "model.safetensors"
    require(weight.is_file() and weight.stat().st_size == WEIGHT_BYTES, "checkpoint size/missing")
    check_file(weight, WEIGHT_SHA256)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    require(len(gpu) == 1, "expected exactly one GPU")
    name, memory = [part.strip() for part in gpu[0].split(",", 1)]
    expected_name = "H100" if gpu_policy == "h100" else "A100"
    require(expected_name in name and int(memory) >= 75_000,
            f"{expected_name} >=75,000 MiB required; got {gpu[0]}")
    free_bytes = shutil.disk_usage(checkpoint_dir).free
    require(free_bytes >= 60 * 1024**3, "less than 60 GiB disk free after checkpoint staging")
    return {"weight_bytes": WEIGHT_BYTES, "weight_sha256": WEIGHT_SHA256,
            "gpu_policy": gpu_policy, "gpu": name, "gpu_memory_mib": int(memory),
            "disk_free_gib": round(free_bytes / 1024**3, 1),
            "pinned_packages_verified": versions}


def self_test() -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sample = root / "sample.txt"
        sample.write_bytes(b"voicechat-local-preflight")
        check_file(sample, "8e501842ca3e72f910942093f8bfafa2e123b423757765d625ebc3590aaf6093")
        try:
            check_file(sample, "0" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("hash drift did not fail closed")
        fixture = {"status": "frozen_voicechat_dual_head_fixture",
                   "tts_voice_ids": ["tts_one", "tts_two"], "human_voice_id": None,
                   "human_script_ids": [], "trailing_silence_samples": 16_000,
                   "wavs": []}
        for key in ("system_prompt", "tool_schema", "stub_calendar"):
            item = root / f"{key}.txt"
            item.write_text(key)
            fixture[key] = {"path": item.name, "sha256": sha256(item)}
        def add_wav(script: str, voice: str) -> None:
            path = root / f"{script}_{voice}.wav"
            sample_id = len(fixture["wavs"]) + 1
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(sample_id.to_bytes(2, "little") + b"\x00\x00" * 16_000)
            item = {"script_id": script, "voice_id": voice,
                    "path": path.name, "sha256": sha256(path), "transcript": "Test speech"}
            for kind, field in (("listening", "listening_receipt"),
                                ("independent_asr", "asr_receipt")):
                receipt = {"schema": "voicechat_wav_qa_receipt_v1", "kind": kind,
                           "verdict": "pass", "wav_sha256": item["sha256"],
                           "script_id": script, "voice_id": voice,
                           "reference_transcript": item["transcript"],
                           "observed_transcript": item["transcript"],
                           "reviewed_utc": "2026-11-18T00:00:00Z"}
                if kind == "listening":
                    receipt["reviewer_id"] = "synthetic-self-test-listener"
                else:
                    receipt.update(model_id="synthetic-independent-asr",
                                   model_revision="self-test", settings={})
                receipt_path = root / f"{script}_{voice}_{kind}.json"
                receipt_path.write_text(json.dumps(receipt))
                item[field] = {"path": receipt_path.name,
                               "sha256": sha256(receipt_path)}
            fixture["wavs"].append(item)

        def expect_fixture_failure(mutated: dict, label: str) -> None:
            manifest.write_text(json.dumps(mutated))
            try:
                check_fixtures(manifest)
            except (ValueError, KeyError, TypeError):
                pass
            else:
                raise AssertionError(f"{label} did not fail closed")
        for script in SCRIPT_IDS:
            for voice in fixture["tts_voice_ids"]:
                add_wav(script, voice)
        manifest = root / "FROZEN_FIXTURE.json"
        manifest.write_text(json.dumps(fixture))
        require(check_fixtures(manifest)["qa_receipts_verified"] == 24,
                "12-case receipt self-test failed")
        add_wav("A", "human_one")
        fixture["human_voice_id"] = "human_one"
        fixture["human_script_ids"] = ["A"]
        manifest.write_text(json.dumps(fixture))
        require(check_fixtures(manifest)["qa_receipts_verified"] == 26,
                "human-extension receipt self-test failed")

        missing = copy.deepcopy(fixture)
        missing["wavs"].pop(0)
        expect_fixture_failure(missing, "missing mandatory TTS case")
        duplicate_path = copy.deepcopy(fixture)
        duplicate_path["wavs"][1]["path"] = duplicate_path["wavs"][0]["path"]
        duplicate_path["wavs"][1]["sha256"] = duplicate_path["wavs"][0]["sha256"]
        expect_fixture_failure(duplicate_path, "reused WAV path")
        duplicate_bytes = copy.deepcopy(fixture)
        second_wav = root / duplicate_bytes["wavs"][1]["path"]
        original_second_wav = second_wav.read_bytes()
        second_wav.write_bytes((root / duplicate_bytes["wavs"][0]["path"]).read_bytes())
        duplicate_bytes["wavs"][1]["sha256"] = sha256(second_wav)
        expect_fixture_failure(duplicate_bytes, "reused WAV bytes")
        second_wav.write_bytes(original_second_wav)
        absolute = copy.deepcopy(fixture)
        absolute["system_prompt"]["path"] = str((root / absolute["system_prompt"]["path"]).resolve())
        expect_fixture_failure(absolute, "absolute prompt path")
        absolute_wav = copy.deepcopy(fixture)
        absolute_wav["wavs"][0]["path"] = str((root / absolute_wav["wavs"][0]["path"]).resolve())
        expect_fixture_failure(absolute_wav, "absolute WAV path")
        for bad_path in ("./system_prompt.txt", "nested/../system_prompt.txt",
                         "nested//system_prompt.txt", "C:/system_prompt.txt"):
            unnormalized = copy.deepcopy(fixture)
            unnormalized["system_prompt"]["path"] = bad_path
            expect_fixture_failure(unnormalized, f"unnormalized path {bad_path}")
        no_qa = copy.deepcopy(fixture)
        no_qa["wavs"][0].pop("listening_receipt")
        no_qa["wavs"][0].update(listened=True, independent_asr_checked=True)
        expect_fixture_failure(no_qa, "self-asserted QA flags")
        tampered_qa = copy.deepcopy(fixture)
        qa_path = root / tampered_qa["wavs"][0]["asr_receipt"]["path"]
        qa_path.write_text(qa_path.read_text() + " ")
        expect_fixture_failure(tampered_qa, "tampered ASR receipt")
        wrong_binding = copy.deepcopy(fixture)
        qa_data = json.loads(qa_path.read_text())
        qa_data["wav_sha256"] = "0" * 64
        qa_path.write_text(json.dumps(qa_data))
        wrong_binding["wavs"][0]["asr_receipt"]["sha256"] = sha256(qa_path)
        expect_fixture_failure(wrong_binding, "ASR receipt bound to wrong WAV")

        source = root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        (source / ".gitignore").write_text(
            "ignored.py\nignored.json\nignored.jinja\n*.egg-info/\n"
            "__pycache__/\nbuild/\n")
        tracked = source / "tracked.py"
        tracked.write_text("x = 1\n")
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "-c", "user.name=Preflight",
                        "-c", "user.email=preflight@example.invalid", "commit", "-qm", "base"],
                       check=True)
        require(check_clean_worktree(source) == 0, "clean source self-test failed")
        egg_info = source / "nemo_toolkit.egg-info"
        egg_info.mkdir()
        for name in ("PKG-INFO", "SOURCES.txt"):
            (egg_info / name).write_text("generated setuptools metadata\n")
        require(check_clean_worktree(source) == 2,
                "editable-install metadata should be allowed")
        for label, dirty_file in (("modified", tracked), ("untracked", source / "extra.py"),
                                  ("ignored Python", source / "ignored.py"),
                                  ("ignored config", source / "ignored.json"),
                                  ("ignored template", source / "ignored.jinja"),
                                  ("ignored pycache", source / "__pycache__" / "module.pyc"),
                                  ("ignored build code", source / "build" / "module.py"),
                                  ("egg-info code", egg_info / "evil.py")):
            dirty_file.parent.mkdir(exist_ok=True)
            dirty_file.write_text("x = 2\n")
            try:
                check_clean_worktree(source)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{label} NeMo source did not fail closed")
            if dirty_file == tracked:
                tracked.write_text("x = 1\n")
            else:
                dirty_file.unlink()

        nano_home = root / "nano_hf_home"
        nano_repo = nano_home / "hub" / NANO_CACHE_REPO
        nano_ref = nano_repo / "refs" / "main"
        nano_snapshot = nano_repo / "snapshots" / NANO_COMMIT
        nano_ref.parent.mkdir(parents=True)
        nano_snapshot.mkdir(parents=True)
        (nano_repo / "blobs").mkdir()
        nano_ref.write_text(NANO_COMMIT + "\n")
        synthetic_hashes = {}
        for filename in NANO_FILES_SHA256:
            artifact = nano_snapshot / filename
            artifact.write_bytes(("synthetic " + filename).encode())
            synthetic_hashes[filename] = sha256(artifact)

        def expect_nano_failure(label: str) -> None:
            try:
                check_nano_cache(nano_home, expected_files=synthetic_hashes)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{label} Nano cache did not fail closed")

        require(check_nano_cache(nano_home, expected_files=synthetic_hashes)
                ["nano_files_verified"] == 6, "six-file Nano cache self-test failed")
        extra_root_file = nano_repo / "model.safetensors"
        extra_root_file.write_bytes(b"unexpected weight")
        expect_nano_failure("Nano repository root weight")
        extra_root_file.unlink()
        extra_root_dir = nano_repo / "other"
        extra_root_dir.mkdir()
        expect_nano_failure("Nano repository root directory")
        extra_root_dir.rmdir()
        nano_ref.write_text("0" * 40 + "\n")
        expect_nano_failure("wrong refs/main")
        nano_ref.write_text(NANO_COMMIT + "\n")
        token_config = nano_snapshot / "tokenizer_config.json"
        token_config_bytes = token_config.read_bytes()
        token_config.unlink()
        expect_nano_failure("missing Nano tokenizer config")
        token_config.write_bytes(token_config_bytes)
        token_config.write_bytes(b"changed")
        expect_nano_failure("changed Nano tokenizer config")
        token_config.write_bytes(token_config_bytes)
        extra_snapshot = nano_repo / "snapshots" / ("1" * 40)
        extra_snapshot.mkdir()
        expect_nano_failure("unexpected Nano snapshot")
        extra_snapshot.rmdir()
        outside = root / "outside-config.json"
        outside.write_bytes((nano_snapshot / "config.json").read_bytes())
        config_file = nano_snapshot / "config.json"
        config_file.unlink()
        config_file.symlink_to(outside)
        expect_nano_failure("Nano snapshot symlink escape")
        config_file.unlink()
        blobs = nano_repo / "blobs"
        config_blob = blobs / "synthetic-config-blob"
        config_blob.write_bytes(outside.read_bytes())
        config_file.symlink_to(Path("../../blobs") / config_blob.name)
        require(check_nano_cache(nano_home, expected_files=synthetic_hashes)
                ["nano_files_verified"] == 6, "normal HF blob symlink self-test failed")
        extra_blob = blobs / "unreferenced-weight-like-blob"
        extra_blob.write_bytes(b"unexpected")
        expect_nano_failure("unreferenced Nano blob")
        extra_blob.unlink()

        check_pristine_modules_cache(nano_home)
        modules = nano_home / "modules"
        module_root = (modules / "transformers_modules" / "nvidia" /
                       "NVIDIA-Nemotron-Nano-9B-v2" / NANO_COMMIT)
        module_root.mkdir(parents=True)
        seeded = module_root / "modeling_nemotron_h.py"
        seeded.write_bytes(b"unverified preseeded code")
        try:
            check_pristine_modules_cache(nano_home)
        except ValueError:
            pass
        else:
            raise AssertionError("preseeded Nano dynamic code did not fail closed")
        shutil.rmtree(modules)
        check_pristine_modules_cache(nano_home)
        module_root.mkdir(parents=True)
        for name in ("configuration_nemotron_h.py", "modeling_nemotron_h.py"):
            (module_root / name).write_bytes((nano_snapshot / name).read_bytes())
        require(check_generated_modules_cache(nano_home, expected_files=synthetic_hashes)
                ["nano_dynamic_module_files_verified"] == 2,
                "copied Nano custom-code hash self-test failed")
        (module_root / "extra.py").write_bytes(b"unexpected executable code")
        try:
            check_generated_modules_cache(nano_home, expected_files=synthetic_hashes)
        except ValueError:
            pass
        else:
            raise AssertionError("extra Nano dynamic code did not fail closed")
        (module_root / "extra.py").unlink()
        (module_root / "modeling_nemotron_h.py").write_bytes(b"changed custom code")
        try:
            check_generated_modules_cache(nano_home, expected_files=synthetic_hashes)
        except ValueError:
            pass
        else:
            raise AssertionError("changed Nano dynamic code did not fail closed")
    return {"status": "self_test_passed_no_model_or_network"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("source", "ready"))
    parser.add_argument("--nemo-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--nano-hf-home", type=Path,
                        help="dedicated Hugging Face cache containing the six pinned Nano metadata files")
    parser.add_argument("--gpu-policy", choices=("h100", "a100-80gb"), default="h100")
    args = parser.parse_args()
    try:
        if args.self_test:
            require(args.phase is None, "--self-test excludes --phase")
            result = self_test()
        else:
            require(args.phase is not None and args.nemo_dir and args.checkpoint_dir,
                    "--phase requires --nemo-dir and --checkpoint-dir")
            result = {"status": "local_source_preflight_passed_no_model_or_network"}
            result.update(check_source(args.nemo_dir, args.checkpoint_dir))
            if args.phase == "ready":
                require(args.fixture_manifest is not None, "--ready requires --fixture-manifest")
                require(args.nano_hf_home is not None, "--ready requires --nano-hf-home")
                result.update(check_fixtures(args.fixture_manifest))
                result.update(check_nano_cache(args.nano_hf_home))
                result.update(check_runtime(args.checkpoint_dir, args.gpu_policy))
                check_pristine_modules_cache(args.nano_hf_home.resolve())
                result.update(verify_nano_offline(args.nano_hf_home.resolve()))
                result.update(check_generated_modules_cache(args.nano_hf_home.resolve()))
                check_nano_cache(args.nano_hf_home)
                result["status"] = "local_ready_preflight_passed_no_model_or_network"
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "failed_closed_no_model_or_network",
                          "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
