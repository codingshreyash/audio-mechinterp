"""Offline-only source, checkpoint, fixture, and GPU readiness checks.

This script never downloads weights, imports NeMo, or calls model inference.
Run --self-test locally. Run --phase source after staging code/config, and
--phase ready only on a prepared H100 or explicitly selected A100 80GB host.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


NEMO_COMMIT = "097dfe9e2f55baf653b83035868bdc89849f1b47"
VOICECHAT_COMMIT = "443794ea956ef0065f001967ffd00e77f519cb39"
NANO_COMMIT = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
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


def check_source(nemo_dir: Path, checkpoint_dir: Path) -> dict:
    require(nemo_dir.is_dir() and checkpoint_dir.is_dir(), "source/checkpoint directory missing")
    head = subprocess.run(
        ["git", "-C", str(nemo_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(head == NEMO_COMMIT, f"NeMo HEAD drift: {head}")
    for relative, expected in SOURCE_SHA256.items():
        check_file(nemo_dir / relative, expected)
    config_path = checkpoint_dir / "config.json"
    check_file(config_path, CONFIG_SHA256)
    config = json.loads(config_path.read_text())
    stt = config["model"]["stt"]["model"]
    require(stt["use_function_head"] is True, "function head disabled")
    require(stt["duplex_function_channel_weight"] == 2.0, "function channel weight drift")
    require(stt["pretrained_llm"] == "nvidia/NVIDIA-Nemotron-Nano-9B-v2", "backbone drift")
    return {"nemo_commit": head, "voicechat_commit": VOICECHAT_COMMIT,
            "source_files_verified": len(SOURCE_SHA256), "config_sha256": CONFIG_SHA256}


def check_fixtures(path: Path) -> dict:
    """Require all 12 TTS cases and exactly the declared optional human cases."""
    require(path.is_file(), f"fixture manifest missing: {path}")
    spec = json.loads(path.read_text())
    require(spec.get("status") == "frozen_voicechat_dual_head_fixture", "fixture not frozen")
    base = path.parent.resolve()
    for key in ("system_prompt", "tool_schema", "stub_calendar"):
        item = spec[key]
        local = (base / item["path"]).resolve()
        require(local.is_relative_to(base), f"fixture escapes manifest directory: {key}")
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
    for item in wavs:
        key = (item["script_id"], item["voice_id"])
        require(key not in seen, f"duplicate WAV case: {key}")
        seen.add(key)
        local = (base / item["path"]).resolve()
        require(local.is_relative_to(base), "WAV escapes manifest directory")
        check_file(local, item["sha256"])
        with wave.open(str(local), "rb") as wav:
            require((wav.getnchannels(), wav.getframerate(), wav.getsampwidth(),
                     wav.getcomptype()) == (1, 16000, 2, "NONE"),
                    f"WAV must be 16-kHz mono PCM16: {key}")
            require(wav.getnframes() > trailing, f"WAV lacks speech before trailing silence: {key}")
            wav.setpos(wav.getnframes() - trailing)
            require(wav.readframes(trailing) == b"\x00" * (2 * trailing),
                    f"frozen trailing silence is not exact digital zero: {key}")
        require(item.get("listened") is True and item.get("independent_asr_checked") is True,
                f"WAV not listening/ASR checked: {key}")
        require(isinstance(item.get("transcript"), str) and item["transcript"].strip(),
                f"missing frozen transcript: {key}")
    require(seen == expected, f"WAV case matrix differs: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    return {"fixture_manifest_sha256": sha256(path), "wav_fixtures_verified": len(wavs),
            "tts_wavs": 12, "human_wavs": len(human_scripts),
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
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(b"\x01\x00" + b"\x00\x00" * 16_000)
            fixture["wavs"].append({"script_id": script, "voice_id": voice,
                                    "path": path.name, "sha256": sha256(path),
                                    "transcript": "Test speech", "listened": True,
                                    "independent_asr_checked": True})
        for script in SCRIPT_IDS:
            for voice in fixture["tts_voice_ids"]:
                add_wav(script, voice)
        manifest = root / "FROZEN_FIXTURE.json"
        manifest.write_text(json.dumps(fixture))
        require(check_fixtures(manifest)["tts_wavs"] == 12, "12-case self-test failed")
        add_wav("A", "human_one")
        fixture["human_voice_id"] = "human_one"
        fixture["human_script_ids"] = ["A"]
        manifest.write_text(json.dumps(fixture))
        require(check_fixtures(manifest)["human_wavs"] == 1, "human-extension self-test failed")
        fixture["wavs"].pop(0)
        manifest.write_text(json.dumps(fixture))
        try:
            check_fixtures(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("missing mandatory TTS case did not fail closed")
    return {"status": "self_test_passed_no_model_or_network"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("source", "ready"))
    parser.add_argument("--nemo-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
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
                result.update(check_fixtures(args.fixture_manifest))
                result.update(check_runtime(args.checkpoint_dir, args.gpu_policy))
                result["status"] = "local_ready_preflight_passed_no_model_or_network"
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "failed_closed_no_model_or_network",
                          "error_type": type(exc).__name__, "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
