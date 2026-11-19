"""Two-stage, source-only VoiceChat acoustic action corpus builder.

This module never imports VoiceChat or runs ASR. Stage 1 renders exact new
scripts. Stage 2 requires two independent, model-blind interval annotations
and an adjudicated interval before making the 48 fixed acoustic arms.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import wave
from typing import Any, Callable

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "spec.json"
SPEC_SHA256 = "3325f4f693461cf48cf994bfda486e81b2bc6535d0c855a3ac838601644d414d"
ROOT = HERE.parents[2]
V1 = ROOT / "models/second_architecture/VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V1.md"
V2 = ROOT / "models/second_architecture/VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V2.md"
SOURCE_ADMISSION_GO = HERE / "SOURCE_RENDERED_INDEPENDENT_ADMISSION_GO_V1.json"
SOURCE_REVIEW_GO = HERE / "SOURCE_RENDERED_INDEPENDENT_REVIEW_GO_V1.md"
SAMPLE_RATE = 16000
TAIL_FRAMES = 80000
ARMS = ("clean", "full_noise", "decision_noise", "displaced_noise")
SCRIPT_IDS = ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2")
VOICE_IDS = ("piper_amy_en_us", "apple_samantha_en_us", "kokoro_bella_en_us")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("xb") as sink:
        sink.write(_json_bytes(value))


def _spec() -> dict[str, Any]:
    _require(_sha_file(SPEC_PATH) == SPEC_SHA256, "Frozen source spec bytes changed")
    spec = json.loads(SPEC_PATH.read_text(encoding="ascii"))
    _require(spec.get("schema") == "voicechat_acoustic_action_source_spec_v1"
             and spec.get("sample_rate_hz") == SAMPLE_RATE
             and spec.get("terminal_zero_frames") == TAIL_FRAMES
             and spec.get("nominal_noise_snr_db") == 20.0
             and spec.get("min_interval_gap_frames") == 2560,
             "Frozen source spec format changed")
    _require(spec.get("conversion") == {
        "resampler": "scipy.signal.resample_poly", "window": ["kaiser", 8.6],
        "padtype": "constant", "cval": 0.0, "normalization": "uniform_level_match",
        "target_rms": 0.1, "max_peak": 0.85, "subtype": "PCM_16",
    }, "Frozen conversion/level rule changed")
    scripts = spec.get("scripts")
    phrases = spec.get("decision_phrases")
    plan = spec.get("source_plan")
    _require(isinstance(scripts, dict) and tuple(scripts) == SCRIPT_IDS
             and isinstance(phrases, dict) and set(phrases) == set(SCRIPT_IDS)
             and isinstance(plan, list) and len(plan) == 12,
             "Frozen script or source population changed")
    expected = {(sid, "piper_amy_en_us", "development") for sid in ("A1", "B1", "C1", "D1")}
    expected |= {(sid, voice, "locked") for sid in ("A2", "B2", "C2", "D2")
                 for voice in VOICE_IDS[1:]}
    observed = {(row.get("script_id"), row.get("voice_id"), row.get("split"))
                for row in plan if isinstance(row, dict)}
    _require(observed == expected and len(observed) == len(plan),
             "Frozen 12-source plan changed")
    for sid, script in scripts.items():
        _require(isinstance(script, str) and script.isascii() and script.strip() == script
                 and isinstance(phrases[sid], str)
                 and phrases[sid].casefold() in script.casefold(),
                 f"Invalid exact script or decision phrase for {sid}")
    return spec


def _read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        _require(wav.getnchannels() == 1 and wav.getframerate() == SAMPLE_RATE
                 and wav.getsampwidth() == 2 and wav.getcomptype() == "NONE",
                 f"Not mono 16-kHz PCM16: {path}")
        data = wav.readframes(wav.getnframes())
    _require(bool(data) and len(data) % 2 == 0, f"Empty or odd PCM: {path}")
    return np.frombuffer(data, dtype="<i2").copy()


def _write_pcm(path: Path, samples: np.ndarray) -> None:
    pcm = np.asarray(samples, dtype="<i2").reshape(-1)
    with path.open("xb") as stream:
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())


def _raw_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    mono = np.asarray(samples, dtype="<f4").reshape(-1)
    _require(bool(mono.size) and np.isfinite(mono).all() and sample_rate > 0,
             "Kokoro returned empty/nonfinite audio")
    data = mono.tobytes()
    _require(len(data) < 0xFFFFFFFF - 36, "Raw Kokoro WAV too large")
    header = (struct.pack("<4sI4s", b"RIFF", 36 + len(data), b"WAVE")
              + struct.pack("<4sIHHIIHH", b"fmt ", 16, 3, 1,
                            sample_rate, sample_rate * 4, 4, 32)
              + struct.pack("<4sI", b"data", len(data)))
    with path.open("xb") as sink:
        sink.write(header + data)


def _compute_converted(raw: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct conversion from raw bytes without writing a WAV."""
    samples, source_rate = sf.read(raw, dtype="float64", always_2d=True)
    _require(samples.ndim == 2 and samples.shape[1] == 1 and source_rate > 0
             and len(samples) >= source_rate // 2
             and np.isfinite(samples).all(), "Raw TTS audio invalid or too short")
    up = SAMPLE_RATE // math.gcd(source_rate, SAMPLE_RATE)
    down = source_rate // math.gcd(source_rate, SAMPLE_RATE)
    resampled = resample_poly(samples[:, 0], up, down,
                             window=("kaiser", 8.6), padtype="constant", cval=0.0)
    _require(np.isfinite(resampled).all() and np.max(np.abs(resampled)) > 0
             and len(resampled) < 25 * SAMPLE_RATE,
             "Converted TTS audio invalid, silent, or too long")
    before_rms = float(np.sqrt(np.mean(np.square(resampled))))
    before_peak = float(np.max(np.abs(resampled)))
    _require(before_rms > 0 and before_peak > 0, "Converted TTS level invalid")
    gain = min(0.1 / before_rms, 0.85 / before_peak)
    _require(math.isfinite(gain) and gain > 0, "TTS level-match gain invalid")
    resampled *= gain
    quantized = np.rint(resampled * 32768.0)
    _require(bool(np.all((quantized >= -32768) & (quantized <= 32767))),
             "TTS conversion would clip; preserve failed source")
    pcm = quantized.astype("<i2")
    return pcm, {"raw_sample_rate_hz": source_rate, "raw_frames": len(samples),
                 "resample_up": up, "resample_down": down,
                 "converted_frames": len(pcm), "clean_frames": len(pcm) + TAIL_FRAMES,
                 "pre_gain_rms": before_rms, "pre_gain_peak": before_peak,
                 "level_match_gain": gain,
                 "post_gain_peak": float(np.max(np.abs(resampled))),
                 "raw_pcm_float64_sha256": _sha_bytes(np.asarray(samples[:, 0], dtype="<f8").tobytes()),
                 "converted_pcm_sha256": _sha_bytes(pcm.tobytes())}


def _convert_raw(raw: Path, converted: Path, clean: Path) -> dict[str, Any]:
    pcm, numeric = _compute_converted(raw)
    _write_pcm(converted, pcm)
    _write_pcm(clean, np.concatenate((pcm, np.zeros(TAIL_FRAMES, dtype="<i2"))))
    return {**numeric, "raw_sha256": _sha_file(raw),
            "converted_sha256": _sha_file(converted),
            "clean_sha256": _sha_file(clean),
            "clean_pcm_sha256": _sha_bytes(_read_pcm(clean).tobytes())}


class _RealTTS:
    def __init__(self, spec: dict[str, Any]):
        self.pins = spec["tts"]
        self._piper = None
        self._kokoro = None
        _require(importlib.metadata.version("piper-tts") == "1.8.0"
                 and importlib.metadata.version("kokoro-onnx") == "0.6.1",
                 "Pinned TTS package versions absent")
        piper = self.pins["piper"]
        model = Path(piper["model_path"])
        config = Path(str(model) + ".json")
        _require(model.is_file() and config.is_file()
                 and _sha_file(model) == piper["model_sha256"]
                 and _sha_file(config) == piper["config_sha256"],
                 "Pinned Piper Amy bytes absent or changed")
        kokoro = self.pins["kokoro"]
        km, kv = Path(kokoro["model_path"]), Path(kokoro["voices_path"])
        _require(km.is_file() and kv.is_file()
                 and _sha_file(km) == kokoro["model_sha256"]
                 and _sha_file(kv) == kokoro["voices_sha256"],
                 "Pinned Kokoro model/voices bytes absent or changed")
        apple = self.pins["samantha"]
        say = Path(apple["command"])
        _require(say.is_file() and _sha_file(say) == apple["command_sha256"],
                 "Pinned say binary absent or changed")
        build = subprocess.run(["/usr/bin/sw_vers", "-buildVersion"], check=True,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        listed = subprocess.run([str(say), "-v", "?"], check=True,
                                capture_output=True, text=True, timeout=30).stdout.splitlines()
        _require(build == apple["os_build"] and apple["voice_line"] in listed,
                 "Pinned Samantha voice or OS build changed")

    def identity(self) -> dict[str, Any]:
        return {"piper_package_version": importlib.metadata.version("piper-tts"),
                "kokoro_package_version": importlib.metadata.version("kokoro-onnx"),
                "onnxruntime_version": importlib.metadata.version("onnxruntime"),
                "piper_voice": self.pins["piper"]["voice"],
                "kokoro_voice": self.pins["kokoro"]["voice"],
                "samantha_voice_line": self.pins["samantha"]["voice_line"],
                "piper_model_sha256": self.pins["piper"]["model_sha256"],
                "piper_config_sha256": self.pins["piper"]["config_sha256"],
                "kokoro_model_sha256": self.pins["kokoro"]["model_sha256"],
                "kokoro_voices_sha256": self.pins["kokoro"]["voices_sha256"],
                "say_sha256": self.pins["samantha"]["command_sha256"],
                "os_build": self.pins["samantha"]["os_build"]}

    def __call__(self, voice_id: str, text: str, raw: Path) -> list[str]:
        if voice_id == "piper_amy_en_us":
            if self._piper is None:
                from piper import PiperVoice
                self._piper = PiperVoice.load(self.pins["piper"]["model_path"],
                                               config_path=self.pins["piper"]["model_path"] + ".json",
                                               use_cuda=False)
                _require("CPUExecutionProvider" in self._piper.session.get_providers(),
                         "Piper session lacks CPU provider")
            from piper.config import SynthesisConfig
            config = SynthesisConfig(**self.pins["piper"]["synthesis"])
            with wave.open(str(raw), "wb") as wav:
                self._piper.synthesize_wav(text, wav, syn_config=config)
            return ["PiperVoice.synthesize_wav", "en_US-amy-medium", text]
        if voice_id == "apple_samantha_en_us":
            command = [self.pins["samantha"]["command"], "-v", "Samantha", "-o", str(raw), text]
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            return command
        if voice_id == "kokoro_bella_en_us":
            if self._kokoro is None:
                from kokoro_onnx import Kokoro
                self._kokoro = Kokoro(self.pins["kokoro"]["model_path"],
                                      self.pins["kokoro"]["voices_path"])
                _require("af_bella" in self._kokoro.get_voices()
                         and "CPUExecutionProvider" in self._kokoro.sess.get_providers(),
                         "Pinned Kokoro af_bella voice or CPU provider absent")
            voice = self.pins["kokoro"]
            audio, rate = self._kokoro.create(text, voice=voice["voice"], **voice["synthesis"])
            _raw_float_wav(raw, audio, rate)
            return ["Kokoro.create", voice["voice"], text]
        raise ValueError("Voice outside frozen source plan")


def _source_id(row: dict[str, str]) -> str:
    return f"{row['script_id']}__{row['voice_id']}"


def _committed_exact(path: Path) -> None:
    """Reject a local, uncommitted source-admission self-repin."""
    _require(path.is_file() and not path.is_symlink(),
             f"Independent admission file absent: {path.name}")
    relative = str(path.relative_to(ROOT))
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                             "--", relative], capture_output=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "--", relative], capture_output=True, text=True, check=True)
    committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                               capture_output=True)
    _require(listed.returncode == 0 and not dirty.stdout.strip()
             and committed.returncode == 0 and committed.stdout == path.read_bytes(),
             f"Independent admission is not exact committed bytes: {path.name}")


def _external_source_admission(source_dir: Path, manifest_path: Path,
                               started_path: Path) -> dict[str, Any]:
    """Require a separately committed reviewer pin for production stage 2."""
    _committed_exact(SOURCE_REVIEW_GO)
    _committed_exact(SOURCE_ADMISSION_GO)
    receipt = json.loads(SOURCE_ADMISSION_GO.read_text(encoding="ascii"))
    manifest_sha = _sha_file(manifest_path)
    started_sha = _sha_file(started_path)
    review_text = SOURCE_REVIEW_GO.read_text(encoding="utf-8")
    _require(receipt.get("schema") == "voicechat_acoustic_source_admission_go_v1"
             and receipt.get("status") == "independent_source_byte_audit_go"
             and receipt.get("source_dir") == str(source_dir.resolve())
             and receipt.get("source_manifest_sha256") == manifest_sha
             and receipt.get("started_sha256") == started_sha
             and receipt.get("source_spec_sha256") == _sha_file(SPEC_PATH)
             and receipt.get("source_builder_sha256") == _sha_file(Path(__file__))
             and receipt.get("review_sha256") == _sha_file(SOURCE_REVIEW_GO)
             and "**Decision: GO" in review_text
             and manifest_sha in review_text and started_sha in review_text,
             "Externally reviewed exact stage-1 source manifest pin absent or different")
    return receipt


def _checked_source_provenance(source_dir: Path, spec: dict[str, Any],
                               manifest_path: Path, manifest: dict[str, Any],
                               *, allow_synthetic_test_only: bool) -> tuple[bool, dict[str, Any] | None]:
    started_path = source_dir / "started.json"
    _require(source_dir.is_dir() and not source_dir.is_symlink()
             and started_path.is_file() and not started_path.is_symlink(),
             "Stage-1 source directory or immutable start receipt absent")
    started = json.loads(started_path.read_text(encoding="ascii"))
    expected_start = dict(manifest)
    expected_start["status"] = "started_no_model_inference"
    expected_start["rows"] = []
    _require(started == expected_start,
             "Stage-1 start and completion identity/provenance differ")
    _require(manifest.get("schema") == "voicechat_acoustic_action_sources_v1"
             and manifest.get("status") == "complete_sources_no_human_intervals"
             and manifest.get("spec_sha256") == _sha_file(SPEC_PATH)
             and manifest.get("builder_sha256") == _sha_file(Path(__file__))
             and manifest.get("protocol_v1_sha256") == _sha_file(V1)
             and manifest.get("protocol_v2_sha256") == _sha_file(V2)
             and manifest.get("planned_sources") == 12
             and manifest.get("candidate_inference_authorized") is False
             and isinstance(manifest.get("rows"), list)
             and len(manifest["rows"]) == 12,
             "Stage-1 source identity or population invalid")
    identity = manifest.get("renderer_identity")
    _require(isinstance(identity, dict), "Stage-1 renderer identity absent")
    synthetic = identity.get("synthetic_test_only") is True
    if synthetic:
        _require(allow_synthetic_test_only,
                 "Synthetic test renderer cannot enter the production variant path")
        admission = None
    else:
        pins = spec["tts"]
        _require(identity == {
            "piper_package_version": "1.8.0",
            "kokoro_package_version": "0.6.1",
            "onnxruntime_version": identity.get("onnxruntime_version"),
            "piper_voice": pins["piper"]["voice"],
            "kokoro_voice": pins["kokoro"]["voice"],
            "samantha_voice_line": pins["samantha"]["voice_line"],
            "piper_model_sha256": pins["piper"]["model_sha256"],
            "piper_config_sha256": pins["piper"]["config_sha256"],
            "kokoro_model_sha256": pins["kokoro"]["model_sha256"],
            "kokoro_voices_sha256": pins["kokoro"]["voices_sha256"],
            "say_sha256": pins["samantha"]["command_sha256"],
            "os_build": pins["samantha"]["os_build"],
        } and isinstance(identity.get("onnxruntime_version"), str)
                 and bool(identity["onnxruntime_version"]),
                 "Pinned production TTS identity differs")
        admission = None
    _require([_source_id(row) for row in spec["source_plan"]]
             == [row.get("source_id") for row in manifest["rows"]],
             "Stage-1 source order or IDs differ from frozen plan")
    for planned, row in zip(spec["source_plan"], manifest["rows"], strict=True):
        source_id = _source_id(planned)
        text = spec["scripts"][planned["script_id"]]
        phrase = spec["decision_phrases"][planned["script_id"]]
        raw_suffix = "aiff" if planned["voice_id"] == "apple_samantha_en_us" else "wav"
        _require(all(row.get(key) == value for key, value in planned.items())
                 and row.get("script") == text
                 and row.get("script_sha256") == _sha_bytes(text.encode("ascii"))
                 and row.get("decision_phrase") == phrase
                 and row.get("raw_path") == f"{source_id}.raw.{raw_suffix}"
                 and row.get("converted_path") == f"{source_id}.converted.wav"
                 and row.get("clean_path") == f"{source_id}.clean.wav",
                 f"Stage-1 frozen script/voice/phrase/path differs: {source_id}")
        raw = source_dir / row["raw_path"]
        converted = source_dir / row["converted_path"]
        clean = source_dir / row["clean_path"]
        _require(all(path.is_file() and not path.is_symlink() for path in (raw, converted, clean))
                 and row.get("raw_sha256") == _sha_file(raw)
                 and row.get("converted_sha256") == _sha_file(converted)
                 and row.get("clean_sha256") == _sha_file(clean),
                 f"Stage-1 source file hash differs: {source_id}")
        if not synthetic:
            if planned["voice_id"] == "piper_amy_en_us":
                command = ["PiperVoice.synthesize_wav", pins["piper"]["voice"], text]
            elif planned["voice_id"] == "kokoro_bella_en_us":
                command = ["Kokoro.create", pins["kokoro"]["voice"], text]
            else:
                command = [pins["samantha"]["command"], "-v", "Samantha",
                           "-o", str(raw), text]
            _require(row.get("render_command") == command,
                     f"Stage-1 engine command differs: {source_id}")
        reconstructed, numeric = _compute_converted(raw)
        _require(all(row.get(key) == value for key, value in numeric.items())
                 and row.get("raw_frames") == numeric["raw_frames"]
                 and row.get("converted_frames") == len(reconstructed)
                 and row.get("clean_frames") == len(reconstructed) + TAIL_FRAMES
                 and row.get("converted_pcm_sha256") == _sha_bytes(reconstructed.tobytes()),
                 f"Stage-1 conversion metadata differs: {source_id}")
        expected_clean = np.concatenate((reconstructed, np.zeros(TAIL_FRAMES, dtype="<i2")))
        _require(np.array_equal(_read_pcm(converted), reconstructed)
                 and np.array_equal(_read_pcm(clean), expected_clean)
                 and row.get("clean_pcm_sha256") == _sha_bytes(expected_clean.tobytes()),
                 f"Stage-1 raw-to-converted-to-clean reconstruction differs: {source_id}")
    if not synthetic:
        admission = _external_source_admission(source_dir, manifest_path, started_path)
    return synthetic, admission


def render_sources(output_dir: Path, *, renderer: Callable[[str, str, Path], list[str]] | None = None,
                   renderer_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render the 12 frozen sources; injected renderer is for fake tests only."""
    spec = _spec()
    _require(output_dir.is_absolute() and not output_dir.exists(),
             "Stage-1 output must be a new absolute directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": "voicechat_acoustic_action_sources_v1",
        "status": "started_no_model_inference", "created_utc": _utc(),
        "spec_sha256": _sha_file(SPEC_PATH),
        "protocol_v1_sha256": _sha_file(V1), "protocol_v2_sha256": _sha_file(V2),
        "builder_sha256": _sha_file(Path(__file__)),
        "platform": platform.platform(), "python": sys.version.split()[0],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "soundfile": sf.__version__},
        "planned_sources": 12, "rows": [], "candidate_inference_authorized": False,
    }
    active = None
    try:
        if renderer is None:
            real = _RealTTS(spec)
            renderer = real
            renderer_identity = real.identity()
        else:
            _require(renderer_identity is not None and renderer_identity.get("synthetic_test_only") is True,
                     "Injected renderer requires synthetic-test identity")
        manifest["renderer_identity"] = renderer_identity
        _write_json_exclusive(output_dir / "started.json", manifest)
        for row in spec["source_plan"]:
            active = _source_id(row)
            text = spec["scripts"][row["script_id"]]
            stem = active
            raw = output_dir / f"{stem}.raw.{ 'aiff' if row['voice_id'] == 'apple_samantha_en_us' else 'wav' }"
            converted = output_dir / f"{stem}.converted.wav"
            clean = output_dir / f"{stem}.clean.wav"
            command = renderer(row["voice_id"], text, raw)
            _require(raw.is_file() and not raw.is_symlink(), "Renderer did not create raw file")
            conversion = _convert_raw(raw, converted, clean)
            manifest["rows"].append({**row, "source_id": stem, "script": text,
                                     "script_sha256": _sha_bytes(text.encode("ascii")),
                                     "decision_phrase": spec["decision_phrases"][row["script_id"]],
                                     "render_command": command,
                                     "raw_path": raw.name, "converted_path": converted.name,
                                     "clean_path": clean.name, **conversion})
        manifest["status"] = "complete_sources_no_human_intervals"
        _write_json_exclusive(output_dir / "source_manifest.json", manifest)
        _write_json_exclusive(output_dir / "annotation_template.json",
                              annotation_template(manifest, _sha_file(output_dir / "source_manifest.json")))
        return manifest
    except BaseException as error:
        _write_json_exclusive(output_dir / "failure.json", {
            "status": "incomplete_sources_no_candidate_inference", "active_source": active,
            "error_type": type(error).__name__, "error": str(error),
            "completed_rows": manifest["rows"], "created_utc": _utc(),
        })
        raise


def annotation_template(manifest: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {"schema": "voicechat_acoustic_intervals_v1",
            "source_manifest_sha256": manifest_sha256,
            "sample_rate_hz": SAMPLE_RATE,
            "instructions": "Two people listen to each clean WAV without model output. Each marks decision and equal-width displaced speech windows; adjudicate before noise rendering.",
            "records": [{"source_id": row["source_id"],
                         "clean_wav_sha256": row["clean_sha256"],
                         "clean_wav_path": row["clean_path"],
                         "decision_phrase": row["decision_phrase"],
                         "content_frames": row["converted_frames"],
                         "annotator_a": {"reviewer_id": None, "decision": None, "displaced": None},
                         "annotator_b": {"reviewer_id": None, "decision": None, "displaced": None},
                         "third_annotation": None,
                         "adjudication": {"by": None, "decision": None, "displaced": None, "note": None}}
                        for row in manifest["rows"]]}


def _interval(value: Any, content_frames: int, name: str) -> tuple[int, int]:
    _require(isinstance(value, list) and len(value) == 2
             and all(type(item) is int for item in value), f"{name} must be two integer frames")
    start, end = value
    _require(0 <= start < end <= content_frames, f"{name} outside speech content")
    return start, end


def _window_pair(data: dict[str, Any], content_frames: int, gap: int, name: str) -> tuple[tuple[int,int], tuple[int,int]]:
    decision = _interval(data.get("decision"), content_frames, f"{name}.decision")
    displaced = _interval(data.get("displaced"), content_frames, f"{name}.displaced")
    _require(decision[1] - decision[0] == displaced[1] - displaced[0],
             f"{name} windows differ in width")
    _require(displaced[1] + gap <= decision[0] or decision[1] + gap <= displaced[0],
             f"{name} windows overlap or lack 160-ms gap")
    return decision, displaced


def _checked_annotations(source_dir: Path, spec: dict[str, Any],
                         manifest: dict[str, Any], annotation_path: Path) -> dict[str, Any]:
    data = json.loads(annotation_path.read_text(encoding="ascii"))
    _require(data.get("schema") == "voicechat_acoustic_intervals_v1"
             and data.get("source_manifest_sha256") == _sha_file(source_dir / "source_manifest.json")
             and data.get("sample_rate_hz") == SAMPLE_RATE
             and isinstance(data.get("records"), list) and len(data["records"]) == 12,
             "Annotation source binding or population invalid")
    by_id = {record.get("source_id"): record for record in data["records"]
             if isinstance(record, dict)}
    _require(len(by_id) == 12 and set(by_id) == {row["source_id"] for row in manifest["rows"]},
             "Annotation source IDs duplicated or missing")
    for row in manifest["rows"]:
        record = by_id[row["source_id"]]
        _require(record.get("clean_wav_sha256") == row["clean_sha256"]
                 and record.get("clean_wav_path") == row["clean_path"]
                 and record.get("decision_phrase") == row["decision_phrase"]
                 and record.get("content_frames") == row["converted_frames"],
                 "Annotation clean WAV or phrase binding differs")
        content = row["converted_frames"]
        proposed = []
        for key in ("annotator_a", "annotator_b"):
            observer = record.get(key)
            _require(isinstance(observer, dict)
                     and isinstance(observer.get("reviewer_id"), str)
                     and len(observer["reviewer_id"].strip()) >= 2,
                     f"{key} human reviewer ID missing")
            proposed.append((observer["reviewer_id"],
                             _window_pair(observer, content, spec["min_interval_gap_frames"], key)))
        _require(proposed[0][0] != proposed[1][0], "Annotators must have distinct IDs")
        adjudication = record.get("adjudication")
        _require(isinstance(adjudication, dict)
                 and isinstance(adjudication.get("by"), str)
                 and len(adjudication["by"].strip()) >= 2
                 and isinstance(adjudication.get("note"), str)
                 and bool(adjudication["note"].strip()),
                 "Final interval adjudication absent")
        final_pair = _window_pair(adjudication, content,
                                  spec["min_interval_gap_frames"], "adjudication")
        disagreement = proposed[0][1] != proposed[1][1]
        drift = final_pair != proposed[0][1] or final_pair != proposed[1][1]
        third = record.get("third_annotation")
        if disagreement or drift:
            _require(isinstance(third, dict)
                     and isinstance(third.get("reviewer_id"), str)
                     and len(third["reviewer_id"].strip()) >= 2
                     and third["reviewer_id"] not in {proposed[0][0], proposed[1][0]}
                     and adjudication["by"] == third["reviewer_id"],
                     "Disagreement or final drift requires distinct third adjudicator")
            third_pair = _window_pair(third, content,
                                      spec["min_interval_gap_frames"], "third_annotation")
            _require(final_pair in (proposed[0][1], proposed[1][1], third_pair),
                     "Final intervals must follow a recorded proposal")
        else:
            _require(third is None, "Unneeded third annotation changes agreed result")
    return data


def _noise_seed(script_id: str, voice_id: str) -> int:
    text = f"voicechat_acoustic_action_v1|{script_id}|{voice_id}".encode("ascii")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def _noisy_arm(clean: np.ndarray, content_frames: int,
               noise: np.ndarray, rms: float,
               mask: tuple[int, int]) -> tuple[np.ndarray, dict[str, float | int]]:
    _require(0 <= mask[0] < mask[1] <= content_frames, "Noise mask outside content")
    original = clean.astype(np.float64) / 32768.0
    changed = np.zeros(content_frames, dtype=np.float64)
    changed[mask[0]:mask[1]] = 0.1 * rms * noise[mask[0]:mask[1]]
    rendered = np.rint((original[:content_frames] + changed) * 32768.0)
    _require(bool(np.all((rendered >= -32768) & (rendered <= 32767))),
             "Noise quantization would clip")
    result = clean.copy()
    result[:content_frames] = rendered.astype("<i2")
    _require(np.array_equal(result[content_frames:], clean[content_frames:]),
             "Terminal zero tail changed")
    realized = (result[:content_frames].astype(np.float64)
                - clean[:content_frames].astype(np.float64)) / 32768.0
    noise_rms = float(np.sqrt(np.mean(np.square(realized))))
    snr = None if noise_rms == 0 else float(20 * math.log10(rms / noise_rms))
    return result, {"realized_noise_rms": noise_rms,
                    "realized_snr_db_content": snr,
                    "changed_samples": int(np.count_nonzero(result != clean)),
                    "mask_start": mask[0], "mask_end": mask[1]}


def make_variants(source_dir: Path, annotation_path: Path,
                  output_dir: Path, *, allow_synthetic_test_only: bool = False) -> dict[str, Any]:
    """Create exactly 48 arms after human interval annotation; no inference."""
    _require(output_dir.is_absolute() and not output_dir.exists(),
             "Stage-2 output must be a new absolute directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    active = None
    rows: list[dict[str, Any]] = []
    try:
        spec = _spec()
        manifest_path = source_dir / "source_manifest.json"
        _require(manifest_path.is_file() and not manifest_path.is_symlink(),
                 "Complete source manifest absent")
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        synthetic_source, source_admission = _checked_source_provenance(
            source_dir, spec, manifest_path, manifest,
            allow_synthetic_test_only=allow_synthetic_test_only)
        _require(annotation_path.is_file() and not annotation_path.is_symlink(),
                 "Two-person annotation file absent")
        annotations = _checked_annotations(source_dir, spec, manifest, annotation_path)
        by_id = {record["source_id"]: record for record in annotations["records"]}
        for row in manifest["rows"]:
            active = row["source_id"]
            clean_path = source_dir / row["clean_path"]
            for label in ("raw", "converted"):
                path = source_dir / row[f"{label}_path"]
                _require(path.is_file() and not path.is_symlink()
                         and _sha_file(path) == row[f"{label}_sha256"],
                         f"Source {label} file absent or changed")
            _require(clean_path.is_file() and not clean_path.is_symlink()
                     and _sha_file(clean_path) == row["clean_sha256"],
                     "Source clean WAV absent or changed")
            clean = _read_pcm(clean_path)
            content_frames = row["converted_frames"]
            _require(len(clean) == content_frames + TAIL_FRAMES
                     and np.count_nonzero(clean[content_frames:]) == 0,
                     "Source PCM length or terminal zeros changed")
            clean_float = clean[:content_frames].astype(np.float64) / 32768.0
            rms = float(np.sqrt(np.mean(np.square(clean_float))))
            _require(math.isfinite(rms) and rms > 0, "Clean content RMS invalid")
            seed = _noise_seed(row["script_id"], row["voice_id"])
            noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(content_frames)
            noise -= float(noise.mean())
            noise_rms = float(np.sqrt(np.mean(np.square(noise))))
            _require(math.isfinite(noise_rms) and noise_rms > 0,
                     "Standardized noise RMS invalid")
            noise /= noise_rms
            final = by_id[active]["adjudication"]
            masks = {
                "full_noise": (0, content_frames),
                "decision_noise": tuple(final["decision"]),
                "displaced_noise": tuple(final["displaced"]),
            }
            for arm in ARMS:
                destination = output_dir / f"{active}__{arm}.wav"
                if arm == "clean":
                    shutil.copyfile(clean_path, destination)
                    outcome = {"realized_noise_rms": 0.0,
                               "realized_snr_db_content": None,
                               "changed_samples": 0,
                               "mask_start": None, "mask_end": None}
                else:
                    samples, outcome = _noisy_arm(clean, content_frames,
                                                  noise, rms, masks[arm])
                    _write_pcm(destination, samples)
                    _require(np.array_equal(_read_pcm(destination)[content_frames:],
                                            clean[content_frames:]),
                             "Written zero tail changed")
                rows.append({"source_id": active, "script_id": row["script_id"],
                             "voice_id": row["voice_id"], "split": row["split"],
                             "arm": arm, "wav_path": destination.name,
                             "wav_sha256": _sha_file(destination),
                             "pcm_sha256": _sha_bytes(_read_pcm(destination).tobytes()),
                             "frames": len(clean), "seed": seed,
                             "clean_content_rms": rms, **outcome})
        result = {"schema": "voicechat_acoustic_action_variants_v1",
                  "status": "complete_48_wavs_no_human_qa_or_model_inference",
                  "created_utc": _utc(), "spec_sha256": _sha_file(SPEC_PATH),
                  "source_manifest_sha256": _sha_file(manifest_path),
                  "started_sha256": _sha_file(source_dir / "started.json"),
                  "source_admission_sha256": _sha_file(SOURCE_ADMISSION_GO)
                  if source_admission is not None else None,
                  "annotation_sha256": _sha_file(annotation_path),
                  "builder_sha256": _sha_file(Path(__file__)),
                  "synthetic_test_only": synthetic_source,
                  "numpy_version": np.__version__, "planned_sources": 12,
                  "planned_arms": 48, "rows": rows,
                  "candidate_inference_authorized": False}
        _write_json_exclusive(output_dir / "variant_manifest.json", result)
        return result
    except BaseException as error:
        _write_json_exclusive(output_dir / "failure.json", {
            "status": "incomplete_variants_no_candidate_inference",
            "active_source": active, "completed_rows": rows,
            "error_type": type(error).__name__, "error": str(error),
            "created_utc": _utc(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    render = sub.add_parser("render-sources")
    render.add_argument("--output-dir", type=Path, required=True)
    variants = sub.add_parser("make-variants")
    variants.add_argument("--source-dir", type=Path, required=True)
    variants.add_argument("--annotations", type=Path, required=True)
    variants.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "render-sources":
        result = render_sources(args.output_dir)
        print(json.dumps({"status": result["status"], "rows": len(result["rows"])},
                         sort_keys=True))
    else:
        result = make_variants(args.source_dir, args.annotations, args.output_dir)
        print(json.dumps({"status": result["status"], "rows": len(result["rows"])},
                         sort_keys=True))


if __name__ == "__main__":
    main()
