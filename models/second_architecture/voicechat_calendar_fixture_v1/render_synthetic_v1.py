"""Render the frozen VoiceChat calendar scripts with two independent TTS engines.

Source candidate only. Do not run before independent source review. This writes
raw sources and final 16-kHz PCM16 WAVs with a fixed five-second zero tail;
it does no ASR, listening, VoiceChat inference, or quality selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import gcd
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from append_trailing_silence import prepare


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts.json"
RECIPE = HERE / "audio_recipe.json"
PADDING = HERE / "append_trailing_silence.py"
PIPER = Path("/Users/shrranjan/Garak/garak/.venv/bin/piper")
APPLE_SAY = Path("/usr/bin/say")
PIPER_MODEL = Path("/Users/shrranjan/Garak/garak/knowledge/multilingual_mechinterp_pilot/english_tts_transfer_20261117/models/en_US-amy-medium.onnx")
PIPER_CONFIG = PIPER_MODEL.with_suffix(".onnx.json")
EXPECTED = {
    "scripts": "c3416890395c89c03bdbf2bad9e62e0215bf46d6d9a749f8cb043af1cbe2f99d",
    "recipe": "70888859521c6705c70b23bda9dfe6bc26df337680ca62eefe68e4db74fe3e17",
    "padding": "ee3ef78ac58db232611526064c94eb4535fc7980a66be68804c2379155c0c444",
    "piper": "0b8f50b978d1da0ba9b6c770c0e58f385e093d1d33bbbb4ca0e6e9b8dabec67c",
    "model": "cf9ccdec59f05adee47173f6fb4d89e8b0c5f30c178748d2572fe4efd1d2724c",
    "config": "27490c2197e3e7b566b051fc0b1dae21200e72b44c209731419effdbbe0789e5",
    "apple_say": "b9f92936d6bb411e891f4136ae714b85a8876585ba449ea3c8d5e557765ebe8f",
}
VOICES = ("apple_samantha_en_us", "piper_amy_en_us")
APPLE_OS_BUILD = "25G241"
APPLE_VOICE_LINE = "Samantha            en_US    # Hello! My name is Samantha."


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_inputs() -> tuple[list[dict], dict]:
    for key, path in (("scripts", SCRIPTS), ("recipe", RECIPE), ("padding", PADDING),
                      ("piper", PIPER), ("model", PIPER_MODEL), ("config", PIPER_CONFIG),
                      ("apple_say", APPLE_SAY)):
        if not path.is_file() or sha(path) != EXPECTED[key]:
            raise ValueError(f"Frozen renderer dependency changed: {key}")
    scripts = json.loads(SCRIPTS.read_text(encoding="ascii"))
    recipe = json.loads(RECIPE.read_text(encoding="ascii"))
    if (scripts.get("fixture_version") != "voicechat_calendar_fixture_v1"
            or [case["id"] for case in scripts["cases"]]
            != ["A", "B", "C", "D", "A_literal", "B_literal"]
            or any(not isinstance(case["script"], str) or not case["script"].isascii()
                   for case in scripts["cases"])
            or [voice["voice_id"] for voice in recipe["required_synthetic_voices"]]
            != list(VOICES)
            or recipe["output_format"]["appended_zero_frames"] != 80000):
        raise ValueError("Frozen VoiceChat scripts or audio recipe changed")
    voice_result = subprocess.run([str(APPLE_SAY), "-v", "?"], check=True,
                                  capture_output=True, text=True, timeout=30)
    voice_lines = [line for line in voice_result.stdout.splitlines()
                   if line.startswith("Samantha ")]
    os_build = subprocess.run(["/usr/bin/sw_vers", "-buildVersion"], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    if voice_lines != [APPLE_VOICE_LINE] or os_build != APPLE_OS_BUILD:
        raise ValueError("Selected Samantha voice or macOS build changed")
    return scripts["cases"], {"apple_say_sha256": sha(APPLE_SAY),
                              "apple_voice_line": voice_lines[0],
                              "apple_os_build": os_build}


def render_one(voice: str, script: str, source: Path) -> list[str]:
    if voice == VOICES[0]:
        command = [str(APPLE_SAY), "-v", "Samantha", "-o", str(source), script]
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    elif voice == VOICES[1]:
        command = [str(PIPER), "--model", str(PIPER_MODEL), "--output_file", str(source)]
        subprocess.run(command, input=(script + "\n").encode("ascii"), check=True,
                       capture_output=True, timeout=120)
    else:
        raise ValueError("Unknown frozen voice")
    return command


def output_paths(output: Path, case_id: str, voice: str) -> tuple[Path, Path, Path]:
    if voice not in VOICES or case_id not in {"A", "B", "C", "D", "A_literal", "B_literal"}:
        raise ValueError("Unknown frozen case or voice")
    stem = f"{case_id}__{voice}"
    suffix = "aiff" if voice == VOICES[0] else "wav"
    source = output / f"{stem}.source.{suffix}"
    converted = output / f"{stem}.converted.wav"
    final = output / f"{stem}.final.wav"
    if len({source, converted, final}) != 3:
        raise AssertionError("Render paths overlap")
    return source, converted, final


def convert_one(source: Path, converted: Path) -> dict:
    samples, source_rate = sf.read(source, dtype="float32", always_2d=True)
    if samples.shape[1] != 1 or samples.shape[0] < source_rate // 2:
        raise ValueError("TTS did not produce a mono speech-length recording")
    samples = samples[:, 0]
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) <= 0:
        raise ValueError("TTS audio is silent or nonfinite")
    factor = gcd(source_rate, 16000)
    audio = resample_poly(samples, 16000 // factor, source_rate // factor).astype(np.float32)
    before_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    before_peak = float(np.max(np.abs(audio)))
    if before_rms <= 0 or before_peak <= 0:
        raise ValueError("Resampled TTS audio is silent")
    target_rms = 10 ** (-20 / 20)
    gain = min(target_rms / before_rms, 0.95 / before_peak)
    audio *= gain
    if len(audio) >= 25 * 16000 or np.max(np.abs(audio)) > 0.951:
        raise ValueError("TTS recording duration or peak is out of range")
    sf.write(converted, audio, 16000, subtype="PCM_16")
    return {"source_rate_hz": source_rate, "source_frames": len(samples),
            "resampled_frames": len(audio), "pre_gain_rms": before_rms,
            "pre_gain_peak": before_peak, "linear_gain": float(gain),
            "post_gain_peak": float(np.max(np.abs(audio)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    meta = {"status": "started_synthetic_render_no_qa", "started_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha(Path(__file__)), "dependencies_sha256": EXPECTED,
            "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "soundfile": sf.__version__,
            "scipy": __import__("scipy").__version__, "voices": list(VOICES),
            "planned_scripts": ["A", "B", "C", "D", "A_literal", "B_literal"]}
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    rows = []
    active_case = None
    active_voice = None
    try:
        cases, apple_identity = checked_inputs()
        (output / "preflight.json").write_text(json.dumps({
            "status": "render_inputs_verified_no_audio_yet", "apple_identity": apple_identity,
            "dependency_sha256": EXPECTED}, indent=2) + "\n")
        for voice in VOICES:
            for case in cases:
                active_case, active_voice = case["id"], voice
                source, converted, final = output_paths(output, case["id"], voice)
                command = render_one(voice, case["script"], source)
                conversion = convert_one(source, converted)
                padding = prepare(converted, final)
                rows.append({"case_id": case["id"], "voice_id": voice,
                             "script": case["script"], "command": command,
                             "source_path": source.name, "source_sha256": sha(source),
                             "converted_path": converted.name,
                             "converted_sha256": sha(converted),
                             "final_path": final.name, "final_sha256": sha(final),
                             "conversion": conversion, "padding": padding})
        result = {"status": "complete_synthetic_render_no_qa", "planned": 12,
                  "rendered": len(rows), "meta_sha256": sha(output / "meta.json"),
                  "preflight_sha256": sha(output / "preflight.json"),
                  "rows": rows, "model_inference_authorized": False}
        with (output / "render_manifest.json").open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "rendered": len(rows),
                          "manifest_sha256": sha(output / "render_manifest.json")}, indent=2))
    except BaseException as exc:
        (output / "error.json").write_text(json.dumps({"status": "incomplete_synthetic_render",
                  "type": type(exc).__name__, "error": str(exc), "rendered": len(rows),
                  "active_case": active_case, "active_voice": active_voice,
                  "stderr": exc.stderr.decode("utf-8", "replace")[:4096]
                            if isinstance(exc, subprocess.CalledProcessError)
                            and isinstance(exc.stderr, bytes) else None}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
