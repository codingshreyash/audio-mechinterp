"""Render exact non-speech PCM probes. This module never loads model weights."""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import wave
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "spec.json"
SPEC_SHA256 = "2dd69b687b00d90ed4e2d8bbc266039943498c31532634440f2d3bd0a0ca0951"
REQUIRED_NUMPY = "2.3.5"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_spec() -> dict:
    raw = SPEC_PATH.read_bytes()
    if sha(raw) != SPEC_SHA256:
        raise ValueError("Frozen non-speech spec bytes changed")
    spec = json.loads(raw)
    if (spec.get("schema") != "nonspeech_probe_source_spec_v1"
            or spec.get("sample_rate_hz") != 16000
            or spec.get("durations_seconds") != [2, 8]
            or spec.get("families") != ["steady_tone", "beep_train", "click_train", "band_noise"]
            or spec.get("rms_levels") != [0.005, 0.05]
            or spec.get("output_cases") != 20):
        raise ValueError("Frozen non-speech source plan changed")
    return spec


def _scale_rms(x: np.ndarray, target: float, peak_ceiling: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("Nonzero source has invalid RMS")
    y = x * (target / rms)
    if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > peak_ceiling:
        raise ValueError("Frozen source would exceed peak ceiling")
    return y


def _base(family: str, seconds: int, spec: dict) -> np.ndarray:
    sr = spec["sample_rate_hz"]
    n = seconds * sr
    t = np.arange(n, dtype=np.float64) / sr
    if family == "steady_tone":
        x = np.sin(2 * np.pi * spec["tone_hz"] * t)
        ramp = min(80, n // 2)
        taper = np.sin(np.linspace(0, np.pi / 2, ramp, endpoint=False)) ** 2
        x[:ramp] *= taper
        x[-ramp:] *= taper[::-1]
        return x
    if family == "beep_train":
        period = spec["beep_period_samples"]
        width = spec["beep_on_samples"]
        phase = np.arange(n) % period
        envelope = (phase < width).astype(np.float64)
        ramp = 80
        onset = phase < ramp
        offset = (phase >= width - ramp) & (phase < width)
        envelope[onset] *= np.sin(np.pi * phase[onset] / (2 * ramp)) ** 2
        envelope[offset] *= np.sin(np.pi * (width - phase[offset]) / (2 * ramp)) ** 2
        return envelope * np.sin(2 * np.pi * spec["beep_hz"] * t)
    if family == "click_train":
        width = spec["click_width_samples"]
        period = spec["click_period_samples"]
        x = np.zeros(n, dtype=np.float64)
        pulse = np.hanning(width)
        # Alternating polarity removes a long-run DC offset.
        for index, start in enumerate(range(0, n, period)):
            stop = min(start + width, n)
            x[start:stop] = (1 if index % 2 == 0 else -1) * pulse[:stop - start]
        return x
    if family == "band_noise":
        seed = spec[f"seed_band_noise_{seconds}s"]
        rng = np.random.Generator(np.random.PCG64(seed))
        x = rng.standard_normal(n)
        spectrum = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, 1 / sr)
        spectrum[(freq < spec["band_noise_low_hz"]) |
                 (freq > spec["band_noise_high_hz"])] = 0
        return np.fft.irfft(spectrum, n=n)
    raise ValueError("Unknown frozen family")


def cases(spec: dict) -> list[tuple[str, np.ndarray]]:
    if np.__version__ != REQUIRED_NUMPY:
        raise ValueError("Pinned NumPy version changed")
    result: list[tuple[str, np.ndarray]] = []
    for seconds in spec["durations_seconds"]:
        n = seconds * spec["sample_rate_hz"]
        result.append((f"d{seconds:02d}_silence", np.zeros(n, dtype=np.int16)))
        rng = np.random.Generator(np.random.PCG64(spec[f"seed_dither_{seconds}s"]))
        result.append((f"d{seconds:02d}_dither", rng.integers(-1, 2, n, dtype=np.int16)))
        for family in spec["families"]:
            base = _base(family, seconds, spec)
            for target in spec["rms_levels"]:
                normalized = _scale_rms(base, target, spec["pcm_peak_ceiling"])
                pcm = np.rint(normalized * 32768).astype("<i2")
                label = "low" if target == spec["rms_levels"][0] else "high"
                result.append((f"d{seconds:02d}_{family}_{label}", pcm))
    if len(result) != spec["output_cases"] or len({name for name, _ in result}) != len(result):
        raise ValueError("Frozen non-speech case count or names changed")
    return result


def wav_bytes(pcm: np.ndarray, sr: int = 16000) -> bytes:
    if pcm.dtype != np.dtype("int16") or pcm.ndim != 1:
        raise ValueError("PCM16 mono required")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.astype("<i2", copy=False).tobytes())
    return buffer.getvalue()


def render(output_dir: Path) -> dict:
    """Create one exclusive source attempt; partial output is retained on failure."""
    spec = read_spec()
    if output_dir.exists() or not output_dir.is_absolute() or not output_dir.parent.is_dir():
        raise ValueError("New absolute output directory required")
    output_dir.mkdir()
    rows = []
    started = {"schema": "nonspeech_probe_render_started_v1",
               "status": "started_no_model_inference",
               "spec_sha256": sha(SPEC_PATH.read_bytes()),
               "source_sha256": sha(Path(__file__).read_bytes()),
               "numpy_version": np.__version__, "platform": platform.platform()}
    (output_dir / "started.json").write_text(json.dumps(started, sort_keys=True, indent=2) + "\n")
    try:
        for case_id, pcm in cases(spec):
            raw = wav_bytes(pcm)
            path = output_dir / f"{case_id}.wav"
            with path.open("xb") as handle:
                handle.write(raw)
            values = pcm.astype(np.float64) / 32768.0
            rows.append({"case_id": case_id, "wav": path.name,
                         "wav_sha256": sha(raw), "pcm_sha256": sha(pcm.astype("<i2").tobytes()),
                         "frames": int(len(pcm)), "sample_rate_hz": 16000,
                         "rms": float(np.sqrt(np.mean(values * values))),
                         "peak": float(np.max(np.abs(values)))})
        result = {**started, "schema": "nonspeech_probe_render_manifest_v1",
                  "status": "complete_source_only_no_model_inference", "rows": rows}
        with (output_dir / "manifest.json").open("x") as handle:
            handle.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return result
    except Exception as error:
        failure = {**started, "schema": "nonspeech_probe_render_failure_v1",
                   "status": "incomplete_source_only", "error": repr(error), "completed_rows": rows}
        with (output_dir / "failure.json").open("x") as handle:
            handle.write(json.dumps(failure, sort_keys=True, indent=2) + "\n")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": render(args.output_dir)["status"]}))
