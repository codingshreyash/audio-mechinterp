"""Frozen, source-only WAV construction for the natural-tail Whisper follow-up.

This module never imports Transformers or loads model weights. Rendering is an
explicit operation into a new directory; importing it does not write files.
"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np

from .render import wav_bytes


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/render_20261119_0740Z"
SOURCE_MANIFEST_SHA256 = "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90"
SAMPLE_RATE = 16_000
TAIL_SAMPLES = (8_000, 32_000)
NOISE_PCM_AMPLITUDE = 32  # Exactly 32/32768 = 0.0009765625 RMS, before no taper.
NOISE_SEEDS = {8_000: (9405, 9406), 32_000: (9420, 9421)}
CONDITIONS = ("base", "z8000", "n8000_s0", "n8000_s1",
              "z32000", "n32000_s0", "n32000_s1")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def noise_tail(samples: int, seed_index: int) -> np.ndarray:
    """A fixed, low-level, two-valued broadband background in PCM16 units."""
    if samples not in TAIL_SAMPLES or seed_index not in (0, 1):
        raise ValueError("Unfrozen noise condition")
    rng = np.random.Generator(np.random.PCG64(NOISE_SEEDS[samples][seed_index]))
    signs = rng.integers(0, 2, size=samples, dtype=np.int8).astype(np.int16) * 2 - 1
    return np.asarray(signs * NOISE_PCM_AMPLITUDE, dtype="<i2")


def construct_pcm(original: np.ndarray, condition: str) -> np.ndarray:
    if original.ndim != 1 or original.dtype != np.dtype("int16"):
        raise ValueError("Mono PCM16 source required")
    if condition not in CONDITIONS:
        raise ValueError("Unfrozen condition")
    if condition == "base":
        return original.copy()
    kind, detail = condition[0], condition[1:]
    if kind == "z":
        samples = int(detail)
        if samples not in TAIL_SAMPLES:
            raise ValueError("Unfrozen duration")
        tail = np.zeros(samples, dtype="<i2")
    else:
        duration, seed = detail.split("_s")
        tail = noise_tail(int(duration), int(seed))
    output = np.concatenate((original, tail)).astype("<i2", copy=False)
    if not np.array_equal(output[: len(original)], original):
        raise AssertionError("Source PCM prefix changed")
    return output


def trusted_sources() -> list[tuple[str, np.ndarray, dict]]:
    manifest_bytes = (SOURCE / "manifest.json").read_bytes()
    if digest(manifest_bytes) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Source manifest changed")
    rows = json.loads(manifest_bytes)["rows"]
    if len(rows) != 20 or len({row["case_id"] for row in rows}) != 20:
        raise ValueError("Source denominator changed")
    sources = []
    for row in rows:
        path = SOURCE / row["wav"]
        if path.is_symlink() or digest(path.read_bytes()) != row["wav_sha256"]:
            raise ValueError(f"Source WAV changed: {row['case_id']}")
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                    stream.getnframes()) != (1, 2, SAMPLE_RATE, row["frames"]):
                raise ValueError(f"Source format changed: {row['case_id']}")
            pcm_bytes = stream.readframes(row["frames"])
        if digest(pcm_bytes) != row["pcm_sha256"]:
            raise ValueError(f"Source PCM changed: {row['case_id']}")
        sources.append((row["case_id"], np.frombuffer(pcm_bytes, dtype="<i2").copy(), row))
    return sources


def render_new(output_dir: Path) -> dict:
    """Create the 140 fixed WAVs and a manifest, with no model inference."""
    if output_dir.exists() or not output_dir.is_absolute() or not output_dir.parent.is_dir():
        raise ValueError("A new absolute output directory is required")
    sources = trusted_sources()
    if np.__version__ != "2.3.5":
        raise ValueError("NumPy version differs from frozen source renderer")
    output_dir.mkdir()
    rows = []
    for case_id, original, source_row in sources:
        source_prefix = original.astype("<i2", copy=False).tobytes()
        for condition in CONDITIONS:
            pcm = construct_pcm(original, condition)
            raw = wav_bytes(pcm, SAMPLE_RATE)
            name = f"{case_id}__{condition}.wav"
            (output_dir / name).write_bytes(raw)
            rows.append({"case_id": case_id, "condition": condition, "wav": name,
                         "wav_sha256": digest(raw),
                         "pcm_sha256": digest(pcm.astype("<i2", copy=False).tobytes()),
                         "source_prefix_sha256": digest(source_prefix),
                         "source_wav_sha256": source_row["wav_sha256"],
                         "frames": len(pcm), "sample_rate_hz": SAMPLE_RATE})
    result = {"schema": "whisper_natural_tail_waveform_v1",
              "status": "complete_source_only_no_model_inference",
              "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
              "builder_sha256": digest(Path(__file__).read_bytes()), "rows": rows}
    (output_dir / "manifest.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
