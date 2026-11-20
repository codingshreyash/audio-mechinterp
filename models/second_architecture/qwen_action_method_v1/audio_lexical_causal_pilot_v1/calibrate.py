"""Freeze a symmetric Piper pair with word-only 0.73 PCM gain.

The operation changes no sample outside the source red/blue difference support.
It was selected from a processor-only check, before any Qwen model-weight call.
"""
from __future__ import annotations

import hashlib
import json
import struct
import wave
from pathlib import Path

from . import render

HERE = Path(__file__).resolve().parent
WAV = {word: HERE / f"piper_{word}_word_gain_073_zero_pad_16s.wav"
       for word in ("red", "blue")}
MANIFEST = HERE / "CALIBRATED_MANIFEST_V1.json"
GAIN_NUMERATOR = 73
GAIN_DENOMINATOR = 100


def _scale_sample(sample: int) -> int:
    """Nearest integer with exact ties-to-even arithmetic, retaining sign."""
    sign = -1 if sample < 0 else 1
    quotient, remainder = divmod(abs(sample) * GAIN_NUMERATOR, GAIN_DENOMINATOR)
    if remainder > 50 or (remainder == 50 and quotient % 2):
        quotient += 1
    return sign * quotient


def make() -> tuple[dict[str, bytes], dict]:
    original = render.verify()
    lo, hi = original["pcm_difference_support"]
    import io
    outputs = {}
    for word, source in render.WAV.items():
        with wave.open(str(source), "rb") as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                    reader.getnframes()) != (1, 2, 16_000, 256_000):
                raise ValueError(f"{word} padded source geometry changed")
            pcm = reader.readframes(256_000)
        samples = list(struct.unpack("<256000h", pcm))
        for index in range(lo, hi):
            samples[index] = _scale_sample(samples[index])
        body = struct.pack("<256000h", *samples)
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(body)
        outputs[word] = output.getvalue()
    metadata = {
        "schema": "qwen_lexical_calibrated_audio_manifest_v1",
        "original_manifest_sha256": render.sha(HERE / "RAW_MANIFEST_V1.json"),
        "original_wav_sha256": {word: render.sha(path)
                                for word, path in render.WAV.items()},
        "gain_numerator": GAIN_NUMERATOR,
        "gain_denominator": GAIN_DENOMINATOR,
        "rounding": "nearest_integer_ties_to_even_exact_rational",
        "edited_pcm_support": [lo, hi],
        "calibrated_wav_sha256": {word: hashlib.sha256(data).hexdigest()
                                  for word, data in outputs.items()},
        "selection": "processor_only_before_model_weights",
    }
    return outputs, metadata


def verify() -> dict:
    expected, metadata = make()
    for word, data in expected.items():
        path = WAV[word]
        if not path.is_file() or path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"Calibrated {word} WAV changed")
    if json.loads(MANIFEST.read_text()) != metadata:
        raise ValueError("Calibrated blue manifest changed")
    return metadata


if __name__ == "__main__":
    outputs, metadata = make()
    for word, data in outputs.items():
        WAV[word].write_bytes(data)
    MANIFEST.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
