"""Prepare a future rendered WAV: exactly five seconds of appended PCM zeros.

This source-only utility is not invoked by fixture preparation. The caller
must first convert/render a single utterance to mono 16-kHz signed PCM16 WAV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave


ZERO_FRAMES = 5 * 16000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, destination: Path) -> dict:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    if destination.exists():
        raise FileExistsError(destination)
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if (params.nchannels, params.sampwidth, params.framerate, params.comptype) != (1, 2, 16000, "NONE"):
            raise ValueError("Expected mono 16-kHz PCM16 WAV")
        original_frames = reader.readframes(params.nframes)
    # Remove only exact terminal zero samples from the raw render. No nonzero
    # samples, including very quiet speech, are removed.
    end = len(original_frames)
    while end >= 2 and original_frames[end - 2:end] == b"\x00\x00":
        end -= 2
    trimmed = original_frames[:end]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(trimmed + b"\x00\x00" * ZERO_FRAMES)
    with wave.open(str(destination), "rb") as reader:
        final_frames = reader.readframes(reader.getnframes())
        final_count = reader.getnframes()
    if final_frames[-ZERO_FRAMES * 2:] != b"\x00\x00" * ZERO_FRAMES:
        raise RuntimeError("Trailing silence validation failed")
    if len(trimmed) >= 2 and trimmed[-2:] == b"\x00\x00":
        raise RuntimeError("Pre-padding audio still ends with zero sample")
    return {"source": str(source), "source_sha256": sha256(source),
            "destination": str(destination), "destination_sha256": sha256(destination),
            "source_frames": len(original_frames) // 2,
            "removed_existing_zero_frames": (len(original_frames) - len(trimmed)) // 2,
            "appended_zero_frames": ZERO_FRAMES, "final_frames": final_count,
            "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
