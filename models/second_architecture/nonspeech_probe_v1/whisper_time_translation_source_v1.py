"""Deterministically render the prospective Whisper time-translation inputs.

This module uses only the two already-open, engineered beep-train WAVs. It
does not import Whisper or inspect model outputs. Each layout has the same
duration for a given offset: prefix silence moves the acoustic boundaries;
suffix silence keeps their original positions.
"""
from __future__ import annotations

import hashlib
import json
import os
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "runs/natural_tail_waveform_v1/source_v1"
OUTPUT = HERE / "runs/whisper_time_translation_v1/source_v1"
SOURCE = {
    "zero": ("d08_beep_train_low__z32000.wav",
             "3e650da071ceb47c6a83b06bb00d13d8f4fde316dcc30d9cc2eb80fa16aa1220"),
    "noise": ("d08_beep_train_low__n32000_s0.wav",
              "28fa81a018d66cbb23ee2e471d56def5ace2113a73230bacf71de699d02a275b"),
}
OFFSETS = (8_000, 16_000)
FRAMES = 160_000
SAMPLE_RATE = 16_000


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pcm(path: Path, expected_sha: str) -> bytes:
    require(path.is_file() and not path.is_symlink()
            and sha(path.read_bytes()) == expected_sha, "source WAV hash changed")
    with wave.open(str(path), "rb") as source:
        require((source.getnchannels(), source.getsampwidth(),
                 source.getframerate(), source.getnframes())
                == (1, 2, SAMPLE_RATE, FRAMES), "source WAV geometry changed")
        pcm = source.readframes(FRAMES)
    require(len(pcm) == FRAMES * 2, "source PCM incomplete")
    return pcm


def encode_wav(path: Path, pcm: bytes) -> None:
    require(len(pcm) % 2 == 0, "PCM alignment changed")
    with path.open("xb") as stream:
        with wave.open(stream, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm)
        stream.flush()
        os.fsync(stream.fileno())


def render(output_dir: Path = OUTPUT) -> dict:
    """Create an immutable 8-WAV source directory and return its manifest."""
    require(not output_dir.exists() and not output_dir.is_symlink(),
            "translated source path already consumed")
    source_pcm = {kind: read_pcm(ORIGINAL / name, expected)
                  for kind, (name, expected) in SOURCE.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for offset in OFFSETS:
        silence = bytes(offset * 2)
        for layout in ("prepend", "append"):
            for kind in ("zero", "noise"):
                pcm = (silence + source_pcm[kind] if layout == "prepend"
                       else source_pcm[kind] + silence)
                name = f"d08_beep_train_low__{layout}_{offset}__{kind}.wav"
                path = output_dir / name
                encode_wav(path, pcm)
                with wave.open(str(path), "rb") as audio:
                    require((audio.getnchannels(), audio.getsampwidth(),
                             audio.getframerate(), audio.getnframes())
                            == (1, 2, SAMPLE_RATE, FRAMES + offset),
                            "rendered WAV geometry changed")
                    require(audio.readframes(FRAMES + offset) == pcm,
                            "rendered PCM reconstruction failed")
                rows.append({"layout": layout, "offset_samples": offset,
                             "kind": kind, "wav": name,
                             "frames": FRAMES + offset,
                             "wav_sha256": sha(path.read_bytes()),
                             "pcm_sha256": sha(pcm)})
    manifest = {"schema": "whisper_time_translation_source_v1",
                "status": "rendered_no_model_inference",
                "source": {kind: {"wav": name, "wav_sha256": expected,
                                   "pcm_sha256": sha(source_pcm[kind])}
                           for kind, (name, expected) in SOURCE.items()},
                "rows": rows}
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with (output_dir / "manifest.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def verify(output_dir: Path = OUTPUT) -> dict:
    """Rebuild every expected PCM in memory and check all saved bytes."""
    manifest = json.loads((output_dir / "manifest.json").read_bytes())
    require(manifest.get("schema") == "whisper_time_translation_source_v1"
            and len(manifest.get("rows", [])) == 8, "source manifest changed")
    source_pcm = {kind: read_pcm(ORIGINAL / name, expected)
                  for kind, (name, expected) in SOURCE.items()}
    expected_rows = []
    for offset in OFFSETS:
        silence = bytes(offset * 2)
        for layout in ("prepend", "append"):
            for kind in ("zero", "noise"):
                name = f"d08_beep_train_low__{layout}_{offset}__{kind}.wav"
                path = output_dir / name
                pcm = silence + source_pcm[kind] if layout == "prepend" else source_pcm[kind] + silence
                require(path.is_file() and not path.is_symlink(), "translated WAV missing")
                with wave.open(str(path), "rb") as audio:
                    require((audio.getnchannels(), audio.getsampwidth(),
                             audio.getframerate(), audio.getnframes())
                            == (1, 2, SAMPLE_RATE, FRAMES + offset)
                            and audio.readframes(FRAMES + offset) == pcm,
                            "translated PCM differs")
                expected_rows.append({"layout": layout, "offset_samples": offset,
                                      "kind": kind, "wav": name,
                                      "frames": FRAMES + offset,
                                      "wav_sha256": sha(path.read_bytes()),
                                      "pcm_sha256": sha(pcm)})
    require(manifest["rows"] == expected_rows
            and set(p.name for p in output_dir.iterdir()) ==
            {"manifest.json", *(r["wav"] for r in expected_rows)},
            "translated source manifest/inventory differs")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("render", "verify"))
    args = parser.parse_args()
    result = render() if args.mode == "render" else verify()
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]),
                      "manifest_sha256": sha((OUTPUT / "manifest.json").read_bytes())},
                     sort_keys=True))
