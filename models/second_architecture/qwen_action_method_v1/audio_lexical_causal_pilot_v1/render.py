"""Reconstruct two canonical 16-second PCM16 WAVs from the controlled Piper pair."""
from __future__ import annotations

import hashlib
import io
import json
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_MANIFEST = ROOT / "eval/tts/outputs/controlled_pair_v1/controlled_pair_manifest.json"
SOURCE = {word: ROOT / f"eval/tts/outputs/controlled_pair_v1/piper_{word}.wav"
          for word in ("red", "blue")}
WAV = {word: HERE / f"piper_{word}_zero_pad_16s.wav" for word in SOURCE}
SAMPLES = 256_000
SPLICE_START = 18_082
SPLICE_END_EXCLUSIVE = 26_626


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                reader.getnframes()) != (1, 2, 16_000, 49_724):
            raise ValueError(f"Unexpected source WAV geometry: {path}")
        pcm = reader.readframes(49_724)
    if len(pcm) != 99_448:
        raise ValueError("Source PCM incomplete")
    return pcm


def render() -> tuple[dict[str, bytes], dict]:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    entries = source_manifest["engines"]["piper"]["outputs"]
    pcm = {word: _pcm(path) for word, path in SOURCE.items()}
    for word, path in SOURCE.items():
        if sha(path) != entries[word]["file_sha256"]:
            raise ValueError("Source manifest WAV hash mismatch")
    first = next(i for i in range(49_724) if pcm["red"][i*2:i*2+2] != pcm["blue"][i*2:i*2+2])
    last = next(i for i in range(49_723, -1, -1)
                if pcm["red"][i*2:i*2+2] != pcm["blue"][i*2:i*2+2]) + 1
    if (first, last) != (SPLICE_START, SPLICE_END_EXCLUSIVE):
        raise ValueError("Observed PCM difference support changed")
    outputs = {}
    for word, body in pcm.items():
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(body + bytes((SAMPLES-49_724)*2))
        outputs[word] = buffer.getvalue()
    metadata = {"schema": "qwen_lexical_audio_manifest_v1", "source_manifest_sha256": sha(SOURCE_MANIFEST),
                "source_wav_sha256": {word: sha(path) for word, path in SOURCE.items()},
                "source_pcm_samples": 49_724, "final_pcm_samples": SAMPLES,
                "sample_rate": 16_000, "pcm_difference_support": [first, last],
                "final_wav_sha256": {word: hashlib.sha256(data).hexdigest()
                                     for word, data in outputs.items()},
                "claim_scope": "fit-only synthetic speech; no independent human lexical QA"}
    return outputs, metadata


def verify() -> dict:
    outputs, metadata = render()
    for word, expected in outputs.items():
        path = WAV[word]
        if not path.is_file() or path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"Frozen padded WAV changed: {word}")
    saved = json.loads((HERE / "RAW_MANIFEST_V1.json").read_text())
    if saved != metadata:
        raise ValueError("Frozen raw manifest changed")
    return metadata


if __name__ == "__main__":
    outputs, metadata = render()
    for word, body in outputs.items():
        WAV[word].write_bytes(body)
    (HERE / "RAW_MANIFEST_V1.json").write_text(json.dumps(metadata, sort_keys=True, indent=2)+"\n")
