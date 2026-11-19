"""Deterministic LibriSpeech human-speech source renderer; no model inference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = Path.home() / ".cache/audio-mechinterp/librispeech/dev-clean.tar.gz"
DEFAULT_OUTPUT = HERE / "source_v1"
ARCHIVE_MD5 = "42e2234ba48799c1f50f24a7926300a1"
ARCHIVE_SHA256 = "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
TAIL_FRAMES = 32000
SAMPLE_RATE = 16000
SPEAKERS = 4


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path, name: str) -> str:
    h = hashlib.new(name)
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def sorted_speakers(members: dict[str, tarfile.TarInfo]) -> list[str]:
    prefix = "LibriSpeech/dev-clean/"
    ids = {name[len(prefix):].split("/", 1)[0] for name in members
           if name.startswith(prefix) and name.endswith(".trans.txt")
           and len(name[len(prefix):].split("/")) == 3}
    require(len(ids) >= SPEAKERS and all(x.isdecimal() for x in ids),
            "dev-clean speaker set invalid")
    return sorted(ids, key=lambda x: (
        sha256_bytes(f"human-audio-v1:speaker:{x}".encode()), int(x)))[:SPEAKERS]


def read_member(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo],
                name: str) -> bytes:
    info = members.get(name)
    require(info is not None and info.isfile() and not info.issym() and not info.islnk(),
            f"archive member missing or unsafe: {name}")
    stream = archive.extractfile(info)
    require(stream is not None, f"cannot read archive member: {name}")
    return stream.read()


def transcripts_for_speaker(archive: tarfile.TarFile,
                            members: dict[str, tarfile.TarInfo], speaker: str) -> list[tuple[str, str, str]]:
    prefix = f"LibriSpeech/dev-clean/{speaker}/"
    entries: list[tuple[str, str, str]] = []
    for name in sorted(members):
        if not name.startswith(prefix) or not name.endswith(".trans.txt"):
            continue
        text = read_member(archive, members, name).decode("utf-8")
        for line in text.splitlines():
            parts = line.strip().split(maxsplit=1)
            require(len(parts) == 2 and parts[0].startswith(speaker + "-"),
                    f"bad transcript line: {name}")
            utterance, transcript = parts
            flac_name = str(Path(name).parent / f"{utterance}.flac")
            entries.append((utterance, transcript, flac_name))
    return sorted(entries)


def decode_flac(data: bytes) -> np.ndarray:
    with sf.SoundFile(io.BytesIO(data), mode="r") as audio:
        require(audio.channels == 1 and audio.samplerate == SAMPLE_RATE
                and audio.subtype == "PCM_16", "FLAC geometry/subtype invalid")
        pcm = audio.read(dtype="int16", always_2d=False)
    require(pcm.ndim == 1 and 2 * SAMPLE_RATE <= len(pcm) <= 12 * SAMPLE_RATE,
            "source duration outside frozen 2–12-second window")
    return np.ascontiguousarray(pcm, dtype="<i2")


def pick_one(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo],
             speaker: str) -> tuple[str, str, str, bytes, np.ndarray]:
    for utterance, transcript, flac_name in transcripts_for_speaker(archive, members, speaker):
        if not 5 <= len(transcript.split()) <= 16:
            continue
        flac = read_member(archive, members, flac_name)
        try:
            pcm = decode_flac(flac)
        except ValueError as error:
            if str(error) == "source duration outside frozen 2–12-second window":
                continue
            raise
        return utterance, transcript, flac_name, flac, pcm
    raise ValueError(f"no eligible utterance for selected speaker {speaker}")


def noise_tail(utterance: str, seed: int) -> np.ndarray:
    require(seed in (0, 1), "unfrozen seed")
    raw = hashlib.shake_256(f"human-audio-v1:{utterance}:{seed}".encode()).digest(TAIL_FRAMES)
    values = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
    return np.ascontiguousarray(values % 65 - 32, dtype="<i2")


def wav_bytes(pcm: np.ndarray) -> bytes:
    require(pcm.ndim == 1 and pcm.dtype == np.dtype("<i2"), "PCM16 array expected")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def render(archive_path: Path, output: Path) -> dict:
    require(archive_path.is_file() and not archive_path.is_symlink(), "archive unavailable")
    require(hash_file(archive_path, "md5") == ARCHIVE_MD5
            and hash_file(archive_path, "sha256") == ARCHIVE_SHA256,
            "official archive checksums changed")
    require(not output.exists(), "one-shot output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".human_audio_source_v1_", dir=output.parent))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            require(len(members) == len(archive.getmembers()), "duplicate archive member names")
            license_data = read_member(archive, members, "LibriSpeech/LICENSE.TXT")
            require(b"Creative Commons Attribution 4.0" in license_data,
                    "archive license changed")
            (staging / "LICENSE.TXT").write_bytes(license_data)
            chosen = sorted_speakers(members)
            rows = []
            for index, speaker in enumerate(chosen):
                utterance, transcript, flac_name, flac, active = pick_one(archive, members, speaker)
                flac_out = f"{utterance}.flac"
                (staging / flac_out).write_bytes(flac)
                active_bytes = active.tobytes()
                zero = np.zeros(TAIL_FRAMES, dtype="<i2")
                tail0, tail1 = noise_tail(utterance, 0), noise_tail(utterance, 1)
                require(not np.array_equal(tail0, tail1), "noise seeds collided")
                conditions = []
                for label, tail in (("base", np.zeros(0, dtype="<i2")),
                                    ("z32000", zero), ("n32000_s0", tail0),
                                    ("n32000_s1", tail1)):
                    pcm = np.concatenate((active, tail)).astype("<i2", copy=False)
                    require(pcm.tobytes()[:len(active_bytes)] == active_bytes,
                            "active PCM changed")
                    name = f"{utterance}__{label}.wav"
                    final = wav_bytes(pcm)
                    (staging / name).write_bytes(final)
                    conditions.append({"condition": label, "wav": name,
                                       "wav_sha256": sha256_bytes(final),
                                       "pcm_sha256": sha256_bytes(pcm.tobytes()),
                                       "tail_sha256": sha256_bytes(tail.tobytes()),
                                       "frames": int(len(pcm))})
                rows.append({"speaker_id": speaker,
                             "split": "development" if index < 2 else "sealed_test",
                             "utterance_id": utterance,
                             "reference_transcript": transcript,
                             "flac_member": flac_name,
                             "flac": flac_out,
                             "flac_sha256": sha256_bytes(flac),
                             "active_pcm_sha256": sha256_bytes(active_bytes),
                             "active_frames": int(len(active)),
                             "sample_rate": SAMPLE_RATE,
                             "conditions": conditions})
        require(len(rows) == 4 and all(len(row["conditions"]) == 4 for row in rows),
                "source denominator changed")
        manifest = {"schema": "human_audio_source_v1", "status": "source_only_no_model_inference",
                    "source_url": "https://openslr.elda.org/resources/12/dev-clean.tar.gz",
                    "license": "CC BY 4.0", "archive_md5": ARCHIVE_MD5,
                    "archive_sha256": ARCHIVE_SHA256,
                    "license_sha256": sha256_bytes(license_data),
                    "renderer_sha256": hash_file(Path(__file__), "sha256"),
                    "protocol_sha256": hash_file(HERE / "PROTOCOL.md", "sha256"),
                    "tail_frames": TAIL_FRAMES,
                    "rows": rows}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        os.rename(staging, output)
        return manifest
    except Exception:
        # Keep a failed staging directory for forensics; no partial final output.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = render(args.archive, args.output)
    print(json.dumps({"status": data["status"], "rows": len(data["rows"]),
                      "wavs": sum(len(row["conditions"]) for row in data["rows"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
