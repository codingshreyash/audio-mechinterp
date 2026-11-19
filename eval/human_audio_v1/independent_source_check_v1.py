"""Independent, read-only check of the four-speaker human source corpus.

This does not import the renderer or its auditor and does not load a target model.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ARCHIVE = Path.home() / ".cache/audio-mechinterp/librispeech/dev-clean.tar.gz"
SOURCE = HERE / "source_v1"
OFFICIAL_MD5 = "42e2234ba48799c1f50f24a7926300a1"
ARCHIVE_SHA256 = "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
MANIFEST_SHA256 = "c0593a4631191b7643714651c751fde29ebc6323bf1d59680d3a4c1d1bd38051"


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            h.update(block)
    return h.hexdigest()


def get_member(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str) -> bytes:
    item = members.get(name)
    need(item is not None and item.isfile() and not item.issym() and not item.islnk(),
         f"unsafe or missing archive member: {name}")
    stream = archive.extractfile(item)
    need(stream is not None, f"unreadable archive member: {name}")
    return stream.read()


def flac_pcm(data: bytes) -> bytes | None:
    with sf.SoundFile(io.BytesIO(data)) as sound:
        if not (sound.channels == 1 and sound.samplerate == 16000
                and sound.subtype == "PCM_16" and 32000 <= sound.frames <= 192000):
            return None
        values = sound.read(dtype="int16", always_2d=False)
    return np.asarray(values, dtype="<i2").tobytes()


def first_eligible(archive: tarfile.TarFile,
                   members: dict[str, tarfile.TarInfo], speaker: str) -> tuple[str, str, str, bytes]:
    choices: list[tuple[str, str, str]] = []
    prefix = f"LibriSpeech/dev-clean/{speaker}/"
    for name in sorted(members):
        if not (name.startswith(prefix) and name.endswith(".trans.txt")):
            continue
        for line in get_member(archive, members, name).decode("utf-8").splitlines():
            utterance, transcript = line.split(maxsplit=1)
            need(utterance.startswith(speaker + "-"), "transcript speaker changed")
            if 5 <= len(transcript.split()) <= 16:
                flac_name = str(PurePosixPath(name).parent / f"{utterance}.flac")
                choices.append((utterance, transcript, flac_name))
    for utterance, transcript, flac_name in sorted(choices):
        flac = get_member(archive, members, flac_name)
        pcm = flac_pcm(flac)
        if pcm is not None:
            return utterance, transcript, flac_name, pcm
    raise ValueError(f"no eligible utterance for {speaker}")


def check() -> dict:
    need(ARCHIVE.is_file() and not ARCHIVE.is_symlink(), "official archive not cached")
    need(file_digest(ARCHIVE, "md5") == OFFICIAL_MD5
         and file_digest(ARCHIVE) == ARCHIVE_SHA256, "archive digest changed")
    manifest_path = SOURCE / "manifest.json"
    need(file_digest(manifest_path) == MANIFEST_SHA256, "manifest digest changed")
    manifest = json.loads(manifest_path.read_bytes())
    need(manifest["schema"] == "human_audio_source_v1"
         and manifest["status"] == "source_only_no_model_inference"
         and manifest["archive_md5"] == OFFICIAL_MD5
         and manifest["archive_sha256"] == ARCHIVE_SHA256,
         "manifest identity changed")
    need(manifest["renderer_sha256"] == file_digest(HERE / "render_source_v1.py")
         and manifest["protocol_sha256"] == file_digest(HERE / "PROTOCOL.md"),
         "renderer or protocol bytes changed")
    rows = manifest["rows"]
    need(len(rows) == 4 and [row["split"] for row in rows]
         == ["development", "development", "sealed_test", "sealed_test"],
         "four-speaker split changed")
    expected_files = {"manifest.json", "LICENSE.TXT"}
    wav_hashes: list[str] = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        all_members = archive.getmembers()
        members = {member.name: member for member in all_members}
        need(len(members) == len(all_members), "duplicate archive member")
        speakers = {parts[2] for member in all_members
                    if (parts := member.name.split("/")) and len(parts) == 5
                    and parts[:2] == ["LibriSpeech", "dev-clean"]
                    and member.name.endswith(".trans.txt")}
        need(all(x.isdecimal() for x in speakers), "speaker IDs changed")
        ranked = sorted(speakers, key=lambda speaker: (
            digest(f"human-audio-v1:speaker:{speaker}".encode()), int(speaker)))[:4]
        need([row["speaker_id"] for row in rows] == ranked,
             "chosen speaker IDs or order changed")
        license_data = get_member(archive, members, "LibriSpeech/LICENSE.TXT")
        need(b"Creative Commons Attribution 4.0" in license_data
             and (SOURCE / "LICENSE.TXT").read_bytes() == license_data
             and digest(license_data) == manifest["license_sha256"]
             and manifest["license"] == "CC BY 4.0", "license changed")
        for row in rows:
            speaker = row["speaker_id"]
            utterance, transcript, flac_member, active = first_eligible(
                archive, members, speaker)
            need((row["utterance_id"], row["reference_transcript"], row["flac_member"])
                 == (utterance, transcript, flac_member),
                 f"not first eligible utterance: {speaker}")
            flac = get_member(archive, members, flac_member)
            flac_name = f"{utterance}.flac"
            need(row["flac"] == flac_name
                 and (SOURCE / flac_name).read_bytes() == flac
                 and row["flac_sha256"] == digest(flac)
                 and row["active_pcm_sha256"] == digest(active)
                 and row["active_frames"] == len(active) // 2
                 and row["sample_rate"] == 16000, "source FLAC/PCM changed")
            expected_files.add(flac_name)
            labels = ["base", "z32000", "n32000_s0", "n32000_s1"]
            need([condition["condition"] for condition in row["conditions"]] == labels,
                 "condition order changed")
            for condition in row["conditions"]:
                label = condition["condition"]
                name = f"{utterance}__{label}.wav"
                need(condition["wav"] == name and (SOURCE / name).is_file()
                     and not (SOURCE / name).is_symlink(), "WAV path changed")
                data = (SOURCE / name).read_bytes()
                need(condition["wav_sha256"] == digest(data), "WAV digest changed")
                with wave.open(io.BytesIO(data), "rb") as sound:
                    need((sound.getnchannels(), sound.getsampwidth(), sound.getframerate())
                         == (1, 2, 16000), "WAV format changed")
                    pcm = sound.readframes(sound.getnframes())
                    need(sound.getnframes() == condition["frames"], "WAV frame count changed")
                need(pcm.startswith(active) and digest(pcm) == condition["pcm_sha256"],
                     "active PCM or final PCM changed")
                tail = pcm[len(active):]
                if label == "base":
                    expected_tail = b""
                elif label == "z32000":
                    expected_tail = b"\0" * 64000
                else:
                    seed = int(label.rsplit("s", 1)[1])
                    raw = hashlib.shake_256(
                        f"human-audio-v1:{utterance}:{seed}".encode()).digest(32000)
                    samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) % 65 - 32)
                    expected_tail = np.asarray(samples, dtype="<i2").tobytes()
                need(tail == expected_tail and digest(tail) == condition["tail_sha256"],
                     "tail bytes changed")
                expected_files.add(name)
                wav_hashes.append(digest(data))
    need({path.name for path in SOURCE.iterdir() if path.is_file()} == expected_files,
         "rendered file census changed")
    need(len(set(wav_hashes)) == 16, "WAV identity collision")
    receipt_path = HERE / "SOURCE_AUDIT_V1.json"
    receipt = json.loads(receipt_path.read_bytes())
    need(receipt == {"schema": "human_audio_source_audit_v1", "status": "PASS",
                     "manifest_sha256": MANIFEST_SHA256, "archive_sha256": ARCHIVE_SHA256,
                     "speakers": ranked, "source_recordings": 4, "wavs": 16,
                     "file_count": 22}, "source audit receipt changed")
    return {"status": "PASS", "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": MANIFEST_SHA256, "speakers": ranked,
            "source_recordings": 4, "wavs": 16, "files": 22}


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True))
