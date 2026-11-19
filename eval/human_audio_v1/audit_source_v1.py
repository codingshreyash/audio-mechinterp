"""Read-only byte and geometry audit for the four-speaker source render."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_v1"
ARCHIVE = Path.home() / ".cache/audio-mechinterp/librispeech/dev-clean.tar.gz"
EXPECTED_MD5 = "42e2234ba48799c1f50f24a7926300a1"
EXPECTED_SHA256 = "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"


def h_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def h_file(path: Path, name: str = "sha256") -> str:
    digest = hashlib.new(name)
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def need(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(what)


def member_bytes(tf: tarfile.TarFile, name: str) -> bytes:
    member = tf.getmember(name)
    need(member.isfile() and not member.issym() and not member.islnk(), "unsafe archive member")
    source = tf.extractfile(member)
    need(source is not None, "archive member unreadable")
    return source.read()


def expected_noise(utterance: str, seed: int) -> bytes:
    data = hashlib.shake_256(f"human-audio-v1:{utterance}:{seed}".encode()).digest(32000)
    samples = (np.frombuffer(data, dtype=np.uint8).astype(np.int16) % 65 - 32).astype("<i2")
    return samples.tobytes()


def audit(archive: Path, source: Path) -> dict:
    need(archive.is_file() and not archive.is_symlink(), "archive unavailable")
    need(h_file(archive, "md5") == EXPECTED_MD5
         and h_file(archive) == EXPECTED_SHA256, "official archive digest mismatch")
    manifest_path = source / "manifest.json"
    need(manifest_path.is_file() and not manifest_path.is_symlink(), "manifest unavailable")
    data = json.loads(manifest_path.read_text())
    need(data.get("schema") == "human_audio_source_v1"
         and data.get("status") == "source_only_no_model_inference"
         and data.get("archive_md5") == EXPECTED_MD5
         and data.get("archive_sha256") == EXPECTED_SHA256,
         "manifest source identity mismatch")
    rows = data["rows"]
    need(len(rows) == 4 and len({r["speaker_id"] for r in rows}) == 4,
         "speaker denominator changed")
    need([r["split"] for r in rows] == ["development", "development", "sealed_test", "sealed_test"],
         "speaker split changed")
    speaker_order = sorted((r["speaker_id"] for r in rows), key=lambda s: (
        hashlib.sha256(f"human-audio-v1:speaker:{s}".encode()).hexdigest(), int(s)))
    need([r["speaker_id"] for r in rows] == speaker_order, "speaker ordering changed")
    observed_files = {p.name for p in source.iterdir() if p.is_file()}
    expected_files = {"manifest.json", "LICENSE.TXT"}
    with tarfile.open(archive, "r:gz") as tf:
        all_speakers = {p[2] for member in tf.getmembers()
                        if (p := member.name.split("/")) and len(p) == 5
                        and p[:2] == ["LibriSpeech", "dev-clean"]
                        and member.name.endswith(".trans.txt")}
        frozen_speakers = sorted(all_speakers, key=lambda s: (
            hashlib.sha256(f"human-audio-v1:speaker:{s}".encode()).hexdigest(), int(s)))[:4]
        need([r["speaker_id"] for r in rows] == frozen_speakers,
             "frozen four-speaker selection changed")
        license_data = member_bytes(tf, "LibriSpeech/LICENSE.TXT")
        need((source / "LICENSE.TXT").read_bytes() == license_data
             and h_bytes(license_data) == data["license_sha256"], "license changed")
        for row in rows:
            need(row["sample_rate"] == 16000, "sample rate changed")
            utterance = row["utterance_id"]
            need(utterance.startswith(row["speaker_id"] + "-"), "speaker/utterance mismatch")
            flac = member_bytes(tf, row["flac_member"])
            need(row["flac"] == f"{utterance}.flac"
                 and (source / row["flac"]).read_bytes() == flac
                 and h_bytes(flac) == row["flac_sha256"], "FLAC source mismatch")
            expected_files.add(row["flac"])
            trans_name = str(Path(row["flac_member"]).parent /
                             f"{'-'.join(utterance.split('-')[:2])}.trans.txt")
            lines = member_bytes(tf, trans_name).decode().splitlines()
            matched = [line.split(maxsplit=1)[1] for line in lines
                       if line.split(maxsplit=1)[0] == utterance]
            need(matched == [row["reference_transcript"]], "reference transcript mismatch")
            with sf.SoundFile(io.BytesIO(flac)) as input_audio:
                need(input_audio.channels == 1 and input_audio.samplerate == 16000
                     and input_audio.subtype == "PCM_16", "FLAC geometry mismatch")
                active = np.asarray(input_audio.read(dtype="int16"), dtype="<i2").tobytes()
            need(len(active) == row["active_frames"] * 2
                 and h_bytes(active) == row["active_pcm_sha256"], "active PCM mismatch")
            conditions = row["conditions"]
            need([c["condition"] for c in conditions] ==
                 ["base", "z32000", "n32000_s0", "n32000_s1"], "condition census changed")
            for condition in conditions:
                name = condition["wav"]
                wav_path = source / name
                need(name == f"{utterance}__{condition['condition']}.wav"
                     and wav_path.is_file() and not wav_path.is_symlink(), "WAV missing")
                wav_data = wav_path.read_bytes()
                need(h_bytes(wav_data) == condition["wav_sha256"], "WAV hash mismatch")
                with wave.open(io.BytesIO(wav_data), "rb") as output_audio:
                    need((output_audio.getnchannels(), output_audio.getsampwidth(),
                          output_audio.getframerate(), output_audio.getnframes()) ==
                         (1, 2, 16000, condition["frames"]), "WAV geometry mismatch")
                    pcm = output_audio.readframes(output_audio.getnframes())
                need(h_bytes(pcm) == condition["pcm_sha256"] and pcm.startswith(active),
                     "WAV active prefix or PCM hash mismatch")
                tail = pcm[len(active):]
                label = condition["condition"]
                expected_tail = (b"" if label == "base" else
                                 b"\0" * 64000 if label == "z32000" else
                                 expected_noise(utterance, int(label[-1])))
                need(tail == expected_tail and h_bytes(tail) == condition["tail_sha256"],
                     "tail construction mismatch")
                expected_files.add(name)
    need(observed_files == expected_files, "source file census changed")
    return {"schema": "human_audio_source_audit_v1", "status": "PASS",
            "manifest_sha256": h_file(manifest_path),
            "archive_sha256": EXPECTED_SHA256,
            "speakers": [r["speaker_id"] for r in rows],
            "source_recordings": len(rows),
            "wavs": sum(len(r["conditions"]) for r in rows),
            "file_count": len(expected_files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=HERE / "SOURCE_AUDIT_V1.json")
    args = parser.parse_args()
    result = audit(args.archive, args.source)
    with args.output.open("x") as out:
        out.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
