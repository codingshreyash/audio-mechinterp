"""Independent no-weight check of the Whisper time-translation stimuli.

Reads only pinned local processor assets and the eight open non-speech WAVs.
Does not import the candidate time-translation source/plan or model weights.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ADMISSION = HERE / "NATURAL_TAIL_MULTILINGUAL_ADMISSION_V4_2.json"
SOURCE = HERE / "runs/natural_tail_waveform_v1/source_v1"
TRANSLATED = HERE / "runs/whisper_time_translation_v1/source_v1"
MANIFEST_SHA256 = "eec732fb24a4b238ad0666d52d168f5ba41ec38c9d315a057a39e0d53be1435b"
SOURCE_SHA256 = {
    "zero": "3e650da071ceb47c6a83b06bb00d13d8f4fde316dcc30d9cc2eb80fa16aa1220",
    "noise": "28fa81a018d66cbb23ee2e471d56def5ace2113a73230bacf71de699d02a275b",
}
SOURCE_WAV = {
    "zero": "d08_beep_train_low__z32000.wav",
    "noise": "d08_beep_train_low__n32000_s0.wav",
}
OFFSETS = (8000, 16000)
WINDOWS = ((50, 750), (850, 950))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def pcm(path: Path, digest: str, frames: int) -> bytes:
    require(path.is_file() and not path.is_symlink() and sha(path) == digest,
            "WAV identity changed")
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                 wav.getnframes()) == (1, 2, 16000, frames),
                "WAV geometry changed")
        data = wav.readframes(frames)
    require(len(data) == 2 * frames, "PCM data incomplete")
    return data


def features(processor: object, data: bytes) -> tuple[np.ndarray, np.ndarray]:
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / np.float32(32768)
    out = processor(samples, sampling_rate=16000, return_tensors="np",
                    return_attention_mask=True)
    mel = np.asarray(out.input_features)[0]
    mask = np.asarray(out.attention_mask)[0]
    require(mel.shape == (80, 3000) and mask.shape == (3000,)
            and np.isfinite(mel).all(), "processor geometry changed")
    return mel, mask


def main() -> None:
    from transformers import WhisperProcessor

    admission = json.loads(ADMISSION.read_bytes())
    snapshot = Path(admission["assets"]["snapshot"])
    require(snapshot.is_dir() and not snapshot.is_symlink()
            and snapshot.name == admission["assets"]["revision"],
            "pinned processor snapshot missing")
    small_assets = {name: digest for name, digest in
                    admission["assets"]["files_sha256"].items()
                    if name != "model.safetensors"}
    require(all(sha(snapshot / name) == digest for name, digest in small_assets.items()),
            "processor asset changed")
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    manifest_path = TRANSLATED / "manifest.json"
    require(sha(manifest_path) == MANIFEST_SHA256, "translated source manifest changed")
    manifest = json.loads(manifest_path.read_bytes())
    rows = {(row["kind"], row["layout"], row["offset_samples"]): row
            for row in manifest["rows"]}
    require(len(rows) == 8, "translated WAV census changed")
    report_rows = []
    for kind in ("zero", "noise"):
        original_data = pcm(SOURCE / SOURCE_WAV[kind], SOURCE_SHA256[kind], 160000)
        original_mel, original_mask = features(processor, original_data)
        require(int(original_mask.sum()) == 1000, "original active mask changed")
        for offset in OFFSETS:
            shift = offset // 160
            translated = {}
            for layout in ("prepend", "append"):
                row = rows[(kind, layout, offset)]
                path = TRANSLATED / row["wav"]
                raw = pcm(path, row["wav_sha256"], 160000 + offset)
                expected = ((bytes(2 * offset) + original_data) if layout == "prepend"
                            else (original_data + bytes(2 * offset)))
                require(raw == expected, "translated PCM reconstruction changed")
                translated[layout] = features(processor, raw)
            pre_mel, pre_mask = translated["prepend"]
            app_mel, app_mask = translated["append"]
            require(np.array_equal(pre_mask, app_mask)
                    and int(pre_mask.sum()) == 1000 + shift,
                    "equal-length attention mask changed")
            comparisons = []
            for start, end in WINDOWS:
                ref = original_mel[:, start:end]
                pre = pre_mel[:, start + shift:end + shift]
                app = app_mel[:, start:end]
                pre_equal = bool(np.array_equal(pre, ref))
                app_equal = bool(np.array_equal(app, ref))
                comparisons.append({"original_frame_start": start,
                                    "original_frame_end_exclusive": end,
                                    "prepend_bitwise_equal_after_shift": pre_equal,
                                    "append_bitwise_equal_at_fixed_time": app_equal,
                                    "prepend_max_abs_error": float(np.max(np.abs(pre-ref))),
                                    "append_max_abs_error": float(np.max(np.abs(app-ref)))})
            require(all(c["prepend_bitwise_equal_after_shift"]
                        and c["append_bitwise_equal_at_fixed_time"] for c in comparisons),
                    "interior mel translation changed")
            report_rows.append({"kind": kind, "offset_samples": offset,
                                "mel_frame_shift": shift,
                                "encoder_row_shift": shift // 2,
                                "active_mask_frames": int(pre_mask.sum()),
                                "comparisons": comparisons})
    report = {"schema": "whisper_time_translation_independent_preprocessing_v1",
              "status": "PASS_NO_WEIGHTS", "admission_sha256": sha(ADMISSION),
              "processor_snapshot": str(snapshot),
              "processor_revision": snapshot.name,
              "processor_asset_sha256": small_assets,
              "transformers_version": importlib.metadata.version("transformers"),
              "translated_manifest_sha256": MANIFEST_SHA256,
              "source_wav_sha256": SOURCE_SHA256,
              "translated_wav_sha256": {row["wav"]: row["wav_sha256"] for row in manifest["rows"]},
              "rows": report_rows, "model_weights_loaded": False}
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
