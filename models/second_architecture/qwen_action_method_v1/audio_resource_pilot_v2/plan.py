"""Frozen non-study audio stimulus and one-shot technical pilot contract."""
from __future__ import annotations

import hashlib
import io
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "models/qwen_omni/sample_apple_5p5s.wav"
WAV = HERE.parent / "audio_resource_pilot_v1/nonstudy_apple_zero_pad_16s.wav"
SOURCE_SHA256 = "ba226916abdd36bef212a0b478d49eb63028fb541d84a7934f50e4871fe6d21d"
WAV_SHA256 = "b84ce83e90ad498e93063a27e8cba9a3a860985e5801c55385a95e0483735973"
CASE_ID = "NON_STUDY_APPLE_ZERO_PAD_16S"
SITES = ("text.residual.17", "text.residual.34")
ATTEMPT = HERE.parent / "runs/audio_resource_pilot_v2/attempt_002"
ADMISSION = HERE / "ADMISSION_V2.json"  # Absent until a separate admission.
REVIEW = HERE / "INDEPENDENT_EXECUTABLE_REVIEW_V2.md"  # Absent now.
GO = HERE / "GO_V2.json"  # Separate committed GO after independent executable review.
MAX_SECONDS = 900
GPU_BYTES = 60 * 1024**3
RSS_BYTES = 24 * 1024**3
EVIDENCE_BYTES = 64 * 1024**2
MAX_NEW_TOKENS = 12
FIRST_LOGIT_TOLERANCE = 1e-5
SYSTEM_PROMPT = "Respond briefly to the user's audio."
EXPECTED_BATCH_SHAPES = {"input_ids": [1, 423], "attention_mask": [1, 423],
                         "input_features": [1, 128, 30000],
                         "feature_attention_mask": [1, 30000]}
EXPECTED_MEL_FRAMES = 1600
EXPECTED_ENCODER_ROWS = 800
EXPECTED_PROJECTED_ROWS = 400
EXPECTED_AUDIO_POSITIONS = list(range(17, 417))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> bytes:
    """Build canonical PCM16 WAV bytes; no study script or D/H case is used."""
    if sha(SOURCE) != SOURCE_SHA256:
        raise ValueError("non-study source WAV changed")
    with wave.open(str(SOURCE), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                reader.getnframes()) != (1, 2, 16000, 88000):
            raise ValueError("non-study source PCM geometry changed")
        pcm = reader.readframes(88000)
    if len(pcm) != 176000:
        raise ValueError("non-study source PCM incomplete")
    data = io.BytesIO()
    with wave.open(data, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm + bytes((256000 - 88000) * 2))
    return data.getvalue()


def verify_wav() -> dict:
    if not WAV.is_file() or WAV.is_symlink() or sha(WAV) != WAV_SHA256:
        raise ValueError("frozen non-study WAV missing or changed")
    if WAV.read_bytes() != render():
        raise ValueError("non-study WAV differs from deterministic renderer")
    return {"case_id": CASE_ID, "source_sha256": SOURCE_SHA256,
            "wav_sha256": WAV_SHA256, "sample_rate": 16000,
            "pcm_samples": 256000, "duration_seconds": 16}


def verify_geometry(record: dict) -> None:
    """Freeze processor-only sizing observed before target inference."""
    if (record.get("batch_shapes") != EXPECTED_BATCH_SHAPES
            or record.get("mel_frames") != EXPECTED_MEL_FRAMES
            or record.get("encoder_row_count") != EXPECTED_ENCODER_ROWS
            or record.get("projected_row_count") != EXPECTED_PROJECTED_ROWS
            or record.get("audio_placeholder_positions") != EXPECTED_AUDIO_POSITIONS):
        raise ValueError("non-study 16-second processor geometry changed")
