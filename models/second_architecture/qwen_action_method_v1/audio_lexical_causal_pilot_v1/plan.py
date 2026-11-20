"""Frozen FIT-only Qwen red/blue causal pilot; no model inference on import."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import calibrate, render

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ATTEMPT = HERE.parent / "runs/audio_lexical_causal_pilot_v1/attempt_001"
ADMISSION = HERE / "ADMISSION_V1.json"
REVIEW = HERE / "INDEPENDENT_EXECUTABLE_REVIEW_V1.md"
GO = HERE / "GO_V1.json"
PROCESSOR_RECEIPT = HERE / "PROCESSOR_ONLY_RECEIPT_V1.json"
PROCESSOR_RECEIPT_SHA256 = "fee2f98a6e32ffa8137198d29b0f5adf171912396fc794d57c5f0a03c453800c"

CALIBRATED = calibrate.WAV
ORIGINAL = render.WAV
EXPECTED_WAV_SHA256 = {
    "calibrated": {"red": "df3927b62935c6fa27de3e6d4a137965e547f94bb3f3a6aa68886ae37c274214",
                   "blue": "d1c93162f9fdf0976ca2a4d600b82397e62828a4e1073e6fe6e6a86b7c43f17a"},
    "original": {"red": "4c8a5174ba10a634777332dd46b23604e7351b4bf2b39f1300986becffd94d5e",
                 "blue": "a1cd7aaa954c764d058e0b23bd9150c94e7ee3c71e153c1251dc0ec5ee7a7d77"},
}

SITES = ("audio.pre_attention.0", "audio.to_language")
# Exact first-chunk row windows, derived from the frozen mel support and pinned
# conv receptive field. After attention, these are nominal rows, not a phoneme.
WINDOWS = {
    "audio.pre_attention.0": {"local": [58, 81], "chunk": [0, 100],
                              "relocated": [0, 23], "total": 800, "width": 1280},
    "audio.to_language": {"local": [29, 41], "chunk": [0, 50],
                          "relocated": [0, 12], "total": 400, "width": 2048},
}
ARMS = ("local_donor", "chunk_donor", "full_donor", "self",
        "relocated_delta", "equal_l2_noise")
WORDS = ("red", "blue")
MAX_NEW_TOKENS = 24
EOS_ID = 151645
FIRST_LOGIT_TOLERANCE = 1e-5
MIN_LOCAL_DELTA_L2 = 0.01
MAX_PREPATCH_DRIFT_FRACTION = 0.01
SOURCE_DRIFT_TOLERANCE = 1e-5
OUTSIDE_SUPPORT_TOLERANCE = 1e-5
MAX_SECONDS = 900
GPU_BYTES = 60 * 1024**3
RSS_BYTES = 24 * 1024**3
EVIDENCE_BYTES = 256 * 1024**2
NOISE_SEED = 731902
SYSTEM_PROMPT = "Listen to the audio. Reply with the single color word spoken, and nothing else."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> dict:
    original = render.verify()
    calibrated = calibrate.verify()
    for family, paths in (("original", ORIGINAL), ("calibrated", CALIBRATED)):
        for word, path in paths.items():
            if sha(path) != EXPECTED_WAV_SHA256[family][word]:
                raise ValueError(f"Frozen {family}/{word} WAV hash changed")
    if sha(PROCESSOR_RECEIPT) != PROCESSOR_RECEIPT_SHA256:
        raise ValueError("Processor-only receipt changed")
    receipt = json.loads(PROCESSOR_RECEIPT.read_text())
    if (receipt.get("schema") != "qwen_lexical_processor_only_v1"
            or receipt.get("system_prompt") != SYSTEM_PROMPT
            or receipt["comparison"]["calibrated"]["feature_difference_support_exclusive"] != [118, 159]
            or receipt["comparison"]["calibrated"]["changed_feature_columns"] != 41
            or receipt["comparison"]["calibrated"]["max_abs_padding_difference"] != 0):
        raise ValueError("Processor-only feature receipt invalid")
    return {"original": original, "calibrated": calibrated,
            "processor_receipt_sha256": PROCESSOR_RECEIPT_SHA256,
            "wav_sha256": EXPECTED_WAV_SHA256}


def verify_runtime_processor_receipt(receipt: dict) -> None:
    """Check platform-stable properties against the frozen Mac receipt.

    Whisper FFT/mel arithmetic is not bitwise portable between Mac and Linux.
    Admission binds the *complete* Linux receipt after this no-weight check.
    """
    saved = json.loads(PROCESSOR_RECEIPT.read_text())
    if (receipt.get("schema") != saved["schema"]
            or receipt.get("model_revision") != saved["model_revision"]
            or receipt.get("system_prompt") != saved["system_prompt"]
            or set(receipt.get("records", {})) != {"original", "calibrated"}):
        raise ValueError("Runtime processor receipt identity changed")
    for family in ("original", "calibrated"):
        if set(receipt["records"][family]) != {"red", "blue"}:
            raise ValueError("Runtime processor pair incomplete")
        for word in WORDS:
            actual_record = json.loads(json.dumps(receipt["records"][family][word]))
            saved_record = json.loads(json.dumps(saved["records"][family][word]))
            # Only this tensor may have platform-dependent STFT/mel bytes.
            for row in (actual_record, saved_record):
                row["tensor_sha256"].pop("input_features")
            if actual_record != saved_record:
                raise ValueError(f"Runtime processor geometry differs: {family}/{word}")
    comp = receipt.get("comparison", {}).get("calibrated", {})
    if (comp.get("feature_difference_support_exclusive") != [118, 159]
            or comp.get("changed_feature_columns") != 41
            or comp.get("max_abs_padding_difference") != 0.0
            or comp.get("red_feature_max") != comp.get("blue_feature_max")
            or comp.get("red_padding_floor") != comp.get("blue_padding_floor")
            or not all(isinstance(comp.get(key), (int, float))
                       and math.isfinite(comp[key]) for key in
                       ("red_feature_max", "blue_feature_max", "red_padding_floor",
                        "blue_padding_floor", "max_abs_feature_difference"))
            or comp["max_abs_feature_difference"] <= 0):
        raise ValueError("Runtime calibrated features lost their local support")


def verify_record(family: str, word: str, record: dict, runtime_receipt: dict) -> None:
    saved = runtime_receipt["records"][family][word]
    if record != saved:
        raise ValueError(f"Processor batch differs from admitted {family}/{word} receipt")


def validate_windows() -> None:
    for site, windows in WINDOWS.items():
        lo, hi = windows["local"]
        a, b = windows["chunk"]
        x, y = windows["relocated"]
        if not (0 <= a <= x < y <= lo < hi <= b <= windows["total"]
                and hi-lo == y-x):
            raise ValueError(f"Invalid frozen row windows: {site}")


validate_windows()
