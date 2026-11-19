"""Source-bound V3 development gate over raw token and score records.

No model, processor, or audio dependency. V1/V2 remain immutable HOLD records.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np

from . import endpoint_v2, readout_v2
from .endpoint import LABELS, SPEC_PATH, load_spec

V1_SPEC_SHA256 = "a6ddd9b007649173d23e38996883920467ef861f15b4491acde3cdd096e4ef18"
V2_FREEZE_SHA256 = "eb4b4f5d12827d2d2055d6e97d44b01e3a32d66b20514bbfdc41a16b07178d91"
V2_FREEZE_PATH = SPEC_PATH.with_name("READOUT_FREEZE_V2.json")
REQUIRED = {"case_id", "continuation_ids", "actual_scores", "vjp_scores"}
ASSERTIONS = {"status", "label", "canonical_label_path", "predicted_label",
              "actual_score_top"}


def _expected_rows() -> tuple[list[str], list[str], list[str], list[str]]:
    if sha256(SPEC_PATH.read_bytes()).hexdigest() != V1_SPEC_SHA256:
        raise ValueError("V1 script manifest bytes changed")
    if sha256(V2_FREEZE_PATH.read_bytes()).hexdigest() != V2_FREEZE_SHA256:
        raise ValueError("V2 readout freeze bytes changed")
    spec = load_spec()
    fit = [row for row in spec["scripts"] if row["split"] == "fit_text"]
    development = [row for row in spec["scripts"] if row["split"] == "development"]
    return ([f"{row['id']}__typed" for row in fit],
            [row["label"] for row in development],
            [f"{row['id']}__typed" for row in development],
            [f"{row['id']}__piper_en_US_amy_medium" for row in development])


def _fit_scores(records: list[dict[str, Any]], expected_ids: list[str]) -> np.ndarray:
    if len(records) != 18 or [row.get("case_id") for row in records] != expected_ids:
        raise ValueError("FIT manifest IDs/order changed")
    values = []
    for row in records:
        if set(row) != {"case_id", "scores"}:
            raise ValueError("FIT score record fields changed")
        value = np.asarray(row["scores"], dtype=np.float64)
        if value.shape != (len(LABELS),) or not np.isfinite(value).all():
            raise ValueError("FIT score shape/value invalid")
        values.append(value)
    return np.stack(values)


def _rows(
    records: list[dict[str, Any]], expected_ids: list[str], tokenizer: Any
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if len(records) != 6 or [row.get("case_id") for row in records] != expected_ids:
        raise ValueError("development manifest IDs/order changed")
    verified = []
    actual, vjp = [], []
    for row in records:
        if set(row) - (REQUIRED | ASSERTIONS) or not REQUIRED.issubset(row):
            raise ValueError("unexpected or missing development record field")
        endpoint = endpoint_v2.score_complete_answer(row["continuation_ids"], tokenizer)
        actual_row = np.asarray(row["actual_scores"], dtype=np.float64)
        vjp_row = np.asarray(row["vjp_scores"], dtype=np.float64)
        if (actual_row.shape != (len(LABELS),) or vjp_row.shape != (len(LABELS),)
                or not np.isfinite(actual_row).all() or not np.isfinite(vjp_row).all()):
            raise ValueError("invalid complete-label score row")
        derived = {"status": endpoint["status"], "label": endpoint["label"],
                   "canonical_label_path": endpoint["canonical_label_path"],
                   "predicted_label": LABELS[int(np.argmax(vjp_row))],
                   "actual_score_top": LABELS[int(np.argmax(actual_row))]}
        for key in ASSERTIONS & row.keys():
            if type(row[key]) is not type(derived[key]) or row[key] != derived[key]:
                raise ValueError(f"supplied {key} contradicts raw record")
        verified.append({"case_id": row["case_id"], "endpoint": endpoint,
                         "vjp_top": derived["predicted_label"],
                         "actual_score_top": derived["actual_score_top"]})
        actual.append(actual_row)
        vjp.append(vjp_row)
    return verified, np.stack(actual), np.stack(vjp)


def development_gate(
    typed_records: list[dict[str, Any]], audio_records: list[dict[str, Any]],
    fit_records: list[dict[str, Any]], tokenizer: Any,
) -> dict[str, Any]:
    """Derive every gate class/flag from frozen IDs, raw IDs and score matrices."""
    fit_ids, truth, typed_ids, audio_ids = _expected_rows()
    typed, typed_actual, typed_vjp = _rows(typed_records, typed_ids, tokenizer)
    audio, audio_actual, audio_vjp = _rows(audio_records, audio_ids, tokenizer)
    fit = _fit_scores(fit_records, fit_ids)
    constant = readout_v2.constant_score_baseline(fit, 6)
    counts = readout_v2.development_gate(
        truth,
        [row["endpoint"]["label"] for row in typed],
        [row["endpoint"]["label"] for row in audio],
        readout_v2.predicted_labels(typed_vjp),
        readout_v2.predicted_labels(audio_vjp),
        readout_v2.predicted_labels(constant),
        [row["endpoint"]["canonical_label_path"] for row in typed],
        [row["endpoint"]["canonical_label_path"] for row in audio],
        typed_actual, typed_vjp, constant, audio_actual, audio_vjp, constant,
    )
    return {"schema": "qwen_action_method_development_gate_v3",
            "v1_spec_sha256": V1_SPEC_SHA256,
            "v2_freeze_sha256": V2_FREEZE_SHA256,
            "truth_order": truth, "typed": typed, "audio": audio,
            "counts": counts,
            "constant_top": readout_v2.predicted_labels(constant)}
