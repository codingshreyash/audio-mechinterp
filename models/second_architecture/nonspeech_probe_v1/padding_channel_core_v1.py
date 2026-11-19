"""Pure array algebra shared by the padding worker and saved-result auditor.

Importing this module never loads a candidate checkpoint. It deliberately
operates on immutable copies so that a hook cannot change its donor evidence.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import padding_channel_plan_v1 as plan


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def expected_shape(site: str) -> tuple[int, int, int]:
    if site == "log_mel":
        return (1, 80, 3000)
    if site == "pre_block0":
        return (1, 1500, 384)
    raise ValueError(f"unknown site: {site}")


def assert_array(value: np.ndarray, site: str) -> None:
    require(value.dtype == np.float32 and tuple(value.shape) == expected_shape(site)
            and bool(np.isfinite(value).all()), f"{site} tensor invalid")


def time_slice(site: str, start: int, end: int) -> tuple[slice, slice, slice]:
    return (slice(None), slice(None), slice(start, end)) if site == "log_mel" else (
        slice(None), slice(start, end), slice(None))


def selected(row: dict, site: str, phase: str) -> bool:
    if row["hook_policy"] != "edit" or row["site"] != site:
        return False
    if row["arm"] == "padding_detect_only":
        return phase == "detect"
    if row["arm"] == "padding_transcribe_only":
        return phase == "transcribe"
    return True


def edit(native: np.ndarray, recipient: np.ndarray, donor: np.ndarray,
         row: dict, site: str, phase: str) -> tuple[np.ndarray, dict]:
    """Reconstruct one selected-site/phase edit from saved source tensors."""
    assert_array(native, site)
    assert_array(recipient, site)
    assert_array(donor, site)
    require(phase in plan.PHASES, "unknown phase")
    require(np.array_equal(native, recipient), "selected-site native differs from capture")
    active = int(plan.geometry(row["case_id"], site)["active"])
    total = expected_shape(site)[-1 if site == "log_mel" else 1]
    arm = row["arm"]
    if not selected(row, site, phase):
        return native.copy(), {"support": [0, 0], "seed": None,
                               "target_l2": 0.0, "realized_l2": 0.0, "cosine": None}
    if arm == "self":
        start, end, source = 0, total, recipient
    elif arm in ("RD", "padding_detect_only", "padding_transcribe_only"):
        start, end, source = active, total, donor
    elif arm == "DR":
        start, end, source = 0, active, donor
    elif arm == "DD":
        start, end, source = 0, total, donor
    elif arm.startswith("random_active_"):
        start, end, source = 0, active, None
    elif arm.startswith("random_padding_"):
        start, end, source = active, total, None
    else:
        raise ValueError(f"unknown edit arm: {arm}")
    support = time_slice(site, start, end)
    target_displacement = (donor[support] - recipient[support]).astype(np.float64)
    target_l2 = float(np.linalg.norm(target_displacement.ravel()))
    result = recipient.copy()
    seed = None
    if source is not None:
        result[support] = source[support]
    else:
        seed = int(row["seeds"][phase])
        noise = np.random.default_rng(seed).standard_normal(
            recipient[support].shape).astype(np.float32).astype(np.float64)
        norm = float(np.linalg.norm(noise.ravel()))
        require(norm > 0, "zero random draw")
        displacement = (noise / norm * target_l2).astype(np.float32)
        result[support] = (recipient[support] + displacement).astype(np.float32)
    realized_delta = (result[support] - recipient[support]).astype(np.float64)
    realized_l2 = float(np.linalg.norm(realized_delta.ravel()))
    cosine = float(np.dot(realized_delta.ravel(), target_displacement.ravel())
                   / (realized_l2 * target_l2)) if realized_l2 and target_l2 else None
    require(bool(np.isfinite(result).all()), "edited tensor nonfinite")
    return result, {"support": [start, end], "seed": seed,
                    "target_l2": target_l2, "realized_l2": realized_l2,
                    "cosine": cosine}


def maxabs_finite(left: np.ndarray, right: np.ndarray) -> float:
    require(left.shape == right.shape and left.dtype == right.dtype == np.float32
            and bool(np.isfinite(left).all()) and bool(np.isfinite(right).all()),
            "finite parity arrays invalid")
    return float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))


def processed_score_maxabs(left: np.ndarray, right: np.ndarray) -> float:
    require(left.shape == right.shape and left.dtype == right.dtype == np.float32,
            "processed score shape/dtype mismatch")
    require(not np.isnan(left).any() and not np.isnan(right).any()
            and not np.isposinf(left).any() and not np.isposinf(right).any(),
            "processed score NaN/+inf")
    left_mask, right_mask = np.isneginf(left), np.isneginf(right)
    require(np.array_equal(left_mask, right_mask), "processed suppression mask mismatch")
    finite = ~left_mask
    require(bool(finite.any()), "processed scores have no finite entry")
    return float(np.max(np.abs(left[finite].astype(np.float64)
                               - right[finite].astype(np.float64))))
