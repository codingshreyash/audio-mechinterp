"""V5 evidence bound with 20 deduplicated pinned processor input features.

The reviewed V3 schedule and V4 evidence rules remain unchanged. No model load.
"""
from __future__ import annotations

from .padding_channel_plan_v4 import *  # noqa: F403 - versioned plan API
from . import padding_channel_plan_v4 as _v4

FEATURE_CASES = 20
FEATURE_SHAPE = (1, 80, 3000)
FEATURE_BYTES = FEATURE_CASES * FEATURE_SHAPE[0] * FEATURE_SHAPE[1] * FEATURE_SHAPE[2] * 4
FEATURE_HEADER_ALLOWANCE_BYTES = FEATURE_CASES * 1024


def worst_case_core_evidence_bytes() -> int:
    return (_v4.worst_case_core_evidence_bytes() + FEATURE_BYTES
            + FEATURE_HEADER_ALLOWANCE_BYTES)


def resource_budget_fits() -> bool:
    return (worst_case_core_evidence_bytes() + _v4.METADATA_ALLOWANCE_BYTES
            + _v4.EVIDENCE_RESERVE_BYTES < _v4.EVIDENCE_LIMIT_BYTES)
