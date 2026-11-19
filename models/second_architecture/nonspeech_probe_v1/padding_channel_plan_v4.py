"""V4 evidence bound; reuses the independently reviewed V3 call schedule unchanged.

Source-only module: no candidate model import or inference.
"""
from __future__ import annotations

from . import padding_channel_plan_v1 as base
from .padding_channel_plan_v1 import *  # noqa: F403 - reviewed schedule API

FIRST_DECODER_POSITIONS = 4


def worst_case_core_evidence_bytes() -> int:
    # V3 counted one decoder row; the pinned scorer returns [1,4,384].
    return (base.worst_case_core_evidence_bytes()
            + len(base.schedule()) * (FIRST_DECODER_POSITIONS - 1)
            * base.DECODER_WIDTH * 4)


def resource_budget_fits() -> bool:
    return (worst_case_core_evidence_bytes() + base.METADATA_ALLOWANCE_BYTES
            + base.EVIDENCE_RESERVE_BYTES < base.EVIDENCE_LIMIT_BYTES)
