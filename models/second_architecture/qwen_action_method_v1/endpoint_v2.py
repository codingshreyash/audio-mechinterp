"""Strict no-weight complete-label endpoint; V1 remains frozen and on HOLD."""
from __future__ import annotations

from typing import Any

from .endpoint import EOS_ID, LABELS, MAX_NEW_TOKENS
from .readout_v2 import LABEL_TOKEN_IDS

CANONICAL = dict(zip(LABELS, LABEL_TOKEN_IDS, strict=True))


def score_complete_answer(
    continuation_ids: list[int], tokenizer: Any, *, max_new_tokens: int = MAX_NEW_TOKENS
) -> dict[str, Any]:
    """Classify every token path as complete, malformed, or censored.

    The caller must slice `continuation_ids` after the prompt and preserve the
    unsliced generated IDs in its separate raw record. No candidate run exists.
    """
    ids = list(continuation_ids)
    if len(ids) > max_new_tokens or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("invalid continuation IDs or output cap")
    if int(tokenizer.eos_token_id) != EOS_ID:
        raise ValueError("tokenizer EOS differs from pinned config")
    special = {int(x) for x in tokenizer.all_special_ids}
    if EOS_ID not in special:
        raise ValueError("tokenizer special-ID inventory omits EOS")
    if not ids:
        return {"status": "censored", "reason": "empty_continuation",
                "label": None, "raw_payload": "", "continuation_ids": [],
                "terminal_eos": False, "canonical_label_path": False}

    terminal_eos = ids[-1] == EOS_ID
    payload = ids[:-1] if terminal_eos else ids
    decoded = tokenizer.decode(payload, skip_special_tokens=False)
    if EOS_ID in payload:
        status, reason, label = "malformed", "early_or_repeated_eos", None
    elif any(token in special for token in payload):
        status, reason, label = "malformed", "unexpected_special_id", None
    elif not terminal_eos:
        status, reason, label = "censored", "missing_terminal_eos", None
    elif decoded.strip() not in LABELS:
        status, reason, label = "malformed", "not_one_label", None
    else:
        status, reason, label = "complete", None, decoded.strip()
    return {"status": status, "reason": reason, "label": label,
            "raw_payload": decoded, "continuation_ids": ids,
            "terminal_eos": terminal_eos,
            "canonical_label_path": bool(status == "complete"
                                         and ids == [CANONICAL[label], EOS_ID])}


def exact_complete_parity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (left.get("status") == right.get("status") == "complete"
            and left.get("label") == right.get("label")
            and left.get("continuation_ids") == right.get("continuation_ids"))
