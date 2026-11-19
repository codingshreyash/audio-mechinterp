"""No-weight stimulus and complete-answer checks for Qwen action method V1.

This module never imports a model, processor, torch, or an audio decoder.
It defines source-only invariants; actual model output is not available yet.
"""
from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).with_name("frozen_spec_v1.json")
LABELS = ("LIST", "ASK", "CANCEL")
SPLITS = {"fit_text": 18, "development": 6, "locked": 6}
EOS_ID = 151645
MAX_NEW_TOKENS = 12
ROOT = Path(__file__).resolve().parents[3]


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    validate_spec(spec)
    return spec


def validate_local_source(spec: dict[str, Any], root: Path = ROOT) -> dict[str, str]:
    """Hash pinned installed source/config without importing Transformers."""
    pins = json.loads((root / "models/qwen_omni/pins.json").read_text())
    for key in ("model_id", "model_revision", "transformers_revision"):
        if spec[key] != pins[key]:
            raise ValueError(f"{key} differs from existing Qwen pin")
    base = root / "models/qwen_omni/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni"
    files = {
        "modeling": (base / "modeling_qwen2_5_omni.py", spec["installed_modeling_sha256"]),
        "processing": (base / "processing_qwen2_5_omni.py", spec["installed_processing_sha256"]),
        "config": (root / "models/qwen_omni/config.snapshot.json",
                   pins["config_snapshot_sha256"]),
    }
    digests = {name: sha256(path.read_bytes()).hexdigest()
               for name, (path, _) in files.items()}
    if any(digests[name] != expected for name, (_, expected) in files.items()):
        raise ValueError("pinned Qwen source/config bytes changed")
    return digests


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema") != "qwen_action_method_source_v1" or spec.get("labels") != list(LABELS):
        raise ValueError("unexpected protocol schema or labels")
    if spec.get("eos_token_id_from_pinned_config") != EOS_ID:
        raise ValueError("unexpected pinned EOS token")
    if spec.get("generation") != {"max_new_tokens": MAX_NEW_TOKENS,
                                   "do_sample": False, "use_cache": True}:
        raise ValueError("generation protocol changed")
    if not isinstance(spec.get("system_prompt"), str) or not spec["system_prompt"].strip():
        raise ValueError("missing system prompt")
    rows = spec.get("scripts")
    if not isinstance(rows, list) or len(rows) != sum(SPLITS.values()):
        raise ValueError("wrong script census")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate script ID")
    normalized = [" ".join(row["text"].casefold().split()) for row in rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError("exact wording reused across splits")
    for split, expected in SPLITS.items():
        section = [row for row in rows if row["split"] == split]
        if len(section) != expected or Counter(row["label"] for row in section) != {
            label: expected // len(LABELS) for label in LABELS
        }:
            raise ValueError(f"unbalanced {split} split")
        for row in section:
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("empty script")
            if not row["id"].startswith({"fit_text": "F_", "development": "D_",
                                         "locked": "H_"}[split]):
                raise ValueError("split ID prefix mismatch")
    if spec.get("speech_voices") != {
        "development": ["piper_en_US_amy_medium"],
        "locked": ["macos_samantha_en_US", "kokoro_af_bella"]
    }:
        raise ValueError("voice split changed")


def make_typed_conversation(system_prompt: str, spoken_words: str) -> list[dict[str, Any]]:
    if not system_prompt.strip() or not spoken_words.strip():
        raise ValueError("empty prompt or user words")
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": spoken_words}]},
    ]


def score_complete_answer(
    continuation_ids: list[int], tokenizer: Any, *, max_new_tokens: int = MAX_NEW_TOKENS
) -> dict[str, Any]:
    """Score exactly one terminal label; other text and missing EOS are failures.

    `continuation_ids` must be sliced after the prompt. The caller separately
    records the unsliced sequence, prompt IDs, and both token decodes.
    """
    ids = list(continuation_ids)
    if not ids or len(ids) > max_new_tokens or any(not isinstance(x, int) for x in ids):
        raise ValueError("invalid continuation IDs or output cap")
    if int(tokenizer.eos_token_id) != EOS_ID:
        raise ValueError("tokenizer EOS differs from pinned config")
    payload = ids[:-1] if ids[-1] == EOS_ID else ids
    decoded = tokenizer.decode(payload, skip_special_tokens=False)
    status = "censored" if ids[-1] != EOS_ID else "complete"
    if status == "complete" and EOS_ID in payload:
        status = "malformed"
    label = decoded.strip() if status == "complete" else None
    if status == "complete" and label not in LABELS:
        status, label = "malformed", None
    return {"status": status, "label": label, "raw_payload": decoded,
            "continuation_ids": ids, "eos": ids[-1] == EOS_ID}


def exact_complete_parity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Only exact EOS-complete token sequence equality is a control pass."""
    return (left.get("status") == right.get("status") == "complete"
            and left.get("continuation_ids") == right.get("continuation_ids")
            and left.get("label") == right.get("label"))
