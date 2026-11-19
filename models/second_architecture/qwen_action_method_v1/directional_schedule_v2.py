"""Source-only fixed shared-state VJP intervention schedule."""
from __future__ import annotations

from hashlib import sha256

from .endpoint import load_spec
from .plan import audio_cases

ROTATION = {"LIST": "ASK", "ASK": "CANCEL", "CANCEL": "LIST"}
ARMS = ("zero_edit", "target", "random_orthogonal_1", "random_orthogonal_2")


def cases() -> list[dict[str, str]]:
    spec = load_spec()
    typed = [{"id": f"{row['id']}__typed", "script_id": row["id"],
              "split": row["split"], "voice": "typed", "label": row["label"]}
             for row in spec["scripts"] if row["split"] != "fit_text"]
    return typed + audio_cases(spec)


def schedule() -> list[dict[str, str | int]]:
    rows = []
    for case in cases():
        target = ROTATION[case["label"]]
        for arm in ARMS:
            index = int(arm.rsplit("_", 1)[-1]) if arm.startswith("random_") else 0
            seed = (int(sha256(f"qwen_action_method_v2|{case['id']}|{target}|{index}"
                               .encode()).hexdigest()[:16], 16) if index else 0)
            rows.append({"case_id": case["id"], "split": case["split"],
                         "modality": "typed" if case["voice"] == "typed" else "audio",
                         "voice": case["voice"], "reference_label": case["label"],
                         "target_label": target, "arm": arm, "seed": seed})
    return rows
