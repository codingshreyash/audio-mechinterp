"""Pure source-only Qwen action-label case and full-route control schedule."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

from .endpoint import load_spec

PAIRS = {
    "development": (("D_L1", "D_C1"), ("D_A1", "D_C2"), ("D_L2", "D_A2")),
    "locked": (("H_L1", "H_C1"), ("H_A1", "H_C2"), ("H_L2", "H_A2")),
}
ARMS = ("self", "full_donor", "same_label_donor", "random_1", "random_2")


def audio_cases(spec: dict[str, Any] | None = None) -> list[dict[str, str]]:
    spec = load_spec() if spec is None else spec
    cases = []
    for row in spec["scripts"]:
        if row["split"] == "fit_text":
            continue
        for voice in spec["speech_voices"][row["split"]]:
            cases.append({"id": f"{row['id']}__{voice}", "script_id": row["id"],
                          "split": row["split"], "voice": voice,
                          "label": row["label"]})
    return cases


def causal_schedule(spec: dict[str, Any] | None = None) -> list[dict[str, str | int]]:
    """No output-dependent selection; all source pairs and directions remain planned."""
    spec = load_spec() if spec is None else spec
    cases = audio_cases(spec)
    by_id = {case["id"]: case for case in cases}
    out: list[dict[str, str | int]] = []
    for split in ("development", "locked"):
        for voice in spec["speech_voices"][split]:
            for first, second in PAIRS[split]:
                for rec_script, donor_script in ((first, second), (second, first)):
                    recipient = by_id[f"{rec_script}__{voice}"]
                    donor = by_id[f"{donor_script}__{voice}"]
                    same_label = next(case for case in cases
                                      if case["split"] == split and case["voice"] == voice
                                      and case["label"] == recipient["label"]
                                      and case["id"] != recipient["id"])
                    for arm in ARMS:
                        seed = (int(sha256(f"qwen_action_method_v1|{recipient['id']}|"
                                           f"{donor['id']}|{arm}".encode()).hexdigest()[:16], 16)
                                if arm.startswith("random_") else 0)
                        out.append({"split": split, "voice": voice,
                                    "recipient_id": recipient["id"], "donor_id": donor["id"],
                                    "same_label_donor_id": same_label["id"],
                                    "arm": arm, "seed": seed})
    return out
