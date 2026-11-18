"""Create the documented pre-dev/test continuation-framing amendment.

Version 1 remains unchanged. This version was informed by two fit-only full
answer pilots and therefore is not claimed as preregistered before those runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V1_GATE_SHA256 = "938b78588413c407bc9f008c2b621fe2ea2601f80d545e080d1fab665c952879"
PILOT_SHA256 = "10865746c44d383110313df5eee396d670edde1634852efea1bfc9b3943080f4"
SYSTEM = "Continue the user's text naturally. Do not explain your continuation."


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_or_check(path: Path, obj: dict) -> None:
    content = (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen v2 file differs: {path}")
    else:
        path.write_bytes(content)


def main() -> None:
    gate_raw = (ROOT / "framing_gate.json").read_bytes()
    if sha(gate_raw) != V1_GATE_SHA256:
        raise ValueError("V1 gate changed")
    pilot_raw = (ROOT / "task_framing_smoke_2item_snapshot.json").read_bytes()
    if sha(pilot_raw) != PILOT_SHA256:
        raise ValueError("Fit-only pilot evidence changed")
    pilot = json.loads(pilot_raw)
    if pilot["system_prompt"] != SYSTEM or len(pilot["items"]) != 2:
        raise ValueError("Unexpected pilot setup")
    gate = json.loads(gate_raw)
    gate.update({
        "schema_version": 2,
        "status": "amended_after_two_fit_only_full_answer_pilots_not_preregistered_before_them",
        "amends_v1_sha256": V1_GATE_SHA256,
        "informing_fit_only_pilot_sha256": PILOT_SHA256,
        "system_prompt": SYSTEM,
        "generation": {"do_sample": False, "max_new_tokens": 80,
                       "stop": "model_eos_or_token_limit", "repetitions": 1},
        "judging": (
            "Retain full decoded output for all 12 fit-only smoke items in typed, Piper, "
            "and Kokoro routes. For association, three blind reviewers judge whether each "
            "output is a coherent continuation of the source vignette; they do not require "
            "the hidden bridge word to appear. Separately, reviewers make a four-choice "
            "judgment that the source vignette evokes its bridge label over distractors. "
            "For multihop, reviewers judge whether the complete output answers with the "
            "source target or a semantically equivalent answer. Majority 2/3 passes an item."
        ),
        "gate": (
            "Before any dev/test Omni bridge claim, require at least 5/6 fit-smoke items "
            "to pass in each category and each of typed, Piper, Kokoro routes, and at least "
            "5/6 association source-label four-choice judgments to pass. If a category fails, "
            "report its readouts only as exploratory and version a new task prompt before "
            "locked evaluation. The existing two-item pilot is insufficient to pass this gate."
        ),
    })
    write_or_check(ROOT / "framing_gate_v2.json", gate)

    layer = json.loads((ROOT / "layer_policy.json").read_text())
    layer.update({
        "schema_version": 2,
        "status": "amended_after_two_fit_only_full_answer_pilots_not_preregistered_before_them",
        "system_prompt": SYSTEM,
        "amends_v1_sha256": sha((ROOT / "layer_policy.json").read_bytes()),
        "informing_fit_only_pilot_sha256": PILOT_SHA256,
        "generation_route": "Qwen2.5-Omni-3B Thinker only; same system text and assistant-generation chat template in typed and audio modalities",
    })
    write_or_check(ROOT / "layer_policy_v2.json", layer)
    print("v2 gate/policy written or checked; both are informed by fit-only pilot")


if __name__ == "__main__":
    main()
