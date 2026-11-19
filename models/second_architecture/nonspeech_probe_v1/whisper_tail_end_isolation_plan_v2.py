"""Frozen, source-only schedule for isolated Whisper tail/end V2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import whisper_time_translation_plan_v1 as prior
from . import whisper_time_translation_source_v1 as source

HERE = Path(__file__).resolve().parent
PLAN = HERE / "WHISPER_TAIL_END_ISOLATION_PLAN_V2.json"
PROTOCOL = HERE / "WHISPER_TAIL_END_ISOLATION_PROSPECTIVE_V2.md"
PRIOR_PLAN_SHA256 = "b1aa22f0b0c9a357a641dcc7d3ee5e1a789d2337d9b0bbb81ae131e6e3162f61"
SOURCE_MANIFEST_SHA256 = "eec732fb24a4b238ad0666d52d168f5ba41ec38c9d315a057a39e0d53be1435b"
STAGES = ("original", "offset_8000", "offset_16000")
SITE = "post_conv_pos"
REGION = "tail_end"
BASE_ROWS = list(range(498, 503))
NONZERO_DELTA_L2 = 1e-3
KINDS = ("target_transfer", "identity_sham", "source_random",
         "norm_matched_shifted_delta")


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(offset_samples: int, location: str) -> list[int]:
    need(location in ("fixed", "translated")
         and offset_samples in (0, 8000, 16000), "invalid tail/end coordinate")
    shift = offset_samples // 320 if location == "translated" else 0
    return [row + shift for row in BASE_ROWS]


def arm(stage: str, layout: str, location: str, kind: str) -> dict:
    need(stage in STAGES and kind in (*KINDS, "full_donor", "relocated_correct_delta"),
         "invalid V2 arm")
    offset = 0 if stage == "original" else int(stage.removeprefix("offset_"))
    target = list(range(1500)) if kind == "full_donor" else rows(offset, location)
    result = {"stage": stage, "offset_samples": offset, "layout": layout,
              "site": SITE, "region": REGION if kind != "full_donor" else "all",
              "location": location, "recipient": "noise", "donor": "zero",
              "kind": kind, "target_rows": target,
              "source_rows": [] if kind in ("full_donor", "relocated_correct_delta")
              else prior.displaced_rows(REGION), "phase": "generate_only"}
    if kind == "relocated_correct_delta":
        correct = "translated" if layout == "prepend" else "fixed"
        need(stage != "original" and location != correct,
             "relocation must target competing translated window")
        result["delta_source_rows"] = rows(offset, correct)
        need(not set(result["delta_source_rows"]) & set(target),
             "relocation source and destination overlap")
    return result


def arms_for(stage: str) -> list[dict]:
    need(stage in STAGES, "unknown V2 stage")
    if stage == "original":
        result = [arm(stage, "original", "fixed", kind) for kind in KINDS]
        result.append(arm(stage, "original", "fixed", "full_donor"))
        need(len(result) == 5, "original V2 arm census changed")
        return result
    result = []
    for layout in ("prepend", "append"):
        for location in ("fixed", "translated"):
            result.extend(arm(stage, layout, location, kind) for kind in KINDS)
        result.append(arm(stage, layout, "fixed", "full_donor"))
        competing = "fixed" if layout == "prepend" else "translated"
        result.append(arm(stage, layout, competing, "relocated_correct_delta"))
    need(len(result) == 20, "translated V2 arm census changed")
    return result


def build() -> dict:
    need(sha(prior.PLAN) == PRIOR_PLAN_SHA256, "V1 plan bytes changed")
    need(sha(source.OUTPUT / "manifest.json") == SOURCE_MANIFEST_SHA256,
         "V1 rendered source manifest changed")
    source.verify()
    inherited = json.loads(prior.PLAN.read_text())
    need(inherited == prior.build(), "V1 plan source changed")
    need(set(inherited["original"]) == {"zero", "noise"}
         and len(inherited["translated"]) == 8,
         "V1 frozen pair/source inventory changed")
    return {"schema": "whisper_tail_end_isolation_plan_v2",
            "status": "source_only_no_inference",
            "input_sha256": {"protocol": sha(PROTOCOL),
                             "v1_plan": PRIOR_PLAN_SHA256,
                             "translated_manifest": SOURCE_MANIFEST_SHA256},
            "original": inherited["original"],
            "translated": inherited["translated"],
            "site": SITE, "region": REGION, "base_rows": BASE_ROWS,
            "nonzero_delta_l2": NONZERO_DELTA_L2,
            "stages": {stage: {"arms": arms_for(stage),
                               "baseline_calls": 2 if stage == "original" else 4,
                               "max_calls": 7 if stage == "original" else 24}
                       for stage in STAGES},
            "stop_rule": "translated_no_complete_contrast_skips_layout_arms",
            "original_replay_failure": "attempt_invalid_blocks_later_stages",
            "original_control_failure": "later_stages_descriptive_only",
            "censored_baseline": "record_ineligible_and_skip_layout_arms",
            "censored_arm": "record_failed_complete_answer_control",
            "security_claim_authorized": False}


if __name__ == "__main__":
    need(not PLAN.exists(), "frozen V2 plan path already consumed")
    PLAN.write_text(json.dumps(build(), sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "source_only_no_inference", "sha256": sha(PLAN)}))
