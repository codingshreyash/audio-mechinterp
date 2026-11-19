"""Frozen, no-model arm schedule for the Whisper time-translation probe."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from . import whisper_tail_causal_plan_v1 as original
from . import whisper_time_translation_source_v1 as source

HERE = Path(__file__).resolve().parent
PLAN = HERE / "WHISPER_TIME_TRANSLATION_PLAN_V1.json"
PROTOCOL = HERE / "WHISPER_TIME_TRANSLATION_PROSPECTIVE_V1.md"
MANIFEST_SHA256 = "eec732fb24a4b238ad0666d52d168f5ba41ec38c9d315a057a39e0d53be1435b"
SITES = ("post_conv_pos", "post_block0")
KINDS = ("target_transfer", "identity_sham", "source_random",
         "norm_matched_shifted_delta")
ORIGINAL_WINDOWS = ("union", "source_tail", "tail_end")
TRANSLATED_WINDOWS = (("union", "post_conv_pos"),
                      ("union", "post_block0"),
                      ("source_tail", "post_conv_pos"))
BASE = {"union": list(range(398, 403)) + list(range(498, 503)),
        "source_tail": list(range(398, 403)),
        "tail_end": list(range(498, 503))}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def displaced_rows(window: str) -> list[int]:
    count = len(BASE[window])
    seed = hashlib.sha256(f"whisper-time-translation-v1:{window}".encode()).digest()
    return sorted(random.Random(int.from_bytes(seed, "big")).sample(range(20, 301), count))


def rows(window: str, shift: int) -> list[int]:
    return [row + shift for row in BASE[window]]


def arm(*, stage: str, offset_samples: int, layout: str,
        site: str, window: str, location: str, kind: str) -> dict:
    shift = offset_samples // 320 if location == "translated" else 0
    target = list(range(1500)) if kind == "full_donor" else rows(window, shift)
    result = {"stage": stage, "offset_samples": offset_samples, "layout": layout,
              "site": site, "region": window, "location": location,
              "recipient": "noise", "donor": "zero", "kind": kind,
              "target_rows": target,
              "source_rows": [] if kind in ("full_donor", "relocated_correct_delta")
              else displaced_rows(window),
              "phase": "generate_only"}
    if kind == "relocated_correct_delta":
        correct = "translated" if layout == "prepend" else "fixed"
        require(location != correct, "relocation must target competing window")
        result["delta_source_rows"] = rows(window, offset_samples // 320
                                           if correct == "translated" else 0)
        require(len(result["delta_source_rows"]) == len(target)
                and not set(result["delta_source_rows"]) & set(target),
                "relocation rows overlap or differ in count")
    return result


def arms_for(stage: str) -> list[dict]:
    result = []
    if stage == "original":
        for site in SITES:
            for window in ORIGINAL_WINDOWS:
                for kind in KINDS:
                    result.append(arm(stage=stage, offset_samples=0, layout="original",
                                      site=site, window=window, location="fixed", kind=kind))
            result.append(arm(stage=stage, offset_samples=0, layout="original",
                              site=site, window="all", location="fixed", kind="full_donor"))
        require(len(result) == 26, "original arm census changed")
        return result
    require(stage in ("offset_8000", "offset_16000"), "unknown translation stage")
    offset = int(stage.removeprefix("offset_"))
    for layout in ("prepend", "append"):
        for window, site in TRANSLATED_WINDOWS:
            for location in ("fixed", "translated"):
                for kind in KINDS:
                    result.append(arm(stage=stage, offset_samples=offset, layout=layout,
                                      site=site, window=window, location=location, kind=kind))
            competing = "fixed" if layout == "prepend" else "translated"
            result.append(arm(stage=stage, offset_samples=offset, layout=layout,
                              site=site, window=window, location=competing,
                              kind="relocated_correct_delta"))
        for site in SITES:
            result.append(arm(stage=stage, offset_samples=offset, layout=layout,
                              site=site, window="all", location="fixed", kind="full_donor"))
    require(len(result) == 58, "translated arm census changed")
    return result


def build() -> dict:
    manifest_path = source.OUTPUT / "manifest.json"
    require(sha(manifest_path) == MANIFEST_SHA256, "frozen source manifest changed")
    manifest = source.verify()
    old_plan = original.build()
    old_unit = next(unit for unit in old_plan["units"]
                    if unit["model"] == "whisper_multilingual")
    require(old_unit["case_id"] == "d08_beep_train_low"
            and old_unit["zero"]["termination"] == "eos"
            and old_unit["noise"]["termination"] == "eos"
            and old_unit["zero"]["sequence_ids"] != old_unit["noise"]["sequence_ids"],
            "original reviewed contrast unavailable")
    original_inputs = {kind: {"wav": old_unit[kind]["wav"],
                              "wav_sha256": old_unit[kind]["wav_sha256"],
                              "frames": old_unit[kind]["frames"],
                              "sequence_ids": old_unit[kind]["sequence_ids"],
                              "prefix_ids": old_unit[kind]["prefix_ids"],
                              "termination": old_unit[kind]["termination"],
                              "mel": old_unit[kind]["mel"],
                              "mask": old_unit[kind]["mask"],
                              "first_raw_logits": old_unit[kind]["first_raw_logits"],
                              "first_processed_scores": old_unit[kind]["first_processed_scores"]}
                       for kind in ("zero", "noise")}
    result = {"schema": "whisper_time_translation_plan_v1",
              "status": "source_only_no_inference",
              "input_sha256": {"protocol": sha(PROTOCOL),
                               "translated_manifest": sha(manifest_path),
                               "old_plan": sha(original.PLAN_PATH)},
              "original": original_inputs,
              "translated": manifest["rows"],
              "sites": list(SITES), "windows": BASE,
              "stages": {stage: {"arms": arms_for(stage),
                                 "baseline_calls": 2 if stage == "original" else 4,
                                 "max_calls": 28 if stage == "original" else 62}
                         for stage in ("original", "offset_8000", "offset_16000")},
              "stop_rule": "baseline_no_complete_contrast_skips_that_pair_arms",
              "security_claim_authorized": False}
    return result


if __name__ == "__main__":
    require(not PLAN.exists(), "frozen plan path already consumed")
    with PLAN.open("x") as out:
        out.write(json.dumps(build(), sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "source_only_no_inference", "sha256": sha(PLAN)}))
