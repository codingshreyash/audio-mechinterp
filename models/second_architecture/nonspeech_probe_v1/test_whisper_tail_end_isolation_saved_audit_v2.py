"""No-model tests for V2's isolated, full-path decision rule."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from . import whisper_tail_end_isolation_plan_v2 as plan
from . import whisper_tail_end_isolation_saved_audit_v2 as audit


def checked(token: int) -> dict:
    return {"output": {"termination": "eos", "eos_seen": True,
                       "censored": False, "sequence_ids": [token, 50257]},
            "raw": np.array([token], dtype=np.float32)}


def row(layout: str, location: str, kind: str, result: dict,
        norm: float = 1.0) -> dict:
    return {"plan": {"type": "arm", "layout": layout, "location": location,
                     "kind": kind}, "checked": result,
            "patch": {"inside_delta_l2": norm, "target_delta_l2": norm}}


def fixture(stage: str):
    baselines = {(layout, kind): checked(10 if kind == "zero" else 20)
                 for layout in (("original",) if stage == "original"
                                else ("prepend", "append"))
                 for kind in ("zero", "noise")}
    states = {}
    evidence = []
    for layout in {key[0] for key in baselines}:
        zero = np.zeros((1500, 384), dtype=np.float32)
        noise = zero.copy()
        aligned = "translated" if layout == "prepend" else "fixed"
        if layout == "original":
            aligned = "fixed"
        rows = plan.rows(0 if stage == "original" else int(stage.removeprefix("offset_")),
                         aligned)
        zero[rows, 0] = 2.0
        states[(layout, "zero")] = {plan.SITE: zero}
        states[(layout, "noise")] = {plan.SITE: noise}
        locations = ("fixed",) if stage == "original" else ("fixed", "translated")
        for location in locations:
            correct = location == aligned
            for kind, output in (("target_transfer", baselines[(layout, "zero")]
                                 if correct else baselines[(layout, "noise")]),
                                 ("identity_sham", baselines[(layout, "noise")]),
                                 ("source_random", baselines[(layout, "noise")]),
                                 ("norm_matched_shifted_delta", baselines[(layout, "noise")])):
                evidence.append(row(layout, location, kind, output))
        evidence.append(row(layout, "fixed", "full_donor", baselines[(layout, "zero")]))
        if stage != "original":
            competing = "fixed" if layout == "prepend" else "translated"
            evidence.append(row(layout, competing, "relocated_correct_delta",
                                baselines[(layout, "noise")]))
    eligible = {layout: {"eligible": True} for layout in {key[0] for key in baselines}}
    return evidence, baselines, states, eligible


class SavedAuditTests(unittest.TestCase):
    def test_original_isolated_viability_uses_nonzero_native_delta(self) -> None:
        evidence, baselines, states, eligible = fixture("original")
        outcome = audit.summarize("original", evidence, baselines, states,
                                  eligible, {"status": "not_translated"})
        self.assertTrue(outcome["original_qualified"]["qualified"])
        self.assertGreater(outcome["original_qualified"]["native_delta_l2"], 0.001)
        states[("original", "zero")][plan.SITE][:] = 0
        outcome = audit.summarize("original", evidence, baselines, states,
                                  eligible, {"status": "not_translated"})
        self.assertFalse(outcome["original_qualified"]["qualified"])

    def test_original_cannot_qualify_without_clean_complete_contrast(self) -> None:
        evidence, baselines, states, eligible = fixture("original")
        # Even a forged positive arm bundle cannot replace the clean contrast.
        baselines[("original", "noise")] = baselines[("original", "zero")]
        for arm in evidence:
            arm["checked"] = baselines[("original", "zero")]
        eligible["original"]["eligible"] = False
        outcome = audit.summarize("original", evidence, baselines, states,
                                  eligible, {"status": "not_translated"})
        self.assertFalse(outcome["original_qualified"]["qualified"])
        self.assertFalse(outcome["original_qualified"]["clean_contrast_eligible"])

    def test_strict_crossover_requires_relocation_and_tracks_new_end_null(self) -> None:
        evidence, baselines, states, eligible = fixture("offset_8000")
        result = audit.summarize("offset_8000", evidence, baselines, states,
                                 eligible, {"pass_le_1e_5": True})
        self.assertTrue(result["strict_translated_crossover"])
        self.assertTrue(result["secondary_opposite_direct_preserved_recipient"])
        append = next(item for item in result["layouts"] if item["layout"] == "append")
        self.assertTrue(append["append_new_end_native_below_threshold"])
        relocation = next(item for item in evidence
                          if item["plan"]["layout"] == "append"
                          and item["plan"]["kind"] == "relocated_correct_delta")
        relocation["checked"] = baselines[("append", "zero")]
        result = audit.summarize("offset_8000", evidence, baselines, states,
                                 eligible, {"pass_le_1e_5": True})
        self.assertFalse(result["strict_translated_crossover"])

    def test_third_answer_is_nonspecific_and_secondary_fails(self) -> None:
        evidence, baselines, states, eligible = fixture("offset_16000")
        opposite = next(item for item in evidence
                        if item["plan"]["layout"] == "append"
                        and item["plan"]["location"] == "translated"
                        and item["plan"]["kind"] == "target_transfer")
        opposite["checked"] = checked(30)
        result = audit.summarize("offset_16000", evidence, baselines, states,
                                 eligible, {"pass_le_1e_5": True})
        self.assertTrue(result["strict_translated_crossover"])
        self.assertFalse(result["secondary_opposite_direct_preserved_recipient"])
        append = next(item for item in result["layouts"] if item["layout"] == "append")
        self.assertTrue(append["competing_direct_third_or_censored"])

    def test_joint_mode_reaudits_fixed_attempts(self) -> None:
        with mock.patch.object(audit, "audit", side_effect=ValueError("raw tampered")) as rerun:
            with self.assertRaisesRegex(ValueError, "raw tampered"):
                audit.audit_joint()
            rerun.assert_called_once_with("original", audit.gate.ADMISSIONS["original"],
                                          audit.gate.GOS["original"])

    def test_joint_rejects_out_of_order_attempts(self) -> None:
        identity = {"source": "same"}
        def report(stage: str, start: str, end: str, monotonic: float) -> dict:
            return {"schema": "whisper_tail_end_isolation_saved_audit_v2",
                    "status": "PASS", "stage": stage, "source_identity": identity,
                    "plan_sha256": "pinned", "parent_start_utc": start,
                    "parent_end_utc": end, "parent_start_monotonic": monotonic,
                    "whole_elapsed_seconds": 5.0, "boot_time_unix_us": 1,
                    "summary": {"original_qualified": {"qualified": True}} if stage == "original"
                    else {"strict_translated_crossover": True,
                          "secondary_opposite_direct_preserved_recipient": True,
                          "layouts": [{"layout": "append",
                                       "append_new_end_native_below_threshold": True}]}}
        original = report("original", "2026-11-19T10:00:00+00:00",
                          "2026-11-19T10:00:05+00:00", 100.0)
        offset8 = report("offset_8000", "2026-11-19T10:01:00+00:00",
                         "2026-11-19T10:01:05+00:00", 160.0)
        offset16 = report("offset_16000", "2026-11-19T10:02:00+00:00",
                          "2026-11-19T10:02:05+00:00", 220.0)
        result = audit._join_verified_reports(original, offset8, offset16)
        self.assertEqual(result["strict_count_out_of_2"], 2)
        offset8["parent_start_utc"] = "2026-11-19T10:00:04+00:00"
        with self.assertRaisesRegex(ValueError, "UTC order"):
            audit._join_verified_reports(original, offset8, offset16)
        offset8["parent_start_utc"] = "2026-11-19T10:01:00+00:00"
        offset8["parent_start_monotonic"] = 104.0
        with self.assertRaisesRegex(ValueError, "monotonic stage order"):
            audit._join_verified_reports(original, offset8, offset16)


if __name__ == "__main__":
    unittest.main()
