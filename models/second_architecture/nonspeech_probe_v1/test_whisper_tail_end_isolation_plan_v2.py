"""No-model V2 schedule and source-freeze checks."""
from __future__ import annotations

import json
import unittest

from . import whisper_tail_end_isolation_plan_v2 as plan


class PlanTests(unittest.TestCase):
    def test_committed_schedule_matches_builder(self) -> None:
        self.assertEqual(json.loads(plan.PLAN.read_text()), plan.build())
        frozen = plan.build()
        self.assertEqual([frozen["stages"][stage]["max_calls"] for stage in plan.STAGES],
                         [7, 24, 24])
        self.assertEqual(frozen["nonzero_delta_l2"], 0.001)

    def test_both_offsets_and_competing_new_end_control(self) -> None:
        for stage, shift in (("offset_8000", 25), ("offset_16000", 50)):
            arms = plan.arms_for(stage)
            self.assertEqual(len(arms), 20)
            for layout in ("prepend", "append"):
                selected = [item for item in arms if item["layout"] == layout]
                self.assertEqual(len(selected), 10)
                for location in ("fixed", "translated"):
                    self.assertEqual({item["kind"] for item in selected
                                      if item["location"] == location
                                      and item["kind"] in plan.KINDS}, set(plan.KINDS))
                relocation = next(item for item in selected
                                  if item["kind"] == "relocated_correct_delta")
                aligned = [row + shift for row in plan.BASE_ROWS] if layout == "prepend" \
                    else plan.BASE_ROWS
                competing = plan.BASE_ROWS if layout == "prepend" \
                    else [row + shift for row in plan.BASE_ROWS]
                self.assertEqual(relocation["delta_source_rows"], aligned)
                self.assertEqual(relocation["target_rows"], competing)
                self.assertFalse(set(aligned) & set(competing))
                self.assertEqual({item["site"] for item in selected}, {plan.SITE})

    def test_original_viability_and_displaced_controls_are_prespecified(self) -> None:
        arms = plan.arms_for("original")
        self.assertEqual([item["kind"] for item in arms],
                         [*plan.KINDS, "full_donor"])
        self.assertTrue(all(item["target_rows"] == plan.BASE_ROWS
                            for item in arms[:-1]))
        source_rows = arms[0]["source_rows"]
        self.assertEqual(len(source_rows), 5)
        self.assertTrue(all(20 <= row <= 300 for row in source_rows))
        self.assertFalse(set(source_rows) & set(plan.BASE_ROWS))
        self.assertTrue(all(item["source_rows"] == source_rows
                            for item in arms[:-1]))


if __name__ == "__main__":
    unittest.main()
