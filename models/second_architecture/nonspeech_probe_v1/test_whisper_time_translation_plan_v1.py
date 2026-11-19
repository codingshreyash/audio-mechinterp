"""No-weight checks of the predeclared time-translation schedule."""
from __future__ import annotations

import json
import unittest

from . import whisper_time_translation_plan_v1 as plan


class PlanTests(unittest.TestCase):
    def test_committed_plan_matches_builder(self) -> None:
        self.assertEqual(json.loads(plan.PLAN.read_text()), plan.build())

    def test_census_and_disjoint_displaced_controls(self) -> None:
        self.assertEqual(len(plan.arms_for("original")), 26)
        for stage in ("offset_8000", "offset_16000"):
            arms = plan.arms_for(stage)
            self.assertEqual(len(arms), 58)
            for layout in ("prepend", "append"):
                self.assertEqual(sum(a["layout"] == layout for a in arms), 29)
        for stage in ("original", "offset_8000", "offset_16000"):
            for arm in plan.arms_for(stage):
                if arm["kind"] == "full_donor":
                    self.assertEqual(arm["target_rows"], list(range(1500)))
                    continue
                if arm["kind"] == "relocated_correct_delta":
                    self.assertEqual(arm["source_rows"], [])
                    self.assertEqual(len(arm["target_rows"]),
                                     len(arm["delta_source_rows"]))
                    self.assertFalse(set(arm["target_rows"]) &
                                     set(arm["delta_source_rows"]))
                    continue
                self.assertEqual(len(arm["target_rows"]), len(arm["source_rows"]))
                self.assertFalse(set(arm["target_rows"]) & set(arm["source_rows"]))
                self.assertEqual(len(arm["target_rows"]), len(set(arm["target_rows"])))

    def test_nominal_translation_rows(self) -> None:
        for offset, delta in ((8000, 25), (16000, 50)):
            fixed = plan.rows("union", 0)
            moved = plan.rows("union", delta)
            self.assertEqual([row + delta for row in fixed], moved)
            stage = f"offset_{offset}"
            shifted = [a for a in plan.arms_for(stage)
                       if a["layout"] == "prepend" and a["site"] == "post_conv_pos"
                       and a["region"] == "union" and a["location"] == "translated"
                       and a["kind"] == "target_transfer"]
            self.assertEqual(len(shifted), 1)
            self.assertEqual(shifted[0]["target_rows"], moved)


if __name__ == "__main__":
    unittest.main()
