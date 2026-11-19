"""No-weight checks for the frozen non-speech causal schedule."""
from __future__ import annotations

import collections
import unittest

from . import causal_plan_v1 as plan


class CausalPlanTests(unittest.TestCase):
    def test_manifest_population_and_exact_schedule(self) -> None:
        names = plan.frozen_cases()
        self.assertEqual(len(names), 20)
        self.assertEqual(len([name for name in names if name not in plan.DONORS]), 18)
        rows = plan.schedule()
        self.assertEqual(len(rows), 328)
        self.assertEqual(collections.Counter(row["stage"] for row in rows),
                         {"donor": 4, "A": 180, "B": 144})
        self.assertEqual([(row["case_id"], row["arm"]) for row in rows[:4]],
                         [(plan.DONORS[0], "no_hook"), (plan.DONORS[0], "read_only"),
                          (plan.DONORS[1], "no_hook"), (plan.DONORS[1], "read_only")])
        self.assertEqual(len({(row["stage"], row["case_id"], row["arm"])
                              for row in rows}), len(rows))
        for case in names:
            if case in plan.DONORS:
                continue
            self.assertEqual({row["arm"] for row in rows if row["stage"] == "A"
                              and row["case_id"] == case}, set(plan.STAGE_A_ARMS))
            self.assertEqual({row["arm"] for row in rows if row["stage"] == "B"
                              and row["case_id"] == case}, set(plan.STAGE_B_ARMS))

    def test_duration_donor_and_early_geometry(self) -> None:
        for row in plan.schedule():
            self.assertEqual(row["donor_id"], plan.donor_for(row["case_id"]))
            expected = 100 if row["case_id"].startswith("d02_") else 400
            self.assertEqual(row["active_rows"], expected)
            self.assertLessEqual(2 * expected, 1500)

    def test_random_seeds_unique_by_case_arm_and_phase(self) -> None:
        rows = plan.schedule()
        pairs = [(row["case_id"], row["stage"], row["arm"], phase, seed)
                 for row in rows if "seeds" in row
                 for phase, seed in row["seeds"].items()]
        self.assertEqual(len(pairs), 144)
        self.assertEqual(len({seed for *_, seed in pairs}), len(pairs))
        for case, stage, arm, phase, seed in pairs:
            self.assertEqual(seed, plan.random_seed(case, stage, arm, phase))
        for row in rows:
            if "seeds" in row:
                self.assertEqual(row["random_support_rows"], [0, 1500])
                self.assertEqual(row["random_norm_reference"],
                                 "full_donor_minus_recipient_l2_same_site_phase")

    def test_full_donor_logits_and_worst_case_binary_budget(self) -> None:
        rows = plan.schedule()
        full = [row for row in rows if row["save_full_raw_logits_each_step"]]
        self.assertEqual(len(full), 38)  # two silence donors and 36 full-donor arms
        self.assertEqual(sum(row["stage"] == "donor" for row in full), 2)
        self.assertTrue(all((row["stage"] == "donor" and row["arm"] == "no_hook")
                            or row["arm"] == "donor_both"
                            for row in full))
        self.assertLessEqual(plan.worst_case_core_evidence_bytes()
                             + plan.METADATA_ALLOWANCE_BYTES
                             + plan.EVIDENCE_RESERVE_BYTES,
                             plan.MAX_EVIDENCE_BYTES)


if __name__ == "__main__":
    unittest.main()
