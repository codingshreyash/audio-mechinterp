"""No candidate-model load or inference; validate the prospective census."""
from collections import Counter
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v1 as plan


class PaddingChannelPlanTests(unittest.TestCase):
    def test_frozen_population_and_full_factorial(self) -> None:
        rows = plan.schedule()
        self.assertEqual(len(rows), 434)
        self.assertEqual(len({(row["case_id"], row["site"], row["arm"])
                              for row in rows}), 434)
        self.assertEqual(Counter(row["arm"] for row in rows), {
            "no_hook": 2, **{arm: 36 for arm in
                              plan.CELLS + plan.CONTROLS + plan.PHASE_SPLITS}})
        cases = set(plan.population()) - set(plan.SILENCE)
        self.assertEqual(len(cases), 18)
        for site in plan.SITES:
            for case_id in cases:
                actual = {row["arm"] for row in rows
                          if row["site"] == site and row["case_id"] == case_id}
                self.assertEqual(actual,
                                 set(plan.CELLS + plan.CONTROLS + plan.PHASE_SPLITS))
                self.assertIn(plan.donor_for(case_id), plan.SILENCE)


    def test_fixed_geometry_and_boundary_partition(self) -> None:
        expected = {
            ("d02_silence", "log_mel"): (200, 2800, 3000),
            ("d08_silence", "log_mel"): (800, 2200, 3000),
            ("d02_silence", "pre_block0"): (100, 1400, 1500),
            ("d08_silence", "pre_block0"): (400, 1100, 1500),
        }
        for key, triple in expected.items():
            geom = plan.geometry(*key)
            self.assertEqual((geom["active"], geom["padding"], geom["total"]), triple)
            self.assertEqual(geom["active"] + geom["padding"], geom["total"])


    def test_random_seeds_are_stable_distinct_and_restricted(self) -> None:
        rows = [row for row in plan.schedule() if row["arm"].startswith("random_")]
        seeds = [row["seeds"][phase] for row in rows for phase in plan.PHASES]
        self.assertEqual(len(seeds), 18 * 2 * 4 * 2)
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual(rows[0]["seeds"]["detect"], plan.seed(
            rows[0]["case_id"], rows[0]["site"], rows[0]["arm"], "detect"))
        with self.assertRaises(ValueError):
            plan.seed("d02_beep_train_low", "log_mel", "self", "detect")


    def test_manifest_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "manifest.json"
            changed.write_text('{"rows":[]}')
            with patch.object(plan, "MANIFEST", changed):
                with self.assertRaisesRegex(ValueError, "manifest changed"):
                    plan.schedule()


if __name__ == "__main__":
    unittest.main()
