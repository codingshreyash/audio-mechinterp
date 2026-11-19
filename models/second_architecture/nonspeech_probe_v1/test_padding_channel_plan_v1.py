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
        self.assertEqual(len(rows), 436)
        self.assertEqual(len({(row["case_id"], row["site"], row["arm"])
                              for row in rows}), 436)
        self.assertEqual(Counter(row["arm"] for row in rows), {
            "no_hook": 2, "read_only_donor_capture": 2,
            **{arm: 36 for arm in
                              plan.CELLS + plan.CONTROLS + plan.PHASE_SPLITS}})
        self.assertEqual(Counter(row["hook_policy"] for row in rows),
                         {"none": 38, "capture_both_sites_both_phases": 38,
                          "edit": 360})
        for donor in plan.SILENCE:
            donor_rows = [row for row in rows if row["case_id"] == donor]
            self.assertEqual([row["arm"] for row in donor_rows],
                             ["no_hook", "read_only_donor_capture"])
            self.assertEqual([row["hook_policy"] for row in donor_rows],
                             ["none", "capture_both_sites_both_phases"])
        cases = set(plan.population()) - set(plan.SILENCE)
        self.assertEqual(len(cases), 18)
        for site in plan.SITES:
            for case_id in cases:
                actual = {row["arm"] for row in rows
                          if row["site"] == site and row["case_id"] == case_id}
                self.assertEqual(actual,
                                 set(plan.CELLS + plan.CONTROLS + plan.PHASE_SPLITS))
                ordered = [row["arm"] for row in rows
                           if row["site"] == site and row["case_id"] == case_id]
                self.assertEqual(ordered, list(plan.ARM_ORDER))
                self.assertIn(plan.donor_for(case_id), plan.SILENCE)
                rr = next(row for row in rows if row["site"] == site
                          and row["case_id"] == case_id and row["arm"] == "RR")
                read_only = next(row for row in rows if row["site"] == site
                                 and row["case_id"] == case_id
                                 and row["arm"] == "read_only")
                self.assertEqual(rr["hook_policy"], "none")
                self.assertEqual(read_only["hook_policy"],
                                 "capture_both_sites_both_phases")

    def test_capture_must_precede_every_edit(self) -> None:
        rows = list(plan.schedule())
        first_rr = next(i for i, row in enumerate(rows) if row["arm"] == "RR")
        self.assertEqual([row["arm"] for row in rows[first_rr:first_rr + 5]],
                         ["RR", "read_only", "RD", "DR", "DD"])
        rows[first_rr + 1], rows[first_rr + 2] = rows[first_rr + 2], rows[first_rr + 1]
        with self.assertRaisesRegex(ValueError, "edit precedes"):
            plan.validate_capture_dependencies(rows)


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

    def test_source_pins_and_resource_ceilings(self) -> None:
        plan.verify_installed_source_bytes()
        self.assertEqual(plan.PINNED_MODEL,
                         "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
        self.assertEqual(plan.WALL_SECONDS, 1800)
        self.assertEqual(plan.RSS_LIMIT_BYTES, 4 * 1024**3)
        self.assertEqual(plan.EVIDENCE_LIMIT_BYTES, 2 * 1024**3)
        self.assertEqual(plan.worst_case_core_evidence_bytes(), 1_375_203_488)
        self.assertTrue(plan.resource_budget_fits())
        with patch.object(plan, "EVIDENCE_LIMIT_BYTES",
                          plan.worst_case_core_evidence_bytes()):
            self.assertFalse(plan.resource_budget_fits())
        with patch.dict(plan.SOURCE_SHA256,
                        {"modeling_whisper.py": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                plan.verify_installed_source_bytes()


if __name__ == "__main__":
    unittest.main()
