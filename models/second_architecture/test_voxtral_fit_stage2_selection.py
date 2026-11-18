"""CPU-only checks of the v1 stop and fit-informed v2 selection rules."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import voxtral_fit_stage2_gate_v1 as gate_v1  # noqa: E402
import voxtral_fit_v2_select as select_v2  # noqa: E402


class GateAndSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads(gate_v1.INDEX.read_text())
        cls.pilot = json.loads(gate_v1.PILOT.read_text())

    def test_v1_stage2_stops_on_all_selected_natural_endpoints(self):
        report = gate_v1.evaluate(self.index, self.pilot)
        self.assertEqual(report["status"], "gate_failed")
        self.assertEqual(report["passing_natural_endpoints"], 14)
        self.assertEqual(len(report["failures"]), 2)
        self.assertEqual({(x["pair_id"], x["word"]) for x in report["failures"]}, {("animal_06", "deer")})

    def test_v1_stop_not_changed_by_aggregate_capability_pass(self):
        self.assertTrue(self.pilot["fit_capability_gate_pass"])
        self.assertFalse(gate_v1.evaluate(self.index, self.pilot)["v1_stage2_all_eight_records_natural_endpoints_pass"])

    def test_v2_uses_unopened_fit_pairs_only(self):
        ids, rows = select_v2.select(self.index)
        self.assertEqual(ids, ["animal_08", "color_05", "nature_09", "object_05"])
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r["split"] == "fit_pool" and r["map_fit"] for r in rows))
        self.assertTrue(set(ids).isdisjoint(select_v2.OLD_PILOT_IDS))
        self.assertTrue(all(w not in select_v2.HOMOPHONE_FAMILIES for r in rows for w in r["words"]))

    def test_v2_rejects_missing_category(self):
        changed = copy.deepcopy(self.index)
        for row in changed["rows"]:
            if row["pair_id"].startswith("color_") and row["split"] == "fit_pool":
                row["map_fit"] = False
        with self.assertRaises(ValueError):
            select_v2.select(changed)


if __name__ == "__main__":
    unittest.main()
