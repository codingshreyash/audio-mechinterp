"""CPU-only tests for the frozen v2 FIT selection and two-stage gate."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import voxtral_fit_pilot_capability_v2 as v2  # noqa: E402


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads(v2.INDEX.read_text())
        cls.selection = json.loads(v2.SELECTION.read_text())

    def test_exactly_32_unopened_fit_clips(self):
        jobs = v2.select_v2(self.index, self.selection)
        self.assertEqual(len(jobs), 32)
        self.assertEqual({r["pair_id"] for r, _, _ in jobs}, set(v2.PAIR_IDS))
        self.assertTrue(all(r["split"] == "fit_pool" and r["map_fit"] and not r["pilot_fit"] for r, _, _ in jobs))
        self.assertEqual({r["engine"] for r, _, _ in jobs}, {"piper", "kokoro"})
        self.assertEqual(len({(r["pair_id"], r["engine"], r["phrasing_id"]) for r, _, _ in jobs}), 16)

    def test_rejects_old_pilot_or_development_clip(self):
        for bad_pair in ("animal_06", "animal_05"):
            changed = copy.deepcopy(self.selection)
            changed["clips"][0]["pair_id"] = bad_pair
            with self.assertRaises(ValueError):
                v2.select_v2(self.index, changed)

    def test_rejects_selected_wav_hash_change(self):
        changed = copy.deepcopy(self.selection)
        changed["clips"][0]["wav_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            v2.select_v2(self.index, changed)

    def synthetic_records(self):
        records = []
        for row, word, _ in v2.select_v2(self.index, self.selection):
            records.append({
                "pair_id": row["pair_id"], "split": row["split"],
                "engine": row["engine"], "phrasing_id": row["phrasing_id"],
                "expected_word": word, "raw_decoded": word.capitalize(),
                "generated_token_ids": [1, 2],
                "classification": {"exact_expected_word_and_eos": True},
            })
        return records

    def test_stage2_requires_all_16_even_when_capability_passes(self):
        records = self.synthetic_records()
        good = v2.stage2_gate(records, True)
        self.assertTrue(good["intervention_gate_pass"])
        first = min(r["phrasing_id"] for r in records)
        target = next(r for r in records if r["pair_id"] == "animal_08" and r["engine"] == "piper" and r["phrasing_id"] == first)
        target["classification"]["exact_expected_word_and_eos"] = False
        failed = v2.stage2_gate(records, True)
        self.assertEqual(failed["first_template_natural_endpoints_passing"], 15)
        self.assertFalse(failed["intervention_gate_pass"])
        self.assertTrue(failed["capability_gate_pass"])

    def test_capability_failure_stops_even_if_stage2_exact(self):
        gate = v2.stage2_gate(self.synthetic_records(), False)
        self.assertEqual(gate["first_template_natural_endpoints_passing"], 16)
        self.assertFalse(gate["intervention_gate_pass"])


if __name__ == "__main__":
    unittest.main()
