"""Saved-evidence fidelity tests for the audited causal viewer."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("build_causal_v1", HERE / "build_causal_v1.py")
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class CausalViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = builder.build_data()

    def test_inventory_and_controls(self) -> None:
        data = self.data
        self.assertEqual(len(data["cases"]), 18)
        self.assertEqual(len(data["donors"]), 2)
        self.assertEqual(len(data["informative_cases"]), 4)
        self.assertEqual(set(data["informative_cases"]), {
            "d02_beep_train_high", "d02_beep_train_low", "d08_beep_train_high", "d08_beep_train_low"})
        self.assertEqual(sum(len(c["stages"][s]) for c in data["cases"].values() for s in ("A", "B")) + 4, 328)
        self.assertEqual({r["call_index"] for donor in data["donors"].values() for r in donor.values()}, {1, 2, 3, 4})
        for donor in data["donors"].values():
            baseline, read_only = donor["no_hook"], donor["read_only"]
            self.assertEqual(baseline["sequence_ids"], read_only["sequence_ids"])
            self.assertEqual(baseline["transcript"], read_only["transcript"])
            self.assertTrue(baseline["eos_seen"] and read_only["eos_seen"])
            self.assertFalse(baseline["censored"] or read_only["censored"])
            self.assertTrue(all(read_only["parity"].values()))
        for case in data["cases"].values():
            for stage in ("A", "B"):
                base = case["stages"][stage]["no_hook"]["sequence_ids"]
                self.assertEqual(base, case["stages"][stage]["read_only"]["sequence_ids"])
                self.assertEqual(base, case["stages"][stage]["self_both"]["sequence_ids"])

    def test_censor_and_transfer(self) -> None:
        data = self.data
        self.assertEqual(len(data["censored_calls"]), 5)
        for name in data["censored_calls"]:
            self.assertTrue(any(r["call_name"] == name and r["censored"] and not r["eos_seen"]
                                for c in data["cases"].values() for s in c["stages"].values() for r in s.values()))
        self.assertEqual(data["counts"]["A"]["donor_detect"]["changed_full_sequence"], 0)
        self.assertEqual(data["counts"]["A"]["donor_transcribe"]["changed_full_sequence"], 4)
        self.assertEqual(data["counts"]["B"]["active_both"]["changed_full_sequence"], 4)
        self.assertEqual(data["counts"]["B"]["displaced_both"]["censored"], 1)
        self.assertEqual(data["counts"]["A"]["random_2_both"]["matched_donor"], 13)
        for case_id in data["informative_cases"]:
            c = data["cases"][case_id]
            donor = data["donors"][c["donor_id"]]["no_hook"]
            for stage, arm in (("A", "donor_transcribe"), ("A", "donor_both"),
                               ("B", "donor_both"), ("B", "active_both")):
                self.assertEqual(c["stages"][stage][arm]["sequence_ids"], donor["sequence_ids"])
            self.assertEqual(c["mel_padding"]["padded_frames_with_difference_gt_1e_6"],
                             c["mel_padding"]["padded_mel_frames"])

    def test_deterministic_output_and_hash_guard(self) -> None:
        rendered = builder.render(self.data)
        self.assertEqual(rendered, (HERE / "causal_v1.html").read_text())
        self.assertEqual(rendered, builder.render(builder.build_data()))
        self.assertIn(self.data["hashes"]["records_sha256"], rendered)
        self.assertIn(self.data["hashes"]["independent_memo_sha256"], rendered)
        self.assertNotIn("https://", rendered)
        original = builder.EXPECTED["records_sha256"]
        try:
            builder.EXPECTED["records_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
                builder.build_data()
        finally:
            builder.EXPECTED["records_sha256"] = original


if __name__ == "__main__":
    unittest.main()
