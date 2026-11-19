"""No-weight tests for the FIT noise pilot source and admission gate."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.second_architecture import acoustic_fit_noise_pilot_v1 as pilot


class NoisePilotSourceTests(unittest.TestCase):
    def test_frozen_matrix_and_locality(self):
        clips = pilot.stimuli()
        self.assertEqual(set(clips), {"piper_silver", "piper_teal", "kokoro_silver", "kokoro_teal"})
        for key, arms in clips.items():
            self.assertEqual(set(arms), set(pilot.ARMS))
            clean = arms["clean"]
            self.assertTrue(np.array_equal(clean, arms["clean_repeat"]))
            s, e = pilot.EXPECTED[key.split("_")[0]]["slot"]
            self.assertTrue(np.array_equal(clean[:s], arms["word_noise"][:s]))
            self.assertTrue(np.array_equal(clean[e:], arms["word_noise"][e:]))
            self.assertTrue(np.array_equal(clean[s:e], arms["outside_noise"][s:e]))
            self.assertTrue(np.array_equal(arms["full_noise"] - clean,
                                           arms["word_noise"] - clean + arms["outside_noise"] - clean))

    def test_hash_mismatch_fails_closed(self):
        expected = pilot.EXPECTED["piper"]["noise"]
        pilot.EXPECTED["piper"]["noise"] = "0" * 64
        try:
            with self.assertRaisesRegex(ValueError, "frozen noise differs"):
                pilot.stimuli()
        finally:
            pilot.EXPECTED["piper"]["noise"] = expected

    def test_run_requires_receipt_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "run"
            with self.assertRaises((FileNotFoundError, ValueError)):
                pilot.require_review(root / "missing.json", root / "missing_receipt.json", {})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
