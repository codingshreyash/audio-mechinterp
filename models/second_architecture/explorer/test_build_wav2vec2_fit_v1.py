"""Source-integrity and display-reduction checks for the offline Wav2Vec2 viewer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import build_wav2vec2_fit_v1 as viewer


class Wav2Vec2ViewerTests(unittest.TestCase):
    def test_max_pool_retains_spike_and_exact_zero(self) -> None:
        trace = np.array([0, 0, 0, 9, 0, 0, 0, 0], dtype=np.float32)
        self.assertEqual(viewer.max_pool(trace, bins=2), [9.0, 0.0])
        self.assertEqual(viewer.max_pool(np.zeros(9, dtype=np.float32), bins=3),
                         [0.0, 0.0, 0.0])

    def test_max_pool_rejects_nonfinite_or_negative_trace(self) -> None:
        with self.assertRaises(ValueError):
            viewer.max_pool(np.array([0, np.nan], dtype=np.float32))
        with self.assertRaises(ValueError):
            viewer.max_pool(np.array([0, -1], dtype=np.float32))

    def test_unreviewed_result_cannot_build(self) -> None:
        with patch.dict(viewer.EXPECTED_SHA, {"causal_review": ""}):
            with self.assertRaisesRegex(ValueError, "independent result review SHA not pinned"):
                viewer.load_verified()

    def test_changed_causal_records_rejected_before_display(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records.jsonl").write_text('{"type":"complete"}\n')
            with patch.object(viewer, "CAUSAL", root):
                with self.assertRaisesRegex(ValueError, "missing or changed source file"):
                    viewer.load_verified()

    def test_verified_frozen_population(self) -> None:
        if not viewer.EXPECTED_SHA["causal_review"]:
            self.skipTest("Independent result review not yet committed")
        data = viewer.load_verified()
        self.assertEqual(len(data["clips"]), 8)
        self.assertEqual(len(data["traces"]), 8)
        self.assertEqual(len(data["baselines"]), 16)
        self.assertEqual(len(data["patches"]), 224)
        self.assertEqual(sum(r["primary_eligible"] and r["arm"] == "W"
                             for r in data["patches"]), 8)
        self.assertEqual(sum(r["target_only_transfer"] and r["primary_eligible"]
                             and r["arm"] == "W" for r in data["patches"]), 8)


if __name__ == "__main__":
    unittest.main()
