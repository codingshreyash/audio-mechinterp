"""No-model tests of translation-worker arm and branch semantics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from . import whisper_time_translation_plan_v1 as plan
from . import whisper_time_translation_gate_v1 as gate
from . import whisper_time_translation_worker_v1 as worker


def fake_outcome(sidecars: Path, stem: str, tokens: list[int], mask: np.ndarray) -> dict:
    name = f"{stem}_mask.npy"
    np.save(sidecars / name, mask, allow_pickle=False)
    return {"output": {"termination": "eos", "eos_seen": True,
                       "censored": False, "sequence_ids": tokens,
                       "prefix_ids": [1, 2]},
            "sidecars": {"mask": {"file": name}}}


class WorkerTests(unittest.TestCase):
    def test_direct_worker_without_parent_attempt_refuses_before_weights(self) -> None:
        stage = "original"
        with mock.patch.object(worker.v4.old, "_snapshot_load") as load:
            with self.assertRaisesRegex(ValueError, "fixed expected"):
                worker.run(stage, gate.ATTEMPTS[stage], gate.ADMISSIONS[stage],
                           gate.GOS[stage], -1)
            load.assert_not_called()

    def test_pinned_plan_baseline_and_arm_counts(self) -> None:
        frozen = json.loads(plan.PLAN.read_text())
        for stage, baseline_count, maximum in (
            ("original", 2, 28), ("offset_8000", 4, 62),
            ("offset_16000", 4, 62)):
            self.assertEqual(len(worker.baselines_for(stage, frozen)), baseline_count)
            layouts = ("original",) if stage == "original" else ("prepend", "append")
            calls, skipped = worker.planned_calls(
                stage, frozen, {name: True for name in layouts})
            self.assertEqual(len(calls), maximum)
            self.assertEqual(skipped, [])

    def test_conditional_skip_is_whole_layout_and_deterministic(self) -> None:
        frozen = json.loads(plan.PLAN.read_text())
        calls, skipped = worker.planned_calls(
            "offset_8000", frozen, {"prepend": False, "append": True})
        self.assertEqual(len(calls), 4 + 29)
        self.assertEqual(skipped, list(range(1, 30)))
        self.assertEqual({c["layout"] for c in calls[4:]}, {"append"})
        calls, skipped = worker.planned_calls(
            "offset_8000", frozen, {"prepend": False, "append": False})
        self.assertEqual(len(calls), 4)
        self.assertEqual(skipped, list(range(1, 59)))

    def test_cross_window_mapping_preserves_exact_delta_and_norm(self) -> None:
        arms = plan.arms_for("offset_8000")
        for arm in (a for a in arms if a["kind"] == "relocated_correct_delta"):
            mapped = worker.execution_arm(arm)
            self.assertEqual(mapped["kind"], "norm_matched_shifted_delta")
            self.assertEqual(mapped["target_rows"], arm["delta_source_rows"])
            self.assertEqual(mapped["source_rows"], arm["target_rows"])
            # This is the exact patch arithmetic used by the inherited V3 hook.
            recipient = np.zeros((1500, 3), dtype=np.float32)
            donor = np.zeros_like(recipient)
            delta = np.arange(1, len(mapped["target_rows"]) * 3 + 1,
                              dtype=np.float32).reshape(-1, 3)
            donor[mapped["target_rows"]] = delta
            edited = recipient.copy()
            edited[mapped["source_rows"]] += (
                donor[mapped["target_rows"]] - recipient[mapped["target_rows"]])
            self.assertTrue(np.array_equal(edited[mapped["source_rows"]], delta))
            self.assertEqual(float(np.linalg.norm(edited)),
                             float(np.linalg.norm(delta)))
            self.assertEqual(int(np.count_nonzero(edited[mapped["target_rows"]])), 0)

    def test_pair_eligibility_requires_complete_contrast_and_equal_masks(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            sidecars = Path(folder)
            mask = np.ones(3000, dtype=np.int64)
            zero = fake_outcome(sidecars, "z", [10, 20], mask)
            noise = fake_outcome(sidecars, "n", [10, 21], mask)
            self.assertTrue(worker.pair_eligibility(zero, noise, sidecars)["eligible"])
            noise["output"]["sequence_ids"] = [10, 20]
            self.assertFalse(worker.pair_eligibility(zero, noise, sidecars)["eligible"])
            noise["output"]["sequence_ids"] = [10, 21]
            noise["output"]["censored"] = True
            self.assertFalse(worker.pair_eligibility(zero, noise, sidecars)["eligible"])
            noise["output"]["censored"] = False
            np.save(sidecars / "n_mask.npy", np.zeros(3000, dtype=np.int64),
                    allow_pickle=False)
            self.assertFalse(worker.pair_eligibility(zero, noise, sidecars)["eligible"])


if __name__ == "__main__":
    unittest.main()
