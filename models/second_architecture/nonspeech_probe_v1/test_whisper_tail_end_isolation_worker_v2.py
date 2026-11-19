"""No-weight tests of V2 conditional schedule and relocation arithmetic."""
from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from . import whisper_tail_end_isolation_gate_v2 as gate
from . import whisper_tail_end_isolation_plan_v2 as plan
from . import whisper_tail_end_isolation_worker_v2 as worker


class WorkerTests(unittest.TestCase):
    def test_every_stage_has_fixed_baselines_and_maximum_calls(self) -> None:
        frozen = json.loads(plan.PLAN.read_text())
        for stage, baseline_count, maximum in (("original", 2, 7),
                                               ("offset_8000", 4, 24),
                                               ("offset_16000", 4, 24)):
            self.assertEqual(len(worker.baselines_for(stage, frozen)), baseline_count)
            layouts = ("original",) if stage == "original" else ("prepend", "append")
            calls, skipped = worker.planned_calls(
                stage, frozen, {layout: True for layout in layouts})
            self.assertEqual(len(calls), maximum)
            self.assertEqual(skipped, [])

    def test_original_failure_and_one_layout_failure_keep_baselines(self) -> None:
        frozen = json.loads(plan.PLAN.read_text())
        calls, skipped = worker.planned_calls("original", frozen, {"original": False})
        self.assertEqual(len(calls), 2)
        self.assertEqual(skipped, [1, 2, 3, 4, 5])
        calls, skipped = worker.planned_calls("offset_8000", frozen,
                                              {"prepend": False, "append": True})
        self.assertEqual(len(calls), 14)
        self.assertEqual(skipped, list(range(1, 11)))
        self.assertEqual({call["layout"] for call in calls[:4]}, {"prepend", "append"})

    def test_relocation_uses_aligned_nonzero_delta_at_competing_rows(self) -> None:
        for stage in ("offset_8000", "offset_16000"):
            for arm in (item for item in plan.arms_for(stage)
                        if item["kind"] == "relocated_correct_delta"):
                mapped = worker.execution_arm(arm)
                self.assertEqual(mapped["kind"], "norm_matched_shifted_delta")
                self.assertEqual(mapped["target_rows"], arm["delta_source_rows"])
                self.assertEqual(mapped["source_rows"], arm["target_rows"])
                donor = np.zeros((1500, 384), dtype=np.float32)
                recipient = donor.copy()
                donor[mapped["target_rows"], 0] = 2.0
                edited = recipient.copy()
                edited[mapped["source_rows"], 0] = (
                    donor[mapped["target_rows"]] - recipient[mapped["target_rows"]])[:, 0]
                self.assertEqual(float(np.linalg.norm(edited)),
                                 float(np.linalg.norm(donor[mapped["target_rows"]])))
                self.assertEqual(int(np.count_nonzero(edited[mapped["target_rows"]])), 0)

    def test_direct_worker_without_fixed_attempt_stops_before_weights(self) -> None:
        with mock.patch.object(worker.v4.old, "_snapshot_load") as load:
            with self.assertRaisesRegex(ValueError, "fixed expected V2"):
                worker.run("original", gate.ATTEMPTS["original"],
                           gate.ADMISSIONS["original"], gate.GOS["original"], -1)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
