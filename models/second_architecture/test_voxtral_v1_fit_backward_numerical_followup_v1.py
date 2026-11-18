"""CPU tests for frozen signed-grid arithmetic and one-shot receipts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import voxtral_v1_fit_backward_numerical_followup_v1 as followup


class NumericalFollowupTests(unittest.TestCase):
    def test_fixed_grid_has_three_repeats_and_symmetric_alphas(self):
        self.assertEqual(len(followup.CYCLES), 3)
        self.assertEqual(followup.CYCLES[1], tuple(reversed(followup.CYCLES[0])))
        self.assertEqual(followup.CYCLES[2], followup.CYCLES[0])
        self.assertEqual({alpha for order in followup.CYCLES for alpha in order},
                         {-0.1, -0.025, -0.00625, 0.00625, 0.025, 0.1})

    def test_realized_edits_and_contractions_use_native_rounding(self):
        source = torch.tensor([[1.0], [4.0]], dtype=torch.bfloat16)
        direction = torch.tensor([0.5], dtype=torch.float32)
        zero = followup.realized_edit(source, direction, 0.0, torch.bfloat16)
        edit = followup.realized_edit(source, direction, 0.025, torch.bfloat16)
        self.assertTrue(torch.equal(zero["edited"], source))
        self.assertEqual(float(zero["effective"].abs().sum()), 0.0)
        grad = torch.tensor([[[1.0], [-1.0]]] * 4)
        terms = followup.contract(grad, edit)
        self.assertEqual(terms["uniform"], [0.0] * 4)
        self.assertNotEqual(terms["rowwise"], [0.0] * 4)
        fp32_source = source.float()
        fp32_edit = followup.realized_edit(fp32_source, direction, 0.025, torch.float32)
        self.assertEqual(fp32_edit["edited"].dtype, torch.float32)
        self.assertEqual(fp32_edit["unchanged_fraction"], 0.0)

    def test_ulp_spacing_uses_native_precision(self):
        bf16 = followup.ulp_spacing(torch.ones(4, dtype=torch.bfloat16))
        fp32 = followup.ulp_spacing(torch.ones(4, dtype=torch.float32))
        self.assertEqual(len(bf16), 4)
        self.assertTrue(all(a > b > 0 for a, b in zip(bf16, fp32)))

    def test_median_and_per_repeat_errors_are_distinct(self):
        records = []
        for cycle, order in enumerate(followup.CYCLES):
            for alpha in order:
                zero = torch.zeros(3072)
                alpha_terminal = zero.clone()
                alpha_terminal[0] = alpha + (0.001 if cycle == 0 else -0.001 if cycle == 2 else 0)
                records.append({"cycle": cycle, "alpha": alpha, "zero_terminal": zero,
                                "alpha_terminal": alpha_terminal,
                                "prediction": {"rowwise": [alpha, 0, 0, 0],
                                               "uniform": [alpha, 0, 0, 0],
                                               "ideal": [alpha, 0, 0, 0]},
                                "edit_metrics": {}, "zero_sha": "zero", "alpha_sha": "alpha",
                                "zero_source_sha": "source0", "alpha_source_sha": "sourcea",
                                "zero_ulp_four": [1.0] * 4})
        analysis = followup.analyze_site(records, torch.ones(4, 1, 1))
        group = analysis["alpha_groups"][str(0.025)]
        self.assertAlmostEqual(group["median_absolute_l2_error"], 0.0, places=8)
        self.assertEqual(len(group["per_repeat"]), 3)
        self.assertGreater(group["per_repeat"][0]["absolute_l2_error"], 0.0)
        self.assertAlmostEqual(analysis["odd_even"][str(0.025)]["measured_odd_four"][0], 0.025)

    def test_preflight_failure_is_receipted_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            with patch.object(followup, "preflight", side_effect=ValueError("fixture")):
                with self.assertRaisesRegex(ValueError, "fixture"):
                    followup.run("bf16", output)
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            error = json.loads((output / "error.json").read_text())
            self.assertEqual(progress[0]["phase"], "attempt_started")
            self.assertEqual(error["error_type"], "ValueError")
            with self.assertRaises(FileExistsError):
                followup.run("bf16", output)

    def test_mps_failure_is_receipted_without_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            with patch.object(followup, "preflight", return_value={}), \
                 patch.object(torch.backends.mps, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "MPS unavailable"):
                    followup.run("fp32", output)
            error = json.loads((output / "error.json").read_text())
            self.assertEqual(error["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
