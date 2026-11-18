"""CPU-only checks for the bounded FP32 fit diagnostic driver."""

from __future__ import annotations

import copy
import unittest

from fit_fp32_diagnostic_v1 import mps_cap_fraction, relabel_fp32_edits
from fit_gate_v1 import MAX_MPS_DRIVER_BYTES


class FitFP32DiagnosticTests(unittest.TestCase):
    def test_mps_fraction_never_exceeds_fixed_driver_ceiling(self) -> None:
        self.assertEqual(mps_cap_fraction(MAX_MPS_DRIVER_BYTES), 1.0)
        self.assertEqual(mps_cap_fraction(2 * MAX_MPS_DRIVER_BYTES), 0.5)
        self.assertEqual(mps_cap_fraction(MAX_MPS_DRIVER_BYTES // 2), 1.0)
        with self.assertRaises(ValueError):
            mps_cap_fraction(0)

    def test_edit_norm_is_labeled_as_fp32_without_changing_value(self) -> None:
        source = {"conditions": [{
            "alpha": 0.05,
            "effective_bf16_delta_norm": 1.234,
            "predicted_delta_score": 0.2,
        }]}
        result = relabel_fp32_edits(copy.deepcopy(source))
        condition = result["conditions"][0]
        self.assertNotIn("effective_bf16_delta_norm", condition)
        self.assertEqual(condition["effective_fp32_delta_norm"], 1.234)
        self.assertEqual(condition["effective_edit_dtype"], "float32")
        self.assertEqual(condition["predicted_delta_score"], 0.2)
        with self.assertRaises(ValueError):
            relabel_fp32_edits({"conditions": [{"alpha": 0.05}]})


if __name__ == "__main__":
    unittest.main()
