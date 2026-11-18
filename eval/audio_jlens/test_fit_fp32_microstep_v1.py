"""CPU-only precision checks for the encoder FP32 microstep driver."""

from __future__ import annotations

import unittest

import torch

from fit_fp32_microstep_v1 import (
    ALPHAS, SITES_TO_CHECK, contrast, edit_pair_with_bf16_cast,
)


class FitFP32MicrostepTests(unittest.TestCase):
    def test_frozen_sites_alpha_grid_and_donor_orientation(self) -> None:
        self.assertEqual(SITES_TO_CHECK, ("encoder15", "encoder31"))
        self.assertEqual(ALPHAS, (0.0, 0.001, 0.0025, 0.005, 0.01, 0.025))
        self.assertEqual(contrast({"red": -2.0, "blue": -1.5}), 0.5)

    def test_fp32_edit_and_hypothetical_bf16_cast_are_distinct(self) -> None:
        base = torch.tensor([[1.0, 0.0], [2.0, 3.0]], dtype=torch.float32)
        donor = torch.tensor([[1.1, 0.002], [8.0, 9.0]], dtype=torch.float32)
        fp32, cast = edit_pair_with_bf16_cast(base, donor, [0], 0.001)
        self.assertEqual(fp32.dtype, torch.float32)
        self.assertEqual(cast.dtype, torch.float32)
        self.assertGreater(float(fp32[0, 0]), 0)
        self.assertEqual(float(cast[0, 0]), 0.0)
        self.assertTrue(torch.equal(base[1], torch.tensor([2.0, 3.0])))
        zero_fp32, zero_cast = edit_pair_with_bf16_cast(base, donor, [0], 0.0)
        self.assertEqual(float(zero_fp32.norm()), 0.0)
        self.assertEqual(float(zero_cast.norm()), 0.0)
        with self.assertRaises(ValueError):
            edit_pair_with_bf16_cast(base.half(), donor, [0], 0.001)


if __name__ == "__main__":
    unittest.main()
