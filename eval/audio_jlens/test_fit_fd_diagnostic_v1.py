"""CPU checks for the separate fit-only finite-difference diagnostic."""

from __future__ import annotations

import unittest

import torch

from fit_fd_diagnostic_v1 import score_grad_mode_patch
from fit_gate_v1 import patch_tensor, score_and_gradient, score_with_patch
from test_fit_gate_v1 import TinyAudioAnswer


class FitFiniteDifferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyAudioAnswer()
        self.batch = {
            "input_ids": torch.tensor([[4, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def test_grad_mode_zero_and_small_directional_patch(self) -> None:
        candidate = [2, 3]
        vjp_score, base, grad = score_and_gradient(
            self.model, self.batch, candidate, self.model.audio
        )
        donor = base.clone()
        donor[0] += torch.tensor([0.001, -0.002])
        zero, zero_delta = patch_tensor(base, donor, [0], 0.0)
        self.assertEqual(float(zero_delta.norm()), 0.0)
        zero_grad = score_grad_mode_patch(
            self.model, self.batch, candidate, self.model.audio, zero, [0], base
        )
        zero_no_grad = score_with_patch(
            self.model, self.batch, candidate, self.model.audio, zero, [0]
        )
        self.assertAlmostEqual(zero_grad, vjp_score, places=6)
        self.assertAlmostEqual(zero_grad, zero_no_grad, places=6)

        replacement, delta = patch_tensor(base, donor, [0], 1.0)
        predicted = float((grad[0] * delta[0]).sum())
        patched_grad = score_grad_mode_patch(
            self.model, self.batch, candidate, self.model.audio,
            replacement, [0], base,
        )
        patched_no_grad = score_with_patch(
            self.model, self.batch, candidate, self.model.audio,
            replacement, [0],
        )
        self.assertAlmostEqual(patched_grad - zero_grad, predicted, delta=2e-5)
        self.assertAlmostEqual(patched_no_grad - zero_no_grad, predicted, delta=2e-5)
        self.assertEqual(len(self.model.audio._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
