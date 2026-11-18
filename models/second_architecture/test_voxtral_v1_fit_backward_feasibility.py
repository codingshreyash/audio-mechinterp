"""CPU tests for the frozen, non-running backward feasibility math."""

import unittest

import torch

import voxtral_v1_fit_backward_feasibility as pilot


class BackwardFeasibilityMathTests(unittest.TestCase):
    def test_zero_bf16_edit_is_exact(self):
        recipient = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.bfloat16)
        donor = torch.tensor([[2.0, 1.0], [2.0, 1.0]], dtype=torch.bfloat16)
        edit = pilot.uniform_edit(recipient, donor, [0, 1], 0.0)
        self.assertTrue(torch.equal(edit["edited"], recipient))
        self.assertEqual(float(edit["effective"].abs().sum()), 0.0)

    def test_rowwise_rounding_changes_uniform_contraction(self):
        # BF16 spacing differs between the two recipient rows. The intended
        # uniform edit can therefore become a nonuniform realized edit.
        recipient = torch.tensor([[1.0], [4.0]], dtype=torch.bfloat16)
        donor = torch.tensor([[1.5], [4.5]], dtype=torch.bfloat16)
        edit = pilot.uniform_edit(recipient, donor, [0, 1], 0.025)
        grad = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
        terms = pilot.contract_rows(grad, edit)
        self.assertEqual(terms["uniform_mean"], 0.0)
        self.assertEqual(terms["ideal_uniform"], 0.0)
        self.assertNotEqual(terms["rowwise_oracle"], 0.0)

    def test_both_source_vjps_share_one_graph(self):
        # Encoder -> projector -> terminal. Both derivatives must come from
        # the same graph, as in the planned Voxtral backward pass.
        encoder = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        projector = 2.0 * encoder
        terminal = (3.0 * projector).sum()
        ge, gp = torch.autograd.grad(terminal, (encoder, projector))
        self.assertTrue(torch.equal(ge, torch.full_like(encoder, 6.0)))
        self.assertTrue(torch.equal(gp, torch.full_like(projector, 3.0)))

    def test_vector_error_requires_four_finite_coordinates(self):
        self.assertEqual(pilot.vector_error([0, 1, 2, 3], [0, 1, 2, 3])["absolute_l2"], 0.0)
        with self.assertRaises(ValueError):
            pilot.vector_error([1], [1])
        with self.assertRaises(ValueError):
            pilot.vector_error([0, 1, float("nan"), 3], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
