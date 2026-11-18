"""CPU toy checks for FIT-only local-Jacobian heterogeneity math."""

import sys
import time
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_jacobian_heterogeneity_v1 import (  # noqa: E402
    N_CASES, check_deadline, pair_heldout_means, similarity_strata,
    terminal_metrics,
)
from fit_rank16_transport_v1 import accumulate_one_vjp, empty_accumulator  # noqa: E402


class HeterogeneityMathTests(unittest.TestCase):
    def test_wall_deadline_fails_closed(self):
        check_deadline(time.monotonic() + 10)
        with self.assertRaises(TimeoutError):
            check_deadline(time.monotonic() - 1)

    def test_uniform_bin_vjp_is_row_sum_and_matches_finite_difference(self):
        source = torch.arange(24, dtype=torch.float64).reshape(12, 2).requires_grad_(True)
        rows = {site: [[2*i, 2*i+1] for i in range(6)] for site in ("encoder31", "connector")}
        stage1 = {"target_basis": torch.zeros(2048, 16), "target_basis_sha256": "toy",
                  "metadata": {}, "source_bin_means": {
                      "encoder31": torch.zeros(48, 6, 2),
                      "connector": torch.zeros(48, 6, 2)}}
        accumulator = empty_accumulator(stage1)
        weights = torch.arange(1, 25, dtype=torch.float64).reshape(12, 2)
        score = (source * weights).sum()
        grad = torch.autograd.grad(score, source)[0]
        accumulate_one_vjp(accumulator,
                           {"encoder31": grad, "connector": grad}, rows, 0)
        delta = torch.tensor([[.1, -.2]] * 6, dtype=torch.float64)
        directional = torch.einsum("bd,bd->", accumulator["jacobian_sums"]["encoder31"][:, 0], delta)
        eps = 1e-5
        row_edit = torch.repeat_interleave(delta, 2, dim=0)
        fd = (((source.detach() + eps * row_edit) * weights).sum()
              - ((source.detach() - eps * row_edit) * weights).sum()) / (2 * eps)
        self.assertAlmostEqual(float(directional), float(fd), places=7)

    def test_pair_heldout_uses_other_44_and_not_own_four(self):
        local = torch.arange(N_CASES, dtype=torch.float32)[:, None, None, None]
        pairs = [f"pair_{i // 4}" for i in range(N_CASES)]
        output = pair_heldout_means(local, pairs)
        self.assertEqual(tuple(output.shape), tuple(local.shape))
        self.assertAlmostEqual(float(output[0]), float(local[4:].mean()))
        self.assertTrue(torch.equal(output[0], output[3]))
        self.assertNotEqual(float(output[0]), float(local.mean()))

    def test_fixed_similarity_strata_and_terminal_metrics(self):
        cases = []
        for p in range(12):
            for voice in ("piper", "kokoro"):
                for direction in range(2):
                    cases.append({"pair_id": f"pair_{p}",
                                  "category": ("animal", "color", "nature", "object")[p // 3],
                                  "engine": voice, "direction": direction})
        local = torch.randn(48, 6, 16, 2, generator=torch.Generator().manual_seed(3))
        strata = similarity_strata(local, cases)
        self.assertEqual({k: v["n"] for k, v in strata.items()}, {
            "within_pair_same_voice": 24, "within_pair_cross_voice": 48,
            "between_pair_same_category": 192, "across_categories": 864,
            "same_voice_all": 552, "cross_voice_all": 576,
        })
        basis = torch.eye(3)[:, :2]
        metrics = terminal_metrics(torch.tensor([1., 0., 0.]),
                                   torch.tensor([1., 0., 2.]), basis)
        self.assertAlmostEqual(metrics["full_terminal_l2_error"], 2)
        self.assertAlmostEqual(metrics["projected_terminal_l2_error"], 0)
        self.assertAlmostEqual(metrics["rank16_capture_fraction"], .2)


if __name__ == "__main__":
    unittest.main()
