"""CPU-only checks for FIT calibration intervention and baseline math."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_uniform_calibration_v1 import (add_uniform_bins, clean_forward,
                                        finish_summary, patched_forward,
                                        prefix_score,
                                        projected_map_diagnostic,
                                        require_matching_source_shapes)
from fit_uniform_baselines_v1 import (feature_matrix, pair_cv, ridge_fit,
                                      ridge_predict)


class FitUniformCalibrationTests(unittest.TestCase):
    def test_full_donor_recipient_shape_required(self):
        current = torch.zeros(6, 2)
        recipient = {"source_shapes": {"connector": [6, 2]}}
        donor = {"source_shapes": {"connector": [6, 2]}}
        require_matching_source_shapes(current, recipient, donor, "connector")
        donor["source_shapes"]["connector"] = [7, 2]
        with self.assertRaisesRegex(ValueError, "full source tensor shapes"):
            require_matching_source_shapes(current, recipient, donor, "connector")

    def test_rank16_map_diagnostic_identity_case(self):
        basis = torch.eye(2048, 16)
        jacobian = torch.zeros(6, 16, 2)
        jacobian[1, 0, 0] = 2
        delta = torch.zeros(6, 2)
        delta[1, 0] = 3
        measured = torch.zeros(2048)
        measured[0] = 1.5
        result = projected_map_diagnostic(basis, jacobian, delta, measured, .25)
        self.assertAlmostEqual(result["rank16_capture_fraction"], 1)
        self.assertAlmostEqual(result["projected_relative_error"], 0)
        self.assertAlmostEqual(result["projected_cosine"], 1)

    def test_clean_and_patched_hooks_measure_same_toy_edit(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Identity()
                self.connector = torch.nn.Identity()
                self.terminal = torch.nn.Linear(2, 2048, bias=False)
                with torch.no_grad():
                    self.terminal.weight.zero_()
                    self.terminal.weight[0, 0] = 1

            def forward(self, input_ids, **kwargs):
                source = self.connector(self.encoder(input_ids))
                target = self.terminal(source).unsqueeze(0)
                return SimpleNamespace(logits=target)

        model = Toy()
        rows = [[i] for i in range(6)]
        delta = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        modules = {"encoder31": model.encoder, "connector": model.connector,
                   "text.residual.35": model.terminal}
        # The real target hook sees [1,seq,2048]; wrap the toy layer to match.
        class Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                return self.inner(x).unsqueeze(0)
        model.terminal = Wrap(model.terminal)
        modules["text.residual.35"] = model.terminal
        model.forward = lambda input_ids, **kwargs: SimpleNamespace(logits=model.terminal(
            model.connector(model.encoder(input_ids))))
        batch = {"input_ids": torch.ones(6, 2)}
        clean, _ = clean_forward(model, batch, modules)
        h_patch, _, calls = patched_forward(
            model, batch, model.connector, model.terminal, rows, delta, 0.25
        )
        self.assertEqual(calls, {"source": 1, "target": 1})
        self.assertAlmostEqual(float(h_patch[0] - clean["text.residual.35"][0, -1, 0]),
                               float(0.25 * delta[-1, 0]))

    def test_prefix_log_odds_and_disjoint_uniform_rows(self):
        logits = torch.tensor([0.0, 2.0, -1.0, 1.0])
        self.assertAlmostEqual(prefix_score(logits, [1, 3], [2]),
                               float(torch.logsumexp(logits[[1, 3]].double(), 0) + 1))
        self.assertAlmostEqual(prefix_score(logits, [2], [1, 3]),
                               -prefix_score(logits, [1, 3], [2]))
        with self.assertRaises(ValueError):
            prefix_score(logits, [1, 2], [2])
        source = torch.zeros(6, 2)
        rows = [[i] for i in range(6)]
        deltas = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        result = add_uniform_bins(source, rows, deltas, 0.25)
        self.assertTrue(torch.allclose(result, deltas * 0.25))
        self.assertTrue(torch.equal(source, torch.zeros_like(source)))
        with self.assertRaises(ValueError):
            add_uniform_bins(source, [[0], [0], [2], [3], [4], [5]], deltas, 1)

    def test_row_weighted_global_feature(self):
        record = {"sites": {"connector": {
            "delta_bins": torch.tensor([[1.], [2.], [3.], [4.], [5.], [6.]]),
            "source_rows": [[0], [1, 2], [3], [4], [5], [6]],
        }}}
        global_x = feature_matrix([record], "connector", "global")
        self.assertAlmostEqual(float(global_x[0, 0]), 23 / 7)
        six_x = feature_matrix([record], "connector", "six_bin")
        self.assertEqual(tuple(six_x.shape), (1, 6))

    def test_dual_ridge_and_pair_disjoint_selection(self):
        g = torch.Generator().manual_seed(31)
        x = torch.randn(48, 7, generator=g, dtype=torch.float64)
        w = torch.randn(7, 16, generator=g, dtype=torch.float64)
        y = x @ w
        pair_ids = [f"pair_{i}" for i in range(12) for _ in range(4)]
        fit, table = pair_cv(x, y, pair_ids)
        self.assertEqual(len(table), 6)
        self.assertEqual(tuple(ridge_predict(fit, x).shape), (48, 16))
        self.assertLess(float((ridge_predict(fit, x) - y).square().mean()),
                        float(y.square().mean()) * 1e-3)
        with self.assertRaises(ValueError):
            ridge_fit(torch.zeros(48, 7), y, 1e-4)

    def test_floor_uses_all_fit_zero_repeats(self):
        records = [{"sites": {site: {"zero_v2_scores": [0.0, 0.0, 0.0]}
                              for site in ("encoder31", "connector")}}
                   for _ in range(48)]
        summary = finish_summary(records)
        self.assertEqual(summary["numerical_floor_v2_by_site"],
                         {"encoder31": 1e-4, "connector": 1e-4})


if __name__ == "__main__":
    unittest.main()
