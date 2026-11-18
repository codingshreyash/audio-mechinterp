"""Synthetic CPU checks for the prospective FIT-only router implementation."""

import math
import time
import unittest

import numpy as np
import torch

import fit_conditional_routing_diagnostic_v1 as assay


class RouterDiagnosticTests(unittest.TestCase):
    def test_pair_exclusion_and_wrong_router_slots(self):
        ids = [f"pair_{p:02d}/fit_chosen/{voice}/{word}"
               for p in range(12) for voice in ("kokoro", "piper")
               for word in ("apple", "berry")]
        pairs, groups, wrong = assay.pair_and_slot(ids)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len(groups), 12)
        self.assertEqual(len(wrong), 48)
        for i, j in wrong.items():
            a = assay.case_parts(ids[i])
            b = assay.case_parts(ids[j])
            self.assertNotEqual(a[0], b[0])
            self.assertEqual(a[2:], b[2:])

    def test_jacobian_edit_contraction_preserves_bin_rank_axes(self):
        j = torch.zeros(2, 6, assay.RANK, 3)
        d = torch.zeros(2, 6, 3)
        j[0, 4, 7, 2] = 3.0
        j[1, 2, 3, 0] = -2.0
        d[0, 4, 2] = 5.0
        d[1, 2, 0] = 7.0
        out = assay.response_coordinates(j, d)
        self.assertEqual(tuple(out.shape), (2, 2, assay.RANK))
        for query in range(2):
            for source in range(2):
                expected = assay.ALPHA * torch.einsum("bkd,bd->k", j[source], d[query])
                self.assertTrue(torch.allclose(out[query, source], expected.double()))
        self.assertAlmostEqual(out[0, 0, 7].item(), 0.375)
        self.assertAlmostEqual(out[1, 1, 3].item(), -0.35)

    def test_train_only_pca_and_routing_weights(self):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(48, 6, 4))
        groups = [f"pair_{i//4:02d}" for i in range(48)]
        train = np.arange(4, 48)
        phi, train_phi, h = assay.fit_fold_features(features, train, groups)
        self.assertEqual(phi.shape, (48, 8))
        self.assertEqual(train_phi.shape, (44, 8))
        self.assertTrue(math.isfinite(h) and h > 0)
        weights = assay.gaussian_weights(phi[0], train_phi, h)
        self.assertEqual(len(weights), 44)
        self.assertAlmostEqual(weights.sum(), 1.0)
        changed = features.copy()
        changed[:4] += 1000.0
        phi_changed, train_changed, h_changed = assay.fit_fold_features(changed, train, groups)
        self.assertTrue(np.allclose(train_phi, train_changed))
        self.assertEqual(h, h_changed)
        self.assertFalse(np.allclose(phi[:4], phi_changed[:4]))

    def test_case_error_and_fail_closed_geometry(self):
        basis = torch.eye(2048, dtype=torch.float64)[:, :assay.RANK]
        pred = torch.zeros(assay.RANK)
        target = torch.ones(assay.RANK)
        actual = basis @ target.double()
        item = assay.case_errors(pred, target, actual, basis)
        self.assertAlmostEqual(item["projected_terminal_l2"], 4.0)
        self.assertAlmostEqual(item["full_terminal_l2"], 4.0)
        with self.assertRaises(ValueError):
            assay.response_coordinates(torch.zeros(2, 6, assay.RANK, 3),
                                       torch.zeros(2, 5, 3))

    def test_synthetic_full_evaluation_keeps_48_and_oracle_ceiling(self):
        ids = [f"pair_{p:02d}/fit_chosen/{voice}/{word}"
               for p in range(12) for voice in ("kokoro", "piper")
               for word in ("apple", "berry")]
        pairs, groups, wrong = assay.pair_and_slot(ids)
        gen = torch.Generator().manual_seed(23)
        features = torch.randn(48, 6, 4, generator=gen)
        jac = torch.randn(48, 6, assay.RANK, 4, generator=gen)
        delta = torch.randn(48, 6, 4, generator=gen)
        local = assay.ALPHA * torch.einsum("ibkd,ibd->ik", jac, delta)
        actual = torch.zeros(48, 2048)
        actual[:, :assay.RANK] = local
        basis = torch.eye(2048, dtype=torch.float64)[:, :assay.RANK]
        data = {"case_ids": ids, "pairs": pairs, "by_pair": groups,
                "wrong_query": wrong, "basis": basis,
                "sites": {site: {"features": features, "jacobians": jac,
                                 "deltas": delta, "actuals": actual}
                          for site in assay.SITES}}
        result = assay.evaluate(data, time.monotonic())
        for site in assay.SITES:
            self.assertEqual(len(result[site]["rows"]), 48)
            self.assertLess(result[site]["summary"]["local_in_sample_oracle"]
                            ["mean_projected_terminal_l2"], 1e-5)
            self.assertGreater(result[site]["summary"]["global_pair_excluded"]
                               ["mean_projected_terminal_l2"], 1e-4)
            self.assertEqual(len(result[site]["pair_bootstrap_95_percentile_ratio"]), 2)


if __name__ == "__main__":
    unittest.main()
