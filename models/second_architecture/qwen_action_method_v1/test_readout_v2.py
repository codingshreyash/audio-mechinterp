"""No-weight numerical tests for the target-restricted VJP readout."""
from __future__ import annotations

import unittest

import numpy as np

from . import directional_schedule_v2 as schedule
from . import readout_v2 as r


class ReadoutV2Tests(unittest.TestCase):
    def test_frozen_directional_schedule_has_all_cells_and_unique_randoms(self):
        cells = schedule.cases()
        self.assertEqual(len(cells), 30)
        rows = schedule.schedule()
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(row["split"] == "development" for row in rows), 48)
        self.assertEqual(sum(row["split"] == "locked" for row in rows), 72)
        self.assertEqual(len({(row["case_id"], row["arm"]) for row in rows}), 120)
        seeds = [row["seed"] for row in rows if row["arm"].startswith("random_")]
        self.assertEqual(len(seeds), 60)
        self.assertEqual(len(set(seeds)), 60)
        self.assertTrue(all(row["reference_label"] != row["target_label"] for row in rows))

    def test_full_label_score_includes_eos_conditional(self):
        first = np.zeros(151646, dtype=np.float64)
        after_a = np.zeros_like(first)
        after_b = np.zeros_like(first)
        after_b[r.EOS_TOKEN_ID] = 5.0
        a = r.complete_label_log_score(first, after_a, r.LABEL_TOKEN_IDS[0])
        b = r.complete_label_log_score(first, after_b, r.LABEL_TOKEN_IDS[0])
        self.assertGreater(b, a)
        with self.assertRaisesRegex(ValueError, "unfrozen"):
            r.complete_label_log_score(first, after_a, 123)

    def test_toy_mean_vjp_is_exact_for_linear_scores(self):
        h = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        g = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0]])
        y = h @ g.T + np.array([1.0, 2.0, 3.0])
        fit = r.fit_mean_vjp(h, y, np.repeat(g[None, :, :], 3, axis=0),
                             expected_fit=3)
        point = np.array([[2.0, -3.0]])
        np.testing.assert_allclose(fit.predict(point), point @ g.T + [1.0, 2.0, 3.0])
        u = r.target_direction(fit, "CANCEL", "LIST")
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0)
        a = r.matched_random_direction(u, 13)
        b = r.matched_random_direction(u, 14)
        self.assertAlmostEqual(float(np.dot(u, a)), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0)
        self.assertFalse(np.array_equal(a, b))

    def test_ridge_and_constant_are_finite_fixed_baselines(self):
        h = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        y = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0],
                      [-1.0, -1.0, 2.0]])
        fit = r.fit_fixed_ridge(h, y, expected_fit=3)
        self.assertGreater(fit.regularization, 0)
        self.assertEqual(fit.predict(h).shape, (3, 3))
        np.testing.assert_allclose(r.constant_score_baseline(y, 2)[0], y.mean(axis=0))
        self.assertEqual(r.predicted_labels(np.array([[1, 0, 0], [0, 1, 0]])),
                         ["LIST", "ASK"])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            r.fit_fixed_ridge(np.zeros((3, 2)), y, expected_fit=3)

    def test_development_gate_is_all_six_and_frozen(self):
        truth = ["LIST", "LIST", "ASK", "ASK", "CANCEL", "CANCEL"]
        constant = ["LIST"] * 6
        actual = np.repeat(np.eye(3), 2, axis=0)
        zero = np.zeros((6, 3))
        arrays = (actual, actual, zero, actual, actual, zero)
        canonical = [True] * 6
        passed = r.development_gate(truth, truth, truth, truth, truth, constant,
                                    canonical, canonical, *arrays)
        self.assertTrue(passed["pass"])
        self.assertEqual(passed["constant"], 2)
        self.assertEqual(passed["audio_vjp_to_constant_mae"], 0)
        wrong = truth.copy()
        wrong[0] = wrong[1] = None
        failed = r.development_gate(truth, wrong, truth, truth, truth, constant,
                                    canonical, canonical, *arrays)
        self.assertFalse(failed["pass"])
        flat = (actual, zero, zero, actual, zero, zero)
        score_failed = r.development_gate(truth, truth, truth, truth, truth,
                                          constant, canonical, canonical, *flat)
        self.assertFalse(score_failed["pass"])
        off_path = [True] * 4 + [False] * 2
        path_failed = r.development_gate(truth, truth, truth, truth, truth,
                                         constant, canonical, off_path, *arrays)
        self.assertFalse(path_failed["pass"])
        with self.assertRaisesRegex(ValueError, "census"):
            r.development_gate(truth[:-1], truth, truth, truth, truth,
                               constant, canonical, canonical, *arrays)


if __name__ == "__main__":
    unittest.main()
