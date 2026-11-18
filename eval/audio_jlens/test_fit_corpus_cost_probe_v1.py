"""CPU-only checks for the bounded fit-only cost probe."""

import unittest

import torch

from fit_corpus_cost_probe_v1 import bin_rows, six_intervals, vjp_rows


class CostProbeTests(unittest.TestCase):
    def test_six_bins_and_disjoint_row_assignment(self):
        intervals = six_intervals(1.0, 5.0)
        self.assertEqual(intervals, [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0),
                                     (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)])
        centers = [0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5]
        row_map = {
            "encoder_rows": [{"nominal_center_seconds": c} for c in centers],
            "projected_rows": [{"nominal_center_mel_frame": c} for c in centers],
            "nominal_mel_frame_period_seconds": 1.0,
        }
        for site in ("encoder31", "connector"):
            rows = bin_rows(row_map, site, intervals, len(centers))
            self.assertEqual(rows, [[0, 1], [2], [3], [4], [5], [6]])
        with self.assertRaisesRegex(ValueError, "Row-map/source shape mismatch"):
            bin_rows(row_map, "encoder31", intervals, len(centers) + 1)
        missing = {"encoder_rows": [e for e in row_map["encoder_rows"]
                                    if e["nominal_center_seconds"] != 2.5]}
        with self.assertRaisesRegex(ValueError, "no source rows"):
            bin_rows(missing, "encoder31", intervals, 6)

    def test_both_source_vjps_share_one_graph(self):
        encoder = torch.arange(6, dtype=torch.float32).reshape(2, 3).detach().requires_grad_(True)
        connector = (encoder[:, :2] * 2).clone()
        terminal = torch.zeros((1, 2, 4), dtype=torch.float32)
        value = torch.stack((connector.sum(), encoder.sum(), connector[0].sum(), encoder[1].sum()))
        terminal = terminal.clone()
        terminal[0, -1, :] = value
        rows = list(vjp_rows(terminal, {"encoder31": encoder, "connector": connector}, (0, 1)))
        self.assertEqual(len(rows), 2)
        self.assertTrue(torch.allclose(rows[0][1]["connector"], torch.ones_like(connector)))
        self.assertTrue(torch.allclose(rows[0][1]["encoder31"],
                                       torch.tensor([[2., 2., 0.], [2., 2., 0.]])))
        self.assertTrue(torch.allclose(rows[1][1]["connector"], torch.zeros_like(connector)))
        self.assertTrue(torch.allclose(rows[1][1]["encoder31"], torch.ones_like(encoder)))
        self.assertGreaterEqual(rows[0][2], 0)


if __name__ == "__main__":
    unittest.main()
