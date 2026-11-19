"""No-weight checks for the frozen FIT normalization counterfactual algebra."""

import unittest

import numpy as np
import torch
import torch.nn.functional as functional

from models.second_architecture import wav2vec2_fit_noise_norm_causal_source_v1 as source


class FrozenSourcePlanTests(unittest.TestCase):
    def test_saved_plan_covers_every_pair_and_forward(self):
        plan = source.no_weight_plan()  # verifies source/manifest/PCM/result hashes
        self.assertEqual(plan["weight_loads"], 0)
        self.assertEqual(len(plan["pairs"]), 8)
        self.assertEqual(len(plan["stage_ii_cases"]), 4)
        self.assertEqual(plan["scored_forward_count"], 48)
        self.assertEqual(plan["total_forward_ceiling"], 80)
        self.assertEqual(
            [(p["case"], p["noise_arm"]) for p in plan["pairs"]],
            [(f"{engine}_{word}", arm) for engine in source.ENGINES
             for word in source.WORDS for arm in source.NOISE_ARMS],
        )
        for pair in plan["pairs"]:
            self.assertEqual(
                [item["arm"] for item in pair["prior_stock"]],
                ["clean", pair["noise_arm"]],
            )
            for item in pair["prior_stock"]:
                self.assertEqual(len(item["pcm_sha256"]), 64)
                self.assertEqual(len(item["argmax_ids_sha256"]), 64)

    def test_input_statistics_and_crossed_statistics(self):
        clean = np.array([-3000, -1000, 1000, 3000, 0, 400], dtype="<i2")
        noisy = clean.copy()
        noisy[2] += 1300
        clean_stats = source.input_statistics(clean)
        noisy_stats = source.input_statistics(noisy)
        x = clean.astype(np.float32) / np.float32(32768)
        expected = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        self.assertTrue(np.array_equal(
            source.normalize_with_statistics(clean, *clean_stats), expected))
        self.assertFalse(np.array_equal(
            source.normalize_with_statistics(clean, *clean_stats),
            source.normalize_with_statistics(clean, *noisy_stats)))
        with self.assertRaisesRegex(ValueError, "PCM16"):
            source.input_statistics(clean.astype(np.float32))

    def test_shared_outside_statistics_requires_exact_pcm_invariant(self):
        clean = np.array([-900, -500, 200, 500, 1000, -200], dtype="<i2")
        noisy = clean.copy()
        noisy[2:4] += 300
        stats = source.common_outside_statistics(clean, noisy, (2, 4))
        normalized_clean = source.normalize_with_statistics(clean, *stats)
        normalized_noisy = source.normalize_with_statistics(noisy, *stats)
        self.assertTrue(np.array_equal(normalized_clean[:2], normalized_noisy[:2]))
        self.assertTrue(np.array_equal(normalized_clean[4:], normalized_noisy[4:]))
        changed_outside = noisy.copy()
        changed_outside[0] += 1
        with self.assertRaisesRegex(ValueError, "outside"):
            source.common_outside_statistics(clean, changed_outside, (2, 4))

    def test_receptive_field_overlap_boundary(self):
        # Frames [0..6] cover samples [0..39] with kernel 10/stride 5.
        mask = source.first_conv_outside_mask(7, (15, 21))
        self.assertEqual(mask.tolist(), [True, True, False, False, False, True, True])

    def test_first_groupnorm_manual_own_stats_matches_torch(self):
        generator = torch.Generator().manual_seed(20261119)
        raw = torch.randn((1, 512, 31), generator=generator, dtype=torch.float32)
        gamma = torch.randn((512,), generator=generator, dtype=torch.float32)
        beta = torch.randn((512,), generator=generator, dtype=torch.float32)
        own_stats = source.groupnorm_statistics(raw)
        manual = source.groupnorm_with_statistics(raw, own_stats, gamma, beta)
        native = functional.group_norm(raw, num_groups=512, weight=gamma,
                                       bias=beta, eps=source.FIRST_GN_EPSILON)
        self.assertLessEqual((manual - native).abs().max().item(), 2e-5)

    def test_shared_stats_preserve_outside_raw_equality(self):
        generator = torch.Generator().manual_seed(20261119)
        clean = torch.randn((1, 512, 31), generator=generator, dtype=torch.float32)
        noisy = clean.clone()
        noisy[:, :, 12:17] += 0.5
        gamma = torch.ones((512,), dtype=torch.float32)
        beta = torch.zeros((512,), dtype=torch.float32)
        clean_stats = source.groupnorm_statistics(clean)
        clean_norm = source.groupnorm_with_statistics(clean, clean_stats, gamma, beta)
        noisy_norm = source.groupnorm_with_statistics(noisy, clean_stats, gamma, beta)
        self.assertTrue(torch.equal(clean_norm[:, :, :12], noisy_norm[:, :, :12]))
        self.assertTrue(torch.equal(clean_norm[:, :, 17:], noisy_norm[:, :, 17:]))
        crossed = source.groupnorm_with_statistics(
            clean, source.groupnorm_statistics(noisy), gamma, beta)
        self.assertFalse(torch.equal(clean_norm[:, :, :12], crossed[:, :, :12]))
        with self.assertRaisesRegex(ValueError, "shape invalid"):
            source.groupnorm_with_statistics(clean, clean_stats, gamma[:10], beta)


if __name__ == "__main__":
    unittest.main()
