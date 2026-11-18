"""CPU checks for FIT-only FP32 phase-1 score and control construction."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import fit_fp32_phase1_v1 as phase1
from fit_gate_v1 import patch_tensor, score_and_gradient
from test_fit_gate_v1 import TinyAudioAnswer


class FP32Phase1Tests(unittest.TestCase):
    def test_baseline_progress_hash_binds_all_eight_saved_cases(self) -> None:
        results = [
            {"case_id": f"case_{i}", "measured_delta_score": i / 10}
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit.progress.jsonl"
            path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in results))
            digest = phase1.hash_complete_fit_progress(path, results)
            self.assertEqual(digest, phase1.sha(path))
            with self.assertRaises(ValueError):
                phase1.hash_complete_fit_progress(path, results[:-1])
            changed = [dict(r) for r in results]
            changed[0]["measured_delta_score"] = 999
            with self.assertRaises(ValueError):
                phase1.hash_complete_fit_progress(path, changed)

    def test_controls_preserve_declared_rows_and_effective_norms(self) -> None:
        base = torch.tensor([
            [0.2, 0.1], [0.4, -0.1],
            [0.0, 0.0], [0.0, 0.0],
        ], dtype=torch.float32)
        donor = base.clone()
        donor[0] += torch.tensor([0.3, 0.1])
        donor[1] += torch.tensor([-0.1, 0.2])
        controls = phase1.make_controls(base, donor, [0, 1], [2, 3])
        by_name = {c["name"]: c for c in controls}
        self.assertEqual(len(controls), 7)
        self.assertEqual(float(by_name["self_patch"]["effective"].norm()), 0.0)
        self.assertEqual(float(by_name["wrong_time_natural"]["effective"].norm()), 0.0)
        word_norm = float((donor[:2] - base[:2]).norm())
        for name in (
            "random_same_rows_seed_0", "random_same_rows_seed_1",
            "random_same_rows_seed_2", "row_shuffled_donor",
        ):
            self.assertAlmostEqual(
                float(by_name[name]["effective"].norm()), word_norm,
                delta=2e-6,
            )
        self.assertFalse(by_name["norm_scaled_offword_synthetic"]["eligible"])
        self.assertEqual(by_name["norm_scaled_offword_synthetic"]["note"], "ineligible_zero_offword_delta")
        self.assertEqual(by_name["norm_scaled_offword_synthetic"]["rows"], [2, 3])
        self.assertTrue(torch.equal(
            by_name["row_shuffled_donor"]["replacement"][2:], base[2:]
        ))
        permutation = np.random.default_rng(0).permutation(2).tolist()
        self.assertTrue(torch.equal(
            by_name["row_shuffled_donor"]["effective"],
            (donor[:2] - base[:2])[permutation],
        ))
        self.assertTrue(torch.equal(
            phase1.make_controls(base, donor, [0, 1], [2, 3])[2]["effective"],
            by_name["random_same_rows_seed_0"]["effective"],
        ))

    def test_offword_selection_prefers_disjoint_chunk_then_lowest_start(self) -> None:
        row_map = {
            "encoder_rows": [
                {"row_index": i, "nominal_center_seconds": i * .1,
                 "chunk_index": 0 if i < 4 else 1 if i < 8 else 2}
                for i in range(12)
            ]
        }
        rows, offword, rule = phase1.controls_rows(row_map, "encoder31", .4, .6)
        self.assertEqual(rows, [4, 5])
        self.assertEqual(offword, [0, 1])
        self.assertEqual(rule, "disjoint_attention_chunk_lowest_start")
        short_map = {"encoder_rows": row_map["encoder_rows"][4:8]}
        for index, entry in enumerate(short_map["encoder_rows"]):
            entry = dict(entry)
            entry["row_index"] = index
            short_map["encoder_rows"][index] = entry
        rows, offword, rule = phase1.controls_rows(short_map, "encoder31", .4, .6)
        self.assertEqual(rows, [0, 1])
        self.assertEqual(offword, [2, 3])
        self.assertEqual(rule, "lowest_nonoverlapping_fallback")

    def test_teacher_forced_contrast_and_row_prediction(self) -> None:
        model = TinyAudioAnswer()
        batch = {
            "input_ids": torch.tensor([[4, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        donor_score, base_d, donor_grad = score_and_gradient(
            model, batch, [2, 3], model.audio
        )
        recipient_score, base_r, recipient_grad = score_and_gradient(
            model, batch, [1, 3], model.audio
        )
        self.assertTrue(torch.equal(base_d, base_r))
        donor = base_d.clone()
        donor[1, 0] += 0.001
        replacement, effective = patch_tensor(base_d, donor, [1], 1.0)
        grad = donor_grad - recipient_grad
        terms = phase1.predict_row_contributions(grad, effective, [1])
        self.assertEqual(len(terms), 1)
        with patch.object(phase1, "ANSWER_IDS", {"red": [1, 3], "blue": [2, 3]}):
            baseline = phase1.score_contrast(
                model, batch, model.audio, base_d, [1], "red", "blue"
            )
            patched = phase1.score_contrast(
                model, batch, model.audio, replacement, [1], "red", "blue"
            )
        self.assertAlmostEqual(
            baseline["contrast"], donor_score - recipient_score, places=6
        )
        self.assertAlmostEqual(
            patched["contrast"] - baseline["contrast"],
            float(sum(terms)), delta=2e-5,
        )
        self.assertEqual(len(model.audio._forward_hooks), 0)

    def test_fit_only_norm_baseline_uses_exact_frozen_alpha_rows(self) -> None:
        results = []
        for site in phase1.SITES_TO_CHECK:
            for engine in phase1.ENGINES:
                for recipient, donor in phase1.DIRECTIONS:
                    results.append({
                        "case_id": f"{engine}/{recipient}_to_{donor}/{site}",
                        "site": site, "word_rows": [0, 1],
                        "source_audio_sha256": "a", "donor_audio_sha256": "b",
                        "conditions": [
                            {"alpha": alpha, "effective_fp32_edit_norm": 2.0,
                             "measured_delta_score": 0.4}
                            for alpha in phase1.ALPHAS
                        ],
                    })
        fitted = phase1.fit_norm_baseline(results)
        for site in phase1.SITES_TO_CHECK:
            self.assertEqual(len(fitted[site]["fit_rows"]), 12)
            self.assertAlmostEqual(fitted[site]["beta"], 0.2)
        results[0]["conditions"][1]["measured_delta_score"] = -100.0
        fitted_with_negative = phase1.fit_norm_baseline(results)
        self.assertEqual(fitted_with_negative["connector"]["beta"], 0.0)

    def test_frozen_fit_stop_rule_counts_flips_and_determinate_signs(self) -> None:
        clean = [{"eligible": True} for _ in range(4)]
        results = []
        for site in phase1.SITES_TO_CHECK:
            for engine in phase1.ENGINES:
                for recipient, donor in phase1.DIRECTIONS:
                    results.append({
                        "case_id": f"{engine}/{recipient}_to_{donor}/{site}",
                        "engine": engine, "recipient": recipient,
                        "donor": donor, "site": site,
                        "zero_contrast_max_abs_deviation_from_mean": 0.0,
                        "conditions": [{
                            "alpha": alpha,
                            "measured_delta_score": 0.1 if alpha in phase1.FIT_BASELINE_ALPHAS else 0.0,
                            "predicted_delta_score": 0.08 if alpha in phase1.FIT_BASELINE_ALPHAS else 0.0,
                            "generated": {
                                "answer": donor if alpha == 1 else recipient,
                                "stopped_on_eos": True,
                            },
                        } for alpha in phase1.ALPHAS],
                    })
        passed = phase1.fit_gate_summary(results, clean)
        self.assertTrue(passed["pass"])
        self.assertTrue(passed["development_permitted_by_fit_gate"])
        for site in phase1.SITES_TO_CHECK:
            summary = passed["sites"][site]
            self.assertEqual(summary["numerical_floor"], 1e-4)
            self.assertEqual(summary["complete_donor_flips"], 4)
            self.assertEqual(summary["small_step_determinate"], 12)
            self.assertEqual(summary["small_step_sign_matches"], 12)

        results[0]["conditions"][-1]["generated"]["answer"] = "red"
        failed_flip = phase1.fit_gate_summary(results, clean)
        self.assertFalse(failed_flip["pass"])
        self.assertEqual(failed_flip["sites"]["connector"]["complete_donor_flips"], 3)
        self.assertIn(
            "fewer_than_four_alpha_one_complete_donor_flips",
            failed_flip["sites"]["connector"]["failure_reasons"],
        )
        results[0]["conditions"][-1]["generated"]["answer"] = "blue"
        for case in results:
            if case["site"] == "encoder31":
                for condition in case["conditions"]:
                    if condition["alpha"] in phase1.FIT_BASELINE_ALPHAS:
                        condition["measured_delta_score"] = 1e-5
        indeterminate = phase1.fit_gate_summary(results, clean)
        self.assertFalse(indeterminate["pass"])
        self.assertEqual(indeterminate["sites"]["encoder31"]["small_step_determinate"], 0)
        self.assertIn(
            "no_determinate_small_step_effects",
            indeterminate["sites"]["encoder31"]["failure_reasons"],
        )
        for case in results:
            if case["site"] == "encoder31":
                for condition in case["conditions"]:
                    if condition["alpha"] in phase1.FIT_BASELINE_ALPHAS:
                        condition["measured_delta_score"] = -0.1
        wrong_sign = phase1.fit_gate_summary(results, clean)
        self.assertFalse(wrong_sign["pass"])
        self.assertEqual(wrong_sign["sites"]["encoder31"]["small_step_sign_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
