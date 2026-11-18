"""CPU-only checks for the frozen development runner and its scoring gate."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import development_fp32_v1 as dev
from fit_gate_v1 import patch_tensor, score_and_gradient
from test_fit_gate_v1 import TinyAudioAnswer


class FakeTokenizer:
    eos_token_id = 151645

    def encode(self, surface, add_special_tokens=False):
        reverse = {v: k for k, v in dev.ANSWER_SURFACES.items()}
        return dev.ANSWER_IDS[reverse[surface]][:-1]


def synthetic_results() -> tuple[list[dict], dict]:
    results = []
    for site in dev.SITES_TO_CHECK:
        for engine in dev.EXPECTED_VOICES:
            for category, words in dev.EXPECTED_WORDS.items():
                for phrasing in ("chosen", "remember"):
                    for recipient, donor in ((words[0], words[1]), (words[1], words[0])):
                        case_id = f"{engine}/{category}/{phrasing}/{recipient}_to_{donor}/{site}"
                        results.append({
                            "case_id": case_id, "site": site, "engine": engine,
                            "category": category, "phrasing": phrasing,
                            "recipient": recipient, "donor": donor,
                            "pair_clean_eligible": True,
                            "controls_planned": phrasing == "chosen",
                            "conditions": [{
                                "alpha": alpha,
                                "effective_fp32_edit_norm": 1.0,
                                "measured_delta_score": 0.1 if alpha in dev.FIT_BASELINE_ALPHAS else 0.0,
                                "predicted_delta_score": 0.08 if alpha in dev.FIT_BASELINE_ALPHAS else 0.0,
                                "generated": {"answer": donor if alpha == 1 else recipient,
                                              "stopped_on_eos": True},
                            } for alpha in dev.ALPHAS],
                        })
    baseline = {
        "fit_numerical_floors": {site: 1e-4 for site in dev.SITES_TO_CHECK},
        "coefficients": {site: {"beta": 0.0} for site in dev.SITES_TO_CHECK},
    }
    return results, baseline


class DevelopmentFP32Tests(unittest.TestCase):
    def test_pinned_fit_and_development_inputs_are_present(self) -> None:
        fit, baseline = dev.verify_fit_artifacts()
        manifest = dev.verify_dev_manifest()
        self.assertEqual(fit["status"], "fit_only_gate_passed")
        self.assertEqual(len(manifest["pairs"]), 8)
        self.assertEqual(set(baseline["coefficients"]), set(dev.SITES_TO_CHECK))

    def test_candidate_answer_plus_eos_token_validation(self) -> None:
        dev.verify_candidate_tokens(SimpleNamespace(tokenizer=FakeTokenizer()))
        bad = FakeTokenizer()
        bad.eos_token_id = 1
        with self.assertRaises(ValueError):
            dev.verify_candidate_tokens(SimpleNamespace(tokenizer=bad))

    def test_complete_answer_score_contrast_matches_toy_gradient(self) -> None:
        model = TinyAudioAnswer()
        batch = {
            "input_ids": torch.tensor([[4, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        donor_score, base, donor_grad = score_and_gradient(
            model, batch, [2, 3], model.audio
        )
        recipient_score, _, recipient_grad = score_and_gradient(
            model, batch, [1, 3], model.audio
        )
        donor = base.clone()
        donor[1, 0] += 0.001
        patched, effective = patch_tensor(base, donor, [1], 1.0)
        predicted = float(((donor_grad - recipient_grad)[1] * effective[0]).sum())
        with patch.object(dev, "ANSWER_IDS", {"red": [1, 3], "blue": [2, 3]}):
            clean = dev.score_contrast(model, batch, model.audio, base, [1], "red", "blue")
            changed = dev.score_contrast(model, batch, model.audio, patched, [1], "red", "blue")
        self.assertAlmostEqual(clean["contrast"], donor_score - recipient_score, places=6)
        self.assertAlmostEqual(changed["contrast"] - clean["contrast"], predicted, delta=2e-5)

    def test_gate_keeps_all_planned_and_clean_eligible_denominators(self) -> None:
        results, baseline = synthetic_results()
        gate = dev.evaluate_development_gate(results, baseline)
        self.assertTrue(gate["pass"])
        for site in dev.SITES_TO_CHECK:
            summary = gate["sites"][site]
            self.assertEqual(summary["all_planned_directions"], 16)
            self.assertEqual(summary["clean_eligible_directions"], 16)
            self.assertEqual(summary["controls_planned_directions"], 8)
            self.assertEqual(summary["determinate_clean_small_step_conditions"], 48)
            self.assertAlmostEqual(summary["jac_mean_abs_error"], 0.02)
            self.assertAlmostEqual(summary["zero_mean_abs_error"], 0.1)
            self.assertEqual(summary["alpha_one_complete_donor_flips_all_planned"], 16)
            self.assertEqual(summary["alpha_one_complete_donor_flip_rate_all_planned"], 1.0)
            self.assertEqual(len(summary["category_phrasing_voice_breakdown"]), 8)
        one_ineligible = copy.deepcopy(results)
        one_ineligible[0]["pair_clean_eligible"] = False
        gate = dev.evaluate_development_gate(one_ineligible, baseline)
        self.assertEqual(gate["sites"]["connector"]["all_planned_directions"], 16)
        self.assertEqual(gate["sites"]["connector"]["clean_eligible_directions"], 15)
        self.assertEqual(gate["sites"]["connector"]["alpha_one_clean_eligible_denominator"], 15)
        self.assertEqual(gate["sites"]["connector"]["alpha_one_complete_donor_flip_rate_clean_eligible"], 1.0)

    def test_gate_fails_zero_comparator_and_wrong_sign(self) -> None:
        results, baseline = synthetic_results()
        for case in results:
            if case["site"] == "encoder31":
                for condition in case["conditions"]:
                    if condition["alpha"] in dev.FIT_BASELINE_ALPHAS:
                        condition["predicted_delta_score"] = -0.08
        wrong_sign = dev.evaluate_development_gate(results, baseline)
        self.assertFalse(wrong_sign["pass"])
        self.assertEqual(wrong_sign["sites"]["encoder31"]["determinate_sign_rate"], 0.0)
        zero_effect = copy.deepcopy(results)
        for case in zero_effect:
            if case["site"] == "connector":
                for condition in case["conditions"]:
                    if condition["alpha"] in dev.FIT_BASELINE_ALPHAS:
                        condition["measured_delta_score"] = 0.0
        failed = dev.evaluate_development_gate(zero_effect, baseline)
        self.assertFalse(failed["pass"])
        self.assertIn("zero_baseline_mae_zero_or_unavailable",
                      failed["sites"]["connector"]["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
