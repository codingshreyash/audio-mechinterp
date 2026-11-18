"""CPU-only checks for the frozen locked-test FP32 runner."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import locked_test_fp32_v1 as locked
from fit_gate_v1 import patch_tensor, score_and_gradient
from test_fit_gate_v1 import TinyAudioAnswer


class FakeTokenizer:
    eos_token_id = 151645

    def encode(self, surface, add_special_tokens=False):
        reverse = {v: k for k, v in locked.ANSWER_SURFACES.items()}
        return locked.ANSWER_IDS[reverse[surface]][:-1]


def synthetic_results() -> tuple[list[dict], dict]:
    results = []
    for site in locked.SITES_TO_CHECK:
        for engine in locked.EXPECTED_VOICES:
            for category, words in locked.EXPECTED_WORDS.items():
                for phrasing in ("codeword", "term"):
                    for recipient, donor in ((words[0], words[1]), (words[1], words[0])):
                        case_id = f"{engine}/{category}/{phrasing}/{recipient}_to_{donor}/{site}"
                        results.append({
                            "case_id": case_id, "site": site, "engine": engine,
                            "category": category, "phrasing": phrasing,
                            "recipient": recipient, "donor": donor,
                            "pair_clean_eligible": True,
                            "controls_planned": phrasing == "codeword",
                            "conditions": [{
                                "alpha": alpha,
                                "effective_fp32_edit_norm": 1.0,
                                "measured_delta_score": 0.1 if alpha in locked.FIT_BASELINE_ALPHAS else 0.0,
                                "predicted_delta_score": 0.08 if alpha in locked.FIT_BASELINE_ALPHAS else 0.0,
                                "generated": {"answer": donor if alpha == 1 else recipient,
                                              "stopped_on_eos": True},
                            } for alpha in locked.ALPHAS],
                        })
    baseline = {
        "fit_numerical_floors": {site: 1e-4 for site in locked.SITES_TO_CHECK},
        "coefficients": {site: {"beta": 0.0} for site in locked.SITES_TO_CHECK},
    }
    return results, baseline


class LockedTestFP32Tests(unittest.TestCase):
    def test_frozen_fit_dev_and_test_inputs_are_present(self) -> None:
        fit, baseline, development = locked.verify_prerequisites()
        manifest = locked.verify_locked_manifest()
        self.assertEqual(fit["status"], "fit_only_gate_passed")
        self.assertEqual(development["status"], "development_gate_passed")
        self.assertEqual(len(manifest["pairs"]), 8)
        self.assertEqual(set(baseline["coefficients"]), set(locked.SITES_TO_CHECK))

    def test_title_case_complete_answer_token_validation(self) -> None:
        locked.verify_candidate_tokens(SimpleNamespace(tokenizer=FakeTokenizer()))
        self.assertEqual(locked.ANSWER_IDS["north"], [25221, 151645])
        self.assertEqual(locked.ANSWER_IDS["south"], [25018, 151645])
        self.assertEqual(locked.ANSWER_IDS["seven"], [59528, 151645])
        self.assertEqual(locked.ANSWER_IDS["eight"], [59085, 151645])
        bad = FakeTokenizer()
        bad.eos_token_id = 1
        with self.assertRaises(ValueError):
            locked.verify_candidate_tokens(SimpleNamespace(tokenizer=bad))

    def test_complete_answer_contrast_and_patch_orientation(self) -> None:
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
        with patch.object(locked, "ANSWER_IDS", {"north": [1, 3], "south": [2, 3]}):
            clean = locked.score_contrast(model, batch, model.audio, base, [1], "north", "south")
            changed = locked.score_contrast(model, batch, model.audio, patched, [1], "north", "south")
        self.assertAlmostEqual(clean["contrast"], donor_score - recipient_score, places=6)
        self.assertAlmostEqual(changed["contrast"] - clean["contrast"], predicted, delta=2e-5)

    def test_locked_gate_counts_all_and_eligible_denominators(self) -> None:
        results, baseline = synthetic_results()
        gate = locked.evaluate_locked_test_gate(results, baseline)
        self.assertTrue(gate["pass"])
        for site in locked.SITES_TO_CHECK:
            summary = gate["sites"][site]
            self.assertEqual(summary["all_planned_directions"], 16)
            self.assertEqual(summary["clean_eligible_directions"], 16)
            self.assertEqual(summary["controls_planned_directions"], 8)
            self.assertEqual(summary["determinate_clean_small_step_conditions"], 48)
            self.assertEqual(summary["alpha_one_complete_donor_flip_rate_all_planned"], 1.0)
            self.assertEqual(len(summary["category_phrasing_voice_breakdown"]), 8)
        one_ineligible = copy.deepcopy(results)
        one_ineligible[0]["pair_clean_eligible"] = False
        gate = locked.evaluate_locked_test_gate(one_ineligible, baseline)
        self.assertEqual(gate["sites"]["connector"]["all_planned_directions"], 16)
        self.assertEqual(gate["sites"]["connector"]["clean_eligible_directions"], 15)

    def test_locked_gate_fails_wrong_sign(self) -> None:
        results, baseline = synthetic_results()
        for case in results:
            if case["site"] == "encoder31":
                for condition in case["conditions"]:
                    if condition["alpha"] in locked.FIT_BASELINE_ALPHAS:
                        condition["predicted_delta_score"] = -0.08
        gate = locked.evaluate_locked_test_gate(results, baseline)
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["sites"]["encoder31"]["determinate_sign_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
