"""CPU checks for frozen candidate mass and fit-only selection."""

import math
import unittest

import torch

from fit_pool_capability_v1 import load_fit_cases
from fit_pool_single_token_score_v1 import first_step_metrics, selected_cases


class SingleTokenScoreTests(unittest.TestCase):
    def test_full_vocab_log_mass_and_tie(self):
        logits = torch.tensor([0.0, 2.0, 2.0, 1.0, -1.0])
        m = first_step_metrics(logits, [1, 2, 2], [3])
        self.assertEqual(m["expected_alias_token_ids"], [1, 2])
        self.assertEqual(m["top_vocab_token_id"], 1)
        self.assertAlmostEqual(m["correct_minus_foil_log_odds"],
                               math.log(2) + 1, places=6)
        self.assertAlmostEqual(m["expected_mass"] + m["foil_mass"] +
                               float(torch.softmax(logits, 0)[[0, 4]].sum()), 1.0, places=6)
        self.assertTrue(m["correct_mass_gt_foil"])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            first_step_metrics(logits, [1], [1, 3])

    def test_only_preselected_fit_contexts(self):
        spec, _manifest, all_cases = load_fit_cases()
        cases = selected_cases(spec, all_cases)
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({case["pair_id"] for case in cases}), 12)
        self.assertEqual({case["template"] for case in cases}, {"fit_chosen"})
        self.assertEqual({case["engine"] for case in cases}, {"piper", "kokoro"})
        self.assertEqual({case["category"] for case in cases},
                         {"animal", "color", "object", "nature"})
        self.assertEqual(len({case["audio_sha256"] for case in cases}), 48)
        self.assertTrue(all(case["foil_word"] != case["word"] for case in cases))


if __name__ == "__main__":
    unittest.main()
