"""CPU-only checks of V2 lexical coverage and complete-sequence aggregation."""

import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import fit_pool_scoring_v2 as v2
from fit_pool_capability_v1 import load_fit_cases
from fit_pool_single_token_score_v1 import selected_cases


HERE = Path(__file__).resolve().parent


class V2ScoringTests(unittest.TestCase):
    def test_artifact_and_fit_only_coverage(self):
        spec, _manifest, all_cases = load_fit_cases()
        cases = selected_cases(spec, all_cases)
        policy = v2.load_scoring(spec, cases)
        self.assertEqual(len(policy["words"]), 96)
        self.assertEqual(len(policy["pairs"]), 48)
        capability = json.loads((HERE / "fit_pool_capability_v1_results.json").read_text())
        rows = capability["cases"]
        correct = [r for r in rows if r["classification"]["strict_word_plus_eos"]]
        self.assertEqual(len(correct), 180)
        self.assertEqual(sum(r["generated_token_ids"][0] in
                             policy["words"][r["word"]]["first_token_ids"] for r in correct), 180)
        self.assertEqual(sum(r["generated_token_ids"] in
                             policy["words"][r["word"]]["complete_answer_sequences_plus_eos"] for r in correct), 180)

    def test_sequence_logsumexp_has_no_length_normalization(self):
        values = {(4, 5): -2.0, (6, 7, 8): -3.0}
        with patch.object(v2, "scored_forward", side_effect=lambda _m, _b, ids: torch.tensor(values[tuple(ids)])):
            score, parts = v2.complete_class_score(None, {}, [[4, 5], [6, 7, 8]])
        self.assertEqual(parts, [-2.0, -3.0])
        self.assertAlmostEqual(score, math.log(math.exp(-2) + math.exp(-3)))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            v2.complete_class_score(None, {}, [[4, 5], [4, 5]])


if __name__ == "__main__":
    unittest.main()
