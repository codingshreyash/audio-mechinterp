"""CPU-only checks for feature-factorial construction and full-reply scoring."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voxtral_preprocessing_decomposition import make_conditions, teacher_forced_candidate_score


class PreprocessingDecompositionTests(unittest.TestCase):
    def test_feature_factorial_and_padding_swaps(self):
        red = torch.zeros((1, 128, 3000), dtype=torch.float32)
        blue = torch.ones_like(red)
        word = list(range(112, 168))
        conditions = make_conditions({"red": red, "blue": blue}, word, 311)
        self.assertEqual(len(conditions), 6)
        self.assertTrue(torch.equal(conditions["word_red_outside_red"]["features"], red))
        self.assertTrue(torch.equal(conditions["word_blue_outside_blue"]["features"], blue))
        self.assertEqual(float(conditions["word_red_outside_blue"]["features"][0, 0, 112]), 0.0)
        self.assertEqual(float(conditions["word_red_outside_blue"]["features"][0, 0, 111]), 1.0)
        self.assertEqual(float(conditions["word_blue_outside_red"]["features"][0, 0, 167]), 1.0)
        self.assertEqual(float(conditions["word_blue_outside_red"]["features"][0, 0, 168]), 0.0)
        red_pad_blue = conditions["red_with_blue_padding"]["features"]
        self.assertEqual(float(red_pad_blue[0, 0, 310]), 0.0)
        self.assertEqual(float(red_pad_blue[0, 0, 311]), 1.0)
        blue_pad_red = conditions["blue_with_red_padding"]["features"]
        self.assertEqual(float(blue_pad_red[0, 0, 310]), 1.0)
        self.assertEqual(float(blue_pad_red[0, 0, 311]), 0.0)

    def test_teacher_forcing_scores_candidate_then_eos(self):
        class FakeModel:
            def __call__(self, **kwargs):
                assert kwargs["input_ids"].tolist() == [[11, 12, 3]]
                assert kwargs["attention_mask"].tolist() == [[1, 1, 1]]
                assert kwargs["input_features"].shape == (1, 128, 3000)
                logits = torch.zeros((1, 3, 5), dtype=torch.float32)
                logits[0, 1, 3] = 2.0  # candidate after prompt
                logits[0, 2, 2] = 4.0  # EOS after candidate
                return SimpleNamespace(logits=logits)

        batch = {
            "input_ids": torch.tensor([[11, 12]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "input_features": torch.zeros((1, 128, 3000)),
        }
        result = teacher_forced_candidate_score(FakeModel(), batch, [3], eos=2, device="cpu")
        expected = (2.0 - math.log(math.exp(2.0) + 4)) + (4.0 - math.log(math.exp(4.0) + 4))
        self.assertAlmostEqual(result["complete_reply_logprob"], expected, places=5)
        self.assertEqual(len(result["token_logprobs_including_eos"]), 2)


if __name__ == "__main__":
    unittest.main()
