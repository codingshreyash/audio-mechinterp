"""No-model tests for exact CTC sequence scores and reference encoding."""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from .analyze import ctc_logp, target_ids, tokens


def enumerate_ctc(logits: np.ndarray, target: list[int], blank: int = 0) -> float:
    """Independent tiny-alphabet exhaustive path sum."""
    x = logits.astype(np.float64)
    p = np.exp(x - x.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    total = 0.0
    for path in itertools.product(range(x.shape[1]), repeat=x.shape[0]):
        collapsed = [token for i, token in enumerate(path)
                     if i == 0 or token != path[i-1]]
        emitted = [token for token in collapsed if token != blank]
        if emitted == target:
            total += math.prod(p[t, token] for t, token in enumerate(path))
    return math.log(total)


class AnalyzeTests(unittest.TestCase):
    def test_ctc_matches_exhaustive_repeated_letter_and_blank_paths(self) -> None:
        logits = np.array([[0.4, 0.2, -0.3],[-0.7, 0.1, 0.9],
                           [0.6, -0.5, 0.2],[0.3, 1.1, -0.2]], dtype=np.float32)
        for target in ([1], [1,1], [1,2], [2,1]):
            with self.subTest(target=target):
                self.assertAlmostEqual(ctc_logp(logits, target),
                                       enumerate_ctc(logits, target), places=11)

    def test_complete_reference_encoding_uses_vocabulary_word_separator(self) -> None:
        vocab = {"|":4, "A":7, "B":24, "C":19}
        ref = tokens("A, b C!")
        text, ids = target_ids(ref, vocab)
        self.assertEqual(text, "A B C")
        self.assertEqual(ids, [7,4,24,4,19])


if __name__ == "__main__":
    unittest.main()
