"""Small synthetic rank and paired-metric checks, no model or tokenizer load."""

from __future__ import annotations

import unittest

from scoring import aggregate_scores, paired_audio_typed, rank_candidates, score_item


ITEM = {
    "id": "association:toy", "category": "association", "split": "test",
    "scoring_status": "scorable",
    "scorable_labels": [{"label": "bridge", "token_surfaces": [
        {"surface": " bridge", "token_id": 2},
        {"surface": "bridge", "token_id": 3},
    ]}],
}


class ScoreTest(unittest.TestCase):
    def test_candidate_best_and_ties(self) -> None:
        self.assertEqual(rank_candidates([4, 5, 2, 3, 6], [2, 3]), 4)
        self.assertEqual(rank_candidates([3, 5, 2, 3, 6], [2, 3]), 3)
        self.assertEqual(rank_candidates([4, 5, 2, 3, 6], [2, 3], allowed_ids=[0, 2, 3]), 2)

    def test_min_layer_and_paired_item(self) -> None:
        typed = score_item(ITEM, {0: [4, 5, 2, 3, 6], 1: [0, 1, 5, 4, 2]},
                           modality="typed", method="jlens", head_vocab_size=5,
                           allowed_ids=[0, 2, 3], expected_layers=(0, 1))
        audio = score_item(ITEM, {0: [4, 5, 2, 3, 6], 1: [4, 5, 2, 3, 6]},
                           modality="audio", method="jlens", head_vocab_size=5,
                           audio_source="piper",
                           allowed_ids=[0, 2, 3], expected_layers=(0, 1))
        self.assertEqual(typed["labels"]["bridge"]["full_min_rank"], 1)
        self.assertEqual(audio["labels"]["bridge"]["full_min_rank"], 4)
        self.assertEqual(aggregate_scores([typed, audio])[("typed", None, "jlens", "association", "test")]["pass@1"], 1)
        paired = paired_audio_typed([audio], [typed], k=1)
        self.assertEqual(paired[("association", "piper")]["mean_audio_minus_typed"], -1)


if __name__ == "__main__":
    unittest.main()
