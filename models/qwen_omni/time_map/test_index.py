"""CPU checks against pinned Transformers' audio row-indexing functions."""

from __future__ import annotations

import unittest

import torch
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    get_audio_cu_seqlens,
    get_pool_indices,
)

from index import build_row_index


class RowIndexTests(unittest.TestCase):
    def test_index_matches_pinned_source_over_chunk_boundaries(self):
        for length in (2, 3, 4, 199, 200, 201, 399, 400, 401, 550, 751):
            with self.subTest(length=length):
                projected_count = (((length - 1) // 2 + 1) - 2) // 2 + 1
                placeholders = list(range(17, 17 + projected_count))
                row_map = build_row_index(
                    mel_frames=length, audio_token_positions=placeholders,
                    n_window=100, hop_length=160, sample_rate=16000,
                )
                chunks = row_map["chunks"]
                chunk_lengths = torch.tensor(
                    [chunk["mel_frame_end_exclusive"] - chunk["mel_frame_start"] for chunk in chunks]
                )
                source_cu = get_audio_cu_seqlens(chunk_lengths).tolist()
                own_cu = [0] + [chunk["encoder_row_end_exclusive"] for chunk in chunks]
                self.assertEqual(source_cu, own_cu)
                source_pool = get_pool_indices(torch.tensor([length])).tolist()
                own_pool = [row["encoder_row_pair"][0] for row in row_map["projected_rows"]]
                self.assertEqual(source_pool, own_pool)
                self.assertEqual(len(row_map["encoder_rows"]), (length - 1) // 2 + 1)
                self.assertEqual(len(row_map["projected_rows"]), len(placeholders))
                self.assertEqual(
                    [row["audio_placeholder_token_position"] for row in row_map["projected_rows"]],
                    placeholders,
                )

    def test_rejects_wrong_placeholder_count(self):
        with self.assertRaisesRegex(ValueError, "processor audio placeholders"):
            build_row_index(mel_frames=550, audio_token_positions=[1], n_window=100, hop_length=160, sample_rate=16000)

    def test_index_contains_no_word_alignment_claim(self):
        result = build_row_index(
            mel_frames=550, audio_token_positions=list(range(137)),
            n_window=100, hop_length=160, sample_rate=16000,
        )
        self.assertEqual([c["mel_frame_end_exclusive"] for c in result["chunks"]], [200, 400, 550])
        self.assertEqual([c["encoder_row_end_exclusive"] for c in result["chunks"]], [100, 200, 275])
        self.assertEqual(result["projected_rows"][0]["encoder_row_pair"], [0, 1])
        self.assertEqual(result["projected_rows"][-1]["encoder_row_pair"], [272, 273])
        self.assertIn("not word timestamps", result["timing_warning"])


if __name__ == "__main__":
    unittest.main()
