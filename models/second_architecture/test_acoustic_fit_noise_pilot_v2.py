"""No-weight V2 capture-contract tests for the frozen FIT acoustic pilot."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models.second_architecture import acoustic_fit_noise_pilot_v2 as worker


class WavCaptureV2Tests(unittest.TestCase):
    def test_frozen_v1_stimuli_are_unchanged(self):
        clips = worker.stimuli()
        self.assertEqual(len(clips), 4)
        self.assertTrue(all(set(arms) == set(worker.ARMS) for arms in clips.values()))

    def test_block_capture_keeps_batch_axis(self):
        result = torch.zeros((1, 286, 768), dtype=torch.float32)
        for stage in ("post_block0", "post_block5", "post_block11"):
            self.assertIs(worker.select_wav_capture_v2(stage, "post_tensor", result), result)
            with self.assertRaisesRegex(ValueError, "batch-preserving Tensor"):
                worker.select_wav_capture_v2(stage, "post_tensor", (result,))

    def test_feature_projection_still_selects_tuple_first_tensor(self):
        hidden = torch.zeros((1, 286, 768), dtype=torch.float32)
        normalization = torch.ones_like(hidden)
        self.assertIs(worker.select_wav_capture_v2("feature_projection", "post_first",
                                                   (hidden, normalization)), hidden)
        with self.assertRaisesRegex(ValueError, "two-tensor tuple"):
            worker.select_wav_capture_v2("feature_projection", "post_first", hidden)

    def test_other_stages_reject_tuple(self):
        hidden = torch.zeros((1, 286, 32), dtype=torch.float32)
        self.assertIs(worker.select_wav_capture_v2("logits", "post", hidden), hidden)
        with self.assertRaisesRegex(ValueError, "must return a Tensor"):
            worker.select_wav_capture_v2("logits", "post", (hidden,))

    def test_v2_direct_worker_rejects_whisper_before_preflight(self):
        with self.assertRaisesRegex(ValueError, "only Wav2Vec2"):
            worker.run_model("whisper", Path("missing"), Path("missing"), Path("missing"))


if __name__ == "__main__":
    unittest.main()
