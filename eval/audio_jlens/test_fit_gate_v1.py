"""CPU checks for complete-answer teacher forcing and audio-hook VJP math."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fit_gate_v1 import (
    extension_with_candidate, patch_tensor, score_and_gradient,
    score_from_logits, score_with_patch, record_mps_memory,
)


class TinyAudioAnswer(torch.nn.Module):
    """Two audio rows control first-answer and EOS probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.audio = torch.nn.Identity()
        self.register_buffer("audio_values", torch.tensor([[0.4, -0.2], [-0.3, 0.7]]))

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                return_dict=True, **_unused):
        state = self.audio(self.audio_values)
        length = input_ids.shape[1]
        logits = torch.zeros((1, length, 5), dtype=state.dtype)
        # Prompt length is two. IDs 1/2 are answer words; ID 3 is EOS.
        logits[0, 1, 1] = state[0, 0]
        logits[0, 1, 2] = state[1, 0]
        if length > 2:
            logits[0, 2, 3] = state[0, 1] + state[1, 1] + 0.1 * input_ids[0, 2].float()
        return SimpleNamespace(logits=logits)


class FitGateToyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyAudioAnswer()
        self.batch = {
            "input_ids": torch.tensor([[4, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def test_teacher_forcing_scores_answer_and_eos_at_correct_positions(self) -> None:
        candidate = [2, 3]
        extended, prompt_len = extension_with_candidate(self.batch, candidate)
        self.assertEqual(prompt_len, 2)
        self.assertEqual(extended["input_ids"].tolist(), [[4, 4, 2]])
        self.assertEqual(extended["attention_mask"].tolist(), [[1, 1, 1]])
        logits = self.model(**extended).logits
        expected = (logits[0, 1].log_softmax(-1)[2]
                    + logits[0, 2].log_softmax(-1)[3])
        actual = score_from_logits(logits, prompt_len, candidate)
        self.assertAlmostEqual(float(actual), float(expected), places=6)
        modified = logits.clone()
        modified[0, 2, 3] += 1.0
        self.assertGreater(float(score_from_logits(modified, prompt_len, candidate)), float(actual))

    def test_audio_row_gradient_predicts_small_patch_with_correct_orientation(self) -> None:
        candidate = [2, 3]
        base_score, base, grad = score_and_gradient(
            self.model, self.batch, candidate, self.model.audio
        )
        self.assertEqual(tuple(base.shape), (2, 2))
        self.assertEqual(tuple(grad.shape), (2, 2))
        self.assertEqual(len(self.model.audio._forward_hooks), 0)
        donor = base.clone()
        donor[0] += torch.tensor([0.001, -0.002])
        replacement, effective = patch_tensor(base, donor, [0], 1.0)
        predicted = float((grad[0] * effective[0]).sum())
        patched = score_with_patch(
            self.model, self.batch, candidate, self.model.audio,
            replacement, [0],
        )
        measured = patched - base_score
        self.assertAlmostEqual(predicted, measured, delta=2e-5)
        self.assertEqual(len(self.model.audio._forward_hooks), 0)
        zero, delta = patch_tensor(base, donor, [0], 0.0)
        self.assertTrue(torch.equal(zero, base))
        self.assertEqual(float(delta.norm()), 0.0)

    def test_candidate_specific_gradients_form_contrast_gradient(self) -> None:
        blue_score, blue_state, blue_grad = score_and_gradient(
            self.model, self.batch, [2, 3], self.model.audio
        )
        red_score, red_state, red_grad = score_and_gradient(
            self.model, self.batch, [1, 3], self.model.audio
        )
        self.assertTrue(torch.equal(blue_state, red_state))
        contrast_grad = blue_grad - red_grad
        donor = blue_state.clone()
        donor[1, 0] += 0.001
        replacement, effective = patch_tensor(blue_state, donor, [1], 1.0)
        predicted = float((contrast_grad[1] * effective[0]).sum())
        blue_patched = score_with_patch(
            self.model, self.batch, [2, 3], self.model.audio, replacement, [1]
        )
        red_patched = score_with_patch(
            self.model, self.batch, [1, 3], self.model.audio, replacement, [1]
        )
        measured = (blue_patched - red_patched) - (blue_score - red_score)
        self.assertAlmostEqual(predicted, measured, delta=2e-5)

    def test_memory_guard_persists_measurement_before_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            with patch("torch.mps.current_allocated_memory", return_value=2 * 1024**3), \
                 patch("torch.mps.driver_allocated_memory", return_value=31 * 1024**3):
                with self.assertRaises(MemoryError):
                    record_mps_memory(path, "cpu_toy")
            saved = json.loads(path.read_text().strip())
            self.assertEqual(saved["phase"], "cpu_toy")
            self.assertTrue(saved["over_limit"])
            self.assertEqual(saved["driver_allocated_bytes"], 31 * 1024**3)


if __name__ == "__main__":
    unittest.main()
