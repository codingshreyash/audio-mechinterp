"""No-weight tests of causal patch geometry, hooks, and saved-data parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import wav2vec2_fit_causal_patch_audit_v1 as auditor
import wav2vec2_fit_causal_patch_v1 as runner


class TupleLayer(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value + 1,)


class Projection(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (value + 2, value + 3)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wav2vec2 = torch.nn.Module()
        self.wav2vec2.feature_projection = Projection()
        self.wav2vec2.encoder = torch.nn.Module()
        self.wav2vec2.encoder.layers = torch.nn.ModuleList([TupleLayer() for _ in range(12)])

    def forward(self, *, input_values: torch.Tensor, attention_mask: None,
                output_hidden_states: bool, return_dict: bool) -> SimpleNamespace:
        initial = torch.zeros((1, 4, 768), dtype=torch.float32)
        hidden, _unused = self.wav2vec2.feature_projection(initial)
        for layer in self.wav2vec2.encoder.layers:
            hidden = layer(hidden)[0]
        logits = hidden[:, :, :32].contiguous()
        return SimpleNamespace(logits=logits)


class CausalNoWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipient = np.zeros((1, 8, 768), dtype=np.float32)
        self.donor = np.ones((1, 8, 768), dtype=np.float32)
        self.win = {"W": (3, 5), "P": (1, 3), "Q": (5, 7)}

    def test_frozen_windows(self) -> None:
        self.assertEqual(runner.windows((30527, 42788), 249),
                         {"W": (95, 134), "P": (56, 95), "Q": (134, 173)})
        with self.assertRaisesRegex(ValueError, "outside"):
            runner.windows((0, 320), 2)

    def test_patch_support_and_replay(self) -> None:
        for arm, expected in (("W", (3, 5)), ("P", (1, 3)), ("Q", (5, 7))):
            patched, norm = runner.candidate_state(
                self.recipient, self.donor, arm=arm, window=self.win, seed=None)
            support = np.flatnonzero(np.any(patched[0] != 0, axis=-1))
            self.assertEqual(support.tolist(), list(range(*expected)))
            self.assertGreater(norm, 0)
            self.assertTrue(np.array_equal(
                patched, auditor.independent_patch(self.recipient, self.donor,
                                                    arm, self.win, None)))
        for arm, seed in (("R1", 971103), ("R2", 971104)):
            patched, _ = runner.candidate_state(
                self.recipient, self.donor, arm=arm, window=self.win, seed=seed)
            self.assertTrue(np.array_equal(
                patched, auditor.independent_patch(self.recipient, self.donor,
                                                    arm, self.win, seed)))
            self.assertTrue(np.array_equal(patched[:, :3], self.recipient[:, :3]))
            self.assertTrue(np.array_equal(patched[:, 5:], self.recipient[:, 5:]))
        self.assertFalse(np.array_equal(
            runner.candidate_state(self.recipient, self.donor,
                                   arm="R1", window=self.win, seed=971103)[0],
            runner.candidate_state(self.recipient, self.donor,
                                   arm="R2", window=self.win, seed=971104)[0]))

    def test_self_full_and_zero_random_rejection(self) -> None:
        self.assertTrue(np.array_equal(
            runner.candidate_state(self.recipient, self.donor,
                                   arm="self_W", window=self.win, seed=None)[0],
            self.recipient))
        self.assertTrue(np.array_equal(
            runner.candidate_state(self.recipient, self.donor,
                                   arm="full_donor", window=self.win, seed=None)[0],
            self.donor))
        with self.assertRaisesRegex(ValueError, "zero"):
            runner.candidate_state(self.recipient, self.recipient,
                                   arm="R1", window=self.win, seed=971103)

    def test_tuple_preserving_feature_and_final_layer_hooks(self) -> None:
        model = FakeModel().eval()
        input_tensor = torch.zeros((1, 4), dtype=torch.float32)
        projected = np.full((1, 4, 768), 2, dtype=np.float32)
        final = np.full((1, 4, 768), 14, dtype=np.float32)
        for site, state, replacement in (
            ("feature_projection", projected, np.full_like(projected, 4)),
            ("post_block11", final, np.full_like(final, 19)),
        ):
            logits, count = runner.patched_forward(
                model, input_tensor, site=site, clean_state=state,
                patched_state=replacement)
            self.assertEqual(count, 1)
            expected = 16 if site == "feature_projection" else 19
            self.assertTrue(np.array_equal(logits, np.full((1, 4, 32), expected,
                                                             dtype=np.float32)))
        self.assertEqual(len(model.wav2vec2.feature_projection._forward_hooks), 0)
        self.assertEqual(len(model.wav2vec2.encoder.layers[11]._forward_hooks), 0)

    def test_target_only_requires_complete_baseline_transfer(self) -> None:
        recipient = {"word": "plate", "normalized_words": ["choose", "plate", "today"]}
        donor = {"word": "cup", "normalized_words": ["choose", "cup", "today"]}
        self.assertTrue(runner.target_only_transfer(recipient, donor,
                                                    ["choose", "cup", "today"]))
        self.assertFalse(runner.target_only_transfer(recipient, donor,
                                                     ["choose", "cub", "today"]))
        self.assertFalse(runner.target_only_transfer(recipient, donor,
                                                     ["choose", "cup", "tomorrow"]))
        self.assertFalse(auditor.independent_target_only(recipient, donor,
                                                          ["choose", "cub", "today"]))
        with self.assertRaisesRegex(ValueError, "target-word outcome"):
            auditor.verify_target_receipt(
                {"primary_eligible": True, "target_only_transfer": True},
                recipient, donor, ["choose", "cub", "today"])
        self.assertTrue(auditor.verify_target_receipt(
            {"primary_eligible": True, "target_only_transfer": False},
            recipient, donor, ["choose", "cub", "today"]))

    def test_saved_data_reader_rejects_partial_and_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            path.write_bytes(b'{"a":1}')
            with self.assertRaisesRegex(ValueError, "truncated"):
                auditor.read_rows(path)
            path.write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                auditor.read_rows(path)


if __name__ == "__main__":
    unittest.main()
