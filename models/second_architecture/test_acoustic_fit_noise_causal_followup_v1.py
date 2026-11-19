"""No-model tests for the post hoc FIT acoustic causal patch source."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import acoustic_fit_noise_causal_followup_v1 as causal  # noqa: E402


class PatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipient = np.zeros(causal.STATE_SHAPE, dtype=np.float32)
        self.donor = np.ones(causal.STATE_SHAPE, dtype=np.float32)

    def test_frozen_arm_matrix_and_region_parity(self) -> None:
        self.assertEqual(len(causal.ARMS), 8)
        for recipient_arm in causal.PCM_HASHES:
            for arm in causal.ARMS:
                patched, proof = causal.construct_patch(
                    self.recipient, self.donor,
                    recipient_arm=recipient_arm, arm=arm)
                self.assertEqual(patched.shape, causal.STATE_SHAPE)
                self.assertEqual(proof["arm"], arm)
                self.assertEqual(proof["realized_edit_sha256"],
                                 causal.array_sha(patched - self.recipient))
                self.assertEqual(proof["realized_edit_shape"],
                                 list(causal.STATE_SHAPE))
                self.assertEqual(proof["realized_edit_dtype"], "float32")
                outside = np.setdiff1d(np.arange(286), causal._region_indices(arm))
                np.testing.assert_array_equal(patched[:, outside, :],
                                              self.recipient[:, outside, :])
                if arm == "self_W":
                    np.testing.assert_array_equal(patched, self.recipient)
                if arm == "full_donor":
                    np.testing.assert_array_equal(patched, self.donor)
                if arm.startswith("R"):
                    self.assertLessEqual(
                        abs(proof["realized_l2"] - proof["target_l2"]),
                        1e-4 * proof["target_l2"])

    def test_bad_input_fails_closed(self) -> None:
        bad = self.recipient.copy()
        bad[0, 1, 1] = np.nan
        with self.assertRaises(ValueError):
            causal.construct_patch(bad, self.donor,
                                   recipient_arm="clean", arm="W_donor")
        with self.assertRaises(ValueError):
            causal.construct_patch(self.recipient, self.donor,
                                   recipient_arm="unfrozen", arm="W_donor")
        with self.assertRaises(ValueError):
            causal.construct_patch(self.recipient, self.recipient,
                                   recipient_arm="clean", arm="R1_W")

    def test_private_hook_exact_recipient_and_tuple(self) -> None:
        import torch

        class Projection(torch.nn.Module):
            def forward(self, state):
                return state, state

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wav2vec2 = SimpleNamespace(feature_projection=Projection())

            def forward(self, *, input_values, **_kwargs):
                state, _ = self.wav2vec2.feature_projection(input_values)
                return SimpleNamespace(logits=state[:, :, :32].contiguous())

        fake = Fake()
        patched, _ = causal.construct_patch(
            self.recipient, self.donor,
            recipient_arm="clean", arm="W_donor")
        logits, count = causal.patched_forward(
            fake, torch.from_numpy(self.recipient),
            recipient_state=self.recipient, patched_state=patched)
        self.assertEqual(count, 1)
        self.assertEqual(logits.shape, (1, 286, 32))
        self.assertTrue(np.all(logits[:, 105:179, :] == 1))
        self.assertTrue(np.all(logits[:, :105, :] == 0))
        self.assertEqual(len(fake.wav2vec2.feature_projection._forward_hooks), 0)
        with self.assertRaises(ValueError):
            causal.patched_forward(
                fake, torch.from_numpy(self.recipient),
                recipient_state=self.donor, patched_state=patched)

    def test_saved_selection_read_only(self) -> None:
        failed = causal._failed_v1_evidence()
        self.assertEqual(failed["parent_result_sha256"],
                         causal.FAILED_V1_RESULT_SHA)
        selected = causal._selected_rows()
        self.assertEqual(set(selected), {"clean", "word_noise"})
        self.assertEqual(selected["clean"]["output"]["transcript"],
                         causal.TRANSCRIPTS["clean"])
        self.assertEqual(selected["word_noise"]["output"]["transcript"],
                         causal.TRANSCRIPTS["word_noise"])

    def test_failed_v1_cannot_be_mutated_or_relabeled_complete(self) -> None:
        with patch.object(causal, "FAILED_V1_RESULT_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                causal._failed_v1_evidence()
        with patch.object(
                causal.parent_v1, "audit_saved_attempt",
                return_value={"status": "complete_evidence_pass",
                              "result_sha256": causal.FAILED_V1_RESULT_SHA}):
            with self.assertRaisesRegex(ValueError, "bound failure"):
                causal._failed_v1_evidence()


if __name__ == "__main__":
    unittest.main()
