"""CPU-only fake-model and provenance tests for the Whisper FIT runner."""

from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from models.second_architecture import whisper_object02_fit_causal_v1 as assay


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.layer_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(encoder=FakeEncoder())


class WhisperObject02FitCausalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recipient = torch.zeros((1, 1500, 384), dtype=torch.float32)
        self.donor = self.recipient.clone()
        self.donor[:, 90:139, 0] = 1.0

    def test_frozen_manifest_and_pcm_pair(self) -> None:
        cases, windows = assay.check_case_manifest()
        self.assertEqual({case.key for case in cases}, {"piper_plate", "piper_cup", "kokoro_plate", "kokoro_cup"})
        self.assertEqual(windows["piper"]["word_window"], [90, 139])
        self.assertEqual(windows["kokoro"]["prefix_window"], [10, 84])

    def test_donor_patch_only_changes_selected_rows(self) -> None:
        edited = assay.patch_rows(self.recipient, self.donor, (90, 139), "donor")
        self.assertTrue(torch.equal(edited[:, 90:139], self.donor[:, 90:139]))
        self.assertTrue(torch.equal(edited[:, :90], self.recipient[:, :90]))
        self.assertTrue(torch.equal(edited[:, 139:], self.recipient[:, 139:]))

    def test_random_edit_reproducible_and_equal_target_norm(self) -> None:
        audit1, audit2 = {}, {}
        a = assay.patch_rows(self.recipient, self.donor, (90, 139), "random", assay.derive_arm_seed("piper", "plate", "cup", 20261118), audit1)
        b = assay.patch_rows(self.recipient, self.donor, (90, 139), "random", assay.derive_arm_seed("piper", "plate", "cup", 20261118), audit2)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(audit1, audit2)
        self.assertEqual(audit1["random_edit_sha256"], hashlib.sha256(a[:, 90:139, :].numpy().tobytes()).hexdigest())
        self.assertAlmostEqual(audit1["random_edit_norm"], audit1["random_target_norm"], places=5)
        self.assertTrue(torch.equal(a[:, :90], self.recipient[:, :90]))

    def test_all_random_arms_have_distinct_seeds_and_edit_bytes(self) -> None:
        seeds = []
        edits = []
        for engine in ("piper", "kokoro"):
            for recipient_word, donor_word in (("plate", "cup"), ("cup", "plate")):
                for base_seed in assay.BASE_SEEDS:
                    seed = assay.derive_arm_seed(engine, recipient_word, donor_word, base_seed)
                    seeds.append(seed)
                    audit = {}
                    # Equal-sized states make this a direct test of the per-arm
                    # random draw, independent of model output or target norms.
                    assay.patch_rows(self.recipient, self.donor, (90, 139), "random", seed, audit)
                    edits.append(audit["random_edit_sha256"])
        self.assertEqual(len(set(seeds)), 8)
        self.assertEqual(len(set(edits)), 8)
        self.assertNotEqual(
            assay.derive_arm_seed("piper", "plate", "cup", 20261118),
            assay.derive_arm_seed("piper", "cup", "plate", 20261118),
        )

    def test_random_target_zero_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero or nonfinite"):
            assay.patch_rows(self.recipient, self.recipient, (90, 139), "random", assay.derive_arm_seed("piper", "plate", "cup", 20261118))

    def test_unfrozen_seed_and_invalid_shape_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unfrozen"):
            assay.patch_rows(self.recipient, self.donor, (90, 139), "random", 7)
        with self.assertRaisesRegex(ValueError, "encoder state"):
            assay.patch_rows(self.recipient[:, :10], self.donor, (0, 10), "donor")

    def test_hooks_at_all_three_sites_and_single_call(self) -> None:
        for site in assay.SITES:
            model = FakeModel()
            count = [0]
            handle = assay.install_patch_hook(model, site, self.donor, (90, 139), "donor", None, count)
            try:
                result = model.model.encoder(self.recipient)
                self.assertTrue(torch.equal(result[:, 90:139], self.donor[:, 90:139]))
                self.assertEqual(count, [1])
                with self.assertRaisesRegex(ValueError, "more than once"):
                    model.model.encoder(self.recipient)
            finally:
                handle.remove()

    def test_full_donor_and_self_controls(self) -> None:
        self.assertTrue(torch.equal(assay.patch_rows(self.recipient, self.donor, (0, 1500), "full_donor"), self.donor))
        self.assertTrue(torch.equal(assay.patch_rows(self.recipient, self.recipient, (90, 139), "self"), self.recipient))
        with self.assertRaisesRegex(ValueError, "all rows"):
            assay.patch_rows(self.recipient, self.donor, (90, 139), "full_donor")

    def test_complete_transcript_target_replacement(self) -> None:
        original = "The word I chose is plate. What word did I choose?"
        good = assay.transcript_outcome("The word I chose is cup. What word did I choose?", "plate", "cup", original)
        bad = assay.transcript_outcome("The word I chose is cup. I changed the rest.", "plate", "cup", original)
        self.assertTrue(good["target_replacement"])
        self.assertFalse(bad["target_replacement"])
        self.assertTrue(bad["donor_once_recipient_absent"])
        self.assertGreater(bad["wer_to_recipient"], good["wer_to_recipient"])

    def test_zero_hook_count_keeps_completed_decode_available(self) -> None:
        class Model:
            generation_config = SimpleNamespace(eos_token_id=2)

            def generate(self, **_kwargs):
                return SimpleNamespace(sequences=torch.tensor([[1, 2]]), scores=[torch.tensor([[0.0, 1.0, 2.0]])])

        class Processor:
            tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1])

            def batch_decode(self, *_args, **_kwargs):
                return ["plate"]

        decoded = assay.run_model(Model(), Processor(), {}, hook_count=[0])
        self.assertEqual(decoded["token_ids"], [1, 2])
        self.assertEqual(decoded["hook_count"], 0)
        self.assertTrue(decoded["eos_seen"])


if __name__ == "__main__":
    unittest.main()
