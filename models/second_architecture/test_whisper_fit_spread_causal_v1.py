"""CPU-only provenance, region, hook and fake-generation tests for spread assay."""

from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from models.second_architecture import whisper_fit_spread_causal_v1 as assay


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


class WhisperSpreadRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recipient = torch.zeros((1, 1500, 384), dtype=torch.float32)
        self.donor = self.recipient.clone()
        self.donor[:, :90, 0] = 1.0
        self.donor[:, 90:161, 0] = 2.0
        self.donor[:, 161:272, 0] = 3.0
        self.donor[:, 272:, 0] = 4.0
        cases, self.regions = assay.check_case_manifest()
        self.cases = {case.key: case for case in cases}
        self.case = self.cases["color_05_piper_silver"]

    def test_all_eight_frozen_inputs_and_regions(self) -> None:
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(self.regions["color_05"]["piper"], {
            "slot_samples": [30527, 49847], "W": [90, 161], "P": [10, 81],
            "Q": [171, 242], "N": 272, "C": [[0, 90], [161, 272]],
        })
        self.assertEqual(self.regions["object_02"]["kokoro"]["Q"], [185, 259])
        self.assertEqual(self.cases["object_02_kokoro_plate"].active_rows, 279)

    def test_region_masks_are_fixed_and_disjoint_where_required(self) -> None:
        self.assertEqual(assay.region_spans(self.case, "word"), ((90, 161),))
        self.assertEqual(assay.region_spans(self.case, "prefix"), ((10, 81),))
        self.assertEqual(assay.region_spans(self.case, "postword"), ((171, 242),))
        self.assertEqual(assay.region_spans(self.case, "complement"), ((0, 90), (161, 272)))
        self.assertEqual(assay.region_spans(self.case, "random"), assay.region_spans(self.case, "complement"))
        self.assertEqual(assay.region_spans(self.case, "full_donor"), ((0, 1500),))

    def test_complement_patch_preserves_word_and_padding(self) -> None:
        spans = assay.region_spans(self.case, "complement")
        edited = assay.patch_rows(self.recipient, self.donor, spans, "donor")
        self.assertTrue(torch.equal(edited[:, :90], self.donor[:, :90]))
        self.assertTrue(torch.equal(edited[:, 161:272], self.donor[:, 161:272]))
        self.assertTrue(torch.equal(edited[:, 90:161], self.recipient[:, 90:161]))
        self.assertTrue(torch.equal(edited[:, 272:], self.recipient[:, 272:]))

    def test_word_patch_preserves_complement(self) -> None:
        edited = assay.patch_rows(self.recipient, self.donor, assay.region_spans(self.case, "word"), "donor")
        self.assertTrue(torch.equal(edited[:, 90:161], self.donor[:, 90:161]))
        self.assertTrue(torch.equal(edited[:, :90], self.recipient[:, :90]))
        self.assertTrue(torch.equal(edited[:, 161:], self.recipient[:, 161:]))

    def test_random_complement_edit_is_deterministic_and_norm_matched(self) -> None:
        seed = assay.derive_arm_seed("color_05", "piper", "silver", "teal", "post_block0", 20261118)
        spans = assay.region_spans(self.case, "complement")
        audit_a, audit_b = {}, {}
        first = assay.patch_rows(self.recipient, self.donor, spans, "random", seed, audit_a)
        second = assay.patch_rows(self.recipient, self.donor, spans, "random", seed, audit_b)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(audit_a, audit_b)
        self.assertAlmostEqual(audit_a["random_edit_norm"], audit_a["random_target_norm"], places=4)
        self.assertTrue(torch.equal(first[:, 90:161], self.recipient[:, 90:161]))
        self.assertTrue(torch.equal(first[:, 272:], self.recipient[:, 272:]))
        local_edit = torch.cat((first[:, :90], first[:, 161:272]), dim=1)
        self.assertEqual(audit_a["random_edit_sha256"], hashlib.sha256(local_edit.numpy().tobytes()).hexdigest())

    def test_all_32_random_arms_have_unique_seeds_and_edit_bytes(self) -> None:
        seeds, hashes = set(), set()
        for pair_id, spec in assay.PAIR_SPECS.items():
            left, right = spec["words"]
            for engine in assay.ENGINES:
                case = self.cases[f"{pair_id}_{engine}_{left}"]
                donor = self.recipient.clone()
                for start, end in assay.region_spans(case, "complement"):
                    donor[:, start:end, 0] = 1.0
                for recipient_word, donor_word in ((left, right), (right, left)):
                    for site in assay.SITES:
                        for base in assay.BASE_SEEDS:
                            seed = assay.derive_arm_seed(pair_id, engine, recipient_word, donor_word, site, base)
                            seeds.add(seed)
                            audit = {}
                            assay.patch_rows(self.recipient, donor, assay.region_spans(case, "complement"), "random", seed, audit)
                            hashes.add(audit["random_edit_sha256"])
        self.assertEqual(len(seeds), 32)
        self.assertEqual(len(hashes), 32)

    def test_invalid_region_seed_and_zero_target_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlapping"):
            assay.patch_rows(self.recipient, self.donor, ((0, 100), (90, 120)), "donor")
        with self.assertRaisesRegex(ValueError, "unfrozen"):
            assay.patch_rows(self.recipient, self.donor, ((0, 90), (161, 272)), "random", 7)
        seed = assay.derive_arm_seed("color_05", "piper", "silver", "teal", "post_norm", 20261118)
        with self.assertRaisesRegex(ValueError, "zero or nonfinite"):
            assay.patch_rows(self.recipient, self.recipient, ((0, 90), (161, 272)), "random", seed)

    def test_two_post_block_hooks_and_single_call(self) -> None:
        for site in assay.SITES:
            model = FakeModel()
            count = [0]
            handle = assay.install_patch_hook(model, site, self.donor, ((90, 161),), "donor", None, count)
            try:
                output = model.model.encoder(self.recipient)
                self.assertTrue(torch.equal(output[:, 90:161], self.donor[:, 90:161]))
                self.assertEqual(count, [1])
                with self.assertRaisesRegex(ValueError, "more than once"):
                    model.model.encoder(self.recipient)
            finally:
                handle.remove()

    def test_full_donor_self_and_matrix_count(self) -> None:
        self.assertTrue(torch.equal(assay.patch_rows(self.recipient, self.donor, ((0, 1500),), "full_donor"), self.donor))
        self.assertTrue(torch.equal(assay.patch_rows(self.recipient, self.recipient, ((90, 161),), "self"), self.recipient))
        self.assertEqual(len(assay.expected_arm_matrix()), 120)
        self.assertEqual(sum(key[5] == "random" for key in assay.expected_arm_matrix()), 32)
        with self.assertRaisesRegex(ValueError, "all rows"):
            assay.patch_rows(self.recipient, self.donor, ((90, 161),), "full_donor")

    def test_complete_transcript_outcome_requires_other_words_unchanged(self) -> None:
        original = "The word I chose is silver. What word did I choose?"
        good = assay.transcript_outcome("The word I chose is teal. What word did I choose?", "silver", "teal", original)
        bad = assay.transcript_outcome("The word I chose is teal. I changed the rest.", "silver", "teal", original)
        self.assertTrue(good["target_replacement"])
        self.assertFalse(bad["target_replacement"])
        self.assertTrue(bad["donor_once_recipient_absent"])

    def test_fake_generation_zero_hook_retained_and_bad_eos_censored(self) -> None:
        class Model:
            generation_config = SimpleNamespace(eos_token_id=2)

            def __init__(self, ids: list[int]) -> None:
                self.ids = ids

            def generate(self, **_kwargs):
                return SimpleNamespace(sequences=torch.tensor([self.ids]), scores=[torch.tensor([[0.0, 1.0, 2.0, 3.0]])])

        class Processor:
            tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1])

            def batch_decode(self, *_args, **_kwargs):
                return ["silver"]

        good = assay.run_model(Model([1, 2]), Processor(), {}, pair_id="color_05", hook_count=[0])
        self.assertEqual(good["hook_count"], 0)
        self.assertTrue(good["eos_terminated"])
        self.assertFalse(good["censored"])
        for ids in ([1, 2, 3], [1, 2, 2]):
            with self.subTest(ids=ids):
                bad = assay.run_model(Model(ids), Processor(), {}, pair_id="color_05")
                self.assertTrue(bad["censored"])
                self.assertFalse(bad["eos_terminated"])


if __name__ == "__main__":
    unittest.main()
