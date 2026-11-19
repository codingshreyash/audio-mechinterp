"""No-weight Track B admission, geometry and hook tests."""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import wav2vec2_dev_word_disjoint_audit_v1 as dev_audit
import wav2vec2_dev_word_disjoint_v1 as dev
import wav2vec2_fit_complement_patch_v1 as fit_w2v

from . import candidate_common as common
from . import audit_whisper
from . import run_wav2vec2
from . import run_whisper


class CandidateSourceTests(unittest.TestCase):
    def test_exact_parser_and_full_reference(self) -> None:
        expected = ["i", "said", "banana", "and", "then", "stopped"]
        self.assertEqual(common.reference_tokens("banana"), expected)
        self.assertEqual(common.tokens("Ｉ said banana, and then stopped!"), expected)
        self.assertEqual(common.tokens("I_said_banana_and_then_stopped"), expected)
        for value in ("I said banana banana and then stopped",
                      "I said banana tomato and then stopped",
                      "I said banana then stopped",
                      "I said banana and then stopped 2", ""):
            self.assertNotEqual(common.tokens(value), expected)

    def test_exact_fixture_and_bounded_matrices_without_weights(self) -> None:
        cases, _ = common.fixture_cases()
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({c.word for c in cases}), 16)
        self.assertEqual({c.engine for c in cases}, {"piper", "samantha"})
        self.assertEqual(len(common.directions(cases, ("S", "C"))), 64)
        self.assertEqual(len(common.directions(cases, ("stock",))), 32)
        self.assertEqual(len(run_wav2vec2.shared_stats(cases)), 16)
        self.assertLess(run_wav2vec2.byte_ledger(cases)["total_bound"],
                        run_wav2vec2.MAX_OUTPUT_BYTES)
        self.assertLess(run_whisper.byte_ledger(cases)["total_bound"],
                        run_whisper.MAX_OUTPUT_BYTES)
        for case in cases:
            n = run_whisper.active_frames(case)
            a, b = run_whisper.window(case)
            self.assertTrue(0 < a < b < n)
            t = (case.frames - 400) // 320 + 1
            x, y = dev.frame_window(case.slot, t)
            self.assertTrue(0 < x < y < t)

    def test_human_qa_absence_closes_inference_gate(self) -> None:
        cases, _ = common.fixture_cases()
        with patch.object(common, "QA_RECEIPT", Path("/definitely-absent/QA_GO.json")):
            with self.assertRaisesRegex(ValueError, "human listening QA receipt"):
                common.qa_gate(cases)

    def test_whisper_regions_and_independent_patch_reconstruction(self) -> None:
        r = np.zeros((1, 1500, 384), dtype=np.float32)
        d = np.ones((1, 1500, 384), dtype=np.float32)
        n, w = 120, (40, 65)
        for arm in common.ARMS:
            seed = 713 if arm.startswith("random") else None
            candidate, norms = run_whisper.patch_state(r, d, n, w, arm, seed)
            independent = audit_whisper.patch_independent(r, d, n, w, arm, seed)
            self.assertTrue(np.array_equal(candidate, independent), arm)
            self.assertEqual(dev.tensor_sha(candidate), dev_audit.tensor_sha(independent))
            self.assertTrue(np.array_equal(candidate[:, n:], r[:, n:]))
            self.assertGreaterEqual(norms["realized_l2"], 0)
            if arm in ("W", "random_W"):
                self.assertTrue(np.array_equal(candidate[:, :w[0]], r[:, :w[0]]))
                self.assertTrue(np.array_equal(candidate[:, w[1]:], r[:, w[1]:]))
            if arm in ("O", "random_O"):
                self.assertTrue(np.array_equal(candidate[:, w[0]:w[1]], r[:, w[0]:w[1]]))
        self.assertTrue(np.array_equal(run_whisper.patch_state(r, d, n, w, "full", None)[0][:, n:],
                                       r[:, n:]))

    def test_whisper_fake_hook_changes_full_ids_and_is_removed(self) -> None:
        import torch

        class FakeProcessor:
            def batch_decode(self, sequences, skip_special_tokens=True):
                ids = sequences[0].tolist() if hasattr(sequences[0], "tolist") else sequences[0]
                return ["donor" if ids[1] == 3 else "recipient"]

        class FakeModel:
            def __init__(self):
                encoder = types.SimpleNamespace(
                    layers=[torch.nn.Identity()], layer_norm=torch.nn.Identity())
                self.model = types.SimpleNamespace(encoder=encoder)
                self.generation_config = types.SimpleNamespace(eos_token_id=2)

            def generate(self, **kwargs):
                x = torch.zeros((1, 1500, 384), dtype=torch.float32)
                x = self.model.encoder.layers[0](x)
                x = self.model.encoder.layer_norm(x)
                token = 3 if x[0, 50, 0] > 0 else 4
                return types.SimpleNamespace(sequences=torch.tensor([[1, token, 2]]))

        model = FakeModel()
        processor = FakeProcessor()
        clean = np.zeros((1, 1500, 384), dtype=np.float32)
        donor = clean.copy()
        donor[0, 50, 0] = 1.0
        for site in common.SITES_WHISPER:
            result, counts = run_whisper.generate(
                model, processor, {}, patch=(site, clean, donor))
            self.assertEqual(result["ids"], [1, 3, 2])
            self.assertEqual(counts[site], 1)
            self.assertEqual(len(run_whisper._hook_module(model, site)._forward_hooks), 0)

    def test_wav2vec2_fake_two_site_hooks_and_live_identity(self) -> None:
        import torch

        class TupleLayer(torch.nn.Module):
            def forward(self, x):
                return (x, None)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = TupleLayer()
                self.final = TupleLayer()
                self.wav2vec2 = types.SimpleNamespace(
                    feature_projection=self.projection,
                    encoder=types.SimpleNamespace(layers=[torch.nn.Identity() for _ in range(11)]
                                                  + [self.final]))

            def forward(self, input_values, **kwargs):
                t = (input_values.shape[1] - 400) // 320 + 1
                x = torch.zeros((1, t, 768), dtype=torch.float32)
                x = self.projection(x)[0]
                x = self.final(x)[0]
                logits = torch.zeros((1, t, 32), dtype=torch.float32)
                logits[:, :, 1] = x[:, :, 0]
                return types.SimpleNamespace(logits=logits)

        model = FakeModel()
        frames = 16000
        tensor = torch.zeros((1, frames), dtype=torch.float32)
        states, logits = run_wav2vec2.capture_clean(model, tensor, frames)
        self.assertEqual(set(states), set(common.SITES_W2V))
        self.assertTrue(np.array_equal(states["feature_projection"],
                                       states["post_block11"]))
        for site in common.SITES_W2V:
            edited = states[site].copy()
            edited[:, 20:25, 0] = 1.0
            patched, count = fit_w2v.patched_forward(
                model, tensor, site=site, clean_state=states[site],
                patched_state=edited)
            self.assertEqual(count, 1)
            self.assertTrue(np.all(patched[:, 20:25, 1] == 1))
            self.assertEqual(len((model.projection if site == "feature_projection"
                                  else model.final)._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
