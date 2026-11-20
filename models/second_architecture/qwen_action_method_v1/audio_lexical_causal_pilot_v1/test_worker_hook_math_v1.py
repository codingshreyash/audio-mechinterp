"""No-weight tests for exact lexical worker hook footprints and controls."""
from __future__ import annotations

import unittest

import torch

from . import plan, worker


class FakeTokenizer:
    all_special_ids = [151645, 151644, 151646]

    def decode(self, ids, **_kwargs):
        return "".join({1: "red", 2: "blue", 3: ".", 151645: "<eos>",
                        151644: "<bad>", 151646: "<bad2>"}.get(x, "?") for x in ids)


class PreLayer(torch.nn.Module):
    def forward(self, hidden_states, cu_seqlens):
        return hidden_states


class WorkerHookMathTests(unittest.TestCase):
    def test_strict_complete_output_rejects_hidden_special(self):
        tok = FakeTokenizer()
        self.assertEqual(worker.classify([1, 151645], tok)["label"], "red")
        self.assertEqual(worker.classify([2, 3, 151645], tok)["label"], "blue")
        self.assertIsNone(worker.classify([1, 151644, 151645], tok)["label"])
        self.assertIsNone(worker.classify([1, 151645, 3], tok)["label"])
        self.assertIsNone(worker.classify([1] * plan.MAX_NEW_TOKENS, tok)["label"])

    def test_hook_sites_are_pre_attention_input_and_post_projection_output(self):
        pre = PreLayer()
        post = torch.nn.Identity()
        modules = {"audio.pre_attention.0": pre, "audio.to_language": post}
        pre_value = torch.zeros((800, 1280), dtype=torch.float32)
        projection_value = torch.zeros((400, 2048), dtype=torch.float32)
        seen = []
        def observe(site, value):
            seen.append((site, list(value.shape)))
            return None
        handles = worker.hook_state(modules, observe)
        try:
            torch.testing.assert_close(pre(pre_value, cu_seqlens=torch.tensor([0, 100])), pre_value)
            torch.testing.assert_close(post(projection_value), projection_value)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen, [("audio.pre_attention.0", [800, 1280]),
                                ("audio.to_language", [400, 2048])])
        hook = worker.hook_one(modules, "audio.pre_attention.0", lambda x: x + 1)
        try:
            torch.testing.assert_close(pre(pre_value, cu_seqlens=torch.tensor([0, 100])),
                                       torch.ones_like(pre_value))
        finally:
            hook.remove()
        hook = worker.hook_one(modules, "audio.to_language", lambda x: x + 2)
        try:
            torch.testing.assert_close(post(projection_value),
                                       torch.full_like(projection_value, 2))
        finally:
            hook.remove()

    def test_six_arms_edit_only_frozen_windows_and_control_norms(self):
        for site in plan.SITES:
            with self.subTest(site=site):
                w = plan.WINDOWS[site]
                clean = torch.zeros((w["total"], w["width"]), dtype=torch.float32)
                donor = clean.clone()
                lo, hi = w["local"]
                donor[lo:hi] = 1
                donor[w["chunk"][0]] = 3
                for arm in plan.ARMS:
                    patched, info = worker.make_edit(site, arm, clean, clean, donor,
                                                      plan.NOISE_SEED, torch)
                    delta = patched-clean
                    changed = torch.any(delta != 0, dim=1).nonzero().flatten().tolist()
                    if arm == "local_donor":
                        self.assertEqual(changed, list(range(lo, hi)))
                    elif arm == "chunk_donor":
                        self.assertTrue(all(w["chunk"][0] <= i < w["chunk"][1] for i in changed))
                    elif arm == "full_donor":
                        torch.testing.assert_close(patched, donor)
                    elif arm == "self":
                        torch.testing.assert_close(patched, clean)
                    elif arm == "relocated_delta":
                        x, y = w["relocated"]
                        self.assertEqual(changed, list(range(x, y)))
                        self.assertAlmostEqual(info["realized_edit_l2"], info["local_delta_l2"],
                                               places=4)
                    elif arm == "equal_l2_noise":
                        self.assertEqual(changed, list(range(lo, hi)))
                        self.assertAlmostEqual(info["realized_edit_l2"], info["local_delta_l2"],
                                               places=4)
                        repeat, _ = worker.make_edit(site, arm, clean, clean, donor,
                                                     plan.NOISE_SEED, torch)
                        torch.testing.assert_close(repeat, patched)


if __name__ == "__main__":
    unittest.main()
