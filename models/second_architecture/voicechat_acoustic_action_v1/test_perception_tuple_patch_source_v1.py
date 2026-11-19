"""Fake tensors only: no VoiceChat weights, audio, or candidate inference."""

import unittest

import torch

from models.second_architecture.voicechat_acoustic_action_v1 import perception_tuple_patch_source_v1 as patcher


class PerceptionTuplePatchSourceTests(unittest.TestCase):
    def setUp(self):
        self.rec = (
            torch.arange(20, dtype=torch.float32).reshape(1, 5, 4),
            torch.tensor([5], dtype=torch.int64),
            torch.arange(10, dtype=torch.float32).reshape(1, 5, 2),
        )
        self.donor = (self.rec[0] + 100, self.rec[1].clone(), self.rec[2] + 200)

    def test_capture_and_read_only_do_not_change_recipient(self):
        saved = patcher.capture(self.rec)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(saved, self.rec)))
        self.assertIsNot(saved[0], self.rec[0])
        seen = patcher.patch(self.rec, mode="read_only")
        self.assertIs(seen[0], self.rec[0])

    def test_self_and_full_donor_exact_replay(self):
        own = patcher.patch(self.rec, mode="self")
        got = patcher.patch(self.rec, self.donor, mode="full_donor")
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(own, self.rec)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, self.donor)))
        self.assertTrue(torch.equal(self.rec[0], torch.arange(20).reshape(1, 5, 4)))

    def test_local_patch_covers_both_retained_tensors_only_inside_rows(self):
        got = patcher.patch(self.rec, self.donor, mode="rows_donor", rows=(1, 3))
        for i in range(5):
            expected = self.donor if i in (1, 2) else self.rec
            self.assertTrue(torch.equal(got[0][:, i], expected[0][:, i]))
            self.assertTrue(torch.equal(got[2][:, i], expected[2][:, i]))
        self.assertTrue(torch.equal(got[1], self.rec[1]))

    def test_geometry_and_length_mismatch_hold(self):
        bad_len = (self.donor[0], torch.tensor([4]), self.donor[2])
        bad_asr = (self.donor[0], self.donor[1], self.donor[2][:, :-1])
        bad_dtype = (self.donor[0].half(), self.donor[1], self.donor[2])
        for bad in (bad_len, bad_asr, bad_dtype):
            with self.subTest(bad=str(tuple(x.shape for x in bad))):
                with self.assertRaises(ValueError):
                    patcher.patch(self.rec, bad, mode="full_donor")

    def test_invalid_mode_and_window_hold(self):
        with self.assertRaises(ValueError):
            patcher.patch(self.rec, mode="other")
        for rows in ((-1, 2), (2, 2), (2, 6), (True, 2)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                patcher.patch(self.rec, self.donor, mode="rows_donor", rows=rows)

    def test_nonfinite_and_multibatch_hold(self):
        nonfinite = (self.rec[0].clone(), self.rec[1], self.rec[2])
        nonfinite[0][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            patcher.capture(nonfinite)
        two = tuple(x.repeat(2, *([1] * (x.ndim - 1))) for x in self.rec)
        with self.assertRaises(ValueError):
            patcher.capture(two)

    def test_fake_module_hook_applies_once_and_rejects_pass_leak(self):
        class FakePerception(torch.nn.Module):
            def __init__(self, result):
                super().__init__()
                self.result = result

            def forward(self):
                return self.result

        module = FakePerception(self.rec)
        hook = patcher.OneCallPerceptionHook(mode="full_donor", donor=self.donor)
        handle = module.register_forward_hook(hook)
        try:
            got = module()
            self.assertEqual(hook.calls, 1)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, self.donor)))
            with self.assertRaises(ValueError):
                module()  # a second pass must not be patched silently
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
