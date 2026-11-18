"""CPU-only source tests for the CUDA old-FIT factorial worker."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import voxtral_old_fit_grad_leaf_cuda_runner_v1 as target
from test_voxtral_old_fit_handoff_replay_stage_b_v1 import fake_model_and_batch


class CudaRunnerSourceTests(unittest.TestCase):
    def test_protocol_schedule_caps_and_no_replacement_input(self):
        self.assertEqual(target.CASE_IDS, (
            "animal_08_kokoro_fit_chosen_beaver",
            "nature_09_piper_fit_chosen_waterfall"))
        self.assertEqual(target.full_schedule(), target.mps.full_schedule())
        self.assertEqual(target.direct_schedule(), target.mps.direct_schedule())
        self.assertEqual((len(target.full_schedule()), len(target.direct_schedule())), (16, 16))
        self.assertEqual((target.MAX_CUDA_BYTES, target.MAX_RSS_BYTES,
                          target.MAX_SECONDS, target.MAX_OUTPUT_BYTES),
                         (42 * 1024**3, 110 * 1024**3, 600, 1024**3))

    def test_raw_bytes_distinguish_signed_zero(self):
        positive = torch.tensor([0.0], dtype=torch.float32)
        negative = torch.tensor([-0.0], dtype=torch.float32)
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(target.raw_equal(positive, negative))
        self.assertTrue(target.raw_equal(positive, positive.clone()))

    def test_limited_writer_enforces_cap(self):
        raw = io.BytesIO()
        writer = target.LimitedWriter(raw, 4)
        self.assertEqual(writer.write(b"123"), 3)
        with self.assertRaisesRegex(OSError, "output cap"):
            writer.write(b"45")
        self.assertEqual(raw.getvalue(), b"123")

    def test_bad_admission_cannot_make_attempt_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            with self.assertRaisesRegex(ValueError, "admission"):
                target.run_case(output, case_index=0, attempt_id="fixed",
                                freeze_sha256="a" * 64, admission={})
            self.assertFalse(output.exists())

    def test_passive_capture_both_sites_and_full_direct_hooks(self):
        model, batch = fake_model_and_batch()
        model.eval()
        passive = target.passive_forward(model, batch, ordinal=0, device="cpu")
        self.assertEqual(passive["hook_counts"], {"encoder31": 1, "projector": 1})
        for site in target.SITES:
            source = passive["sources"][site]
            self.assertEqual(source.dtype, torch.float32)
            self.assertEqual(tuple(source.shape), target.replay.SOURCE_SHAPES[site])
            full = target.full_forward(model, batch, site=site, full_source=source,
                                       mode="N0", ordinal=0, repeat=0, device="cpu")
            self.assertEqual(set(full["hook_counts"]), {
                "source_override", "source_observer", "projector", "lm_input",
                "lm0_pre", "lm0", "lm7", "lm14", "lm22", "lm29"})
            self.assertTrue(all(count == 1 for count in full["hook_counts"].values()))
            self.assertTrue(target.raw_equal(full["source_post_full"], source))
            direct = target.direct_forward(
                model, batch, site=site, origin_ordinal=full["ordinal"],
                lm_input=full["lm_input_full"], mode="N0", ordinal=16,
                repeat=0, device="cpu")
            self.assertEqual(set(direct["hook_counts"]), {
                "lm0_pre", "lm0", "lm7", "lm14", "lm22", "lm29"})
            for key in ("lm0_input_full", "first_answer_logits"):
                self.assertTrue(target.raw_equal(full[key], direct[key]), key)
            self.assertTrue(target.raw_equal(full["lm_terminal"]["lm0"],
                                             direct["lm_terminal"]["lm0"]))

    def test_archive_hash_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.pt"
            path.write_bytes(b"bad")
            with patch.object(target.torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    target.checked_beaver_archive(path, target.BEAVER_ARCHIVE_SHA)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
