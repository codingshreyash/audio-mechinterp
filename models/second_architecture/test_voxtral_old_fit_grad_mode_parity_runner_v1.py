"""CPU fake-model tests for the new forward-only diagnostic plumbing."""

from __future__ import annotations

import unittest

import torch

import voxtral_old_fit_grad_mode_parity_runner_v1 as target
from test_voxtral_old_fit_handoff_replay_stage_b_v1 import fake_model_and_batch


class GraphNoGradRunnerTests(unittest.TestCase):
    def test_fixed_call_order_and_caps(self):
        self.assertEqual(target.full_schedule(), (
            ("encoder31", "N", 0), ("encoder31", "G", 0),
            ("encoder31", "G", 1), ("encoder31", "N", 1),
            ("projector", "G", 0), ("projector", "N", 0),
            ("projector", "N", 1), ("projector", "G", 1)))
        self.assertEqual(len(target.direct_schedule()), 16)
        self.assertEqual(target.direct_schedule()[:4], (
            ("encoder31", "N", "N", 0), ("encoder31", "N", "G", 0),
            ("encoder31", "N", "G", 1), ("encoder31", "N", "N", 1)))
        self.assertEqual((target.MODEL_SECONDS, target.MAX_DRIVER_BYTES,
                          target.MAX_OUTPUT_BYTES), (180, 30 * 1024**3, 1024**3))

    def test_full_hook_captures_returned_source_before_any_pair_assertion(self):
        model, batch = fake_model_and_batch()
        model.eval()
        for site in target.SITES:
            source = (batch["input_features"] if site == "encoder31" else
                      model.model.multi_modal_projector(batch["input_features"])).to(torch.bfloat16)
            for mode in ("N", "G"):
                record = target.full_forward(model, batch, site=site,
                                             full_source=source, mode=mode,
                                             ordinal=0, repeat=0, device="cpu")
                self.assertEqual(record["kind"], "full")
                self.assertEqual(record["mode"], mode)
                self.assertTrue(torch.equal(record["source_post_full"], source.float()))
                self.assertTrue(torch.equal(record["projector_full"],
                                            record["lm_input_full"][0, :375]))
                self.assertTrue(torch.equal(record["lm_input_full"],
                                            record["lm0_input_full"]))
                self.assertEqual(set(record["lm_terminal"]),
                                 {"lm0", "lm7", "lm14", "lm22", "lm29"})
                self.assertEqual(tuple(record["first_answer_logits"].shape), (131072,))
                self.assertEqual(record["audio_placeholder_positions"], list(range(375)))

    def test_direct_fixed_input_under_both_modes(self):
        model, batch = fake_model_and_batch()
        model.eval()
        lm_input = torch.zeros(target.replay.LM_INPUT_SHAPE, dtype=torch.float32)
        lm_input[0, 0, 0] = 2.0
        records = [target.direct_forward(model, batch, site="encoder31",
                                         origin_mode="N", origin_ordinal=0,
                                         lm_input=lm_input, mode=mode,
                                         ordinal=i, repeat=0, device="cpu")
                   for i, mode in enumerate(("N", "G"))]
        for record in records:
            self.assertTrue(torch.equal(record["lm_input_full"], lm_input))
            self.assertTrue(torch.equal(record["lm0_input_full"], lm_input))
            self.assertEqual(record["input_sha256"],
                             target.native.first.tensor_sha(lm_input))
        self.assertTrue(torch.equal(records[0]["first_answer_logits"],
                                    records[1]["first_answer_logits"]))

    def test_failed_mode_or_shape_never_executes(self):
        model, batch = fake_model_and_batch()
        source = torch.zeros(target.replay.SOURCE_SHAPES["encoder31"], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "site/mode"):
            target.full_forward(model, batch, site="encoder31", full_source=source,
                                mode="inference_mode", ordinal=0, repeat=0, device="cpu")
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            target.full_forward(model, batch, site="encoder31", full_source=source.float(),
                                mode="N", ordinal=0, repeat=0, device="cpu")


if __name__ == "__main__":
    unittest.main()
