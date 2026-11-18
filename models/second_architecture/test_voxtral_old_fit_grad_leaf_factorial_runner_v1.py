"""CPU tests for the distinct grad-context x leaf-flag diagnostic."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

import voxtral_old_fit_grad_leaf_factorial_runner_v1 as target
from test_voxtral_old_fit_handoff_replay_stage_b_v1 import fake_model_and_batch


class GradLeafFactorialRunnerTests(unittest.TestCase):
    def test_fixed_call_order_and_caps(self):
        self.assertEqual(target.full_schedule(), (
            ("encoder31", "N0", 0), ("encoder31", "G1", 0),
            ("encoder31", "G0", 0), ("encoder31", "N1", 0),
            ("encoder31", "N1", 1), ("encoder31", "G0", 1),
            ("encoder31", "G1", 1), ("encoder31", "N0", 1),
            ("projector", "G1", 0), ("projector", "N0", 0),
            ("projector", "N1", 0), ("projector", "G0", 0),
            ("projector", "G0", 1), ("projector", "N1", 1),
            ("projector", "N0", 1), ("projector", "G1", 1)))
        self.assertEqual(len(target.direct_schedule()), 16)
        self.assertEqual(target.direct_schedule()[:4], (
            ("encoder31", "N0", "N0", 0), ("encoder31", "N0", "G1", 0),
            ("encoder31", "N0", "G0", 0), ("encoder31", "N0", "N1", 0)))
        self.assertEqual((target.MODEL_SECONDS, target.MAX_DRIVER_BYTES,
                          target.MAX_OUTPUT_BYTES), (300, 30 * 1024**3, 1024**3))

    def test_real_preflight_batchfeature_and_exact_pins(self):
        prepared = target.native.preflight_cpu()
        original = prepared["batches"]["beaver"]
        self.assertNotIsInstance(original, dict)
        self.assertEqual(type(original).__name__, "BatchFeature")
        normalized = target.checked_batch(prepared)
        self.assertIsInstance(normalized, dict)
        self.assertEqual(set(normalized), set(target.BATCH_SHA256))
        self.assertEqual({key: target.native.first.tensor_sha(value)
                          for key, value in normalized.items()}, target.BATCH_SHA256)
        self.assertEqual({key: tuple(value.shape) for key, value in normalized.items()},
                         {"input_ids": (1, 396), "attention_mask": (1, 396),
                          "input_features": (1, 128, 3000)})
        bad = dict(normalized)
        bad["input_ids"] = bad["input_ids"].clone()
        bad["input_ids"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "tensor hash"):
            target.checked_batch({"batches": {"beaver": bad}})
        bad["input_ids"] = normalized["input_ids"].to(torch.int32)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            target.checked_batch({"batches": {"beaver": bad}})
        bad = dict(normalized)
        bad["extra"] = normalized["attention_mask"]
        with self.assertRaisesRegex(ValueError, "field set"):
            target.checked_batch({"batches": {"beaver": bad}})

    def test_bad_batch_stops_before_model_factory_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            called = []
            def forbidden_factory(_dtype):
                called.append(True)
                raise AssertionError("model factory reached")
            upstream = {"stage_a": {}, "prior_model_work_seconds": 1.0,
                        "failed_manifest_sha256": "a" * 64,
                        "failed_v1_diagnostic_manifest_sha256": "b" * 64}
            with patch.object(target.driver, "phase_preflight",
                              return_value=({}, (), {})), \
                    patch.object(target, "checked_sources", return_value={}), \
                    patch.object(target.native, "preflight_cpu",
                                 return_value={"batches": {"beaver": object()}}):
                with self.assertRaisesRegex(ValueError, "batch mapping"):
                    target.run_diagnostic(output, upstream=upstream,
                                          model_factory=forbidden_factory,
                                          device="cpu")
            self.assertEqual(called, [])
            self.assertFalse(output.exists())

    def test_full_hook_captures_returned_source_before_any_pair_assertion(self):
        model, batch = fake_model_and_batch()
        model.eval()
        for site in target.SITES:
            source = (batch["input_features"] if site == "encoder31" else
                      model.model.multi_modal_projector(batch["input_features"])).to(torch.bfloat16)
            for mode in target.CONDITIONS:
                record = target.full_forward(model, batch, site=site,
                                             full_source=source, mode=mode,
                                             ordinal=0, repeat=0, device="cpu")
                self.assertEqual(record["kind"], "full")
                self.assertEqual(record["mode"], mode)
                self.assertEqual(record["factor_observed"],
                                 {"grad_enabled": mode.startswith("G"),
                                  "leaf_requires_grad": mode.endswith("1")})
                self.assertEqual(record["downstream_grad_flags"]["lm_input"]
                                 ["grad_enabled"], mode.startswith("G"))
                self.assertEqual(record["downstream_grad_flags"]["lm0_pre"]
                                 ["grad_enabled"], mode.startswith("G"))
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
                                         origin_mode="N0", origin_ordinal=0,
                                         lm_input=lm_input, mode=mode,
                                         ordinal=i, repeat=0, device="cpu")
                   for i, mode in enumerate(target.CONDITIONS)]
        for record in records:
            mode = record["mode"]
            self.assertEqual(record["factor_observed"],
                             {"grad_enabled": mode.startswith("G"),
                              "leaf_requires_grad": mode.endswith("1")})
            self.assertEqual(record["downstream_grad_flags"]["lm_input"],
                             {"grad_enabled": mode.startswith("G"),
                              "requires_grad": mode.endswith("1")})
            self.assertTrue(torch.equal(record["lm_input_full"], lm_input))
            self.assertTrue(torch.equal(record["lm0_input_full"], lm_input))
            self.assertEqual(record["input_sha256"],
                             target.native.first.tensor_sha(lm_input))
        for record in records[1:]:
            self.assertTrue(torch.equal(records[0]["first_answer_logits"],
                                        record["first_answer_logits"]))

    def test_failed_mode_or_shape_never_executes(self):
        model, batch = fake_model_and_batch()
        source = torch.zeros(target.replay.SOURCE_SHAPES["encoder31"], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "site/mode"):
            target.full_forward(model, batch, site="encoder31", full_source=source,
                                mode="inference_mode", ordinal=0, repeat=0, device="cpu")
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            target.full_forward(model, batch, site="encoder31", full_source=source.float(),
                                mode="N0", ordinal=0, repeat=0, device="cpu")


if __name__ == "__main__":
    unittest.main()
