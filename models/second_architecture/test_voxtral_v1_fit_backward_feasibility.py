"""CPU tests for the frozen, non-running backward feasibility math."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

import voxtral_v1_fit_backward_feasibility as pilot


class BackwardFeasibilityMathTests(unittest.TestCase):
    def test_zero_bf16_edit_is_exact(self):
        recipient = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.bfloat16)
        donor = torch.tensor([[2.0, 1.0], [2.0, 1.0]], dtype=torch.bfloat16)
        edit = pilot.uniform_edit(recipient, donor, [0, 1], 0.0)
        self.assertTrue(torch.equal(edit["edited"], recipient))
        self.assertEqual(float(edit["effective"].abs().sum()), 0.0)

    def test_rowwise_rounding_changes_uniform_contraction(self):
        # BF16 spacing differs between the two recipient rows. The intended
        # uniform edit can therefore become a nonuniform realized edit.
        recipient = torch.tensor([[1.0], [4.0]], dtype=torch.bfloat16)
        donor = torch.tensor([[1.5], [4.5]], dtype=torch.bfloat16)
        edit = pilot.uniform_edit(recipient, donor, [0, 1], 0.025)
        grad = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
        terms = pilot.contract_rows(grad, edit)
        self.assertEqual(terms["uniform_mean"], 0.0)
        self.assertEqual(terms["ideal_uniform"], 0.0)
        self.assertNotEqual(terms["rowwise_oracle"], 0.0)

    def test_both_source_vjps_share_one_graph(self):
        # Encoder -> projector -> terminal. Both derivatives must come from
        # the same graph, as in the planned Voxtral backward pass.
        encoder = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        projector = 2.0 * encoder
        terminal = (3.0 * projector).sum()
        ge, gp = torch.autograd.grad(terminal, (encoder, projector))
        self.assertTrue(torch.equal(ge, torch.full_like(encoder, 6.0)))
        self.assertTrue(torch.equal(gp, torch.full_like(projector, 3.0)))

    def test_vector_error_requires_four_finite_coordinates(self):
        self.assertEqual(pilot.vector_error([0, 1, 2, 3], [0, 1, 2, 3])["absolute_l2"], 0.0)
        with self.assertRaises(ValueError):
            pilot.vector_error([1], [1])
        with self.assertRaises(ValueError):
            pilot.vector_error([0, 1, float("nan"), 3], [0, 1, 2, 3])

    def test_bf16_leaf_returns_bf16_vjp_before_fp32_storage(self):
        leaf = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, requires_grad=True)
        target = (leaf * 2).float().sum()
        (raw_vjp,) = torch.autograd.grad(target, (leaf,))
        self.assertEqual(raw_vjp.dtype, torch.bfloat16)
        self.assertEqual(raw_vjp.float().dtype, torch.float32)

    def test_preflight_failure_has_start_and_error_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "one_shot"
            with patch.object(pilot, "preflight", side_effect=ValueError("fixture preflight failure")):
                with self.assertRaisesRegex(ValueError, "fixture preflight failure"):
                    pilot.run(output)
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            error = json.loads((output / "error.json").read_text())
            self.assertEqual(progress[0]["phase"], "attempt_started")
            self.assertEqual(error["error_type"], "ValueError")
            self.assertFalse((output / "meta.json").exists())
            with self.assertRaises(FileExistsError):
                pilot.run(output)

    def test_mps_failure_is_inside_attempt_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "one_shot"
            with patch.object(pilot, "preflight", return_value={"sources": {"modeling_llama": pilot.LLAMA_SHA}}), \
                 patch.object(torch.backends.mps, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "MPS unavailable"):
                    pilot.run(output)
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            error = json.loads((output / "error.json").read_text())
            self.assertEqual([entry["phase"] for entry in progress], ["attempt_started", "preflight_validated"])
            self.assertEqual(error["error_type"], "RuntimeError")

    def test_preflight_time_counts_toward_attempt_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "one_shot"
            with patch.object(pilot, "preflight", return_value={"sources": {}}), \
                 patch.object(pilot, "check_time", side_effect=TimeoutError("cap")):
                with self.assertRaisesRegex(TimeoutError, "cap"):
                    pilot.run(output)
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            error = json.loads((output / "error.json").read_text())
            self.assertEqual([entry["phase"] for entry in progress], ["attempt_started"])
            self.assertEqual(error["error_type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
