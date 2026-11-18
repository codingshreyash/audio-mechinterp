"""CPU-only correctness checks for terminal residual Jacobian fitting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from fit_terminal_jacobian import fit_rows_from_graph, initialize_state


class TerminalJacobianTests(unittest.TestCase):
    def test_start_graph_hook_keeps_values_and_exposes_residual_gradient(self) -> None:
        first = torch.nn.Linear(3, 3, bias=False)
        second = torch.nn.Linear(3, 3, bias=False)
        x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        expected = second(first(x)).detach()
        captured = {}

        def start_graph(_module, _inputs, output):
            residual = output.detach().requires_grad_(True)
            captured["source"] = residual
            return residual

        handle = first.register_forward_hook(start_graph)
        try:
            actual = second(first(x))
        finally:
            handle.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.autograd.grad(actual[0, -1, 0], captured["source"])[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[0, -1].norm()), 0)

    def test_rows_and_partial_resume_match_analytic_jacobian(self) -> None:
        width, dim_batch = 5, 2  # exercises a short final coordinate batch
        a = torch.arange(1, width * width + 1, dtype=torch.float32).reshape(width, width) / 13
        b = torch.arange(width * width, 0, -1, dtype=torch.float32).reshape(width, width) / 17
        meta = {"width": width, "source_layers": [0, 17], "case": "tiny-terminal"}

        def one_forward():
            source0 = torch.arange(dim_batch * 3 * width, dtype=torch.float32).reshape(
                dim_batch, 3, width
            ).requires_grad_(True)
            source17 = source0 @ a
            target = source17 @ b
            return target, {0: source0, 17: source17}

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "partial.pt"
            state = initialize_state(checkpoint, meta)
            target, sources = one_forward()
            saved_batches = []

            def interrupt_after_first(record):
                saved_batches.append(record)
                if len(saved_batches) == 1:
                    raise RuntimeError("simulated interruption after atomic save")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                fit_rows_from_graph(
                    target, sources, state, dim_batch=dim_batch,
                    checkpoint=checkpoint, on_batch=interrupt_after_first,
                )
            self.assertEqual(int(state["completed"].sum()), 2)
            resumed = initialize_state(checkpoint, meta)
            self.assertEqual(int(resumed["completed"].sum()), 2)
            target, sources = one_forward()  # graph cannot be serialized
            fit_rows_from_graph(
                target, sources, resumed, dim_batch=dim_batch,
                checkpoint=checkpoint,
            )
            self.assertTrue(bool(resumed["completed"].all()))
            self.assertEqual(len(resumed["batch_records"]), 3)
            torch.testing.assert_close(resumed["rows"]["0"], (a @ b).T)
            torch.testing.assert_close(resumed["rows"]["17"], b.T)
            self.assertTrue(torch.isfinite(resumed["rows"]["0"]).all())
            self.assertTrue(torch.isfinite(resumed["rows"]["17"]).all())

    def test_mismatched_checkpoint_metadata_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "fit.pt"
            state = initialize_state(checkpoint, {"width": 3, "source_layers": [0]})
            from fit_terminal_jacobian import atomic_torch_save
            atomic_torch_save(checkpoint, state)
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                initialize_state(checkpoint, {"width": 3, "source_layers": [1]})


if __name__ == "__main__":
    unittest.main()
