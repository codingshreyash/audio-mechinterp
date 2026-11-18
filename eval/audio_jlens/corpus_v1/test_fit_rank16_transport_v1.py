"""CPU toy checks for shared source VJPs, basis math, and saved transport."""

import tempfile
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_rank16_transport_v1 import (
    RANK, SITES, TARGET, accumulate_one_vjp, atomic_torch_save,
    canonical_basis, captures_for_forward, category_means, empty_accumulator,
    tensor_sha,
)


class TinyAudioGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.connector = torch.nn.Linear(2, 2, bias=False)
        self.terminal = torch.nn.Linear(2, 2048, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
            self.connector.weight.copy_(2 * torch.eye(2))
            self.terminal.weight.zero_()
            self.terminal.weight[0, 0] = 1

    def forward(self, input_ids, use_cache=False, return_dict=True):
        a = self.encoder(input_ids)
        b = self.connector(a)
        c = self.terminal(b).unsqueeze(0)
        return {"last": c}


class CorpusTransportTests(unittest.TestCase):
    def test_both_sites_share_gradient_graph(self):
        model = TinyAudioGraph()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        modules = {"encoder31": model.encoder, "connector": model.connector,
                   TARGET: model.terminal}
        # Terminal hook in the real model sees [1,seq,2048]. The toy terminal
        # module must expose that shape too.
        class Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                return self.inner(x).unsqueeze(0)
        model.terminal = Wrap(model.terminal)
        modules[TARGET] = model.terminal
        # TinyAudioGraph's forward would add another batch dimension; replace
        # it with a toy forward that exposes the wrapped terminal directly.
        model.forward = lambda input_ids, **kwargs: model.terminal(
            model.connector(model.encoder(input_ids))
        )
        captured = captures_for_forward(
            model, {"input_ids": torch.tensor([[1.0, 2.0]])}, modules,
            gradient=True,
        )
        terminal = captured[TARGET]
        cotangent = torch.zeros_like(terminal)
        cotangent[0, -1, 0] = 1
        ge, gc = torch.autograd.grad(
            terminal, [captured["encoder31"], captured["connector"]],
            grad_outputs=cotangent,
        )
        self.assertTrue(torch.allclose(ge, torch.tensor([[2.0, 0.0]])))
        self.assertTrue(torch.allclose(gc, torch.tensor([[1.0, 0.0]])))

    def test_basis_accumulation_and_serialization(self):
        g = torch.Generator().manual_seed(7)
        deltas = torch.randn(24, 2048, generator=g)
        basis, singular = canonical_basis(deltas)
        self.assertEqual(tuple(basis.shape), (2048, RANK))
        self.assertTrue(torch.allclose(basis.T @ basis, torch.eye(RANK), atol=1e-4))
        self.assertGreater(singular[RANK-1], 0)
        stage1 = {"target_basis": basis, "target_basis_sha256": tensor_sha(basis),
                  "metadata": {"fit": True},
                  "source_bin_means": {
                      "encoder31": torch.zeros(48, 6, 3),
                      "connector": torch.zeros(48, 6, 2),
                  }}
        acc = empty_accumulator(stage1)
        gradients = {
            "encoder31": torch.tensor([[1., 2., 3.]] * 6),
            "connector": torch.tensor([[4., 5.]] * 6),
        }
        rows = {site: [[i] for i in range(6)] for site in SITES}
        accumulate_one_vjp(acc, gradients, rows, 0)
        self.assertTrue(torch.allclose(acc["jacobian_sums"]["encoder31"][:, 0],
                                       gradients["encoder31"].double()))
        self.assertTrue(torch.allclose(acc["jacobian_sums"]["connector"][:, 0],
                                       gradients["connector"].double()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_torch_save(path, acc)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["metadata"], acc["metadata"])
            self.assertTrue(torch.equal(loaded["jacobian_sums"]["connector"],
                                        acc["jacobian_sums"]["connector"]))

    def test_category_means_do_not_use_eval(self):
        cases = [{"category": category} for category in
                 ("animal", "color", "nature", "object") for _ in range(12)]
        target = torch.arange(48, dtype=torch.float32)[:, None].repeat(1, 2048)
        sources = {"encoder31": torch.ones(48, 6, 3),
                   "connector": torch.ones(48, 6, 2) * 2}
        means = category_means(cases, target, sources)
        self.assertEqual(set(means), {"animal", "color", "nature", "object"})
        self.assertAlmostEqual(float(means["animal"]["target"][0]), 5.5)
        self.assertAlmostEqual(float(means["object"]["target"][0]), 41.5)


if __name__ == "__main__":
    unittest.main()
