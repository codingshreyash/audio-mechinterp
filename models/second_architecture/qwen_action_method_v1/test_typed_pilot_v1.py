"""No-weight fake-model tests for the gated typed pilot source."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import typed_pilot_audit_v1 as audit
from . import typed_pilot_backend_v1 as backend
from . import typed_pilot_parent_v1 as parent
from .runner_contract_v1 import validate_teacher_forced_triplet


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Identity()
        self.second = torch.nn.Identity()

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        length = input_ids.shape[1]
        state = input_ids.float().unsqueeze(-1).repeat(1, 1, 2048) / 1000
        state = self.first(state)
        state = self.second(state + 0.5)
        logits = torch.zeros(1, length, 151646, dtype=torch.float32)
        for index, token_id in enumerate((22852, 7384, 88493, 151645)):
            logits[:, :, token_id] = state[:, :, index]
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, **_kwargs):
        self.forward(input_ids, attention_mask=attention_mask)
        ending = torch.tensor([[22852, 151645]], dtype=input_ids.dtype)
        return SimpleNamespace(sequences=torch.cat((input_ids, ending), dim=1))


class Tokenizer:
    eos_token_id = 151645
    all_special_ids = [151645]

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join({22852: "LIST", 7384: "ASK", 88493: "CANCEL"}.get(x, "?")
                       for x in ids)


class TypedPilotTests(unittest.TestCase):
    def test_missing_go_rejects_without_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "unused"
            with patch.object(backend, "source_identity", return_value={}), \
                 patch.object(backend, "GO", Path(temp) / "absent.json"), \
                 patch.object(parent, "ATTEMPT", attempt):
                with self.assertRaisesRegex(ValueError, "independent typed-pilot GO absent"):
                    parent.launch()
            self.assertFalse(attempt.exists())

    def test_fake_teacher_forcing_and_saved_reconstruction(self):
        model = FakeModel()
        batch = {"input_ids": torch.tensor([[8, 9]], dtype=torch.long),
                 "attention_mask": torch.ones((1, 2), dtype=torch.long)}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.object(backend, "measure_mps", return_value=0):
                rows = backend.teacher_forced_triplet(model, batch,
                    {"text.residual.17": model.first,
                     "text.residual.34": model.second}, torch, folder)
            self.assertEqual([row["label"] for row in rows],
                             ["LIST", "ASK", "CANCEL"])
            reloaded, refs = audit.reconstruct_triplet(folder, rows)
            self.assertEqual(len(refs), 18)
            checked = validate_teacher_forced_triplet(reloaded)
            self.assertEqual(checked["cross_suffix_state_maxabs"],
                             {site: 0.0 for site in backend.SITES})
            first = rows[0]["sidecars"]["first_logits"]
            (folder / first["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "sidecar missing or altered"):
                audit.reconstruct_triplet(folder, rows)

    def test_fake_generation_read_only_parity(self):
        model = FakeModel()
        batch = {"input_ids": torch.tensor([[8, 9]], dtype=torch.long),
                 "attention_mask": torch.ones((1, 2), dtype=torch.long)}
        processor = SimpleNamespace(tokenizer=Tokenizer())
        result = backend.generation_pair(model, batch, model.second,
                                         processor, torch)
        self.assertEqual(result["raw_no_hook_ids"], [22852, 151645])
        self.assertEqual(result["no_hook"]["status"], "complete")
        self.assertGreater(result["read_only_hook_calls"], 0)

    def test_worker_refuses_before_output_without_go(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "unused"
            with patch.object(backend, "validate_go",
                              side_effect=ValueError("independent GO absent")):
                with self.assertRaisesRegex(ValueError, "independent GO absent"):
                    backend.run_worker(attempt)
            self.assertFalse(attempt.exists())


if __name__ == "__main__":
    unittest.main()
