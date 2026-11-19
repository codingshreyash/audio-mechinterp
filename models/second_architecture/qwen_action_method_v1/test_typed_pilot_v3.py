"""No-weight fake-model tests for the gated typed pilot source."""
from __future__ import annotations

import json
import builtins
import os
import sys
import tempfile
import unittest
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import typed_pilot_audit_v3 as audit
from . import typed_pilot_backend_v3 as backend
from . import typed_pilot_parent_v3 as parent
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
            self.assertEqual(len(refs), 36)
            checked = validate_teacher_forced_triplet(reloaded)
            self.assertEqual(checked["cross_suffix_state_maxabs"],
                             {site: 0.0 for site in backend.SITES})
            for row in reloaded:
                self.assertTrue(all(value == 0.0 for value in
                                    audit.check_reference_parity(row).values()))
            reloaded[0]["reference"]["no_hook"]["first_logits"][22852] += 0.1
            with self.assertRaisesRegex(ValueError, "saved unmodified/read-only/leaf parity failed"):
                audit.check_reference_parity(reloaded[0])
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

    def test_direct_worker_with_fake_go_and_prehost_cannot_reach_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "attempt_001"
            attempt.mkdir()
            pre = attempt / "pre_host.json"
            pre.write_text("{}")
            fake_go = attempt / "fake_go.json"
            fake_go.write_text("{}")
            original_import = builtins.__import__
            def block_torch(name, *args, **kwargs):
                if name == "torch" or name.startswith("transformers"):
                    raise AssertionError("candidate import reached")
                return original_import(name, *args, **kwargs)
            with patch.object(backend, "validate_go", return_value={"source_identity": {}}), \
                 patch.object(backend, "ATTEMPT", attempt), \
                 patch.object(backend, "GO", fake_go), \
                 patch("models.second_architecture.wav2vec2_fit_noise_norm_causal_host_v1.validate_power"), \
                 patch.object(backend, "measure_mps") as mps, \
                 patch("builtins.__import__", side_effect=block_torch), \
                 patch.object(sys, "argv", ["typed_pilot_backend_v3", "--worker",
                                             "--attempt", str(attempt)]):
                with self.assertRaises(FileNotFoundError):
                    backend.main()
                mps.assert_not_called()
                (attempt / "start.json").write_text(json.dumps({
                    "schema": "qwen_action_typed_pilot_parent_v3",
                    "status": "worker_authorized", "approved_attempt": str(attempt),
                    "go_sha256": backend.sha(fake_go), "source_identity": {},
                    "pre_host_sha256": backend.sha(pre), "parent_pid": os.getppid(),
                    "nonce_sha256": "0" * 64,
                    "deadline_monotonic": time.monotonic() + 30}))
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ValueError, "inherited parent pipe absent"):
                        backend.main()
                mps.assert_not_called()
            self.assertFalse((attempt / "worker_output").exists())

    def test_go_requires_narrow_qa_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            review = folder / "review.md"
            review.write_text("independent fake review")
            receipt = {"schema": "qwen_action_typed_pilot_go_v3",
                       "status": "independent_executable_go",
                       "approved_attempt": str(backend.ATTEMPT),
                       "scope": "one typed F_L1 pilot only",
                       "source_identity": {},
                       "review_sha256": backend.sha(review)}
            go = folder / "go.json"
            go.write_text(json.dumps(receipt))
            with patch.object(backend, "source_identity", return_value={}), \
                 patch.object(backend, "GO", go), \
                 patch.object(backend, "REVIEW", review):
                with self.assertRaisesRegex(ValueError, "GO binding invalid"):
                    backend.validate_go()
                receipt["qa_policy"] = "typed_F_L1_only_no_audio_QA_exception"
                receipt["qa_amendment_sha256"] = backend.sha(backend.QA_AMENDMENT)
                receipt["parent_amendment_sha256"] = backend.sha(backend.PARENT_AMENDMENT)
                go.write_text(json.dumps(receipt))
                self.assertEqual(backend.validate_go(), receipt)

    def test_host_archive_fsync_failure_stops_before_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            fake = {"outputs": {"power_log": "fake log"},
                    "collected_utc": datetime.now(timezone.utc).isoformat()}
            with patch.object(parent.host, "probe", return_value=fake), \
                 patch.object(parent.host, "validate_power"), \
                 patch.object(parent, "_fsync_path",
                              side_effect=OSError("archive fsync failed")):
                with self.assertRaisesRegex(OSError, "archive fsync failed"):
                    parent._probe(attempt, "pre", start=datetime.now(timezone.utc),
                                  deadline=time.monotonic()+30)
            self.assertFalse((attempt / "terminal.json").exists())
            self.assertFalse((attempt / "pre_host.json").exists())

    def test_missing_terminal_never_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            (attempt / "start.json").write_text("{}")
            (attempt / "result.json").write_text("{}")
            with patch.object(backend, "ATTEMPT", attempt), \
                 patch.object(backend, "validate_go", return_value={}):
                with self.assertRaisesRegex(FileNotFoundError, "terminal.json"):
                    audit.audit(attempt)


if __name__ == "__main__":
    unittest.main()
