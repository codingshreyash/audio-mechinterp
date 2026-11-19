"""Source-only fake tests for CUDA stage admission, one-shot, and V3 method parity."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import typed_pilot_cuda_backend_v1 as backend
from . import typed_pilot_cuda_parent_v1 as parent
from . import typed_pilot_audit_v3 as saved_method
from .runner_contract_v1 import validate_teacher_forced_triplet


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Identity()
        self.second = torch.nn.Identity()

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        length = input_ids.shape[1]
        state = input_ids.float().unsqueeze(-1).repeat(1, 1, 2048) / 1000
        state = self.second(self.first(state) + 0.5)
        logits = torch.zeros(1, length, 151646, dtype=torch.float32)
        for index, token in enumerate((22852, 7384, 88493, 151645)):
            logits[:, :, token] = state[:, :, index]
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, **_kwargs):
        self.forward(input_ids, attention_mask=attention_mask)
        suffix = torch.tensor([[22852, 151645]], dtype=input_ids.dtype)
        return SimpleNamespace(sequences=torch.cat((input_ids, suffix), dim=1))


class FakeTokenizer:
    eos_token_id = 151645
    all_special_ids = [151645]

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join({22852: "LIST", 7384: "ASK", 88493: "CANCEL"}.get(x, "?")
                       for x in ids)


class CudaTypedPilotTests(unittest.TestCase):
    def test_absent_go_fails_before_attempt_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "load_001"
            go = Path(temporary) / "missing_go.json"
            with patch.object(backend, "stage_paths", return_value=(attempt, go)), \
                 patch.object(backend, "source_identity", return_value={}):
                with self.assertRaisesRegex(ValueError, "independent CUDA stage GO absent"):
                    parent.launch("load")
            self.assertFalse(attempt.exists())

    def test_runtime_uuid_mismatch_rejects_go_before_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            attempt, go, review = base / "load_001", base / "go.json", base / "review.md"
            review.write_text("independent fake review")
            receipt = {"schema": "qwen_action_typed_pilot_cuda_go_v1",
                "status": "independent_executable_go", "stage": "load",
                "approved_attempt": str(attempt.resolve()),
                "scope": "one typed F_L1 pilot only",
                "qa_policy": "typed_F_L1_only_no_audio_QA_exception",
                "qa_amendment_sha256": backend.sha(backend.QA_AMENDMENT),
                "parent_amendment_sha256": backend.sha(backend.PARENT_AMENDMENT),
                "source_identity": {}, "runtime_identity": {"gpu_uuid": "GPU-A"},
                "review_sha256": backend.sha(review)}
            go.write_text(json.dumps(receipt))
            with patch.object(backend, "stage_paths", return_value=(attempt, go)), \
                 patch.object(backend, "REVIEW", review), \
                 patch.object(backend, "source_identity", return_value={}), \
                 patch.object(backend, "runtime_identity",
                              return_value={"gpu_uuid": "GPU-B"}):
                with self.assertRaisesRegex(ValueError, "live CUDA runtime differs"):
                    parent.launch("load")
            self.assertFalse(attempt.exists())

    def test_direct_worker_without_live_parent_fails_before_candidate_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "unused"
            with patch.object(backend, "parent_admission",
                              side_effect=ValueError("bounded parent absent")):
                with self.assertRaisesRegex(ValueError, "bounded parent absent"):
                    backend.run_worker(attempt, "load")
            self.assertFalse(attempt.exists())

    def test_partial_consumes_one_shot_and_never_writes_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "load_001"
            go = Path(temporary) / "go.json"
            go.write_text("{}")
            runtime = {"gpu_uuid": "GPU-test"}
            probe = {"collected_utc": "2026-11-19T00:00:00+00:00",
                     "runtime_identity": runtime, "gpu_used_bytes": 0,
                     "parent_pid": 1}
            observation = {"worker_returncode": 1, "worker_stop_reason": "nonzero_exit",
                           "worker_elapsed_seconds": 0.01,
                           "worker_peak_tree_rss_bytes": 1,
                           "worker_peak_gpu_used_bytes": 1,
                           "worker_resource_samples": 1,
                           "worker_parent_heartbeats": 1}
            with patch.object(backend, "stage_paths", return_value=(attempt, go)), \
                 patch.object(backend, "validate_go",
                              return_value={"source_identity": {}, "runtime_identity": runtime}), \
                 patch.object(parent, "_host_probe", return_value=probe), \
                 patch.object(parent, "_run_worker", return_value=observation):
                result = parent.launch("load")
                self.assertEqual(result["status"], "invalid_or_partial")
                self.assertFalse((attempt / "terminal.json").exists())
                self.assertTrue((attempt / "result.json").exists())
                with self.assertRaisesRegex(ValueError, "already used"):
                    parent.launch("load")

    def test_fake_triplet_and_complete_generation(self):
        model = FakeModel()
        batch = {"input_ids": torch.tensor([[8, 9]]),
                 "attention_mask": torch.ones((1, 2), dtype=torch.long)}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch.object(backend, "measure_cuda", return_value=0):
                rows = backend.teacher_forced_triplet(model, batch,
                    {"text.residual.17": model.first,
                     "text.residual.34": model.second}, torch, folder)
            self.assertEqual([row["label"] for row in rows],
                             ["LIST", "ASK", "CANCEL"])
            reloaded, sidecars = saved_method.reconstruct_triplet(folder, rows)
            self.assertEqual(len(sidecars), 36)
            self.assertEqual(validate_teacher_forced_triplet(reloaded)
                             ["cross_suffix_state_maxabs"],
                             {site: 0.0 for site in backend.SITES})
            for row in reloaded:
                self.assertTrue(all(value == 0.0 for value in
                                    saved_method.check_reference_parity(row).values()))
            processor = SimpleNamespace(tokenizer=FakeTokenizer())
            generated = backend.generation_pair(model, batch, model.second,
                                                processor, torch)
            self.assertEqual(generated["raw_no_hook_ids"], [22852, 151645])
            self.assertEqual(generated["raw_no_hook_ids"],
                             generated["raw_read_only_ids"])
            self.assertGreater(generated["read_only_hook_calls"], 0)


if __name__ == "__main__":
    unittest.main()
