"""CPU-only fake-worker checks for CUDA admission and launcher bookkeeping."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "voxtral_old_fit_grad_leaf_cuda_launch_gate_v1",
    HERE / "voxtral_old_fit_grad_leaf_cuda_launch_gate_v1.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = self.root / "launch"
        self.worker_root = self.root / "worker"
        self.freeze = self.root / "freeze.json"
        self.receipt = self.root / "receipt.json"
        self.attempts = ["20261118T190000000000Z", "20261118T190001000000Z"]
        gate.put(self.freeze, {"test": "freeze"})
        gate.put(self.receipt, {"test": "receipt"})
        self.patches = [patch.object(gate, "RUNS", self.run_root),
                        patch.object(gate, "WORKER_RUNS", self.worker_root),
                        patch.object(gate, "FREEZE", self.freeze),
                        patch.object(gate, "ENV_RECEIPT", self.receipt)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.fake_freeze = {"attempt_ids": self.attempts}
        self.fake_receipt = {"checkpoint": {"tree_sha256": "e" * 64},
                             "storage": {"input_bundle_path": "/tmp/fake_bundle",
                                         "input_bundle_sha256": "b" * 64}}

    def reserve(self) -> None:
        gate.put(self.run_root / "reservation.json", {
            "status": "cuda_campaign_reserved_no_retry", "attempt_ids": self.attempts,
            "freeze_sha256": gate.sha(self.freeze)})

    def start(self, index: int) -> None:
        gate.put(self.run_root / self.attempts[index] / "launch_start.json", {
            "status": "cuda_case_started", "case_index": index,
            "attempt_id": self.attempts[index], "freeze_sha256": gate.sha(self.freeze)})

    def test_worker_claim_is_atomic_and_cannot_retry(self) -> None:
        self.reserve()
        self.start(0)
        with patch.object(gate, "verify_freeze", return_value=(self.fake_freeze, self.fake_receipt)), \
                patch.object(gate, "verify_live_environment"):
            admission = gate.verify_worker_start(0, self.attempts[0], gate.sha(self.freeze))
            self.assertEqual(admission["checkpoint_tree_sha256"], "e" * 64)
            with self.assertRaises(FileExistsError):
                gate.verify_worker_start(0, self.attempts[0], gate.sha(self.freeze))
            with self.assertRaises(ValueError):
                gate.verify_worker_start(1, self.attempts[0], gate.sha(self.freeze))

    def test_second_case_requires_first_case_audit(self) -> None:
        self.reserve()
        self.start(1)
        with patch.object(gate, "verify_freeze", return_value=(self.fake_freeze, self.fake_receipt)), \
                patch.object(gate, "verify_live_environment"):
            with self.assertRaises(FileNotFoundError):
                gate.verify_worker_start(1, self.attempts[1], gate.sha(self.freeze))
            gate.put(self.run_root / self.attempts[0] / "launch_finish.json", {
                "status": "cuda_case_failed_no_retry"})
            with self.assertRaises(ValueError):
                gate.verify_worker_start(1, self.attempts[1], gate.sha(self.freeze))
            (self.run_root / self.attempts[0] / "launch_finish.json").unlink()
            gate.put(self.run_root / self.attempts[0] / "launch_finish.json", {
                "status": "cuda_case_completed_cpu_audited_only"})
            gate.verify_worker_start(1, self.attempts[1], gate.sha(self.freeze))

    def test_fake_worker_watchdog_and_stream_cap(self) -> None:
        self.run_root.mkdir()
        fake = self.root / "fake_worker.py"
        fake.write_text("import time\ntime.sleep(0.4)\n")
        with patch.object(gate, "resource_usage", return_value=(gate.MAX_CUDA, 0)):
            code, _, reason = gate.bounded([sys.executable, str(fake)], self.run_root,
                                           ("out1", "err1"), self.worker_root, 5)
        self.assertEqual(reason, "external_cuda_memory_cap")
        self.assertNotEqual(code, 0)
        loud = self.root / "loud.py"
        loud.write_text("import sys\nsys.stdout.write('x' * 4096)\n")
        with patch.object(gate, "MAX_STREAM_BYTES", 1024), \
                patch.object(gate, "resource_usage", return_value=(0, 0)):
            _, _, reason = gate.bounded([sys.executable, str(loud)], self.run_root,
                                        ("out2", "err2"), self.worker_root, 5)
        self.assertEqual(reason, "capture_byte_cap")

    def test_case_result_provenance_and_caps(self) -> None:
        self.run_root.mkdir()
        self.worker_root.mkdir()
        self.start(0)
        gate.put(self.run_root / self.attempts[0] / "worker_claim.json", {
            "status": "cuda_worker_claimed_no_retry", "case_index": 0,
            "attempt_id": self.attempts[0], "freeze_sha256": gate.sha(self.freeze)})
        out = self.worker_root / self.attempts[0]
        gate.put(out / "meta.json", {"environment_receipt_sha256": gate.sha(self.receipt),
                                     "checkpoint_tree_sha256": "e" * 64})
        result = {"status": "completed", "schema": "voxtral_old_fit_grad_leaf_cuda_result_v1",
                  "case_id": gate.CASE_IDS[0], "case_index": 0, "attempt_id": self.attempts[0],
                  "freeze_sha256": gate.sha(self.freeze), "runner_sha256": "r" * 64,
                  "protocol_v1_sha256": "1" * 64, "protocol_v2_sha256": "2" * 64,
                  "schema_sha256": "3" * 64, "elapsed_seconds": 1,
                  "peak_cuda_allocated_bytes": 1, "peak_cuda_reserved_bytes": 1,
                  "peak_process_rss_bytes": 1}
        gate.put(out / "result.json", result)
        real_sha = gate.sha
        def fake_sha(path):
            return {gate.RUNNER: "r" * 64, gate.PROTOCOL_V1: "1" * 64,
                    gate.PROTOCOL_V2: "2" * 64, gate.RESULT_SCHEMA: "3" * 64}.get(path, real_sha(path))
        with patch.object(gate, "sha", side_effect=fake_sha):
            gate._case_receipts(0, self.attempts[0], gate.sha(self.freeze), self.fake_receipt)
            result["peak_cuda_reserved_bytes"] = gate.MAX_CUDA
            (out / "result.json").unlink()
            gate.put(out / "result.json", result)
            with self.assertRaisesRegex(ValueError, "cap/receipt"):
                gate._case_receipts(0, self.attempts[0], gate.sha(self.freeze), self.fake_receipt)

    def test_missing_freeze_remains_closed(self) -> None:
        with self.assertRaises((FileNotFoundError, ValueError)):
            gate.verify_freeze()


if __name__ == "__main__":
    unittest.main()
