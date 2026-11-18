"""CPU-only tests for immutable failure binding and fail-closed admission."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import voxtral_old_fit_grad_mode_parity_launch_gate_v1 as target


class GraphNoGradGateTests(unittest.TestCase):
    def test_failed_attempt_remains_failed_and_time_is_preserved(self):
        manifest, stage_a, prior = target.verify_failed_provenance()
        self.assertEqual(manifest["attempt_id"], target.FAILED_ID)
        self.assertEqual(manifest["status"],
                         "frozen_failed_before_saved_result_or_audit_no_retry")
        self.assertEqual(manifest["parent_model_work_elapsed_seconds"],
                         target.FAILED_SECONDS)
        self.assertEqual(prior, 246.84183400077745 + target.FAILED_SECONDS)
        self.assertEqual(stage_a["original_bf16_launch_status"], "launch_failed")

    def test_missing_review_fails_closed_even_after_future_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            absent = Path(temp) / "absent_review.md"
            with mock.patch.dict(target.BOUND, {"source_review_sha256": absent}):
                with mock.patch.object(target, "verify_failed_provenance",
                                       return_value=({}, {"original_bf16_launch_status": "launch_failed"}, 288.0)):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        target.verify_freeze()
        with mock.patch.object(target, "verify_freeze", side_effect=ValueError("freeze absent")):
            with self.assertRaisesRegex(ValueError, "freeze absent"):
                target.verify_worker_start("20261118T150000000000Z", "0" * 64)

    def test_no_worker_claim_for_bad_attempt_id(self):
        with self.assertRaisesRegex(ValueError, "attempt ID"):
            target.verify_worker_start("../../old-stage-b", "0" * 64)
        with self.assertRaisesRegex(ValueError, "attempt ID"):
            target.launch("../../old-stage-b")

    def test_bounded_process_group_timeout_and_stream_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            capture.mkdir()
            output = Path(temp) / "output"
            command = [sys.executable, "-c", "import time; time.sleep(10)"]
            code, elapsed, reason = target._bounded(
                command, capture, ("out.txt", "err.txt"), output, 1)
            self.assertIsNotNone(code)
            self.assertLess(elapsed, 3)
            self.assertEqual(reason, "diagnostic_process_timeout")
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            capture.mkdir()
            output = Path(temp) / "output"
            command = [sys.executable, "-c", "print('x'*4096, flush=True)"]
            with mock.patch.object(target, "MAX_CAPTURE_BYTES", 1024):
                code, _, reason = target._bounded(
                    command, capture, ("out.txt", "err.txt"), output, 10)
            self.assertIsNotNone(code)
            self.assertEqual(reason, "diagnostic_capture_byte_cap")

    def test_cap_edge_still_writes_bounded_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            capture.mkdir()
            output = Path(temp) / "output"
            output.mkdir()
            (capture / "worker_stdout.txt").write_bytes(b"x" * 3000)
            (capture / "worker_stderr.txt").write_bytes(b"y" * 2000)
            (output / "partial.bin").write_bytes(b"z" * 500)
            finish = {"status": "diagnostic_launch_completed_cpu_audited_only"}
            with mock.patch.object(target, "MAX_OUTPUT_BYTES", 6000), \
                    mock.patch.object(target, "FINISH_RESERVE_BYTES", 512):
                target._write_finish_preserving_cap(capture, output, finish)
                self.assertLess(target._tree_bytes(capture) + target._tree_bytes(output), 6000)
            self.assertEqual(target._json(capture / "launch_finish.json")["status"],
                             "diagnostic_launch_failed_no_retry")
            self.assertEqual(target._json(capture / "launch_finish.json")["failure_reason"],
                             "diagnostic_combined_output_byte_cap")
            receipt = target._json(capture / "launch_finish.json")
            self.assertEqual(receipt["trimmed_streams"]["worker_stdout.txt"], {
                "original_bytes": 3000,
                "original_sha256": hashlib.sha256(b"x" * 3000).hexdigest(),
                "retained_prefix_bytes": 512,
            })
            self.assertEqual((capture / "worker_stdout.txt").read_bytes(), b"x" * 512)


if __name__ == "__main__":
    unittest.main()
