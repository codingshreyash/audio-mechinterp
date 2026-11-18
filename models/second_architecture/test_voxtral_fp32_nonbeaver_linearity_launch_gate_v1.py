"""CPU-only checks for the four-entry, fail-closed non-beaver gate."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import voxtral_fp32_nonbeaver_linearity_launch_gate_v1 as gate


class FourEntryGateTests(unittest.TestCase):
    def test_fixed_four_context_scope_and_caps(self):
        self.assertEqual(tuple(gate.runner.CONTEXT_LABELS),
                         ("piper_silver", "piper_teal", "kokoro_silver", "kokoro_teal"))
        self.assertEqual(gate.MAX_OUTPUT_BYTES, 2 * 1024**3)
        self.assertEqual(gate.MAX_CAMPAIGN_BYTES, 8 * 1024**3)
        self.assertEqual(gate.MIN_FREE_BYTES_BEFORE_RESERVATION, 12 * 1024**3)
        self.assertEqual(gate.PROTOCOL_SHA,
                         "99de95003190de6b5fb55c5ba8bed446dfea7cdd8cf53c0569d304bcb942b49b")
        self.assertEqual(gate.runner.COUNTS["recipient_passive_captures"], 1)
        self.assertTrue(set(gate.runner.SOURCE_DEPENDENCIES).issubset(
            gate.REVIEWED_RUNNER_KEYS))

    def test_missing_external_reviews_hold_without_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = root / "review.md"
            with mock.patch.object(gate, "CONCEPTUAL_REVIEW", missing), \
                    mock.patch.dict(gate.BOUND, {"conceptual_review_sha256": missing}), \
                    mock.patch.object(gate, "_check_previous", return_value=368.4824181678705), \
                    mock.patch.object(gate, "RUNS", root / "launch"):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    gate.verify_freeze()
                self.assertFalse((root / "launch").exists())

    def test_reserve_all_records_exact_four_attempts_before_first_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            now = datetime.now(timezone.utc)
            frozen = {"reviewed_utc": (now - timedelta(days=1)).isoformat()}
            upstream = {"prior_model_work_seconds": 368.4824181678705}
            campaign = "20261118T170000000001Z"
            with mock.patch.object(gate, "RUNS", root / "launch"), \
                    mock.patch.object(gate.runner, "RUNS", root / "worker"), \
                    mock.patch.object(gate, "verify_freeze", return_value=(frozen, upstream)), \
                    mock.patch.object(gate, "sha", return_value="f" * 64), \
                    mock.patch.object(gate.shutil, "disk_usage",
                                      return_value=SimpleNamespace(free=13 * 1024**3)):
                saved = gate.reserve_all(campaign)
                self.assertEqual(saved["context_labels"], list(gate.runner.CONTEXT_LABELS))
                self.assertEqual(saved["attempt_ids"],
                                 [f"{campaign}_{i:02d}" for i in range(4)])
                self.assertEqual(json.loads((root / "launch/campaign_reservation.json").read_text()),
                                 saved)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    gate.reserve_all(campaign)

    def test_reservation_requires_free_disk_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = {"reviewed_utc": "2026-11-17T00:00:00+00:00"}
            upstream = {"prior_model_work_seconds": 368.4824181678705}
            with mock.patch.object(gate, "RUNS", root / "launch"), \
                    mock.patch.object(gate.runner, "RUNS", root / "worker"), \
                    mock.patch.object(gate, "verify_freeze", return_value=(frozen, upstream)), \
                    mock.patch.object(gate.shutil, "disk_usage",
                                      return_value=SimpleNamespace(free=11 * 1024**3)):
                with self.assertRaisesRegex(ValueError, "12 GiB"):
                    gate.reserve_all("20261118T170000000001Z")
            self.assertFalse((root / "launch/campaign_reservation.json").exists())

    def test_byte_budget_requires_v5_and_two_subcap_engine_rehearsals(self):
        budget = gate._json(gate.BYTE_BUDGET)
        gate.verify_byte_budget(budget, budget["runner_sha256"])
        old_cap = json.loads(json.dumps(budget))
        old_cap["max_output_bytes_per_context"] = 1024**3
        with self.assertRaisesRegex(ValueError, "byte budget"):
            gate.verify_byte_budget(old_cap, budget["runner_sha256"])
        over = json.loads(json.dumps(budget))
        over["measurements"][0]["serialized_bytes"] = gate.runner.MAX_WORKER_BYTES
        with self.assertRaisesRegex(ValueError, "byte budget"):
            gate.verify_byte_budget(over, budget["runner_sha256"])
        wrong_protocol = json.loads(json.dumps(budget))
        wrong_protocol["protocol_v5_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "byte budget"):
            gate.verify_byte_budget(wrong_protocol, budget["runner_sha256"])

    def test_prior_entry_must_finish_and_later_entry_must_not_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)]}
            with mock.patch.object(gate, "RUNS", root):
                with self.assertRaises(FileNotFoundError):
                    gate._check_prior_entries(1, reservation)
                prior = root / gate.runner.CONTEXT_LABELS[0] / reservation["attempt_ids"][0]
                gate._put(prior / "launch_finish.json", {
                    "status": gate.FINISH_FAILED, "context_index": 0,
                    "attempt_id": reservation["attempt_ids"][0],
                    "model_work_elapsed_seconds": 17.25,
                })
                self.assertEqual(gate._check_prior_entries(1, reservation), 17.25)
                later = root / gate.runner.CONTEXT_LABELS[2] / reservation["attempt_ids"][2]
                gate._put(later / "launch_start.json", {"status": "context_launch_started"})
                with self.assertRaisesRegex(ValueError, "out of order"):
                    gate._check_prior_entries(1, reservation)

    def test_prior_entry_elapsed_must_be_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)]}
            prior = root / gate.runner.CONTEXT_LABELS[0] / reservation["attempt_ids"][0]
            gate._put(prior / "launch_finish.json", {
                "status": gate.FINISH_FAILED, "context_index": 0,
                "attempt_id": reservation["attempt_ids"][0],
                "model_work_elapsed_seconds": 301,
            })
            with mock.patch.object(gate, "RUNS", root):
                with self.assertRaisesRegex(ValueError, "model-work seconds"):
                    gate._check_prior_entries(1, reservation)

    def test_prior_output_overshoot_blocks_next_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)]}
            prior = root / gate.runner.CONTEXT_LABELS[0] / reservation["attempt_ids"][0]
            gate._put(prior / "launch_finish.json", {
                "status": gate.FINISH_FAILED, "context_index": 0,
                "attempt_id": reservation["attempt_ids"][0],
                "model_work_elapsed_seconds": 1,
            })
            (prior / "large.bin").write_bytes(b"x" * 1024)
            with mock.patch.object(gate, "RUNS", root), \
                    mock.patch.object(gate, "MAX_OUTPUT_BYTES", 1000):
                with self.assertRaisesRegex(ValueError, "output cap"):
                    gate._check_prior_entries(1, reservation)

    def test_bad_id_rejected_before_worker_claim(self):
        with self.assertRaisesRegex(ValueError, "Attempt ID"):
            gate.verify_worker_start(0, "../../old-stage", "0" * 64)
        with self.assertRaisesRegex(ValueError, "Context index"):
            gate.verify_worker_start(4, "20261118T170000000001Z_04", "0" * 64)

    def test_parent_watchdog_times_out_child_process(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            capture.mkdir()
            code, elapsed, reason = gate._bounded(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                capture, ("out.txt", "err.txt"), Path(temp) / "output", 1)
            self.assertIsNotNone(code)
            self.assertLess(elapsed, 3)
            self.assertEqual(reason, "context_process_timeout")

    def test_output_overshoot_preserves_bounded_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / "capture"
            output = Path(temp) / "output"
            capture.mkdir()
            output.mkdir()
            (capture / "worker_stdout.txt").write_bytes(b"x" * 3000)
            (capture / "worker_stderr.txt").write_bytes(b"y" * 2000)
            (output / "partial.bin").write_bytes(b"z" * 500)
            finish = {"status": gate.FINISH_OK}
            with mock.patch.object(gate, "MAX_OUTPUT_BYTES", 6000), \
                    mock.patch.object(gate, "FINISH_RESERVE_BYTES", 512):
                gate._write_finish(capture, output, finish)
                self.assertLess(gate._tree_bytes(capture) + gate._tree_bytes(output), 6000)
            receipt = gate._json(capture / "launch_finish.json")
            self.assertEqual(receipt["status"], gate.FINISH_FAILED)
            self.assertEqual(receipt["failure_reason"], "context_combined_output_byte_cap")
            self.assertEqual(receipt["trimmed_streams"]["worker_stdout.txt"], {
                "original_bytes": 3000,
                "original_sha256": hashlib.sha256(b"x" * 3000).hexdigest(),
                "retained_prefix_bytes": 512,
            })


if __name__ == "__main__":
    unittest.main()
