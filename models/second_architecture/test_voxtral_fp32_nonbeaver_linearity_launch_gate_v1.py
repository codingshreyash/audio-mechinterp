"""CPU-only checks for the four-entry, fail-closed non-beaver gate."""

from __future__ import annotations

import json
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
                         "a222ce0f06a0e82b2c22bf6231e605983d726610b2f51dfbe7299de3ea8d79bd")

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
                })
                gate._check_prior_entries(1, reservation)
                later = root / gate.runner.CONTEXT_LABELS[2] / reservation["attempt_ids"][2]
                gate._put(later / "launch_start.json", {"status": "context_launch_started"})
                with self.assertRaisesRegex(ValueError, "out of order"):
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


if __name__ == "__main__":
    unittest.main()
