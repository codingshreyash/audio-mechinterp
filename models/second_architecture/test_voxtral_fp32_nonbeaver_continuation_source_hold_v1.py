"""Source-only reproductions of continuation gate review blockers.

These reproductions remain as regression tests for the corrected source gate.
"""

from __future__ import annotations

import plistlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as gate


class ContinuationSourceHoldTests(unittest.TestCase):
    def test_preflight_rejects_stale_power_log_without_recent_coverage(self):
        old = (datetime.now(timezone.utc) - timedelta(days=1)).astimezone()
        stale_log = old.strftime("%Y-%m-%d %H:%M:%S %z") + " Assertions\tOld event\n"
        outputs = iter((
            "Now drawing from 'AC Power'\n",
            plistlib.dumps([{"AppleClamshellState": False}]).decode(),
            plistlib.dumps([{"ExternalConnected": True}]).decode(),
            stale_log,
        ))
        with mock.patch.object(gate, "_run_command", side_effect=lambda _argv: next(outputs)):
            with self.assertRaisesRegex(ValueError, "coverage|stale|recent"):
                gate._host_probe()

    def test_saved_preflight_rejects_stale_power_log_without_recent_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp)
            now = datetime.now(timezone.utc)
            old = (now - timedelta(days=1)).astimezone()
            stale_log = old.strftime("%Y-%m-%d %H:%M:%S %z") + " Assertions\tOld event\n"
            probe = {
                "collected_utc": now.isoformat(),
                "commands": gate.HOST_COMMANDS,
                "outputs": {
                    "power": "Now drawing from 'AC Power'\n",
                    "clamshell": plistlib.dumps([{"AppleClamshellState": False}]).decode(),
                    "battery": plistlib.dumps([{"ExternalConnected": True}]).decode(),
                    "power_log": stale_log,
                },
            }
            gate._store_host_probe(capture, probe, "pre")
            start = {"wall_utc": (now + timedelta(seconds=1)).isoformat(), "pre_host": probe}
            with self.assertRaisesRegex(ValueError, "coverage|stale|recent"):
                gate._check_pre_host_record(capture, start)

    def test_saved_postrun_rejects_log_truncated_before_run(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp)
            now = datetime.now(timezone.utc)
            old = (now - timedelta(days=1)).astimezone()
            stale_log = old.strftime("%Y-%m-%d %H:%M:%S %z") + " Assertions\tOld event\n"
            probe = {
                "collected_utc": (now + timedelta(seconds=2)).isoformat(),
                "commands": gate.HOST_COMMANDS,
                "outputs": {
                    "power": "Now drawing from 'AC Power'\n",
                    "clamshell": plistlib.dumps([{"AppleClamshellState": False}]).decode(),
                    "battery": plistlib.dumps([{"ExternalConnected": True}]).decode(),
                    "power_log": stale_log,
                },
            }
            gate._store_host_probe(capture, probe, "post")
            start = {"wall_utc": (now - timedelta(seconds=10)).isoformat()}
            finish = {"wall_utc": now.isoformat(), "post_host": probe}
            with self.assertRaisesRegex(ValueError, "coverage|stale|recent"):
                gate._check_post_host_record(capture, start, finish)

    def test_completed_slot_rejects_start_before_actual_predecessor_finish(self):
        self._assert_completed_slot_rejects_backdated_start(2)

    def test_final_slot_rejects_start_before_actual_predecessor_finish(self):
        self._assert_completed_slot_rejects_backdated_start(3)

    def _assert_completed_slot_rejects_backdated_start(self, index: int):
        reservation = gate.v1._json(gate.v1.RUNS / "campaign_reservation.json")
        label = gate.v1.runner.CONTEXT_LABELS[index]
        attempt = reservation["attempt_ids"][index]
        start_at = datetime(2026, 11, 18, 19, 59, tzinfo=timezone.utc)
        predecessor_finish = start_at + timedelta(minutes=1)
        fixed = "f" * 64
        start = {
            "status": "context_launch_started", "context_index": index,
            "context_label": label, "attempt_id": attempt,
            "freeze_sha256": gate.V1_FREEZE_SHA,
            "continuation_freeze_sha256": fixed, "entrypoint_sha256": fixed,
            "worker_command": [sys.executable, str(gate.WORKER.resolve()),
                               "--run-context", str(index), "--attempt-id", attempt,
                               "--continuation-freeze-sha256", fixed],
            "worker_output": str(gate.v1.runner.RUNS / label / attempt),
            "prior_model_work_seconds": reservation["prior_model_work_seconds"],
            "cumulative_prior_model_work_seconds": 659.2050019588787,
            "process_cap_seconds": 300, "max_mps_driver_bytes": 30 * 1024**3,
            "max_output_bytes": 2 * 1024**3,
            "max_worker_bytes": gate.v1.runner.MAX_WORKER_BYTES,
            "max_capture_bytes": 8 * 1024**2,
            "planned_counts": gate.v1.runner.COUNTS,
            "wall_utc": start_at.isoformat(),
        }
        finish = {
            "status": gate.v1.FINISH_OK, "context_index": index,
            "context_label": label, "attempt_id": attempt,
            "freeze_sha256": gate.V1_FREEZE_SHA,
            "continuation_freeze_sha256": fixed, "entrypoint_sha256": fixed,
            "prior_model_work_seconds": reservation["prior_model_work_seconds"],
            "cumulative_prior_model_work_seconds": 659.2050019588787,
            "worker_returncode": 0, "auditor_returncode": 0,
            "post_run_source_verified": True, "error_sha256": None,
            "result_sha256": fixed, "audit_sha256": fixed,
            "worker_output_bytes": 0,
            "worker_started_utc": (start_at + timedelta(seconds=1)).isoformat(),
            "worker_finished_utc": (start_at + timedelta(seconds=10)).isoformat(),
            "wall_utc": (start_at + timedelta(seconds=20)).isoformat(),
            "elapsed_seconds": 20.0, "model_work_elapsed_seconds": 9.0,
        }
        claim = {
            "status": "context_worker_claimed_no_retry", "context_index": index,
            "context_label": label, "attempt_id": attempt,
            "freeze_sha256": gate.V1_FREEZE_SHA,
            "continuation_freeze_sha256": fixed, "entrypoint_sha256": fixed,
            "wall_utc": (start_at + timedelta(seconds=2)).isoformat(),
        }
        result = {
            "status": gate.v1.runner.STATUS, "attempt_id": attempt,
            "freeze_sha256": gate.V1_FREEZE_SHA,
            "context": {"index": index, "label": label},
            "actual_counts": gate.v1.runner.COUNTS,
        }
        audit = {
            "status": "audited_voxtral_fp32_nonbeaver_saved_data_only",
            "attempt_id": attempt, "freeze_sha256": gate.V1_FREEZE_SHA,
            "result_sha256": fixed, "worker_invalidity_disagreement": False,
        }

        def raw(path):
            if path.name == "launch_finish.json":
                return {"status": gate.v1.FINISH_OK,
                        "context_index": index - 1,
                        "context_label": gate.v1.runner.CONTEXT_LABELS[index - 1],
                        "attempt_id": reservation["attempt_ids"][index - 1],
                        "wall_utc": predecessor_finish.isoformat()}
            return {"launch_start.json": start, "worker_claim.json": claim,
                    "result.json": result, "audit.json": audit}[path.name]

        with mock.patch.object(gate.v1, "_json", side_effect=raw), \
                mock.patch.object(gate.v1, "sha", return_value=fixed), \
                mock.patch.object(gate.v1, "_tree_bytes", return_value=0), \
                mock.patch.object(gate, "_check_pre_host_record"), \
                mock.patch.object(gate, "_check_post_host_record"), \
                mock.patch.object(gate.v1, "_success_files"), \
                mock.patch.object(gate.v1, "_rerun_saved_prior_audit"):
            with self.assertRaisesRegex(ValueError, "start receipt differs"):
                gate._check_new_completed(index, reservation, 659.2050019588787,
                                          predecessor_finish, preview=finish)
            # The launcher's synthetic `start - 1us` value must not turn the
            # same backdated start into a valid completed slot.
            with self.assertRaisesRegex(ValueError, "start receipt differs"):
                gate._check_new_completed(index, reservation, 659.2050019588787,
                                          start_at - timedelta(microseconds=1), preview=finish)


if __name__ == "__main__":
    unittest.main()
