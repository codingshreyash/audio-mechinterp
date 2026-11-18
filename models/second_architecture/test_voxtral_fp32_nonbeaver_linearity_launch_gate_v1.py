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
    def _write_prior_failure(self, root: Path, reservation: dict,
                             *, elapsed: float = 17.25) -> Path:
        label = gate.runner.CONTEXT_LABELS[0]
        attempt = reservation["attempt_ids"][0]
        capture = root / label / attempt
        output = root / "worker" / label / attempt
        start_at = datetime(2026, 11, 18, 17, 0, 1, tzinfo=timezone.utc)
        worker_done_at = start_at + timedelta(seconds=elapsed)
        finish_at = worker_done_at + timedelta(seconds=1)
        gate._put(capture / "launch_start.json", {
            "status": "context_launch_started", "context_index": 0,
            "context_label": label, "attempt_id": attempt,
            "freeze_sha256": reservation["freeze_sha256"],
            "wall_utc": start_at.isoformat(),
            "worker_command": [sys.executable, str(Path(gate.runner.__file__).resolve()),
                               "--run-context", "0", "--attempt-id", attempt,
                               "--freeze-sha256", reservation["freeze_sha256"]],
            "worker_output": str(output),
            "prior_model_work_seconds": reservation["prior_model_work_seconds"],
            "cumulative_prior_model_work_seconds": reservation["prior_model_work_seconds"],
            "process_cap_seconds": gate.runner.MODEL_SECONDS,
            "max_mps_driver_bytes": gate.runner.MAX_DRIVER_BYTES,
            "max_output_bytes": gate.MAX_OUTPUT_BYTES,
            "max_worker_bytes": gate.runner.MAX_WORKER_BYTES,
            "max_capture_bytes": gate.MAX_CAPTURE_BYTES,
            "planned_counts": gate.runner.COUNTS,
        })
        gate._put(capture / "launch_finish.json", {
            "status": gate.FINISH_FAILED, "context_index": 0,
            "context_label": label, "attempt_id": attempt,
            "freeze_sha256": reservation["freeze_sha256"],
            "wall_utc": finish_at.isoformat(),
            "worker_started_utc": start_at.isoformat(),
            "worker_finished_utc": worker_done_at.isoformat(),
            "model_work_elapsed_seconds": elapsed,
            "elapsed_seconds": elapsed + 1,
            "prior_model_work_seconds": reservation["prior_model_work_seconds"],
            "cumulative_prior_model_work_seconds": reservation["prior_model_work_seconds"],
            "worker_returncode": None, "auditor_returncode": None,
            "post_run_source_verified": False,
            "result_sha256": None, "audit_sha256": None,
            "error_sha256": None, "failure_reason": "worker failed before claim",
            "worker_output_bytes": 0,
            "launcher_output_bytes_before_finish":
                (capture / "launch_start.json").stat().st_size,
            "combined_output_bytes_before_finish":
                (capture / "launch_start.json").stat().st_size,
        })
        return capture

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

    def test_reservation_rejects_false_prior_time_and_stale_utc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            baseline = {
                "status": gate.CAMPAIGN_STATUS, "campaign_id": campaign,
                "attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)],
                "context_labels": list(gate.runner.CONTEXT_LABELS),
                "freeze_sha256": "f" * 64,
                "wall_utc": "2026-11-18T17:00:00+00:00",
                "prior_model_work_seconds": 368.4824181678705,
            }
            frozen = {"reviewed_utc": "2026-11-18T16:00:00+00:00"}
            upstream = {"prior_model_work_seconds": 368.4824181678705}
            with mock.patch.object(gate, "RUNS", root), \
                    mock.patch.object(gate, "sha", return_value="f" * 64):
                path = root / "campaign_reservation.json"
                gate._put(path, {**baseline, "prior_model_work_seconds": -999999.0})
                with self.assertRaisesRegex(ValueError, "prior model-work"):
                    gate._reservation(frozen, upstream)
                path.unlink()
                gate._put(path, {**baseline, "wall_utc": "2020-01-01T00:00:00+00:00"})
                with self.assertRaisesRegex(ValueError, "reservation differs"):
                    gate._reservation(frozen, upstream)

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
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)],
                           "prior_model_work_seconds": 368.4824181678705,
                           "freeze_sha256": "f" * 64,
                           "wall_utc": "2026-11-18T17:00:00+00:00"}
            with mock.patch.object(gate, "RUNS", root), \
                    mock.patch.object(gate.runner, "RUNS", root / "worker"):
                with self.assertRaises(FileNotFoundError):
                    gate._check_prior_entries(1, reservation)
                self._write_prior_failure(root, reservation)
                self.assertEqual(gate._check_prior_entries(1, reservation), 17.25)
                later = root / gate.runner.CONTEXT_LABELS[2] / reservation["attempt_ids"][2]
                gate._put(later / "launch_start.json", {"status": "context_launch_started"})
                with self.assertRaisesRegex(ValueError, "out of order"):
                    gate._check_prior_entries(1, reservation)

    def test_prior_entry_elapsed_must_be_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)],
                           "prior_model_work_seconds": 368.4824181678705,
                           "freeze_sha256": "f" * 64,
                           "wall_utc": "2026-11-18T17:00:00+00:00"}
            with mock.patch.object(gate, "RUNS", root), \
                    mock.patch.object(gate.runner, "RUNS", root / "worker"):
                prior = self._write_prior_failure(root, reservation, elapsed=301)
                with self.assertRaisesRegex(ValueError, "model-work seconds"):
                    gate._check_prior_entries(1, reservation)

    def test_minimal_success_and_mutated_elapsed_cannot_admit_next_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)],
                           "prior_model_work_seconds": 368.4824181678705,
                           "freeze_sha256": "f" * 64,
                           "wall_utc": "2026-11-18T17:00:00+00:00"}
            prior = root / gate.runner.CONTEXT_LABELS[0] / reservation["attempt_ids"][0]
            with mock.patch.object(gate, "RUNS", root), \
                    mock.patch.object(gate.runner, "RUNS", root / "worker"):
                gate._put(prior / "launch_finish.json", {
                    "status": gate.FINISH_OK, "context_index": 0,
                    "attempt_id": reservation["attempt_ids"][0],
                    "model_work_elapsed_seconds": 0,
                })
                with self.assertRaises((ValueError, FileNotFoundError)):
                    gate._check_prior_entries(1, reservation)
                (prior / "launch_finish.json").unlink()
                self._write_prior_failure(root, reservation)
                saved = gate._json(prior / "launch_finish.json")
                saved["model_work_elapsed_seconds"] = 0
                (prior / "launch_finish.json").write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, "parent-measured"):
                    gate._check_prior_entries(1, reservation)
                saved["model_work_elapsed_seconds"] = 25
                (prior / "launch_finish.json").write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, "model-work seconds"):
                    gate._check_prior_entries(1, reservation)

    def test_completed_prior_requires_claim_and_matching_result_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)],
                           "prior_model_work_seconds": 368.4824181678705,
                           "freeze_sha256": "f" * 64,
                           "wall_utc": "2026-11-18T17:00:00+00:00"}
            with mock.patch.object(gate, "RUNS", root), \
                    mock.patch.object(gate.runner, "RUNS", root / "worker"):
                capture = self._write_prior_failure(root, reservation, elapsed=8)
                attempt = reservation["attempt_ids"][0]
                label = gate.runner.CONTEXT_LABELS[0]
                output = root / "worker" / label / attempt
                context = {"index": 0, "label": label, "engine": "piper",
                           "recipient": "silver", "donor": "teal"}
                gate._put(capture / "worker_claim.json", {
                    "status": "context_worker_claimed_no_retry", "context_index": 0,
                    "context_label": label, "attempt_id": attempt,
                    "freeze_sha256": reservation["freeze_sha256"],
                    "wall_utc": "2026-11-18T17:00:02+00:00",
                })
                (capture / "worker_stdout.txt").write_bytes(b"")
                (capture / "worker_stderr.txt").write_bytes(b"")
                gate._put(output / "result.json", {
                    "status": "completed_fp32_nonbeaver_local_linearity_raw_only",
                    "context": context, "attempt_id": attempt,
                    "freeze_sha256": reservation["freeze_sha256"],
                    "actual_counts": gate.runner.COUNTS, "elapsed_seconds": 6,
                })
                result_sha = gate.sha(output / "result.json")
                gate._put(output / "audit.json", {
                    "status": "audited_voxtral_fp32_nonbeaver_saved_data_only",
                    "context": context, "attempt_id": attempt,
                    "freeze_sha256": reservation["freeze_sha256"],
                    "result_sha256": result_sha,
                    "preflight_receipt_sha256": gate.sha(gate.CPU_PREFLIGHT),
                    "protocol_v5_sha256": gate.PROTOCOL_SHA,
                    "worker_invalidity_disagreement": False,
                })
                finish_path = capture / "launch_finish.json"
                saved = gate._json(finish_path)
                saved.update({
                    "status": gate.FINISH_OK, "worker_returncode": 0,
                    "auditor_returncode": 0, "post_run_source_verified": True,
                    "result_sha256": result_sha,
                    "audit_sha256": gate.sha(output / "audit.json"),
                    "worker_output_bytes": gate._tree_bytes(output),
                })
                saved.pop("failure_reason")
                saved["combined_output_bytes_before_finish"] = (
                    gate._tree_bytes(capture) - finish_path.stat().st_size +
                    gate._tree_bytes(output))
                finish_path.write_text(json.dumps(saved))
                self.assertEqual(gate._check_prior_entries(1, reservation), 8)
                (capture / "worker_claim.json").unlink()
                with self.assertRaisesRegex(ValueError, "claim absent"):
                    gate._check_prior_entries(1, reservation)

    def test_prior_output_overshoot_blocks_next_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = "20261118T170000000001Z"
            reservation = {"attempt_ids": [f"{campaign}_{i:02d}" for i in range(4)],
                           "prior_model_work_seconds": 368.4824181678705,
                           "freeze_sha256": "f" * 64,
                           "wall_utc": "2026-11-18T17:00:00+00:00"}
            with mock.patch.object(gate, "RUNS", root), \
                    mock.patch.object(gate.runner, "RUNS", root / "worker"), \
                    mock.patch.object(gate, "MAX_OUTPUT_BYTES", 1000):
                prior = self._write_prior_failure(root, reservation, elapsed=1)
                (prior / "large.bin").write_bytes(b"x" * 1024)
                saved = gate._json(prior / "launch_finish.json")
                saved["combined_output_bytes_before_finish"] += 1024
                (prior / "launch_finish.json").write_text(json.dumps(saved))
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
