"""CPU-only fail-closed tests for the two-slot continuation boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import plistlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as gate


class ContinuationGateTests(unittest.TestCase):
    def test_manifest_rejects_changed_raw_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launch, worker = root / "launch", root / "worker"
            paths = [launch / "campaign_reservation.json"]
            for index in (0, 1):
                label, attempt = gate.v1.runner.CONTEXT_LABELS[index], f"{gate.CAMPAIGN}_{index:02d}"
                paths.extend(launch / label / attempt / f"launch_{n}.json" for n in range(6))
                paths.extend(worker / label / attempt / f"worker_{n}.json" for n in range(7))
            self.assertEqual(len(paths), 27)
            rows = []
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(index))
                rows.append({"path": str(path.relative_to(root)),
                             "bytes": path.stat().st_size,
                             "sha256": gate.v1.sha(path)})
            digest = hashlib.sha256("".join(
                f"{row['sha256']}  {row['path']}\n" for row in rows).encode()).hexdigest()
            manifest = {"schema": "voxtral_fp32_nonbeaver_piper_evidence_manifest_v1",
                        "raw_file_count": 27, "raw_manifest_sha256": digest,
                        "freeze": {"sha256": gate.V1_FREEZE_SHA},
                        "failed_admission": {"receipt_sha256": gate.FAILED_SHA,
                                             "remaining_attempt_directories_absent": True},
                        "reserved_context_labels": list(gate.v1.runner.CONTEXT_LABELS),
                        "raw_files": rows}
            evidence = root / "manifest.json"
            evidence.write_text(json.dumps(manifest))
            with mock.patch.object(gate, "REPO", root), \
                    mock.patch.object(gate, "PIPER_MANIFEST", evidence), \
                    mock.patch.object(gate, "RAW_MANIFEST_SHA", digest), \
                    mock.patch.object(gate.v1, "RUNS", launch), \
                    mock.patch.object(gate.v1.runner, "RUNS", worker):
                self.assertEqual(gate._verify_manifest()["raw_file_count"], 27)
                paths[-1].write_text("tampered")
                with self.assertRaisesRegex(ValueError, "changed or absent"):
                    gate._verify_manifest()

    def test_fixed_scope_and_absent_external_freeze_hold(self):
        self.assertEqual(gate.CAMPAIGN, "20261118T182308460007Z")
        self.assertEqual(gate.PRIOR_WORK, (138.8538012499921, 151.86878254101612))
        self.assertFalse(gate.FREEZE.exists())
        self.assertFalse(gate.SOURCE_REVIEW.exists())
        with self.assertRaises((ValueError, FileNotFoundError)):
            gate.verify_freeze()

    def test_gap_arithmetic_and_narrow_exception(self):
        start = {"wall_utc": "2026-11-18T18:28:15.345155+00:00"}
        finish = {"wall_utc": "2026-11-18T18:32:10.910491+00:00",
                  "worker_started_utc": "2026-11-18T18:28:43.591852+00:00",
                  "worker_finished_utc": "2026-11-18T18:31:29.268246+00:00",
                  "elapsed_seconds": 210.4409978329204,
                  "model_work_elapsed_seconds": gate.PRIOR_WORK[1]}
        # Use the actual frozen receipt timestamps for the exact signed-gap check.
        prior = gate.v1.RUNS / "piper_teal" / f"{gate.CAMPAIGN}_01"
        actual_start = gate.v1._json(prior / "launch_start.json")
        actual_finish = gate.v1._json(prior / "launch_finish.json")
        launch_gap, worker_gap = gate._signed_gaps(actual_start, actual_finish)
        self.assertAlmostEqual(launch_gap, gate.EXCEPTION_LAUNCH_GAP, places=9)
        self.assertAlmostEqual(worker_gap, gate.EXCEPTION_WORKER_GAP, places=9)
        changed = copy.deepcopy(actual_finish)
        changed["elapsed_seconds"] += 0.01
        self.assertGreater(abs(gate._signed_gaps(actual_start, changed)[0] -
                               gate.EXCEPTION_LAUNCH_GAP), .001)
        changed["elapsed_seconds"] = actual_finish["elapsed_seconds"] + 40
        self.assertLess(gate._signed_gaps(actual_start, changed)[0], 0)
        changed = copy.deepcopy(actual_finish)
        changed["worker_finished_utc"] = actual_start["wall_utc"]
        with self.assertRaisesRegex(ValueError, "chronology"):
            gate._signed_gaps(actual_start, changed)
        self.assertEqual(start["wall_utc"][:10], actual_start["wall_utc"][:10])
        self.assertEqual(finish["elapsed_seconds"], actual_finish["elapsed_seconds"])

    def test_new_context_five_second_rule_checks_both_clocks(self):
        start = {"wall_utc": "2026-11-18T20:00:00+00:00"}
        finish = {"wall_utc": "2026-11-18T20:00:20+00:00",
                  "worker_started_utc": "2026-11-18T20:00:01+00:00",
                  "worker_finished_utc": "2026-11-18T20:00:11+00:00",
                  "elapsed_seconds": 20.0, "model_work_elapsed_seconds": 10.0}
        gate._require_strict_timing(start, finish)
        launch_bad = {**finish, "elapsed_seconds": 14.99}
        with self.assertRaisesRegex(ValueError, "five-second"):
            gate._require_strict_timing(start, launch_bad)
        worker_bad = {**finish, "model_work_elapsed_seconds": 4.99}
        with self.assertRaisesRegex(ValueError, "five-second"):
            gate._require_strict_timing(start, worker_bad)

    def test_future_attempt_artifact_rejects_out_of_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reservation = gate.v1._json(gate.v1.RUNS / "campaign_reservation.json")
            later = root / "launch" / gate.v1.runner.CONTEXT_LABELS[3] / reservation["attempt_ids"][3]
            later.mkdir(parents=True)
            with mock.patch.object(gate, "RUNS", root / "launch"), \
                    mock.patch.object(gate.v1.runner, "RUNS", root / "worker"), \
                    mock.patch.object(gate.v1, "_check_prior_entries", return_value=gate.PRIOR_WORK[0]), \
                    mock.patch.object(gate, "_check_piper_one",
                                      return_value=(gate.PRIOR_WORK[1],
                                                    datetime(2026, 11, 18, tzinfo=timezone.utc))):
                with self.assertRaisesRegex(ValueError, "out of order"):
                    gate.check_prior(2, reservation)

    def test_power_event_parser_excludes_wake_requests(self):
        log = ("2026-11-18 14:00:00 -0500 Wake Requests\tMaintenance\n"
               "2026-11-18 14:01:00 -0500 Sleep\tThermal Emergency Sleep\n"
               "2026-11-18 14:02:00 -0500 DarkWake\tDarkWake from Deep Idle\n")
        start = datetime(2026, 11, 18, 19, 0, tzinfo=timezone.utc)
        found = gate._power_events(log, start, start + timedelta(minutes=3))
        self.assertEqual(len(found), 2)
        self.assertIn("Sleep", found[0])
        self.assertIn("DarkWake", found[1])

    def test_power_log_requires_old_event_and_recent_terminal_snapshot(self):
        collected = datetime(2026, 11, 18, 19, 20, tzinfo=timezone.utc)
        old = "2026-11-18 14:00:00 -0500 Assertions\told event\n"
        recent = "2026-11-18 14:19:59 -0500 Assertions\trecent event\n"
        terminal = ("2026-11-18 14:20:00 -0500 : Showing all currently held "
                    "IOKit power assertions\n")
        gate._power_log_coverage(old + terminal, collected - timedelta(minutes=10),
                                 collected, collected)
        with self.assertRaisesRegex(ValueError, "coverage"):
            gate._power_log_coverage(old + recent, collected - timedelta(minutes=10),
                                     collected, collected)
        stale_terminal = ("2026-11-18 14:05:00 -0500 : Showing all currently held "
                          "IOKit power assertions\n")
        with self.assertRaisesRegex(ValueError, "coverage"):
            gate._power_log_coverage(old + stale_terminal,
                                     collected - timedelta(minutes=10), collected, collected)
        with self.assertRaisesRegex(ValueError, "coverage"):
            gate._power_log_coverage(recent + terminal,
                                     collected - timedelta(minutes=10), collected, collected)

    def test_host_probe_rejects_battery_closed_lid_and_recent_sleep(self):
        now = datetime.now(timezone.utc)
        recent = now.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        old = (now - timedelta(days=1)).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        terminal = f"{recent} : Showing all currently held IOKit power assertions\n"
        quiet_log = f"{old} Sleep\tOld event\n" + terminal
        event_log = f"{old} Sleep\tOld event\n{recent} DarkWake\tRecent event\n" + terminal

        def values(*, ac=True, open_lid=True, log=quiet_log):
            return iter(("Now drawing from 'AC Power'\n" if ac else
                         "Now drawing from 'Battery Power'\n",
                         plistlib.dumps([{"AppleClamshellState": not open_lid}]).decode(),
                         plistlib.dumps([{"ExternalConnected": ac}]).decode(), log))

        for kwargs, message in (({"ac": False}, "external power"),
                                ({"open_lid": False}, "clamshell"),
                                ({"log": event_log}, "recent sleep")):
            iterator = values(**kwargs)
            with mock.patch.object(gate, "_run_command", side_effect=lambda _argv: next(iterator)):
                with self.assertRaisesRegex(ValueError, message):
                    gate._host_probe()
        iterator = values()
        with mock.patch.object(gate, "_run_command", side_effect=lambda _argv: next(iterator)):
            self.assertIn("AC Power", gate._host_probe()["outputs"]["power"])

    def test_power_log_archive_exact_bytes_and_post_run_transition(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp)
            start_at = datetime(2026, 11, 18, 19, 0, tzinfo=timezone.utc)
            log = ("2026-11-17 14:00:00 -0500 Sleep\told event\n"
                   "2026-11-18 14:01:00 -0500 DarkWake\ttransition\n"
                   "2026-11-18 14:02:00 -0500 : Showing all currently held IOKit power assertions\n")
            probe = {"collected_utc": (start_at + timedelta(minutes=2)).isoformat(),
                     "commands": gate.HOST_COMMANDS,
                     "outputs": {"power_log": log,
                                 "power": "Now drawing from 'AC Power'\n",
                                 "battery": plistlib.dumps([{"ExternalConnected": True}]).decode(),
                                 "clamshell": plistlib.dumps([{"AppleClamshellState": False}]).decode()}}
            gate._store_host_probe(capture, probe, "post")
            self.assertEqual(gate._load_power_log(capture, probe), log)
            start = {"wall_utc": start_at.isoformat()}
            finish = {"wall_utc": (start_at + timedelta(minutes=2)).isoformat(),
                      "post_host": probe}
            with self.assertRaisesRegex(ValueError, "slept"):
                gate._check_post_host_record(capture, start, finish)
            archive = capture / "post_power_log.txt.gz"
            archive.write_bytes(archive.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "archive changed"):
                gate._load_power_log(capture, probe)

    def test_no_retry_claim_is_atomic_and_same_attempt_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = root / "continuation.json"
            freeze.write_text("{}")
            reservation = gate.v1._json(gate.v1.RUNS / "campaign_reservation.json")
            index = 2
            attempt = reservation["attempt_ids"][index]
            label = gate.v1.runner.CONTEXT_LABELS[index]
            capture = root / "launch" / label / attempt
            output = root / "worker" / label / attempt
            continuation_sha = gate.v1.sha(freeze)
            command = [gate.sys.executable, str(gate.WORKER.resolve()),
                       "--run-context", "2", "--attempt-id", attempt,
                       "--continuation-freeze-sha256", continuation_sha]
            gate.v1._put(capture / "launch_start.json", {
                "wall_utc": "2026-11-18T20:00:00+00:00",
                "worker_command": command, "worker_output": str(output),
                "continuation_freeze_sha256": continuation_sha,
                "entrypoint_sha256": gate.v1.sha(gate.WORKER),
                "freeze_sha256": gate.V1_FREEZE_SHA,
                "cumulative_prior_model_work_seconds": 659.2050019588787,
                "prior_model_work_seconds": 368.4824181678705,
                "context_index": 2, "context_label": label, "attempt_id": attempt,
                "planned_counts": gate.v1.runner.COUNTS, "process_cap_seconds": 300,
                "max_mps_driver_bytes": 30 * 1024**3,
                "max_output_bytes": 2 * 1024**3,
                "max_worker_bytes": gate.v1.runner.MAX_WORKER_BYTES,
                "max_capture_bytes": 8 * 1024**2})
            frozen = {"reviewed_utc": "2026-11-18T19:00:00+00:00"}
            upstream = {"prior_model_work_seconds": 368.4824181678705}
            with mock.patch.object(gate, "RUNS", root / "launch"), \
                    mock.patch.object(gate, "FREEZE", freeze), \
                    mock.patch.object(gate.v1.runner, "RUNS", root / "worker"), \
                    mock.patch.object(gate, "verify_freeze", return_value=(frozen, upstream, reservation)), \
                    mock.patch.object(gate, "check_prior", return_value=sum(gate.PRIOR_WORK)), \
                    mock.patch.object(gate, "_check_pre_host_record"):
                with self.assertRaisesRegex(ValueError, "admission failed"):
                    gate.verify_worker_start(3, reservation["attempt_ids"][3], continuation_sha)
                self.assertFalse((capture / "worker_claim.json").exists())
                self.assertEqual(gate.verify_worker_start(index, attempt, continuation_sha), upstream)
                self.assertTrue((capture / "worker_claim.json").exists())
                with self.assertRaisesRegex(ValueError, "admission failed"):
                    gate.verify_worker_start(index, attempt, continuation_sha)


if __name__ == "__main__":
    unittest.main()
