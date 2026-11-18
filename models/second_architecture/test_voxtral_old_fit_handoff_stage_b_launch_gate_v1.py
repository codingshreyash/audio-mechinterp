"""CPU-only provenance and no-retry checks for the Stage-B external launcher."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import voxtral_old_fit_handoff_stage_b_launch_gate_v1 as gate
import voxtral_old_fit_handoff_stage_b_audit_adapter_v1 as adapter


class GitFixture(unittest.TestCase):
    def git(self, repo: Path, *args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                text=True, check=True)
        return result.stdout.strip()

    def commit(self, repo: Path, message: str) -> str:
        self.git(repo, "add", ".")
        self.git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-q", "-m", message)
        return self.git(repo, "rev-parse", "HEAD")

    def test_first_add_rejects_edit_and_same_byte_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.git(repo, "init", "-q")
            receipt = repo / "freeze.json"
            copy = repo / "copy.json"
            receipt.write_text('{"reviewed_utc":"2026-11-18T14:00:00+00:00"}\n')
            copy.write_bytes(receipt.read_bytes())
            first = self.commit(repo, "external freeze")
            self.assertEqual(gate.first_add(repo, receipt), first)
            receipt.unlink()
            receipt.symlink_to(copy.name)
            self.assertIn("T freeze.json", self.git(repo, "status", "--short"))
            with self.assertRaisesRegex(ValueError, "contains symlink"):
                gate.first_add(repo, receipt)
            receipt.unlink()
            receipt.write_text('{"reviewed_utc":"2026-11-18T14:00:01+00:00"}\n')
            self.commit(repo, "edit freeze timestamp")
            with self.assertRaisesRegex(ValueError, "changed after first-add"):
                gate.first_add(repo, receipt)

    def test_symlinked_parent_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.git(repo, "init", "-q")
            (repo / "actual").mkdir()
            (repo / "actual" / "x.txt").write_text("same bytes\n")
            self.commit(repo, "actual evidence")
            (repo / "link").symlink_to("actual", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains symlink"):
                gate.regular(repo, repo / "link" / "x.txt")


class GateTest(unittest.TestCase):
    def test_real_stage_a_has_failed_bf16_and_single_fp32(self) -> None:
        stage_a = gate.same_stage_a()
        self.assertEqual(stage_a["original_bf16_launch_status"], "launch_failed")
        self.assertEqual(set(stage_a["arms"]), {"bf16", "fp32"})
        self.assertAlmostEqual(stage_a["stage_a_model_work_seconds"], 218.5633733337745)

    def test_source_scaffold_cannot_open_without_external_freeze(self) -> None:
        with mock.patch.object(gate, "_bounded") as child:
            with self.assertRaises(ValueError):
                gate.admission("bf16_direct_lm", "20261118T140000000000Z")
            child.assert_not_called()

    def test_invalid_attempt_id_rejected_before_freeze(self) -> None:
        with mock.patch.object(gate, "verify_phase_freeze") as frozen:
            with self.assertRaisesRegex(ValueError, "attempt ID invalid"):
                gate.admission("bf16_direct_lm", "../retry")
            frozen.assert_not_called()

    def test_bf16_no_retry_reservation_precedes_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bf16_direct_lm_reservation.json").write_text("{}\n")
            freeze = root / "freeze.json"
            freeze.write_text("{}\n")
            stage_a = {"stage_a_model_work_seconds": 218.5633733337745,
                       "arms": {"bf16": {"attempt": root / "a"},
                                "fp32": {"attempt": root / "b"}}}
            with (mock.patch.object(gate, "RUNS", root),
                  mock.patch.object(gate, "FREEZE", freeze),
                  mock.patch.object(gate.driver, "RUNS", root / "driver"),
                  mock.patch.object(gate, "verify_phase_freeze", return_value=({}, stage_a, None)),
                  mock.patch.object(gate, "_bounded") as child):
                with self.assertRaisesRegex(ValueError, "already reserved"):
                    gate.admission("bf16_direct_lm", "20261118T140000000000Z")
                child.assert_not_called()

    def test_worker_rejects_mismatched_reservation_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase, attempt_id = "bf16_direct_lm", "20261118T140000000000Z"
            freeze = root / "freeze.json"
            freeze.write_text("{}\n")
            attempt = root / phase / attempt_id
            attempt.mkdir(parents=True)
            worker_output = root / "driver" / phase / attempt_id
            stage_a = {"stage_a_model_work_seconds": 218.5633733337745}
            start = {"status": "launch_started", "phase": phase,
                     "attempt_id": attempt_id, "wall_utc": "2026-11-18T14:01:00+00:00",
                     "stage_b_freeze_sha256": gate.sha(freeze),
                     "prior_model_work_seconds": stage_a["stage_a_model_work_seconds"],
                     "worker_output": str(worker_output)}
            (attempt / "launch_start.json").write_text(json.dumps(start))
            (root / f"{phase}_reservation.json").write_text(json.dumps({
                "status": "phase_reserved_no_retry", "phase": phase,
                "attempt_id": "different-attempt", "stage_b_freeze_sha256": gate.sha(freeze)}))
            with (mock.patch.object(gate, "RUNS", root),
                  mock.patch.object(gate, "FREEZE", freeze),
                  mock.patch.object(gate.driver, "RUNS", root / "driver"),
                  mock.patch.object(gate, "verify_phase_freeze", return_value=(
                      {"reviewed_utc": "2026-11-18T14:00:00+00:00"}, stage_a, None))):
                with self.assertRaisesRegex(ValueError, "reservation/start receipt invalid"):
                    gate.verify_worker_start(phase, attempt_id, gate.sha(freeze))

    def test_fp32_freeze_requires_bf16_result_before_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            freeze = Path(tmp) / "freeze.json"
            freeze.write_text("{}\n")
            with (mock.patch.object(gate, "FREEZE", freeze),
                  mock.patch.object(gate, "verify_freeze", return_value=({}, {})),
                  mock.patch.object(gate, "_prior_bf16_b", side_effect=ValueError("BF16 review absent")),
                  mock.patch.object(gate, "verify_fp32_freeze") as fp32_freeze):
                with self.assertRaisesRegex(ValueError, "BF16 review absent"):
                    gate.verify_phase_freeze("fp32_replay")
                fp32_freeze.assert_not_called()

    def test_dangling_symlink_in_phase_tree_blocks_no_retry_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "bf16_direct_lm"
            folder.mkdir()
            (folder / "ghost").symlink_to("missing")
            with self.assertRaisesRegex(ValueError, "unexpected file or symlink"):
                gate._dirs(folder)

    def test_audit_adapter_requires_parent_gate_before_loading_tensors(self) -> None:
        with (mock.patch.object(gate, "verify_audit_inputs", side_effect=ValueError("start absent")),
              mock.patch.object(adapter.auditor, "verify_bf16_posthoc") as tensors):
            with self.assertRaisesRegex(ValueError, "start absent"):
                adapter.audit_existing("bf16_direct_lm", "20261118T140000000000Z",
                                       "0" * 64, 218.5633733337745)
            tensors.assert_not_called()

    def test_parent_watchdog_terminates_timed_out_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "launch"
            capture.mkdir()
            code, elapsed, reason = gate._bounded(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                capture_dir=capture, worker_output=None, timeout=.15,
                output_name="stdout.txt", error_name="stderr.txt")
            self.assertEqual(reason, "child_process_timeout")
            self.assertLess(elapsed, 2.0)
            self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
