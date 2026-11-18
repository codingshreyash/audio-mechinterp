"""CPU-only provenance and no-retry checks for the Stage-B external launcher."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
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

    def test_reviewed_external_freeze_accepts_then_rejects_recommit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.git(repo, "init", "-q")
            source_keys = {"prospective_protocol_sha256", "replay_primitives_sha256",
                           "driver_sha256", "driver_tests_sha256", "driver_amendment_sha256",
                           "auditor_sha256", "auditor_tests_sha256",
                           *gate.REVIEWED_SOURCE_KEYS}
            bound = {key: repo / f"{key}.txt" for key in source_keys}
            bound["driver_review_sha256"] = repo / "driver_review.md"
            bound["auditor_review_sha256"] = repo / "auditor_review.md"
            review = bound["stage_b_launcher_review_sha256"]
            fp32_result_review = repo / "fp32_result_review.md"
            for key in source_keys - {"stage_b_launcher_review_sha256"}:
                bound[key].write_text(f"reviewed {key}\n")
            self.commit(repo, "sources")
            bound["driver_review_sha256"].write_text(gate.DRIVER_REVIEW_GO + "\n")
            self.commit(repo, "driver review")
            bound["auditor_review_sha256"].write_text(gate.AUDITOR_REVIEW_GO + "\n")
            self.commit(repo, "auditor review")
            fp32_result_review.write_text("Stage A result GO\n")
            self.commit(repo, "Stage A result review")
            review.write_text(gate.LAUNCHER_REVIEW_GO + "\n")
            launcher_review_commit = self.commit(repo, "launcher review")
            freeze = repo / "freeze.json"
            receipt = {
                "status": gate.STATUS, "permitted_phases": ["bf16_direct_lm"],
                "audit_scope": gate.AUDIT_SCOPE,
                "launcher_review_commit": launcher_review_commit,
                "reviewed_utc": "2026-11-18T15:00:00+00:00",
                "stage_a_model_work_seconds": 218.5633733337745,
                "max_total_bytes": gate.MAX_TOTAL_BYTES,
                "max_total_seconds": gate.driver.MAX_TOTAL_SECONDS,
                **{key: gate.sha(path) for key, path in bound.items()},
            }
            freeze.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            self.commit(repo, "external BF16 freeze")
            freeze_review = repo / "freeze_review.md"
            freeze_review.write_text(gate.FREEZE_REVIEW_GO + "\n" + gate.sha(freeze) + "\n")
            self.commit(repo, "independent freeze review")
            pins = {
                "AUDITOR_SHA": gate.sha(bound["auditor_sha256"]),
                "AUDITOR_TEST_SHA": gate.sha(bound["auditor_tests_sha256"]),
                "DRIVER_SHA": gate.sha(bound["driver_sha256"]),
                "DRIVER_TEST_SHA": gate.sha(bound["driver_tests_sha256"]),
                "DRIVER_AMENDMENT_SHA": gate.sha(bound["driver_amendment_sha256"]),
            }
            with ExitStack() as stack:
                for name, value in {"REPO": repo, "BOUND_FILES": bound, "FREEZE": freeze,
                                    "FREEZE_REVIEW": freeze_review,
                                    "REVIEW": review,
                                    "DRIVER_REVIEW": bound["driver_review_sha256"],
                                    "AUDITOR_REVIEW": bound["auditor_review_sha256"],
                                    **pins}.items():
                    stack.enter_context(mock.patch.object(gate, name, value))
                stack.enter_context(mock.patch.object(gate.driver, "STAGE_A_FP32_RESULT_REVIEW",
                                                      fp32_result_review))
                stack.enter_context(mock.patch.object(gate.auditor, "PROTOCOL_SHA",
                                                      gate.sha(bound["prospective_protocol_sha256"])))
                stack.enter_context(mock.patch.object(gate.auditor, "PRIMITIVES_SHA",
                                                      gate.sha(bound["replay_primitives_sha256"])))
                stack.enter_context(mock.patch.object(gate, "same_stage_a", return_value={
                    "stage_a_model_work_seconds": 218.5633733337745}))
                accepted, _ = gate.verify_freeze()
                self.assertEqual(accepted["launcher_review_commit"], launcher_review_commit)
                receipt["reviewed_utc"] = "2026-11-18T15:00:01+00:00"
                freeze.write_text(json.dumps(receipt, sort_keys=True) + "\n")
                self.commit(repo, "later timestamp edit")
                with self.assertRaisesRegex(ValueError, "changed after first-add"):
                    gate.verify_freeze()


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

    def test_launcher_passes_hard_phase_deadline_not_cap_plus_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze = root / "freeze.json"
            freeze.write_text("{}\n")
            stage_a = {"stage_a_model_work_seconds": 218.5633733337745}
            with (mock.patch.object(gate, "RUNS", root / "gate_runs"),
                  mock.patch.object(gate.driver, "RUNS", root / "driver_runs"),
                  mock.patch.object(gate, "FREEZE", freeze),
                  mock.patch.object(gate, "admission", return_value=(
                      {"reviewed_utc": "2026-11-18T14:00:00+00:00"}, stage_a, None)),
                  mock.patch.object(gate, "_bounded", return_value=(1, .2, "child_process_timeout")) as child):
                result = gate.launch("bf16_direct_lm", "20261118T140000000000Z")
                self.assertEqual(result["status"], "launch_failed")
                self.assertEqual(child.call_args.kwargs["timeout"],
                                 gate.driver.CAP_SECONDS["bf16_direct_lm"])
                self.assertEqual(child.call_args.kwargs["termination_grace"],
                                 gate.PROCESS_GRACE_SECONDS)
                start = json.loads((root / "gate_runs" / "bf16_direct_lm" /
                                    "20261118T140000000000Z" / "launch_start.json").read_text())
                self.assertEqual(start["process_timeout_seconds"],
                                 gate.driver.CAP_SECONDS["bf16_direct_lm"])
                self.assertEqual(start["process_grace_seconds"], gate.PROCESS_GRACE_SECONDS)

    def test_fast_exiting_child_cannot_bypass_stream_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "launch"
            capture.mkdir()
            code, _, reason = gate._bounded(
                [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 9437184)"],
                capture_dir=capture, worker_output=None, timeout=5,
                output_name="stdout.txt", error_name="stderr.txt")
            self.assertEqual(code, 0)
            self.assertEqual(reason, "capture_byte_cap")
            self.assertGreater((capture / "stdout.txt").stat().st_size, gate.MAX_CAPTURE_BYTES)

    def test_fast_exiting_child_cannot_bypass_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "launch"
            capture.mkdir()
            code, _, reason = gate._bounded(
                [sys.executable, "-c", "import time; time.sleep(.04)"],
                capture_dir=capture, worker_output=None, timeout=.01,
                output_name="stdout.txt", error_name="stderr.txt")
            self.assertEqual(code, 0)
            self.assertEqual(reason, "child_process_timeout")


if __name__ == "__main__":
    unittest.main()
