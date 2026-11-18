"""CPU-only synthetic tests for the external Stage-A launch scaffold."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import voxtral_old_fit_handoff_stage_a_launch_gate_v1 as gate


class StageALaunchGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.files = {key: self.repo / f"{key}.txt" for key in gate.FILES}
        for key, path in self.files.items():
            path.write_text(f"fixture {key}\n")
        self.pins = {key: gate.sha(path) for key, path in self.files.items()}
        self._commit("source review")
        self.source_commit = self._head()
        self._commit("auditor review")
        self.auditor_commit = self._head()
        self._commit("launcher review")
        self.launcher_commit = self._head()
        self.freeze = self.repo / "external_freeze.json"
        self.runs = self.repo / "runs"
        self.manifest = self.repo / "bf16_manifest.json"

    def _commit(self, message):
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.test", "commit", "-q",
                        "--allow-empty", "-m", message], check=True)

    def _head(self):
        return subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                                       text=True).strip()

    def _freeze(self, commit=True):
        value = {"status": "frozen_old_fit_handoff_stage_a_v1",
                 "permitted_arms": ["bf16", "fp32"], "audit_scope": gate.SCOPE,
                 "source_review_commit": self.source_commit,
                 "auditor_review_commit": self.auditor_commit,
                 "launcher_review_commit": self.launcher_commit,
                 "reviewed_utc": "2026-11-18T13:00:00Z", **self.pins}
        self.freeze.write_text(json.dumps(value) + "\n")
        if commit:
            self._commit("separate external freeze")
        return value

    def _verified(self):
        pinned = {key: value for key, value in self.pins.items()
                  if key not in gate.REVIEWED_LAUNCH_FILES}
        return gate.verify_freeze(self.freeze, files=self.files, pinned=pinned,
                                  repo=self.repo)

    def _bf16_manifest(self, frozen, *, elapsed=24.0, status=gate.LAUNCH_STATUS):
        attempt = self.runs / "bf16" / "attempt_1"
        (attempt / "runner").mkdir(parents=True)
        (attempt / "runner/blobs").mkdir()
        for name in ("meta.json", "model_layout.json", "progress.jsonl"):
            (attempt / "runner" / name).write_text(f"fixture {name}\n")
        blobs = {}
        for name in ("probes", "details"):
            payload = f"fixture {name}\n".encode()
            digest = gate.sha_bytes(payload)
            (attempt / "runner/blobs" / f"{digest}.pt").write_bytes(payload)
            blobs[f"{name}_sha256"] = digest
            blobs[f"{name}_path"] = f"blobs/{digest}.pt"
            blobs[f"{name}_bytes"] = len(payload)
        result = {"status": gate.RESULT_STATUS, "arm": "bf16",
                  "script_sha256": frozen["receipt"]["runner_sha256"],
                  "native_source_sha256": frozen["receipt"]["native_source_sha256"],
                  "protocol_sha256": frozen["receipt"]["protocol_sha256"],
                  "meta_sha256": gate.sha(attempt / "runner/meta.json"),
                  "model_layout_sha256": gate.sha(attempt / "runner/model_layout.json"),
                  "progress_sha256": gate.sha(attempt / "runner/progress.jsonl"), **blobs}
        gate.exclusive_json(attempt / "runner/result.json", result)
        result_sha = gate.sha(attempt / "runner/result.json")
        audit = {"status": gate.AUDIT_STATUS, "arm": "bf16",
                 "result_sha256": result_sha,
                 "auditor_sha256": frozen["receipt"]["auditor_sha256"],
                 "runner_sha256": frozen["receipt"]["runner_sha256"],
                 "native_source_sha256": frozen["receipt"]["native_source_sha256"],
                 "protocol_sha256": frozen["receipt"]["protocol_sha256"],
                 "probes_sha256": blobs["probes_sha256"],
                 "details_sha256": blobs["details_sha256"]}
        gate.exclusive_json(attempt / "audit.json", audit)
        audit_sha = gate.sha(attempt / "audit.json")
        gate.exclusive_json(attempt / "launch_start.json", {
            "arm": "bf16", "freeze_receipt_sha256": frozen["receipt_sha256"],
            "runner_sha256": frozen["receipt"]["runner_sha256"],
            "auditor_sha256": frozen["receipt"]["auditor_sha256"]})
        gate.exclusive_json(attempt / "launch_finish.json", {
            "status": status, "arm": "bf16", "freeze_receipt_sha256": frozen["receipt_sha256"],
            "runner_result_sha256": result_sha, "audit_sha256": audit_sha,
            "post_run_source_verified": True, "runner_returncode": 0, "auditor_returncode": 0,
            "model_work_elapsed_seconds": elapsed})
        value = {"status": status, "arm": "bf16", "attempt_relative_path": "attempt_1",
                 "freeze_receipt_sha256": frozen["receipt_sha256"],
                 "launch_start_sha256": gate.sha(attempt / "launch_start.json"),
                 "launch_finish_sha256": gate.sha(attempt / "launch_finish.json"),
                 "result_sha256": result_sha, "audit_sha256": audit_sha,
                 "model_work_elapsed_seconds": elapsed}
        self.manifest.write_text(json.dumps(value) + "\n")
        self._commit("external BF16 attempt manifest")
        return attempt

    def test_reviewed_source_pins_match_and_live_freeze_is_absent(self):
        self.assertEqual(set(gate.LIVE_SOURCE_PINS), set(gate.FILES) - set(gate.REVIEWED_LAUNCH_FILES))
        for name, expected in gate.LIVE_SOURCE_PINS.items():
            self.assertEqual(gate.sha(gate.FILES[name]), expected, name)
        with self.assertRaisesRegex(ValueError, "External freeze receipt absent"):
            gate.verify_freeze()

    def test_absent_and_uncommitted_freeze_fail(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            self._verified()
        self._freeze(commit=False)
        with self.assertRaisesRegex(ValueError, "Missing committed blob"):
            self._verified()

    def test_committed_freeze_verifies_then_tamper_fails(self):
        self._freeze()
        frozen = self._verified()
        self.assertEqual(frozen["receipt"]["permitted_arms"], ["bf16", "fp32"])
        self.files["runner_sha256"].write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "Uncommitted or changed"):
            self._verified()

    def test_review_commit_does_not_cover_changed_launcher(self):
        self._freeze()
        self.files["launcher_sha256"].write_text("changed after review\n")
        self._commit("unreviewed launcher")
        value = json.loads(self.freeze.read_text())
        value["launcher_sha256"] = gate.sha(self.files["launcher_sha256"])
        self.freeze.write_text(json.dumps(value) + "\n")
        self._commit("false refreeze")
        with self.assertRaisesRegex(ValueError, "Launcher review commit did not contain"):
            self._verified()

    def test_freeze_historical_readd_is_rejected(self):
        self._freeze()
        self.assertEqual(gate.first_add_commit(self.repo, self.freeze), self._head())
        saved = self.freeze.read_bytes()
        self.freeze.unlink()
        self._commit("remove freeze")
        self.freeze.write_bytes(saved)
        self._commit("readd freeze")
        with self.assertRaisesRegex(ValueError, "exactly one Git first-add"):
            self._verified()

    def test_bf16_admission_then_no_retry(self):
        self._freeze()
        frozen = self._verified()
        output = self.runs / "bf16" / "attempt_1"
        self.assertEqual(gate.admission("bf16", output, frozen=frozen, runs=self.runs,
                                        manifest=self.manifest, repo=self.repo), 0)
        gate.reserve_attempt(output, "bf16", frozen, ["fixture"], 0, runs=self.runs)
        with self.assertRaisesRegex(ValueError, "no retry"):
            gate.admission("bf16", self.runs / "bf16" / "attempt_2", frozen=frozen,
                           runs=self.runs, manifest=self.manifest, repo=self.repo)
        with self.assertRaises(FileExistsError):
            gate.reserve_attempt(self.runs / "bf16" / "attempt_2", "bf16",
                                 frozen, ["fixture"], 0, runs=self.runs)

    def test_fp32_requires_committed_audited_bf16_manifest(self):
        self._freeze()
        frozen = self._verified()
        target = self.runs / "fp32" / "attempt_1"
        with self.assertRaisesRegex(ValueError, "Uncommitted or changed"):
            gate.admission("fp32", target, frozen=frozen, runs=self.runs,
                           manifest=self.manifest, repo=self.repo)
        self._bf16_manifest(frozen)
        self.assertEqual(gate.admission("fp32", target, frozen=frozen, runs=self.runs,
                                        manifest=self.manifest, repo=self.repo), 24.0)

    def test_fp32_rejects_tampered_or_failed_bf16(self):
        self._freeze()
        frozen = self._verified()
        attempt = self._bf16_manifest(frozen)
        (attempt / "audit.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "file hash mismatch"):
            gate.admission("fp32", self.runs / "fp32" / "attempt_1", frozen=frozen,
                           runs=self.runs, manifest=self.manifest, repo=self.repo)

    def test_no_extra_attempt_allowed_before_fp32(self):
        self._freeze()
        frozen = self._verified()
        self._bf16_manifest(frozen)
        (self.runs / "bf16" / "attempt_2").mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected Stage-A attempts"):
            gate.admission("fp32", self.runs / "fp32" / "attempt_1", frozen=frozen,
                           runs=self.runs, manifest=self.manifest, repo=self.repo)

    def test_cumulative_cap_counts_prior_attempt(self):
        self._freeze()
        frozen = self._verified()
        self._bf16_manifest(frozen, elapsed=899.0)
        original = gate.CUMULATIVE_MODEL_SECONDS
        gate.CUMULATIVE_MODEL_SECONDS = 2000
        try:
            with self.assertRaisesRegex(ValueError, "Cumulative model-work cap"):
                gate.admission("fp32", self.runs / "fp32" / "attempt_1", frozen=frozen,
                               runs=self.runs, manifest=self.manifest, repo=self.repo)
        finally:
            gate.CUMULATIVE_MODEL_SECONDS = original

    def test_bounded_stdout_flood_is_stopped(self):
        output = self.repo / "capture"
        output.mkdir()
        code, elapsed, reason = gate.run_bounded_child(
            [sys.executable, "-c", "import sys; sys.stdout.write('x'*1000000)"],
            output=output, env={}, timeout_seconds=5,
            cap_bytes=100000, stream_cap_bytes=1024)
        self.assertNotEqual(code, 0)
        self.assertLess(elapsed, 5)
        self.assertEqual(reason, "runner_stdout_capture_cap_reached")
        self.assertEqual((output / "runner_stdout.txt").stat().st_size, 1024)

    def test_parent_directory_cap_stops_child(self):
        output = self.repo / "capture"
        output.mkdir()
        (output / "already.bin").write_bytes(b"x" * 1000)
        code, _, reason = gate.run_bounded_child(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            output=output, env={}, timeout_seconds=5,
            cap_bytes=1000, stream_cap_bytes=1024)
        self.assertNotEqual(code, 0)
        self.assertEqual(reason, "entire_attempt_byte_cap_reached")

    def test_bf16_sidecar_deletion_blocks_fp32(self):
        self._freeze()
        frozen = self._verified()
        attempt = self._bf16_manifest(frozen)
        result = gate.read_json(attempt / "runner/result.json")
        (attempt / "runner" / result["details_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "sidecar missing or changed"):
            gate.admission("fp32", self.runs / "fp32" / "attempt_1", frozen=frozen,
                           runs=self.runs, manifest=self.manifest, repo=self.repo)


if __name__ == "__main__":
    unittest.main()
