"""CPU-only synthetic tests for the committed external launch gate."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import voxtral_v1_fit_backward_numerical_launch_gate_v1 as gate


class ExternalFreezeTests(unittest.TestCase):
    def test_reviewed_preexisting_pins_match_repository_files(self):
        for name, expected in gate.PINNED.items():
            self.assertEqual(gate.sha(gate.FILES[name]), expected, name)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.files = {name: self.repo / f"{name}.txt" for name in gate.FILES}
        for name, path in self.files.items():
            path.write_text(f"fixed fixture for {name}\n")
        self.pinned = {name: gate.sha(path) for name, path in self.files.items()}
        self.expected = {name: value for name, value in self.pinned.items()
                         if name not in gate.REVIEWED_LAUNCH_FILES}
        self._commit("source and independent review fixture")
        self.review_commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        self.freeze = self.repo / "external_freeze.json"

    def _commit(self, message):
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.test", "commit", "-q", "-m", message], check=True)

    def _freeze(self, *, commit=True):
        receipt = {"status": "frozen_old_fit_numerical_launch_v1",
                   "permitted_arms": ["bf16", "fp32"], "audit_scope": gate.SCOPE,
                   "launcher_review_commit": self.review_commit,
                   "reviewed_utc": "2026-11-18T12:00:00+00:00", **self.pinned}
        self.freeze.write_text(json.dumps(receipt, indent=2) + "\n")
        if commit:
            self._commit("external freeze fixture")

    def test_no_freeze_fails_before_any_attempt(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            gate.verify_freeze(self.freeze, files=self.files, pinned=self.expected, repo=self.repo)
        self.assertFalse((self.repo / "attempt").exists())

    def test_uncommitted_receipt_fails_closed(self):
        self._freeze(commit=False)
        with self.assertRaisesRegex(ValueError, "committed blob"):
            gate.verify_freeze(self.freeze, files=self.files, pinned=self.expected, repo=self.repo)

    def test_valid_committed_receipt_then_runner_tamper_rejected(self):
        self._freeze()
        checked = gate.verify_freeze(self.freeze, files=self.files, pinned=self.expected, repo=self.repo)
        self.assertEqual(checked["receipt"]["runner_sha256"], self.pinned["runner_sha256"])
        self.files["runner_sha256"].write_text("tampered after freeze\n")
        with self.assertRaisesRegex(ValueError, "Uncommitted or changed"):
            gate.verify_freeze(self.freeze, files=self.files, pinned=self.expected, repo=self.repo)

    def test_review_commit_must_contain_same_launcher_source(self):
        self._freeze()
        self.files["launcher_sha256"].write_text("changed launcher\n")
        self._commit("changed launcher after independent review")
        receipt = json.loads(self.freeze.read_text())
        receipt["launcher_sha256"] = gate.sha(self.files["launcher_sha256"])
        self.freeze.write_text(json.dumps(receipt, indent=2) + "\n")
        self._commit("attempted external refreeze")
        with self.assertRaisesRegex(ValueError, "review commit did not contain"):
            gate.verify_freeze(self.freeze, files=self.files, pinned=self.expected, repo=self.repo)

    def test_output_collision_does_not_overwrite_start_receipt(self):
        self._freeze()
        frozen = gate.verify_freeze(self.freeze, files=self.files, pinned=self.expected, repo=self.repo)
        output = self.repo / "attempt"
        gate.reserve_attempt(output, "bf16", frozen, ["python", "runner.py"])
        original = (output / "launch_start.json").read_bytes()
        with self.assertRaises(FileExistsError):
            gate.reserve_attempt(output, "bf16", frozen, ["python", "runner.py"])
        self.assertEqual((output / "launch_start.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
