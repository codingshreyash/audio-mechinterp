"""Fake-process and host admission checks; never loads a candidate model."""
from __future__ import annotations

import plistlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acoustic_fit_noise_parent_launch_v1 as parent


class ParentLauncherTests(unittest.TestCase):
    def test_bounded_process_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, timed_out, elapsed = parent.run_bounded(
                [sys.executable, "-c", "print('fake')"],
                root / "stdout", root / "stderr", timeout_seconds=2)
            self.assertEqual(rc, 0)
            self.assertFalse(timed_out)
            self.assertLess(elapsed, 2)
            self.assertIn("fake", (root / "stdout").read_text())

    def test_timeout_kills_fake_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, timed_out, elapsed = parent.run_bounded(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                root / "stdout", root / "stderr", timeout_seconds=0.05)
            self.assertTrue(timed_out)
            self.assertNotEqual(rc, 0)
            self.assertLess(elapsed, 2)

    def test_worker_review_failure_does_not_create_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "attempt"
            with (mock.patch.object(parent.pilot, "preflight", return_value={}),
                  mock.patch.object(parent.pilot, "require_review", side_effect=ValueError("HOLD"))):
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    parent.execute("whisper", root / "receipt", root / "worker_go",
                                   root / "parent_go", attempt)
            self.assertFalse(attempt.exists())

    def test_worker_matrix_rejects_duplicate_and_wrong_pcm(self):
        expected = {f"case_{i}": {arm: f"sha_{i}_{arm}" for arm in parent.pilot.ARMS}
                    for i in range(4)}
        cases = [{"type": "case", "model": "whisper", "case": case,
                  "arm": arm, "pcm_sha256": digest}
                 for case, arms in expected.items() for arm, digest in arms.items()]
        rows = [{"type": "start", "model": "whisper"}, *cases,
                {"type": "complete", "planned_cases": 20}]
        self.assertTrue(parent.valid_worker_rows(rows, "whisper", expected))
        duplicate = [*rows]
        duplicate[1] = duplicate[2]
        self.assertFalse(parent.valid_worker_rows(duplicate, "whisper", expected))
        wrong = [*rows]
        wrong[1] = {**wrong[1], "pcm_sha256": "wrong"}
        self.assertFalse(parent.valid_worker_rows(wrong, "whisper", expected))

    def test_closed_lid_rejected_before_probe_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = {"collected_utc": parent.utc(), "outputs": {
                "power": "Now drawing from 'AC Power'",
                "clamshell": plistlib.dumps([{"AppleClamshellState": True}]).decode(),
                "battery": plistlib.dumps([{"ExternalConnected": True}]).decode(),
                "power_log": "fake"}}
            with mock.patch.object(parent.host_gate, "_host_probe", return_value=evidence):
                with self.assertRaisesRegex(ValueError, "lid is closed"):
                    parent.probe_host(root, "pre")
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
