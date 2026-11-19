"""Fake, no-model tests of V3 terminal write ordering and deadline closure."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture.nonspeech_probe_v1 import padding_channel_parent_v3 as parent


class ParentV3Tests(unittest.TestCase):
    def _attempt(self, root: Path) -> Path:
        attempt = root / "attempt_003"
        attempt.mkdir()
        parent.worker.write_once(attempt / "launch_result.json",
                                 {"schema": "whisper_padding_launch_v3",
                                  "status": "candidate_complete"})
        return attempt

    def test_slow_completion_write_cannot_leave_terminal_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = self._attempt(Path(temp))
            clock = [100.0]
            original = parent.write_parent_json

            def slow_write(path, filename, payload, **kwargs):
                original(path, filename, payload, **kwargs)
                if filename == "launch_completion.json":
                    clock[0] = 1801.0

            with patch.object(parent.time, "monotonic", side_effect=lambda: clock[0]), \
                 patch.object(parent, "write_parent_json", side_effect=slow_write):
                with self.assertRaisesRegex(ValueError, "wall deadline"):
                    parent.finalize_completion(attempt, start_mono=0.0,
                                               deadline=1800.0)
            self.assertEqual(json.loads((attempt / "launch_completion.json").read_text())
                             ["status"], "pending_terminal_deadline_proof")
            self.assertFalse((attempt / "launch_terminal.json").exists())

    def test_slow_terminal_write_is_removed_before_failed_return(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = self._attempt(Path(temp))
            clock = [100.0]
            original = parent.write_parent_json

            def slow_write(path, filename, payload, **kwargs):
                original(path, filename, payload, **kwargs)
                if filename == "launch_terminal.json":
                    clock[0] = 1801.0

            with patch.object(parent.time, "monotonic", side_effect=lambda: clock[0]), \
                 patch.object(parent, "write_parent_json", side_effect=slow_write):
                with self.assertRaisesRegex(ValueError, "wall deadline"):
                    parent.finalize_completion(attempt, start_mono=0.0,
                                               deadline=1800.0)
            self.assertTrue((attempt / "launch_completion.json").is_file())
            self.assertFalse((attempt / "launch_terminal.json").exists())

    def test_successful_terminal_binds_result_and_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = self._attempt(Path(temp))
            with patch.object(parent.time, "monotonic", return_value=100.0):
                parent.finalize_completion(attempt, start_mono=0.0,
                                           deadline=1800.0)
            terminal = json.loads((attempt / "launch_terminal.json").read_text())
            self.assertEqual(terminal["completion_sha256"], parent.worker.sha_file(
                attempt / "launch_completion.json"))
            self.assertEqual(terminal["result_sha256"], parent.worker.sha_file(
                attempt / "launch_result.json"))


if __name__ == "__main__":
    unittest.main()
