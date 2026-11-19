"""No-weight V3 audit destination and terminal-proof tests."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from models.second_architecture.nonspeech_probe_v1 import padding_channel_audit_v3 as audit
from models.second_architecture.nonspeech_probe_v1 import padding_channel_parent_v3 as parent


class AuditV3Tests(unittest.TestCase):
    def test_fixed_destination_never_targets_prior_attempt(self):
        self.assertEqual(audit.AUDIT_PATH.name, "attempt_003_audit.json")
        self.assertEqual(audit.AUDIT_PATH.parent, audit.worker.ATTEMPT.parent)
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "attempt_003"
            attempt.mkdir()
            output = Path(temp) / "attempt_003_audit.json"
            with patch.object(audit.worker, "ATTEMPT", attempt), \
                 patch.object(audit, "AUDIT_PATH", output):
                audit.verify_audit_destination(attempt, output)
                for wrong in (Path(temp) / "attempt_001_audit.json",
                              Path(temp) / "attempt_002_audit.json"):
                    with self.assertRaisesRegex(ValueError, "fixed V3 audit"):
                        audit.verify_audit_destination(attempt, wrong)
                with self.assertRaisesRegex(ValueError, "fixed V3 audit"):
                    audit.verify_audit_destination(Path(temp) / "attempt_001", output)
                output.write_text("occupied")
                with self.assertRaisesRegex(ValueError, "exclusive write"):
                    audit.verify_audit_destination(attempt, output)

    def test_pending_completion_alone_cannot_pass_saved_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            start = {"start_utc": datetime.now(timezone.utc).isoformat()}
            result = {"status": "candidate_complete"}
            audit.worker.write_once(attempt / "launch_result.json", result)
            completion = {"schema": "whisper_padding_completion_v3",
                          "status": "pending_terminal_deadline_proof",
                          "result_sha256": audit.worker.sha_file(
                              attempt / "launch_result.json"),
                          "parent_elapsed_seconds": 100.0,
                          "completed_utc": datetime.now(timezone.utc).isoformat()}
            audit.worker.write_once(attempt / "launch_completion.json", completion)
            self.assertIsNone(audit.verify_terminal_deadline(attempt, start, result))

    def test_terminal_receipt_checks_post_completion_and_own_mtime(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            start = {"start_utc": datetime.now(timezone.utc).isoformat()}
            result = {"status": "candidate_complete"}
            audit.worker.write_once(attempt / "launch_result.json", result)
            with patch.object(parent.time, "monotonic", return_value=100.0):
                parent.finalize_completion(attempt, start_mono=0.0,
                                           deadline=1800.0)
            self.assertIsNotNone(audit.verify_terminal_deadline(attempt, start,
                                                                 result))
            terminal_path = attempt / "launch_terminal.json"
            terminal = json.loads(terminal_path.read_text())
            terminal["parent_elapsed_seconds_after_completion_write"] = 1801.0
            terminal_path.write_text(audit.worker.canonical(terminal) + "\n")
            with self.assertRaisesRegex(ValueError, "terminal deadline"):
                audit.verify_terminal_deadline(attempt, start, result)
            terminal["parent_elapsed_seconds_after_completion_write"] = 100.0
            terminal_path.write_text(audit.worker.canonical(terminal) + "\n")
            late = datetime.fromisoformat(start["start_utc"]).timestamp() + 1801
            os.utime(terminal_path, (late, late))
            with self.assertRaisesRegex(ValueError, "terminal deadline"):
                audit.verify_terminal_deadline(attempt, start, result)


if __name__ == "__main__":
    unittest.main()
