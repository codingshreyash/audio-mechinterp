"""No-weight V3 admission path and source-inventory tests."""
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture.nonspeech_probe_v1 import padding_channel_worker_v3 as worker


class WorkerV3Tests(unittest.TestCase):
    def test_fixed_attempt_and_exclusive_future_go(self):
        self.assertEqual(worker.ATTEMPT.name, "attempt_003")
        self.assertEqual(worker.ADMISSION.name, "PADDING_CHANNEL_ADMISSION_V3.json")
        self.assertEqual(worker.GO.name, "PADDING_CHANNEL_EXECUTABLE_GO_V3.json")
        with self.assertRaisesRegex(ValueError, "fixed admission/GO paths"):
            worker.validate_go(Path("PADDING_CHANNEL_ADMISSION_V2.json"), worker.GO)

    def test_source_inventory_includes_v3_deadline_and_auditor(self):
        sources = worker.source_identity()
        for name in ("padding_channel_worker_v3.py", "padding_channel_parent_v3.py",
                     "padding_channel_audit_v3.py", "test_padding_channel_parent_v3.py",
                     "test_padding_channel_audit_v3.py",
                     "WHISPER_PADDING_CHANNEL_PROSPECTIVE_V6.md"):
            self.assertIn(name, sources)
        with patch.object(worker, "sha_file", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "host-gate source changed"):
                worker.source_identity()


if __name__ == "__main__":
    unittest.main()
