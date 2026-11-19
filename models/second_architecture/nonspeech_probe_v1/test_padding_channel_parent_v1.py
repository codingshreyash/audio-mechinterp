"""No-weight parent census and fail-closed launch tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture.nonspeech_probe_v1 import padding_channel_parent_v1 as parent
from models.second_architecture.nonspeech_probe_v1 import padding_channel_worker_v1 as worker
from models.second_architecture.nonspeech_probe_v1 import causal_parent_v2 as bounded


class PaddingParentTests(unittest.TestCase):
    def test_exact_unique_436_call_names(self):
        names = parent.expected_names()
        self.assertEqual(len(names), 436)
        self.assertEqual(len(set(names)), 436)
        self.assertEqual(names[:4], [
            "none_d02_silence_no_hook", "both_d02_silence_read_only_donor_capture",
            "none_d08_silence_no_hook", "both_d08_silence_read_only_donor_capture"])

    def test_progress_classifies_unattempted_and_inflight(self):
        names = parent.expected_names()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            rows = [{"type": "attempt", "call_index": 1, "call_name": names[0]},
                    {"type": "complete", "call_index": 1, "call_name": names[0]},
                    {"type": "attempt", "call_index": 2, "call_name": names[1]}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = bounded.progress_census(path, names)
            self.assertEqual(result["completed_call_names"], names[:1])
            self.assertEqual(result["failed_attempted_call_name"], names[1])
            self.assertEqual(result["censored_call_names"], names[2:])

    def test_no_go_does_not_create_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "absent"
            with patch.object(parent, "ATTEMPT", attempt), \
                 patch.object(worker, "validate_go", side_effect=ValueError("GO missing")):
                with self.assertRaisesRegex(ValueError, "GO missing"):
                    parent.launch(Path(temp) / "admission", Path(temp) / "go")
            self.assertFalse(attempt.exists())


if __name__ == "__main__":
    unittest.main()
