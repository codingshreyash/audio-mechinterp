"""No-weight whole-attempt wall and pre-write evidence-cap tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

from models.second_architecture.nonspeech_probe_v1 import padding_channel_parent_v2 as parent
from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v5 as plan


class ParentV2Tests(unittest.TestCase):
    def test_remaining_worker_time_includes_host_and_result_reserve(self):
        deadline = 1800.0
        with patch.object(parent.time, "monotonic", return_value=300.0):
            self.assertEqual(parent.remaining_worker_seconds(deadline), 1020.0)
        with patch.object(parent.time, "monotonic", return_value=1320.0):
            with self.assertRaisesRegex(ValueError, "no worker time remains"):
                parent.remaining_worker_seconds(deadline)
        with patch.object(parent.time, "monotonic", return_value=1800.1):
            with self.assertRaisesRegex(ValueError, "wall deadline"):
                parent.guard_deadline(deadline)

    def test_near_cap_parent_write_rejected_before_file_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            near = plan.EVIDENCE_LIMIT_BYTES - plan.EVIDENCE_RESERVE_BYTES - 5
            with patch.object(parent.bounded, "output_bytes", return_value=near):
                with self.assertRaisesRegex(ValueError, "before write"):
                    parent.write_parent_json(attempt, "host.json", {"payload": "long"})
            self.assertFalse((attempt / "host.json").exists())
            with patch.object(parent.bounded, "output_bytes",
                              return_value=plan.EVIDENCE_LIMIT_BYTES-5):
                with self.assertRaisesRegex(ValueError, "before write"):
                    parent.write_parent_json(attempt, "launch_result.json",
                                             {"status": "invalid_or_partial"},
                                             terminal=True)
            self.assertFalse((attempt / "launch_result.json").exists())

    def test_near_cap_host_archive_rejected_before_archive_write(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            clock = 0.0
            evidence = {"collected_utc": datetime.now(timezone.utc).isoformat(),
                        "outputs": {"power_log": "mock host log"}}
            with patch.object(parent.time, "monotonic", return_value=clock), \
                 patch.object(parent.host_gate, "probe", return_value=evidence), \
                 patch.object(parent.bounded, "output_bytes",
                              return_value=plan.EVIDENCE_LIMIT_BYTES
                              -plan.EVIDENCE_RESERVE_BYTES-1), \
                 patch.object(parent.host_gate, "archive") as archive:
                with self.assertRaisesRegex(ValueError, "before write"):
                    parent.host_probe(attempt, "pre", deadline=1800.0)
                archive.assert_not_called()
            self.assertFalse((attempt / "pre_power_log.txt.gz").exists())

    def test_fake_post_overrun_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "attempt"
            admission, go = Path(temp) / "admission", Path(temp) / "go"
            admission.write_text("admission")
            go.write_text("go")
            clock = [0.0]

            def fake_probe(_attempt, label, **_kwargs):
                clock[0] += 200.0 if label == "pre" else 500.0
                return {"collected_utc": datetime.now(timezone.utc).isoformat()}

            def fake_run(_command, _attempt, **kwargs):
                clock[0] += kwargs["seconds"]
                return {"worker_returncode": 0, "worker_stop_reason": "normal_exit",
                        "worker_timed_out": False,
                        "worker_elapsed_seconds": kwargs["seconds"],
                        "worker_peak_tree_rss_bytes": 1,
                        "worker_rss_samples": 1}

            with patch.object(parent, "ATTEMPT", attempt), \
                 patch.object(parent.worker, "validate_go",
                              return_value={"sources": {}}), \
                 patch.object(parent.time, "monotonic", side_effect=lambda: clock[0]), \
                 patch.object(parent, "host_probe", side_effect=fake_probe), \
                 patch.object(parent.bounded, "run_bounded", side_effect=fake_run):
                result = parent.launch(admission, go)
            self.assertEqual(result["status"], "invalid_or_partial")
            self.assertFalse((attempt / "launch_completion.json").exists())
            self.assertIn("wall deadline", result["launch_error"]["message"])


if __name__ == "__main__":
    unittest.main()
