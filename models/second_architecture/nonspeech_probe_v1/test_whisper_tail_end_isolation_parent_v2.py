"""No-model parent evidence and fail-closed tests."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from . import whisper_tail_end_isolation_parent_v2 as parent
from . import whisper_tail_end_isolation_saved_audit_v2 as saved_audit


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ParentTests(unittest.TestCase):
    def test_fake_child_receives_live_parent_and_is_monitored(self):
        script = ("import os,sys,time\n"
                  "fd=int(sys.argv[-1]); hello=os.read(fd,39)\n"
                  "assert hello.startswith(b'HELLO ') and hello.endswith(b'\\n')\n"
                  "time.sleep(.45)\n")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = {"monotonic_seconds": time.monotonic()}
            with (root / "host_samples.jsonl").open("wb") as host_out:
                result = parent.run_bounded(
                    [sys.executable, "-B", "-c", script], root, "original",
                    nonce="0" * 32, deadline=time.monotonic() + 300,
                    samples=[sample], host_out=host_out)
            self.assertEqual(result["worker_returncode"], 0)
            self.assertEqual(result["worker_stop_reason"], "normal_exit")
            self.assertGreater(result["worker_rss_samples"], 0)
            self.assertGreater(result["parent_heartbeat_count"], 0)

    def test_rss_cap_kills_fake_process_group(self):
        script = "import time; time.sleep(10)"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = {"monotonic_seconds": time.monotonic()}
            with (root / "host_samples.jsonl").open("wb") as host_out, \
                 patch.object(parent, "tree_rss_bytes",
                              return_value=parent.gate.RSS_BYTES):
                result = parent.run_bounded(
                    [sys.executable, "-B", "-c", script], root, "original",
                    nonce="0" * 32, deadline=time.monotonic() + 300,
                    samples=[sample], host_out=host_out)
            self.assertEqual(result["worker_stop_reason"], "rss_limit")
            self.assertNotEqual(result["worker_returncode"], 0)

    def test_original_requires_all_five_arms_after_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_rows(root / "progress.jsonl", [
                {"type": "attempt", "index": i} if step == 0 else
                {"type": "complete", "index": i}
                for i in range(1, 3) for step in (0, 1)])
            rows = [{"type": "call", "index": i} for i in (1, 2)]
            rows.append({"type": "complete", "stage": "original",
                         "executed_calls": 2, "planned_max_calls": 7,
                         "skipped_arms": list(range(1, 6))})
            write_rows(root / "records.jsonl", rows)
            self.assertFalse(parent.complete_worker_output(root, "original"))
            progress = [{"type": kind, "index": i}
                        for i in range(1, 8)
                        for kind in ("attempt", "complete")]
            write_rows(root / "progress.jsonl", progress)
            rows = [{"type": "call", "index": i} for i in range(1, 8)]
            rows.append({"type": "complete", "stage": "original",
                         "executed_calls": 7, "planned_max_calls": 7,
                         "skipped_arms": []})
            write_rows(root / "records.jsonl", rows)
            self.assertTrue(parent.complete_worker_output(root, "original"))
            rows[-1]["planned_max_calls"] = 62
            write_rows(root / "records.jsonl", rows)
            self.assertFalse(parent.complete_worker_output(root, "original"))

    def test_offset_requires_all_four_baselines_and_bounded_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            progress = [{"type": kind, "index": i}
                        for i in range(1, 5)
                        for kind in ("attempt", "complete")]
            write_rows(root / "progress.jsonl", progress)
            rows = [{"type": "call", "index": i} for i in range(1, 5)]
            rows.append({"type": "complete", "stage": "offset_8000",
                         "executed_calls": 4, "planned_max_calls": 24,
                         "skipped_arms": list(range(1, 21))})
            write_rows(root / "records.jsonl", rows)
            self.assertTrue(parent.complete_worker_output(root, "offset_8000"))
            rows[-1]["skipped_arms"][0] = 0
            write_rows(root / "records.jsonl", rows)
            self.assertFalse(parent.complete_worker_output(root, "offset_8000"))
            write_rows(root / "progress.jsonl", progress[:-1])
            self.assertFalse(parent.complete_worker_output(root, "offset_8000"))

    def test_excess_and_linked_evidence_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "normal").write_bytes(b"abc")
            self.assertEqual(parent.evidence_bytes(root), 3)
            (root / "link").symlink_to(root / "normal")
            with self.assertRaisesRegex(ValueError, "linked"):
                parent.evidence_bytes(root)

    def test_no_go_stops_before_host_or_process(self):
        with patch.object(parent.gate, "validate_go", side_effect=ValueError("no GO")), \
             patch.object(parent.host, "quick_probe") as host_probe, \
             patch.object(parent.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "no GO"):
                parent.launch("original", parent.gate.ADMISSIONS["original"],
                              parent.gate.GOS["original"])
            host_probe.assert_not_called()
            spawn.assert_not_called()

    def test_translated_stage_requires_prior_full_saved_audit(self):
        with patch.object(parent.gate, "validate_go", return_value={}), \
             patch.object(saved_audit, "audit", side_effect=ValueError("prior raw invalid")) as prior_audit, \
             patch.object(parent.host, "quick_probe") as host_probe, \
             patch.object(parent.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "prior raw invalid"):
                parent.launch("offset_8000", parent.gate.ADMISSIONS["offset_8000"],
                              parent.gate.GOS["offset_8000"])
            prior_audit.assert_called_once_with(
                "original", parent.gate.ADMISSIONS["original"],
                parent.gate.GOS["original"])
            host_probe.assert_not_called()
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
