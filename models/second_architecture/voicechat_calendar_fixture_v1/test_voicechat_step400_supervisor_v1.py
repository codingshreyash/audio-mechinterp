"""No-model, no-NVML tests for the candidate step-400 parent supervisor."""
from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from . import voicechat_step400_supervisor_v1 as parent


CHILD = """import os,sys,time
fd=int(os.environ['VOICECHAT_DIAG_HEARTBEAT_FD'])
os.write(fd,b'\\x01')
print('fake only',flush=True)
time.sleep(float(sys.argv[1]))
sys.exit(int(sys.argv[2]))
"""


class SupervisorFakeTests(unittest.TestCase):
    def run_fake(self, *, sleep: float = 0.03, code: int = 0, sample=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        staged = root / "staged"
        staged.mkdir()
        output = root / "attempt_001"
        if sample is None:
            sample = lambda _pid: parent.GpuSample(100, 90, True)
        result = parent.supervise_candidate(
            command=[sys.executable, "-c", CHILD, str(sleep), str(code)],
            environment=os.environ.copy(), output_dir=output,
            staged_checkpoint_dir=staged, gpu_sample=sample,
            disk_free_bytes=lambda _path: 31 * 1024**3,
        )
        return result, output, staged

    def test_fake_child_receipt_and_poll_timestamps(self):
        result, output, _ = self.run_fake()
        self.assertEqual(result["status"], "bounded_child_complete_pending_saved_audit")
        self.assertEqual(result["heartbeat_count"], 1)
        rows = [json.loads(line) for line in (output / "gpu_poll.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), result["sample_count"])
        self.assertTrue(all(row["sample_begin_monotonic"] <= row["sample_end_monotonic"]
                            for row in rows))
        self.assertTrue(all(rows[i]["sample_end_monotonic"] -
                            rows[i-1]["sample_end_monotonic"] <= 0.2
                            for i in range(1, len(rows))))
        stdout_hash = hashlib.sha256(b"stdout.log").hexdigest()
        self.assertEqual(result["evidence_inventory_before_parent_receipt"][stdout_hash]["bytes"],
                         len(b"fake only\n"))
        with self.assertRaisesRegex(ValueError, "consumed diagnostic attempt"):
            parent.supervise_candidate(command=[sys.executable], environment=os.environ.copy(),
                                       output_dir=output, staged_checkpoint_dir=output,
                                       gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                                       disk_free_bytes=lambda _path: 31 * 1024**3)

    def test_gpu_cap_kills_child(self):
        result, _, _ = self.run_fake(sleep=5, sample=lambda _pid: parent.GpuSample(130001, 1, True))
        self.assertEqual(result["reason"], "gpu_memory_cap")
        self.assertEqual(result["status"], "invalid_or_partial")
        self.assertLess(result["elapsed_monotonic_seconds"], 2)

    def test_actual_nvml_call_gap_fails_coverage(self):
        def slow(_pid):
            time.sleep(.25)
            return parent.GpuSample(100, 90, True)
        result, _, _ = self.run_fake(sleep=5, sample=slow)
        self.assertEqual(result["reason"], "gpu_telemetry_gap")
        self.assertGreater(result["largest_sample_gap_seconds"], .2)

    def test_missing_heartbeat_fails_closed(self):
        script = "import time; time.sleep(5)"
        with tempfile.TemporaryDirectory() as name, patch.object(parent, "HEARTBEAT_DEADLINE_SECONDS", .12):
            root = Path(name)
            staged = root / "staged"
            staged.mkdir()
            result = parent.supervise_candidate(
                command=[sys.executable, "-c", script], environment=os.environ.copy(),
                output_dir=root / "attempt", staged_checkpoint_dir=staged,
                gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                disk_free_bytes=lambda _path: 31 * 1024**3)
        self.assertEqual(result["reason"], "heartbeat_timeout")

    def test_nonzero_child_and_output_log_limit(self):
        result, _, _ = self.run_fake(code=7)
        self.assertEqual(result["reason"], "child_nonzero")
        with patch.object(parent, "MAX_LOG_BYTES", 2):
            result, _, _ = self.run_fake(sleep=5)
        self.assertEqual(result["reason"], "stdout_log_cap")

    def test_disk_floor_precedes_attempt_creation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            staged = root / "staged"
            staged.mkdir()
            output = root / "attempt"
            with self.assertRaisesRegex(ValueError, "30 GiB"):
                parent.supervise_candidate(
                    command=[sys.executable], environment=os.environ.copy(),
                    output_dir=output, staged_checkpoint_dir=staged,
                    gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                    disk_free_bytes=lambda _path: 1)
            self.assertFalse(output.exists())

    def test_aggregate_evidence_is_trimmed_after_kill(self):
        script = """import os,time
os.write(int(os.environ['VOICECHAT_DIAG_HEARTBEAT_FD']),b'\\x01')
open('a.bin','wb').write(b'a'*10000)
open('b.bin','wb').write(b'b'*10000)
time.sleep(5)
"""
        with tempfile.TemporaryDirectory() as name, patch.object(parent, "MAX_EVIDENCE_BYTES", 32768), \
                patch.object(parent, "TERMINAL_RESERVE_BYTES", 16384):
            root = Path(name)
            staged = root / "staged"
            staged.mkdir()
            output = root / "attempt"
            result = parent.supervise_candidate(
                command=[sys.executable, "-c", script], environment=os.environ.copy(),
                output_dir=output, staged_checkpoint_dir=staged,
                gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                disk_free_bytes=lambda _path: 31 * 1024**3)
            retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        self.assertEqual(result["reason"], "evidence_cap")
        self.assertTrue(result["truncated_evidence"])
        self.assertLessEqual(retained, 32768)

    def test_child_symlink_is_preserved_with_external_no_go_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            staged = root / "staged"
            staged.mkdir()
            target = root / "outside.txt"
            target.write_text("untouched")
            secret_name = "PRIVATE_user_prompt_calendar_name.link"
            script = ("import os,time; "
                      "os.write(int(os.environ['VOICECHAT_DIAG_HEARTBEAT_FD']),b'\\x01'); "
                      f"os.symlink({str(target)!r},{secret_name!r}); time.sleep(5)")
            output = root / "attempt"
            result = parent.supervise_candidate(
                command=[sys.executable, "-c", script], environment=os.environ.copy(),
                output_dir=output, staged_checkpoint_dir=staged,
                gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                disk_free_bytes=lambda _path: 31 * 1024**3)
            receipt = root / "attempt.parent_failure.json"
            self.assertTrue(receipt.is_file())
            self.assertTrue((output / secret_name).is_symlink())
            self.assertEqual(target.read_text(), "untouched")
            self.assertEqual(result["status"], "invalid_or_partial")
            self.assertEqual(result["reason"], "unsealed_child_evidence")
            self.assertIs(result["no_go"], True)
            self.assertEqual(result["cap_policy_status"], "failed_no_64mib_proof")
            self.assertNotIn(secret_name.encode(), receipt.read_bytes())
            self.assertNotIn(b"untouched", receipt.read_bytes())

    def test_thirty_three_child_files_get_bounded_failure_receipt(self):
        script = """import os,time
os.write(int(os.environ['VOICECHAT_DIAG_HEARTBEAT_FD']),b'\\x01')
for i in range(33):
    open('extra_%02d.bin'%i,'wb').write(b'x')
time.sleep(5)
"""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            staged = root / "staged"
            staged.mkdir()
            output = root / "attempt"
            result = parent.supervise_candidate(
                command=[sys.executable, "-c", script], environment=os.environ.copy(),
                output_dir=output, staged_checkpoint_dir=staged,
                gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                disk_free_bytes=lambda _path: 31 * 1024**3)
            self.assertTrue((root / "attempt.parent_failure.json").is_file())
            self.assertEqual(result["status"], "invalid_or_partial")
            self.assertEqual(result["reason"], "unsealed_child_evidence")
            self.assertEqual(len(list(output.iterdir())), 36)

    def test_exited_child_cannot_leave_session_descendant_running(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            staged = root / "staged"
            staged.mkdir()
            marker = root / "descendant_survived.txt"
            grandchild = ("import time,pathlib; time.sleep(.8); "
                          f"pathlib.Path({str(marker)!r}).write_text('survived')")
            script = ("import os,subprocess,sys; "
                      "os.write(int(os.environ['VOICECHAT_DIAG_HEARTBEAT_FD']),b'\\x01'); "
                      f"subprocess.Popen([sys.executable,'-c',{grandchild!r}],"
                      "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)")
            result = parent.supervise_candidate(
                command=[sys.executable, "-c", script], environment=os.environ.copy(),
                output_dir=root / "attempt", staged_checkpoint_dir=staged,
                gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                disk_free_bytes=lambda _path: 31 * 1024**3)
            time.sleep(.9)
            self.assertFalse(marker.exists())
            self.assertEqual(result["status"], "bounded_child_complete_pending_saved_audit")


if __name__ == "__main__":
    unittest.main()
