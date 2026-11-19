"""No-weight faults for V2 executable gate and live parent channel."""
from __future__ import annotations

import json
import os
import tempfile
import subprocess
import time
import unittest
from pathlib import Path
from unittest import mock

from . import asr_proxy_gate_v2 as gate
from . import asr_proxy_parent_v2 as parent
from . import asr_proxy_worker_v1 as old
from . import asr_proxy_worker_v2 as wrapped
from . import asr_proxy_saved_audit_v2 as saved


class GateFaultTests(unittest.TestCase):
    def test_missing_go_fails_before_host_and_attempt(self) -> None:
        with mock.patch.object(gate, "validate_go", side_effect=ValueError("GO missing")), \
             mock.patch.object(parent.host, "quick_probe", side_effect=AssertionError("host touched")):
            with self.assertRaisesRegex(ValueError, "GO missing"):
                parent.launch()

    def test_go_field_fault_and_chronology_fault(self) -> None:
        identity = {"asr_proxy_worker_v2.py": "a" * 64}
        expected = {"model": "facebook/wav2vec2-base-960h", "model_revision": "r",
                    "source_identity": identity, "calls": 16}
        snapshot = b"committed bytes"
        digest = gate.sha_bytes(snapshot)
        go = {"schema": "human_asr_proxy_executable_go_v2", "status": "GO",
              "model": expected["model"], "model_revision": "r",
              "attempt": str(old.ATTEMPT), "calls": 16,
              "admission_v1_sha256": digest, "admission_v2_sha256": digest,
              "source_identity": identity, "review_sha256": digest}
        original_read = Path.read_text

        def fake_read(path: Path, *args, **kwargs):
            if path == gate.ADMISSION_V2:
                return json.dumps(expected)
            if path == gate.GO:
                return json.dumps(go)
            return original_read(path, *args, **kwargs)

        def fake_git(argv: tuple[str, ...]) -> bytes:
            if argv[0] == "show":
                return snapshot
            if argv[0] == "log":
                return ("g" if str(gate.GO.name) in argv[-1] else
                        "r" if str(gate.REVIEW.name) in argv[-1] else "s").encode() * 40
            raise AssertionError(argv)

        with mock.patch.object(gate, "tracked_clean"), \
             mock.patch.object(gate, "admission", return_value=expected), \
             mock.patch.object(gate, "sha", return_value=digest), \
             mock.patch.object(gate, "_git_output", side_effect=fake_git), \
             mock.patch.object(Path, "read_text", fake_read):
            go["attempt"] = "wrong-attempt"
            with self.assertRaisesRegex(ValueError, "GO fields invalid"):
                gate.validate_go()
            go["attempt"] = str(old.ATTEMPT)
            with mock.patch.object(gate, "_ancestor", return_value=False):
                with self.assertRaisesRegex(ValueError, "review must precede GO"):
                    gate.validate_go()
            with mock.patch.object(gate, "_ancestor", side_effect=[True, False]):
                with self.assertRaisesRegex(ValueError, "source/admission must precede"):
                    gate.validate_go()
            expected["source_identity"] = {"asr_proxy_worker_v2.py": "changed"}
            with self.assertRaisesRegex(ValueError, "GO fields invalid"):
                gate.validate_go()
        self.assertIn(gate.HOST_SOURCE, gate.source_paths())

    def test_saved_audit_rejects_wrong_go_hash_before_v1_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            (attempt / "launch.json").write_text(json.dumps({
                "schema_v2": "human_asr_proxy_parent_v2", "go_sha256": "wrong"}))
            (attempt / "parent_result.json").write_text(json.dumps({
                "schema_v2": "human_asr_proxy_parent_result_v2", "go_sha256": "wrong"}))
            with mock.patch.object(old, "ATTEMPT", attempt), \
                 mock.patch.object(gate, "validate_go", return_value={
                     "go_sha256": "correct", "admission_v2_sha256": "a", "source_identity": {}}), \
                 mock.patch.object(saved.old_audit, "audit",
                                   side_effect=AssertionError("V1 audit reached")):
                with self.assertRaisesRegex(ValueError, "binding invalid"):
                    saved.audit()

    def test_worker_requires_live_fd_and_nonce(self) -> None:
        with self.assertRaisesRegex(ValueError, "inherited channel"):
            wrapped.run(old.ATTEMPT, old.ADMISSION, -1)
        go = {"go_sha256": "g", "admission_v1_sha256": "a",
              "admission_v2_sha256": "b", "source_identity": {}}
        launch = {"schema": "human_asr_proxy_parent_v1",
                  "schema_v2": "human_asr_proxy_parent_v2",
                  "worker_parent_pid": os.getppid(), "admission_sha256": "a",
                  "admission_v2_sha256": "b", "go_sha256": "g",
                  "source_identity_v2": {}, "parent_nonce_sha256": "x"}
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"0" * 64 + b"\nbeat\n")
            with mock.patch.object(gate, "validate_go", return_value=go), \
                 mock.patch.object(Path, "read_text", return_value=json.dumps(launch)), \
                 mock.patch.object(old, "run", side_effect=AssertionError("weights reached")):
                with self.assertRaisesRegex(ValueError, "nonce invalid"):
                    wrapped.run(old.ATTEMPT, old.ADMISSION, read_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_closed_channel_exit_race_is_bounded(self) -> None:
        process = mock.Mock()
        process.wait.return_value = 0
        self.assertEqual(parent.classify_closed_channel(process, time.monotonic()),
                         ("normal_exit", 0))
        self.assertEqual(parent.classify_closed_channel(
            process, time.monotonic() - parent.WALL_SECONDS - 1)[0],
            "live_channel_closed")
        process.wait.side_effect = subprocess.TimeoutExpired("worker", 0.2)
        self.assertEqual(parent.classify_closed_channel(process, time.monotonic()),
                         ("live_channel_closed", None))


if __name__ == "__main__":
    unittest.main()
