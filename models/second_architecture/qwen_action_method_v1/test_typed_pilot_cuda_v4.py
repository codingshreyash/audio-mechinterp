"""No-weight V4 pilot-only gate and method-inheritance regressions."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from . import typed_pilot_cuda_backend_v3 as v3
from . import typed_pilot_cuda_backend_v4 as backend
from . import typed_pilot_cuda_parent_v4 as parent
from . import typed_pilot_cuda_posthoc_audit_v3 as posthoc


class V4NoWeightTests(unittest.TestCase):
    def test_only_fresh_pilot_path_and_no_go(self) -> None:
        self.assertTrue(str(backend.PILOT_ATTEMPT).endswith("typed_pilot_cuda_v4/pilot_003"))
        with self.assertRaisesRegex(ValueError, "pilot stage only"):
            backend.stage_paths("load")
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "pilot_003"
            with mock.patch.object(backend, "stage_paths",
                                   return_value=(attempt, Path(temp) / "missing_go")), \
                 mock.patch.object(backend, "source_identity", return_value={}), \
                 mock.patch.object(backend, "load_prerequisite", return_value={}), \
                 mock.patch.object(backend, "ADMISSION_CANDIDATE",
                                   Path(temp) / "missing_candidate"):
                with self.assertRaisesRegex(ValueError, "admission candidate absent"):
                    parent.launch("pilot")
            self.assertFalse(attempt.exists())

    def test_scientific_functions_have_same_ast_as_v3(self) -> None:
        def definitions(path: Path) -> dict[str, ast.FunctionDef]:
            return {node.name: node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef)}
        old = definitions(Path(v3.__file__))
        new = definitions(Path(backend.__file__))
        for name in ("fixed_case", "unmodified_references", "teacher_forced_triplet",
                     "generation_pair", "save_array", "measure_cuda"):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)

    def test_approved_posthoc_load_recomputed_and_mismatch_rejected(self) -> None:
        saved = json.loads(backend.LOAD_AUDIT_PATH.read_text())
        with mock.patch.object(posthoc, "audit", return_value=saved):
            lineage = backend.load_prerequisite()
        self.assertEqual(lineage["status"], "load_only_pass_no_forward_or_vjp")
        self.assertEqual(len(lineage["load_evidence_sha256"]), 12)
        with mock.patch.object(posthoc, "audit", return_value={**saved, "status": "bad"}):
            with self.assertRaisesRegex(ValueError, "did not pass independently"):
                backend.load_prerequisite()

    def test_dirty_load_evidence_rejected_before_audit(self) -> None:
        original = subprocess.run
        def fake_run(command, *args, **kwargs):
            if command[:2] == ("git", "diff") and command[-1].endswith("load_002/start.json"):
                return subprocess.CompletedProcess(command, 1)
            return original(command, *args, **kwargs)
        with mock.patch.object(backend.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(posthoc, "audit") as audit:
            with self.assertRaisesRegex(ValueError, "not tracked/clean"):
                backend.load_prerequisite()
            audit.assert_not_called()

    def test_go_cannot_rebind_a_different_load_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = root / "pilot_003"
            go, review, candidate = (root / name for name in ("go.json", "review.md", "candidate.json"))
            review.write_text("review")
            identity, load = {"source": "frozen"}, {"posthoc_audit_sha256": "A"}
            candidate.write_text(json.dumps({"schema": "qwen_action_cuda_admission_candidate_v4",
                "status": "source_only_no_weights", "source_identity": identity,
                "load_prerequisite": load,
                "approved_pilot_attempt": str(attempt.resolve()),
                "runtime_identity": {"gpu_uuid": "GPU-A"}, "gpu_baseline": {}}))
            go.write_text(json.dumps({"schema": "qwen_action_typed_pilot_cuda_go_v4",
                "status": "independent_executable_go", "stage": "pilot",
                "approved_attempt": str(attempt.resolve()),
                "scope": "one typed F_L1 pilot only",
                "qa_policy": "typed_F_L1_only_no_audio_QA_exception",
                "qa_amendment_sha256": backend.sha(backend.QA_AMENDMENT),
                "parent_amendment_sha256": backend.sha(backend.PARENT_AMENDMENT),
                "admission_candidate_sha256": backend.sha(candidate),
                "source_identity": identity, "load_prerequisite": {"posthoc_audit_sha256": "B"},
                "runtime_identity": {"gpu_uuid": "GPU-A"},
                "gpu_baseline": {}, "review_sha256": backend.sha(review)}))
            with mock.patch.object(backend, "stage_paths", return_value=(attempt, go)), \
                 mock.patch.object(backend, "source_identity", return_value=identity), \
                 mock.patch.object(backend, "load_prerequisite", return_value=load), \
                 mock.patch.object(backend, "ADMISSION_CANDIDATE", candidate), \
                 mock.patch.object(backend, "PILOT_ATTEMPT", attempt), \
                 mock.patch.object(backend, "REVIEW", review), \
                 mock.patch.object(backend, "require_independent_source_review"):
                with self.assertRaisesRegex(ValueError, "GO binding invalid"):
                    backend.validate_go("pilot", verify_runtime=False)

    def test_capture_is_pilot_only_and_requires_approved_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate.json"
            load_go = root / "load_go.json"
            load_go.write_text(json.dumps({"runtime_identity": {"gpu_uuid": "GPU-A"}}))
            with mock.patch.object(backend, "ADMISSION_CANDIDATE", candidate), \
                 mock.patch.object(backend, "PILOT_ATTEMPT", root / "pilot_003"), \
                 mock.patch.object(backend, "LOAD_GO", load_go), \
                 mock.patch.object(backend, "source_identity", return_value={"v4_sources": {}}), \
                 mock.patch.object(backend, "load_prerequisite", return_value={"status": "load_only_pass_no_forward_or_vjp"}), \
                 mock.patch.object(backend, "runtime_identity", return_value={"gpu_uuid": "GPU-B"}):
                with self.assertRaisesRegex(ValueError, "runtime differs"):
                    parent.capture_admission()
                self.assertFalse(candidate.exists())

    def test_pilot_timing_uses_bounded_parent_clocks(self) -> None:
        start = {"start_utc": "2026-11-19T00:00:00+00:00"}
        result = {"end_utc": "2026-11-19T00:00:10+00:00",
                  "worker_exit_utc": "2026-11-19T00:00:08+00:00",
                  "elapsed_seconds_before_result": 10.0}
        terminal = {"terminal_utc": "2026-11-19T00:00:10.004+00:00",
                    "elapsed_seconds_after_result": 10.004}
        pre = {"collected_utc": "2026-11-19T00:00:01+00:00"}
        post = {"collected_utc": "2026-11-19T00:00:09+00:00"}
        self.assertEqual(posthoc.verify_parent_timing("pilot", start, result,
                         terminal, pre, post)["wall_after_seconds"], 10.004)
        with self.assertRaisesRegex(ValueError, "deadline proof invalid"):
            posthoc.verify_parent_timing("pilot", start, result,
                {**terminal, "elapsed_seconds_after_result": 900.001}, pre, post)

    def test_direct_worker_without_live_parent_pipe_rejects_before_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = root / "pilot_003"
            attempt.mkdir()
            go = root / "go.json"
            go.write_text("{}")
            pre = attempt / "pre_host.json"
            pre.write_text(json.dumps({"runtime_identity": {"gpu_uuid": "GPU-A"}}))
            stage = attempt / "stage_start_gpu.json"
            stage.write_text(json.dumps({"gpu_uuid": "GPU-A", "compute_processes": [],
                                         "gpu_used_bytes": 0}))
            (attempt / "start.json").write_text(json.dumps({
                "schema": "qwen_action_typed_pilot_cuda_parent_v4",
                "status": "worker_authorized", "stage": "pilot",
                "approved_attempt": str(attempt.resolve()),
                "go_sha256": backend.sha(go), "pre_host_sha256": backend.sha(pre),
                "stage_start_gpu_sha256": backend.sha(stage),
                "parent_pid": os.getppid(),
                "nonce_sha256": hashlib.sha256(b"fake").hexdigest(),
                "deadline_monotonic": time.monotonic() + 10}))
            original_import = __import__
            imported_weights = []
            def guard_import(name, *args, **kwargs):
                if name in ("torch", "transformers"):
                    imported_weights.append(name)
                    raise AssertionError("candidate import preceded parent nonce")
                return original_import(name, *args, **kwargs)
            try:
                with mock.patch.object(backend, "stage_paths", return_value=(attempt, go)), \
                     mock.patch.dict(os.environ, {"QWEN_TYPED_PILOT_PARENT_FD": ""}), \
                     mock.patch("builtins.__import__", side_effect=guard_import):
                    with self.assertRaisesRegex(ValueError, "parent pipe absent"):
                        backend.run_worker(attempt, "pilot")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            self.assertFalse(imported_weights)
            self.assertFalse((attempt / "worker_output").exists())

    def test_v4_source_review_and_go_follow_candidate(self) -> None:
        with mock.patch.object(backend, "_tracked_clean"), \
             mock.patch.object(backend, "_ancestor", return_value=True):
            def commits(path):
                if path == backend.ADMISSION_CANDIDATE:
                    return "candidate"
                if path == backend.REVIEW:
                    return "source"
                if path == backend.PILOT_GO:
                    return "go"
                return "source"
            with mock.patch.object(backend, "_commit_of", side_effect=commits):
                with self.assertRaisesRegex(ValueError, "independent source review/GO"):
                    backend.require_independent_source_review(backend.PILOT_GO)
            def same_review_go(path):
                if path == backend.ADMISSION_CANDIDATE:
                    return "candidate"
                if path in (backend.REVIEW, backend.PILOT_GO):
                    return "review"
                return "source"
            with mock.patch.object(backend, "_commit_of", side_effect=same_review_go):
                with self.assertRaisesRegex(ValueError, "independent source review/GO"):
                    backend.require_independent_source_review(backend.PILOT_GO)


if __name__ == "__main__":
    unittest.main()
