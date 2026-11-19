"""No-model source tests for the external step-400 admission candidate."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.second_architecture.voicechat_calendar_fixture_v1 import voicechat_step400_freeze_launcher_v1 as gate


class FreezeLauncherTests(unittest.TestCase):
    def test_command_is_fixed_argv_and_runs_candidate_only(self) -> None:
        args = gate.command(Path("/nemo"), Path("/weights"), Path("/out/diag-001"))
        self.assertEqual(args[:3], [gate.sys.executable, "-B", "-c"])
        self.assertIn("voicechat_step400_worker_v1 import main", args[3])
        self.assertNotIn("run_fc_offline_inference", args[3])
        self.assertEqual(args[4:], ["--nemo-dir", "/nemo", "--checkpoint-dir",
                                    "/weights", "--output-dir", "/out/diag-001"])

    def test_bootstrap_uses_frozen_project_packages_not_shadow_initializer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nemo = root / "nemo"
            project = root / "project"
            fixture = project / "models/second_architecture/voicechat_calendar_fixture_v1"
            output = root / "output"
            for path in (nemo / "models", fixture, output):
                path.mkdir(parents=True)
            (nemo / "models/__init__.py").write_text(
                'from pathlib import Path\nPath("malicious").write_text("ran")\n')
            (fixture / "voicechat_step400_worker_v1.py").write_text(
                'from pathlib import Path\ndef main(): Path("trusted").write_text("ran")\n')
            with patch.object(gate, "REPO_ROOT", project), patch.object(gate, "HERE", fixture):
                argv = gate.command(nemo, root / "weights", output)
            env = {"PYTHONPATH": str(nemo), "PYTHONDONTWRITEBYTECODE": "1",
                   "PATH": os.environ.get("PATH", "")}
            result = subprocess.run(argv, cwd=output, env=env, capture_output=True,
                                    text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "trusted").exists())
            self.assertFalse((output / "malicious").exists())

    def test_untracked_project_initializer_changes_source_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            project = root / "project"
            fixture = project / "models/second_architecture/voicechat_calendar_fixture_v1"
            fixture.mkdir(parents=True)
            source = fixture / "candidate.py"
            source.write_text("pass\n")
            prior = Path.cwd()
            try:
                os.chdir(project)
                with patch.object(gate, "REPO_ROOT", project), patch.object(
                    gate, "HERE", fixture), patch.object(
                    gate, "SOURCE_NAMES", ("candidate.py",)), patch.object(
                    gate.sample, "hash_file", return_value=("0" * 64, 5)), patch.object(
                    gate.subprocess, "run", return_value=SimpleNamespace(stdout="a" * 40)):
                    gate.project_source()
                    (project / "models/__init__.py").write_text("raise RuntimeError()\n")
                    with self.assertRaisesRegex(ValueError, "unreviewed package initializer"):
                        gate.project_source()
            finally:
                os.chdir(prior)

    def test_signed_review_is_freeze_specific_and_rejects_tampering(self) -> None:
        freeze = {"expected": {"project": {"local_source_sha256": {"a.py": "0" * 64}}}}
        raw = gate.canonical(freeze)
        key = bytes(range(32))
        unsigned = {"schema": gate.REVIEW_SCHEMA,
                    "status": "independent_one_attempt_go",
                    "freeze_sha256": gate.digest(raw),
                    "project_source_sha256": gate.digest(gate.canonical(
                        freeze["expected"]["project"]["local_source_sha256"])),
                    "reviewer_id": "different-reviewer", "reviewed_utc":
                    "2026-11-19T18:00:00+00:00"}
        review = dict(unsigned, hmac_sha256=hmac.new(
            key, gate.canonical(unsigned), hashlib.sha256).hexdigest())
        gate.verify_review(freeze, raw, review, key)
        for bad in (dict(review, freeze_sha256="1" * 64),
                    dict(review, hmac_sha256="2" * 64)):
            with self.assertRaises(ValueError):
                gate.verify_review(freeze, raw, bad, key)

    def test_unsigned_build_binds_ready_preload_and_consumes_freeze_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = dict(nemo_dir=root / "nemo", checkpoint_dir=root / "weights",
                        nano_home=root / "nano", output_dir=root / "diag-001",
                        freeze_path=root / "freeze.json", review_path=root / "review.json")
            for name in ("nemo_dir", "checkpoint_dir", "nano_home"):
                dirs[name].mkdir()
            phases = [{"phase": "ready", "source": {}, "sample_sha256": {},
                       "nano": {}, "nano_ref_sha256": "hash", "runtime":
                       {"disk_free_gib": 35.0}, "offline": {}, "dynamic": {}},
                      {"phase": "preload", "source": {}, "sample_sha256": {},
                       "nano": {}, "nano_ref_sha256": "hash", "runtime":
                       {"disk_free_gib": 34.9}, "offline": {}, "dynamic": {}}]
            with patch.object(gate, "check_paths"), patch.object(
                gate.sample, "sample_ready_preload", return_value={
                    "status": "sample_only_ready_preload_passed_no_model_work",
                    "receipts": phases}), patch.object(
                gate, "expected", return_value={"command": ["python", "-B"]}):
                receipt = gate.build(**dirs)
                self.assertEqual(receipt["status"], "unsigned_requires_independent_review")
                content = json.loads(dirs["freeze_path"].read_text())
                self.assertEqual(content["ready_preload"][1]["phase"], "preload")
                self.assertNotIn("disk_free_gib", content["ready_preload"][1]["runtime"])
                with self.assertRaises(ValueError):
                    gate.build(**dirs)

    def test_launch_rejects_missing_review_before_host_or_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = dict(nemo_dir=root / "nemo", checkpoint_dir=root / "weights",
                          nano_home=root / "nano", output_dir=root / "diag-001",
                          freeze_path=root / "freeze.json", review_path=root / "review.json",
                          review_key=bytes(range(32)))
            with patch.object(gate, "check_paths"), patch.object(
                gate, "expected") as host, patch.object(
                gate.parent, "supervise_candidate") as child:
                with self.assertRaises(ValueError):
                    gate.launch(**kwargs)
                host.assert_not_called()
                child.assert_not_called()

    def test_valid_review_reaches_only_bounded_parent_with_exact_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "diag-001"
            freeze_path, review_path = root / "freeze.json", root / "review.json"
            expected = {"project": {"local_source_sha256": {"worker.py": "0" * 64}},
                        "command": ["python", "-B", "-c", "reviewed-bootstrap"],
                        "child_environment": {"PYTHONPATH": str(root / "nemo")}}
            phase = {"phase": "preload"}
            freeze = {"schema": gate.SCHEMA, "status": "unsigned_source_and_host_freeze",
                      "attempt_id": output.name, "expected": expected,
                      "ready_preload": [{"phase": "ready"}, phase]}
            raw = gate.canonical(freeze)
            freeze_path.write_bytes(raw)
            key = bytes(range(32))
            unsigned = {"schema": gate.REVIEW_SCHEMA,
                        "status": "independent_one_attempt_go",
                        "freeze_sha256": gate.digest(raw),
                        "project_source_sha256": gate.digest(gate.canonical(
                            expected["project"]["local_source_sha256"])),
                        "reviewer_id": "separate-reviewer", "reviewed_utc":
                        "2026-11-19T18:00:00+00:00"}
            review = dict(unsigned, hmac_sha256=hmac.new(
                key, gate.canonical(unsigned), hashlib.sha256).hexdigest())
            review_path.write_bytes(gate.canonical(review))
            kwargs = dict(nemo_dir=root / "nemo", checkpoint_dir=root / "weights",
                          nano_home=root / "nano", output_dir=output,
                          freeze_path=freeze_path, review_path=review_path,
                          review_key=key)
            with patch.object(gate, "check_paths"), patch.object(
                gate, "expected", return_value=expected), patch.object(
                gate, "launch_preload", return_value=phase), patch.object(
                gate.parent, "supervise_candidate", return_value={"status": "fake"}) as child:
                self.assertEqual(gate.launch(**kwargs), {"status": "fake"})
                child.assert_called_once_with(
                    command=expected["command"],
                    environment=expected["child_environment"],
                    output_dir=output, staged_checkpoint_dir=root / "weights")
                self.assertTrue(gate.claim_path(output).is_file())
                with self.assertRaises(FileExistsError):
                    gate.launch(**kwargs)
                child.assert_called_once()

    def test_key_is_removed_before_any_preflight_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = dict(nemo_dir=root / "nemo", checkpoint_dir=root / "weights",
                          nano_home=root / "nano", output_dir=root / "diag-001",
                          freeze_path=root / "freeze.json", review_path=root / "review.json",
                          review_key=bytes(range(32)))
            observed: list[bool] = []
            def host(*_args: object) -> dict:
                observed.append(gate.KEY_ENV in os.environ)
                raise ValueError("stop before child")
            with patch.dict(os.environ, {gate.KEY_ENV: "f" * 64}), patch.object(
                gate, "check_paths"), patch.object(gate, "read_json", return_value=(
                    {"schema": gate.SCHEMA, "status": "unsigned_source_and_host_freeze",
                     "attempt_id": "diag-001", "expected": {}, "ready_preload": []}, b"{}\n")), patch.object(
                gate, "verify_review"), patch.object(gate, "expected", side_effect=host):
                with self.assertRaisesRegex(ValueError, "stop before child"):
                    gate.launch(**kwargs)
                self.assertEqual(observed, [False])
                self.assertNotIn(gate.KEY_ENV, os.environ)

    def test_noncanonical_or_duplicate_review_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaises(ValueError):
                gate.read_json(p, 64)
            p.write_bytes(b'{ "a":1}\n')
            with self.assertRaises(ValueError):
                gate.read_json(p, 64)

    def test_read_cap_holds_if_file_grows_between_stat_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "freeze.json"
            p.write_bytes(b'{"a":1}\n')
            original_read = os.read
            changed = False
            def grow_then_read(fd: int, count: int) -> bytes:
                nonlocal changed
                if not changed:
                    changed = True
                    with p.open("ab") as stream:
                        stream.write(b"x" * 100)
                return original_read(fd, count)
            with patch.object(gate.os, "read", side_effect=grow_then_read):
                with self.assertRaisesRegex(ValueError, "changed during bounded read"):
                    gate.read_json(p, 64)


if __name__ == "__main__":
    unittest.main()
