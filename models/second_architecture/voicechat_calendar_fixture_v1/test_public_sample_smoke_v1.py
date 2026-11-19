"""CPU-only synthetic admission and launcher tests; never load VoiceChat."""

from __future__ import annotations

import hmac
import hashlib
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import public_sample_smoke_v1 as smoke


class FakeChecks:
    def __init__(self, fail: str | None = None):
        self.calls: list[str] = []
        self.fail = fail

    def _call(self, name: str):
        self.calls.append(name)
        if name == self.fail:
            raise ValueError("synthetic preflight failure")
        return {"synthetic_check": name}

    def check_source(self, *_): return self._call("source")
    def check_nano_cache(self, *_): return self._call("nano")
    def check_runtime(self, *_): return self._call("runtime")
    def check_pristine_modules_cache(self, *_): self._call("pristine")
    def verify_nano_offline(self, *_): return self._call("offline")
    def check_preload_environment(self, *_): return self._call("preload_env")
    def check_generated_modules_cache(self, *_): return self._call("dynamic")


class SampleSmokeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.nemo = self.root / "nemo"
        self.checkpoint = self.root / "checkpoint"
        self.nano = self.root / "nano"
        self.out = self.root / "run0001"
        for path in (self.nemo, self.checkpoint, self.nano):
            path.mkdir()
        self.freeze_path = self.root / "freeze.json"
        self.review_path = self.root / "review.json"
        self.source_review_path = self.root / "independent_source_review.md"
        self.key = b"K" * 32
        self.env = patch.dict(os.environ, {
            "PYTHONPATH": str(self.nemo), "HF_HOME": str(self.nano),
            "HF_HUB_CACHE": str(self.nano / "hub"),
            "HF_MODULES_CACHE": str(self.nano / "modules"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1"}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sample = patch.object(smoke, "check_sample", return_value=smoke.SAMPLE_SHA256)
        self.sample.start()
        self.addCleanup(self.sample.stop)
        self.nano_ref = patch.object(smoke, "check_exact_nano_ref", return_value="synthetic-ref-hash")
        self.nano_ref.start()
        self.addCleanup(self.nano_ref.stop)
        self.environment_receipt = patch.object(smoke, "installed_environment_receipt",
                                                return_value={
            "schema": "voicechat_full_python_environment_v1",
            "python_executable": sys.executable,
            "pip_freeze_all_sorted_sha256": "0" * 64,
            "pip_freeze_all_line_count": 7,
            "runner_package_versions": dict(smoke.RUNNER_PACKAGE_PINS),
            "mamba_dependency_versions": {name: None for name in smoke.MAMBA_DEPENDENCY_SPECS},
            "mamba_dependency_specs": dict(smoke.MAMBA_DEPENDENCY_SPECS),
        })
        self.environment_receipt_mock = self.environment_receipt.start()
        self.addCleanup(self.environment_receipt.stop)
        self.native_receipt = patch.object(smoke, "native_ffmpeg_runtime", return_value={
            "schema": "voicechat_native_ffmpeg_runtime_v1", "ffmpeg_major": 7,
            "isolated_libdir": str(self.root / "voicechat-sample-ffmpeg-conda7" / "lib"),
            "public_wav_sha256": smoke.SAMPLE_SHA256[smoke.SAMPLE_WAV],
        })
        self.native_receipt_mock = self.native_receipt.start()
        self.addCleanup(self.native_receipt.stop)
        self.calls: list[list[str]] = []

    def freeze(self, *, cap: int = 65536):
        expected = smoke.expected_freeze_fields(
            self.nemo, self.checkpoint, self.nano, self.out, "h100")
        source_record = {"schema": "voicechat_public_sample_source_review_v1",
                         "decision": "GO", "source_sha256": expected["source_sha256"],
                         "reviewer_id": "synthetic-source-reviewer",
                         "reviewed_utc": "2026-11-19T00:00:00Z"}
        self.source_review_path.write_bytes(smoke.canonical(source_record))
        source_review_sha = smoke.hash_file(self.source_review_path)[0]
        freeze = {"schema": "voicechat_public_sample_smoke_freeze_v1",
                  "status": "externally_frozen_sample_only", "run_id": self.out.name,
                  "expected": expected,
                  "caps": {"wall_seconds": 10, "vram_mib": 1000,
                           "total_evidence_bytes": cap},
                  "source_review": {"path": str(self.source_review_path),
                                    "sha256": source_review_sha}}
        self.freeze_path.write_bytes(smoke.canonical(freeze))
        review = {"schema": "voicechat_public_sample_freeze_review_v1",
                  "status": "independently_reviewed_executable_freeze",
                  "freeze_sha256": smoke.hash_file(self.freeze_path)[0],
                  "source_review_sha256": source_review_sha,
                  "reviewer_id": "synthetic-reviewer", "reviewed_utc": "2026-11-19T00:00:00Z"}
        review["hmac_sha256"] = hmac.new(self.key, smoke.canonical(review),
                                          hashlib.sha256).hexdigest()
        self.review_path.write_bytes(smoke.canonical(review))

    def fake_executor(self, command, output_dir, caps, environment):
        self.calls.append(command)
        self.assertEqual(command, smoke.fixed_command(self.nemo, self.checkpoint, self.out))
        self.assertEqual(command.count("--checkpoint"), 1)
        self.assertEqual(command.count("--wav"), 1)
        self.assertEqual(command.count("--api-response-json"), 1)
        self.assertEqual(command.count("--output-dir"), 1)
        self.assertEqual(len(command), 10)
        self.assertFalse(any("calendar" in value.lower() for value in command))
        self.assertEqual(environment["PYTHONPATH"], str(self.nemo))
        (output_dir / "official_outputs").mkdir()
        (output_dir / "official_outputs" / "sample_fc.txt").write_text("synthetic output")
        (output_dir / "stdout.log").write_text("synthetic stdout\n")
        (output_dir / "stderr.log").write_text("")
        return {"returncode": 0, "process_id": 100, "elapsed_monotonic_seconds": .1,
                "peak_vram_mib": 10, "cap_reason": None, "logs_complete": True}

    def run_fake(self, checks=None, executor=None):
        return smoke.run_sample(
            nemo_dir=self.nemo, checkpoint_dir=self.checkpoint, nano_home=self.nano,
            output_dir=self.out, freeze_path=self.freeze_path,
            review_path=self.review_path, gpu_policy="h100", review_key=self.key,
            checks=checks or FakeChecks(), executor=executor or self.fake_executor)

    def test_four_argument_official_sample_and_saved_evidence(self):
        self.freeze()
        checks = FakeChecks()
        result = self.run_fake(checks)
        self.assertEqual(result["status"], "environment_smoke_only")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([part["phase"] for part in json.loads(
            (self.out / "admission.json").read_text())["receipts"]],
                         ["ready", "preload"])
        self.assertEqual(checks.calls, [
            "source", "nano", "runtime", "pristine", "offline", "dynamic", "nano",
            "source", "nano", "runtime", "preload_env", "dynamic", "nano"])
        self.assertIn("official_outputs/sample_fc.txt", result["artifact_sha256"])
        self.assertTrue((self.out / "scope.json").is_file())
        self.assertTrue((self.out / "result.json").is_file())
        final_manifest = json.loads((self.out / "final_manifest.json").read_text())
        result_row = next(row for row in final_manifest["entries"]
                          if row["path"] == "result.json")
        self.assertEqual(result_row["retained_sha256"],
                         smoke.hash_file(self.out / "result.json")[0])
        self.assertEqual((self.out / "final_manifest.sha256").read_bytes(),
                         smoke._root_digest((self.out / "final_manifest.json").read_bytes()))

    def test_missing_review_and_bad_mac_reject_before_attempt(self):
        self.freeze()
        self.review_path.unlink()
        with self.assertRaisesRegex(ValueError, "Not a regular unlinked file"):
            self.run_fake()
        self.assertFalse(self.out.exists())
        self.freeze()
        review = json.loads(self.review_path.read_text())
        review["reviewer_id"] = "changed"
        self.review_path.write_bytes(smoke.canonical(review))
        with self.assertRaisesRegex(ValueError, "signature differs"): self.run_fake()
        self.assertFalse(self.out.exists())

    def test_changed_source_review_or_relaxed_caps_reject(self):
        self.freeze()
        self.source_review_path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "source review changed"): self.run_fake()
        self.assertFalse(self.out.exists())
        self.freeze(cap=smoke.MAX_EVIDENCE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "caps invalid"): self.run_fake()
        self.assertFalse(self.out.exists())

    def test_preload_failure_consumes_one_attempt_before_worker(self):
        self.freeze()
        checks = FakeChecks(fail="preload_env")
        with self.assertRaisesRegex(ValueError, "synthetic preflight failure"):
            self.run_fake(checks)
        self.assertTrue((self.out / "failure.json").is_file())
        self.assertEqual(self.calls, [])
        with self.assertRaises(FileExistsError): self.run_fake()

    def test_output_cap_and_missing_artifact_fail_closed(self):
        self.freeze(cap=65536)
        def oversized(command, output_dir, caps, environment):
            (output_dir / "official_outputs").mkdir()
            (output_dir / "official_outputs" / "large.bin").write_bytes(b"x" * 70000)
            return {"returncode": 0, "process_id": 100,
                    "elapsed_monotonic_seconds": .1, "peak_vram_mib": 10,
                    "cap_reason": None, "logs_complete": True}
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            self.run_fake(executor=oversized)
        self.assertTrue(self.out.is_dir())
        self.assertFalse((self.out / "result.json").exists())
        self.assertTrue((self.out / "failure.json").is_file())
        self.assertTrue((self.out / "final_manifest.json").is_file())
        self.assertEqual(smoke._root_digest((self.out / "final_manifest.json").read_bytes()),
                         (self.out / "final_manifest.sha256").read_bytes())
        manifest = json.loads((self.out / "final_manifest.json").read_text())
        large = next(row for row in manifest["entries"]
                     if row["path"] == "official_outputs/large.bin")
        self.assertEqual(large["original_bytes"], 70000)
        self.assertEqual(large["original_sha256"], hashlib.sha256(b"x" * 70000).hexdigest())
        self.assertTrue(large["truncated"])
        self.assertEqual(large["retained_sha256"],
                         smoke.hash_file(self.out / large["path"])[0])
        self.assertLessEqual(smoke.evidence_bytes(self.out, 65536), 65536)

    def test_missing_transport_artifact_fails_closed(self):
        self.freeze()
        def no_output(command, output_dir, caps, environment):
            (output_dir / "stdout.log").write_text("synthetic completed")
            (output_dir / "stderr.log").write_text("")
            return {"returncode": 0, "process_id": 100,
                    "elapsed_monotonic_seconds": .1, "peak_vram_mib": 10,
                    "cap_reason": None, "logs_complete": True}
        with self.assertRaisesRegex(ValueError, "no transport artifact"):
            self.run_fake(executor=no_output)
        self.assertTrue((self.out / "failure.json").is_file())

    def test_two_artifact_aggregate_overrun_is_sealed_with_partial_hashes(self):
        self.freeze(cap=65536)
        def two_outputs(command, output_dir, caps, environment):
            folder = output_dir / "official_outputs"
            folder.mkdir()
            (folder / "one.bin").write_bytes(b"a" * 40000)
            (folder / "two.bin").write_bytes(b"b" * 40000)
            (output_dir / "stdout.log").write_text("synthetic")
            (output_dir / "stderr.log").write_text("")
            return {"returncode": 0, "process_id": 100,
                    "elapsed_monotonic_seconds": .1, "peak_vram_mib": 10,
                    "cap_reason": None, "logs_complete": True}
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            self.run_fake(executor=two_outputs)
        failure = json.loads((self.out / "failure.json").read_text())
        self.assertGreater(failure["observed_bytes_before_preservation"], 65536)
        self.assertTrue(failure["partial_preservation"])
        manifest = json.loads((self.out / "final_manifest.json").read_text())
        rows = {row["path"]: row for row in manifest["entries"]}
        self.assertEqual(rows["official_outputs/one.bin"]["original_sha256"],
                         hashlib.sha256(b"a" * 40000).hexdigest())
        self.assertEqual(rows["official_outputs/two.bin"]["original_sha256"],
                         hashlib.sha256(b"b" * 40000).hexdigest())
        self.assertTrue(any(rows[name]["truncated"] for name in
                            ("official_outputs/one.bin", "official_outputs/two.bin")))
        self.assertLessEqual(smoke.evidence_bytes(self.out, 65536), 65536)

    def test_review_key_does_not_enter_official_child_environment(self):
        os.environ["VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX"] = self.key.hex()
        os.environ["VOICECHAT_CONTROLLER_KEY_HEX"] = "c" * 64
        self.addCleanup(os.environ.pop, "VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX", None)
        self.addCleanup(os.environ.pop, "VOICECHAT_CONTROLLER_KEY_HEX", None)
        self.freeze()
        def inspect_env(command, output_dir, caps, environment):
            self.assertNotIn("VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX", environment)
            self.assertNotIn("VOICECHAT_CONTROLLER_KEY_HEX", environment)
            self.assertNotIn("VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX", os.environ)
            return self.fake_executor(command, output_dir, caps, environment)
        self.run_fake(executor=inspect_env)

    def test_sample_and_nano_ref_exact_bytes(self):
        # Override only these two functions; check the real byte validators.
        self.sample.stop()
        self.nano_ref.stop()
        wav = self.nemo / smoke.SAMPLE_WAV
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF" + b"0" * 10)
        response = self.nemo / smoke.SAMPLE_RESPONSE
        response.parent.mkdir(parents=True)
        response.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "Pinned public sample changed"):
            smoke.check_sample(self.nemo)
        ref = (self.nano / "hub" / smoke.pinned.NANO_CACHE_REPO / "refs" / "main")
        ref.parent.mkdir(parents=True)
        ref.write_bytes((smoke.pinned.NANO_COMMIT + "\n").encode())
        with self.assertRaisesRegex(ValueError, "exact bytes differ"):
            smoke.check_exact_nano_ref(self.nano)
        ref.write_bytes(smoke.pinned.NANO_COMMIT.encode())
        self.assertEqual(smoke.check_exact_nano_ref(self.nano),
                         smoke.sha(ref.read_bytes()))

    def test_nano_ref_resolves_with_huggingface_hub(self):
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            self.skipTest("huggingface-hub is unavailable in this no-model test environment")
        self.nano_ref.stop()
        repo = self.nano / "hub" / smoke.pinned.NANO_CACHE_REPO
        ref = repo / "refs" / "main"
        snapshot = repo / "snapshots" / smoke.pinned.NANO_COMMIT
        snapshot.mkdir(parents=True)
        ref.parent.mkdir(parents=True)
        config = snapshot / "config.json"
        config.write_text("{}")
        ref.write_bytes(smoke.pinned.NANO_COMMIT.encode("ascii"))
        self.assertEqual(
            try_to_load_from_cache(smoke.pinned.NANO_REPO_ID, "config.json",
                                   cache_dir=str(self.nano / "hub"), revision="main"),
            str(config))
        ref.write_bytes((smoke.pinned.NANO_COMMIT + "\n").encode("ascii"))
        self.assertIsNone(
            try_to_load_from_cache(smoke.pinned.NANO_REPO_ID, "config.json",
                                   cache_dir=str(self.nano / "hub"), revision="main"))
        with self.assertRaisesRegex(ValueError, "exact bytes differ"):
            smoke.check_exact_nano_ref(self.nano)

    def test_command_and_environment_are_freeze_bound(self):
        self.freeze()
        os.environ["PYTHONHASHSEED"] = "123"
        self.addCleanup(os.environ.pop, "PYTHONHASHSEED", None)
        with self.assertRaisesRegex(ValueError, "freeze scope differs"):
            self.run_fake()
        self.assertFalse(self.out.exists())

    def test_full_installed_environment_drift_rejected_before_output(self):
        self.freeze()
        changed = copy.deepcopy(self.environment_receipt_mock.return_value)
        changed["pip_freeze_all_sorted_sha256"] = "f" * 64
        self.environment_receipt_mock.return_value = changed
        with self.assertRaisesRegex(ValueError, "freeze scope differs"):
            self.run_fake()
        self.assertFalse(self.out.exists())

    def test_full_installed_environment_drift_rejected_after_preload(self):
        self.freeze()
        original = copy.deepcopy(self.environment_receipt_mock.return_value)
        changed = copy.deepcopy(original)
        changed["mamba_dependency_versions"]["tilelang"] = "0.1.8"
        self.environment_receipt_mock.side_effect = [original, changed]
        with self.assertRaisesRegex(ValueError, "bound source/environment changed after preload"):
            self.run_fake()
        self.assertEqual(self.calls, [])
        self.assertTrue((self.out / "failure.json").is_file())

    def test_native_ffmpeg_drift_rejected_before_output_and_after_preload(self):
        self.freeze()
        original = copy.deepcopy(self.native_receipt_mock.return_value)
        changed = copy.deepcopy(original)
        changed["library_targets"] = {"libavcodec.so.61": {"sha256": "f" * 64}}
        self.native_receipt_mock.return_value = changed
        with self.assertRaisesRegex(ValueError, "freeze scope differs"):
            self.run_fake()
        self.assertFalse(self.out.exists())
        self.native_receipt_mock.side_effect = [original, changed]
        with self.assertRaisesRegex(ValueError, "bound source/environment changed after preload"):
            self.run_fake()
        self.assertEqual(self.calls, [])
        self.assertTrue((self.out / "failure.json").is_file())

    def test_installed_environment_receipt_sorts_full_freeze_and_checks_versions(self):
        try:
            import packaging.specifiers  # noqa: F401
        except ImportError:
            self.skipTest("packaging is unavailable in this no-model test environment")
        self.environment_receipt.stop()
        versions = {**smoke.RUNNER_PACKAGE_PINS,
                    "tilelang": "0.1.8", "apache-tvm-ffi": "0.1.9",
                    "quack-kernels": "0.3.4"}
        def installed_version(name):
            return versions[name]
        first = SimpleNamespace(returncode=0, stdout="zpkg==1\nnumpy==1.26.4\nApkg==2\n")
        second = SimpleNamespace(returncode=0, stdout="Apkg==2\nzpkg==1\nnumpy==1.26.4\n")
        with patch.object(smoke.subprocess, "run", side_effect=[first, second]) as run_pip, \
             patch.object(smoke.importlib.metadata, "version", side_effect=installed_version):
            left = smoke.installed_environment_receipt()
            right = smoke.installed_environment_receipt()
        self.assertEqual(left, right)
        self.assertEqual(left["pip_freeze_all_line_count"], 3)
        self.assertEqual(left["pip_freeze_all_sorted_sha256"],
                         smoke.sha(b"Apkg==2\nnumpy==1.26.4\nzpkg==1\n"))
        self.assertEqual(run_pip.call_args.args[0],
                         [sys.executable, "-m", "pip", "freeze", "--all"])
        versions["pyarrow"] = "26.0.1"
        with patch.object(smoke.subprocess, "run", return_value=first), \
             patch.object(smoke.importlib.metadata, "version", side_effect=installed_version):
            with self.assertRaisesRegex(ValueError, "pyarrow==25.0.1"):
                smoke.installed_environment_receipt()
        versions["pyarrow"] = "25.0.1"
        versions["tilelang"] = "0.1.9"
        with patch.object(smoke.subprocess, "run", return_value=first), \
             patch.object(smoke.importlib.metadata, "version", side_effect=installed_version):
            with self.assertRaisesRegex(ValueError, "tilelang violates ==0.1.8"):
                smoke.installed_environment_receipt()

    def test_extra_file_symlink_and_tamper_are_rejected(self):
        self.freeze()
        self.run_fake()
        files, _ = smoke.evidence_inventory(self.out, 65536)
        self.assertIn("official_outputs/sample_fc.txt", files)
        link = self.out / "official_outputs" / "external"
        link.symlink_to(self.root / "freeze.json")
        with self.assertRaisesRegex(ValueError, "link rejected"):
            smoke.evidence_inventory(self.out, 65536)

    def test_worker_symlink_failure_keeps_hashed_partial_manifest(self):
        self.freeze()
        def linked_output(command, output_dir, caps, environment):
            folder = output_dir / "official_outputs"
            folder.mkdir()
            (folder / "sample_fc.txt").write_text("partial")
            (folder / "link").symlink_to(self.freeze_path)
            (output_dir / "stdout.log").write_text("partial log")
            (output_dir / "stderr.log").write_text("")
            return {"returncode": 0, "process_id": 100,
                    "elapsed_monotonic_seconds": .1, "peak_vram_mib": 10,
                    "cap_reason": None, "logs_complete": True}
        with self.assertRaisesRegex(ValueError, "link rejected"):
            self.run_fake(executor=linked_output)
        manifest = json.loads((self.out / "final_manifest.json").read_text())
        self.assertEqual(manifest["removed_nonfiles"][0]["kind"], "symlink_removed")
        self.assertIn("failure.json", {row["path"] for row in manifest["entries"]})
        self.assertLessEqual(smoke.evidence_bytes(self.out, 65536), 65536)

    def test_real_supervisor_loop_with_fake_subprocess_drains_logs(self):
        self.out.mkdir()
        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.pid = 999_991
                (self.out_dir / "official_outputs").mkdir()
                (self.out_dir / "official_outputs" / "sample_fc.txt").write_text("fake")
                self.stdout = self._pipe(b"fake stdout\n")
                self.stderr = self._pipe(b"fake stderr\n")
            @staticmethod
            def _pipe(data):
                read, write = os.pipe()
                os.write(write, data)
                os.close(write)
                return os.fdopen(read, "rb", buffering=0)
            def poll(self): return 0
            def wait(self, timeout=None): return 0
        FakeProcess.out_dir = self.out
        caps = {"wall_seconds": 10, "vram_mib": 1000,
                "total_evidence_bytes": 8192}
        with patch.object(smoke.subprocess, "Popen", FakeProcess):
            result = smoke.execute_official(["fake", "--four-fixed-args"],
                                            self.out, caps, {}, telemetry=lambda pid: 12)
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["logs_complete"])
        self.assertEqual(result["peak_vram_mib"], 12)
        self.assertEqual(json.loads((self.out / "process_start.json").read_text())
                         ["process_id"], 999_991)
        self.assertEqual((self.out / "stdout.log").read_bytes(), b"fake stdout\n")
        self.assertEqual((self.out / "stderr.log").read_bytes(), b"fake stderr\n")

    def test_real_supervisor_loop_with_fake_subprocess_stops_on_log_cap(self):
        self.out.mkdir()
        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.pid = 999_992
                self.stdout = self._pipe(b"a" * 3000)
                self.stderr = self._pipe(b"b" * 3000)
            @staticmethod
            def _pipe(data):
                read, write = os.pipe()
                os.write(write, data)
                os.close(write)
                return os.fdopen(read, "rb", buffering=0)
            def poll(self): return 0
            def wait(self, timeout=None): return 0
        caps = {"wall_seconds": 10, "vram_mib": 1000,
                "total_evidence_bytes": 4096}
        with patch.object(smoke.subprocess, "Popen", FakeProcess), \
             patch.object(smoke, "_kill_tree") as killed:
            result = smoke.execute_official(["fake"], self.out, caps, {},
                                            telemetry=lambda pid: 12)
        self.assertEqual(result["cap_reason"], "log_evidence_cap")
        self.assertFalse(result["logs_complete"])
        killed.assert_called_once()
        self.assertLessEqual(smoke.evidence_bytes(self.out, 4096), 4096)

    def test_abnormal_telemetry_kills_fake_process_and_seals_failure(self):
        self.freeze()
        def telemetry_failure(command, output_dir, caps, environment):
            (output_dir / "process_start.json").write_text('{"process_id":123}')
            (output_dir / "stdout.log").write_text("partial log")
            (output_dir / "official_outputs").mkdir()
            (output_dir / "official_outputs" / "sample_fc.txt").write_text("partial")
            raise RuntimeError("synthetic telemetry unavailable")
        with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
            self.run_fake(executor=telemetry_failure)
        manifest = json.loads((self.out / "final_manifest.json").read_text())
        names = {row["path"] for row in manifest["entries"]}
        self.assertIn("process_start.json", names)
        self.assertIn("stdout.log", names)
        self.assertIn("official_outputs/sample_fc.txt", names)
        self.assertIn("failure.json", names)

    def test_supervisor_telemetry_exception_kills_fake_process(self):
        self.out.mkdir()
        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.pid = 999_993
                self.stdout = self._pipe()
                self.stderr = self._pipe()
            @staticmethod
            def _pipe():
                read, write = os.pipe()
                os.close(write)
                return os.fdopen(read, "rb", buffering=0)
            def poll(self): return None
            def wait(self, timeout=None): return 1
        def broken_telemetry(pid):
            raise RuntimeError("synthetic nvidia-smi failure")
        caps = {"wall_seconds": 10, "vram_mib": 1000,
                "total_evidence_bytes": 65536}
        with patch.object(smoke.subprocess, "Popen", FakeProcess), \
             patch.object(smoke, "_kill_tree") as killed:
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi failure"):
                smoke.execute_official(["fake"], self.out, caps, {},
                                       telemetry=broken_telemetry)
        killed.assert_called_once()
        self.assertEqual(json.loads((self.out / "process_start.json").read_text())
                         ["process_id"], 999_993)


class NativeFFmpegReceiptTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.root = self.base / "voicechat-sample-ffmpeg-conda7"
        self.libdir = self.root / "lib"
        self.libdir.mkdir(parents=True)
        self.metadata = self.root / "conda-meta"
        self.metadata.mkdir()
        archive_name = "ffmpeg-7.0.0-h123_0.conda"
        archive_bytes = b"synthetic conda package archive"
        archive_dir = self.root / "package-archives"
        archive_dir.mkdir()
        (archive_dir / archive_name).write_bytes(archive_bytes)
        (self.metadata / "ffmpeg-7.0.0-h123_0.json").write_bytes(smoke.canonical({
            "name": "ffmpeg", "sha256": smoke.sha(archive_bytes)}))
        spec = b"@EXPLICIT\nhttps://conda.anaconda.org/conda-forge/linux-64/ffmpeg-7.0.0-h123_0.conda#" + b"a" * 32 + b"\n"
        (self.root / "conda-explicit.txt").write_bytes(spec)
        (self.root / "native-source-manifest.json").write_bytes(smoke.canonical({
            "schema": "voicechat_native_source_manifest_v1", "kind": "conda_explicit",
            "ffmpeg_major": 7, "explicit_spec": {
                "path": "conda-explicit.txt", "sha256": smoke.sha(spec), "bytes": len(spec)},
            "package_archives": [{"path": f"package-archives/{archive_name}",
                                  "sha256": smoke.sha(archive_bytes),
                                  "bytes": len(archive_bytes)}],
        }))
        for soname in smoke.FFMPEG_REQUIRED_SONAMES.values():
            (self.libdir / soname).write_bytes(soname.encode())
        (self.libdir / "libavcodec.so").symlink_to("libavcodec.so.61")
        self.core = self.base / "libtorchcodec_core7.so"
        self.core.write_bytes(b"synthetic torchcodec core 7")
        self.env = patch.dict(os.environ, {"LD_LIBRARY_PATH": str(self.libdir)}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for target, name, value in [
            (smoke, "_torchcodec_core_path", self.core),
            (smoke, "_os_release_identity", {"id": "ubuntu", "version_id": "22.04"}),
            (smoke, "check_sample", smoke.SAMPLE_SHA256),
            (smoke.platform, "system", "Linux"),
            (smoke.platform, "machine", "x86_64"),
        ]:
            mocked = patch.object(target, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_receipt_binds_every_alias_target_core_source_and_wav(self):
        receipt = smoke.native_ffmpeg_runtime(self.base)
        self.assertEqual(receipt["ffmpeg_major"], 7)
        self.assertEqual(receipt["required_sonames"], smoke.FFMPEG_REQUIRED_SONAMES)
        self.assertEqual(receipt["library_aliases"]["libavcodec.so"]["resolved_target"],
                         "libavcodec.so.61")
        self.assertEqual(receipt["library_targets"]["libavcodec.so.61"]["sha256"],
                         smoke.sha(b"libavcodec.so.61"))
        self.assertEqual(receipt["torchcodec_core"]["sha256"],
                         smoke.sha(b"synthetic torchcodec core 7"))
        self.assertEqual(receipt["public_wav_sha256"], smoke.SAMPLE_SHA256[smoke.SAMPLE_WAV])
        self.assertEqual(receipt["source"]["explicit_spec"]["package_count"], 1)
        self.assertEqual(receipt["source"]["package_archives_count"], 1)

    def test_tampered_target_and_source_spec_reject_or_change_receipt(self):
        first = smoke.native_ffmpeg_runtime(self.base)
        (self.libdir / "libavcodec.so.61").write_bytes(b"changed")
        second = smoke.native_ffmpeg_runtime(self.base)
        self.assertNotEqual(first, second)
        (self.root / "conda-explicit.txt").write_text("@EXPLICIT\nchanged\n")
        with self.assertRaisesRegex(ValueError, "Conda explicit spec changed"):
            smoke.native_ffmpeg_runtime(self.base)

    def test_symlink_escape_missing_soname_and_unexpected_major_reject(self):
        (self.libdir / "libavutil.so.59").unlink()
        (self.libdir / "libavutil.so.59").symlink_to(self.core)
        with self.assertRaisesRegex(ValueError, "symlink escapes"):
            smoke.native_ffmpeg_runtime(self.base)
        (self.libdir / "libavutil.so.59").unlink()
        with self.assertRaisesRegex(ValueError, "Required FFmpeg soname missing"):
            smoke.native_ffmpeg_runtime(self.base)
        (self.libdir / "libavutil.so.59").write_bytes(b"restored")
        (self.libdir / "libavutil.so.58").write_bytes(b"wrong major")
        with self.assertRaisesRegex(ValueError, "Unexpected FFmpeg library major"):
            smoke.native_ffmpeg_runtime(self.base)

    def test_archive_or_installed_metadata_tamper_rejects(self):
        archive = self.root / "package-archives" / "ffmpeg-7.0.0-h123_0.conda"
        archive.write_bytes(b"different archive bytes")
        with self.assertRaisesRegex(ValueError, "archive bytes changed"):
            smoke.native_ffmpeg_runtime(self.base)
        archive.write_bytes(b"synthetic conda package archive")
        metadata = self.metadata / "ffmpeg-7.0.0-h123_0.json"
        metadata.write_bytes(smoke.canonical({"name": "ffmpeg", "sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "does not match installed metadata"):
            smoke.native_ffmpeg_runtime(self.base)

    def test_unlisted_archive_rejects(self):
        (self.root / "package-archives" / "unlisted.conda").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            smoke.native_ffmpeg_runtime(self.base)


if __name__ == "__main__":
    unittest.main()
