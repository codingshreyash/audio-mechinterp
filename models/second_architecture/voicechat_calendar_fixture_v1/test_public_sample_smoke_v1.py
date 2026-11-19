"""CPU-only synthetic admission and launcher tests; never load VoiceChat."""

from __future__ import annotations

import hmac
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
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
        self.freeze(cap=8192)
        def oversized(command, output_dir, caps, environment):
            (output_dir / "official_outputs").mkdir()
            (output_dir / "official_outputs" / "large.bin").write_bytes(b"x" * 9000)
            return {"returncode": 0, "process_id": 100,
                    "elapsed_monotonic_seconds": .1, "peak_vram_mib": 10,
                    "cap_reason": None, "logs_complete": True}
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            self.run_fake(executor=oversized)
        self.assertTrue(self.out.is_dir())
        self.assertFalse((self.out / "result.json").exists())

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
        ref.write_bytes(smoke.pinned.NANO_COMMIT.encode())
        with self.assertRaisesRegex(ValueError, "exact bytes differ"):
            smoke.check_exact_nano_ref(self.nano)
        ref.write_bytes((smoke.pinned.NANO_COMMIT + "\n").encode())
        self.assertEqual(smoke.check_exact_nano_ref(self.nano),
                         smoke.sha(ref.read_bytes()))

    def test_command_and_environment_are_freeze_bound(self):
        self.freeze()
        os.environ["PYTHONHASHSEED"] = "123"
        self.addCleanup(os.environ.pop, "PYTHONHASHSEED", None)
        with self.assertRaisesRegex(ValueError, "freeze scope differs"):
            self.run_fake()
        self.assertFalse(self.out.exists())

    def test_extra_file_symlink_and_tamper_are_rejected(self):
        self.freeze()
        self.run_fake()
        files, _ = smoke.evidence_inventory(self.out, 65536)
        self.assertIn("official_outputs/sample_fc.txt", files)
        link = self.out / "official_outputs" / "external"
        link.symlink_to(self.root / "freeze.json")
        with self.assertRaisesRegex(ValueError, "link rejected"):
            smoke.evidence_inventory(self.out, 65536)

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


if __name__ == "__main__":
    unittest.main()
