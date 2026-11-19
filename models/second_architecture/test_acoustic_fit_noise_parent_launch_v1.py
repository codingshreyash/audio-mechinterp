"""Fake-process and host admission checks; never loads a candidate model."""
from __future__ import annotations

import gzip
import json
import plistlib
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acoustic_fit_noise_parent_launch_v1 as parent


class ParentLauncherTests(unittest.TestCase):
    def test_bounded_process_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, timed_out, elapsed = parent.run_bounded(
                [sys.executable, "-c", "print('fake')"],
                root / "stdout", root / "stderr", timeout_seconds=2)
            self.assertEqual(rc, 0)
            self.assertFalse(timed_out)
            self.assertLess(elapsed, 2)
            self.assertIn("fake", (root / "stdout").read_text())

    def test_timeout_kills_fake_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, timed_out, elapsed = parent.run_bounded(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                root / "stdout", root / "stderr", timeout_seconds=0.05)
            self.assertTrue(timed_out)
            self.assertNotEqual(rc, 0)
            self.assertLess(elapsed, 2)

    def test_worker_review_failure_does_not_create_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "attempt"
            with (mock.patch.object(parent.pilot, "preflight", return_value={}),
                  mock.patch.object(parent.pilot, "require_review", side_effect=ValueError("HOLD"))):
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    parent.execute("whisper", root / "receipt", root / "worker_go",
                                   root / "parent_go", attempt)
            self.assertFalse(attempt.exists())

    def test_worker_matrix_rejects_duplicate_and_wrong_pcm(self):
        # Use real frozen case names/geometries and synthetic hashes, no model.
        expected = {key: {arm: f"sha_{key}_{arm}" for arm in parent.pilot.ARMS}
                    for key in ("piper_silver", "piper_teal", "kokoro_silver", "kokoro_teal")}
        cases = []
        for key in sorted(expected):
            for arm in parent.pilot.ARMS:
                metrics = {}
                for stage in parent.WHISPER_STAGES:
                    shape, window = parent.expected_geometry("whisper", key, stage)
                    metrics[stage] = {"shape": shape, "nominal_window": window,
                                      "sha256": "a" * 64, "whole_l2": 0.0,
                                      "word_l2": 0.0, "outside_l2": 0.0,
                                      "outside_maxabs": 0.0}
                cases.append({"type": "case", "model": "whisper", "case": key,
                              "arm": arm, "pcm_sha256": expected[key][arm],
                              "output": {"token_ids": [1, 2], "transcript": "silver",
                                         "eos_seen": True, "censored": False},
                              "hook_counts": {s: 1 for s in parent.WHISPER_STAGES[1:]},
                              "metrics": metrics, "elapsed_seconds": len(cases) + 1,
                              "rss_bytes": 100_000_000})
        rows = [{"type": "start", "model": "whisper", "preflight_sha256": "pre",
                 "review_sha256": "review"}, *cases,
                {"type": "complete", "planned_cases": 20, "elapsed_seconds": 21}]
        check = lambda value: parent.valid_worker_rows(value, "whisper", expected, "pre", "review")
        self.assertTrue(check(rows))
        duplicate = [*rows]
        duplicate[1] = duplicate[2]
        self.assertFalse(check(duplicate))
        wrong = [*rows]
        wrong[1] = {**wrong[1], "pcm_sha256": "wrong"}
        self.assertFalse(check(wrong))
        drift = [*rows]
        drift[2] = {**drift[2], "output": {**drift[2]["output"], "transcript": "teal"}}
        self.assertFalse(check(drift))
        state_drift = [*rows]
        edited = {**state_drift[2]["metrics"]}
        edited["post_block0"] = {**edited["post_block0"], "sha256": "b" * 64}
        state_drift[2] = {**state_drift[2], "metrics": edited}
        self.assertFalse(check(state_drift))
        missing_metric = [*rows]
        missing_metric[3] = {**missing_metric[3], "metrics": {}}
        self.assertFalse(check(missing_metric))
        bool_metric = [*rows]
        edited = {**bool_metric[3]["metrics"]}
        edited["post_block0"] = {**edited["post_block0"], "whole_l2": True}
        bool_metric[3] = {**bool_metric[3], "metrics": edited}
        self.assertFalse(check(bool_metric))
        over_time = [*rows]
        over_time[-1] = {**over_time[-1], "elapsed_seconds": 1801}
        self.assertFalse(check(over_time))
        self.assertFalse(parent.valid_worker_rows(rows, "whisper", expected, "wrong", "review"))

    def test_elapsed_deadline_even_if_wait_returns_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = mock.Mock()
            fake.returncode = 0
            fake.wait.side_effect = lambda **_kw: time.sleep(0.02)
            with mock.patch.object(parent.subprocess, "Popen", return_value=fake):
                rc, timed_out, elapsed = parent.run_bounded(
                    [sys.executable], root / "out", root / "err", timeout_seconds=0.001)
            self.assertEqual(rc, 0)
            self.assertTrue(timed_out)
            self.assertGreater(elapsed, 0.001)

    def test_wav2vec2_metrics_and_full_ids_are_required(self):
        expected = {key: {arm: f"pcm_{key}_{arm}" for arm in parent.pilot.ARMS}
                    for key in ("piper_silver", "piper_teal", "kokoro_silver", "kokoro_teal")}
        cases = []
        for key in sorted(expected):
            for arm in parent.pilot.ARMS:
                metrics = {}
                for stage in parent.WAV_STAGES:
                    shape, window = parent.expected_geometry("wav2vec2", key, stage)
                    metrics[stage] = {"shape": shape, "nominal_window": window,
                                      "sha256": "a" * 64, "whole_l2": 0.0,
                                      "word_l2": 0.0, "outside_l2": 0.0,
                                      "outside_maxabs": 0.0}
                frames = metrics["logits"]["shape"][1]
                cases.append({"type": "case", "model": "wav2vec2", "case": key,
                              "arm": arm, "pcm_sha256": expected[key][arm],
                              "output": {"argmax_ids": [0] * frames, "transcript": "SILVER"},
                              "hook_counts": {s: 1 for s in parent.WAV_STAGES},
                              "metrics": metrics, "elapsed_seconds": len(cases) + 1,
                              "rss_bytes": 100_000_000})
        rows = [{"type": "start", "model": "wav2vec2",
                 "preflight_sha256": "pre", "review_sha256": "review"},
                *cases, {"type": "complete", "planned_cases": 20,
                         "elapsed_seconds": 21}]
        check = lambda value: parent.valid_worker_rows(value, "wav2vec2", expected, "pre", "review")
        self.assertTrue(check(rows))
        short_ids = [*rows]
        short_ids[1] = {**short_ids[1], "output": {"argmax_ids": [0], "transcript": "SILVER"}}
        self.assertFalse(check(short_ids))
        missing_logits = [*rows]
        missing_logits[3] = {**missing_logits[3],
                             "metrics": {k: v for k, v in missing_logits[3]["metrics"].items()
                                         if k != "logits"}}
        self.assertFalse(check(missing_logits))

    def test_host_evidence_binding_rejects_changed_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("pre", "post"):
                packed = gzip.compress(b"bounded host log", mtime=0)
                (root / f"{label}_power_log.txt.gz").write_bytes(packed)
                (root / f"{label}_host.json").write_text(json.dumps({
                    "power_log_archive": {
                        "file": f"{label}_power_log.txt.gz", "raw_bytes": 16,
                        "raw_sha256": parent.hashlib.sha256(b"bounded host log").hexdigest(),
                        "compressed_sha256": parent.pilot.sha_file(root / f"{label}_power_log.txt.gz")}}))
            inv = parent.evidence_inventory(root)
            self.assertTrue(parent.host_sidecars_valid(root, inv))
            (root / "post_power_log.txt.gz").write_bytes(b"changed")
            self.assertFalse(parent.host_sidecars_valid(root, parent.evidence_inventory(root)))

    def test_read_only_audit_rejects_saved_evidence_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "dummy.txt").write_text("first")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"cases": {}}))
            worker_review = root / "worker_go.json"
            worker_review.write_text("{}")
            launcher_review = root / "launcher_go.json"
            launcher_review.write_text("{}")
            binding = {"source": "fake"}
            result = {"schema": "acoustic_fit_noise_parent_result_v1",
                      "model": "whisper", "attempt_path": str(attempt.resolve()),
                      "source_binding": binding,
                      "preflight_sha256": parent.pilot.sha_file(receipt),
                      "worker_review_sha256": parent.pilot.sha_file(worker_review),
                      "launcher_review_sha256": parent.pilot.sha_file(launcher_review),
                      "status": "invalid_or_partial",
                      "evidence_sha256": parent.evidence_inventory(attempt)}
            result["capture_bytes_before_result"] = sum(
                value["bytes"] for value in result["evidence_sha256"].values())
            (attempt / "launch_result.json").write_text(json.dumps(result))
            with (mock.patch.object(parent, "source_binding", return_value=binding),
                  mock.patch.object(parent.pilot, "require_review"),
                  mock.patch.object(parent, "require_launcher_go")):
                valid = parent.audit_saved_attempt("whisper", receipt, worker_review,
                                                   launcher_review, attempt)
                self.assertEqual(valid["status"], "bound_invalid_or_partial")
                (attempt / "dummy.txt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                    parent.audit_saved_attempt("whisper", receipt, worker_review,
                                               launcher_review, attempt)
                result["evidence_sha256"] = parent.evidence_inventory(attempt, allow_result=True)
                (attempt / "launch_result.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "capture byte count mismatch"):
                    parent.audit_saved_attempt("whisper", receipt, worker_review,
                                               launcher_review, attempt)
                (attempt / "large.bin").write_bytes(b"x" * (17 * 1024**2))
                result["evidence_sha256"] = parent.evidence_inventory(attempt, allow_result=True)
                result["capture_bytes_before_result"] = sum(
                    value["bytes"] for value in result["evidence_sha256"].values())
                result["status"] = "complete"
                (attempt / "launch_result.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "byte cap"):
                    parent.audit_saved_attempt("whisper", receipt, worker_review,
                                               launcher_review, attempt)

    def test_timeline_rejects_negative_and_bool_intervals(self):
        result = {"start_utc": "2026-11-19T01:00:00+00:00",
                  "end_utc": "2026-11-19T01:00:10+00:00",
                  "start_monotonic": 100.0, "end_monotonic": 110.0,
                  "wall_seconds": 10.0, "monotonic_seconds": 10.0}
        pre = "2026-11-19T01:00:01+00:00"
        post = "2026-11-19T01:00:09+00:00"
        self.assertTrue(parent.valid_timeline(result, pre, post))
        self.assertFalse(parent.valid_timeline({**result, "end_monotonic": 90.0}, pre, post))
        self.assertFalse(parent.valid_timeline({**result, "wall_seconds": True}, pre, post))
        self.assertFalse(parent.valid_timeline(result, post, pre))

    def test_launcher_review_cannot_share_worker_review_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker.json"
            worker.write_text("{}")
            review = root / "parent.json"
            binding = {"launcher_commit": "a" * 40, "tests_commit": "a" * 40,
                       "handoff_commit": "a" * 40}
            review.write_text(json.dumps({
                "schema": "acoustic_fit_noise_parent_source_go_v1", "status": "GO",
                "source_binding": binding, "preflight_sha256": "pre",
                "worker_review_sha256": parent.pilot.sha_file(worker)}))
            with (mock.patch.object(parent.pilot, "source_commit", return_value="b" * 40),
                  mock.patch.object(parent.subprocess, "run", return_value=mock.Mock(returncode=0))):
                with self.assertRaisesRegex(ValueError, "not later"):
                    parent.require_launcher_go(review, binding, "pre", worker)

    def test_closed_lid_rejected_before_probe_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = {"collected_utc": parent.utc(), "outputs": {
                "power": "Now drawing from 'AC Power'",
                "clamshell": plistlib.dumps([{"AppleClamshellState": True}]).decode(),
                "battery": plistlib.dumps([{"ExternalConnected": True}]).decode(),
                "power_log": "fake"}}
            with mock.patch.object(parent.host_gate, "_host_probe", return_value=evidence):
                with self.assertRaisesRegex(ValueError, "lid is closed"):
                    parent.probe_host(root, "pre")
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
