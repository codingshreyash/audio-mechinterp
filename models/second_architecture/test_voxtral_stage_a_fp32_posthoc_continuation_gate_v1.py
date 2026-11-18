"""CPU-only admission checks for the Stage-A FP32 continuation scaffold."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import voxtral_stage_a_fp32_posthoc_continuation_gate_v1 as gate


class ContinuationGateTest(unittest.TestCase):
    def test_preserved_original_failure_and_all_manifest_bytes(self) -> None:
        saved = gate.verify_preserved_bf16()
        self.assertEqual(saved["finish"]["status"], "launch_failed")
        self.assertEqual(saved["finish"]["auditor_returncode"], 1)
        self.assertEqual(saved["finish"]["audit_sha256"], None)
        self.assertEqual(len(saved["manifest"]["files"]), 12)
        self.assertEqual(sum(not x["git_committed"] for x in saved["manifest"]["files"]), 2)

    def test_committed_posthoc_receipt_binds_reviewed_raw(self) -> None:
        saved = gate.verify_preserved_bf16()
        posthoc = gate.verify_posthoc_receipt(saved)
        self.assertEqual(posthoc["receipt"]["original_launch_status"], "launch_failed")
        self.assertEqual(posthoc["review_commit"], "1f5f4f4dd7f1ce5258f0552f6515999617875cf2")

    def test_invalid_raw_hash_blocks_before_external_receipt(self) -> None:
        original_sha = gate.old.sha

        def tampered_hash(path):
            if path == gate.ORIGINAL_ATTEMPT / "launch_finish.json":
                return "0" * 64
            return original_sha(path)

        with (mock.patch.object(gate.old, "sha", side_effect=tampered_hash),
              mock.patch.object(gate, "verify_posthoc_receipt") as posthoc):
            with self.assertRaisesRegex(ValueError, "Original evidence changed"):
                gate.admission(gate.RUNS / "fp32" / "test_attempt")
            posthoc.assert_not_called()

    def test_later_committed_freeze_timestamp_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            def git(*args: str) -> str:
                result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                                        capture_output=True, text=True)
                return result.stdout.strip()

            def commit(message: str) -> str:
                git("add", ".")
                git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-q", "-m", message)
                return git("rev-parse", "HEAD")

            git("init", "-q")
            (repo / "source.txt").write_text("reviewed source\n")
            source_commit = commit("source review")
            bound = {
                "posthoc_receipt_sha256": repo / "receipt.json",
                "posthoc_audit_sha256": repo / "audit.json",
                "posthoc_result_review_sha256": repo / "result_review.md",
                "posthoc_review_sha256": repo / "receipt_review.md",
                "continuation_launcher_sha256": repo / "launcher.py",
                "continuation_tests_sha256": repo / "tests.py",
                "continuation_protocol_sha256": repo / "protocol.md",
                "continuation_review_sha256": repo / "launcher_review.md",
            }
            for key in tuple(bound)[:4]:
                bound[key].write_text(key + "\n")
            receipt_review_commit = commit("receipt review")
            for key in tuple(bound)[4:]:
                bound[key].write_text(key + "\n")
            bound["continuation_review_sha256"].write_text(
                "Decision: GO for FP32-only launcher source; model run HOLD until external freeze.\n")
            launcher_review_commit = commit("launcher review")
            freeze_path = repo / "freeze.json"
            freeze = {
                "status": gate.FREEZE_STATUS, "permitted_arm": "fp32",
                "audit_scope": gate.SCOPE, "receipt_review_commit": receipt_review_commit,
                "launcher_review_commit": launcher_review_commit,
                "reviewed_utc": "2026-11-18T14:00:00+00:00",
                **{key: gate.old.sha(path) for key, path in bound.items()},
            }
            freeze_path.write_text(json.dumps(freeze, sort_keys=True) + "\n")
            freeze_copy = repo / "freeze_copy.json"
            freeze_copy.write_bytes(freeze_path.read_bytes())
            commit("external freeze")
            posthoc = {"review_commit": receipt_review_commit,
                       "receipt_sha256": gate.old.sha(bound["posthoc_receipt_sha256"])}
            with (mock.patch.object(gate, "REPO", repo),
                  mock.patch.object(gate, "FREEZE", freeze_path),
                  mock.patch.object(gate, "POSTHOC_AUDIT", bound["posthoc_audit_sha256"]),
                  mock.patch.object(gate, "REVIEW", bound["continuation_review_sha256"]),
                  mock.patch.object(gate, "BOUND_FILES", bound),
                  mock.patch.object(gate, "FIXED_PINS", {}),
                  mock.patch.object(gate, "V2_SOURCE_REVIEW_COMMIT", source_commit)):
                accepted = gate.verify_continuation_freeze({}, posthoc)
                self.assertEqual(accepted["receipt"]["reviewed_utc"], freeze["reviewed_utc"])
                freeze_path.unlink()
                freeze_path.symlink_to(freeze_copy.name)
                self.assertIn("T freeze.json", git("status", "--short"))
                # The older helper resolves the symlink to the same-byte copy.
                gate.old.committed_same(repo, "HEAD", freeze_path)
                with self.assertRaisesRegex(ValueError, "contains a symlink"):
                    gate.verify_continuation_freeze({}, posthoc)
                freeze_path.unlink()
                freeze_path.write_text(json.dumps(freeze, sort_keys=True) + "\n")
                freeze["reviewed_utc"] = "2026-11-18T14:00:01+00:00"
                freeze_path.write_text(json.dumps(freeze, sort_keys=True) + "\n")
                commit("later freeze timestamp edit")
                with self.assertRaisesRegex(ValueError, "changed after first-add"):
                    gate.verify_continuation_freeze({}, posthoc)


if __name__ == "__main__":
    unittest.main()
