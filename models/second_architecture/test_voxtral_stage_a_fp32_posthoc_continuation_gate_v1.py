"""CPU-only admission checks for the Stage-A FP32 continuation scaffold."""

from __future__ import annotations

import unittest
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

    def test_missing_receipt_stops_before_freeze_or_model(self) -> None:
        with (mock.patch.object(gate, "verify_preserved_bf16", return_value={}),
              mock.patch.object(gate, "verify_continuation_freeze") as freeze,
              mock.patch.object(gate.old, "run_bounded_child") as child):
            with self.assertRaisesRegex(ValueError, "Uncommitted or changed bound file"):
                gate.admission(gate.RUNS / "fp32" / "test_attempt")
            freeze.assert_not_called()
            child.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
