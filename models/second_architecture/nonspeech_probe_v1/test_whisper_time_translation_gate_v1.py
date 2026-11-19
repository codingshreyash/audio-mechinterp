"""No-model admission checks for the three time-translation stages."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import whisper_time_translation_gate_v1 as gate


class GateTests(unittest.TestCase):
    def test_three_distinct_fixed_one_shot_paths(self):
        self.assertEqual(set(gate.ATTEMPTS), set(gate.STAGES))
        self.assertEqual(len(set(gate.ATTEMPTS.values())), 3)
        self.assertEqual(len(set(gate.GOS.values())), 3)
        self.assertTrue(all(path.name.endswith("attempt_001")
                            for path in gate.ATTEMPTS.values()))

    def test_admission_binds_stage_plan_source_and_limits(self):
        with patch.object(gate, "source_identity", return_value={"source.py": "a" * 64}), \
             patch.object(gate.v4, "admission", return_value={"model": "whisper_multilingual"}):
            original = gate.admission("original")
            shifted = gate.admission("offset_8000")
        self.assertNotEqual(original["attempt"], shifted["attempt"])
        self.assertEqual(original["plan_sha256"], shifted["plan_sha256"])
        self.assertEqual(original["limits"]["tree_rss_bytes"], 4 * 1024**3)
        self.assertEqual(original["limits"]["evidence_bytes"], 1024**3)
        self.assertEqual(original["limits"]["max_power_probe_gap_seconds"], 15)
        with self.assertRaisesRegex(ValueError, "unknown"):
            gate.admission("offset_4000")

    def test_missing_fixed_go_rejected_before_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "go.json"
            with self.assertRaisesRegex(ValueError, "fixed stage"):
                gate.validate_go("original", gate.ADMISSIONS["original"], wrong)

    def test_source_census_includes_execution_and_auditor(self):
        required = {"whisper_time_translation_worker_v1.py",
                    "whisper_time_translation_parent_v1.py",
                    "whisper_time_translation_saved_audit_v1.py",
                    "whisper_tail_causal_saved_audit_v3.py",
                    "whisper_tail_causal_parent_v3.py",
                    "whisper_tail_causal_plan_v1.py",
                    "whisper_tail_causal_plan_v3.py",
                    "WHISPER_TIME_TRANSLATION_EXECUTABLE_HANDOFF_V1.md",
                    "WHISPER_TIME_TRANSLATION_PLAN_V1.json",
                    "runs/whisper_time_translation_v1/source_v1/manifest.json"}
        self.assertTrue(required.issubset(gate.SOURCE_NAMES))


if __name__ == "__main__":
    unittest.main()
