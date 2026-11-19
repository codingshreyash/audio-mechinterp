"""No-weight checks for the authentic-speech target admission precursor."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import whisper_human_preflight_v1 as gate


class WhisperHumanPreflightTests(unittest.TestCase):
    def test_real_frozen_plan_checks_all_source_wavs(self) -> None:
        plan = gate.frozen_plan()
        self.assertEqual(len(plan["calls"]), 16)
        self.assertEqual(len(plan["comparisons"]), 8)
        self.assertEqual(sum(r["split"] == "development" for r in plan["calls"]), 8)

    def test_source_wav_drift_fails_closed(self) -> None:
        first = json.loads(gate.PLAN.read_text())["calls"][0]
        source = gate.ROOT / first["wav"]
        real_sha = gate.sha
        with mock.patch.object(gate, "sha", side_effect=lambda p: "0" * 64
                               if p == source else real_sha(p)):
            with self.assertRaisesRegex(ValueError, "source WAV hash changed"):
                gate.frozen_plan()

    def test_missing_human_exports_keep_target_on_hold(self) -> None:
        from models.second_architecture.nonspeech_probe_v1 import descriptive_worker_v1
        assets = {"repo": "openai/whisper-tiny",
                  "revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                  "files_sha256": {"model.safetensors":
                    "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"}}
        with mock.patch.object(descriptive_worker_v1, "verify_local_assets", return_value=assets), \
             mock.patch.object(descriptive_worker_v1, "verify_installed_sources",
                               return_value={"versions": {}}):
            report = gate.readiness()
        self.assertEqual(report["status"], "HOLD_NO_TARGET_INFERENCE")
        self.assertIn("two blinded review exports", report["reason"])
        self.assertEqual(report["development_calls"], 8)
        self.assertEqual(report["sealed_calls"], 8)

    def test_false_human_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b, receipt = (root / name for name in ("a.json", "b.json", "receipt.json"))
            for path in (a, b, receipt):
                path.write_text("{}")
            third = root / "third"
            third.mkdir()
            (third / "ADJUDICATION_MAPPING.json").write_text("{}")
            with self.assertRaises((ValueError, KeyError)):
                gate.reviewed_listening(gate.HERE / "blind_bundle_v1", third,
                                        a, b, None, receipt)


if __name__ == "__main__":
    unittest.main()
