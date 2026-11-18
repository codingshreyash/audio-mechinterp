"""Integrity smoke checks for the read-only Voxtral FIT explorer builder."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_voxtral_fit_v2 as explorer


class ExplorerBuilderTests(unittest.TestCase):
    def test_audited_population_and_deterministic_html(self):
        data = explorer.load_verified()
        self.assertEqual(len(data["rows"]), 800)
        self.assertEqual(len(data["case_ids"]), 16)
        self.assertFalse(data["summary"]["sites"]["encoder31"]["restore"]["time_preference_gate"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.html"
            first = explorer.build(target)
            second = explorer.build(target)
            self.assertEqual(first["output_sha256"], second["output_sha256"])
            self.assertIn("Voxtral word-state intervention explorer", target.read_text())

    def test_modified_attempt_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            shutil.copytree(explorer.RUN, run)
            attempt = run / "record_01_attempt_01.jsonl"
            rows = attempt.read_text().splitlines()
            item = json.loads(rows[8])
            item["raw_decoded"] = "TAMPERED"
            rows[8] = json.dumps(item)
            attempt.write_text("\n".join(rows) + "\n")
            with patch.object(explorer, "RUN", run):
                with self.assertRaisesRegex(ValueError, "Attempt 1 SHA mismatch"):
                    explorer.load_verified()

    def test_changed_audit_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            shutil.copytree(explorer.RUN, run)
            audit = run / "audit.json"
            receipt = json.loads(audit.read_text())
            receipt["status"] = "passed_read_only_full_fit_audit"
            receipt["generations"] = 799
            audit.write_text(json.dumps(receipt))
            with patch.object(explorer, "RUN", run):
                with self.assertRaisesRegex(ValueError, "Pinned audit SHA mismatch"):
                    explorer.load_verified()


if __name__ == "__main__":
    unittest.main()
