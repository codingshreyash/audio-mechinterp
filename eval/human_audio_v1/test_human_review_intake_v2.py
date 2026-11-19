"""Synthetic-only fixtures for blinded review import; no human responses."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import blinded_listening_v1 as blind
from . import human_review_intake_v2 as intake


def fake_review(label: str, mapping: dict, references: dict[str, str],
                wrong_case: str | None = None,
                wrong_text: str = "WRONG WORDS") -> dict:
    queue = mapping["queues"][label]
    records = []
    for row in queue["mapping"]:
        transcript = references[row["case_id"]]
        if row["case_id"] == wrong_case:
            transcript = wrong_text
        records.append({"opaque_id": row["opaque_id"],
                        "wav_sha256": row["wav_sha256"], "reviewer_id": "FAKE_" + label,
                        "submitted_at_utc": "2000-01-01T00:00:00Z",
                        "verbatim_transcript": transcript,
                        "clarity": "clear", "tail_audible": "yes", "notes": "fake fixture"})
    return {"schema": "human_audio_blind_review_v1", "reviewer_id": "FAKE_" + label,
            "queue_sha256": queue["queue_sha256"],
            "bundle_binding": mapping["bundle_binding"],
            "planned_clips": len(records), "records": records}


class IntakeTests(unittest.TestCase):
    def test_majority_requires_two_votes(self) -> None:
        self.assertEqual(intake.majority(["a", "b", "a"]), "a")
        self.assertIsNone(intake.majority(["a", "b", "c"]))
        self.assertIsNone(intake.majority(["a", "b"]))

    def test_fake_export_binding_and_third_resolution(self) -> None:
        # This is a scratch-only test. It does not constitute human listening QA.
        bundle = blind.BUNDLE
        mapping = json.loads((bundle / "TECHNICAL_MAPPING.json").read_text())
        source = json.loads((blind.SOURCE / "manifest.json").read_text())
        references = {f"{r['utterance_id']}|{c['condition']}": r["reference_transcript"]
                      for r in source["rows"] for c in r["conditions"]}
        all_cases = blind.cases()
        wrong_case = all_cases[0]["case_id"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_path, b_path, c_path = (root / f"fake_{label}.json" for label in "ABC")
            a_path.write_text(json.dumps(fake_review("A", mapping, references)))
            b_path.write_text(json.dumps(fake_review("B", mapping, references,
                                                     wrong_case=wrong_case)))
            third = root / "third"
            with mock.patch.object(blind, "audit_bundle", return_value={
                     "status": "PASS_SOURCE_ONLY"}), \
                 mock.patch.object(blind, "cases", return_value=all_cases):
                receipt = blind.build_third(bundle, a_path, b_path, third)
                self.assertEqual(receipt["disagreements"], 1)
                c_mapping = {"queues": {"C": receipt["reviewer_C"]},
                             "bundle_binding": mapping["bundle_binding"]}
                c_path.write_text(json.dumps(fake_review("C", c_mapping, references)))
                final = intake.finalize(bundle, third, a_path, b_path, c_path)
                self.assertEqual((final["denominator"], final["passed"],
                                  final["disagreements"]), (16, 16, 1))
                with self.assertRaisesRegex(ValueError, "C review required"):
                    intake.finalize(bundle, third, a_path, b_path, None)
                c_path.write_text(json.dumps(fake_review("C", c_mapping, references,
                                                         wrong_case=wrong_case,
                                                         wrong_text="ANOTHER WRONG")))
                self.assertEqual(intake.finalize(bundle, third, a_path, b_path,
                                                 c_path)["passed"], 15)
                c_data = json.loads(c_path.read_text())
                c_path.write_text(json.dumps({**c_data, "reviewer_id": "FAKE_A"}))
                with self.assertRaisesRegex(ValueError, "record incomplete|identity"):
                    intake.finalize(bundle, third, a_path, b_path, c_path)
                c_path.write_text(json.dumps(c_data))
                b_data = json.loads(b_path.read_text())
                b_path.write_text(json.dumps({**b_data, "reviewer_id": "ALTERED"}))
                with self.assertRaisesRegex(ValueError, "hashes changed"):
                    intake.finalize(bundle, third, a_path, b_path, c_path)

    def test_fake_agreed_wrong_reference_fails_without_c(self) -> None:
        bundle = blind.BUNDLE
        mapping = json.loads((bundle / "TECHNICAL_MAPPING.json").read_text())
        source = json.loads((blind.SOURCE / "manifest.json").read_text())
        references = {f"{r['utterance_id']}|{c['condition']}": r["reference_transcript"]
                      for r in source["rows"] for c in r["conditions"]}
        all_cases = blind.cases()
        wrong_case = all_cases[0]["case_id"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_path, b_path = root / "fake_A.json", root / "fake_B.json"
            for label, path in (("A", a_path), ("B", b_path)):
                path.write_text(json.dumps(fake_review(label, mapping, references,
                                                       wrong_case=wrong_case)))
            with mock.patch.object(blind, "audit_bundle", return_value={
                     "status": "PASS_SOURCE_ONLY"}), \
                 mock.patch.object(blind, "cases", return_value=all_cases):
                third = root / "third"
                receipt = blind.build_third(bundle, a_path, b_path, third)
                self.assertEqual(receipt["disagreements"], 0)
                final = intake.finalize(bundle, third, a_path, b_path, None)
                self.assertEqual((final["denominator"], final["passed"]), (16, 15))


if __name__ == "__main__":
    unittest.main()
