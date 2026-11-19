"""Synthetic V2 chronology and attestation tests; no real listening record."""
from __future__ import annotations

import hashlib
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from . import human_qa_v2 as qa
from .test_human_qa_v1 import fixture

NOW = datetime(2026, 11, 19, 12, 2, tzinfo=timezone.utc)


class HumanQAV2Tests(unittest.TestCase):
    def test_sparse_third_directory_is_versioned(self) -> None:
        self.assertEqual(qa.THIRD.name, "third_review_002")

    def test_first_export_in_2099_rejected_before_reference_compare(self) -> None:
        mapping, a, b, expected = fixture()
        a["locked_utc"] = b["locked_utc"] = "2099-01-01T00:00:00.000Z"
        with patch.object(qa, "now_utc", return_value=NOW):
            with self.assertRaisesRegex(ValueError, "future-skew cap"):
                qa.assess_first_pair(a, b, mapping, expected)

    def test_five_minute_boundary_is_frozen(self) -> None:
        with patch.object(qa, "now_utc", return_value=NOW):
            self.assertEqual(qa.timestamp("2026-11-19T12:07:00.000Z"),
                             datetime(2026, 11, 19, 12, 7, tzinfo=timezone.utc))
            with self.assertRaisesRegex(ValueError, "future-skew cap"):
                qa.timestamp("2026-11-19T12:07:00.001Z")

    def test_third_creation_in_2099_rejected_even_if_both_firsts_valid(self) -> None:
        mapping, a, b, _ = fixture()
        hashes = {"A": "a" * 64, "B": "c" * 64}
        locks = {"A": a["locked_utc"], "B": b["locked_utc"]}
        sources = {"case_00": {"final_sha256": "d" * 64}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "TECHNICAL_MAPPING.json").write_text("synthetic")
            data = {"schema": "qwen_action_sparse_third_bundle_v1",
                    "status": "human_third_review_pending",
                    "bundle_binding": hashlib.sha256((mapping["bundle_binding"]
                                                      + hashes["A"] + hashes["B"])
                                                     .encode()).hexdigest(),
                    "base_mapping_sha256": qa.sha(folder / "TECHNICAL_MAPPING.json"),
                    "first_export_sha256": hashes,
                    "first_export_locks_utc": locks,
                    "created_utc": "2099-01-01T00:00:00.000Z",
                    "queues": {"C": [{"ordinal": 1, "opaque_file": "q01.wav",
                                      "case_id": "case_00", "wav_sha256": "d" * 64}]},
                    "html_sha256": "e" * 64}
            with patch.object(qa, "now_utc", return_value=NOW), \
                 patch.object(qa, "BUNDLE", folder):
                with self.assertRaisesRegex(ValueError, "future-skew cap"):
                    qa.validate_third_mapping(data, mapping, hashes,
                                              ["case_00"], sources, locks)
                data["created_utc"] = "2026-11-19T12:01:00.000Z"
                data["bundle_binding"] = "altered"
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    qa.validate_third_mapping(data, mapping, hashes,
                                              ["case_00"], sources, locks)

    def test_third_export_locked_in_2099_rejected(self) -> None:
        mapping, a, _, _ = fixture()
        row = dict(a["rows"][0])
        third_mapping = {"bundle_binding": "f" * 64,
                         "queues": {"C": [{"ordinal": 1,
                                            "opaque_file": "q01.wav"}]}}
        row["ordinal"] = 1
        document = {"schema": "qwen_action_blind_listening_v1", "queue": "C",
                    "bundle_binding": third_mapping["bundle_binding"],
                    "reviewer": "reviewer_C", "rows": [row],
                    "locked_utc": "2099-01-01T00:00:00.000Z"}
        with patch.object(qa, "now_utc", return_value=NOW):
            with self.assertRaisesRegex(ValueError, "future-skew cap"):
                qa.validate_export(document, "C", third_mapping)

    def test_attestation_must_bind_both_exports_and_assert_independence(self) -> None:
        mapping, a, b, expected = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "TECHNICAL_MAPPING.json").write_text("synthetic")
            ap, bp = root / "a.json", root / "b.json"
            ap.write_text(json.dumps(a)); bp.write_text(json.dumps(b))
            with patch.object(qa, "now_utc", return_value=NOW), \
                 patch.object(qa, "BUNDLE", root), \
                 patch.object(qa, "_load_bound", return_value=(
                     mapping, a, b, expected)):
                template = qa.attestation_template(ap, bp)
                document = {**copy.deepcopy(template), "coordinator_id": "coordinator",
                            "independent_people": True,
                            "separate_blind_folders": True,
                            "no_shared_answers_before_both_locks": True,
                            "signed_utc": "2026-11-19T12:01:00.000Z",
                            "notes": "Both people confirmed separate review."}
                first = qa.assess_first_pair(a, b, mapping, expected)
                qa.validate_attestation(document, template, first["first_export_times"])
                document["independent_people"] = False
                with self.assertRaisesRegex(ValueError, "statement missing"):
                    qa.validate_attestation(document, template,
                                            first["first_export_times"])
                document["independent_people"] = True
                document["first_export_sha256"]["A"] = "altered"
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    qa.validate_attestation(document, template,
                                            first["first_export_times"])
                document["first_export_sha256"]["A"] = template[
                    "first_export_sha256"]["A"]
                document["signed_utc"] = "2099-01-01T00:00:00.000Z"
                with self.assertRaisesRegex(ValueError, "future-skew cap"):
                    qa.validate_attestation(document, template,
                                            first["first_export_times"])

    def test_final_receipt_keeps_candidate_inference_false(self) -> None:
        mapping, a, b, expected = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir, bundle_dir = root / "source", root / "bundle"
            source_dir.mkdir(); bundle_dir.mkdir()
            (source_dir / "manifest.json").write_text("synthetic")
            (bundle_dir / "TECHNICAL_MAPPING.json").write_text("synthetic")
            ap, bp, cp = root / "a.json", root / "b.json", root / "coordinator.json"
            ap.write_text(json.dumps(a)); bp.write_text(json.dumps(b))
            with patch.object(qa, "now_utc", return_value=NOW), \
                 patch.object(qa, "SOURCE", source_dir), \
                 patch.object(qa, "BUNDLE", bundle_dir), \
                 patch.object(qa, "_load_bound", return_value=(
                     mapping, a, b, expected)):
                attestation = qa.attestation_template(ap, bp)
                attestation.update({"coordinator_id": "coordinator",
                                    "independent_people": True,
                                    "separate_blind_folders": True,
                                    "no_shared_answers_before_both_locks": True,
                                    "signed_utc": "2026-11-19T12:01:00.000Z",
                                    "notes": "Two separate reviewers."})
                cp.write_text(json.dumps(attestation))
                report = qa.audit_reviews(ap, bp, cp)
                self.assertEqual(report["schema"], "qwen_action_human_qa_audit_v2")
                self.assertEqual(report["status"],
                                 "human_qa_pass_needs_independent_review")
                self.assertFalse(report["candidate_model_inference_authorized"])
                self.assertEqual(report["coordinator_attestation_sha256"], qa.sha(cp))


if __name__ == "__main__":
    unittest.main()
