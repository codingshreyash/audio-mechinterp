"""Synthetic reviewer records only; no real human response is manufactured."""
from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import human_qa_v1 as qa


def fixture() -> tuple[dict, dict, dict, dict]:
    expected = {}
    ids = [f"case_{i:02d}" for i in range(18)]
    for index, case_id in enumerate(ids):
        label = ("LIST", "ASK", "CANCEL")[index % 3]
        expected[case_id] = {"script": f"Test sentence number {index} for {label}.",
                             "label": label, "source_frames": 90000}
    mapping = {"bundle_binding": "b" * 64,
               "queues": {name: [{"case_id": case_id, "ordinal": pos,
                                    "opaque_file": f"q{pos:02d}.wav"}
                                  for pos, case_id in enumerate(order, 1)]
                          for name, order in (("A", ids), ("B", list(reversed(ids))))}}
    reviews = {}
    for queue in ("A", "B"):
        rows = []
        for plan in mapping["queues"][queue]:
            details = expected[plan["case_id"]]
            rows.append({"ordinal": plan["ordinal"], "file": plan["opaque_file"],
                         "transcript": details["script"],
                         "meaning": qa.INTENTS[details["label"]],
                         "date": "November nineteenth, twenty twenty-six",
                         "event": ("none" if details["label"] == "LIST"
                                   else "Design Review"),
                         "phrase_start": "1.0", "phrase_end": "2.0",
                         "notes": "none"})
        reviews[queue] = {"schema": "qwen_action_blind_listening_v1",
                          "queue": queue, "bundle_binding": mapping["bundle_binding"],
                          "reviewer": f"reviewer_{queue}", "rows": rows,
                          "locked_utc": "2026-11-19T12:00:00.000Z"}
    return mapping, reviews["A"], reviews["B"], expected


class HumanQAIntakeTests(unittest.TestCase):
    def test_complete_independent_pair_is_only_provisional(self) -> None:
        mapping, a, b, expected = fixture()
        b["rows"][0]["date"] = "November 19, 2026"
        result = qa.assess_first_pair(a, b, mapping, expected)
        self.assertTrue(result["first_pair_all_pass"])
        self.assertEqual(result["third_required_case_ids"], [])
        self.assertEqual(len(result["cases"]), 18)

    def test_critical_disagreement_requires_sparse_third_and_fails_initial_pair(self) -> None:
        mapping, a, b, expected = fixture()
        row = next(row for row, plan in zip(b["rows"], mapping["queues"]["B"])
                   if plan["case_id"] == "case_00")
        row["meaning"] = "confirm_now"
        result = qa.assess_first_pair(a, b, mapping, expected)
        self.assertEqual(result["third_required_case_ids"], ["case_00"])
        self.assertFalse(result["first_pair_all_pass"])

    def test_missing_swapped_or_wrong_bound_rows_fail_before_reference_compare(self) -> None:
        mapping, a, b, expected = fixture()
        a["rows"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            qa.assess_first_pair(a, b, mapping, expected)
        mapping, a, b, expected = fixture()
        a["rows"][0], a["rows"][1] = a["rows"][1], a["rows"][0]
        with self.assertRaisesRegex(ValueError, "identity/order"):
            qa.assess_first_pair(a, b, mapping, expected)
        mapping, a, b, expected = fixture()
        a["bundle_binding"] = "altered"
        with self.assertRaisesRegex(ValueError, "binding"):
            qa.assess_first_pair(a, b, mapping, expected)

    def test_reviewer_identity_and_phrase_time_are_checked(self) -> None:
        mapping, a, b, expected = fixture()
        b["reviewer"] = a["reviewer"]
        with self.assertRaisesRegex(ValueError, "independent reviewer"):
            qa.assess_first_pair(a, b, mapping, expected)
        mapping, a, b, expected = fixture()
        a["rows"][0]["phrase_end"] = "nan"
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            qa.assess_first_pair(a, b, mapping, expected)

    def test_source_mismatch_is_explicit_failure_not_drop(self) -> None:
        mapping, a, b, expected = fixture()
        a["rows"][0]["event"] = "Design Review"
        result = qa.assess_first_pair(a, b, mapping, expected)
        self.assertFalse(result["first_pair_all_pass"])
        self.assertIn("event", result["cases"][0]["reviewer_A_failures"])
        self.assertEqual(result["third_required_case_ids"], ["case_00"])

    def test_final_report_requires_third_only_when_flagged_and_never_authorizes_model(self) -> None:
        mapping, a, b, expected = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir, bundle_dir = root / "source", root / "bundle"
            source_dir.mkdir(); bundle_dir.mkdir()
            (source_dir / "manifest.json").write_text("synthetic")
            (bundle_dir / "TECHNICAL_MAPPING.json").write_text("synthetic")
            ap, bp = root / "a.json", root / "b.json"
            ap.write_text(json.dumps(a)); bp.write_text(json.dumps(b))
            with patch.object(qa, "SOURCE", source_dir), \
                 patch.object(qa, "BUNDLE", bundle_dir), \
                 patch.object(qa, "_load_bound", return_value=(
                     mapping, a, b, expected)):
                result = qa.audit_reviews(ap, bp)
                self.assertEqual(result["status"],
                                 "human_qa_pass_needs_independent_review")
                self.assertFalse(result["candidate_model_inference_authorized"])
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    qa.audit_reviews(ap, bp, root / "unneeded_third.json")
                b["rows"][-1]["meaning"] = "unclear"
                with self.assertRaisesRegex(ValueError, "third human review required"):
                    qa.audit_reviews(ap, bp)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.json"
            path.write_text('{"queue":"A","queue":"B"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                qa.read_json(path)

    def test_third_queue_binding_rejects_changed_first_or_wav(self) -> None:
        mapping, a, b, expected = fixture()
        pair_hashes = {"A": "a" * 64, "B": "c" * 64}
        locks = {"A": a["locked_utc"], "B": b["locked_utc"]}
        sources = {case: {"final_sha256": f"{index:064x}"}
                   for index, case in enumerate(expected)}
        with tempfile.TemporaryDirectory() as temp:
            bundle_root = Path(temp)
            (bundle_root / "TECHNICAL_MAPPING.json").write_text("synthetic mapping")
            third = {
                "schema": "qwen_action_sparse_third_bundle_v1",
                "status": "human_third_review_pending",
                "bundle_binding": hashlib.sha256((mapping["bundle_binding"]
                                                  + pair_hashes["A"]
                                                  + pair_hashes["B"]).encode()).hexdigest(),
                "base_mapping_sha256": qa.sha(bundle_root / "TECHNICAL_MAPPING.json"),
                "first_export_sha256": pair_hashes,
                "first_export_locks_utc": locks,
                "created_utc": "2026-11-19T12:01:00.000Z",
                "queues": {"C": [{"ordinal": 1, "opaque_file": "q01.wav",
                                  "case_id": "case_00",
                                  "wav_sha256": sources["case_00"]["final_sha256"]}]},
                "html_sha256": "d" * 64,
            }
            with patch.object(qa, "BUNDLE", bundle_root):
                self.assertEqual(len(qa.validate_third_mapping(
                    third, mapping, pair_hashes, ["case_00"], sources, locks)), 1)
                third["queues"]["C"][0]["wav_sha256"] = "e" * 64
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    qa.validate_third_mapping(third, mapping, pair_hashes,
                                              ["case_00"], sources, locks)


if __name__ == "__main__":
    unittest.main()
