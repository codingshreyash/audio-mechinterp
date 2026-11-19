"""Synthetic reviewer-record tests; no human evidence is created or saved."""

from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

from . import human_qa_audit as qa


def stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fake_export(ctx: dict, reviewer: str) -> dict:
    start = datetime(2026, 11, 19, 1, 0, tzinfo=timezone.utc)
    records = []
    for i, ident in enumerate(ctx["order"]):
        case = ctx["by_id"][ident]
        records.append({
            "opaque_id": ident,
            "reviewer_id": reviewer,
            "wav_sha256": case["wav_sha256"],
            "submitted_at_utc": stamp(start + timedelta(seconds=i + 1)),
            "verbatim_transcript": case["reference_script"],
            "middle_word": case["word"],
            "clear_or_unclear": "clear",
            "confidence": 5,
        })
    return {
        "schema": qa.EXPORT_SCHEMA, "reviewer_id": reviewer,
        "queue_sha256": ctx["queue_sha256"], "planned_clips": 32,
        "opaque_ids_in_order": ctx["order"].copy(),
        "started_at_utc": stamp(start), "records": records,
        "exported_at_utc": stamp(start + timedelta(minutes=5)),
    }


class HumanQaAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ctx = qa.context()

    def pair(self) -> dict:
        return {
            "primary": fake_export(self.ctx, "reviewer_A"),
            "secondary": fake_export(self.ctx, "reviewer_B"),
        }

    def score(self, docs: dict) -> dict:
        return qa.evaluate(self.ctx, docs, {stage: stage * 64 for stage in docs})

    def test_full_valid_synthetic_records_and_unicode_punctuation(self) -> None:
        docs = self.pair()
        variant = (
            docs["primary"]["records"][0]["verbatim_transcript"]
            .replace("I said", "Ｉ said").replace(".", "!")
        )
        docs["primary"]["records"][0]["verbatim_transcript"] = variant
        docs["secondary"]["records"][0]["verbatim_transcript"] = variant
        report = self.score(docs)
        self.assertEqual(report["status"], "human_qa_32_of_32_pass")
        self.assertEqual(report["passed_clips"], 32)
        self.assertEqual(report["disagreement_clips"], 0)
        self.assertFalse(report["candidate_model_inference_authorized"])

    def test_raw_verbatim_difference_triggers_third_even_when_tokens_match(self) -> None:
        docs = self.pair()
        original = docs["secondary"]["records"][0]["verbatim_transcript"]
        docs["primary"]["records"][0]["verbatim_transcript"] = (
            original.replace("I said", "i said").replace(".", "!")
        )
        self.assertEqual(qa.tokens(docs["primary"]["records"][0]["verbatim_transcript"]),
                         qa.tokens(original))
        without = self.score(docs)
        self.assertEqual(without["status"], "human_qa_hold")
        self.assertEqual(without["third_review_required_missing"], 1)
        self.assertTrue(without["cases"][0]["disagreement"])
        self.assertEqual(without["cases"][0]["reason"], "third_review_required")
        docs["tertiary"] = fake_export(self.ctx, "reviewer_C")
        with_third = self.score(docs)
        self.assertEqual(with_third["status"], "human_qa_32_of_32_pass")
        self.assertEqual(with_third["cases"][0]["reason"], "two_of_three_correct")
        self.assertNotEqual(
            with_third["cases"][0]["responses"]["primary"]["verbatim_transcript"],
            with_third["cases"][0]["responses"]["secondary"]["verbatim_transcript"],
        )

    def test_disagreement_requires_third_and_two_of_three_can_resolve(self) -> None:
        docs = self.pair()
        docs["primary"]["records"][0]["middle_word"] = "something else"
        without = self.score(docs)
        self.assertEqual(without["status"], "human_qa_hold")
        self.assertEqual(without["passed_clips"], 31)
        self.assertEqual(without["third_review_required_missing"], 1)
        docs["tertiary"] = fake_export(self.ctx, "reviewer_C")
        resolved = self.score(docs)
        self.assertEqual(resolved["status"], "human_qa_32_of_32_pass")
        self.assertEqual(resolved["cases"][0]["reason"], "two_of_three_correct")
        docs["tertiary"]["records"][0]["clear_or_unclear"] = "unclear"
        unresolved = self.score(docs)
        self.assertEqual(unresolved["status"], "human_qa_hold")
        self.assertEqual(unresolved["cases"][0]["reason"], "unresolved_or_incorrect")

    def test_two_primary_agree_but_wrong_cannot_be_rescued(self) -> None:
        docs = self.pair()
        for stage in docs:
            docs[stage]["records"][0]["verbatim_transcript"] = "I said a number and then stopped."
        docs["tertiary"] = fake_export(self.ctx, "reviewer_C")
        report = self.score(docs)
        self.assertEqual(report["passed_clips"], 31)
        self.assertEqual(report["cases"][0]["reason"], "two_primary_agree_but_incorrect")

    def test_duplicate_reviewer_or_missing_record_rejected(self) -> None:
        docs = self.pair()
        docs["secondary"]["reviewer_id"] = "reviewer_A"
        for record in docs["secondary"]["records"]:
            record["reviewer_id"] = "reviewer_A"
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.score(docs)
        docs = self.pair()
        docs["primary"]["records"].pop()
        with self.assertRaisesRegex(ValueError, "32 ordered"):
            self.score(docs)

    def test_tampered_queue_wav_and_timestamps_rejected(self) -> None:
        docs = self.pair()
        docs["primary"]["queue_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "queue/schema"):
            self.score(docs)
        docs = self.pair()
        docs["secondary"]["records"][0]["wav_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "WAV binding"):
            self.score(docs)
        docs = self.pair()
        docs["primary"]["records"][0]["submitted_at_utc"] = "2026-11-18T00:00:00.000Z"
        with self.assertRaisesRegex(ValueError, "order/time"):
            self.score(docs)

    def test_parser_does_not_drop_extra_words_or_digits(self) -> None:
        self.assertEqual(qa.tokens("Ｉ_said banana, and then stopped!"),
                         ["i", "said", "banana", "and", "then", "stopped"])
        self.assertNotEqual(qa.tokens("I said banana 2 and then stopped."),
                            qa.tokens("I said banana and then stopped."))
        self.assertNotEqual(qa.tokens("I said banana banana and then stopped."),
                            qa.tokens("I said banana and then stopped."))
        self.assertEqual(qa.tokens(""), [])


if __name__ == "__main__":
    unittest.main()
