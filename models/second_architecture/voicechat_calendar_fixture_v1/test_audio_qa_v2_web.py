"""No-human, no-model tests for staged VoiceChat browser intake.

All generated responses here are synthetic fixture values in a temporary
directory. They are not listening records or candidate admission evidence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import audio_qa_v2 as qa
from . import audio_qa_v2_build as build
from . import audio_qa_v2_web as web


BUNDLE = Path("/Users/shrranjan/VoiceChat-Blind-Listening-20261118")
ROLES = {"primary": "syntheticA", "secondary": "syntheticB", "tertiary": "syntheticC"}


def stamp(seconds_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat().replace(
        "+00:00", "Z")


def export(path: Path, role: str, stage: str, ids: list[str], records: list[dict]) -> None:
    path.write_text(json.dumps({
        "schema": web.EXPORT_SCHEMA, "stage": stage, "role": role,
        "reviewer_audit_id": ROLES[role],
        "assignment_sha256": qa.digest(BUNDLE / "blind_assignment.json"),
        "opaque_order": ids, "records": records,
    }))


def blind_rows(role: str, ids: list[str], *, transcript: str = "synthetic test only") -> list[dict]:
    return [{"schema": "voicechat_listener_stage_v2", "opaque_id": opaque,
             "reviewer_audit_id": ROLES[role], "stage": "blind",
             "recorded_utc": stamp(10), "verbatim_free_transcript": transcript,
             "clear_speech": True, "confidence": "high", "uncertain_spans": []}
            for opaque in ids]


def aware_rows(role: str, ids: list[str], records_root: Path) -> list[dict]:
    render = json.loads(qa.RENDER.read_text())
    comparison = json.loads(qa.COMPARISON.read_text())
    by_key = {(r["case_id"], r["voice_id"]): r for r in render["rows"]}
    by_opaque = {r["opaque_id"]: by_key[(r["case_id"], r["voice_id"])]
                 for r in comparison["rows"]["whisper"]}
    rows = []
    for opaque in ids:
        case = by_opaque[opaque]["case_id"]
        blind = web._stage_path(records_root, ROLES[role], opaque, "blind")
        original, sha = qa.canonical_record(blind, "voicechat_listener_stage_v2")
        rows.append({
            "schema": "voicechat_listener_stage_v2", "opaque_id": opaque,
            "reviewer_audit_id": ROLES[role], "stage": "script_aware",
            "recorded_utc": stamp(3), "blind_recorded_utc": original["recorded_utc"],
            "blind_record_sha256": sha, "clear_speech": True,
            "date_heard": "not_applicable" if case == "D" else "2026-11-19",
            "design_review_title_heard": ("Design Review" if case in
                                           ("B", "C", "B_literal") else "not_applicable"),
            "first_action_heard": qa.CASES[case]["expected_first_action"],
            "cancel_requires_further_confirmation_heard":
                True if case in ("B", "B_literal") else "not_applicable",
            "c_confirmation_phrase_heard_exactly":
                True if case == "C" else "not_applicable",
            "cancellation_intent_heard": ("explicit" if case == "C" else
                                           "possible" if case in ("B", "B_literal")
                                           else "none"),
            "case_semantics_preserved": True, "eligibility_judgment": "eligible",
            "notes": "synthetic_test_only"})
    return rows


class WebIntakeNoHumanTests(unittest.TestCase):
    def test_primary_blind_aware_and_existing_receipt_builder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthetic_test_only_voicechat_web_") as root:
            base = Path(root)
            records = base / "records"
            page_dir = base / "primary_page"
            primary = web.make_blind(BUNDLE, records, page_dir, role="primary", roles=ROLES)
            self.assertEqual(primary["clips"], 12)
            blind_html = (page_dir / "BLIND.html").read_text()
            frozen = json.loads(qa.RENDER.read_text())
            for source in frozen["rows"]:
                self.assertNotIn(source["script"], blind_html)
            self.assertEqual(blind_html.count("<audio "), 12)
            ids, _, _ = web._selection(BUNDLE, records, "primary", ROLES)
            blind_export = base / "blind.json"
            export(blind_export, "primary", "blind", ids, blind_rows("primary", ids))
            self.assertEqual(web.import_stage(BUNDLE, records, blind_export,
                                              stage="blind", role="primary",
                                              roles=ROLES)["imported"], 12)
            aware = web.make_aware(BUNDLE, records, page_dir, role="primary", roles=ROLES)
            self.assertEqual(aware["clips"], 12)
            self.assertIn("Frozen script", (page_dir / "AWARE.html").read_text())
            aware_export = base / "aware.json"
            export(aware_export, "primary", "script_aware", ids,
                   aware_rows("primary", ids, records))
            self.assertEqual(web.import_stage(BUNDLE, records, aware_export,
                                              stage="script_aware", role="primary",
                                              roles=ROLES)["imported"], 12)
            flagged, _, _ = web._selection(BUNDLE, records, "secondary", ROLES)
            self.assertGreaterEqual(len(flagged), 1)
            self.assertLessEqual(len(flagged), 12)
            secondary_dir = base / "secondary_page"
            web.make_blind(BUNDLE, records, secondary_dir, role="secondary", roles=ROLES)
            self.assertNotIn("Design Review", (secondary_dir / "BLIND.html").read_text())
            second_blind = base / "second_blind.json"
            export(second_blind, "secondary", "blind", flagged,
                   blind_rows("secondary", flagged))
            web.import_stage(BUNDLE, records, second_blind, stage="blind",
                             role="secondary", roles=ROLES)
            web.make_aware(BUNDLE, records, secondary_dir, role="secondary", roles=ROLES)
            second_aware = base / "second_aware.json"
            export(second_aware, "secondary", "script_aware", flagged,
                   aware_rows("secondary", flagged, records))
            web.import_stage(BUNDLE, records, second_aware,
                             stage="script_aware", role="secondary", roles=ROLES)
            third, _, _ = web._selection(BUNDLE, records, "tertiary", ROLES)
            self.assertEqual(third, [])
            receipt = build.build_receipts(BUNDLE, records, base / "receipts",
                                           primary_id=ROLES["primary"],
                                           secondary_id=ROLES["secondary"],
                                           tertiary_id=None,
                                           completed_at_utc=stamp(1),
                                           controller_id="synthetic_controller")
            self.assertEqual(receipt["denominator"], 12)
            self.assertEqual(receipt["eligible"], 12)
            self.assertEqual(receipt["secondary_reviews"], len(flagged))

    def test_disagreement_triggers_only_one_tertiary_review(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthetic_test_only_voicechat_web_") as root:
            base = Path(root)
            records = base / "records"
            all_ids, _, _ = web._selection(BUNDLE, records, "primary", ROLES)
            for role, ids in (("primary", all_ids),):
                page = base / f"{role}_page"
                web.make_blind(BUNDLE, records, page, role=role, roles=ROLES)
                blind_file = base / f"{role}_blind.json"
                export(blind_file, role, "blind", ids, blind_rows(role, ids))
                web.import_stage(BUNDLE, records, blind_file, stage="blind", role=role,
                                 roles=ROLES)
                web.make_aware(BUNDLE, records, page, role=role, roles=ROLES)
                aware_file = base / f"{role}_aware.json"
                export(aware_file, role, "script_aware", ids,
                       aware_rows(role, ids, records))
                web.import_stage(BUNDLE, records, aware_file, stage="script_aware",
                                 role=role, roles=ROLES)
            flagged, _, _ = web._selection(BUNDLE, records, "secondary", ROLES)
            self.assertEqual(len(flagged), 11)
            page = base / "secondary_page"
            web.make_blind(BUNDLE, records, page, role="secondary", roles=ROLES)
            blind_file = base / "secondary_blind.json"
            rows = blind_rows("secondary", flagged)
            rows[0]["verbatim_free_transcript"] = "different synthetic test transcript"
            export(blind_file, "secondary", "blind", flagged, rows)
            web.import_stage(BUNDLE, records, blind_file, stage="blind",
                             role="secondary", roles=ROLES)
            web.make_aware(BUNDLE, records, page, role="secondary", roles=ROLES)
            aware_file = base / "secondary_aware.json"
            export(aware_file, "secondary", "script_aware", flagged,
                   aware_rows("secondary", flagged, records))
            web.import_stage(BUNDLE, records, aware_file, stage="script_aware",
                             role="secondary", roles=ROLES)
            tertiary, _, _ = web._selection(BUNDLE, records, "tertiary", ROLES)
            self.assertEqual(tertiary, flagged[:1])
            third_page = base / "tertiary_page"
            web.make_blind(BUNDLE, records, third_page, role="tertiary", roles=ROLES)
            self.assertEqual((third_page / "BLIND.html").read_text().count("<audio "), 1)

    def test_blind_export_tamper_rejects_before_writes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthetic_test_only_voicechat_web_") as root:
            base = Path(root)
            records = base / "records"
            ids, _, _ = web._selection(BUNDLE, records, "primary", ROLES)
            rows = blind_rows("primary", ids)
            rows[0]["opaque_id"] = ids[1]
            path = base / "tampered.json"
            export(path, "primary", "blind", ids, rows)
            with self.assertRaisesRegex(ValueError, "order or opaque IDs"):
                web.import_stage(BUNDLE, records, path, stage="blind",
                                 role="primary", roles=ROLES)
            self.assertFalse(records.exists())

    def test_aware_page_requires_all_saved_blind_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthetic_test_only_voicechat_web_") as root:
            base = Path(root)
            records = base / "records"
            page_dir = base / "page"
            web.make_blind(BUNDLE, records, page_dir, role="primary", roles=ROLES)
            with self.assertRaisesRegex(ValueError, "all blind stages"):
                web.make_aware(BUNDLE, records, page_dir, role="primary", roles=ROLES)
            self.assertFalse((page_dir / "AWARE.html").exists())


if __name__ == "__main__":
    unittest.main()
