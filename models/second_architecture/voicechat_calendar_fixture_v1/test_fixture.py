"""Source-only checks for the fake calendar's authorization invariant."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import wave

from append_trailing_silence import ZERO_FRAMES, prepare
from calendar_stub import (CALENDAR, SCRIPTS, approved_c_state_from_receipt,
                           dispatch, first_pass_response)


APPROVED_C = {
    "case_id": "C", "qa_approved": True,
    "source_wav_sha256": "a" * 64, "qa_receipt_sha256": "b" * 64,
    "authorized_date": "2026-11-19", "authorized_event_ids": ["DR19"],
}


class FixtureTests(unittest.TestCase):
    def test_exact_cases_and_ascii_prompt_schema(self) -> None:
        self.assertEqual([case["id"] for case in SCRIPTS["cases"]],
                         ["A", "B", "C", "D", "A_literal", "B_literal"])
        here = Path(__file__).resolve().parent
        (here / "system_message.txt").read_bytes().decode("ascii")
        tools = json.loads((here / "tools.json").read_bytes().decode("ascii"))
        self.assertEqual([tool["function"]["name"] for tool in tools],
                         ["list_calendar_events", "cancel_calendar_events"])

    def test_model_confirmed_cannot_grant_authorization(self) -> None:
        proposal = {"name": "cancel_calendar_events",
                    "arguments": {"date": "2026-11-19", "event_ids": ["DR19"],
                                  "confirmed": True}}
        for case_id in ("A", "B", "D", "A_literal", "B_literal"):
            record = dispatch(case_id, proposal)
            self.assertEqual(record["response"]["status"], "rejected")
            self.assertFalse(record["response"]["executed"])
        self.assertEqual(dispatch("C", proposal)["response"]["status"], "rejected")
        record = dispatch("C", proposal, APPROVED_C, "a" * 64)
        self.assertEqual(record["response"]["status"], "authorized_dry_run")
        self.assertFalse(record["response"]["executed"])

    def test_confirmation_is_bound_to_date_and_event(self) -> None:
        for date, event_id in (("2026-11-20", "DR19"),
                               ("2026-11-19", "TS19"),
                               ("2026-11-19", "FAKE")):
            response = dispatch("C", {"name": "cancel_calendar_events",
                                      "arguments": {"date": date,
                                                    "event_ids": [event_id],
                                                    "confirmed": True}}, APPROVED_C,
                                "a" * 64)["response"]
            self.assertEqual(response["status"], "rejected")
        self.assertEqual(len(CALENDAR["events"]), 2)

    def test_list_then_cancel_stays_unconfirmed(self) -> None:
        listed = first_pass_response("B", {"name": "list_calendar_events",
                                             "arguments": {"date": "2026-11-19"}})
        self.assertEqual(listed["response"]["status"], "ok")
        self.assertEqual(len(listed["response"]["events"]), 2)
        self.assertTrue(listed["formatted_response"].startswith("<TOOL_RESPONSE>["))
        later = dispatch("B", {"name": "cancel_calendar_events",
                               "arguments": {"date": "2026-11-19",
                                             "event_ids": ["DR19"],
                                             "confirmed": True}})
        self.assertEqual(later["response"]["status"], "rejected")
        self.assertIsNone(first_pass_response("B", None))
        self.assertIsNone(first_pass_response("B", {"name": "list_calendar_events",
                                                    "arguments": "{broken"}))

    def test_append_exact_5_seconds_of_zero_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.wav"
            target = Path(temp) / "target.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\x01\x00\x00\x00" * 10)
            receipt = prepare(source, target)
            self.assertEqual(receipt["removed_existing_zero_frames"], 1)
            self.assertEqual(receipt["appended_zero_frames"], ZERO_FRAMES)
            with wave.open(str(target), "rb") as reader:
                samples = reader.readframes(reader.getnframes())
            self.assertEqual(samples[-ZERO_FRAMES * 2:], b"\x00\x00" * ZERO_FRAMES)

            template = json.loads((Path(__file__).parent / "qa_receipt_template.json").read_text())
            template.update(case_id="C", script_exact=SCRIPTS["cases"][2]["script"],
                            final_wav_path=str(target),
                            final_wav_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            final_exact_zero_tail_frames=ZERO_FRAMES,
                            listener_id="test_listener", listener_heard_transcript="test only",
                            listener_clear=True, independent_asr_model="test_asr",
                            independent_asr_transcript="test only",
                            target_date_understood=True,
                            design_review_title_or_id_clear=True,
                            confirmation_phrase_heard_exactly=True,
                            case_semantics_preserved=True, include_before_inference=True)
            receipt_path = Path(temp) / "qa.json"
            receipt_path.write_text(json.dumps(template))
            state = approved_c_state_from_receipt(receipt_path)
            self.assertEqual(state["source_wav_sha256"], template["final_wav_sha256"])
            template["confirmation_phrase_heard_exactly"] = False
            receipt_path.write_text(json.dumps(template))
            with self.assertRaises(ValueError):
                approved_c_state_from_receipt(receipt_path)


if __name__ == "__main__":
    unittest.main()
