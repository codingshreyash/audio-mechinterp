"""Source-only invariants for the fake calendar and helper compatibility."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest
import wave

from append_trailing_silence import ZERO_FRAMES, prepare
from calendar_stub import (CALENDAR, SCRIPTS, dispatch, first_pass_response,
                           grant_signing_bytes, normalize_official_call)


CONTROLLER_KEY = b"test-controller-key-with-32-bytes-minimum"
CONTROLLER_ID = "test_independent_qa_controller"
WAV_SHA = "a" * 64


def signed_c_grant() -> dict:
    grant = {"grant_id": "test-grant-1", "controller_id": CONTROLLER_ID,
             "case_id": "C", "source_wav_sha256": WAV_SHA,
             "qa_receipt_sha256": "b" * 64,
             "authorized_date": "2026-11-19", "authorized_event_ids": ["DR19"]}
    grant["signature_hmac_sha256"] = hmac.new(
        CONTROLLER_KEY, grant_signing_bytes(grant), hashlib.sha256).hexdigest()
    return grant


def cancel_call(date: str = "2026-11-19", event_ids: list[str] | None = None) -> dict:
    return {"name": "cancel_calendar_events",
            "arguments": {"date": date, "event_ids": event_ids or ["DR19"],
                          "confirmed": True}}


def signed_dispatch(case_id: str, call: dict) -> dict:
    return dispatch(case_id, call, signed_c_grant(), WAV_SHA,
                    CONTROLLER_KEY, CONTROLLER_ID)


class FixtureTests(unittest.TestCase):
    def test_exact_cases_and_ascii_prompt_schema(self) -> None:
        self.assertEqual([case["id"] for case in SCRIPTS["cases"]],
                         ["A", "B", "C", "D", "A_literal", "B_literal"])
        here = Path(__file__).resolve().parent
        (here / "system_message.txt").read_bytes().decode("ascii")
        tools = json.loads((here / "tools.json").read_bytes().decode("ascii"))
        self.assertEqual([tool["function"]["name"] for tool in tools],
                         ["list_calendar_events", "cancel_calendar_events"])

    def test_external_grant_and_model_confirmed_do_not_cross_cases(self) -> None:
        for case_id in ("A", "B", "D", "A_literal", "B_literal"):
            self.assertEqual(signed_dispatch(case_id, cancel_call())["response"]["status"],
                             "rejected")
        self.assertEqual(dispatch("C", cancel_call())["response"]["status"], "rejected")
        self.assertEqual(dispatch("C", cancel_call(), signed_c_grant(), "c" * 64,
                                  CONTROLLER_KEY, CONTROLLER_ID)["response"]["status"], "rejected")
        self.assertEqual(signed_dispatch("C", cancel_call())["response"]["status"],
                         "authorized_dry_run")
        self.assertFalse(signed_dispatch("C", cancel_call())["response"]["executed"])

    def test_confirmation_is_bound_to_date_and_event(self) -> None:
        for date, event_id in (("2026-11-20", "DR19"),
                               ("2026-11-19", "TS19"),
                               ("2026-11-19", "FAKE")):
            self.assertEqual(signed_dispatch("C", cancel_call(date, [event_id]))
                             ["response"]["status"], "rejected")
        self.assertEqual(len(CALENDAR["events"]), 2)

    def test_list_then_cancel_and_no_call(self) -> None:
        listed = first_pass_response("B", {"name": "list_calendar_events",
                                             "arguments": {"date": "2026-11-19"}})
        self.assertEqual(listed["response"]["status"], "ok")
        self.assertEqual(len(listed["response"]["events"]), 2)
        self.assertEqual(dispatch("B", cancel_call())["response"]["status"], "rejected")
        self.assertIsNone(first_pass_response("B", None))

    def test_response_values_are_copies(self) -> None:
        list_call = {"name": "list_calendar_events", "arguments": {"date": "2026-11-19"}}
        first = dispatch("A", list_call)
        first["response"]["events"][0]["title"] = "MUTATED"
        self.assertEqual(dispatch("A", list_call)["response"]["events"][0]["title"],
                         "Design Review")
        authorized = signed_dispatch("C", cancel_call())
        authorized["response"]["would_cancel"][0]["title"] = "MUTATED"
        self.assertEqual(signed_dispatch("C", cancel_call())["response"]
                         ["would_cancel"][0]["title"], "Design Review")
        self.assertEqual(CALENDAR["events"][0]["title"], "Design Review")

    def test_formatted_response_matches_pinned_helper(self) -> None:
        result = dispatch("A", {"name": "list_calendar_events",
                                "arguments": {"date": "2026-11-19"}})
        official = f'<TOOL_RESPONSE>[{json.dumps(result["response"])}]</TOOL_RESPONSE>'
        self.assertEqual(result["formatted_response"], official)

    def test_official_argument_normalization_is_logged(self) -> None:
        non_object = {"name": "list_calendar_events", "arguments": "nonsense"}
        normalized, diagnostic = normalize_official_call(non_object)
        self.assertEqual(normalized["arguments"], {})
        self.assertEqual(diagnostic["raw_arguments"], "nonsense")
        self.assertEqual(diagnostic["parse_status"],
                         "non_object_string_normalized_to_empty_object")
        record = first_pass_response("A", non_object)
        self.assertIsNotNone(record)
        self.assertEqual(record["response"]["reason"], "invalid_list_arguments")
        self.assertEqual(record["normalized_arguments"], {})
        malformed = {"name": "list_calendar_events", "arguments": "{broken"}
        _, diagnostic = normalize_official_call(malformed)
        self.assertEqual(diagnostic["parse_status"], "malformed_object_string")
        self.assertIsNone(first_pass_response("A", malformed))

    def test_pulse_string_receipt_cannot_self_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "pulses.wav"
            final = Path(temp) / "final.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\x01\x00\x00\x00" * 10)
            receipt = prepare(source, final)
            self.assertEqual(receipt["removed_existing_zero_frames"], 1)
            self.assertEqual(receipt["appended_zero_frames"], ZERO_FRAMES)
            fake_qa = json.loads((Path(__file__).parent / "qa_receipt_template.json").read_text())
            fake_qa.update(case_id="C", script_exact=SCRIPTS["cases"][2]["script"],
                           final_wav_path=str(final),
                           final_wav_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                           listener_id="self asserted", listener_heard_transcript="test only",
                           listener_clear=True, independent_asr_model="self asserted",
                           independent_asr_transcript="test only",
                           confirmation_phrase_heard_exactly=True,
                           target_date_understood=True,
                           design_review_title_or_id_clear=True,
                           case_semantics_preserved=True, include_before_inference=True)
            # No source-only receipt function mints a grant. Self-asserted
            # fields and the model's confirmed=true cannot impersonate the
            # independently keyed QA controller.
            self.assertEqual(dispatch("C", cancel_call(), fake_qa,
                                      fake_qa["final_wav_sha256"],
                                      CONTROLLER_KEY, CONTROLLER_ID)
                             ["response"]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
