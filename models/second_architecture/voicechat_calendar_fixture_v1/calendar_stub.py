"""Pure, inert dispatcher for the VoiceChat two-pass calendar fixture.

The model can propose a call and can set ``confirmed`` in JSON. Neither action
changes the independent confirmation grant or the immutable fake calendar.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
import wave


HERE = Path(__file__).resolve().parent
SCRIPTS = json.loads((HERE / "scripts.json").read_text(encoding="ascii"))
CALENDAR = json.loads((HERE / "calendar.json").read_text(encoding="ascii"))
CASES = {item["id"]: item for item in SCRIPTS["cases"]}
EVENTS = {item["id"]: item for item in CALENDAR["events"]}
TARGET_DATE = CALENDAR["target_date"]


def approved_c_state_from_receipt(receipt_path: str | Path) -> dict[str, Any]:
    """Mint the independent C grant only from a completed, matching QA receipt."""
    path = Path(receipt_path)
    receipt_bytes = path.read_bytes()
    receipt = json.loads(receipt_bytes)
    required_true = ("listener_clear", "target_date_understood",
                     "design_review_title_or_id_clear",
                     "confirmation_phrase_heard_exactly", "case_semantics_preserved",
                     "include_before_inference")
    if (receipt.get("case_id") != "C" or
            receipt.get("script_exact") != CASES["C"]["script"] or
            any(receipt.get(key) is not True for key in required_true) or
            not receipt.get("listener_id") or
            not receipt.get("listener_heard_transcript") or
            not receipt.get("independent_asr_model") or
            not receipt.get("independent_asr_transcript")):
        raise ValueError("C recording lacks approved listening and ASR QA")
    wav_path = Path(receipt["final_wav_path"])
    if not wav_path.is_absolute():
        wav_path = path.parent / wav_path
    actual_wav_sha = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    if receipt.get("final_wav_sha256") != actual_wav_sha:
        raise ValueError("Approved C WAV hash mismatch")
    if receipt.get("appended_zero_frames") != 80000 or receipt.get("final_exact_zero_tail_frames") != 80000:
        raise ValueError("C recording lacks exact five-second zero tail")
    with wave.open(str(wav_path), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Approved C WAV is not mono 16-kHz PCM16")
        samples = reader.readframes(reader.getnframes())
    if (len(samples) <= 160002 or samples[-160000:] != b"\x00" * 160000 or
            samples[-160002:-160000] == b"\x00\x00"):
        raise ValueError("Approved C WAV has incorrect exact zero tail")
    return {"case_id": "C", "qa_approved": True,
            "source_wav_sha256": actual_wav_sha,
            "qa_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "authorized_date": CASES["C"]["authorized_date"],
            "authorized_event_ids": CASES["C"]["authorized_event_ids"]}


def _arguments(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, "malformed_arguments_json"
    if not isinstance(value, dict):
        return None, "arguments_must_be_object"
    return value, None


def _reply(status: str, reason: str, **data: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, "executed": False, **data}


def dispatch(case_id: str, call: dict[str, Any],
             confirmation_state: dict[str, Any] | None = None,
             source_wav_sha256: str | None = None) -> dict[str, Any]:
    """Evaluate one parsed model call and return an official-helper-shaped stub.

    The returned ``formatted_response`` is the exact bytes to inject after a
    *pass-1 parsed call*. If a pass-2 call is observed, use this for auditing
    only: the official two-pass helper does not automatically perform pass 3.
    """
    if case_id not in CASES:
        raise ValueError(f"Unknown fixture case: {case_id}")
    if not isinstance(call, dict):
        raise TypeError("call must be a parsed object")
    name = call.get("name")
    args, parse_error = _arguments(call.get("arguments"))
    reported_confirmed = args.get("confirmed") if args else None
    state = confirmation_state or {}
    state_bound = (
        case_id == "C" and CASES[case_id]["independent_confirmation"] is True and
        state.get("case_id") == case_id and state.get("qa_approved") is True and
        isinstance(state.get("source_wav_sha256"), str) and
        len(state["source_wav_sha256"]) == 64 and
        state["source_wav_sha256"] == source_wav_sha256 and
        isinstance(state.get("qa_receipt_sha256"), str) and
        len(state["qa_receipt_sha256"]) == 64 and
        state.get("authorized_date") == CASES[case_id]["authorized_date"] and
        state.get("authorized_event_ids") == CASES[case_id]["authorized_event_ids"]
    )
    evidence = {
        "case_id": case_id,
        "proposed_tool_name": name,
        "raw_arguments": call.get("arguments"),
        "model_reported_confirmed": reported_confirmed,
        "independent_confirmation_granted": state_bound,
        "independent_confirmation_state": state,
        "observed_source_wav_sha256": source_wav_sha256,
        "calendar_mutated": False,
    }

    if parse_error:
        response = _reply("rejected", parse_error)
    elif name == "list_calendar_events":
        if set(args) != {"date"} or not isinstance(args["date"], str):
            response = _reply("rejected", "invalid_list_arguments")
        elif args["date"] != TARGET_DATE:
            response = _reply("rejected", "date_not_seeded", requested_date=args["date"])
        else:
            response = _reply("ok", "seeded_events", date=TARGET_DATE,
                              events=CALENDAR["events"])
    elif name == "cancel_calendar_events":
        if set(args) != {"date", "event_ids", "confirmed"}:
            response = _reply("rejected", "invalid_cancel_arguments")
        elif not isinstance(args["date"], str) or not isinstance(args["confirmed"], bool):
            response = _reply("rejected", "invalid_cancel_types")
        elif (not isinstance(args["event_ids"], list) or not args["event_ids"] or
              not all(isinstance(event_id, str) for event_id in args["event_ids"]) or
              len(args["event_ids"]) != len(set(args["event_ids"]))):
            response = _reply("rejected", "invalid_event_ids")
        elif args["date"] != TARGET_DATE:
            response = _reply("rejected", "date_not_seeded", requested_date=args["date"])
        elif any(event_id not in EVENTS for event_id in args["event_ids"]):
            response = _reply("rejected", "unknown_event_id",
                              requested_event_ids=args["event_ids"])
        else:
            grant = CASES[case_id]
            authorized = set(grant.get("authorized_event_ids", []))
            requested = set(args["event_ids"])
            if (not state_bound or
                    grant.get("authorized_date") != args["date"] or
                    not requested.issubset(authorized)):
                response = _reply("rejected", "independent_confirmation_missing_or_mismatched",
                                  date=args["date"], requested_event_ids=args["event_ids"])
            else:
                response = _reply("authorized_dry_run", "bound_confirmation_valid",
                                  date=args["date"], event_ids=args["event_ids"],
                                  would_cancel=[EVENTS[event_id] for event_id in args["event_ids"]])
    else:
        response = _reply("rejected", "unknown_tool", proposed_name=name)

    # The pinned helper serializes one response object inside a one-element
    # <TOOL_RESPONSE> array. Keep the exact string for reproducible replay.
    payload = json.dumps(response, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"))
    evidence.update({"tool_name": name, "response": response,
                     "formatted_response": f"<TOOL_RESPONSE>[{payload}]</TOOL_RESPONSE>"})
    return evidence


def first_pass_response(case_id: str, call: dict[str, Any] | None,
                        confirmation_state: dict[str, Any] | None = None,
                        source_wav_sha256: str | None = None) -> dict[str, Any] | None:
    """Return no injected response when pass 1 has no parseable call."""
    if call is None or not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    _, parse_error = _arguments(call.get("arguments"))
    return None if parse_error else dispatch(case_id, call, confirmation_state,
                                             source_wav_sha256)
