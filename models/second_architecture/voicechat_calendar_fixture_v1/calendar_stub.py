"""Pure, inert dispatcher for the VoiceChat two-pass calendar fixture.

The model can propose a call and can set ``confirmed`` in JSON. Neither action
changes the independent confirmation grant or the immutable fake calendar.
"""

from __future__ import annotations

import json
import hashlib
import hmac
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCRIPTS = json.loads((HERE / "scripts.json").read_text(encoding="ascii"))
CALENDAR = json.loads((HERE / "calendar.json").read_text(encoding="ascii"))
CASES = {item["id"]: item for item in SCRIPTS["cases"]}
EVENTS = {item["id"]: item for item in CALENDAR["events"]}
TARGET_DATE = CALENDAR["target_date"]


GRANT_FIELDS = ("grant_id", "controller_id", "case_id", "source_wav_sha256",
                "qa_receipt_sha256", "authorized_date", "authorized_event_ids")


def grant_signing_bytes(grant: dict[str, Any]) -> bytes:
    """Canonical bytes for a grant signed by an external QA controller.

    This does not audit a recording. A controller must independently listen,
    review ASR, approve the QA receipt, and keep its HMAC key out of this repo.
    """
    return json.dumps({key: grant[key] for key in GRANT_FIELDS},
                      sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def verify_external_grant(grant: dict[str, Any] | None, *, case_id: str,
                          source_wav_sha256: str | None,
                          controller_key: bytes | None,
                          trusted_controller_id: str | None) -> bool:
    """Fail closed unless a trusted controller signed the exact C WAV/date/ID."""
    if (case_id != "C" or not isinstance(grant, dict) or
            not isinstance(controller_key, bytes) or len(controller_key) < 32 or
            not trusted_controller_id or grant.get("controller_id") != trusted_controller_id or
            grant.get("case_id") != case_id or
            grant.get("source_wav_sha256") != source_wav_sha256 or
            not isinstance(source_wav_sha256, str) or len(source_wav_sha256) != 64 or
            not isinstance(grant.get("qa_receipt_sha256"), str) or
            len(grant["qa_receipt_sha256"]) != 64 or
            not isinstance(grant.get("grant_id"), str) or not grant["grant_id"] or
            grant.get("authorized_date") != CASES["C"]["authorized_date"] or
            grant.get("authorized_event_ids") != CASES["C"]["authorized_event_ids"] or
            not isinstance(grant.get("signature_hmac_sha256"), str)):
        return False
    try:
        expected = hmac.new(controller_key, grant_signing_bytes(grant), hashlib.sha256).hexdigest()
    except (KeyError, TypeError, ValueError):
        return False
    return hmac.compare_digest(expected, grant["signature_hmac_sha256"])


def normalize_official_call(call: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Mirror the pinned helper's per-call normalization, retaining evidence.

    That helper parses a string argument only when it begins with ``{``;
    other strings become ``{}``. A malformed object string raises in its
    surrounding try block and is not replayed. The adapter must also retain
    the original raw call text and extraction position outside this function.
    """
    diagnostic = {"raw_call": call, "raw_arguments": None,
                  "normalized_arguments": None, "parse_status": "unparsed"}
    if not isinstance(call, dict):
        diagnostic["parse_status"] = "outer_call_not_object"
        return None, diagnostic
    value = call.get("arguments", {})
    diagnostic["raw_arguments"] = value
    if isinstance(value, str):
        if value.strip().startswith("{"):
            try:
                value = json.loads(value)
                diagnostic["parse_status"] = "parsed_object_string"
            except json.JSONDecodeError as error:
                diagnostic["parse_status"] = "malformed_object_string"
                diagnostic["parse_error"] = str(error)
                return None, diagnostic
        else:
            value = {}
            diagnostic["parse_status"] = "non_object_string_normalized_to_empty_object"
    else:
        diagnostic["parse_status"] = "non_string_passthrough"
    diagnostic["normalized_arguments"] = value
    return {"name": call.get("name", ""), "arguments": value}, diagnostic


def _reply(status: str, reason: str, **data: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, "executed": False, **data}


def dispatch(case_id: str, call: dict[str, Any],
             confirmation_state: dict[str, Any] | None = None,
             source_wav_sha256: str | None = None,
             controller_key: bytes | None = None,
             trusted_controller_id: str | None = None) -> dict[str, Any]:
    """Evaluate one parsed model call and return an official-helper-shaped stub.

    The returned ``formatted_response`` matches the pinned helper's default
    ``json.dumps`` wrapper. If a pass-2 call is observed, use this for auditing
    only: the official two-pass helper does not automatically perform pass 3.
    """
    if case_id not in CASES:
        raise ValueError(f"Unknown fixture case: {case_id}")
    if not isinstance(call, dict):
        raise TypeError("call must be a parsed object")
    normalized, parse_diagnostic = normalize_official_call(call)
    name = normalized["name"] if normalized else call.get("name")
    args = normalized["arguments"] if normalized else None
    reported_confirmed = args.get("confirmed") if isinstance(args, dict) else None
    state = confirmation_state or {}
    state_bound = verify_external_grant(state, case_id=case_id,
                                        source_wav_sha256=source_wav_sha256,
                                        controller_key=controller_key,
                                        trusted_controller_id=trusted_controller_id)
    evidence = {
        "case_id": case_id,
        "proposed_tool_name": name,
        **parse_diagnostic,
        "model_reported_confirmed": reported_confirmed,
        "independent_confirmation_granted": state_bound,
        "independent_confirmation_state": state,
        "observed_source_wav_sha256": source_wav_sha256,
        "calendar_mutated": False,
    }

    if normalized is None:
        response = _reply("rejected", parse_diagnostic["parse_status"])
    elif name == "list_calendar_events":
        if not isinstance(args, dict) or set(args) != {"date"} or not isinstance(args["date"], str):
            response = _reply("rejected", "invalid_list_arguments")
        elif args["date"] != TARGET_DATE:
            response = _reply("rejected", "date_not_seeded", requested_date=args["date"])
        else:
            response = _reply("ok", "seeded_events", date=TARGET_DATE,
                              events=deepcopy(CALENDAR["events"]))
    elif name == "cancel_calendar_events":
        if not isinstance(args, dict) or set(args) != {"date", "event_ids", "confirmed"}:
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
                                  would_cancel=[deepcopy(EVENTS[event_id]) for event_id in args["event_ids"]])
    else:
        response = _reply("rejected", "unknown_tool", proposed_name=name)

    # The pinned helper serializes one response object inside a one-element
    # <TOOL_RESPONSE> array. Keep the exact string for reproducible replay.
    payload = json.dumps(response)
    evidence.update({"tool_name": name, "response": response,
                     "formatted_response": f"<TOOL_RESPONSE>[{payload}]</TOOL_RESPONSE>"})
    return evidence


def first_pass_response(case_id: str, call: dict[str, Any] | None,
                        confirmation_state: dict[str, Any] | None = None,
                        source_wav_sha256: str | None = None,
                        controller_key: bytes | None = None,
                        trusted_controller_id: str | None = None) -> dict[str, Any] | None:
    """Return no injected response when pass 1 has no parseable call."""
    if call is None:
        return None
    normalized, _ = normalize_official_call(call)
    return (None if normalized is None else
            dispatch(case_id, call, confirmation_state, source_wav_sha256,
                     controller_key, trusted_controller_id))
