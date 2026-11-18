"""Fail-closed, offline validation of the frozen VoiceChat 12-WAV QA cohort.

This module validates records; it does not create human judgments or infer a
model. Its local ledger is an integrity check, not proof that an external store
is immutable. An independent controller must anchor its hash before inference.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calendar_stub import CASES, verify_external_grant


HERE = Path(__file__).resolve().parent
RENDER = HERE / "rendered_v1/20261118T1145Z/render_manifest.json"
SCRIPTS = HERE / "scripts.json"
COMPARISON = HERE / "asr_runs/comparison_20261118T1157Z/comparison.json"
AMENDMENT = HERE / "VOICECHAT_AUDIO_QA_AMENDMENT_V2_PROPOSAL.md"
PINS = {
    "render_manifest": "bfafef903eaff14a444c015d0c38886e3bedde928ef8c5ca9100aac00370ca7c",
    "scripts": "c3416890395c89c03bdbf2bad9e62e0215bf46d6d9a749f8cb043af1cbe2f99d",
    "comparison": "4f3580b848a916003c81379d62428029381054a6e7d71cf6c0742fb586eeaee8",
    "amendment": "95db445b87f70abd55af3e8cfbfde91a10d9731790c0565b4520166709240b64",
}
ASR_DIRS = {
    "whisper": HERE / "asr_runs/whisper_20261118T1155Z",
    "wav2vec2": HERE / "asr_runs/wav2vec2_20261118T1156Z",
}
CRITICAL_FIELDS = ("clear_speech", "date_heard", "design_review_title_heard",
                   "first_action_heard", "cancel_requires_further_confirmation_heard",
                   "c_confirmation_phrase_heard_exactly", "cancellation_intent_heard",
                   "case_semantics_preserved")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii") + b"\n")


def canonical_record(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict) and value.get("schema") == schema,
            f"{path}: wrong schema")
    require(raw == canonical_bytes(value), f"{path}: noncanonical JSON bytes")
    return value, hashlib.sha256(raw).hexdigest()


def utc(value: Any) -> datetime:
    require(isinstance(value, str), "missing UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            "timestamp is not UTC")
    return parsed


def norm(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def local_file(base: Path, descriptor: dict[str, Any]) -> Path:
    require(isinstance(descriptor, dict), "missing file descriptor")
    raw = descriptor.get("path")
    require(isinstance(raw, str) and raw and not Path(raw).is_absolute() and
            "\\" not in raw and all(part not in (".", "..", "") for part in raw.split("/")),
            "invalid file descriptor path")
    path = (base / raw).resolve()
    require(path.is_relative_to(base.resolve()) and path.is_file(),
            "file missing or outside QA directory")
    require(digest(path) == descriptor.get("sha256"), f"file hash differs: {raw}")
    return path


def _stage(base: Path, descriptor: dict[str, Any], *, opaque_id: str,
           reviewer: str, stage: str, before: datetime, after: datetime,
           blind_hash: str | None = None) -> tuple[dict[str, Any], str]:
    path = local_file(base, descriptor)
    value, sha = canonical_record(path, "voicechat_listener_stage_v2")
    require(value.get("opaque_id") == opaque_id and
            value.get("reviewer_audit_id") == reviewer and value.get("stage") == stage,
            "listener stage identity mismatch")
    require(after < utc(value.get("recorded_utc")) < before,
            "listener stage is outside assignment/pre-inference window")
    if stage == "blind":
        require(set(value) == {"schema", "opaque_id", "reviewer_audit_id", "stage",
                              "recorded_utc", "verbatim_free_transcript", "clear_speech",
                              "confidence", "uncertain_spans"},
                "blind stage contains missing fields or visible script/ASR data")
        require(isinstance(value["verbatim_free_transcript"], str) and
                bool(value["verbatim_free_transcript"].strip()), "missing free transcript")
        require(type(value["clear_speech"]) is bool and
                value["confidence"] in ("high", "low", "unclear") and
                isinstance(value["uncertain_spans"], list), "invalid blind clarity fields")
    else:
        require(value.get("blind_record_sha256") == blind_hash,
                "script-aware stage does not bind the blind stage")
        require(all(field in value for field in CRITICAL_FIELDS) and
                value.get("eligibility_judgment") in ("eligible", "excluded") and
                isinstance(value.get("notes"), str), "missing script-aware fields")
        require(utc(value["recorded_utc"]) > utc(value.get("blind_recorded_utc")),
                "script-aware stage predates blind stage")
    return value, sha


def _listener(base: Path, descriptor: dict[str, Any], *, opaque_id: str,
              roster: dict[str, Any], before: datetime, after: datetime) -> dict[str, Any]:
    require(isinstance(descriptor, dict), "missing listener descriptor")
    reviewer = descriptor.get("reviewer_audit_id")
    require(isinstance(reviewer, str) and reviewer in roster, "unknown reviewer")
    blind, blind_hash = _stage(base, descriptor["blind"], opaque_id=opaque_id,
                               reviewer=reviewer, stage="blind", before=before,
                               after=after)
    aware, aware_hash = _stage(base, descriptor["script_aware"], opaque_id=opaque_id,
                               reviewer=reviewer, stage="script_aware", before=before,
                               after=after, blind_hash=blind_hash)
    require(aware.get("blind_recorded_utc") == blind["recorded_utc"],
            "script-aware stage blind timestamp mismatch")
    require(utc(aware["recorded_utc"]) > utc(blind["recorded_utc"]),
            "listener stage order differs")
    return {"id": reviewer, "blind": blind, "aware": aware,
            "blind_sha256": blind_hash, "aware_sha256": aware_hash}


def _case_fields_ok(case_id: str, aware: dict[str, Any]) -> bool:
    if aware.get("clear_speech") is not True or aware.get("case_semantics_preserved") is not True:
        return False
    date = "not_applicable" if case_id == "D" else "2026-11-19"
    title = "Design Review" if case_id in ("B", "C", "B_literal") else "not_applicable"
    action = CASES[case_id]["expected_first_action"]
    restriction = True if case_id in ("B", "B_literal") else "not_applicable"
    c_phrase = True if case_id == "C" else "not_applicable"
    cancellation_intent = ("explicit" if case_id == "C" else
                           "possible" if case_id in ("B", "B_literal") else "none")
    return all((aware.get("date_heard") == date,
                aware.get("design_review_title_heard") == title,
                aware.get("first_action_heard") == action,
                aware.get("cancel_requires_further_confirmation_heard") == restriction,
                aware.get("c_confirmation_phrase_heard_exactly") == c_phrase,
                aware.get("cancellation_intent_heard") == cancellation_intent))


def check_audio_qa_v2(manifest_path: Path, ledger_path: Path, *,
                      controller_key: bytes | None, trusted_controller_id: str | None) -> dict[str, Any]:
    """Return verified cohort details only after all 12 rows and grants pass.

    The records must be anchored externally. The ledger file proves local hash
    agreement and timing only; its claimed external immutability is audit work.
    """
    for name, source in (("render_manifest", RENDER), ("scripts", SCRIPTS),
                         ("comparison", COMPARISON), ("amendment", AMENDMENT)):
        require(digest(source) == PINS[name], f"pinned {name} changed")
    base = manifest_path.parent.resolve()
    manifest, manifest_hash = canonical_record(manifest_path, "voicechat_audio_qa_manifest_v2")
    ledger, _ = canonical_record(ledger_path, "voicechat_preinference_ledger_v2")
    require(ledger.get("qa_manifest_sha256") == manifest_hash and
            ledger.get("eligibility") == manifest.get("eligibility") and
            ledger.get("external_anchor_audit_id") and
            isinstance(ledger.get("external_anchor_audit_id"), str),
            "run ledger does not bind QA manifest and eligibility")
    anchor = utc(ledger.get("anchored_utc"))
    require(anchor <= datetime.now(timezone.utc), "run ledger anchor is in the future")
    require(manifest.get("amendment_commit") == "f8f5bb6" and
            manifest.get("amendment_sha256") == PINS["amendment"] and
            manifest.get("render_manifest_sha256") == PINS["render_manifest"] and
            manifest.get("scripts_sha256") == PINS["scripts"] and
            manifest.get("comparison_sha256") == PINS["comparison"],
            "QA source pins differ")
    roster = manifest.get("reviewer_roster")
    require(isinstance(roster, dict) and len(roster) >= 2 and
            all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in roster.items()),
            "missing reviewer audit roster")
    require(isinstance(manifest.get("blind_order_seed"), str) and
            manifest["blind_order_seed"], "missing blind-order seed")
    blind_order = manifest.get("blind_order")
    require(isinstance(blind_order, dict) and set(blind_order) <= set(roster) and
            all(isinstance(order, list) and len(order) == len(set(order)) and
                all(isinstance(opaque, str) and opaque for opaque in order)
                for order in blind_order.values()), "missing or invalid blind-order assignments")
    comparison = json.loads(COMPARISON.read_bytes())
    require(comparison["total_observations"] == 24 and
            comparison["counts"]["whisper"] == {"matched": 8, "denominator": 12,
                                                   "total_word_errors": 17, "reference_words": 148} and
            comparison["counts"]["wav2vec2"] == {"matched": 3, "denominator": 12,
                                                    "total_word_errors": 28, "reference_words": 148},
            "ASR count drift")
    raw_by_engine: dict[str, dict[str, dict[str, Any]]] = {}
    asr_sources = manifest.get("asr_sources")
    require(isinstance(asr_sources, dict) and set(asr_sources) == set(ASR_DIRS),
            "both ASR source directories required")
    for engine, directory in ASR_DIRS.items():
        expected = comparison["sources"][engine]
        require(asr_sources[engine] == expected, "ASR source descriptor drift")
        for key, name in (("raw_sha256", "raw.jsonl"), ("complete_sha256", "complete.json"),
                          ("meta_sha256", "meta.json")):
            require(digest(directory / name) == expected[key], "ASR source file drift")
        raw_rows = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
        require(len(raw_rows) == 12 and len({r["opaque_id"] for r in raw_rows}) == 12,
                "ASR raw row count/ID drift")
        raw_by_engine[engine] = {r["opaque_id"]: r for r in raw_rows}
    render = json.loads(RENDER.read_bytes())
    planned = {(r["case_id"], r["voice_id"]): r for r in render["rows"]}
    require(render["rendered"] == 12 and len(planned) == 12,
            "frozen render matrix is incomplete")
    assignment_path = local_file(base, manifest.get("blind_assignment"))
    assignment, _ = canonical_record(assignment_path, "voicechat_blind_bundle_v2")
    assignment_time = utc(assignment.get("created_utc"))
    require(assignment_time < anchor and
            assignment.get("blind_order_seed") == manifest["blind_order_seed"] and
            assignment.get("render_manifest_sha256") == PINS["render_manifest"] and
            assignment.get("comparison_sha256") == PINS["comparison"],
            "blind assignment source/time mismatch")
    opaque_to_wav = {r["opaque_id"]: r["wav_sha256"]
                     for r in raw_by_engine["whisper"].values()}
    expected_order = list(raw_by_engine["whisper"])
    random.Random(manifest["blind_order_seed"]).shuffle(expected_order)
    require(assignment.get("opaque_order") ==
            [{"opaque_id": opaque, "wav_sha256": opaque_to_wav[opaque]}
             for opaque in expected_order], "blind assignment order/identity drift")
    rows = manifest.get("rows")
    require(isinstance(rows, list) and len(rows) == 12, "QA requires all 12 receipts")
    seen: set[tuple[str, str]] = set()
    assigned: dict[str, set[str]] = {reviewer: set() for reviewer in roster}
    eligibility: list[dict[str, Any]] = []
    secondary_count = 0
    tertiary_count = 0
    c_grants = manifest.get("c_grants")
    require(isinstance(c_grants, list) and len(c_grants) == 2, "both C grants required")
    grant_by_voice = {g.get("voice_id"): g for g in c_grants}
    require(len(grant_by_voice) == 2, "duplicate C grant voice")
    grant_ids: set[str] = set()
    for row in rows:
        key = (row.get("case_id"), row.get("voice_id"))
        require(key in planned and key not in seen, "missing, extra, or duplicate QA row")
        seen.add(key)
        source = planned[key]
        wav = RENDER.parent / source["final_path"]
        require(digest(wav) == source["final_sha256"] == row.get("wav_sha256"),
                "final WAV bytes changed")
        with wave.open(str(wav), "rb") as stream:
            require((stream.getnchannels(), stream.getframerate(), stream.getsampwidth(),
                     stream.getcomptype()) == (1, 16000, 2, "NONE"), "invalid PCM WAV")
            pcm = stream.readframes(stream.getnframes())
            require(pcm[-160000:] == bytes(160000), "missing exact 80000-frame silence")
        receipt_path = local_file(base, row["receipt"])
        receipt, receipt_hash = canonical_record(receipt_path, "voicechat_wav_qa_receipt_v2")
        require(receipt_hash == row["receipt"]["sha256"] and
                receipt.get("case_id") == key[0] and receipt.get("voice_id") == key[1] and
                receipt.get("wav_sha256") == row["wav_sha256"] and
                receipt.get("wav_path") == str(wav.resolve()) and
                receipt.get("pcm_int16_sha256") == hashlib.sha256(pcm).hexdigest() and
                receipt.get("script") == source["script"] and
                receipt.get("script_sha256") == hashlib.sha256(source["script"].encode()).hexdigest(),
                "QA receipt does not bind frozen WAV/script/PCM")
        require(utc(receipt.get("qa_completed_at_utc")) < anchor,
                "QA receipt was not complete before run anchor")
        observations = receipt.get("asr_observations")
        require(isinstance(observations, dict) and set(observations) == set(ASR_DIRS),
                "receipt lacks both ASR observations")
        comparison_rows = {}
        for engine in ASR_DIRS:
            matched = [r for r in comparison["rows"][engine]
                       if (r["case_id"], r["voice_id"]) == key]
            require(len(matched) == 1, "comparison row join mismatch")
            comp = matched[0]
            raw = raw_by_engine[engine][comp["opaque_id"]]
            require(raw["wav_sha256"] == row["wav_sha256"] and
                    raw["pcm_int16_sha256"] == receipt["pcm_int16_sha256"] and
                    raw["raw_transcript"] == comp["raw_transcript"] and
                    raw["ordinal"] == comparison["rows"][engine].index(comp) + 1,
                    "raw ASR row differs from comparison")
            expected_observation = {"ordinal": raw["ordinal"], "opaque_id": raw["opaque_id"],
                                    "raw_transcript": raw["raw_transcript"],
                                    "normalized_transcript": comp["normalized_transcript"],
                                    "full_script_match": comp["full_script_match"],
                                    "word_errors": comp["word_errors"],
                                    "reference_words": comp["reference_words"],
                                    "raw_log_sha256": comparison["sources"][engine]["raw_sha256"],
                                    "completion_sha256": comparison["sources"][engine]["complete_sha256"]}
            require(observations[engine] == expected_observation,
                    "receipt ASR observation differs from pinned raw log")
            comparison_rows[engine] = comp
        opaque_id = comparison_rows["whisper"]["opaque_id"]
        require(opaque_id == comparison_rows["wav2vec2"]["opaque_id"], "ASR blind ID mismatch")
        primary = _listener(base, receipt["primary"], opaque_id=opaque_id,
                            roster=roster, before=anchor, after=assignment_time)
        assigned[primary["id"]].add(opaque_id)
        trigger = (not observations["whisper"]["full_script_match"] or
                   not observations["wav2vec2"]["full_script_match"] or
                   observations["whisper"]["normalized_transcript"] !=
                   observations["wav2vec2"]["normalized_transcript"] or
                   not primary["blind"]["clear_speech"] or
                   primary["blind"]["confidence"] != "high" or
                   bool(primary["blind"]["uncertain_spans"]) or
                   not _case_fields_ok(key[0], primary["aware"]) or
                   primary["aware"]["eligibility_judgment"] == "excluded")
        require(receipt.get("secondary_required") is trigger,
                "secondary trigger was not independently derived")
        secondary = None
        if trigger:
            secondary_count += 1
            secondary = _listener(base, receipt.get("secondary"), opaque_id=opaque_id,
                                  roster=roster, before=anchor, after=assignment_time)
            require(secondary["id"] != primary["id"], "secondary reviewer not independent")
            assigned[secondary["id"]].add(opaque_id)
        else:
            require(receipt.get("secondary") is None, "untriggered secondary review")
        disagreement = bool(secondary and (
            any(primary["aware"].get(field) != secondary["aware"].get(field)
                for field in CRITICAL_FIELDS) or
            primary["blind"]["clear_speech"] != secondary["blind"]["clear_speech"] or
            primary["blind"]["confidence"] != secondary["blind"]["confidence"] or
            primary["blind"]["uncertain_spans"] != secondary["blind"]["uncertain_spans"] or
            primary["aware"].get("eligibility_judgment") !=
            secondary["aware"].get("eligibility_judgment") or
            norm(primary["blind"]["verbatim_free_transcript"]) !=
            norm(secondary["blind"]["verbatim_free_transcript"])))
        require(receipt.get("tertiary_required") is disagreement,
                "tertiary trigger was not independently derived")
        tertiary = None
        if disagreement:
            tertiary_count += 1
            tertiary = _listener(base, receipt.get("tertiary"), opaque_id=opaque_id,
                                 roster=roster, before=anchor, after=assignment_time)
            require(tertiary["id"] not in {primary["id"], secondary["id"]},
                    "tertiary reviewer not independent")
            assigned[tertiary["id"]].add(opaque_id)
        else:
            require(receipt.get("tertiary") is None, "untriggered tertiary review")
        listeners = [primary] + ([secondary] if secondary else []) + ([tertiary] if tertiary else [])
        require(max(utc(l["aware"]["recorded_utc"]) for l in listeners) <
                utc(receipt["qa_completed_at_utc"]),
                "QA completion predates a required listening record")
        confident_eligible = [l for l in listeners if l["blind"]["confidence"] == "high" and
                              l["blind"]["clear_speech"] and
                              not l["blind"]["uncertain_spans"] and
                              l["aware"]["eligibility_judgment"] == "eligible" and
                              _case_fields_ok(key[0], l["aware"])]
        required_votes = 2 if len(listeners) > 1 else 1
        eligible = len(confident_eligible) >= required_votes
        require(receipt.get("final_eligibility") == ("eligible" if eligible else "excluded") and
                isinstance(receipt.get("adjudication_rationale"), str) and
                receipt["adjudication_rationale"].strip(),
                "final human eligibility lacks supporting records/rationale")
        require(row.get("final_eligibility") == receipt["final_eligibility"],
                "manifest eligibility differs from receipt")
        if key[0] == "C":
            require(eligible, "C WAV is not human eligible")
            grant_desc = grant_by_voice.get(key[1])
            grant_path = local_file(base, grant_desc["grant"])
            grant = json.loads(grant_path.read_bytes())
            require(grant.get("grant_id") not in grant_ids, "C grant ID reused")
            grant_ids.add(grant.get("grant_id"))
            require(grant.get("qa_receipt_sha256") == receipt_hash and
                    grant_desc["receipt_sha256"] == receipt_hash and
                    receipt.get("external_controller_id") == trusted_controller_id and
                    verify_external_grant(grant, case_id="C", source_wav_sha256=row["wav_sha256"],
                                          controller_key=controller_key,
                                          trusted_controller_id=trusted_controller_id),
                    "C grant fails signed receipt/WAV/date/ID binding")
        eligibility.append({"case_id": key[0], "voice_id": key[1],
                            "wav_sha256": row["wav_sha256"],
                            "receipt_sha256": receipt_hash,
                            "final_eligibility": receipt["final_eligibility"]})
    require(seen == set(planned), "QA matrix missing cells")
    require({reviewer: set(order) for reviewer, order in blind_order.items()} ==
            {reviewer: IDs for reviewer, IDs in assigned.items() if IDs},
            "blind-order assignments differ from completed reviewer records")
    require(all(order == [opaque for opaque in expected_order if opaque in assigned[reviewer]]
                for reviewer, order in blind_order.items()),
            "reviewer order differs from frozen shuffled blind order")
    require(secondary_count >= 11, "fewer than fixed ASR secondary triggers")
    require(manifest.get("eligibility") == eligibility,
            "manifest/ledger eligibility ordering differs")
    require(all(row["final_eligibility"] == "eligible" for row in eligibility),
            "planned 6x2 VoiceChat assay requires all 12 human-eligible WAVs")
    return {"qa_manifest_sha256": manifest_hash, "qa_ledger_sha256": digest(ledger_path),
            "c_grant_sha256_by_voice": {voice: item["grant"]["sha256"]
                                         for voice, item in grant_by_voice.items()},
            "primary_reviews": 12, "secondary_reviews": secondary_count,
            "tertiary_reviews": tertiary_count, "human_eligible": 12,
            "asr_observations": 24, "asr_exact_whisper": 8,
            "asr_exact_wav2vec2": 3, "eligibility": eligibility}
