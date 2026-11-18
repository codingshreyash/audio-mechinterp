"""Build QA v2 evidence from immutable listener stages, without model inference.

Receipts are derived from saved human records. A later manifest requires two
separately signed C grants. This module never creates grants, an external
anchor, or a pre-inference ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import audio_qa_v2 as qa
from .calendar_stub import verify_external_grant


STAGE_NAME = re.compile(
    r"(?P<reviewer>[A-Za-z0-9_-]{3,80})__(?P<opaque>[0-9a-f]{20})__"
    r"(?P<stage>blind|script_aware)\.json\Z")
INDEX_SCHEMA = "voicechat_audio_qa_prepared_index_v2"


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(qa.canonical_bytes(value))


def _descriptor(base: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(base).as_posix(), "sha256": qa.digest(path)}


def _sources(bundle: Path) -> tuple[dict, dict, dict, datetime]:
    for name, path in (("render_manifest", qa.RENDER), ("scripts", qa.SCRIPTS),
                       ("comparison", qa.COMPARISON), ("amendment", qa.AMENDMENT)):
        qa.require(qa.digest(path) == qa.PINS[name], f"pinned {name} changed")
    render = json.loads(qa.RENDER.read_bytes())
    comparison = json.loads(qa.COMPARISON.read_bytes())
    qa.require(render["rendered"] == len(render["rows"]) == 12 and
               comparison["total_observations"] == 24, "frozen population differs")
    assignment_path = bundle / "blind_assignment.json"
    qa.require(assignment_path.is_file() and not assignment_path.is_symlink(),
               "blind assignment missing or symlinked")
    assignment, _ = qa.canonical_record(assignment_path, "voicechat_blind_bundle_v2")
    order = [row["opaque_id"] for row in comparison["rows"]["whisper"]]
    qa.require(len(order) == len(set(order)) == 12, "ASR opaque IDs differ")
    random.Random(assignment["blind_order_seed"]).shuffle(order)
    raw_rows = [json.loads(line) for line in
                (qa.ASR_DIRS["whisper"] / "raw.jsonl").read_text().splitlines()]
    expected = {row["opaque_id"]: row["wav_sha256"] for row in raw_rows}
    qa.require(len(expected) == 12 and
               assignment.get("render_manifest_sha256") == qa.PINS["render_manifest"] and
               assignment.get("comparison_sha256") == qa.PINS["comparison"] and
               assignment.get("opaque_order") ==
               [{"opaque_id": opaque, "wav_sha256": expected[opaque]} for opaque in order],
               "blind assignment differs from pinned sources")
    assignment_time = qa.utc(assignment["created_utc"])
    qa.require(qa.digest(qa.RENDER_META) == render["meta_sha256"],
               "render metadata changed")
    render_time = qa.utc(json.loads(qa.RENDER_META.read_bytes())["started_utc"])
    qa.require(render_time < assignment_time <= datetime.now(timezone.utc),
               "blind assignment chronology differs")
    for opaque, digest in expected.items():
        wav = bundle / f"{opaque}.wav"
        qa.require(wav.is_file() and not wav.is_symlink() and qa.digest(wav) == digest,
                   f"blinded WAV differs: {opaque}")
    return render, comparison, assignment, assignment_time


def _stages(root: Path, allowed: set[str], reviewers: set[str]) -> dict[tuple[str, str, str], Path]:
    qa.require(root.is_dir() and not root.is_symlink(), "records root missing or symlinked")
    found = {}
    for path in root.rglob("*"):
        qa.require(not path.is_symlink(), "listener record symlink")
        if path.is_dir():
            continue
        qa.require(path.is_file(), "listener record tree contains a non-file")
        match = STAGE_NAME.fullmatch(path.name)
        qa.require(match is not None, f"unexpected listener file: {path.name}")
        key = (match["reviewer"], match["opaque"], match["stage"])
        qa.require(key[0] in reviewers and key[1] in allowed and key not in found,
                   "unknown or duplicate listener stage")
        value, _ = qa.canonical_record(path, "voicechat_listener_stage_v2")
        qa.require((value.get("reviewer_audit_id"), value.get("opaque_id"),
                    value.get("stage")) == key, "listener filename/content differs")
        found[key] = path
    return found


def _asr(comparison: dict, case: str, voice: str, wav_sha: str,
         pcm_sha: str) -> tuple[str, dict]:
    observations = {}
    opaque = None
    for engine, directory in qa.ASR_DIRS.items():
        source = comparison["sources"][engine]
        for field, name in (("raw_sha256", "raw.jsonl"), ("complete_sha256", "complete.json"),
                            ("meta_sha256", "meta.json")):
            qa.require(qa.digest(directory / name) == source[field], "ASR source changed")
        matching = [row for row in comparison["rows"][engine]
                    if (row["case_id"], row["voice_id"]) == (case, voice)]
        qa.require(len(matching) == 1, "ASR comparison row missing or duplicated")
        comp = matching[0]
        raw_rows = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
        matching_raw = [row for row in raw_rows if row["opaque_id"] == comp["opaque_id"]]
        qa.require(len(raw_rows) == 12 and len(matching_raw) == 1, "ASR raw row differs")
        raw = matching_raw[0]
        qa.require(raw["wav_sha256"] == wav_sha and raw["pcm_int16_sha256"] == pcm_sha and
                   raw["raw_transcript"] == comp["raw_transcript"] and
                   raw["ordinal"] == comparison["rows"][engine].index(comp) + 1,
                   "ASR raw/comparison mismatch")
        qa.require(opaque is None or opaque == comp["opaque_id"], "ASR opaque ID mismatch")
        opaque = comp["opaque_id"]
        observations[engine] = {
            "ordinal": raw["ordinal"], "opaque_id": raw["opaque_id"],
            "raw_transcript": raw["raw_transcript"],
            "normalized_transcript": comp["normalized_transcript"],
            "full_script_match": comp["full_script_match"],
            "word_errors": comp["word_errors"],
            "reference_words": comp["reference_words"],
            "raw_log_sha256": source["raw_sha256"],
            "completion_sha256": source["complete_sha256"]}
    qa.require(opaque is not None, "ASR opaque ID missing")
    return opaque, observations


def _secondary(obs: dict, primary: dict, case: str) -> bool:
    return (not obs["whisper"]["full_script_match"] or
            not obs["wav2vec2"]["full_script_match"] or
            obs["whisper"]["normalized_transcript"] !=
            obs["wav2vec2"]["normalized_transcript"] or
            not primary["blind"]["clear_speech"] or
            primary["blind"]["confidence"] != "high" or
            bool(primary["blind"]["uncertain_spans"]) or
            not qa._case_fields_ok(case, primary["aware"]) or
            primary["aware"]["eligibility_judgment"] == "excluded")


def _tertiary(primary: dict, secondary: dict | None) -> bool:
    return bool(secondary and (
        any(primary["aware"].get(field) != secondary["aware"].get(field)
            for field in qa.CRITICAL_FIELDS) or
        primary["blind"]["clear_speech"] != secondary["blind"]["clear_speech"] or
        primary["blind"]["confidence"] != secondary["blind"]["confidence"] or
        primary["blind"]["uncertain_spans"] != secondary["blind"]["uncertain_spans"] or
        primary["aware"].get("eligibility_judgment") !=
        secondary["aware"].get("eligibility_judgment") or
        qa.norm(primary["blind"]["verbatim_free_transcript"]) !=
        qa.norm(secondary["blind"]["verbatim_free_transcript"])))


def build_receipts(bundle: Path, records_root: Path, out: Path, *,
                   primary_id: str, secondary_id: str, tertiary_id: str | None,
                   completed_at_utc: str, controller_id: str) -> dict:
    """Derive all twelve receipts; keep source records byte-identical."""
    roles = [primary_id, secondary_id] + ([tertiary_id] if tertiary_id else [])
    qa.require(all(isinstance(role, str) and
                   re.fullmatch(r"[A-Za-z0-9_-]{3,80}", role) for role in roles) and
               len(roles) == len(set(roles)), "reviewer audit IDs invalid or duplicated")
    qa.require(isinstance(controller_id, str) and bool(controller_id.strip()),
               "nominated external controller audit ID required")
    completed = qa.utc(completed_at_utc)
    qa.require(completed <= datetime.now(timezone.utc), "QA completion time is in future")
    qa.require(not out.exists() and not out.is_symlink(), "output already exists")
    qa.require(not out.resolve().is_relative_to(records_root.resolve()) and
               not out.resolve().is_relative_to(bundle.resolve()),
               "output cannot be inside record or bundle directory")
    render, comparison, assignment, assignment_time = _sources(bundle)
    qa.require(assignment_time < completed, "QA completion predates assignment")
    order = [row["opaque_id"] for row in assignment["opaque_order"]]
    staged = _stages(records_root, set(order), set(roles))
    roster = {role: role for role in roles}
    used: set[tuple[str, str, str]] = set()
    validated_hashes: dict[Path, str] = {}
    assigned = {role: set() for role in roles}
    planned = []

    def listener(role: str, opaque: str) -> tuple[dict, dict]:
        keys = ((role, opaque, "blind"), (role, opaque, "script_aware"))
        qa.require(all(key in staged for key in keys), "required human listener stage missing")
        original = {"reviewer_audit_id": role}
        copied = {"reviewer_audit_id": role}
        for key, field in zip(keys, ("blind", "script_aware")):
            path = staged[key]
            source_sha = qa.digest(path)
            validated_hashes[path] = source_sha
            original[field] = {"path": path.relative_to(records_root).as_posix(),
                               "sha256": source_sha}
            copied[field] = {"path": f"stages/{path.name}", "sha256": source_sha}
            used.add(key)
        checked = qa._listener(records_root, original, opaque_id=opaque, roster=roster,
                               before=completed, after=assignment_time)
        assigned[role].add(opaque)
        return checked, copied

    for source in render["rows"]:
        case, voice = source["case_id"], source["voice_id"]
        wav = qa.RENDER.parent / source["final_path"]
        qa.require(qa.digest(wav) == source["final_sha256"], "frozen WAV changed")
        with wave.open(str(wav), "rb") as stream:
            qa.require((stream.getnchannels(), stream.getframerate(),
                        stream.getsampwidth(), stream.getcomptype()) ==
                       (1, 16000, 2, "NONE"), "WAV format differs")
            frame_count = stream.getnframes()
            pcm = stream.readframes(frame_count)
        zero_tail = 0
        for pos in range(len(pcm) - 2, -1, -2):
            if pcm[pos:pos + 2] != b"\0\0":
                break
            zero_tail += 1
        qa.require(frame_count == source["padding"]["final_frames"] and
                   zero_tail == source["padding"]["appended_zero_frames"] == 80000,
                   "WAV frames or exact zero tail differ")
        pcm_hash = hashlib.sha256(pcm).hexdigest()
        opaque, observations = _asr(comparison, case, voice, source["final_sha256"],
                                    pcm_hash)
        primary, primary_desc = listener(primary_id, opaque)
        secondary_required = _secondary(observations, primary, case)
        secondary = secondary_desc = None
        if secondary_required:
            secondary, secondary_desc = listener(secondary_id, opaque)
        tertiary_required = _tertiary(primary, secondary)
        tertiary = tertiary_desc = None
        if tertiary_required:
            qa.require(tertiary_id is not None, "disagreement needs third reviewer")
            tertiary, tertiary_desc = listener(tertiary_id, opaque)
        listeners = [primary] + ([secondary] if secondary else []) + ([tertiary] if tertiary else [])
        valid = [item for item in listeners if
                 item["blind"]["confidence"] == "high" and
                 item["blind"]["clear_speech"] and
                 not item["blind"]["uncertain_spans"] and
                 item["aware"]["eligibility_judgment"] == "eligible" and
                 qa._case_fields_ok(case, item["aware"])]
        required = 2 if len(listeners) > 1 else 1
        eligible = len(valid) >= required
        rationale = (f"Derived from saved human records: {len(valid)}/{len(listeners)} "
                     f"reviewers met the frozen rule; required votes={required}; "
                     f"third review={tertiary_required}. Rule evaluation is not a C grant.")
        receipt = {
            "schema": "voicechat_wav_qa_receipt_v2",
            "case_id": case, "voice_id": voice,
            "wav_path": str(wav.resolve()), "wav_sha256": source["final_sha256"],
            "pcm_int16_sha256": pcm_hash, "final_sample_count": frame_count,
            "final_exact_zero_tail_frames": zero_tail,
            "script": source["script"],
            "script_sha256": hashlib.sha256(source["script"].encode()).hexdigest(),
            "asr_observations": observations,
            "primary": primary_desc,
            "secondary_required": secondary_required, "secondary": secondary_desc,
            "tertiary_required": tertiary_required, "tertiary": tertiary_desc,
            "final_eligibility": "eligible" if eligible else "excluded",
            "adjudication_rationale": rationale,
            "qa_completed_at_utc": completed_at_utc,
            "external_controller_id": controller_id if case == "C" else "not_applicable"}
        planned.append(receipt)
    qa.require(set(staged) == used, "extra, untriggered, or conflicting human records")
    qa.require(len(planned) == 12, "missing QA matrix cells")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    try:
        (temporary / "stages").mkdir()
        (temporary / "receipts").mkdir()
        blind_copy = temporary / "blind_assignment.json"
        shutil.copyfile(bundle / "blind_assignment.json", blind_copy)
        qa.require(qa.digest(blind_copy) == qa.digest(bundle / "blind_assignment.json"),
                   "blind assignment changed while copying")
        for path in staged.values():
            target = temporary / "stages" / path.name
            qa.require(not target.exists(), "listener file copy collision")
            qa.require(not path.is_symlink(), "listener source became symlinked")
            expected = validated_hashes[path]
            shutil.copyfile(path, target)
            qa.require(qa.digest(target) == expected == qa.digest(path),
                       "human stage changed while copying")
        rows, eligibility = [], []
        for receipt in planned:
            path = temporary / "receipts" / f"{receipt['case_id']}__{receipt['voice_id']}.json"
            _write(path, receipt)
            row = {"case_id": receipt["case_id"], "voice_id": receipt["voice_id"],
                   "wav_sha256": receipt["wav_sha256"],
                   "receipt": _descriptor(temporary, path),
                   "final_eligibility": receipt["final_eligibility"]}
            rows.append(row)
            eligibility.append({"case_id": row["case_id"], "voice_id": row["voice_id"],
                                "wav_sha256": row["wav_sha256"],
                                "receipt_sha256": row["receipt"]["sha256"],
                                "final_eligibility": row["final_eligibility"]})
        blind_order = {role: [opaque for opaque in order if opaque in assigned[role]]
                       for role in roles if assigned[role]}
        index = {
            "schema": INDEX_SCHEMA, "completed_at_utc": completed_at_utc,
            "controller_id_nominated_not_granted": controller_id,
            "amendment_commit": "f8f5bb6", "amendment_sha256": qa.PINS["amendment"],
            "render_manifest_sha256": qa.PINS["render_manifest"],
            "scripts_sha256": qa.PINS["scripts"],
            "comparison_sha256": qa.PINS["comparison"],
            "asr_sources": comparison["sources"], "reviewer_roster": roster,
            "blind_order_seed": assignment["blind_order_seed"], "blind_order": blind_order,
            "blind_assignment": _descriptor(temporary, blind_copy),
            "rows": rows, "eligibility": eligibility}
        _write(temporary / "prepared_index.json", index)
        qa.require(not out.exists(), "output appeared during preparation")
        os.replace(temporary, out)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return {
        "prepared_dir": str(out), "prepared_index_sha256": qa.digest(out / "prepared_index.json"),
        "eligible": sum(row["final_eligibility"] == "eligible" for row in eligibility),
        "denominator": 12, "secondary_reviews": len(assigned[secondary_id]),
        "tertiary_reviews": len(assigned[tertiary_id]) if tertiary_id else 0,
        "controller_grants_created": 0, "external_anchor_created": False}


def _verify_prepared(base: Path, index: dict) -> None:
    """Recheck every copied human stage and derived receipt before finalization."""
    for name, path in (("render_manifest", qa.RENDER), ("scripts", qa.SCRIPTS),
                       ("comparison", qa.COMPARISON), ("amendment", qa.AMENDMENT)):
        qa.require(qa.digest(path) == qa.PINS[name], f"prepared {name} pin differs")
    qa.require(index.get("amendment_commit") == "f8f5bb6" and
               index.get("amendment_sha256") == qa.PINS["amendment"] and
               index.get("scripts_sha256") == qa.PINS["scripts"] and
               index.get("comparison_sha256") == qa.PINS["comparison"] and
               index.get("render_manifest_sha256") == qa.PINS["render_manifest"],
               "prepared source pins differ")
    render = json.loads(qa.RENDER.read_bytes())
    comparison = json.loads(qa.COMPARISON.read_bytes())
    qa.require(qa.digest(qa.RENDER_META) == render["meta_sha256"] and
               index.get("asr_sources") == comparison["sources"],
               "prepared render/ASR source differs")
    assignment_path = qa.local_file(base, index["blind_assignment"])
    assignment, _ = qa.canonical_record(assignment_path, "voicechat_blind_bundle_v2")
    order = [row["opaque_id"] for row in comparison["rows"]["whisper"]]
    random.Random(index["blind_order_seed"]).shuffle(order)
    raw = [json.loads(line) for line in
           (qa.ASR_DIRS["whisper"] / "raw.jsonl").read_text().splitlines()]
    wav_hash_by_opaque = {row["opaque_id"]: row["wav_sha256"] for row in raw}
    qa.require(len(order) == len(set(order)) == len(wav_hash_by_opaque) == 12 and
               assignment.get("blind_order_seed") == index["blind_order_seed"] and
               assignment.get("render_manifest_sha256") == qa.PINS["render_manifest"] and
               assignment.get("comparison_sha256") == qa.PINS["comparison"] and
               assignment.get("opaque_order") ==
               [{"opaque_id": opaque, "wav_sha256": wav_hash_by_opaque[opaque]}
                for opaque in order], "prepared blind assignment differs")
    completed = qa.utc(index["completed_at_utc"])
    assignment_time = qa.utc(assignment["created_utc"])
    qa.require(qa.utc(json.loads(qa.RENDER_META.read_bytes())["started_utc"]) <
               assignment_time < completed <= datetime.now(timezone.utc),
               "prepared chronology differs")
    roster = index["reviewer_roster"]
    qa.require(isinstance(roster, dict) and len(roster) >= 2 and
               all(isinstance(key, str) and key == value for key, value in roster.items()),
               "prepared reviewer roster differs")
    staged = _stages(base / "stages", set(order), set(roster))
    used: set[tuple[str, str, str]] = set()
    assigned = {role: set() for role in roster}
    expected_eligibility = []
    qa.require(len(index["rows"]) == len(index["eligibility"]) == 12,
               "prepared matrix is incomplete")
    for row, source in zip(index["rows"], render["rows"]):
        case, voice = source["case_id"], source["voice_id"]
        qa.require((row["case_id"], row["voice_id"], row["wav_sha256"]) ==
                   (case, voice, source["final_sha256"]), "prepared row order/identity differs")
        receipt_path = qa.local_file(base, row["receipt"])
        receipt, receipt_sha = qa.canonical_record(receipt_path, "voicechat_wav_qa_receipt_v2")
        wav = qa.RENDER.parent / source["final_path"]
        qa.require(qa.digest(wav) == source["final_sha256"], "prepared WAV changed")
        with wave.open(str(wav), "rb") as stream:
            qa.require((stream.getnchannels(), stream.getframerate(),
                        stream.getsampwidth(), stream.getcomptype()) ==
                       (1, 16000, 2, "NONE"), "prepared WAV format differs")
            frames = stream.getnframes()
            pcm = stream.readframes(frames)
        zero_tail = 0
        for pos in range(len(pcm) - 2, -1, -2):
            if pcm[pos:pos + 2] != b"\0\0":
                break
            zero_tail += 1
        qa.require(frames == source["padding"]["final_frames"] and
                   zero_tail == source["padding"]["appended_zero_frames"] == 80000,
                   "prepared WAV frames or zero tail differ")
        pcm_sha = hashlib.sha256(pcm).hexdigest()
        opaque, observations = _asr(comparison, case, voice, source["final_sha256"], pcm_sha)
        qa.require(receipt_sha == row["receipt"]["sha256"] and
                   receipt["case_id"] == case and receipt["voice_id"] == voice and
                   receipt["wav_sha256"] == source["final_sha256"] and
                   receipt["wav_path"] == str(wav.resolve()) and
                   receipt["pcm_int16_sha256"] == pcm_sha and
                   receipt["final_sample_count"] == frames and
                   receipt["final_exact_zero_tail_frames"] == zero_tail and
                   receipt["script"] == source["script"] and
                   receipt["script_sha256"] ==
                   hashlib.sha256(source["script"].encode()).hexdigest() and
                   receipt["asr_observations"] == observations and
                   receipt["qa_completed_at_utc"] == index["completed_at_utc"] and
                   receipt["external_controller_id"] ==
                   (index["controller_id_nominated_not_granted"] if case == "C"
                    else "not_applicable"), "prepared receipt derivation differs")

        def checked_listener(descriptor: dict) -> dict:
            role = descriptor["reviewer_audit_id"]
            for stage in ("blind", "script_aware"):
                expected_path = f"stages/{role}__{opaque}__{stage}.json"
                qa.require(descriptor[stage]["path"] == expected_path,
                           "prepared stage path differs")
                used.add((role, opaque, stage))
            value = qa._listener(base, descriptor, opaque_id=opaque, roster=roster,
                                 before=completed, after=assignment_time)
            assigned[role].add(opaque)
            return value

        primary = checked_listener(receipt["primary"])
        second_required = _secondary(observations, primary, case)
        qa.require(receipt["secondary_required"] is second_required,
                   "prepared secondary trigger differs")
        secondary = checked_listener(receipt["secondary"]) if second_required else None
        qa.require(second_required or receipt["secondary"] is None,
                   "untriggered secondary stage in receipt")
        if secondary:
            qa.require(secondary["id"] != primary["id"], "secondary reviewer reused")
        third_required = _tertiary(primary, secondary)
        qa.require(receipt["tertiary_required"] is third_required,
                   "prepared tertiary trigger differs")
        tertiary = checked_listener(receipt["tertiary"]) if third_required else None
        qa.require(third_required or receipt["tertiary"] is None,
                   "untriggered tertiary stage in receipt")
        if tertiary:
            qa.require(tertiary["id"] not in {primary["id"], secondary["id"]},
                       "tertiary reviewer reused")
        listeners = [primary] + ([secondary] if secondary else []) + ([tertiary] if tertiary else [])
        valid = [item for item in listeners if
                 item["blind"]["confidence"] == "high" and
                 item["blind"]["clear_speech"] and
                 not item["blind"]["uncertain_spans"] and
                 item["aware"]["eligibility_judgment"] == "eligible" and
                 qa._case_fields_ok(case, item["aware"])]
        required = 2 if len(listeners) > 1 else 1
        eligibility = "eligible" if len(valid) >= required else "excluded"
        rationale = (f"Derived from saved human records: {len(valid)}/{len(listeners)} "
                     f"reviewers met the frozen rule; required votes={required}; "
                     f"third review={third_required}. Rule evaluation is not a C grant.")
        qa.require(receipt["final_eligibility"] == row["final_eligibility"] == eligibility and
                   receipt["adjudication_rationale"] == rationale,
                   "prepared eligibility/rationale differs")
        expected_eligibility.append({"case_id": case, "voice_id": voice,
                                     "wav_sha256": row["wav_sha256"],
                                     "receipt_sha256": receipt_sha,
                                     "final_eligibility": eligibility})
    qa.require(set(staged) == used and index["eligibility"] == expected_eligibility and
               index["blind_order"] ==
               {role: [opaque for opaque in order if opaque in assigned[role]]
                for role in roster if assigned[role]},
               "prepared human stage set or eligibility index differs")


def finalize_manifest(prepared_dir: Path, grant_dir: Path, *,
                      controller_key: bytes, trusted_controller_id: str) -> dict:
    """Require two independently signed grants and write only the QA manifest."""
    qa.require(prepared_dir.is_dir() and not prepared_dir.is_symlink() and
               not (prepared_dir / "manifest.json").exists() and
               not (prepared_dir / "grants").exists() and
               not (prepared_dir / "grants").is_symlink(),
               "prepared directory missing or final evidence already exists")
    index, _ = qa.canonical_record(prepared_dir / "prepared_index.json", INDEX_SCHEMA)
    _verify_prepared(prepared_dir, index)
    qa.require(index.get("controller_id_nominated_not_granted") == trusted_controller_id,
               "controller audit ID differs")
    rows = index["rows"]
    qa.require(len(rows) == 12 and len({(r["case_id"], r["voice_id"]) for r in rows}) == 12 and
               all(row["final_eligibility"] == "eligible" for row in rows),
               "full fixture requires twelve human-eligible WAVs")
    qa.require(grant_dir.is_dir() and not grant_dir.is_symlink() and
               not grant_dir.resolve().is_relative_to(prepared_dir.resolve()),
               "external grant directory invalid")
    files = sorted(grant_dir.iterdir())
    qa.require(len(files) == 2 and all(path.is_file() and not path.is_symlink() and
                                      path.suffix == ".json" for path in files),
               "exactly two external C grant files required")
    c_rows = [row for row in rows if row["case_id"] == "C"]
    qa.require(len(c_rows) == 2, "C receipt matrix differs")
    grants = {}
    grant_ids: set[str] = set()
    for path in files:
        value = json.loads(path.read_bytes())
        matching = [row for row in c_rows
                    if row["wav_sha256"] == value.get("source_wav_sha256") and
                    row["receipt"]["sha256"] == value.get("qa_receipt_sha256")]
        qa.require(len(matching) == 1, "grant does not bind exactly one C receipt")
        row = matching[0]
        grant_id = value.get("grant_id")
        qa.require(isinstance(grant_id, str) and grant_id and
                   grant_id not in grant_ids,
                   "C grant ID reused or missing")
        grant_ids.add(grant_id)
        qa.require(row["voice_id"] not in grants and
                   verify_external_grant(value, case_id="C",
                                         source_wav_sha256=row["wav_sha256"],
                                         controller_key=controller_key,
                                         trusted_controller_id=trusted_controller_id),
                   "external grant signature or identity invalid")
        receipt, _ = qa.canonical_record(
            qa.local_file(prepared_dir, row["receipt"]), "voicechat_wav_qa_receipt_v2")
        qa.require(receipt["external_controller_id"] == trusted_controller_id,
                   "C receipt controller ID differs")
        grants[row["voice_id"]] = path
    qa.require(len(grants) == 2, "external C grants incomplete")
    temporary = Path(tempfile.mkdtemp(prefix=".qa_manifest_staging_", dir=prepared_dir))
    installed_grants = False
    created_manifest = False
    final_grants = prepared_dir / "grants"
    final_manifest = prepared_dir / "manifest.json"
    try:
        (temporary / "grants").mkdir()
        descriptors = []
        for row in c_rows:
            source = grants[row["voice_id"]]
            target = temporary / "grants" / source.name
            expected = qa.digest(source)
            shutil.copyfile(source, target)
            qa.require(qa.digest(target) == expected == qa.digest(source),
                       "external grant changed while copying")
            descriptors.append({"voice_id": row["voice_id"],
                                "receipt_sha256": row["receipt"]["sha256"],
                                "grant": _descriptor(temporary, target)})
        manifest = {key: index[key] for key in (
            "amendment_commit", "amendment_sha256", "render_manifest_sha256",
            "scripts_sha256", "comparison_sha256", "asr_sources", "reviewer_roster",
            "blind_order_seed", "blind_order", "blind_assignment", "rows", "eligibility")}
        manifest["schema"] = "voicechat_audio_qa_manifest_v2"
        manifest["c_grants"] = descriptors
        manifest_bytes = qa.canonical_bytes(manifest)
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        qa.require(not final_grants.exists() and not final_manifest.exists(),
                   "final QA evidence appeared during staging")
        os.replace(temporary / "grants", final_grants)
        installed_grants = True
        with final_manifest.open("xb") as stream:
            created_manifest = True
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if created_manifest:
            final_manifest.unlink(missing_ok=True)
        if installed_grants:
            shutil.rmtree(final_grants)
        raise
    finally:
        shutil.rmtree(temporary)
    return {"manifest_path": str(final_manifest),
            "manifest_sha256": manifest_hash,
            "controller_grants_copied": 2, "external_anchor_created": False,
            "ledger_created": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("receipts")
    for option in ("bundle", "records-root", "out"):
        prep.add_argument(f"--{option}", type=Path, required=True)
    for option in ("primary-id", "secondary-id", "completed-at-utc", "controller-id"):
        prep.add_argument(f"--{option}", required=True)
    prep.add_argument("--tertiary-id")
    final = commands.add_parser("manifest")
    final.add_argument("--prepared-dir", type=Path, required=True)
    final.add_argument("--grant-dir", type=Path, required=True)
    final.add_argument("--trusted-controller-id", required=True)
    args = parser.parse_args()
    if args.command == "receipts":
        result = build_receipts(args.bundle, args.records_root, args.out,
                                primary_id=args.primary_id, secondary_id=args.secondary_id,
                                tertiary_id=args.tertiary_id,
                                completed_at_utc=args.completed_at_utc,
                                controller_id=args.controller_id)
    else:
        hex_key = os.environ.get("VOICECHAT_CONTROLLER_KEY_HEX", "")
        qa.require(bool(re.fullmatch(r"[0-9a-fA-F]{64,}", hex_key)) and
                   len(hex_key) % 2 == 0, "external controller key environment missing")
        result = finalize_manifest(args.prepared_dir, args.grant_dir,
                                   controller_key=bytes.fromhex(hex_key),
                                   trusted_controller_id=args.trusted_controller_id)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
