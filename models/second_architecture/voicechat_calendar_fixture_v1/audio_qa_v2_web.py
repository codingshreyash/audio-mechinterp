"""Local staged HTML intake for the existing VoiceChat QA v2 listener schema.

This makes browser forms; it does not listen, judge speech, authenticate human
identity, sign a C grant, or authorize VoiceChat inference. Blind pages contain
no scripts or ASR. An aware page is created only after all assigned blind stage
files have been imported and validated. The existing QA auditor is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import audio_qa_v2 as qa
from . import audio_qa_v2_build as build


EXPORT_SCHEMA = "voicechat_web_stage_export_v1"
ROLES = ("primary", "secondary", "tertiary")
ID_RE = re.compile(r"[A-Za-z0-9_-]{3,80}\Z")
BLIND_KEYS = {"schema", "opaque_id", "reviewer_audit_id", "stage",
              "recorded_utc", "verbatim_free_transcript", "clear_speech",
              "confidence", "uncertain_spans"}
AWARE_KEYS = {"schema", "opaque_id", "reviewer_audit_id", "stage",
              "recorded_utc", "blind_recorded_utc", "blind_record_sha256",
              "clear_speech", "date_heard", "design_review_title_heard",
              "first_action_heard", "cancel_requires_further_confirmation_heard",
              "c_confirmation_phrase_heard_exactly", "cancellation_intent_heard",
              "case_semantics_preserved", "eligibility_judgment", "notes"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _roles(primary_id: str, secondary_id: str, tertiary_id: str | None) -> dict[str, str | None]:
    result = {"primary": primary_id, "secondary": secondary_id, "tertiary": tertiary_id}
    ids = [x for x in result.values() if x is not None]
    qa.require(all(ID_RE.fullmatch(x) for x in ids) and len(ids) == len(set(ids)),
               "reviewer audit IDs must be distinct simple strings")
    return result


def _stage_path(records_root: Path, reviewer: str, opaque: str, stage: str) -> Path:
    return records_root / f"{reviewer}__{opaque}__{stage}.json"


def _listener(records_root: Path, reviewer: str, opaque: str,
              assignment_time: datetime) -> dict[str, Any]:
    blind = _stage_path(records_root, reviewer, opaque, "blind")
    aware = _stage_path(records_root, reviewer, opaque, "script_aware")
    qa.require(blind.is_file() and aware.is_file() and
               not blind.is_symlink() and not aware.is_symlink(),
               "required earlier listener stage is absent")
    desc = {"reviewer_audit_id": reviewer,
            "blind": {"path": blind.name, "sha256": qa.digest(blind)},
            "script_aware": {"path": aware.name, "sha256": qa.digest(aware)}}
    return qa._listener(records_root, desc, opaque_id=opaque,
                        roster={reviewer: reviewer},
                        before=_now() + timedelta(seconds=1), after=assignment_time)


def _selection(bundle: Path, records_root: Path, role: str,
               roles: dict[str, str | None]) -> tuple[list[str], dict[str, dict], dict]:
    qa.require(role in ROLES and roles[role] is not None, "reviewer role or ID missing")
    render, comparison, assignment, assignment_time = build._sources(bundle)
    by_opaque = {row["opaque_id"]: row for row in assignment["opaque_order"]}
    order = list(by_opaque)
    if role == "primary":
        return order, by_opaque, assignment
    flagged_secondary: set[str] = set()
    flagged_tertiary: set[str] = set()
    for source in render["rows"]:
        wav = qa.RENDER.parent / source["final_path"]
        qa.require(qa.digest(wav) == source["final_sha256"],
                   "frozen source WAV changed")
        with wave.open(str(wav), "rb") as audio:
            pcm_sha = hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()
        opaque, asr = build._asr(comparison, source["case_id"], source["voice_id"],
                                 source["final_sha256"], pcm_sha)
        primary = _listener(records_root, roles["primary"], opaque, assignment_time)
        if not build._secondary(asr, primary, source["case_id"]):
            continue
        flagged_secondary.add(opaque)
        if role == "tertiary":
            secondary = _listener(records_root, roles["secondary"], opaque,
                                  assignment_time)
            if build._tertiary(primary, secondary):
                flagged_tertiary.add(opaque)
    selected = flagged_secondary if role == "secondary" else flagged_tertiary
    return [opaque for opaque in order if opaque in selected], by_opaque, assignment


def _page(title: str, sections: list[str], javascript: str) -> bytes:
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            "<style>body{font:17px/1.45 system-ui,sans-serif;max-width:850px;"
            "margin:2rem auto;padding:0 1rem;color:#17202a}section{padding:1rem;"
            "border:1px solid #aab4be;border-radius:10px;margin:1rem 0}"
            "label{display:block;margin:.55rem 0}input[type=text],textarea,select{"
            "box-sizing:border-box;width:100%;padding:.5rem;font:inherit}"
            "audio{width:100%}button{font:inherit;padding:.6rem 1rem}"
            "#error{color:#a10000;white-space:pre-wrap}small{color:#425466}"
            "</style><main>"
            f"<h1>{html.escape(title)}</h1>"
            "<p>Use headphones. Listen to each clip before answering. One export "
            "contains all assigned clips. Keep that JSON file unchanged and send it "
            "to the study coordinator.</p>"
            + "".join(sections) +
            "<p id='error' role='alert'></p><button id='export' type='button'>"
            "Download completed JSON</button></main><script>\n" + javascript +
            "\n</script></html>\n").encode("utf-8")


def _audio(opaque: str, ordinal: int) -> str:
    return (f"<section data-opaque='{opaque}'><h2>Clip {ordinal} of assigned set</h2>"
            f"<audio controls preload='none' src='{opaque}.wav'></audio>"
            f"<small>Audio ID: {opaque}</small>")


def _select(name: str, label: str, values: tuple[str, ...]) -> str:
    opts = "<option value=''>Select…</option>" + "".join(
        f"<option value='{html.escape(value)}'>{html.escape(value)}</option>"
        for value in values)
    return f"<label>{html.escape(label)}<select data-field='{name}'>{opts}</select></label>"


def _text(name: str, label: str, *, multiline: bool = False) -> str:
    tag = (f"<textarea data-field='{name}' rows='2'></textarea>" if multiline else
           f"<input data-field='{name}' type='text'>")
    return f"<label>{html.escape(label)}{tag}</label>"


def _blind_section(opaque: str, ordinal: int) -> str:
    return (_audio(opaque, ordinal) +
            _text("verbatim_free_transcript", "Write exactly what you heard", multiline=True) +
            _select("clear_speech", "Is the speech clear?", ("yes", "no")) +
            _select("confidence", "Confidence", ("high", "low", "unclear")) +
            _text("uncertain_spans", "Uncertain spans and times; separate with semicolons") +
            "</section>")


def _aware_section(opaque: str, ordinal: int, source: dict) -> str:
    case = source["case_id"]
    script = html.escape(source["script"])
    answer = (_audio(opaque, ordinal) + f"<p><strong>Frozen script:</strong> {script}</p>"
              "<p>Judge what is audible, even if it differs from the script.</p>"
              + _select("clear_speech", "After comparison, is speech clear?", ("yes", "no")))
    if case != "D":
        answer += _text("date_heard", "Date heard (YYYY-MM-DD, unclear, or another date)")
    if case in ("B", "C", "B_literal"):
        answer += _text("design_review_title_heard", "Event title heard (or unclear)")
    answer += _select("first_action_heard", "First action requested",
                      ("list_calendar_events", "cancel_calendar_events", "none", "unclear"))
    if case in ("B", "B_literal"):
        answer += _select("cancel_requires_further_confirmation_heard",
                          "Is listing first and asking before cancellation audible?", ("yes", "no"))
    if case == "C":
        answer += _select("c_confirmation_phrase_heard_exactly",
                          "Is 'Confirm cancel Design Review' heard exactly?", ("yes", "no"))
    return (answer +
            _select("cancellation_intent_heard", "Cancellation intent heard",
                    ("none", "possible", "explicit", "unclear")) +
            _select("case_semantics_preserved", "Does audible request preserve script intent and restrictions?",
                    ("yes", "no")) +
            _select("eligibility_judgment", "Your eligibility judgment", ("eligible", "excluded")) +
            _text("notes", "Notes and uncertain time spans", multiline=True) +
            "</section>")


def _script(payload: dict, stage: str) -> str:
    # The blind payload contains only opaque IDs and hashes, not scripts/ASR.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).replace("<", "\\u003c")
    common = """const payload=PAYLOAD;
const sections=[...document.querySelectorAll('section[data-opaque]')];
const field=(section,name,required=true)=>{
  const node=section.querySelector(`[data-field="${name}"]`);
  if(!node) throw new Error(`Missing form field ${name}`);
  const value=node.value.trim();
  if(required&&!value) throw new Error(`Complete ${name} for ${section.dataset.opaque}`);
  return value;
};
const yes=(value)=>value==='yes';
const download=(name,value)=>{
  const blob=new Blob([JSON.stringify(value,null,2)+'\\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob);const link=document.createElement('a');
  link.href=url;link.download=name;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.getElementById('export').addEventListener('click',()=>{
  try { const records=sections.map(section=>RECORD(section));
    download(`${payload.reviewer_audit_id}__${payload.stage}_export.json`,
      {schema:'voicechat_web_stage_export_v1',stage:payload.stage,
       role:payload.role,reviewer_audit_id:payload.reviewer_audit_id,
       assignment_sha256:payload.assignment_sha256,
       opaque_order:payload.opaque_order,records});
    document.getElementById('error').textContent='Downloaded. Keep the JSON file unchanged.';
  } catch(error) {document.getElementById('error').textContent=String(error.message||error);}
});"""
    if stage == "blind":
        record = """const RECORD=(section)=>({
  schema:'voicechat_listener_stage_v2',opaque_id:section.dataset.opaque,
  reviewer_audit_id:payload.reviewer_audit_id,stage:'blind',
  recorded_utc:new Date().toISOString(),
  verbatim_free_transcript:field(section,'verbatim_free_transcript'),
  clear_speech:yes(field(section,'clear_speech')),
  confidence:field(section,'confidence'),
  uncertain_spans:field(section,'uncertain_spans',false).split(';').map(x=>x.trim()).filter(Boolean)
});"""
    else:
        record = """const RECORD=(section)=>{
  const id=section.dataset.opaque;const context=payload.context[id];
  const applicable=(name)=>section.querySelector(`[data-field="${name}"]`)!==null;
  const optional=(name)=>applicable(name)?field(section,name):'not_applicable';
  const optionalYes=(name)=>applicable(name)?yes(field(section,name)):'not_applicable';
  return {schema:'voicechat_listener_stage_v2',opaque_id:id,
    reviewer_audit_id:payload.reviewer_audit_id,stage:'script_aware',
    recorded_utc:new Date().toISOString(),
    blind_recorded_utc:context.blind_recorded_utc,
    blind_record_sha256:context.blind_record_sha256,
    clear_speech:yes(field(section,'clear_speech')),
    date_heard:optional('date_heard'),
    design_review_title_heard:optional('design_review_title_heard'),
    first_action_heard:field(section,'first_action_heard'),
    cancel_requires_further_confirmation_heard:optionalYes('cancel_requires_further_confirmation_heard'),
    c_confirmation_phrase_heard_exactly:optionalYes('c_confirmation_phrase_heard_exactly'),
    cancellation_intent_heard:field(section,'cancellation_intent_heard'),
    case_semantics_preserved:yes(field(section,'case_semantics_preserved')),
    eligibility_judgment:field(section,'eligibility_judgment'),
    notes:field(section,'notes',false)};
};"""
    return "const PAYLOAD=" + encoded + ";\n" + record + "\n" + common


def make_blind(bundle: Path, records_root: Path, out_dir: Path, *, role: str,
               roles: dict[str, str | None]) -> dict:
    selected, by_opaque, assignment = _selection(bundle, records_root, role, roles)
    qa.require(bool(selected), f"no {role} review is triggered")
    qa.require(not out_dir.exists(), "reviewer page directory already exists")
    out_dir.mkdir(parents=True)
    try:
        for opaque in selected:
            source = bundle / f"{opaque}.wav"
            target = out_dir / source.name
            shutil.copyfile(source, target)
            qa.require(qa.digest(target) == by_opaque[opaque]["wav_sha256"],
                       "reviewer WAV copy changed")
        reviewer = roles[role]
        payload = {"stage": "blind", "role": role, "reviewer_audit_id": reviewer,
                   "assignment_sha256": qa.digest(bundle / "blind_assignment.json"),
                   "opaque_order": selected}
        sections = [_blind_section(opaque, i) for i, opaque in enumerate(selected, 1)]
        page = out_dir / "BLIND.html"
        page.write_bytes(_page(f"VoiceChat blind listening — {reviewer}", sections,
                               _script(payload, "blind")))
        qa.require(b"Design Review" not in page.read_bytes() and
                   b"November nineteenth" not in page.read_bytes(),
                   "blind form accidentally contains reference text")
        return {"page": str(page), "reviewer_audit_id": reviewer,
                "role": role, "clips": len(selected), "page_sha256": qa.digest(page)}
    except BaseException:
        # A failed page directory is consumed; never silently overwrite it.
        raise


def make_aware(bundle: Path, records_root: Path, out_dir: Path, *, role: str,
               roles: dict[str, str | None]) -> dict:
    selected, by_opaque, assignment = _selection(bundle, records_root, role, roles)
    assignment_time = qa.utc(assignment["created_utc"])
    reviewer = roles[role]
    qa.require(out_dir.is_dir() and not out_dir.is_symlink() and
               (out_dir / "BLIND.html").is_file(), "blind reviewer page missing")
    page = out_dir / "AWARE.html"
    qa.require(not page.exists(), "aware page already exists")
    # Derive the private mapping from pinned comparison and render bytes; do not
    # write it to the blind page or reviewer bundle before stage-one intake.
    render = json.loads(qa.RENDER.read_bytes())
    comparison = json.loads(qa.COMPARISON.read_bytes())
    by_key = {(r["case_id"], r["voice_id"]): r for r in render["rows"]}
    sources = {row["opaque_id"]: by_key[(row["case_id"], row["voice_id"])]
               for row in comparison["rows"]["whisper"]}
    context = {}
    sections = []
    for i, opaque in enumerate(selected, 1):
        blind = _stage_path(records_root, reviewer, opaque, "blind")
        qa.require(blind.is_file() and not blind.is_symlink(),
                   "all blind stages must be imported before script reveal")
        value, blind_sha = qa._stage(records_root,
                                     {"path": blind.name, "sha256": qa.digest(blind)},
                                     opaque_id=opaque, reviewer=reviewer,
                                     stage="blind", before=_now() + timedelta(seconds=1),
                                     after=assignment_time)
        qa.require(qa.digest(out_dir / f"{opaque}.wav") == by_opaque[opaque]["wav_sha256"],
                   "aware page WAV differs from blind WAV")
        context[opaque] = {"blind_recorded_utc": value["recorded_utc"],
                           "blind_record_sha256": blind_sha}
        sections.append(_aware_section(opaque, i, sources[opaque]))
    payload = {"stage": "script_aware", "role": role,
               "reviewer_audit_id": reviewer,
               "assignment_sha256": qa.digest(bundle / "blind_assignment.json"),
               "opaque_order": selected, "context": context}
    page.write_bytes(_page(f"VoiceChat script comparison — {reviewer}", sections,
                           _script(payload, "script_aware")))
    return {"page": str(page), "reviewer_audit_id": reviewer,
            "role": role, "clips": len(selected), "page_sha256": qa.digest(page)}


def import_stage(bundle: Path, records_root: Path, export: Path, *, stage: str,
                 role: str, roles: dict[str, str | None]) -> dict:
    qa.require(stage in ("blind", "script_aware"), "unsupported web stage")
    selected, _, assignment = _selection(bundle, records_root, role, roles)
    reviewer = roles[role]
    data = json.loads(export.read_bytes())
    qa.require(isinstance(data, dict) and data.get("schema") == EXPORT_SCHEMA and
               data.get("stage") == stage and data.get("role") == role and
               data.get("reviewer_audit_id") == reviewer and
               data.get("assignment_sha256") == qa.digest(bundle / "blind_assignment.json") and
               data.get("opaque_order") == selected and
               isinstance(data.get("records"), list) and
               len(data["records"]) == len(selected), "web export identity or census changed")
    qa.require([r.get("opaque_id") for r in data["records"]] == selected,
               "web export order or opaque IDs changed")
    records_root.mkdir(parents=True, exist_ok=True)
    qa.require(not records_root.is_symlink(), "records root is linked")
    assignment_time = qa.utc(assignment["created_utc"])
    before = _now() + timedelta(seconds=1)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="voicechat-web-intake-", dir=records_root) as temp:
        temporary = Path(temp)
        for value in data["records"]:
            opaque = value["opaque_id"]
            target = _stage_path(records_root, reviewer, opaque, stage)
            qa.require(not target.exists(), "listener stage already exists; never overwrite")
            qa.require(isinstance(value, dict) and set(value) ==
                       (BLIND_KEYS if stage == "blind" else AWARE_KEYS) and
                       value.get("schema") == "voicechat_listener_stage_v2" and
                       value.get("stage") == stage and
                       value.get("reviewer_audit_id") == reviewer,
                       "web stage fields or reviewer identity changed")
            if stage == "blind":
                blind_hash = None
            else:
                blind = _stage_path(records_root, reviewer, opaque, "blind")
                qa.require(blind.is_file() and not blind.is_symlink(),
                           "aware stage has no saved blind stage")
                original, blind_hash = qa.canonical_record(blind, "voicechat_listener_stage_v2")
                qa.require(value["blind_recorded_utc"] == original["recorded_utc"],
                           "aware stage blind timestamp changed")
            candidate = temporary / target.name
            candidate.write_bytes(qa.canonical_bytes(value))
            qa._stage(temporary, {"path": candidate.name,
                                  "sha256": qa.digest(candidate)},
                      opaque_id=opaque, reviewer=reviewer, stage=stage,
                      before=before, after=assignment_time,
                      blind_hash=blind_hash)
            outputs.append((candidate, target))
        committed = []
        try:
            for candidate, target in outputs:
                os.link(candidate, target)
                committed.append(target)
        except BaseException:
            for target in committed:
                target.unlink()
            raise
    return {"stage": stage, "role": role, "reviewer_audit_id": reviewer,
            "imported": len(outputs), "records_root": str(records_root),
            "files": [{"path": str(target), "sha256": qa.digest(target)}
                      for _, target in outputs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("make-blind", "import-blind",
                                            "make-aware", "import-aware", "plan"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--primary-id", required=True)
    parser.add_argument("--secondary-id", required=True)
    parser.add_argument("--tertiary-id")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    roles = _roles(args.primary_id, args.secondary_id, args.tertiary_id)
    if args.command == "plan":
        selected, _, _ = _selection(args.bundle, args.records_root, args.role, roles)
        result = {"role": args.role, "reviewer_audit_id": roles[args.role],
                  "clips": len(selected), "opaque_ids": selected}
    elif args.command.startswith("make-"):
        qa.require(args.out_dir is not None, "--out-dir is required")
        maker = make_blind if args.command == "make-blind" else make_aware
        result = maker(args.bundle, args.records_root, args.out_dir,
                       role=args.role, roles=roles)
    else:
        qa.require(args.export is not None, "--export is required")
        result = import_stage(args.bundle, args.records_root, args.export,
                              stage="blind" if args.command == "import-blind"
                              else "script_aware", role=args.role, roles=roles)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
