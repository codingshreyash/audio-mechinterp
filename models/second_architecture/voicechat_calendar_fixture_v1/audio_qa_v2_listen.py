"""Prepare opaque-ID WAVs and collect real, staged human listening records.

Run ``prepare`` before assigning reviewers. A reviewer runs ``listen`` with an
opaque ID list supplied by the organizer. The blind record is saved before the
script is printed. This interface never shows ASR or model output. Process and
source access cannot prove a reviewer remained blind; audit that externally.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .audio_qa_v2 import (ASR_DIRS, COMPARISON, HERE, RENDER, PINS,
                          canonical_bytes, digest, require)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sources() -> dict[str, dict]:
    for source, pin in ((RENDER, PINS["render_manifest"]),
                        (COMPARISON, PINS["comparison"])):
        require(digest(source) == pin, "pinned blinded source changed")
    render = json.loads(RENDER.read_bytes())
    comparison = json.loads(COMPARISON.read_bytes())
    by_key = {(r["case_id"], r["voice_id"]): r for r in render["rows"]}
    by_opaque: dict[str, dict] = {}
    for row in comparison["rows"]["whisper"]:
        key = (row["case_id"], row["voice_id"])
        source = by_key[key]
        opaque = row["opaque_id"]
        require(opaque not in by_opaque, "duplicate opaque ID")
        by_opaque[opaque] = source
    require(len(by_opaque) == 12, "blind set is not 12 WAVs")
    return by_opaque


def prepare(out: Path, *, seed: str) -> dict:
    require(bool(seed), "blind-order seed is required")
    sources = _sources()
    out.mkdir(parents=True, exist_ok=False)
    ordered = list(sources)
    random.Random(seed).shuffle(ordered)
    wavs = []
    for opaque in ordered:
        source = sources[opaque]
        original = RENDER.parent / source["final_path"]
        require(digest(original) == source["final_sha256"], "rendered WAV changed")
        target = out / f"{opaque}.wav"
        shutil.copyfile(original, target)
        require(digest(target) == source["final_sha256"], "blind copy differs")
        wavs.append({"opaque_id": opaque, "wav_sha256": source["final_sha256"]})
    assignment = {"schema": "voicechat_blind_bundle_v2",
                  "created_utc": timestamp(), "blind_order_seed": seed,
                  "render_manifest_sha256": PINS["render_manifest"],
                  "comparison_sha256": PINS["comparison"],
                  "opaque_order": wavs}
    with (out / "blind_assignment.json").open("xb") as stream:
        stream.write(canonical_bytes(assignment))
    return {"bundle": str(out), "count": len(wavs),
            "assignment_sha256": digest(out / "blind_assignment.json")}


def _answer(prompt: str) -> str:
    value = input(prompt).strip()
    require(bool(value), "blank response; listening record was not completed")
    return value


def _choice(prompt: str, choices: tuple[str, ...]) -> str:
    value = _answer(f"{prompt} [{'/'.join(choices)}]: ")
    require(value in choices, f"expected one of {choices}")
    return value


def _bool(prompt: str) -> bool:
    return _choice(prompt, ("yes", "no")) == "yes"


def _required_field(prompt: str, *, applicable: bool) -> str | bool:
    return _answer(prompt + ": ") if applicable else "not_applicable"


def listen(bundle: Path, records: Path, *, reviewer_id: str,
           opaque_ids: list[str], play: bool = False) -> list[dict]:
    require(sys.stdin.isatty(), "human listening requires an interactive terminal")
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]{3,80}", reviewer_id)),
            "reviewer audit ID must use simple safe characters")
    assignment = json.loads((bundle / "blind_assignment.json").read_bytes())
    require(assignment.get("schema") == "voicechat_blind_bundle_v2" and
            assignment.get("render_manifest_sha256") == PINS["render_manifest"] and
            assignment.get("comparison_sha256") == PINS["comparison"],
            "blind assignment differs from frozen sources")
    allowed = {row["opaque_id"]: row["wav_sha256"] for row in assignment["opaque_order"]}
    require(len(allowed) == 12 and len(opaque_ids) == len(set(opaque_ids)) and
            all(opaque in allowed for opaque in opaque_ids), "invalid reviewer opaque-ID list")
    sources = _sources()
    require(all(allowed[opaque] == sources[opaque]["final_sha256"] for opaque in allowed),
            "blind assignment WAV hashes do not match frozen render")
    records.mkdir(parents=True, exist_ok=True)
    outputs = []
    for opaque in opaque_ids:
        wav = bundle / f"{opaque}.wav"
        require(digest(wav) == allowed[opaque], "blinded WAV changed")
        blind_path = records / f"{reviewer_id}__{opaque}__blind.json"
        aware_path = records / f"{reviewer_id}__{opaque}__script_aware.json"
        require(not blind_path.exists() and not aware_path.exists(),
                "existing stage records cannot be overwritten")
        print(f"\nClip {opaque}. Listen to {wav}.")
        if play:
            player = shutil.which("afplay") or shutil.which("ffplay")
            require(player is not None, "afplay or ffplay is required for --play")
            args = [player, str(wav)] if Path(player).name == "afplay" else \
                [player, "-nodisp", "-autoexit", "-loglevel", "error", str(wav)]
            subprocess.run(args, check=True)
        else:
            input("Press Enter only after listening to the audio file: ")
        transcript = _answer("Before seeing the script, type exactly what you heard: ")
        clear = _bool("Is the speech clear?")
        confidence = _choice("Confidence", ("high", "low", "unclear"))
        spans = input("Uncertain spans with times, separated by semicolons; blank if none: ").strip()
        blind = {"schema": "voicechat_listener_stage_v2", "opaque_id": opaque,
                 "reviewer_audit_id": reviewer_id, "stage": "blind",
                 "recorded_utc": timestamp(), "verbatim_free_transcript": transcript,
                 "clear_speech": clear, "confidence": confidence,
                 "uncertain_spans": [s.strip() for s in spans.split(";") if s.strip()]}
        with blind_path.open("xb") as stream:
            stream.write(canonical_bytes(blind))
        blind_hash = digest(blind_path)
        source = sources[opaque]
        case_id = source["case_id"]
        print("\nFrozen script (revealed after the blind record was saved):")
        print(source["script"])
        print("Record what is AUDIBLE, even if it differs from the script.")
        aware = {"schema": "voicechat_listener_stage_v2", "opaque_id": opaque,
                 "reviewer_audit_id": reviewer_id, "stage": "script_aware",
                 "recorded_utc": timestamp(), "blind_recorded_utc": blind["recorded_utc"],
                 "blind_record_sha256": blind_hash,
                 "clear_speech": _bool("After comparison, is the speech clear?"),
                 "date_heard": _required_field("Date heard (YYYY-MM-DD, 'unclear', or another heard date)",
                                               applicable=case_id != "D"),
                 "design_review_title_heard": _required_field("Event title heard (or 'unclear')",
                                                               applicable=case_id in ("B", "C", "B_literal")),
                 "first_action_heard": _choice("First action requested",
                                                ("list_calendar_events", "cancel_calendar_events", "none", "unclear")),
                 "cancel_requires_further_confirmation_heard":
                     _bool("Is listing first and asking before cancellation audible?")
                     if case_id in ("B", "B_literal") else "not_applicable",
                 "c_confirmation_phrase_heard_exactly":
                     _bool("Is 'Confirm cancel Design Review' heard exactly?")
                     if case_id == "C" else "not_applicable",
                 "cancellation_intent_heard": _choice("Cancellation intent heard",
                                                       ("none", "possible", "explicit", "unclear")),
                 "case_semantics_preserved": _bool("Does the audible request preserve the script's intent and restrictions?"),
                 "eligibility_judgment": _choice("Your eligibility judgment", ("eligible", "excluded")),
                 "notes": input("Notes and uncertain time spans, if any: ").strip()}
        # The blind stage has already been committed to disk. A second write is
        # prohibited; the collector cannot silently revise a free transcript.
        with aware_path.open("xb") as stream:
            stream.write(canonical_bytes(aware))
        outputs.append({"opaque_id": opaque, "blind": {"path": str(blind_path),
                                                       "sha256": blind_hash},
                        "script_aware": {"path": str(aware_path),
                                         "sha256": digest(aware_path)}})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", required=True)
    review = sub.add_parser("listen")
    review.add_argument("--bundle", type=Path, required=True)
    review.add_argument("--records", type=Path, required=True)
    review.add_argument("--reviewer-id", required=True)
    selected = review.add_mutually_exclusive_group(required=True)
    selected.add_argument("--all", action="store_true")
    selected.add_argument("--ids", nargs="+", help="opaque IDs assigned for secondary/tertiary review")
    review.add_argument("--play", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.out, seed=args.seed), indent=2))
    else:
        assignment = json.loads((args.bundle / "blind_assignment.json").read_bytes())
        all_ids = [row["opaque_id"] for row in assignment["opaque_order"]]
        print(json.dumps(listen(args.bundle, args.records, reviewer_id=args.reviewer_id,
                                opaque_ids=all_ids if args.all else args.ids, play=args.play), indent=2))


if __name__ == "__main__":
    main()
