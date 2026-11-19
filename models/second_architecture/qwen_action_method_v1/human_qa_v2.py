"""V2 fail-closed, no-model intake for Qwen action-label blind listening.

This module checks exported human records. It never makes a human judgment and
never grants candidate inference. It preserves V1 and its independent HOLD.
Synthetic reviews are made only in unit tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import listening_bundle_v2 as bundle
from . import render_sources_v2 as source
from .endpoint import load_spec

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/source_render_v2/attempt_001"
BUNDLE = HERE / "runs/source_render_v2/review_bundle_001"
THIRD = HERE / "runs/source_render_v2/third_review_001"
INTENTS = {"LIST": "list_only", "ASK": "ask_before_change",
           "CANCEL": "confirm_now"}
MEANINGS = set(INTENTS.values()) | {"unclear"}
REVIEWER_ID = re.compile(r"[A-Za-z0-9_.-]{3,64}\Z")
TOKENS = re.compile(r"[a-z0-9]+")
ROW_KEYS = {"ordinal", "file", "transcript", "meaning", "date", "event",
            "phrase_start", "phrase_end", "notes"}
EXPORT_KEYS = {"schema", "queue", "bundle_binding", "reviewer", "rows",
               "locked_utc"}
DATE_FORMS = {
    "november nineteenth twenty twenty six", "november nineteen twenty twenty six",
    "november 19 2026", "november 19th 2026", "11 19 2026",
}
CLOCK_SKEW_SECONDS = 300
ATTESTATION_KEYS = {"schema", "bundle_mapping_sha256", "first_export_sha256",
                    "reviewer_ids", "coordinator_id", "independent_people",
                    "separate_blind_folders", "no_shared_answers_before_both_locks",
                    "signed_utc", "notes"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "review JSON missing or linked")
    require(path.stat().st_size <= 2 * 1024 * 1024, "review JSON exceeds size cap")
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(
                          ValueError(f"nonfinite JSON constant: {x}")))
    require(isinstance(data, dict), "review JSON must be an object")
    return data


def sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), "file missing or linked")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: str) -> str:
    return " ".join(TOKENS.findall(value.casefold()))


def timestamp(value: Any) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"),
            "review lock time must be UTC Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("review lock time invalid") from error
    require(parsed.utcoffset() == timezone.utc.utcoffset(parsed),
            "review lock time not UTC")
    require(parsed <= now_utc() + timedelta(seconds=CLOCK_SKEW_SECONDS),
            "review timestamp exceeds frozen five-minute future-skew cap")
    return parsed


def _time_number(value: Any) -> float:
    require(isinstance(value, str), "phrase time must be form string")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError("phrase time invalid") from error
    require(math.isfinite(result), "phrase time nonfinite")
    return result


def validate_export(document: dict[str, Any], queue: str,
                    mapping: dict[str, Any], *, expected_rows: list[dict] | None = None,
                    expected_binding: str | None = None) -> tuple[datetime, str]:
    """Validate a first, blind export without consulting reference scripts."""
    require(set(document) == EXPORT_KEYS
            and document["schema"] == "qwen_action_blind_listening_v1"
            and document["queue"] == queue
            and document["bundle_binding"] == (
                mapping["bundle_binding"] if expected_binding is None else expected_binding),
            "review schema/queue/binding differs")
    reviewer = document["reviewer"]
    require(isinstance(reviewer, str) and REVIEWER_ID.fullmatch(reviewer) is not None,
            "reviewer ID invalid")
    locked = timestamp(document["locked_utc"])
    rows = document["rows"]
    expected = mapping["queues"][queue] if expected_rows is None else expected_rows
    require(isinstance(rows, list) and len(rows) == len(expected),
            "review queue incomplete")
    for given, planned in zip(rows, expected, strict=True):
        require(isinstance(given, dict) and set(given) == ROW_KEYS
                and type(given["ordinal"]) is int
                and given["ordinal"] == planned["ordinal"]
                and given["file"] == planned["opaque_file"],
                "review row identity/order differs")
        require(all(isinstance(given[key], str) and given[key].strip()
                    for key in ROW_KEYS - {"ordinal"}),
                "review row answer incomplete")
        require(given["meaning"] in MEANINGS, "review meaning invalid")
        start, end = (_time_number(given["phrase_start"]),
                      _time_number(given["phrase_end"]))
        require(0 <= start < end <= 16, "review phrase timing invalid")
    return locked, reviewer


def _expected_by_case() -> dict[str, dict[str, Any]]:
    spec = load_spec()
    scripts = {row["id"]: row for row in spec["scripts"]}
    return {row["case_id"]: {"script": scripts[row["script_id"]]["text"],
                             "label": row["label"],
                             "source_frames": row["source_frames"]}
            for row in read_json(SOURCE / "manifest.json")["rows"]}


def _critical(record: dict[str, Any]) -> tuple[str, str, str]:
    heard_date = norm(record["date"])
    return (record["meaning"],
            "2026-11-19" if heard_date in DATE_FORMS else heard_date,
            norm(record["event"]))


def _row_pass(record: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if norm(record["transcript"]) != norm(expected["script"]):
        failures.append("verbatim_words")
    if record["meaning"] != INTENTS[expected["label"]]:
        failures.append("critical_intent")
    if norm(record["date"]) not in DATE_FORMS:
        failures.append("date")
    wanted_event = "none" if expected["label"] == "LIST" else "design review"
    if norm(record["event"]) != wanted_event:
        failures.append("event")
    if norm(record["notes"]) != "none":
        failures.append("unclear_words")
    # Timestamps remain model-blind human annotations. A later source-to-row
    # review must validate them before a local time-window intervention.
    return not failures, failures


def assess_first_pair(first_a: dict, first_b: dict, mapping: dict,
                      expected: dict[str, dict]) -> dict[str, Any]:
    """Only reveal the frozen references after both full blind exports validate."""
    time_a, reviewer_a = validate_export(first_a, "A", mapping)
    time_b, reviewer_b = validate_export(first_b, "B", mapping)
    require(reviewer_a != reviewer_b, "two independent reviewer IDs required")
    by_case = {name: {planned["case_id"]: row for row, planned in zip(
        first["rows"], mapping["queues"][name], strict=True)}
        for name, first in (("A", first_a), ("B", first_b))}
    require(set(by_case["A"]) == set(by_case["B"]) == set(expected),
            "review source population differs")
    results = []
    third = []
    for case_id in expected:
        a, b = by_case["A"][case_id], by_case["B"][case_id]
        a_pass, a_reasons = _row_pass(a, expected[case_id])
        b_pass, b_reasons = _row_pass(b, expected[case_id])
        disagreement = (_critical(a) != _critical(b)
                        or norm(a["transcript"]) != norm(b["transcript"]))
        if disagreement:
            third.append(case_id)
        results.append({"case_id": case_id,
                        "reviewer_A_pass": a_pass, "reviewer_B_pass": b_pass,
                        "reviewer_A_failures": a_reasons,
                        "reviewer_B_failures": b_reasons,
                        "critical_disagreement": disagreement,
                        "initial_pair_pass": a_pass and b_pass})
    return {"schema": "qwen_action_human_qa_first_pair_v1",
            "first_export_locks_utc": {"A": first_a["locked_utc"],
                                       "B": first_b["locked_utc"]},
            "first_export_times": (time_a, time_b),
            "reviewer_ids": {"A": reviewer_a, "B": reviewer_b},
            "cases": results, "third_required_case_ids": third,
            "first_pair_all_pass": all(row["initial_pair_pass"] for row in results)}


def _load_bound(first_a_path: Path, first_b_path: Path) -> tuple[dict, dict, dict, dict]:
    bundle.audit_bundle(SOURCE, BUNDLE)
    mapping = read_json(BUNDLE / "TECHNICAL_MAPPING.json")
    first_a, first_b = read_json(first_a_path), read_json(first_b_path)
    time_a, _ = validate_export(first_a, "A", mapping)
    time_b, _ = validate_export(first_b, "B", mapping)
    terminal = read_json(SOURCE / "terminal.json")
    render_end = datetime.fromisoformat(terminal["terminal_utc_after_manifest"])
    require(time_a >= render_end and time_b >= render_end,
            "human review predates its source audio")
    return mapping, first_a, first_b, _expected_by_case()


def attestation_template(first_a_path: Path, first_b_path: Path) -> dict[str, Any]:
    """Bind an empty coordinator statement to actual first exports, without signing it."""
    mapping, a, b, expected = _load_bound(first_a_path, first_b_path)
    first = assess_first_pair(a, b, mapping, expected)
    return {"schema": "qwen_action_human_qa_coordinator_attestation_v2",
            "bundle_mapping_sha256": sha(BUNDLE / "TECHNICAL_MAPPING.json"),
            "first_export_sha256": {"A": sha(first_a_path), "B": sha(first_b_path)},
            "reviewer_ids": first["reviewer_ids"],
            "coordinator_id": "", "independent_people": None,
            "separate_blind_folders": None,
            "no_shared_answers_before_both_locks": None,
            "signed_utc": "", "notes": ""}


def validate_attestation(document: dict[str, Any], template: dict[str, Any],
                         first_times: tuple[datetime, datetime]) -> None:
    require(set(document) == ATTESTATION_KEYS
            and document["schema"] == template["schema"]
            and document["bundle_mapping_sha256"]
                == template["bundle_mapping_sha256"]
            and document["first_export_sha256"]
                == template["first_export_sha256"]
            and document["reviewer_ids"] == template["reviewer_ids"],
            "coordinator attestation source binding differs")
    require(isinstance(document["coordinator_id"], str)
            and REVIEWER_ID.fullmatch(document["coordinator_id"]) is not None,
            "coordinator identity missing or invalid")
    require(all(document[key] is True for key in (
        "independent_people", "separate_blind_folders",
        "no_shared_answers_before_both_locks")),
        "coordinator independence/blinding statement missing")
    require(isinstance(document["notes"], str)
            and bool(document["notes"].strip()),
            "coordinator notes missing")
    require(timestamp(document["signed_utc"]) >= max(first_times),
            "coordinator statement predates first review locks")


def third_plan(first_a_path: Path, first_b_path: Path) -> dict:
    mapping, a, b, expected = _load_bound(first_a_path, first_b_path)
    result = assess_first_pair(a, b, mapping, expected)
    require(result["third_required_case_ids"], "no critical disagreement for third review")
    return result


def make_third_bundle(first_a_path: Path, first_b_path: Path,
                      output: Path = THIRD) -> dict:
    from . import listening_bundle_v2 as viewer
    mapping, a, b, expected = _load_bound(first_a_path, first_b_path)
    result = assess_first_pair(a, b, mapping, expected)
    flagged = sorted(result["third_required_case_ids"])
    require(flagged and output.resolve() == THIRD.resolve() and not output.exists(),
            "third bundle requires flagged cases and new fixed directory")
    pair_hashes = {"A": sha(first_a_path), "B": sha(first_b_path)}
    third_binding = hashlib.sha256((mapping["bundle_binding"] + pair_hashes["A"]
                                    + pair_hashes["B"]).encode()).hexdigest()
    created = now_utc()
    require(created >= max(result["first_export_times"]),
            "third bundle cannot precede both first review locks")
    output.mkdir(parents=True, exist_ok=False)
    folder = output / "reviewer_C"
    folder.mkdir()
    manifest = read_json(SOURCE / "manifest.json")
    source_rows = {row["case_id"]: row for row in manifest["rows"]}
    rows = []
    for index, case_id in enumerate(flagged, 1):
        file = f"q{index:02d}.wav"
        original = SOURCE / source_rows[case_id]["final"]
        shutil.copyfile(original, folder / file)
        require(sha(original) == sha(folder / file), "third blinded WAV copy changed")
        rows.append({"ordinal": index, "opaque_file": file,
                     "case_id": case_id, "wav_sha256": sha(original)})
    html = viewer._html("C", [row["opaque_file"] for row in rows], third_binding)
    (folder / "REVIEW.html").write_text(html, encoding="utf-8")
    receipt = {"schema": "qwen_action_sparse_third_bundle_v1",
               "status": "human_third_review_pending",
               "bundle_binding": third_binding,
               "base_mapping_sha256": sha(BUNDLE / "TECHNICAL_MAPPING.json"),
               "first_export_sha256": pair_hashes,
               "first_export_locks_utc": result["first_export_locks_utc"],
               "created_utc": created.isoformat().replace("+00:00", "Z"),
               "queues": {"C": rows},
               "html_sha256": sha(folder / "REVIEW.html")}
    source.write_json(output / "TECHNICAL_MAPPING.json", receipt)
    return receipt


def validate_third_mapping(third_mapping: dict, base_mapping: dict,
                           pair_hashes: dict[str, str], flagged: list[str],
                           source_rows: dict[str, dict],
                           first_locks: dict[str, str]) -> list[dict]:
    """Rebuild the sparse third queue from both first exports and frozen WAVs."""
    ordered = sorted(flagged)
    expected_binding = hashlib.sha256((base_mapping["bundle_binding"]
                                       + pair_hashes["A"] + pair_hashes["B"])
                                      .encode()).hexdigest()
    expected_rows = [
        {"ordinal": index, "opaque_file": f"q{index:02d}.wav",
         "case_id": case_id,
         "wav_sha256": source_rows[case_id]["final_sha256"]}
        for index, case_id in enumerate(ordered, 1)
    ]
    require(set(third_mapping) == {"schema", "status", "bundle_binding",
                                   "base_mapping_sha256", "first_export_sha256",
                                   "first_export_locks_utc", "created_utc",
                                   "queues", "html_sha256"}
            and third_mapping["schema"] == "qwen_action_sparse_third_bundle_v1"
            and third_mapping["status"] == "human_third_review_pending"
            and third_mapping["bundle_binding"] == expected_binding
            and third_mapping["base_mapping_sha256"]
                == sha(BUNDLE / "TECHNICAL_MAPPING.json")
            and third_mapping["first_export_sha256"] == pair_hashes
            and third_mapping["first_export_locks_utc"] == first_locks
            and timestamp(third_mapping["created_utc"])
                >= max(timestamp(first_locks["A"]), timestamp(first_locks["B"]))
            and set(third_mapping["queues"]) == {"C"}
            and third_mapping["queues"]["C"] == expected_rows,
            "third bundle source/review binding differs")
    return expected_rows


def audit_reviews(first_a_path: Path, first_b_path: Path,
                  attestation_path: Path,
                  third_path: Path | None = None) -> dict[str, Any]:
    mapping, a, b, expected = _load_bound(first_a_path, first_b_path)
    first = assess_first_pair(a, b, mapping, expected)
    statement = read_json(attestation_path)
    validate_attestation(statement,
                         attestation_template(first_a_path, first_b_path),
                         first["first_export_times"])
    third_review = None
    third_hash = None
    third_rows: list[dict[str, Any]] = []
    if first["third_required_case_ids"]:
        require(third_path is not None, "third human review required for disagreement")
        third_mapping = read_json(THIRD / "TECHNICAL_MAPPING.json")
        source_rows = {row["case_id"]: row for row in
                       read_json(SOURCE / "manifest.json")["rows"]}
        third_plan_rows = validate_third_mapping(
            third_mapping, mapping,
            {"A": sha(first_a_path), "B": sha(first_b_path)},
            first["third_required_case_ids"], source_rows,
            first["first_export_locks_utc"])
        folder = THIRD / "reviewer_C"
        require(THIRD.is_dir() and not THIRD.is_symlink()
                and {p.name for p in THIRD.iterdir()}
                    == {"TECHNICAL_MAPPING.json", "reviewer_C"}
                and folder.is_dir() and not folder.is_symlink()
                and {p.name for p in folder.iterdir()}
                    == {"REVIEW.html"}
                       | {row["opaque_file"] for row in third_plan_rows},
                "third bundle file inventory changed")
        expected_html = bundle._html("C", [r["opaque_file"] for r in
                                      third_plan_rows],
                                     third_mapping["bundle_binding"])
        require(sha(folder / "REVIEW.html") == third_mapping["html_sha256"]
                and (folder / "REVIEW.html").read_text() == expected_html,
                "third blind form changed")
        for row in third_plan_rows:
            require(sha(folder / row["opaque_file"]) == row["wav_sha256"],
                    "third blind WAV changed")
        third_review = read_json(third_path)
        third_time, third_id = validate_export(
            third_review, "C", third_mapping,
            expected_rows=third_plan_rows,
            expected_binding=third_mapping["bundle_binding"])
        require(third_id not in first["reviewer_ids"].values()
                and third_time >= timestamp(third_mapping["created_utc"]),
                "third reviewer identity/chronology invalid")
        third_hash = sha(third_path)
        for review_row, plan_row in zip(third_review["rows"],
                                        third_plan_rows, strict=True):
            passes, reasons = _row_pass(review_row, expected[plan_row["case_id"]])
            third_rows.append({"case_id": plan_row["case_id"],
                               "third_reviewer_pass": passes,
                               "third_reviewer_failures": reasons})
    else:
        require(third_path is None, "third review forbidden without disagreement")
    all_pass = first["first_pair_all_pass"] and not first["third_required_case_ids"]
    # A third documents disagreement but never retroactively makes two initial
    # first-pass reviews agree or clears a failed source.
    return {"schema": "qwen_action_human_qa_audit_v2",
            "status": "human_qa_pass_needs_independent_review" if all_pass
                      else "human_qa_hold",
            "all_18_initial_pairs_pass": first["first_pair_all_pass"],
            "critical_disagreement_count": len(first["third_required_case_ids"]),
            "third_review_received": third_review is not None,
            "third_case_results": third_rows,
            "case_census": 18, "cases": first["cases"],
            "source_manifest_sha256": sha(SOURCE / "manifest.json"),
            "bundle_mapping_sha256": sha(BUNDLE / "TECHNICAL_MAPPING.json"),
            "first_export_sha256": {"A": sha(first_a_path), "B": sha(first_b_path)},
            "coordinator_attestation_sha256": sha(attestation_path),
            "coordinator_id": statement["coordinator_id"],
            "third_export_sha256": third_hash,
            "reviewer_ids": first["reviewer_ids"],
            "first_export_locks_utc": first["first_export_locks_utc"],
            "candidate_model_inference_authorized": False,
            "identity_and_blinding_attestation_required": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--review-c", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--make-third", action="store_true")
    parser.add_argument("--inspect-first", action="store_true")
    parser.add_argument("--make-attestation-template", action="store_true")
    args = parser.parse_args()
    require(sum((args.make_third, args.inspect_first,
                 args.make_attestation_template)) <= 1,
            "choose only one first-review action")
    if args.make_attestation_template:
        require(args.output is not None and args.output.is_absolute()
                and not args.output.exists() and args.review_c is None
                and args.attestation is None,
                "template needs a new absolute output and two first reviews")
        source.write_json(args.output, attestation_template(args.review_a,
                                                            args.review_b))
        print(json.dumps({"unsigned_template": str(args.output)}, sort_keys=True))
        return
    if args.inspect_first:
        require(args.output is None and args.review_c is None
                and args.attestation is None,
                "first-review inspection accepts only two reviews")
        mapping, a, b, expected = _load_bound(args.review_a, args.review_b)
        first = assess_first_pair(a, b, mapping, expected)
        print(json.dumps({"status": "first_pair_audited_human_qa_pending",
                          "case_census": len(first["cases"]),
                          "first_pair_all_pass": first["first_pair_all_pass"],
                          "third_required_case_ids": first["third_required_case_ids"],
                          "candidate_model_inference_authorized": False}, sort_keys=True))
        return
    if args.make_third:
        require(args.output is None and args.review_c is None
                and args.attestation is None,
                "third bundle command accepts only two first reviews")
        print(json.dumps({"third_bundle": str(THIRD),
                          "flagged": len(make_third_bundle(args.review_a,
                                                            args.review_b)["queues"]["C"])},
                         sort_keys=True))
        return
    require(args.output is not None and args.output.is_absolute()
            and not args.output.exists() and args.attestation is not None,
            "new absolute output and coordinator attestation required")
    result = audit_reviews(args.review_a, args.review_b,
                           args.attestation, args.review_c)
    source.write_json(args.output, result)
    print(json.dumps({"status": result["status"],
                      "case_census": result["case_census"]}, sort_keys=True))


if __name__ == "__main__":
    main()
