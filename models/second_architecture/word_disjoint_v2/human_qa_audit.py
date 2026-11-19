"""Private, no-model audit of 32 blinded human listening exports.

This checks supplied records and the pinned audio bytes. It cannot establish
that a person actually listened, that identities are genuine, or that the
reviewers did not see the private references. No review is generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from . import audit, queue


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RENDERED = HERE / "rendered_v2"
MANIFEST = RENDERED / "manifest.json"
BINDING = RENDERED / "TECHNICAL_BINDING.json"
PRIVATE = RENDERED / "QA_PRIVATE_CASES.json"
PUBLIC = RENDERED / "listening/QUEUE.json"
TECHNICAL_REVIEW = HERE / "RENDERED_V2_TECHNICAL_INDEPENDENT_REVIEW_V1.md"
DEFAULT_REPORT = RENDERED / "QA_HUMAN_AUDIT_V1.json"
EXPECTED = {
    "manifest": "f8fd1f43f38877d6d08c47597c8db9f494e1377a69f0a2451c2f99fbcda71ea2",
    "binding": "027763d4fa758dac95c552c2f2aef60683f1c780b0c034d82327be8bd59e93b9",
    "technical_review": "e16a8048bd05c9a3246d7e8c99c83815e3d207e0174ffa8adddebab629f7faa4",
}
EXPORT_SCHEMA = "word_disjoint_blind_review_v1"
REPORT_SCHEMA = "word_disjoint_human_qa_audit_v1"
EXPORT_KEYS = {
    "schema", "reviewer_id", "queue_sha256", "planned_clips",
    "opaque_ids_in_order", "started_at_utc", "records", "exported_at_utc",
}
RECORD_KEYS = {
    "opaque_id", "reviewer_id", "wav_sha256", "submitted_at_utc",
    "verbatim_transcript", "middle_word", "clear_or_unclear", "confidence",
}
REVIEWER_ID = re.compile(r"[A-Za-z0-9_.-]{3,64}\Z")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing/linked JSON: {path}")
    return json.loads(
        path.read_text(),
        object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON: {value}")
        ),
    )


def tokens(value: str) -> list[str]:
    require(isinstance(value, str), "transcript or middle word is not text")
    return re.sub(
        r"[\W_]+", " ", unicodedata.normalize("NFKC", value).casefold()
    ).split()


def utc(value: str) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), "timestamp is not UTC Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("invalid UTC timestamp") from error
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
            "timestamp has non-UTC offset")
    return parsed


def context() -> dict[str, Any]:
    """Rebuild the private/public case mapping from audited, frozen audio."""
    for key, path in (
        ("manifest", MANIFEST), ("binding", BINDING),
        ("technical_review", TECHNICAL_REVIEW),
    ):
        require(sha(path) == EXPECTED[key], f"pinned {key} changed")
    require("**Decision: GO" in TECHNICAL_REVIEW.read_text(),
            "independent technical review lacks GO")
    manifest, binding = load_json(MANIFEST), load_json(BINDING)
    private, public = load_json(PRIVATE), load_json(PUBLIC)
    audit.validate_provenance(manifest)
    audit.verify_saved_report(manifest, binding, RENDERED, sha(MANIFEST))
    rebuilt_private, rebuilt_public = queue.build(
        manifest, binding, sha(MANIFEST), sha(BINDING)
    )
    require(private == rebuilt_private and public == rebuilt_public,
            "private case map or public queue differs from audited render")
    cases = private["cases"]
    order = public["opaque_ids_in_order"]
    by_id = {case["opaque_id"]: case for case in cases}
    require(len(cases) == len(by_id) == len(order) == len(set(order)) == 32
            and set(by_id) == set(order), "32-case identity/order mismatch")
    for case in cases:
        path = RENDERED / "listening" / (case["opaque_id"] + ".wav")
        require(path.is_file() and not path.is_symlink()
                and sha(path) == case["wav_sha256"],
                "blinded WAV absent or changed")
    return {
        "manifest_sha256": sha(MANIFEST), "binding_sha256": sha(BINDING),
        "private_sha256": sha(PRIVATE), "queue_sha256": sha(PUBLIC),
        "technical_review_sha256": sha(TECHNICAL_REVIEW),
        "auditor_source_sha256": sha(Path(__file__)),
        "order": order, "by_id": by_id,
    }


def validate_export(document: Any, ctx: dict[str, Any]) -> str:
    """Reject altered, incomplete, duplicate, or unbound reviewer exports."""
    require(isinstance(document, dict) and set(document) == EXPORT_KEYS,
            "review export fields differ")
    reviewer = document["reviewer_id"]
    require(isinstance(reviewer, str) and REVIEWER_ID.fullmatch(reviewer) is not None,
            "invalid reviewer ID")
    require(document["schema"] == EXPORT_SCHEMA
            and document["queue_sha256"] == ctx["queue_sha256"]
            and type(document["planned_clips"]) is int
            and document["planned_clips"] == 32
            and document["opaque_ids_in_order"] == ctx["order"],
            "review export queue/schema binding differs")
    records = document["records"]
    require(isinstance(records, list) and len(records) == 32,
            "review export does not contain 32 ordered records")
    begun, ended = utc(document["started_at_utc"]), utc(document["exported_at_utc"])
    require(begun <= ended, "review export time reversed")
    previous = begun
    for ident, record in zip(ctx["order"], records):
        require(isinstance(record, dict) and set(record) == RECORD_KEYS,
                "review record fields differ")
        require(record["opaque_id"] == ident and record["reviewer_id"] == reviewer
                and record["wav_sha256"] == ctx["by_id"][ident]["wav_sha256"],
                "review record identity/WAV binding differs")
        moment = utc(record["submitted_at_utc"])
        require(previous <= moment <= ended, "review submission order/time invalid")
        previous = moment
        require(isinstance(record["verbatim_transcript"], str)
                and isinstance(record["middle_word"], str)
                and record["clear_or_unclear"] in ("clear", "unclear")
                and type(record["confidence"]) is int
                and 1 <= record["confidence"] <= 5,
                "review response type/value invalid")
    return reviewer


def evaluate(ctx: dict[str, Any], documents: dict[str, dict[str, Any]],
             file_hashes: dict[str, str]) -> dict[str, Any]:
    require(set(documents) in ({"primary", "secondary"},
                               {"primary", "secondary", "tertiary"}),
            "primary and secondary exports required")
    reviewers = {stage: validate_export(doc, ctx) for stage, doc in documents.items()}
    require(len(set(reviewers.values())) == len(reviewers),
            "reviewers must have distinct IDs")
    require(set(file_hashes) == set(documents), "review export hashes missing")
    rows = []
    for index, ident in enumerate(ctx["order"]):
        case = ctx["by_id"][ident]
        expected = tokens(case["reference_script"])
        require(expected == ["i", "said", case["word"], "and", "then", "stopped"],
                "private full-script reference changed")
        records = {stage: doc["records"][index] for stage, doc in documents.items()}
        a, b = records["primary"], records["secondary"]
        answers = {
            stage: {
                "full_tokens": tokens(record["verbatim_transcript"]),
                "middle_tokens": tokens(record["middle_word"]),
                "clear": record["clear_or_unclear"] == "clear",
                "correct": tokens(record["verbatim_transcript"]) == expected
                           and tokens(record["middle_word"]) == [case["word"]]
                           and record["clear_or_unclear"] == "clear",
            }
            for stage, record in records.items()
        }
        disagreement = any(
            answers["primary"][key] != answers["secondary"][key]
            for key in ("full_tokens", "middle_tokens", "clear")
        )
        third_required = disagreement
        if third_required and "tertiary" not in records:
            accepted = False
            reason = "third_review_required"
        elif third_required:
            accepted = sum(value["correct"] for value in answers.values()) >= 2
            reason = "two_of_three_correct" if accepted else "unresolved_or_incorrect"
        else:
            accepted = answers["primary"]["correct"] and answers["secondary"]["correct"]
            reason = "two_primary_correct" if accepted else "two_primary_agree_but_incorrect"
        rows.append({
            "opaque_id": ident, "reference_word": case["word"],
            "reference_script": case["reference_script"],
            "disagreement": disagreement, "third_required": third_required,
            "third_present": "tertiary" in records, "quality_pass": accepted,
            "reason": reason, "responses": records, "normalized_answers": answers,
        })
    passed = sum(row["quality_pass"] for row in rows)
    missing_third = sum(row["reason"] == "third_review_required" for row in rows)
    status = ("human_qa_32_of_32_pass" if passed == 32 and missing_third == 0
              else "human_qa_hold")
    return {
        "schema": REPORT_SCHEMA, "status": status,
        "human_quality_gate_passed": status == "human_qa_32_of_32_pass",
        "candidate_model_inference_authorized": False,
        "planned_clips": 32, "passed_clips": passed,
        "disagreement_clips": sum(row["disagreement"] for row in rows),
        "third_review_required_missing": missing_third,
        "reviewer_ids": reviewers,
        "review_export_sha256": file_hashes,
        "manifest_sha256": ctx["manifest_sha256"],
        "technical_binding_sha256": ctx["binding_sha256"],
        "technical_review_sha256": ctx["technical_review_sha256"],
        "private_case_map_sha256": ctx["private_sha256"],
        "public_queue_sha256": ctx["queue_sha256"],
        "auditor_source_sha256": ctx["auditor_source_sha256"],
        "cases": rows,
        "raw_exports": documents,
        "limits": ("File checks cannot establish real human identity, actual "
                   "listening, or whether reviewers were blinded."),
    }


def audit_files(primary: Path, secondary: Path, tertiary: Path | None = None) -> dict[str, Any]:
    ctx = context()
    paths = {"primary": primary, "secondary": secondary}
    if tertiary is not None:
        paths["tertiary"] = tertiary
    require(len({path.resolve() for path in paths.values()}) == len(paths),
            "same export supplied for multiple reviewers")
    for path in paths.values():
        require(not path.resolve().is_relative_to((RENDERED / "listening").resolve()),
                "private reviewer export must not be in public listening directory")
    documents = {stage: load_json(path) for stage, path in paths.items()}
    file_hashes = {stage: sha(path) for stage, path in paths.items()}
    return evaluate(ctx, documents, file_hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--tertiary", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_path = args.report.resolve()
    require(report_path.is_relative_to(RENDERED.resolve())
            and not report_path.is_relative_to((RENDERED / "listening").resolve()),
            "private QA report must stay outside public listening directory")
    report = audit_files(args.primary, args.secondary, args.tertiary)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": report["status"], "passed_clips": report["passed_clips"],
        "planned_clips": 32, "disagreements": report["disagreement_clips"],
        "missing_third": report["third_review_required_missing"],
        "report_path": str(report_path), "report_sha256": sha(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
