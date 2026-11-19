"""Validate authentic reviewer exports and adjudicate a blinded C queue."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import blinded_listening_v1 as blind


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def majority(values: list[str]) -> str | None:
    counts = Counter(values)
    key, count = counts.most_common(1)[0]
    return key if count >= 2 else None


def verify_c_queue(folder: Path, receipt: dict, mapping: dict,
                   selected: list[dict]) -> dict:
    c = receipt["reviewer_C"]
    need(c is not None and folder.is_dir() and not folder.is_symlink(),
         "C queue absent")
    public = json.loads((folder / "QUEUE.json").read_text())
    expected_public, expected_order = blind.queue("C", selected, mapping["bundle_binding"])
    need(public["schema"] == "human_audio_blind_queue_v1"
         and public == expected_public
         and public["label"] == "C"
         and public["bundle_binding"] == mapping["bundle_binding"]
         and public["queue_sha256"] == c["queue_sha256"]
         and blind.sha(folder / "QUEUE.json") == c["queue_file_sha256"]
         and blind.sha(folder / "REVIEW.html") == c["html_sha256"]
         and (folder / "REVIEW.html").read_text() == blind.html(public)
         and len(public["clips"]) == len(c["mapping"])
         and {p.name for p in folder.iterdir()} ==
             {"QUEUE.json", "REVIEW.html"} | {r["file"] for r in public["clips"]},
         "C queue source binding changed")
    source = {r["case_id"]: r for r in blind.cases()}
    for row, visible, selected_case in zip(c["mapping"], public["clips"], expected_order):
        wav = folder / visible["file"]
        need(row["case_id"] in source
             and row == {**source[row["case_id"]], **visible}
             and row["case_id"] == selected_case["case_id"]
             and wav.is_file() and not wav.is_symlink()
             and blind.sha(wav) == visible["wav_sha256"], "C WAV changed")
    return {"queues": {"C": c}, "bundle_binding": mapping["bundle_binding"]}


def finalize(bundle: Path, third: Path, a_path: Path, b_path: Path,
             c_path: Path | None) -> dict:
    need(blind.audit_bundle(bundle)["status"] == "PASS_SOURCE_ONLY",
         "A/B source bundle changed")
    mapping = json.loads((bundle / "TECHNICAL_MAPPING.json").read_text())
    receipt = json.loads((third / "ADJUDICATION_MAPPING.json").read_text())
    need(receipt["schema"] == "human_audio_third_queue_v1"
         and receipt["source_bundle_mapping_sha256"] ==
             blind.sha(bundle / "TECHNICAL_MAPPING.json")
         and receipt["review_A_sha256"] == blind.sha(a_path)
         and receipt["review_B_sha256"] == blind.sha(b_path),
         "A/B source or review hashes changed")
    a, a_cases = blind.validate_review(a_path, "A", mapping)
    b, b_cases = blind.validate_review(b_path, "B", mapping)
    need(a["reviewer_id"] != b["reviewer_id"], "A/B reviewer identity repeated")
    cases = blind.cases()
    def value(record: dict, field: str) -> str:
        raw = record[field]
        return blind.normalize(raw) if field == "verbatim_transcript" else raw

    disputed = [case["case_id"] for case in cases if any(
        value(a_cases[case["case_id"]], field) != value(b_cases[case["case_id"]], field)
        for field in ("verbatim_transcript", "clarity", "tail_audible"))]
    need(receipt["disagreements"] == len(disputed), "disagreement census changed")
    if disputed:
        need(receipt["status"] == "third_review_pending" and c_path is not None,
             "C review required")
        selected = [case for case in cases if case["case_id"] in disputed]
        c_mapping = verify_c_queue(third / "reviewer_C", receipt, mapping, selected)
        c, c_cases = blind.validate_review(c_path, "C", c_mapping)
        need(c["reviewer_id"] not in (a["reviewer_id"], b["reviewer_id"])
             and set(c_cases) == set(disputed), "C reviewer identity/census invalid")
        c_hash = blind.sha(c_path)
    else:
        need(receipt["status"] == "no_disagreements"
             and receipt["reviewer_C"] is None and c_path is None,
             "unexpected C review")
        c_cases, c_hash = {}, None
    references = {f"{row['utterance_id']}|{condition['condition']}":
                  row["reference_transcript"]
                  for row in json.loads((blind.SOURCE / "manifest.json").read_text())["rows"]
                  for condition in row["conditions"]}
    results = []
    for case in cases:
        ident = case["case_id"]
        members = [a_cases[ident], b_cases[ident]]
        if ident in c_cases:
            members.append(c_cases[ident])
        judgments = {
            "transcript": [blind.normalize(r["verbatim_transcript"]) for r in members],
            "clarity": [r["clarity"] for r in members],
            "tail_audible": [r["tail_audible"] for r in members],
        }
        agreed = {key: majority(values) for key, values in judgments.items()}
        reference = blind.normalize(references[ident])
        passed = (agreed["transcript"] == reference and agreed["clarity"] == "clear"
                  and agreed["tail_audible"] in ("yes", "no"))
        results.append({"case_id": ident, "wav_sha256": case["wav_sha256"],
                        "reviewer_ids": [r["reviewer_id"] for r in members],
                        "raw": [{k: r[k] for k in ("verbatim_transcript", "clarity",
                                                  "tail_audible", "notes")} for r in members],
                        "disagreed": ident in disputed, "majority": agreed,
                        "normalized_reference": reference, "listening_screen_pass": passed})
    return {"schema": "human_audio_review_intake_v2", "status": "REVIEWED",
            "source_mapping_sha256": blind.sha(bundle / "TECHNICAL_MAPPING.json"),
            "adjudication_mapping_sha256": blind.sha(third / "ADJUDICATION_MAPPING.json"),
            "review_shas": {"A": blind.sha(a_path), "B": blind.sha(b_path), "C": c_hash},
            "cases": results, "passed": sum(r["listening_screen_pass"] for r in results),
            "denominator": 16, "disagreements": len(disputed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=blind.BUNDLE)
    parser.add_argument("--third", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--review-c", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.bundle, args.third, args.review_a, args.review_b, args.review_c)
    with args.output.open("x") as out:
        out.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"], "passed": result["passed"],
                      "denominator": result["denominator"]}))


if __name__ == "__main__":
    main()
