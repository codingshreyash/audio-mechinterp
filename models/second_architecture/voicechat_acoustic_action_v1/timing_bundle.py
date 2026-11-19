"""Offline human timing bundle; no model or automatic alignment.

The browser exports two independent first impressions.  ``assemble`` checks
them against the audited source bytes and writes the existing stage-2 schema.
This software cannot establish that the named reviewers are distinct people.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs" / "sources_20261119_0645Z"
SOURCE_MANIFEST_SHA256 = "3d7714430bfca4c5888b7142be0d6ca575841f3cad0e74163650f02e0d8b1ca5"
ANNOTATION_TEMPLATE_SHA256 = "ebac0ab9ab1804eb7d225313f47ad0fd9e417b1ab6a80316a8a6dd13fefe3a86"
SOURCE_REVIEW = HERE / "SOURCE_RENDERED_INDEPENDENT_REVIEW_GO_V1.md"
SOURCE_ADMISSION = HERE / "SOURCE_RENDERED_INDEPENDENT_ADMISSION_GO_V1.json"
TEMPLATE = HERE / "timing_form.html"
TEMPLATE_SHA256 = "d5eac51cb53bd76c33deed5e4d91237445825fded8286ef3e4b8c0c05cf72efe"
SCHEMA = "voicechat_timing_independent_review_v1"
GAP = 2560
README_TEXT = (
    "# Model-blind timing annotation\n\n"
    "Open TIMING_FORM.html locally. Each reviewer uses a separate browser profile and unique ID. "
    "Listen to all 12 clips, mark the stated phrase and a same-width speech interval elsewhere, "
    "then export the JSON. Use the Cannot mark option if either interval cannot be identified. "
    "Do not discuss marks, view model outputs, or edit an export before both first impressions are locked. "
    "A third person uses the same form if any intervals differ. The coordinator keeps all original exports. "
    "No interval is supplied or inferred by this bundle.\n"
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    _require(path.is_file() and not path.is_symlink(), f"Missing or linked file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _source() -> tuple[dict, dict]:
    # A pinned local study input, never a model answer.
    from . import corpus

    _require(_sha(SOURCE / "source_manifest.json") == SOURCE_MANIFEST_SHA256,
             "Audited source manifest changed")
    _require(SOURCE_REVIEW.is_file() and SOURCE_ADMISSION.is_file(),
             "Independent source review/admission absent")
    manifest_path = SOURCE / "source_manifest.json"
    manifest = _load(manifest_path)
    synthetic, _admission = corpus._checked_source_provenance(
        SOURCE, corpus._spec(), manifest_path, manifest, allow_synthetic_test_only=False)
    _require(synthetic is False, "Synthetic source cannot enter human annotation")
    _require(_sha(SOURCE / "annotation_template.json") == ANNOTATION_TEMPLATE_SHA256,
             "Audited timing template changed")
    template = _load(SOURCE / "annotation_template.json")
    _require(template.get("schema") == "voicechat_acoustic_intervals_v1"
             and template.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256
             and template.get("sample_rate_hz") == 16000
             and len(template.get("records", [])) == 12,
             "Source timing template differs")
    by_id = {row["source_id"]: row for row in manifest["rows"]}
    for record in template["records"]:
        row = by_id[record["source_id"]]
        path = SOURCE / record["clean_wav_path"]
        _require(record["clean_wav_path"] == row["clean_path"]
                 and record["clean_wav_sha256"] == row["clean_sha256"] == _sha(path)
                 and record["content_frames"] == row["clean_frames"] - 80000
                 and record["decision_phrase"] == row["decision_phrase"],
                 "Timing template/source WAV mismatch")
    return manifest, template


def _queue(template: dict) -> dict:
    rows = [{key: record[key] for key in (
        "source_id", "clean_wav_path", "clean_wav_sha256", "content_frames", "decision_phrase")}
        for record in template["records"]]
    return {"schema": "voicechat_timing_queue_v1", "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "sample_rate_hz": 16000, "min_gap_frames": GAP, "rows": rows}


def _rendered_bytes(queue: dict) -> tuple[bytes, bytes, bytes]:
    _require(_sha(TEMPLATE) == TEMPLATE_SHA256, "Reviewed timing form template changed")
    queue_bytes = (json.dumps(queue, sort_keys=True, indent=2) + "\n").encode()
    payload = json.dumps(queue, separators=(",", ":")).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__EMBEDDED_QUEUE__", payload)
    html = html.replace("__QUEUE_SHA256__", hashlib.sha256(queue_bytes).hexdigest())
    _require("__EMBEDDED_QUEUE__" not in html and "__QUEUE_SHA256__" not in html,
             "Form template insertion failed")
    return queue_bytes, html.encode("utf-8"), README_TEXT.encode("utf-8")


def build(output_dir: Path) -> dict:
    _manifest, template = _source()
    _require(output_dir.is_absolute() and not output_dir.exists(),
             "Bundle output must be a new absolute directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    for record in template["records"]:
        name = record["clean_wav_path"]
        shutil.copyfile(SOURCE / name, output_dir / name)
        _require(_sha(output_dir / name) == record["clean_wav_sha256"], "Bundle WAV copy differs")
    queue = _queue(template)
    queue_bytes, html_bytes, readme_bytes = _rendered_bytes(queue)
    (output_dir / "QUEUE.json").write_bytes(queue_bytes)
    (output_dir / "TIMING_FORM.html").write_bytes(html_bytes)
    (output_dir / "README.md").write_bytes(readme_bytes)
    return {"status": "human_annotation_pending", "rows": 12, "queue_sha256": _sha(output_dir / "QUEUE.json")}


def _window(value: object, content: int) -> tuple[int, int]:
    _require(isinstance(value, list) and len(value) == 2 and
             all(type(v) is int for v in value) and 0 <= value[0] < value[1] <= content,
             "Invalid sample-frame interval")
    return value[0], value[1]


def _record_windows(record: dict, source: dict) -> tuple[list[int], list[int]]:
    _require(record.get("status") == "marked", "Human reviewer could not mark a source")
    content = source["content_frames"]
    decision = _window(record.get("decision"), content)
    displaced = _window(record.get("displaced"), content)
    _require(decision[1] - decision[0] == displaced[1] - displaced[0],
             "Displaced window width differs")
    _require(decision[1] + GAP <= displaced[0] or displaced[1] + GAP <= decision[0],
             "Decision/displaced windows overlap or lack 160 ms gap")
    return list(decision), list(displaced)


def _review(path: Path, queue: dict) -> dict:
    result = _load(path)
    _require(result.get("schema") == SCHEMA
             and result.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256
             and result.get("queue_sha256") == hashlib.sha256(
                 (json.dumps(queue, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
             and isinstance(result.get("reviewer_id"), str)
             and len(result["reviewer_id"].strip()) >= 2
             and isinstance(result.get("records"), list)
             and len(result["records"]) == 12,
             "Independent reviewer export identity or population differs")
    for source, record in zip(queue["rows"], result["records"], strict=True):
        _require(isinstance(record, dict) and record.get("source_id") == source["source_id"]
                 and record.get("clean_wav_sha256") == source["clean_wav_sha256"]
                 and isinstance(record.get("submitted_at_utc"), str)
                 and record.get("status") in ("marked", "unmarkable"),
                 "Reviewer record order/identity incomplete")
        if record["status"] == "marked":
            _record_windows(record, source)
        else:
            _require(isinstance(record.get("note"), str) and record["note"].strip(),
                     "Unmarkable reason missing")
    return result


def assemble(bundle: Path, first: Path, second: Path, output: Path,
             third: Path | None = None) -> dict:
    _manifest, template = _source()
    _require(bundle.is_dir() and not bundle.is_symlink(), "Timing bundle directory invalid")
    expected_names = {record["clean_wav_path"] for record in template["records"]}
    expected_names.update(("QUEUE.json", "TIMING_FORM.html", "README.md"))
    actual_entries = list(bundle.iterdir())
    _require(len(actual_entries) == 15 and {p.name for p in actual_entries} == expected_names
             and all(p.is_file() and not p.is_symlink() for p in actual_entries),
             "Timing bundle inventory differs from reviewed 15 files")
    queue = _load(bundle / "QUEUE.json")
    _require(queue == _queue(template), "Timing bundle queue differs from audited source")
    expected_queue, expected_html, expected_readme = _rendered_bytes(_queue(template))
    _require((bundle / "QUEUE.json").read_bytes() == expected_queue
             and (bundle / "TIMING_FORM.html").read_bytes() == expected_html
             and (bundle / "README.md").read_bytes() == expected_readme,
             "Timing bundle form, queue, or README bytes changed")
    for row in queue["rows"]:
        _require((bundle / row["clean_wav_path"]).is_file()
                 and _sha(bundle / row["clean_wav_path"]) == row["clean_wav_sha256"],
                 "Timing bundle WAV changed")
    a, b = _review(first, queue), _review(second, queue)
    _require(a["reviewer_id"] != b["reviewer_id"], "Independent reviewer IDs must differ")
    c = _review(third, queue) if third else None
    if c:
        _require(c["reviewer_id"] not in {a["reviewer_id"], b["reviewer_id"]},
                 "Third reviewer ID must differ")
    result = json.loads(json.dumps(template))
    result["assembly_provenance"] = {
        "schema": "voicechat_timing_assembly_provenance_v1",
        "queue_sha256": _sha(bundle / "QUEUE.json"),
        "first_export_sha256": _sha(first),
        "second_export_sha256": _sha(second),
        "third_export_sha256": _sha(third) if third else None,
        "assembler_source_sha256": _sha(Path(__file__)),
    }
    disagreements = []
    for idx, source in enumerate(queue["rows"]):
        ar, br = a["records"][idx], b["records"][idx]
        _require(ar["status"] == br["status"] == "marked",
                 "At least one initial reviewer could not mark a source; corpus HOLD")
        apair, bpair = _record_windows(ar, source), _record_windows(br, source)
        record = result["records"][idx]
        record["annotator_a"] = {"reviewer_id": a["reviewer_id"], "decision": apair[0],
                                  "displaced": apair[1]}
        record["annotator_b"] = {"reviewer_id": b["reviewer_id"], "decision": bpair[0],
                                  "displaced": bpair[1]}
        if apair == bpair:
            final = apair
            record["adjudication"] = {"by": "consensus_of_" + a["reviewer_id"] + "_" + b["reviewer_id"],
                                      "decision": final[0], "displaced": final[1],
                                      "note": "Both initial interval pairs agree exactly."}
        else:
            disagreements.append(source["source_id"])
            _require(c is not None, "Third reviewer required for differing intervals")
            cr = c["records"][idx]
            _require(cr["status"] == "marked", "Third reviewer could not mark disagreement")
            final = _record_windows(cr, source)
            record["third_annotation"] = {"reviewer_id": c["reviewer_id"],
                                           "decision": final[0], "displaced": final[1]}
            record["adjudication"] = {"by": c["reviewer_id"], "decision": final[0],
                                      "displaced": final[1],
                                      "note": "Third independent timing annotation selected."}
    _require(output.is_absolute() and not output.exists(), "New absolute assembly output required")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"status": "assembled_human_intervals_not_qa_go", "records": 12,
            "third_reviewed_sources": disagreements, "output_sha256": _sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output-dir", type=Path, required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--first", type=Path, required=True)
    a.add_argument("--second", type=Path, required=True)
    a.add_argument("--third", type=Path)
    a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.output_dir), sort_keys=True))
    else:
        print(json.dumps(assemble(args.bundle, args.first, args.second, args.output, args.third), sort_keys=True))


if __name__ == "__main__":
    main()
