"""Build and audit two blinded human-speech queues; no human answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import unicodedata
from pathlib import Path

from . import audit_source_v1 as source_audit

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_v1"
BUNDLE = HERE / "blind_bundle_v1"
TEMPLATE = HERE / "blinded_form_template.html"
JS = HERE / "blinded_form.js"
MANIFEST_SHA = "c0593a4631191b7643714651c751fde29ebc6323bf1d59680d3a4c1d1bd38051"


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def binding() -> str:
    return hashlib.sha256((MANIFEST_SHA + sha(Path(__file__)) + sha(TEMPLATE) + sha(JS)).encode()).hexdigest()


def cases() -> list[dict]:
    need(sha(SOURCE / "manifest.json") == MANIFEST_SHA, "source manifest changed")
    need(source_audit.audit(source_audit.ARCHIVE, SOURCE)["status"] == "PASS",
         "source technical audit failed")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    result = []
    for row in manifest["rows"]:
        for condition in row["conditions"]:
            result.append({"case_id": f"{row['utterance_id']}|{condition['condition']}",
                           "utterance_id": row["utterance_id"],
                           "condition": condition["condition"],
                           "source_wav": condition["wav"],
                           "wav_sha256": condition["wav_sha256"]})
    need(len(result) == 16 and len({r["wav_sha256"] for r in result}) == 16,
         "16 unique source WAVs required")
    return result


def order(label: str, all_cases: list[dict]) -> list[dict]:
    need(label in ("A", "B", "C"), "unknown reviewer queue")
    rows = all_cases.copy()
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f"human-audio-blind-v1:{MANIFEST_SHA}:{label}".encode()).digest(), "big"))
    rng.shuffle(rows)
    return rows


def queue(label: str, selected: list[dict], binding_sha: str) -> tuple[dict, list[dict]]:
    rows = []
    for case in order(label, selected):
        ident = hashlib.sha256((binding_sha + "|" + label + "|" + case["wav_sha256"]).encode()).hexdigest()[:20]
        rows.append({"opaque_id": ident, "file": ident + ".wav",
                     "wav_sha256": case["wav_sha256"]})
    need(len(rows) == len({r["opaque_id"] for r in rows}), "opaque IDs collided")
    public = {"schema": "human_audio_blind_queue_v1", "label": label,
              "bundle_binding": binding_sha, "source_manifest_sha256": MANIFEST_SHA,
              "clips": rows}
    public["queue_sha256"] = hashlib.sha256(canonical(public)).hexdigest()
    return public, order(label, selected)


def html(public: dict) -> str:
    template = TEMPLATE.read_text()
    need(template.count("__EMBEDDED_DATA__") == 1 and template.count("__APP_JS__") == 1,
         "HTML template placeholders changed")
    embedded = json.dumps(public, sort_keys=True, separators=(",", ":"))
    embedded = embedded.replace("<", "\\u003c").replace("&", "\\u0026")
    return template.replace("__EMBEDDED_DATA__", embedded).replace("__APP_JS__", JS.read_text())


def write_queue(folder: Path, label: str, selected: list[dict], binding_sha: str) -> dict:
    need(not folder.exists(), "reviewer directory already exists")
    folder.mkdir(parents=True)
    public, ordered = queue(label, selected, binding_sha)
    mapped = []
    for visible, case in zip(public["clips"], ordered):
        source = SOURCE / case["source_wav"]
        target = folder / visible["file"]
        shutil.copyfile(source, target)
        need(sha(source) == visible["wav_sha256"] == sha(target),
             "blind WAV copy changed")
        mapped.append({**case, **visible})
    (folder / "QUEUE.json").write_bytes(canonical(public))
    (folder / "REVIEW.html").write_text(html(public))
    return {"label": label, "queue_sha256": public["queue_sha256"],
            "queue_file_sha256": sha(folder / "QUEUE.json"),
            "html_sha256": sha(folder / "REVIEW.html"), "mapping": mapped}


def build(output: Path = BUNDLE) -> dict:
    need(not output.exists(), "one-shot listening bundle exists")
    all_cases = cases()
    binding_sha = binding()
    output.mkdir(parents=True)
    queues = {label: write_queue(output / f"reviewer_{label}", label, all_cases,
                                 binding_sha) for label in ("A", "B")}
    need([r["case_id"] for r in queues["A"]["mapping"]]
         != [r["case_id"] for r in queues["B"]["mapping"]],
         "reviewer orders identical")
    mapping = {"schema": "human_audio_blind_mapping_v1",
               "status": "source_only_zero_human_reviews", "bundle_binding": binding_sha,
               "source_manifest_sha256": MANIFEST_SHA, "builder_sha256": sha(Path(__file__)),
               "template_sha256": sha(TEMPLATE), "js_sha256": sha(JS),
               "queues": queues, "human_reviews": 0}
    (output / "TECHNICAL_MAPPING.json").write_bytes(canonical(mapping))
    return mapping


def audit_bundle(output: Path = BUNDLE) -> dict:
    need(output.is_dir() and not output.is_symlink(), "bundle unavailable")
    mapping = json.loads((output / "TECHNICAL_MAPPING.json").read_text())
    need(mapping.get("schema") == "human_audio_blind_mapping_v1"
         and mapping.get("status") == "source_only_zero_human_reviews"
         and mapping.get("source_manifest_sha256") == MANIFEST_SHA
         and mapping.get("bundle_binding") == binding()
         and mapping.get("builder_sha256") == sha(Path(__file__))
         and mapping.get("template_sha256") == sha(TEMPLATE)
         and mapping.get("js_sha256") == sha(JS)
         and mapping.get("human_reviews") == 0,
         "bundle source identity changed")
    need({p.name for p in output.iterdir()} ==
         {"TECHNICAL_MAPPING.json", "reviewer_A", "reviewer_B"},
         "bundle root inventory changed")
    expected_cases = cases()
    for label in ("A", "B"):
        folder = output / f"reviewer_{label}"
        need(folder.is_dir() and not folder.is_symlink(), "reviewer directory invalid")
        public, ordered = queue(label, expected_cases, binding())
        got = json.loads((folder / "QUEUE.json").read_text())
        need(got == public and mapping["queues"][label]["queue_sha256"] == public["queue_sha256"]
             and mapping["queues"][label]["queue_file_sha256"] == sha(folder / "QUEUE.json")
             and {p.name for p in folder.iterdir()} ==
                {"QUEUE.json", "REVIEW.html"} | {c["file"] for c in public["clips"]},
             "reviewer queue identity changed")
        for visible, case, mapped in zip(public["clips"], ordered,
                                         mapping["queues"][label]["mapping"]):
            wav = folder / visible["file"]
            need(wav.is_file() and not wav.is_symlink()
                 and sha(wav) == sha(SOURCE / case["source_wav"]) == visible["wav_sha256"]
                 and mapped == {**case, **visible}, "blind copy/mapping changed")
        need((folder / "REVIEW.html").read_text() == html(public)
             and sha(folder / "REVIEW.html") == mapping["queues"][label]["html_sha256"],
             "blind HTML changed")
        # Neither case IDs nor transcripts nor condition labels appear in the reviewer-facing files.
        visible_text = (folder / "REVIEW.html").read_text() + (folder / "QUEUE.json").read_text()
        need(all(case["case_id"] not in visible_text
                 and case["source_wav"] not in visible_text for case in expected_cases),
             "reviewer-facing case metadata leaked")
    return {"schema": "human_audio_blind_bundle_audit_v1", "status": "PASS_SOURCE_ONLY",
            "source_manifest_sha256": MANIFEST_SHA,
            "mapping_sha256": sha(output / "TECHNICAL_MAPPING.json"),
            "copied_wavs": 32, "forms": 2, "human_reviews": 0}


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()))


def validate_review(path: Path, label: str, mapping: dict) -> tuple[dict, dict[str, dict]]:
    data = json.loads(path.read_text())
    expected = mapping["queues"][label]
    need(data.get("schema") == "human_audio_blind_review_v1"
         and data.get("queue_sha256") == expected["queue_sha256"]
         and data.get("bundle_binding") == mapping["bundle_binding"]
         and data.get("planned_clips") == len(expected["mapping"])
         and isinstance(data.get("reviewer_id"), str)
         and 2 <= len(data["reviewer_id"]) <= 80
         and len(data.get("records", [])) == len(expected["mapping"]),
         "human review schema/census invalid")
    by_case = {}
    for record, row in zip(data["records"], expected["mapping"]):
        need(record.get("opaque_id") == row["opaque_id"]
             and record.get("wav_sha256") == row["wav_sha256"]
             and record.get("reviewer_id") == data["reviewer_id"]
             and isinstance(record.get("verbatim_transcript"), str)
             and bool(record["verbatim_transcript"].strip())
             and record.get("clarity") in ("clear", "unclear")
             and record.get("tail_audible") in ("yes", "no", "unsure")
             and isinstance(record.get("notes"), str)
             and isinstance(record.get("submitted_at_utc"), str),
             "human review record incomplete/mismatched")
        by_case[row["case_id"]] = record
    return data, by_case


def build_third(bundle: Path, a_file: Path, b_file: Path, output: Path) -> dict:
    need(audit_bundle(bundle)["status"] == "PASS_SOURCE_ONLY", "first bundle changed")
    mapping = json.loads((bundle / "TECHNICAL_MAPPING.json").read_text())
    a, a_cases = validate_review(a_file, "A", mapping)
    b, b_cases = validate_review(b_file, "B", mapping)
    need(a["reviewer_id"] != b["reviewer_id"], "reviewers are not independent identities")
    all_cases = cases()
    disagreed = [case for case in all_cases if (
        normalize(a_cases[case["case_id"]]["verbatim_transcript"])
          != normalize(b_cases[case["case_id"]]["verbatim_transcript"])
        or a_cases[case["case_id"]]["clarity"] != b_cases[case["case_id"]]["clarity"]
        or a_cases[case["case_id"]]["tail_audible"]
          != b_cases[case["case_id"]]["tail_audible"])]
    need(not output.exists(), "one-shot adjudication output exists")
    output.mkdir(parents=True)
    if disagreed:
        review = write_queue(output / "reviewer_C", "C", disagreed, mapping["bundle_binding"])
    else:
        review = None
    receipt = {"schema": "human_audio_third_queue_v1",
               "status": "third_review_pending" if disagreed else "no_disagreements",
               "source_bundle_mapping_sha256": sha(bundle / "TECHNICAL_MAPPING.json"),
               "review_A_sha256": sha(a_file), "review_B_sha256": sha(b_file),
               "disagreements": len(disagreed), "reviewer_C": review}
    (output / "ADJUDICATION_MAPPING.json").write_bytes(canonical(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "audit", "third"))
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--review-a", type=Path)
    parser.add_argument("--review-b", type=Path)
    parser.add_argument("--third-output", type=Path, default=HERE / "adjudication_v1")
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.bundle)
        print(json.dumps({"status": result["status"], "bundle": str(args.bundle),
                          "queue_A": result["queues"]["A"]["queue_sha256"],
                          "queue_B": result["queues"]["B"]["queue_sha256"]}))
    elif args.command == "audit":
        print(json.dumps(audit_bundle(args.bundle), sort_keys=True))
    else:
        need(args.review_a is not None and args.review_b is not None,
             "third queue requires two review JSONs")
        result = build_third(args.bundle, args.review_a, args.review_b, args.third_output)
        print(json.dumps({"status": result["status"],
                          "disagreements": result["disagreements"]}))


if __name__ == "__main__":
    main()
