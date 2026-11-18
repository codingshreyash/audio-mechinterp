"""Metadata-only, blinded review queue for the technically bound v2 WAV corpus.

No WAV is opened here, and no ASR or candidate model is loaded. The private case
map contains targets and paths; the public queue contains only opaque IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "rendered_v2/manifest.json"
BINDING = HERE / "rendered_v2/TECHNICAL_BINDING.json"
CANDIDATES = HERE / "CANDIDATES_UNFROZEN.json"
VALIDATION = HERE / "CANDIDATE_VALIDATION_UNFROZEN.json"
PROTOCOL = HERE / "ASR_LISTENING_QA_PROTOCOL_PROSPECTIVE_V1.md"
PRIVATE = HERE / "QA_CASES_PROSPECTIVE_V1.json"
QUEUE = HERE / "LISTENING_QUEUE_PROSPECTIVE_V1.json"
PINS = {
    "manifest_sha256": "15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d",
    "binding_sha256": "858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93",
    "candidate_sha256": "ac2f9598e4459191569662aa34803c5a72e57ff686d415e60935a4f716a11a93",
    "validation_sha256": "67e3eec5e14b30a61efbbce449547ec2d5083a87d12a84f0aee8d0e715a59d56",
}
SPLITS = {"fit_pool": 288, "development": 96, "locked_test": 96}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def build() -> tuple[dict, dict]:
    for key, path in (("manifest_sha256", MANIFEST), ("binding_sha256", BINDING),
                      ("candidate_sha256", CANDIDATES), ("validation_sha256", VALIDATION)):
        require(path.is_file() and sha(path) == PINS[key], f"pinned metadata changed: {key}")
    manifest = json.loads(MANIFEST.read_text())
    binding = json.loads(BINDING.read_text())
    spec = json.loads(CANDIDATES.read_text())
    validation = json.loads(VALIDATION.read_text())
    require(binding["status"] == "cpu_wav_technical_audit_passed_intelligibility_pending_no_model_output"
            and binding["manifest_sha256"] == PINS["manifest_sha256"]
            and binding["matched_records"] == 240 and binding["wavs"] == 480
            and binding["model_inference_authorized"] is False,
            "technical binding does not cover planned population")
    require(manifest["matched_records"] == 240 and manifest["wavs"] == 480
            and manifest["primary_pairs"] == 60 and manifest["reserve_pairs_used"] == 0
            and manifest["candidate_sha256"] == PINS["candidate_sha256"]
            and manifest["candidate_validation_sha256"] == PINS["validation_sha256"],
            "render manifest population/provenance changed")
    primary = {word for pairs in spec["pairs_by_category"].values()
               for pair in pairs for word in pair}
    reserves = {word for pairs in spec["ordered_reserve_pairs_by_category"].values()
                for pair in pairs for word in pair}
    require(len(primary) == 120 and len(reserves) == 24 and not primary & reserves,
            "primary/reserve vocabulary changed")
    assigned = {pair["pair_id"]: (split, pair)
                for split, pairs in validation["assigned_pairs"].items() for pair in pairs}
    require(len(assigned) == 60 and Counter(split for split, _ in assigned.values()) ==
            {"fit_pool": 36, "development": 12, "locked_test": 12},
            "whole-pair allocation changed")
    cases: list[dict] = []
    record_keys = set()
    paths = set()
    wav_hashes = set()
    for row in manifest["pairs"]:
        pair_id, split, engine, template_id = (row[k] for k in
                ("pair_id", "split", "engine", "phrasing_id"))
        require(pair_id in assigned and assigned[pair_id][0] == split
                and row["words"] == assigned[pair_id][1]["words"]
                and engine in ("piper", "kokoro")
                and template_id in {t["id"] for t in spec["templates_by_split"][split]},
                "record identity or template changed")
        record_key = (pair_id, engine, template_id)
        require(record_key not in record_keys, "duplicate matched record")
        record_keys.add(record_key)
        require(set(row["outputs"]) == set(row["words"]), "matched pair output missing")
        for word in row["words"]:
            item = row["outputs"][word]
            wav_sha = item["file_sha256"]
            path = item["path"]
            require(path not in paths and wav_sha not in wav_hashes
                    and path.startswith("eval/audio_jlens/corpus_v2/rendered_v2/clips/")
                    and word in primary and word not in reserves,
                    "duplicate or nonprimary clip")
            paths.add(path)
            wav_hashes.add(wav_sha)
            other = next(w for w in row["words"] if w != word)
            opaque = hashlib.sha256(("qwen-v2-qa-v1|" + PINS["manifest_sha256"] +
                                     "|" + wav_sha).encode()).hexdigest()[:20]
            cases.append({"opaque_id": opaque, "pair_id": pair_id, "split": split,
                          "category": row["category"], "engine": engine,
                          "template_id": template_id, "target_word": word,
                          "paired_alternative": other,
                          "reference_script": item["spoken_script"],
                          "audio_path": path, "wav_sha256": wav_sha,
                          "pcm_int16_sha256": item["pcm_int16_sha256"],
                          "frames": item["frames"]})
    require(len(record_keys) == 240 and len(cases) == 480
            and len({case["opaque_id"] for case in cases}) == 480
            and Counter(case["split"] for case in cases) == SPLITS
            and all(sum(case["pair_id"] == pair for case in cases) == 8 for pair in assigned),
            "QA population or opaque IDs incomplete")
    cases.sort(key=lambda case: case["opaque_id"])
    case_by_id = {case["opaque_id"]: case for case in cases}
    remaining = sorted(case_by_id, key=lambda ident: hashlib.sha256(
        ("qwen-v2-listening-order-v1|" + PINS["manifest_sha256"] + "|" + ident).encode()
    ).hexdigest())
    queue: list[str] = []
    while remaining:
        previous = case_by_id[queue[-1]]["pair_id"] if queue else None
        chosen = next((i for i, ident in enumerate(remaining)
                       if case_by_id[ident]["pair_id"] != previous), None)
        require(chosen is not None, "cannot keep matched pair clips nonadjacent")
        queue.append(remaining.pop(chosen))
    require(len(queue) == 480 and all(case_by_id[a]["pair_id"] != case_by_id[b]["pair_id"]
                                      for a, b in zip(queue, queue[1:])),
            "listening queue pair adjacency")
    provenance = {**PINS, "protocol_sha256": sha(PROTOCOL),
                  "generator_sha256": sha(Path(__file__))}
    private = {"status": "prospective_metadata_only_no_asr_or_listening_outcomes",
               "scope": "Internal case map: do not display reference/paths to blind listeners",
               "provenance": provenance, "planned_clips": 480,
               "cases": cases}
    public = {"status": "prospective_blinded_queue_no_audio_or_outcomes",
              "scope": "Only opaque IDs and order may be shown to listeners",
              "provenance": provenance, "planned_clips": 480,
              "opaque_ids_in_order": queue}
    return private, public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    private, public = build()
    if args.validate_only:
        print(json.dumps({"status": "validated_metadata_only_no_asr_or_listening",
                          "cases": len(private["cases"]),
                          "queue": len(public["opaque_ids_in_order"]),
                          "manifest_sha256": PINS["manifest_sha256"]}, indent=2))
        return
    require(not PRIVATE.exists() and not QUEUE.exists(), "QA plan already exists")
    with PRIVATE.open("x") as stream:
        json.dump(private, stream, indent=2)
        stream.write("\n")
    with QUEUE.open("x") as stream:
        json.dump(public, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "wrote_metadata_only_qa_plan",
                      "private_sha256": sha(PRIVATE), "queue_sha256": sha(QUEUE)}, indent=2))


if __name__ == "__main__":
    main()
