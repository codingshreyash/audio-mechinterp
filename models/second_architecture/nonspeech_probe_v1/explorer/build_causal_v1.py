#!/usr/bin/env python3
"""Deterministically render the audited attempt_002 causal record as an offline page.

Reads saved evidence only. No model inference, package downloads, or evidence edits.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RECORDS = RUNS / "causal_whisper_v1/attempt_002/worker_output/records.jsonl"
SUMMARY = RUNS / "causal_whisper_v1/attempt_002_independent_summary.json"
MEMO = ROOT / "INDEPENDENT_CAUSAL_ATTEMPT_002_RESULT_REVIEW_20261119.md"
MANIFEST = RUNS / "render_20261119_0740Z/manifest.json"
EXPECTED = {
    "records_sha256": "dbacb49bcb1e158b312ee2e76235c2955b28a75dc05b8af2566826b8dd65753f",
    "independent_summary_sha256": "6974bc4c7fd6f8f38df107522669c6b5224e056ef0b614a010cce774f00a331b",
    "independent_memo_sha256": "07e581427d71d7e0e1e5ecd301f532af731776526df4409a34f9c1aeba638e15",
    "manifest_sha256": "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90",
}
STAGE_ARMS = ("no_hook", "read_only", "self_detect", "self_transcribe", "self_both", "donor_detect", "donor_transcribe", "donor_both", "random_1_both", "random_2_both")
STAGE_B_ARMS = ("no_hook", "read_only", "self_both", "donor_both", "active_both", "displaced_both", "random_1_both", "random_2_both")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(row: dict) -> dict:
    p, o = row["plan_row"], row["output"]
    hooks = []
    for phase, sites in row["hook"].items():
        for site, detail in sites.items():
            if detail["edit_l2"] or p["arm"] in ("read_only", "self_detect", "self_transcribe", "self_both"):
                hooks.append({"phase": phase, "site": site, "support_rows": detail["support_rows"],
                              "edit_l2": detail["edit_l2"], "hook_calls": detail["hook_calls"]})
    return {
        "call_index": row["call_index"], "call_name": row["call_name"],
        "stage": p["stage"], "arm": p["arm"], "case_id": p["case_id"],
        "donor_id": p["donor_id"], "site": p["site"], "active_rows": p["active_rows"],
        "transcript": o["transcript"], "text_with_special_tokens": o["text_with_special_tokens"],
        "prefix_ids": o["prefix_ids"], "generated_ids": o["generated_ids"],
        "sequence_ids": o["sequence_ids"], "steps": o["actual_generation_steps"],
        "termination": o["termination"], "eos_seen": o["eos_seen"], "censored": o["censored"],
        "chosen_language_ids": (o["language_detection"] or {}).get("chosen_language_ids"),
        "language_logits_sha256": (o["language_detection"] or {}).get("raw_logits_sha256"),
        "first_raw_logits_sha256": o["score_rows"][0]["raw_logits_sha256"],
        "parity": row["parity"], "hooks": hooks,
    }


def build_data() -> dict:
    paths = {"records_sha256": RECORDS, "independent_summary_sha256": SUMMARY,
             "independent_memo_sha256": MEMO, "manifest_sha256": MANIFEST}
    for key, path in paths.items():
        actual = sha(path)
        if actual != EXPECTED[key]:
            raise ValueError(f"Evidence hash mismatch: {path}: {actual} != {EXPECTED[key]}")
    summary = json.loads(SUMMARY.read_text())
    assert summary["status"] == "GO_SAVED_EVIDENCE_WITH_CENSORING"
    assert summary["records_sha256"] == EXPECTED["records_sha256"]
    assert summary["call_count"] == 328 and not summary["parity_errors"]
    manifest = json.loads(MANIFEST.read_text())
    assets = {}
    for row in manifest["rows"]:
        wav = RUNS / "render_20261119_0740Z" / row["wav"]
        assert sha(wav) == row["wav_sha256"]
        assets[row["case_id"]] = {"wav_sha256": row["wav_sha256"],
                                   "audio": "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode("ascii")}
    assert len(assets) == 20

    records = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    assert len(records) == 330 and records[0]["type"] == "start" and records[-1]["type"] == "complete"
    forwards = [x for x in records if x["type"] == "forward"]
    assert len(forwards) == 328 and [x["call_index"] for x in forwards] == list(range(1, 329))
    assert all(x["plan_row"]["case_id"] in assets and x["plan_row"]["donor_id"] in assets for x in forwards)
    # Saved parity fields are comparison flags, not parity failures. A false
    # recipient/donor match is expected for an effective intervention.
    by_case = {key: {"audio": value["audio"], "wav_sha256": value["wav_sha256"], "stages": {"A": {}, "B": {}}}
               for key, value in sorted(assets.items()) if not key.endswith("_silence")}
    donors = {"d02_silence": {}, "d08_silence": {}}
    for row in forwards:
        p = row["plan_row"]
        rec = compact(row)
        if p["stage"] == "donor":
            assert p["arm"] not in donors[p["case_id"]]
            donors[p["case_id"]][p["arm"]] = rec
        else:
            stage = by_case[p["case_id"]]["stages"][p["stage"]]
            assert p["arm"] not in stage
            stage[p["arm"]] = rec
    assert len(by_case) == 18 and all(set(d) == {"no_hook", "read_only"} for d in donors.values())
    assert all(set(c["stages"]["A"]) == set(STAGE_ARMS) and set(c["stages"]["B"]) == set(STAGE_B_ARMS) for c in by_case.values())
    for case_id, case in by_case.items():
        donor_id = case_id[:3] + "_silence"
        case["donor_id"] = donor_id
        case["informative"] = case["stages"]["A"]["no_hook"]["sequence_ids"] != donors[donor_id]["no_hook"]["sequence_ids"]
        case["mel_padding"] = summary["mel_padding"][case_id]
        if case_id in summary["beep_preblock0_padding"]:
            case["encoder_padding"] = summary["beep_preblock0_padding"][case_id]
        case["detect_shift"] = next(x for x in summary["donor_detect_language_logit_shift"] if x["case"] == case_id)
        assert case["stages"]["A"]["no_hook"]["sequence_ids"] == case["stages"]["B"]["no_hook"]["sequence_ids"]
    informative = [k for k, v in by_case.items() if v["informative"]]
    assert informative == summary["informative_cases"] and len(informative) == 4
    censored = [x["call_name"] for x in forwards if x["output"]["censored"]]
    assert censored == summary["censored_calls"] and len(censored) == 5
    counts = {"A": {}, "B": {}}
    for stage, arms in (("A", STAGE_ARMS), ("B", STAGE_B_ARMS)):
        for arm in arms:
            rows = [v["stages"][stage][arm] for v in by_case.values()]
            counts[stage][arm] = {
                "censored": sum(x["censored"] for x in rows),
                "changed_full_sequence": sum(x["sequence_ids"] != by_case[x["case_id"]]["stages"][stage]["no_hook"]["sequence_ids"] for x in rows),
                "matched_donor": sum(x["sequence_ids"] == donors[x["donor_id"]]["no_hook"]["sequence_ids"] for x in rows),
            }
            if arm in summary["arm_counts"][stage]:
                assert counts[stage][arm] == summary["arm_counts"][stage][arm]
    assert all(by_case[k]["stages"]["A"]["donor_transcribe"]["sequence_ids"] == donors[by_case[k]["donor_id"]]["no_hook"]["sequence_ids"] for k in informative)
    return {"schema": "nonspeech_causal_viewer_v1", "model": "openai/whisper-tiny",
            "revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
            "hashes": EXPECTED, "counts": counts, "censored_calls": censored,
            "cases": by_case, "donors": donors,
            "audio_donors": {key: assets[key] for key in donors},
            "informative_cases": informative}


def render(data: dict) -> str:
    template = (HERE / "causal_v1_template.html").read_text()
    assert template.count("__EMBEDDED_CAUSAL_JSON__") == 1
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return template.replace("__EMBEDDED_CAUSAL_JSON__", payload.replace("<", "\\u003c"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if checked-in HTML is stale")
    args = parser.parse_args()
    output = HERE / "causal_v1.html"
    built = render(build_data())
    if args.check:
        assert output.read_text() == built, "Checked-in HTML differs from deterministic build"
        print("PASS: pinned causal evidence and deterministic HTML")
    else:
        output.write_text(built)
        print(output)


if __name__ == "__main__":
    main()
