"""Read-only v1 Stage-2 gate check from the frozen 32-clip Voxtral FIT pilot."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
INDEX = HERE / "voxtral_corpus_v1_index.json"
PILOT = HERE / "runs/voxtral_corpus_v1_fit_pilot/20261118T082335918194Z/result.json"
OUTPUT = HERE / "VOXTRAL_FIT_STAGE2_GATE_V1.json"
INDEX_SHA256 = "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971"
PILOT_SHA256 = "a4700406fb5deacf947d2e32fe7f61dc4c0ed8d9a47e2816fb06bf37afbe21d3"
PILOT_IDS = ("animal_06", "color_03", "nature_00", "object_02")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_template_records(index: dict) -> list[tuple]:
    rows = [r for r in index["rows"] if r["pilot_fit"]]
    if len(rows) != 16 or set(index["pilot_pair_ids"]) != set(PILOT_IDS):
        raise ValueError("Frozen pilot rows changed")
    by_pair_voice = defaultdict(list)
    for row in rows:
        if row["split"] != "fit_pool" or not row["map_fit"]:
            raise ValueError("Non-fit row in pilot")
        by_pair_voice[(row["pair_id"], row["engine"])].append(row)
    if len(by_pair_voice) != 8 or any(len(v) != 2 for v in by_pair_voice.values()):
        raise ValueError("Expected 4 pairs × 2 voices × 2 templates")
    selected = []
    for key, group in sorted(by_pair_voice.items()):
        row = min(group, key=lambda r: r["phrasing_id"])
        selected.append((row["pair_id"], row["engine"], row["phrasing_id"], tuple(row["words"])))
    return selected


def evaluate(index: dict, pilot: dict) -> dict:
    if not pilot["fit_capability_gate_pass"] or len(pilot["records"]) != 32:
        raise ValueError("Pilot result is not the completed 32-clip capability pass")
    records = first_template_records(index)
    by_key = {}
    for record in pilot["records"]:
        key = (record["pair_id"], record["engine"], record["phrasing_id"], record["expected_word"])
        if key in by_key:
            raise ValueError("Duplicate pilot result")
        by_key[key] = record
    if len(by_key) != 32:
        raise ValueError("Pilot result missing records")
    selected = []
    failures = []
    for pair_id, engine, phrasing_id, words in records:
        endpoints = []
        for word in words:
            key = (pair_id, engine, phrasing_id, word)
            if key not in by_key:
                raise ValueError(f"Missing required natural endpoint: {key}")
            item = by_key[key]
            passed = item["classification"]["exact_expected_word_and_eos"]
            endpoint = {
                "word": word, "passed": passed,
                "raw_decoded": item["raw_decoded"],
                "generated_token_ids": item["generated_token_ids"],
                "stopped_on_eos": item["stopped_on_eos"],
            }
            endpoints.append(endpoint)
            if not passed:
                failures.append({"pair_id": pair_id, "engine": engine,
                                 "phrasing_id": phrasing_id, **endpoint})
        selected.append({"pair_id": pair_id, "engine": engine,
                         "phrasing_id": phrasing_id, "endpoints": endpoints,
                         "both_natural_endpoints_pass": all(x["passed"] for x in endpoints)})
    return {
        "status": "gate_failed" if failures else "gate_passed",
        "v1_capability_gate_passed": True,
        "v1_stage2_all_eight_records_natural_endpoints_pass": not failures,
        "stage2_pair_records": 8,
        "required_natural_endpoints": 16,
        "passing_natural_endpoints": 16 - len(failures),
        "failures": failures,
        "selected_records": selected,
        "consequence": "Under frozen v1 protocol, do not run feature hybrids or hidden-state interventions" if failures else "Stage-2 intervention is permitted under v1",
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if sha(INDEX) != INDEX_SHA256 or sha(PILOT) != PILOT_SHA256:
        raise ValueError("Frozen index or pilot result SHA-256 mismatch")
    index, pilot = json.loads(INDEX.read_text()), json.loads(PILOT.read_text())
    result = {
        "schema_version": "voxtral_fit_stage2_gate_v1",
        "index_sha256": INDEX_SHA256, "pilot_result_sha256": PILOT_SHA256,
        "gate_script_sha256": sha(Path(__file__)), **evaluate(index, pilot),
    }
    with OUTPUT.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"{result['status']}: {result['passing_natural_endpoints']}/16 natural endpoints; {OUTPUT}")


if __name__ == "__main__":
    main()
