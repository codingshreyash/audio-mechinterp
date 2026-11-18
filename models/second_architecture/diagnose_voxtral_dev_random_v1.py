"""Posthoc descriptive readout of the audited Voxtral development controls.

No model forward, threshold change, locked-test access, or new effect gate.
The frozen development audit remains the decision authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/voxtral_dev_transfer_v1/20261118T1016Z"
RESULT = RUN / "result.json"
AUDIT = RUN / "audit.json"
OUTPUT = HERE / "VOXTRAL_DEV_RANDOM_CONTROL_DIAGNOSTIC_V1.json"
RESULT_SHA = "1c43c22720237a703441854f4e8a7fb7f41fad62bacb1be7f72d4958452e6edd"
AUDIT_SHA = "e79a0ae2fb09526be4830a9be1fa975459f7b9f61099e373c6113261a5490467"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def load() -> tuple[dict, dict, dict]:
    require(sha(RESULT) == RESULT_SHA and sha(AUDIT) == AUDIT_SHA,
            "Audited development result changed")
    result, audit = json.loads(RESULT.read_text()), json.loads(AUDIT.read_text())
    require(audit["status"] == "passed_independent_cpu_only_development_audit"
            and audit["result_sha256"] == RESULT_SHA
            and result["summary"]["development_advance_gate"] is False
            and audit["summary"] == result["summary"]
            and result["directions"] == 96 and result["generations"] == 4800,
            "Frozen development audit or denominator changed")
    cases = defaultdict(dict)
    for ordinal in range(1, 49):
        checkpoint_path = RUN / f"record_{ordinal:02d}.checkpoint.json"
        require(sha(checkpoint_path) == result["checkpoint_sha256"][str(ordinal)],
                "Checkpoint hash mismatch")
        checkpoint = json.loads(checkpoint_path.read_text())
        attempt_path = RUN / checkpoint["attempt_file"]
        require(sha(attempt_path) == checkpoint["attempt_sha256"], "Attempt hash mismatch")
        for line in attempt_path.read_text().splitlines():
            row = json.loads(line)
            if row["type"] != "generation":
                continue
            key = (row["pair_id"], row["engine"], row["phrasing_id"], row["recipient"])
            arm = (row["site"], row["kind"])
            require(arm not in cases[key], "Duplicate generation arm")
            cases[key][arm] = row
    require(len(cases) == 96, "Missing planned directions")
    return result, audit, cases


def summarize_site(cases: dict, site: str) -> dict:
    eligible = []
    all_random = Counter()
    for key, arms in cases.items():
        natural = arms[(None, f"feature:native_{key[3]}")]
        blank = arms[(None, "blank_baseline")]
        random = arms[(site, "patch:loss:random")]
        whole = arms[(site, "patch:loss:whole")]
        all_random[random["classification"]] += 1
        if natural["classification"] == "recipient" and blank["classification"] != "recipient":
            eligible.append((natural, random, whole))
    require(len(eligible) == 87, "Eligible count changed")
    groups = {}
    for label, rows in (
        ("random_valid_loss", [x for x in eligible if x[1]["classification"] == "other"
                               and x[1]["terminal_eos"]]),
        ("random_recipient", [x for x in eligible if x[1]["classification"] == "recipient"
                              and x[1]["terminal_eos"]]),
    ):
        groups[label] = {
            "count_of_96": len(rows),
            "raw_output_counts": dict(sorted(Counter(x[1]["raw_decoded"] for x in rows).items())),
            "median_clean_recipient_minus_opposite": median(
                x[0]["recipient_minus_opposite"] for x in rows) if rows else None,
            "median_random_effective_bf16_norm": median(
                x[1]["edit"]["effective_bf16_norm"] for x in rows) if rows else None,
            "median_main_effective_bf16_norm": median(
                x[2]["edit"]["effective_bf16_norm"] for x in rows) if rows else None,
            "median_random_to_main_effective_norm_ratio": median(
                x[1]["edit"]["effective_bf16_norm"] /
                x[2]["edit"]["effective_bf16_norm"] for x in rows) if rows else None,
        }
    return {"eligible_of_96": len(eligible), "all_random_classes_of_96": dict(all_random),
            "groups": groups}


def main() -> None:
    result, audit, cases = load()
    sites = {site: summarize_site(cases, site) for site in ("encoder31", "projector")}
    require(sites["encoder31"]["groups"]["random_valid_loss"]["count_of_96"]
            == audit["summary"]["sites"]["encoder31"]["loss"]["control_success_of_96"]["random"]
            and sites["projector"]["groups"]["random_valid_loss"]["count_of_96"]
            == audit["summary"]["sites"]["projector"]["loss"]["control_success_of_96"]["random"],
            "Random-control counts differ from frozen audit")
    report = {"status": "posthoc_dev_random_control_description_only",
              "result_sha256": sha(RESULT), "audit_sha256": sha(AUDIT),
              "planned_directions": result["directions"],
              "development_advance_gate": False, "sites": sites,
              "limits": "Constructed hidden-state edits; descriptive, not a new effect or security gate"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output_sha256": sha(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
