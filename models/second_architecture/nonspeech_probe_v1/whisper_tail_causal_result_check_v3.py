"""Independent read-only check of the band-noise Whisper tail-state attempt.

This verifier uses only stdlib, NumPy, the frozen JSON plan, and saved evidence.
It never imports the candidate worker/auditor or loads model weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs/whisper_tail_causal_v3"
V4 = HERE / "runs/natural_tail_waveform_v1"
PLAN = HERE / "WHISPER_TAIL_CAUSAL_PLAN_V3.json"
MODELS = ("whisper_multilingual",)
V1_PLAN = HERE / "WHISPER_TAIL_CAUSAL_PLAN_V1.json"
V1_AUDIT = HERE / "runs/whisper_tail_causal_v1/whisper_multilingual_attempt_001_audit.json"
V1_REVIEW = HERE / "WHISPER_TAIL_CAUSAL_V1_INDEPENDENT_RESULT_REVIEW.md"
V1_INVENTORY = HERE / "WHISPER_TAIL_CAUSAL_V1_RAW_FILE_INVENTORY.json"
V2_AUDIT = HERE / "runs/whisper_tail_causal_v2/whisper_multilingual_attempt_001_audit.json"
V2_HISTORICAL = HERE / "runs/whisper_tail_causal_v2/whisper_multilingual_attempt_001_historical_audit.json"
V2_REVIEW = HERE / "WHISPER_TAIL_CAUSAL_V2_INDEPENDENT_RESULT_REVIEW.md"
V2_INVENTORY = HERE / "WHISPER_TAIL_CAUSAL_V2_RAW_FILE_INVENTORY.json"
ADMISSION = HERE / "WHISPER_TAIL_CAUSAL_MULTILINGUAL_ADMISSION_V3.json"
GO = HERE / "WHISPER_TAIL_CAUSAL_MULTILINGUAL_GO_V3.json"
EXECUTABLE_REVIEW = HERE / "WHISPER_TAIL_CAUSAL_EXECUTABLE_REVIEW_V3.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sidecar(attempt: Path, ref: dict) -> np.ndarray:
    name = ref["file"]
    check(name == Path(name).name, "nonlocal sidecar name")
    path = attempt / "sidecars" / name
    check(path.is_file() and not path.is_symlink() and sha(path) == ref["sha256"],
          f"sidecar hash changed: {name}")
    value = np.load(path, allow_pickle=False)
    check(list(value.shape) == ref["shape"] and str(value.dtype) == ref["dtype"],
          f"sidecar geometry changed: {name}")
    return value


def review_one(model: str, frozen: dict) -> dict:
    attempt = RUNS / f"{model}_attempt_001"
    receipt = load(RUNS / f"{model}_attempt_001_audit.json")
    parent = load(attempt / "parent_result.json")
    admission = load(ADMISSION)
    go = load(GO)
    check(admission["plan_sha256"] == sha(PLAN)
          and go["plan_sha256"] == sha(PLAN)
          and go["admission_sha256"] == sha(ADMISSION)
          and go["review_sha256"] == sha(EXECUTABLE_REVIEW)
          and go["source_identity"] == admission["source_identity"]
          and parent["source_identity"] == admission["source_identity"]
          and parent["admission_sha256"] == sha(ADMISSION)
          and parent["go_sha256"] == sha(GO), "source/admission/GO binding")
    check(all(sha(HERE / name) == digest
              for name, digest in admission["source_identity"].items()),
          "source file bytes changed")
    plan_unit = next(x for x in frozen["units"] if x["model"] == model)
    planned = [{"type": "baseline", "variant": v} for v in ("zero", "noise")]
    planned += [{"type": "arm", **x} for x in frozen["arms"] if x["model"] == model]
    records = [json.loads(x) for x in (attempt / "records.jsonl").read_text().splitlines()]
    check(len(planned) == 56 and len(records) == 57
          and records[-1] == {"type": "complete", "calls": 56}, "call census")
    check(receipt["status"] == "PASS" and receipt["calls"] == 56
          and parent["status"] == "completed_pending_saved_audit", "attempt status")
    check(parent["monitor"]["worker_returncode"] == 0
          and parent["monitor"]["worker_stop_reason"] == "normal_exit"
          and parent["monitor"]["worker_monitor_error"] is None, "worker status")
    check(parent["whole_elapsed_seconds"] < 1800
          and parent["monitor"]["worker_peak_tree_rss_bytes"] < 4 * 1024**3
          and parent["evidence_bytes_before_result"] < 1024**3, "resource cap")
    check(parent["host_samples_count"] == receipt["host_samples"], "host sample count")
    samples = [json.loads(x) for x in (attempt / "host_samples.jsonl").read_text().splitlines()]
    check(sha(attempt / "host_samples.jsonl") == parent["host_samples_sha256"]
          and len(samples) == parent["host_samples_count"]
          and all(x["battery_percent"] > 70 and x["lid_open"] is True
                  and x["thermal_ok"] is True for x in samples),
          "host samples")
    check(samples[0]["battery_percent"] >= 80
          and all(b["monotonic_seconds"] - a["monotonic_seconds"] <= 15
                  for a, b in zip(samples, samples[1:])), "battery start/cadence")
    check(samples[0]["monotonic_seconds"] >= parent["start_monotonic"]
          and parent["host_monitor_end_monotonic"]
          - samples[-1]["monotonic_seconds"] <= 15, "host sample coverage")
    check(max([x["monotonic_seconds"] for x in samples[1:]]
              + [parent["host_monitor_end_monotonic"]])
          <= parent["deadline_monotonic"], "host deadline")
    check(sha(attempt / "worker_rss_samples.jsonl")
          == parent["monitor"]["worker_rss_sample_log_sha256"]
          and sha(attempt / "post_power_log.txt") == parent["post_power_log_sha256"],
          "monitor hashes")
    rss = [json.loads(x) for x in (attempt / "worker_rss_samples.jsonl").read_text().splitlines()]
    check(len(rss) == parent["monitor"]["worker_rss_samples"]
          and max(x["tree_rss_bytes"] for x in rss)
          == parent["monitor"]["worker_peak_tree_rss_bytes"]
          and all(0 < x["tree_rss_bytes"] < 4 * 1024**3
                  and x["evidence_bytes"] < 1024**3 for x in rss), "RSS samples")
    baseline: dict[str, dict] = {}
    states: dict[str, dict[str, np.ndarray]] = {}
    recomputed = []
    complete_count = 0
    referenced: set[str] = set()
    for i, (wanted, row) in enumerate(zip(planned, records[:-1], strict=True), 1):
        check(row["index"] == i and row["plan"] == wanted, f"plan row {i}")
        output = row["output"]["output"]
        check(output["sequence_ids"] == output["prefix_ids"] + output["generated_ids"],
              f"sequence concat {i}")
        complete = output["termination"] == "eos" and output["eos_seen"] is True \
            and output["censored"] is False
        censored = output["termination"] == "length_censored" \
            and output["censored"] is True and output["eos_seen"] is False \
            and len(output["generated_ids"]) == 128
        check(complete or censored, f"termination proof {i}")
        complete_count += int(complete)
        refs = row["output"]["sidecars"]
        check(len(refs) == 5, f"result sidecar census {i}")
        arrays = {key: sidecar(attempt, ref) for key, ref in refs.items()}
        referenced.update(ref["file"] for ref in refs.values())
        if wanted["type"] == "baseline":
            variant = wanted["variant"]
            expected = plan_unit[variant]
            check(output["sequence_ids"] == expected["sequence_ids"]
                  and output["prefix_ids"] == expected["prefix_ids"],
                  f"V4 baseline tokens {variant}")
            for key in ("mel", "mask", "first_raw_logits", "first_processed_scores"):
                v4_ref = expected[key]
                old = V4 / f"{model}_attempt_v4/sidecars/{v4_ref['file']}"
                check(sha(old) == v4_ref["sha256"]
                      and np.array_equal(arrays[key], np.load(old, allow_pickle=False)),
                      f"V4 baseline array {variant}:{key}")
            baseline[variant] = {"tokens": output["sequence_ids"],
                                 "prefix": output["prefix_ids"],
                                 "detection": output["language_detection"],
                                 "mel": arrays["mel"], "mask": arrays["mask"],
                                 "raw": arrays["first_raw_logits"]}
            states[variant] = {key: sidecar(attempt, ref)
                               for key, ref in row["state_refs"].items()}
            referenced.update(ref["file"] for ref in row["state_refs"].values())
        else:
            variant, donor = wanted["recipient"], wanted["donor"]
            check(np.array_equal(arrays["mel"], baseline[variant]["mel"])
                  and np.array_equal(arrays["mask"], baseline[variant]["mask"]),
                  f"arm preprocessing {i}")
            check(output["prefix_ids"] == baseline[variant]["prefix"]
                  and output["language_detection"] == baseline[variant]["detection"],
                  f"arm detection {i}")
            patch = row["patch"]
            native = sidecar(attempt, patch["native_ref"])
            edited = sidecar(attempt, patch["edited_ref"])
            referenced.update((patch["native_ref"]["file"], patch["edited_ref"]["file"]))
            recipient_state = states[variant][wanted["site"]]
            donor_state = states[donor][wanted["site"]]
            check(np.array_equal(native, recipient_state), f"native state {i}")
            kind = wanted["kind"]
            selected = wanted["target_rows"] if kind in (
                "target_transfer", "identity_sham", "full_donor") else wanted["source_rows"]
            expected_state = recipient_state.copy()
            if kind in ("target_transfer", "full_donor", "source_random"):
                expected_state[selected] = donor_state[selected]
            elif kind == "norm_matched_shifted_delta":
                expected_state[selected] += donor_state[wanted["target_rows"]] - \
                    recipient_state[wanted["target_rows"]]
            else:
                check(kind == "identity_sham", f"kind {i}")
            check(np.array_equal(edited, expected_state), f"edited state {i}")
            check(patch["hook_calls"] == 1 and patch["selected_rows"] == selected,
                  f"hook receipt {i}")
            recomputed.append({"index": i, "site": wanted["site"],
                               "region": wanted["region"], "kind": kind,
                               "recipient": variant, "donor": donor,
                               "matches_recipient": complete and output["sequence_ids"]
                               == baseline[variant]["tokens"],
                               "matches_donor": complete and output["sequence_ids"]
                               == baseline[donor]["tokens"],
                               "raw_matches_donor": np.array_equal(
                                   arrays["first_raw_logits"], baseline[donor]["raw"])})
    check(len(referenced) == 394
          and {x.name for x in (attempt / "sidecars").glob("*.npy")} == referenced,
          "394-sidecar evidence census")
    check(len(receipt["outcomes"]) == 54, "receipt outcomes")
    for own, original in zip(recomputed, receipt["outcomes"], strict=True):
        check(all(own[key] == original[key] for key in own if key != "raw_matches_donor"),
              f"receipt outcome {own['index']}")
    target_rows = [x for x in recomputed if x["kind"] == "target_transfer"]
    by = {(x["site"], x["region"], x["recipient"], x["kind"]): x
          for x in recomputed}
    qualified = {}
    for x in target_rows:
        key = (x["site"], x["region"], x["recipient"])
        identity = by[(*key, "identity_sham")]
        random = by[(*key, "source_random")]
        shifted = by[(*key, "norm_matched_shifted_delta")]
        full = by[(x["site"], "all_encoder_rows", x["recipient"], "full_donor")]
        qualified[x["index"]] = all((identity["matches_recipient"],
                                     random["matches_recipient"],
                                     shifted["matches_recipient"],
                                     full["matches_donor"],
                                     full["raw_matches_donor"]))
    check(qualified == {x["target_index"]: x["control_qualified"]
                        for x in receipt["target_qualifications"]},
          "receipt control qualifications")
    target_matches = [{**x, "control_qualified": qualified[x["index"]]}
                      for x in target_rows if x["matches_donor"] and qualified[x["index"]]]
    primary = frozen["primary_cross_family_cells"]
    check(len(primary) == 3 and primary == [
        {"site": "post_conv_pos", "region": "boundary", "recipient": "noise", "donor": "zero"},
        {"site": "post_conv_pos", "region": "tail_interior", "recipient": "noise", "donor": "zero"},
        {"site": "post_block0", "region": "boundary", "recipient": "noise", "donor": "zero"}],
        "predeclared V1/V2 rescue cells changed")
    independent_primary = []
    for cell in primary:
        target = next(x for x in target_rows if all(x[k] == cell[k] for k in cell))
        independent_primary.append({**cell, "target_index": target["index"],
                                    "transferred": bool(qualified[target["index"]]
                                                        and target["matches_donor"])})
    check(independent_primary == receipt["primary_cross_family_cells"]
          and receipt["primary_cross_family_count"] == sum(x["transferred"] for x in independent_primary)
          and receipt["all_three_transfer"] is all(x["transferred"] for x in independent_primary),
          "cross-family receipt does not match raw controls")
    files = []
    for path in sorted(attempt.rglob("*")):
        if path.is_file():
            check(not path.is_symlink(), "symlink raw evidence")
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": sha(path)})
    return {"model": model, "status": "independent_saved_checks_pass",
            "raw_file_count": len(files), "raw_files": files,
            "audit_sha256": sha(RUNS / f"{model}_attempt_001_audit.json"),
            "calls": len(planned), "sidecars": len(referenced),
            "complete_outputs": complete_count, "host_samples": len(samples),
            "elapsed_seconds": parent["whole_elapsed_seconds"],
            "peak_tree_rss_bytes": parent["monitor"]["worker_peak_tree_rss_bytes"],
            "baseline_equal": baseline["zero"]["tokens"] == baseline["noise"]["tokens"],
            "primary_cross_family_cells": independent_primary,
            "qualified_target_matches": target_matches,
            "target_results": [{**x, "control_qualified": qualified[x["index"]]}
                               for x in target_rows]}


def main() -> None:
    frozen = load(PLAN)
    check(frozen["schema"] == "whisper_tail_causal_plan_v3"
          and frozen["generation_calls_total"] == 56, "frozen plan")
    v1 = load(V1_PLAN)
    check(sha(V1_AUDIT) == frozen["inputs"]["prior_v1_audit_sha256"]
          and sha(V1_REVIEW) == frozen["inputs"]["prior_v1_review_sha256"]
          and sha(V1_INVENTORY) == frozen["inputs"]["prior_v1_inventory_sha256"]
          and load(V1_AUDIT)["status"] == "PASS", "V1 source binding changed")
    check(sha(V2_AUDIT) == frozen["inputs"]["prior_v2_audit_sha256"]
          and sha(V2_HISTORICAL) == frozen["inputs"]["prior_v2_historical_sha256"]
          and sha(V2_REVIEW) == frozen["inputs"]["prior_v2_review_sha256"]
          and sha(V2_INVENTORY) == frozen["inputs"]["prior_v2_inventory_sha256"]
          and load(V2_AUDIT)["full_replication"] is True
          and load(V2_HISTORICAL)["status"] == "PASS", "V2 source binding changed")
    check(frozen["sites"] == v1["sites"] and frozen["regions"] == v1["regions"]
          and frozen["units"][0]["random_source_rows"]
          == next(x for x in v1["units"] if x["model"] == "whisper_multilingual")["random_source_rows"],
          "V1 sites/rows/control samples changed")
    check(frozen["arms"] == [x for x in v1["arms"]
                             if x["model"] == "whisper_multilingual"],
          "V1 intervention arm schedule changed")
    results = [review_one(model, frozen) for model in MODELS]
    print(json.dumps({"schema": "whisper_tail_causal_independent_result_v3",
                      "status": "independent_saved_checks_pass",
                      "plan_sha256": sha(PLAN), "models": results}, sort_keys=True))


if __name__ == "__main__":
    main()
