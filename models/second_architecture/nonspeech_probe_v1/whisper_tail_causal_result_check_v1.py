"""Independent read-only check of the two saved Whisper tail-state attempts.

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
RUNS = HERE / "runs/whisper_tail_causal_v1"
V4 = HERE / "runs/natural_tail_waveform_v1"
PLAN = HERE / "WHISPER_TAIL_CAUSAL_PLAN_V1.json"
MODELS = ("whisper_multilingual", "whisper_english")


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
            "qualified_target_matches": target_matches,
            "target_results": [{**x, "control_qualified": qualified[x["index"]]}
                               for x in target_rows]}


def main() -> None:
    frozen = load(PLAN)
    check(frozen["schema"] == "whisper_tail_causal_plan_v1"
          and frozen["generation_calls_total"] == 112, "frozen plan")
    results = [review_one(model, frozen) for model in MODELS]
    print(json.dumps({"schema": "whisper_tail_causal_independent_result_v1",
                      "status": "independent_saved_checks_pass",
                      "plan_sha256": sha(PLAN), "models": results}, sort_keys=True))


if __name__ == "__main__":
    main()
