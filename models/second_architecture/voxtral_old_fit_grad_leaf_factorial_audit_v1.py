"""Independent CPU saved-data audit for the old-FIT 2x2 factorial diagnostic.

This rehashes a completed tensor sidecar and checks saved relationships. It
cannot independently attest that the model forwards actually executed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

import voxtral_old_fit_grad_mode_parity_audit_v1 as base


STATUS = "completed_old_fit_fp32_grad_leaf_factorial_diagnostic_only"
META_STATUS = "started_old_fit_fp32_grad_leaf_factorial_diagnostic_only"
AUDIT_STATUS = "audited_old_fit_fp32_grad_leaf_factorial_saved_data_only"
SCOPE = "one_old_fit_case_grad_leaf_factorial_only_no_vjp_no_security_claim"
SCHEMA = "voxtral_old_fit_grad_leaf_factorial_tensors_v1"
CONDITIONS = ("N0", "N1", "G0", "G1")
ORDERS = {"encoder31": ("N0", "G1", "G0", "N1", "N1", "G0", "G1", "N0"),
          "projector": ("G1", "N0", "N1", "G0", "G0", "N1", "N0", "G1")}
COUNTS = {"full_forwards": 16, "direct_forwards": 16, "vjps": 0, "generations": 0}
V1_FAILED_MANIFEST_SHA = "0a2335dcab3ce5bb4cd66f5d0095074058c37537743b1c42764da06438811280"
SPEC = base.REAL


def full_schedule() -> list[tuple[str, str, int]]:
    rows = []
    for site in ("encoder31", "projector"):
        seen = {condition: 0 for condition in CONDITIONS}
        for condition in ORDERS[site]:
            rows.append((site, condition, seen[condition]))
            seen[condition] += 1
    return rows


def direct_schedule() -> list[tuple[str, str, str, int]]:
    return [(site, "N0", condition, repeat)
            for site, condition, repeat in full_schedule()]


def _boundaries(row: dict[str, Any], kind: str) -> dict[str, torch.Tensor]:
    return base._boundaries(row, kind=kind)


def _checked_flags(row: dict[str, Any], condition: str, label: str) -> None:
    expected = {"grad_enabled": condition.startswith("G"),
                "leaf_requires_grad": condition.endswith("1")}
    factor = row.get("factor_observed")
    base.need(isinstance(factor, dict) and set(factor) == set(expected) and
              all(type(value) is bool for value in factor.values()) and
              factor == expected,
              f"{label}: recorded factor flags differ")
    flags = row.get("downstream_grad_flags")
    base.need(isinstance(flags, dict) and set(flags) == {"lm_input", "lm0_pre"},
              f"{label}: downstream grad flag fields differ")
    for stage in ("lm_input", "lm0_pre"):
        item = flags[stage]
        base.need(isinstance(item, dict) and set(item) == {"grad_enabled", "requires_grad"}
                  and type(item["grad_enabled"]) is bool and
                  type(item["requires_grad"]) is bool and
                  item["grad_enabled"] == expected["grad_enabled"],
                  f"{label}: downstream grad flags invalid")
    if row["kind"] == "direct":
        base.need(flags["lm_input"]["requires_grad"] == expected["leaf_requires_grad"],
                  f"{label}: direct input leaf flag differs")


def _metrics(rows: list[dict[str, Any]], *, kind: str,
             positions: list[int] | None = None) -> dict[str, Any]:
    grouped = {condition: [row for row in rows if row["mode"] == condition]
               for condition in CONDITIONS}
    base.need(all(len(grouped[c]) == 2 for c in CONDITIONS),
              "Factorial condition repeat count differs")
    names = ("lm_input",) if positions is not None else tuple(_boundaries(rows[0], kind))
    contrasts = {f"{left}_vs_{right}": {"first_bitwise_difference": None,
                                      "first_resolved_difference": None,
                                      "boundaries": {}}
                 for left, right in itertools.combinations(CONDITIONS, 2)}
    interaction = {}
    for name in names:
        values = {condition: [_boundaries(row, kind)[name] for row in grouped[condition]]
                  for condition in CONDITIONS}
        if positions is not None:
            values = {condition: [tensor[0, positions] for tensor in pair]
                      for condition, pair in values.items()}
        within = {condition: base.difference(*pair) for condition, pair in values.items()}
        threshold = max(3.0 * max(row["l2"] for row in within.values()), 1e-6)
        for left, right in itertools.combinations(CONDITIONS, 2):
            label = f"{left}_vs_{right}"
            comparisons = [base.difference(a, b)
                           for a in values[left] for b in values[right]]
            bitwise = all(row["bitwise_equal"] for row in comparisons)
            resolved = min(row["l2"] for row in comparisons) > threshold
            item = contrasts[label]
            if item["first_bitwise_difference"] is None and not bitwise:
                item["first_bitwise_difference"] = name
            if item["first_resolved_difference"] is None and resolved:
                item["first_resolved_difference"] = name
            item["boundaries"][name] = {
                "within_condition": within,
                "cross_pairs": [{"left_repeat": li, "right_repeat": ri,
                                 **comparisons[2 * li + ri]}
                                for li in range(2) for ri in range(2)],
                "all_cross_bitwise_equal": bitwise,
                "resolved_beyond_observed_repeat_floor": resolved,
                "resolved_threshold_l2": threshold,
                "all_within_condition_bitwise_stable":
                    all(row["bitwise_equal"] for row in within.values()),
            }
        # The interaction is descriptive: matching repeat indices yield two
        # signed 2x2 contrasts. A factorial claim also requires repeat stability.
        deltas = [values["G1"][i].to(torch.float64) -
                  values["G0"][i].to(torch.float64) -
                  values["N1"][i].to(torch.float64) +
                  values["N0"][i].to(torch.float64) for i in range(2)]
        interaction[name] = {
            "matched_repeat_l2": [float(torch.linalg.vector_norm(d).item()) for d in deltas],
            "matched_repeat_max_abs": [float(d.abs().max().item()) for d in deltas],
            "matched_repeat_changed_coordinates": [int(torch.count_nonzero(d).item())
                                                   for d in deltas],
            "resolved_threshold_l2": threshold,
            "all_within_condition_bitwise_stable":
                all(row["bitwise_equal"] for row in within.values()),
        }
    fixed_direct = kind != "direct" or all(
        entry["boundaries"][name]["all_cross_bitwise_equal"]
        for entry in contrasts.values() for name in ("lm_input", "lm0_input"))
    for entry in contrasts.values():
        stable = all(boundary["all_within_condition_bitwise_stable"]
                     for boundary in entry["boundaries"].values())
        entry["all_within_condition_bitwise_stable"] = stable
        entry["fixed_direct_input_bitwise_equal"] = fixed_direct
        entry["deterministic_factor_contrast_supported"] = (
            stable and fixed_direct and entry["first_resolved_difference"] is not None)
    return {"contrasts": contrasts, "interaction": interaction,
            "fixed_direct_input_bitwise_equal": fixed_direct}


def audit_payload(payload: Any, *, spec: base.ShapeSpec = SPEC) -> dict[str, Any]:
    base.need(isinstance(payload, dict) and
              set(payload) == {"schema", "sources_bf16", "batch", "audio_token_id", "calls"}
              and payload["schema"] == SCHEMA, "Tensor payload schema invalid")
    sources = payload["sources_bf16"]
    base.need(isinstance(sources, dict) and set(sources) == set(spec.source),
              "BF16 source map invalid")
    for site, shape in spec.source.items():
        source = base.float_tensor(sources[site], shape, torch.bfloat16, f"{site} source")
        base.need(base.tensor_sha(source) == spec.source_sha256[site],
                  f"{site} full BF16 source hash differs")
    base.need(payload["audio_token_id"] == spec.audio_token_id,
              "Audio token ID differs")
    batch = payload["batch"]
    base.need(isinstance(batch, dict) and set(batch) == set(spec.batch_sha256),
              "Frozen batch field set invalid")
    ids = base.integer_tensor(batch["input_ids"], (1, spec.lm_input[1]), "input_ids")
    mask = base.integer_tensor(batch["attention_mask"], (1, spec.lm_input[1]),
                               "attention_mask")
    base.need(mask.dtype == torch.int64 and bool((mask == 1).all()),
              "Frozen attention mask differs")
    base.float_tensor(batch["input_features"], spec.input_features, torch.float32,
                      "input_features")
    for name, expected in spec.batch_sha256.items():
        base.need(base.tensor_sha(batch[name]) == expected,
                  f"Frozen batch tensor hash differs: {name}")
    positions = torch.nonzero(ids[0] == spec.audio_token_id).flatten().tolist()
    base.need(len(positions) == spec.audio_count, "Audio placeholder count differs")
    others = [i for i in range(spec.lm_input[1]) if i not in set(positions)]
    calls = payload["calls"]
    base.need(isinstance(calls, list) and len(calls) == 32,
              "Exactly 32 saved calls required")
    full_keys = {"kind", "ordinal", "site", "mode", "repeat", "source_sha256",
                 "factor_observed", "downstream_grad_flags",
                 "source_post_full", "projector_full", "lm_input_full",
                 "lm0_input_full", "lm_terminal", "first_answer_logits",
                 "audio_placeholder_positions"}
    direct_keys = {"kind", "ordinal", "site", "origin_mode",
                   "factor_observed", "downstream_grad_flags",
                   "origin_full_ordinal", "mode", "repeat", "input_sha256",
                   "lm_input_full", "lm0_input_full", "lm_terminal",
                   "first_answer_logits"}
    full_groups = {site: [] for site in spec.source}
    for ordinal, (site, condition, repeat) in enumerate(full_schedule()):
        row = calls[ordinal]
        label = f"full[{ordinal}]"
        base.need(isinstance(row, dict) and set(row) == full_keys and
                  (row["kind"], row["ordinal"], row["site"], row["mode"], row["repeat"])
                  == ("full", ordinal, site, condition, repeat),
                  f"{label}: schedule/schema invalid")
        _checked_flags(row, condition, label)
        base.need(row["source_sha256"] == spec.source_sha256[site] and
                  row["audio_placeholder_positions"] == positions,
                  f"{label}: frozen source or positions differ")
        post = base.float_tensor(row["source_post_full"], spec.source[site],
                                 torch.float32, f"{label} source posthook")
        base.need(base.bitwise_equal(post, sources[site].float()),
                  f"{label}: returned source differs")
        projector = base.float_tensor(row["projector_full"], spec.projector,
                                      torch.float32, f"{label} projector")
        lm = base.float_tensor(row["lm_input_full"], spec.lm_input,
                               torch.float32, f"{label} LM input")
        lm0 = base.float_tensor(row["lm0_input_full"], spec.lm_input,
                                torch.float32, f"{label} LM0 pre")
        base.need(base.bitwise_equal(lm, lm0) and
                  base.bitwise_equal(lm[0, positions], projector),
                  f"{label}: projector/LM handoff differs")
        if site == "projector":
            base.need(base.bitwise_equal(post, projector),
                      f"{label}: projector posthook differs")
        base._stages(row, spec, label)
        full_groups[site].append(row)
    origins = {site: next(row for row in rows if row["mode"] == "N0")
               for site, rows in full_groups.items()}
    direct_groups = {site: [] for site in spec.source}
    for ordinal, (site, origin, condition, repeat) in enumerate(direct_schedule(), start=16):
        row = calls[ordinal]
        label = f"direct[{ordinal}]"
        base.need(isinstance(row, dict) and set(row) == direct_keys and
                  (row["kind"], row["ordinal"], row["site"], row["origin_mode"],
                   row["mode"], row["repeat"])
                  == ("direct", ordinal, site, origin, condition, repeat),
                  f"{label}: schedule/schema invalid")
        _checked_flags(row, condition, label)
        fixed = origins[site]["lm_input_full"]
        base.need(row["origin_full_ordinal"] == origins[site]["ordinal"] and
                  row["input_sha256"] == base.tensor_sha(fixed),
                  f"{label}: origin descriptor/hash invalid")
        direct = base.float_tensor(row["lm_input_full"], spec.lm_input,
                                   torch.float32, f"{label} LM input")
        lm0 = base.float_tensor(row["lm0_input_full"], spec.lm_input,
                                torch.float32, f"{label} LM0 pre")
        base.need(base.bitwise_equal(direct, fixed) and base.bitwise_equal(lm0, fixed),
                  f"{label}: fixed direct input differs")
        base._stages(row, spec, label)
        direct_groups[site].append(row)
    return {
        "status": AUDIT_STATUS, "scope": SCOPE, "saved_call_counts": COUNTS,
        "audio_placeholder_count": len(positions), "non_audio_token_count": len(others),
        "retained_tensor_sha256": base._retained_tensor_hashes(payload),
        "observed_grad_flags": [{"ordinal": row["ordinal"], "kind": row["kind"],
                                 "site": row["site"], "condition": row["mode"],
                                 "factor": row["factor_observed"],
                                 "downstream": row["downstream_grad_flags"]}
                                for row in calls],
        "full_model": {site: _metrics(rows, kind="full")
                       for site, rows in full_groups.items()},
        "direct_lm": {site: _metrics(rows, kind="direct")
                      for site, rows in direct_groups.items()},
        "modality_partition": {site: {
            "audio_lm_input": _metrics(rows, kind="full", positions=positions),
            "non_audio_lm_input": _metrics(rows, kind="full", positions=others),
        } for site, rows in full_groups.items()},
        "interpretation_limit":
            "One old-FIT case; two repeats give only an observed floor. "
            "No model execution or kernel identity can be inferred from saved data alone. "
            "No audio attack or security result was tested.",
    }


def audit_run(run_dir: Path, output: Path, *, spec: base.ShapeSpec = SPEC) -> dict[str, Any]:
    base.need(output.parent.resolve() == run_dir.resolve() and not output.exists(),
              "Audit output must be a new file directly in the attempt directory")
    base.need(run_dir.is_dir() and not run_dir.is_symlink(), "Attempt directory invalid")
    total = 0
    for path in run_dir.rglob("*"):
        base.need(not path.is_symlink(), "Attempt contains a symlink")
        if path.is_file():
            total += path.stat().st_size
            base.need(total < base.MAX_WORKER_BYTES, "Worker output exceeds cap")
    result = base.json_object(run_dir / "result.json")
    meta = base.json_object(run_dir / "meta.json")
    base.need(result.get("status") == STATUS and result.get("scope") == SCOPE and
              meta.get("scope") == SCOPE and meta.get("status") == META_STATUS,
              "Result/meta status or scope invalid")
    base.need(result.get("planned_counts") == COUNTS and
              result.get("actual_counts") == COUNTS and meta.get("planned_counts") == COUNTS,
              "Planned or actual forward counts differ")
    base.need(meta.get("full_schedule") == [list(row) for row in full_schedule()] and
              meta.get("direct_schedule") == [list(row) for row in direct_schedule()],
              "Frozen schedule differs")
    base.need(result.get("source_sha256") == spec.source_sha256 and
              meta.get("source_sha256") == spec.source_sha256 and
              result.get("audio_token_id") == spec.audio_token_id,
              "Frozen source or audio token differs")
    for key, name in (("meta_sha256", "meta.json"),
                      ("model_layout_sha256", "model_layout.json"),
                      ("progress_sha256", "progress.jsonl")):
        base.need(result.get(key) == base.sha(run_dir / name), f"{name} SHA differs")
    layout = base.json_object(run_dir / "model_layout.json")
    base.need(layout.get("valid") is True, "Model layout invalid")
    elapsed, peak = result.get("elapsed_seconds"), result.get("peak_mps_driver_bytes")
    base.need(type(elapsed) in (float, int) and 0 <= elapsed < 300 and
              type(peak) is int and 0 <= peak < 30 * 1024**3,
              "Process time or MPS peak exceeds cap")
    for key in ("protocol_sha256", "runner_sha256",
                "failed_stage_b_fp32_manifest_sha256",
                "failed_v1_diagnostic_manifest_sha256"):
        base.need(result.get(key) == meta.get(key), f"{key} differs between result/meta")
    base.need(result["failed_v1_diagnostic_manifest_sha256"] == V1_FAILED_MANIFEST_SHA,
              "V1 failure manifest pin differs")
    detail_sha, detail_rel = result.get("details_sha256"), result.get("details_path")
    base.need(isinstance(detail_sha, str) and base.HEX64.fullmatch(detail_sha) and
              detail_rel == f"blobs/{detail_sha}.pt", "Content-addressed path differs")
    detail = run_dir / detail_rel
    base.need(detail.is_file() and not detail.is_symlink() and
              not (run_dir / "blobs").is_symlink() and
              detail.stat().st_size == result.get("details_bytes") and
              detail.stat().st_size < base.MAX_SIDECAR_BYTES and
              base.sha(detail) == detail_sha, "Saved tensor sidecar differs")
    payload = torch.load(detail, map_location="cpu", weights_only=True, mmap=True)
    report = audit_payload(payload, spec=spec)
    report.update({"result_sha256": base.sha(run_dir / "result.json"),
                   "meta_sha256": base.sha(run_dir / "meta.json"),
                   "details_sha256": detail_sha,
                   "actual_attempt_bytes_before_audit": total,
                   "protocol_sha256": result["protocol_sha256"],
                   "runner_sha256": result["runner_sha256"],
                   "failed_stage_b_fp32_manifest_sha256":
                       result["failed_stage_b_fp32_manifest_sha256"],
                   "failed_v1_diagnostic_manifest_sha256": V1_FAILED_MANIFEST_SHA})
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run_dir, args.output)
    print(json.dumps({"status": report["status"], "audit_sha256": base.sha(args.output)},
                     allow_nan=False))


if __name__ == "__main__":
    main()
