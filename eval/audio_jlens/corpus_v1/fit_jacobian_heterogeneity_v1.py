"""FIT-only local Jacobians versus frozen and pair-held-out corpus means.

The frozen corpus spec/manifest include metadata for all splits, but only FIT
WAVs, activations, and candidate outputs enter this diagnostic. It cannot
rescue the negative Stage A v1 result. --validate-only loads no model weights.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "eval" / "audio_jlens"
sys.path[:0] = [str(ROOT / "models" / "qwen_omni"), str(EVAL)]
from adapter import TRANSFORMERS_REVISION, resolve_hook_points  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import MAX_MPS_DRIVER_BYTES, load_pinned_thinker, record_mps_memory  # noqa: E402
from fit_pool_capability_v1 import (MANIFEST_SHA, SPEC_SHA, load_fit_cases,
                                    sha, validate_tokenizer)  # noqa: E402
from fit_pool_single_token_score_v1 import selected_cases  # noqa: E402
from fit_rank16_transport_v1 import (SITES, SITE_NAMES, TARGET, RANK,
                                     TARGET_MODE_MAX_ABS_TOL,
                                     accumulate_one_vjp, captures_for_forward,
                                     atomic_json_save, atomic_torch_save,
                                     case_batch, empty_accumulator,
                                     fit_metadata, rows_for_case, tensor_sha)  # noqa: E402
from fit_uniform_calibration_v1 import logits_from_residual, prefix_score  # noqa: E402

PROTOCOL = HERE / "FIT_JACOBIAN_HETEROGENEITY_PROTOCOL_V1.md"
FIT_REPORT = HERE / "fit_rank16_transport_v1.report.json"
LENS = HERE / "fit_rank16_transport_v1.lens.pt"
STAGE1 = HERE / "fit_rank16_transport_v1.stage1.pt"
CALIBRATION = HERE / "fit_uniform_calibration_v1.raw.pt"
CALIBRATION_REPORT = HERE / "fit_uniform_calibration_v1.report.json"
SCORING = HERE / "SCORING_V2.json"
ALPHA = .025
N_CASES = 48
MAX_WALL_SECONDS = 1200
EXPECTED_SHA = {
    "fit_report": "bd5b281dc8eb0e6b2653f496fe0e4901df9cab035b975415dde0c00a686b498e",
    "lens": "fef84a20f437c37ca665e4026bb3645ccd558e549c7ffe2adff20e6a867ba12c",
    "stage1": "8276aca0e24fcaa1397693e0353f9c7ee59190036f3516c0c13ebcbc6208b641",
    "calibration": "76085f0dab8f37b6052024b582ff34533cac3d1c579b27b39a631c536d9cd110",
    "calibration_report": "b130aa4af85a807671f537aaabbaddf922bc8f2632d8721c9188c2b6aa8e7bfa",
    "scoring": "e7ea18cd0d5e2638bec64b8733049a5a2754ecd674080d93766db8409b46dd9b",
}


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("FIT diagnostic invocation reached 20-minute wall bound; resume checkpoint")


def load_fit_inputs():
    spec, manifest, all_cases = load_fit_cases()
    validate_tokenizer(spec)
    cases = selected_cases(spec, all_cases)
    for key, path in {"fit_report": FIT_REPORT, "lens": LENS, "stage1": STAGE1,
                      "calibration": CALIBRATION,
                      "calibration_report": CALIBRATION_REPORT,
                      "scoring": SCORING}.items():
        if sha(path) != EXPECTED_SHA[key]:
            raise ValueError(f"Pinned FIT source changed: {key}")
    report = json.loads(FIT_REPORT.read_text())
    calibration_report = json.loads(CALIBRATION_REPORT.read_text())
    lens = torch.load(LENS, map_location="cpu", weights_only=True)
    stage1 = torch.load(STAGE1, map_location="cpu", weights_only=True)
    calibration = torch.load(CALIBRATION, map_location="cpu", weights_only=True)
    if (len(cases) != N_CASES or sha(LENS) != report["lens_sha256"]
            or sha(STAGE1) != report["stage1_sha256"]
            or sha(CALIBRATION) != calibration_report["raw_sha256"]
            or sha(SCORING) != lens["metadata"]["scoring_v2_sha256"]
            or lens["metadata"] != stage1["metadata"]):
        raise ValueError("Frozen FIT artifact hash or metadata changed")
    if (lens["metadata"] != fit_metadata(spec, cases)
            or calibration["metadata"]["lens_sha256"] != sha(LENS)
            or calibration["metadata"]["stage1_sha256"] != sha(STAGE1)
            or calibration["metadata"]["alpha"] != ALPHA
            or calibration_report["metadata"] != calibration["metadata"]
            or calibration_report["summary"] != calibration["summary"]
            or calibration["summary"]["completed_fit_contexts"] != N_CASES
            or len(calibration["records"]) != N_CASES
            or [r["case_id"] for r in stage1["records"]] != [c["case_id"] for c in cases]
            or [r["case_id"] for r in calibration["records"]] != [c["case_id"] for c in cases]):
        raise ValueError("Frozen FIT cases, basis, or calibration changed")
    by_key = {(c["pair_id"], c["template"], c["engine"], c["word"]): i
              for i, c in enumerate(cases)}
    for i, case in enumerate(cases):
        j = by_key[(case["pair_id"], case["template"], case["engine"], case["foil_word"])]
        if calibration["records"][i]["audio_sha256"] != case["audio_sha256"]:
            raise ValueError("FIT calibration audio hash changed")
        for site in SITES:
            item = calibration["records"][i]["sites"][site]
            expected_delta = (stage1["source_bin_means"][site][j]
                              - stage1["source_bin_means"][site][i])
            if (not torch.equal(item["delta_bins"], expected_delta)
                    or item["source_rows"] != stage1["records"][i]["row_indices"][site]
                    or item["source_rows"] != stage1["records"][j]["row_indices"][site]
                    or item["source_shape"] != stage1["records"][i]["source_shapes"][site]
                    or item["source_shape"] != stage1["records"][j]["source_shapes"][site]):
                raise ValueError("FIT calibration donor delta or source geometry changed")
    policy = json.loads(SCORING.read_text())
    for case in cases:
        a = set(policy["words"][case["word"]]["first_token_ids"])
        b = set(policy["words"][case["foil_word"]]["first_token_ids"])
        if not a or not b or a & b:
            raise ValueError("V2 pair token classes overlap")
    metadata = {
        "status": "fit_only_local_jacobian_heterogeneity_no_v1_rescue",
        "script_sha256": sha(Path(__file__)), "protocol_sha256": sha(PROTOCOL),
        "fit_report_sha256": sha(FIT_REPORT), "lens_sha256": sha(LENS),
        "spec_sha256": SPEC_SHA, "manifest_sha256": MANIFEST_SHA,
        "stage1_sha256": sha(STAGE1), "calibration_sha256": sha(CALIBRATION),
        "calibration_report_sha256": sha(CALIBRATION_REPORT),
        "scoring_v2_sha256": sha(SCORING),
        "case_ids": [c["case_id"] for c in cases],
        "model_revision": lens["metadata"]["model_revision"],
        "transformers_revision": TRANSFORMERS_REVISION,
        "alpha": ALPHA, "rank": RANK, "planned_fit_contexts": N_CASES,
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "max_wall_seconds_per_invocation": MAX_WALL_SECONDS,
    }
    return spec, manifest, cases, stage1, lens, calibration, policy, metadata


def local_matrix(accumulator: dict, site: str) -> torch.Tensor:
    value = accumulator["jacobian_sums"][site].float().contiguous()
    if (value.shape[0:2] != (6, RANK) or not torch.isfinite(value).all()
            or float(value.norm()) <= 0):
        raise ValueError(f"Invalid {site} local Jacobian")
    return value


def local_stage2_case(model, processor, modules, case, group,
                      stage1_record, basis, accumulator, memory, deadline):
    """Pinned FIT VJP semantics with a deadline before each coordinate."""
    check_deadline(deadline)
    batch, row_map, intervals = case_batch(model, processor, case, group)
    captured = captures_for_forward(model, batch, modules, gradient=True)
    check_deadline(deadline)
    rows = rows_for_case(row_map, intervals, captured)
    if (rows != stage1_record["row_indices"] or
            {site: list(captured[site].shape) for site in SITES} != stage1_record["source_shapes"]):
        raise ValueError("Local diagnostic source rows/shapes differ from frozen stage 1")
    mode_gap = float((captured[TARGET][0, -1].detach().float().cpu()
                      - stage1_record["target_state"]).abs().max())
    if mode_gap > TARGET_MODE_MAX_ABS_TOL:
        raise ValueError("Local diagnostic grad/no-grad terminal mismatch")
    target = captured[TARGET]
    sources = [captured[site] for site in SITES]
    if not all(source.requires_grad for source in sources):
        raise ValueError("Audio graph disconnected")
    times = []
    for k in range(RANK):
        check_deadline(deadline)
        cotangent = torch.zeros_like(target)
        cotangent[0, -1] = basis[:, k].to(target.device)
        torch.mps.synchronize()
        tick = time.perf_counter()
        gradients = torch.autograd.grad(target, sources, grad_outputs=cotangent,
                                        retain_graph=k < RANK - 1, allow_unused=False)
        torch.mps.synchronize()
        times.append(time.perf_counter() - tick)
        accumulate_one_vjp(accumulator, dict(zip(SITES, gradients, strict=True)), rows, k)
        record_mps_memory(memory, f"local_{case['case_id']}_coordinate_{k}")
        check_deadline(deadline)
    return {"vjp_seconds": times,
            "grad_vs_no_grad_terminal_max_abs": mode_gap,
            "source_rows_per_bin": {site: list(map(len, rows[site])) for site in SITES}}


def pair_heldout_means(local: torch.Tensor, pair_ids: list[str]) -> torch.Tensor:
    if local.shape[0] != N_CASES or len(pair_ids) != N_CASES:
        raise ValueError("Expected 48 local Jacobians")
    pairs = list(dict.fromkeys(pair_ids))
    if len(pairs) != 12 or any(pair_ids.count(pair) != 4 for pair in pairs):
        raise ValueError("Expected 12 pairs × four FIT contexts")
    total = local.double().sum(0)
    return torch.stack([
        ((total - local[[j for j, p in enumerate(pair_ids) if p == pair]].double().sum(0)) / 44).float()
        for pair in pair_ids
    ])


def similarity_strata(local: torch.Tensor, cases: list[dict]) -> dict:
    if local.shape[0] != N_CASES or len(cases) != N_CASES:
        raise ValueError("Similarity needs 48 fixed contexts")
    x = local.flatten(1).double()
    norms = x.norm(dim=1)
    if bool((norms <= 0).any()):
        raise ValueError("Zero local Jacobian in similarity table")
    x = x / norms[:, None]
    cosine = x @ x.T
    groups = defaultdict(list)
    for i in range(N_CASES):
        for j in range(i + 1, N_CASES):
            a, b = cases[i], cases[j]
            value = float(cosine[i, j])
            if a["pair_id"] == b["pair_id"]:
                groups["within_pair_same_voice" if a["engine"] == b["engine"]
                       else "within_pair_cross_voice"].append(value)
            elif a["category"] == b["category"]:
                groups["between_pair_same_category"].append(value)
            else:
                groups["across_categories"].append(value)
            groups["same_voice_all" if a["engine"] == b["engine"]
                   else "cross_voice_all"].append(value)
    expected = {"within_pair_same_voice": 24, "within_pair_cross_voice": 48,
                "between_pair_same_category": 192, "across_categories": 864,
                "same_voice_all": 552, "cross_voice_all": 576}
    if {key: len(groups[key]) for key in expected} != expected:
        raise ValueError("Similarity strata counts drifted")
    def describe(values):
        sorted_values = torch.tensor(values, dtype=torch.float64)
        return {"n": len(values), "mean": float(sorted_values.mean()),
                "median": float(torch.quantile(sorted_values, .5)),
                "p10": float(torch.quantile(sorted_values, .1)),
                "p90": float(torch.quantile(sorted_values, .9))}
    return {key: describe(groups[key]) for key in expected}


def terminal_metrics(predicted: torch.Tensor, actual: torch.Tensor,
                     basis: torch.Tensor) -> dict:
    projected = basis @ (basis.T @ actual)
    observed_norm = float(actual.norm())
    projected_norm = float(projected.norm())
    prediction_norm = float(predicted.norm())
    return {
        "full_terminal_l2_error": float((predicted - actual).norm()),
        "projected_terminal_l2_error": float((predicted - projected).norm()),
        "rank16_capture_fraction": (projected_norm**2 / observed_norm**2
                                    if observed_norm > 0 else None),
        "projected_cosine": (float(torch.dot(predicted, projected) /
                                   (prediction_norm * projected_norm))
                             if prediction_norm > 0 and projected_norm > 0 else None),
    }


def analyze(model, cases, stage1, lens, calibration, policy, records, *, deadline=None):
    basis = lens["basis"].float()
    per_site = {}
    for site in SITES:
        if deadline is not None:
            check_deadline(deadline)
        local = torch.stack([record["jacobians"][site] for record in records])
        counts = torch.tensor([record["source_rows_per_bin"][site] for record in records],
                              dtype=torch.float32)
        if counts.shape != (N_CASES, 6) or bool((counts <= 0).any()):
            raise ValueError("Invalid row counts for per-row Jacobian similarity")
        row_mean_local = local / counts[:, :, None, None]
        frozen = lens["jacobians"][site]
        average = local.double().mean(0).float()
        difference = float((average - frozen).abs().max())
        if difference > 1e-3:
            raise ValueError(f"{site} recomputed local mean differs from frozen map")
        pair_ids = [c["pair_id"] for c in cases]
        heldout = pair_heldout_means(local, pair_ids)
        row_results = []
        with torch.no_grad():
            for i, case in enumerate(cases):
                if deadline is not None:
                    check_deadline(deadline)
                record = calibration["records"][i]
                item = record["sites"][site]
                delta = item["delta_bins"].float()
                h = stage1["records"][i]["target_state"].float()
                actual = item["terminal_delta_per_alpha"].float() * ALPHA
                donor_ids = policy["words"][case["foil_word"]]["first_token_ids"]
                recipient_ids = policy["words"][case["word"]]["first_token_ids"]
                clean_logits = logits_from_residual(model, h.to(model.device)).detach().float().cpu()
                clean_score = prefix_score(clean_logits, donor_ids, recipient_ids)
                if abs(clean_score - record["baseline_v2_score"]) > 1e-3:
                    raise ValueError("FIT clean score changed from calibration")
                models = {"local": local[i], "frozen_average": frozen,
                          "pair_heldout_average": heldout[i]}
                predictions = {}
                for name, matrix in models.items():
                    coordinates = torch.einsum("bkd,bd->k", matrix, delta) * ALPHA
                    predicted = basis @ coordinates
                    logits = logits_from_residual(model, (h + predicted).to(model.device)).detach().float().cpu()
                    score_delta = prefix_score(logits, donor_ids, recipient_ids) - clean_score
                    predictions[name] = {
                        "score_delta": score_delta,
                        "score_absolute_error": abs(score_delta - item["measured_v2_delta_score"]),
                        **terminal_metrics(predicted, actual, basis),
                    }
                if abs(predictions["frozen_average"]["score_delta"] - item["map_predicted_v2_delta_score"]) > 1e-3:
                    raise ValueError("Frozen average score prediction changed from FIT calibration")
                row_results.append({"case_id": case["case_id"], "pair_id": case["pair_id"],
                                    "category": case["category"], "engine": case["engine"],
                                    "audio_sha256": case["audio_sha256"],
                                    "measured_v2_delta_score": item["measured_v2_delta_score"],
                                    "predictions": predictions})
        names = ("local", "frozen_average", "pair_heldout_average")
        summary = {name: {
            "score_mae": sum(row["predictions"][name]["score_absolute_error"] for row in row_results) / N_CASES,
            "full_terminal_l2_mean": sum(row["predictions"][name]["full_terminal_l2_error"] for row in row_results) / N_CASES,
            "projected_terminal_l2_mean": sum(row["predictions"][name]["projected_terminal_l2_error"] for row in row_results) / N_CASES,
        } for name in names}
        pair_summary = {}
        for pair in dict.fromkeys(pair_ids):
            subset = [row for row in row_results if row["pair_id"] == pair]
            pair_summary[pair] = {name: {
                "score_mae": sum(row["predictions"][name]["score_absolute_error"] for row in subset) / 4,
                "projected_terminal_l2_mean": sum(row["predictions"][name]["projected_terminal_l2_error"] for row in subset) / 4,
            } for name in names}
        per_site[site] = {
            "recomputed_mean_vs_frozen_max_abs": difference,
            "local_matrix_shape": list(local.shape),
            "matrix_hashes": [tensor_sha(t) for t in local],
            "similarity_uniform_edit_derivative": similarity_strata(local, cases),
            "similarity_per_row_mean_gradient": similarity_strata(row_mean_local, cases),
            "metrics": summary, "pair_metrics": pair_summary,
            "rows": row_results,
        }
    return per_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "fit_jacobian_heterogeneity_v1")
    args = parser.parse_args()
    spec, manifest, cases, stage1, lens, calibration, policy, metadata = load_fit_inputs()
    if args.validate_only:
        print(json.dumps({"status": "FIT metadata valid, no model forward",
                          "metadata": metadata}, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("Pinned FIT Jacobian assay requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision changed")
    prefix = args.output_prefix.resolve()
    checkpoint = prefix.with_suffix(".smoke.pt" if args.smoke else ".checkpoint.pt")
    result = prefix.with_suffix(".smoke.json" if args.smoke else ".result.pt")
    report = None if args.smoke else prefix.with_suffix(".report.json")
    memory = prefix.with_suffix(".smoke.memory.jsonl" if args.smoke else ".memory.jsonl")
    if result.exists() or (report is not None and report.exists()):
        raise FileExistsError("Final diagnostic output exists")
    if args.smoke and (checkpoint.exists() or memory.exists()):
        raise FileExistsError("Smoke must start with fresh prefix")
    if checkpoint.exists() and not memory.exists():
        raise ValueError("Checkpoint exists without its memory log")
    started = time.monotonic()
    if not memory.exists():
        memory.write_text("")
    torch.mps.set_per_process_memory_fraction(
        mps_cap_fraction(int(torch.mps.recommended_max_memory())))
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32, device_map="mps",
        low_cpu_mem_usage=True)
    if model.dtype != torch.float32 or processor.tokenizer.eos_token_id != policy["eos_token_id"]:
        raise ValueError("Pinned model dtype/EOS changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record_mps_memory(memory, "after_model_load")
    points = {p.name: p.module for p in resolve_hook_points(
        model, audio_layers=(31,), text_layers=(35,))}
    modules = {site: points[SITE_NAMES[site]] for site in SITES}
    modules[TARGET] = points[TARGET]
    groups = {(p["pair_id"], p["phrasing_id"], p["engine"]): p
              for p in manifest["pairs"] if p["split"] == "fit_pool"}
    group = {"system_prompt": spec["system_prompt"], "groups": groups}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
    if saved is not None and (saved["metadata"] != metadata or args.smoke):
        raise ValueError("Diagnostic checkpoint provenance changed")
    records = saved["records"] if saved else []
    if [r["case_id"] for r in records] != [c["case_id"] for c in cases[:len(records)]]:
        raise ValueError("Diagnostic checkpoint is not frozen FIT prefix")
    limit = 1 if args.smoke else N_CASES
    for i in range(len(records), limit):
        check_deadline(started + MAX_WALL_SECONDS)
        accumulator = empty_accumulator(stage1)
        measured = local_stage2_case(model, processor, modules, cases[i], group,
                                     stage1["records"][i], lens["basis"], accumulator,
                                     memory, started + MAX_WALL_SECONDS)
        record = {"case_id": cases[i]["case_id"],
                  "audio_sha256": cases[i]["audio_sha256"],
                  "vjp_seconds": measured["vjp_seconds"],
                  "grad_vs_no_grad_terminal_max_abs": measured["grad_vs_no_grad_terminal_max_abs"],
                  "source_rows_per_bin": measured["source_rows_per_bin"],
                  "jacobians": {site: local_matrix(accumulator, site) for site in SITES}}
        records.append(record)
        atomic_torch_save(checkpoint, {"metadata": metadata, "records": records})
        record_mps_memory(memory, f"after_case_{i+1}_checkpoint")
        if (i + 1) % 8 == 0:
            print(f"local Jacobians captured {i+1}/{N_CASES}", flush=True)
    if args.smoke:
        atomic_json_save(result, {"status": "FIT first-context VJP smoke, no heterogeneity claim",
                                  "metadata": metadata, "checkpoint_sha256": sha(checkpoint),
                                  "memory_sha256": sha(memory),
                                  "case_id": records[0]["case_id"],
                                  "matrix_shapes": {s: list(records[0]["jacobians"][s].shape) for s in SITES},
                                  "matrix_norms": {s: float(records[0]["jacobians"][s].norm()) for s in SITES},
                                  "vjp_seconds": records[0]["vjp_seconds"]})
        return
    analyzed = analyze(model, cases, stage1, lens, calibration, policy, records,
                       deadline=started + MAX_WALL_SECONDS)
    check_deadline(started + MAX_WALL_SECONDS)
    record_mps_memory(memory, "after_analysis")
    atomic_torch_save(result, {"metadata": metadata, "records": records,
                               "site_analysis": analyzed})
    atomic_json_save(report, {"status": metadata["status"], "metadata": metadata,
                              "result_sha256": sha(result), "checkpoint_sha256": sha(checkpoint),
                              "memory_sha256": sha(memory),
                              "mean_vs_frozen_max_abs": {s: analyzed[s]["recomputed_mean_vs_frozen_max_abs"] for s in SITES},
                              "metrics": {s: analyzed[s]["metrics"] for s in SITES},
                              "similarity_uniform_edit_derivative": {
                                  s: analyzed[s]["similarity_uniform_edit_derivative"] for s in SITES},
                              "similarity_per_row_mean_gradient": {
                                  s: analyzed[s]["similarity_per_row_mean_gradient"] for s in SITES},
                              "pair_metrics": {s: analyzed[s]["pair_metrics"] for s in SITES},
                              "scope": "FIT diagnostic only; fixed all-FIT basis; no v1 rescue or dev/test inference"})
    print(json.dumps({s: analyzed[s]["metrics"] for s in SITES}, indent=2))


if __name__ == "__main__":
    main()
