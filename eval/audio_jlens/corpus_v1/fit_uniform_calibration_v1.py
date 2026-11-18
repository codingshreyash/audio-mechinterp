"""Measure FIT-only six-bin uniform edits for comparator calibration.

This script never prepares development or test audio. It is designed for a
separate MPS run after the corpus lens fit and independent review.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "eval" / "audio_jlens"
sys.path[:0] = [str(ROOT / "models" / "qwen_omni"), str(EVAL)]
from adapter import MODEL_REVISION, TRANSFORMERS_REVISION, resolve_hook_points  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import (MAX_MPS_DRIVER_BYTES, _output_tensor, _replace_output,
                         load_pinned_thinker, record_mps_memory)  # noqa: E402
from fit_pool_capability_v1 import load_fit_cases, sha, validate_tokenizer  # noqa: E402
from fit_pool_single_token_score_v1 import selected_cases  # noqa: E402
from fit_rank16_transport_v1 import (SITES, SITE_NAMES, TARGET,
                                     atomic_json_save, atomic_torch_save,
                                     case_batch,
                                     fit_metadata, rows_for_case)  # noqa: E402


PROTOCOL = HERE / "PRE_DEV_EVAL_PROTOCOL_V1.md"
SCORING = HERE / "SCORING_V2.json"
SCORING_SHA = "e7ea18cd0d5e2638bec64b8733049a5a2754ecd674080d93766db8409b46dd9b"
ALPHA = 0.025
ZERO_REPEATS = 3
ZERO_DRIFT_LIMIT = 1e-4
DEFAULT_LENS = HERE / "fit_rank16_transport_v1.lens.pt"
DEFAULT_STAGE1 = HERE / "fit_rank16_transport_v1.stage1.pt"


def prefix_score(logits: torch.Tensor, donor_ids: list[int],
                 recipient_ids: list[int]) -> float:
    if logits.ndim != 1 or not donor_ids or not recipient_ids or set(donor_ids) & set(recipient_ids):
        raise ValueError("Invalid first-token score classes")
    return float(torch.logsumexp(logits[donor_ids].double(), 0)
                 - torch.logsumexp(logits[recipient_ids].double(), 0))


def logits_from_residual(model, residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 1 or residual.shape[0] != 2048:
        raise ValueError("Expected 2048D prompt-last residual")
    return model.lm_head(model.model.language_model.norm(residual))


def add_uniform_bins(value: torch.Tensor, rows: list[list[int]],
                     deltas: torch.Tensor, alpha: float) -> torch.Tensor:
    if value.ndim != 2 or deltas.shape != (6, value.shape[-1]) or len(rows) != 6:
        raise ValueError("Uniform-edit shape mismatch")
    if not all(group for group in rows):
        raise ValueError("Empty source bin")
    flat = [r for group in rows for r in group]
    if len(set(flat)) != len(flat) or min(flat) < 0 or max(flat) >= value.shape[0]:
        raise ValueError("Invalid/disjoint source rows")
    patched = value.clone()
    for index, group in enumerate(rows):
        patched[group] = patched[group] + alpha * deltas[index].to(value.device)
    return patched


def clean_forward(model, batch, modules) -> tuple[dict, torch.Tensor]:
    captured = {}
    calls = {site: 0 for site in (*SITES, TARGET)}

    def hook_for(site):
        def hook(_module, _inputs, output):
            calls[site] += 1
            if site == TARGET:
                value = output[0] if isinstance(output, tuple) else output
                if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
                    raise ValueError("Clean terminal target shape changed")
            else:
                value = _output_tensor(output)
            captured[site] = value.detach()
            return output
        return hook

    handles = [modules[site].register_forward_hook(hook_for(site))
               for site in (*SITES, TARGET)]
    try:
        with torch.no_grad():
            output = model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if calls != {"encoder31": 1, "connector": 1, TARGET: 1}:
        raise ValueError(f"Unexpected clean hook calls {calls}")
    return captured, output.logits[0, -1].detach().float().cpu()


def patched_forward(model, batch, source_module, target_module, rows,
                    deltas, alpha: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    calls = {"source": 0, "target": 0}
    captured = {}

    def source_hook(_module, _inputs, output):
        calls["source"] += 1
        value = _output_tensor(output)
        if (value.ndim != 2 or value.shape[-1] != deltas.shape[-1]
                or max(r for group in rows for r in group) >= value.shape[0]):
            raise ValueError("Patch source shape/row mismatch")
        captured["source_shape"] = list(value.shape)
        return _replace_output(output, add_uniform_bins(value, rows, deltas, alpha))

    def target_hook(_module, _inputs, output):
        calls["target"] += 1
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("Terminal target shape changed")
        captured["terminal"] = value[0, -1].detach().float().cpu().contiguous()
        return output

    handles = [source_module.register_forward_hook(source_hook),
               target_module.register_forward_hook(target_hook)]
    try:
        with torch.no_grad():
            output = model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if calls != {"source": 1, "target": 1}:
        raise ValueError(f"Unexpected patch hook calls: {calls}")
    return captured["terminal"], output.logits[0, -1].detach().float().cpu(), calls


def calibration_metadata(lens_path: Path, stage1_path: Path, cases: list,
                         spec: dict) -> dict:
    return {
        "status": "fit_only_uniform_calibration_no_development_or_test",
        "script_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(PROTOCOL),
        "scoring_v2_sha256": SCORING_SHA,
        "lens_sha256": sha(lens_path),
        "stage1_sha256": sha(stage1_path),
        "fit_metadata": fit_metadata(spec, cases),
        "alpha": ALPHA, "zero_repeats": ZERO_REPEATS,
        "zero_drift_limit": ZERO_DRIFT_LIMIT,
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
    }


def finish_summary(records: list[dict]) -> dict:
    if len(records) != 48:
        raise ValueError("Calibration requires all 48 FIT contexts")
    floors = {}
    for site in SITES:
        departures = []
        for record in records:
            scores = record["sites"][site]["zero_v2_scores"]
            mean = sum(scores) / ZERO_REPEATS
            departures.extend(abs(score - mean) for score in scores)
        floors[site] = max(ZERO_DRIFT_LIMIT, 5 * max(departures))
    return {"planned_fit_contexts": 48, "completed_fit_contexts": len(records),
            "numerical_floor_v2_by_site": floors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="FIT-only first context, both sites, zero repeats and alpha .025")
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    spec, manifest, all_cases = load_fit_cases()
    validate_tokenizer(spec)
    cases = selected_cases(spec, all_cases)
    if sha(SCORING) != SCORING_SHA:
        raise ValueError("Frozen V2 score policy changed")
    policy = json.loads(SCORING.read_text())
    if args.validate_only:
        print(json.dumps({"status": "fit_inputs_valid_no_model_forward",
                          "fit_contexts": len(cases),
                          "protocol_sha256": sha(PROTOCOL),
                          "scoring_v2_sha256": SCORING_SHA}, indent=2))
        return
    lens_path, stage1_path = args.lens.resolve(), args.stage1.resolve()
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=True)
    if lens["status"] != "fit_only_corpus_transport_no_efficacy_evaluation" or lens["metadata"] != fit_metadata(spec, cases):
        raise ValueError("FIT lens provenance changed")
    if sha(stage1_path) != lens["stage1_sha256"] or stage1["metadata"] != lens["metadata"]:
        raise ValueError("Stage1/lens mismatch")
    if not torch.backends.mps.is_available():
        raise RuntimeError("FIT calibration requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision changed")
    prefix = (args.output_prefix or HERE / (
        "fit_uniform_calibration_v1_smoke" if args.smoke else "fit_uniform_calibration_v1"
    )).resolve()
    checkpoint = prefix.with_suffix(".checkpoint.pt")
    raw = prefix.with_suffix(".raw.pt")
    report = prefix.with_suffix(".report.json")
    smoke_raw = prefix.with_suffix(".smoke.pt")
    smoke_report = prefix.with_suffix(".smoke.json")
    memory = prefix.with_suffix(".memory.jsonl")
    if raw.exists() or report.exists() or smoke_raw.exists() or smoke_report.exists():
        raise FileExistsError("FIT calibration refuses overwrite")
    if args.smoke and checkpoint.exists():
        raise FileExistsError("FIT calibration smoke requires a unique prefix")
    metadata = calibration_metadata(lens_path, stage1_path, cases, spec)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("Calibration checkpoint provenance changed")
        records = saved["records"]
        if [r["case_id"] for r in records] != [c["case_id"] for c in cases[:len(records)]]:
            raise ValueError("Calibration checkpoint cases are not frozen prefix")
    else:
        records = []
    if not memory.exists():
        memory.write_text("")
    torch.mps.set_per_process_memory_fraction(
        mps_cap_fraction(int(torch.mps.recommended_max_memory())))
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32 or processor.tokenizer.eos_token_id != policy["eos_token_id"]:
        raise ValueError("Model dtype/EOS changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record_mps_memory(memory, "after_model_load")
    points = {point.name: point.module for point in resolve_hook_points(
        model, audio_layers=(31,), text_layers=(35,))}
    modules = {site: points[SITE_NAMES[site]] for site in SITES}
    modules[TARGET] = points[TARGET]
    groups = {(p["pair_id"], p["phrasing_id"], p["engine"]): p
              for p in manifest["pairs"] if p["split"] == "fit_pool"}
    group = {"system_prompt": spec["system_prompt"], "groups": groups}
    by_key = {(case["pair_id"], case["template"], case["engine"], case["word"]): i
              for i, case in enumerate(cases)}
    limit = 1 if args.smoke else len(cases)
    for i in range(len(records), limit):
        case = cases[i]
        donor_index = by_key[(case["pair_id"], case["template"],
                              case["engine"], case["foil_word"])]
        batch, row_map, intervals = case_batch(model, processor, case, group)
        clean, raw_clean_logits = clean_forward(model, batch, modules)
        record_mps_memory(memory, f"fit_{i}_clean")
        row_sets = rows_for_case(row_map, intervals, clean)
        if row_sets != stage1["records"][i]["row_indices"]:
            raise ValueError("FIT row indices changed from lens capture")
        h_clean = clean[TARGET][0, -1].detach().float().cpu().contiguous()
        if (h_clean - stage1["records"][i]["target_state"]).abs().max() > 1e-3:
            raise ValueError("Clean terminal state changed from FIT map capture")
        clean_logits = logits_from_residual(model, h_clean.to(model.device)).detach().float().cpu()
        clean_logit_max_abs = float((clean_logits - raw_clean_logits).abs().max())
        if clean_logit_max_abs > 1e-3:
            raise ValueError("Clean terminal residual norm/head logits mismatch")
        donor, recipient = case["foil_word"], case["word"]
        v2_d, v2_r = policy["words"][donor]["first_token_ids"], policy["words"][recipient]["first_token_ids"]
        v1_d = [entry["token_id"] for entry in spec["aliases"][donor]]
        v1_r = [entry["token_id"] for entry in spec["aliases"][recipient]]
        # V1 alias schema is validated by the score-policy runner; still bind
        # the exact current token lists before measuring.
        baseline_v2 = prefix_score(clean_logits, v2_d, v2_r)
        baseline_v1 = prefix_score(clean_logits, v1_d, v1_r)
        site_results = {}
        for site in SITES:
            rows = row_sets[site]
            if rows != stage1["records"][donor_index]["row_indices"][site]:
                raise ValueError("Donor and recipient FIT row sets differ")
            current_means = torch.stack([
                clean[site][group_rows].float().mean(dim=0).cpu()
                for group_rows in rows
            ])
            source_mean_max_abs = float((
                current_means - stage1["source_bin_means"][site][i]
            ).abs().max())
            if source_mean_max_abs > 1e-3:
                raise ValueError("Clean source-bin mean changed from FIT map capture")
            delta = (stage1["source_bin_means"][site][donor_index]
                     - stage1["source_bin_means"][site][i]).float()
            zero_scores = []
            for j in range(ZERO_REPEATS):
                h0, raw_logits, calls = patched_forward(
                    model, batch, modules[site], modules[TARGET], rows, delta, 0)
                score = prefix_score(raw_logits, v2_d, v2_r)
                zero_scores.append(score)
                if (h0 - h_clean).abs().max() > 1e-3 or abs(score - baseline_v2) > ZERO_DRIFT_LIMIT:
                    raise ValueError("Alpha-zero clean-state/score invariant failed")
                record_mps_memory(memory, f"fit_{i}_{site}_zero_{j}")
            h_patch, raw_logits, calls = patched_forward(
                model, batch, modules[site], modules[TARGET], rows, delta, ALPHA)
            final_logits = logits_from_residual(model, h_patch.to(model.device)).detach().float().cpu()
            if (final_logits - raw_logits).abs().max() > 1e-3:
                raise ValueError("Terminal residual norm/head logits mismatch")
            site_results[site] = {
                "delta_bins": delta,
                "terminal_delta_per_alpha": (h_patch - h_clean) / ALPHA,
                "zero_v2_scores": zero_scores,
                "measured_v2_delta_score": prefix_score(raw_logits, v2_d, v2_r) - baseline_v2,
                "measured_v1_delta_score": prefix_score(raw_logits, v1_d, v1_r) - baseline_v1,
                "source_rows": rows,
                "source_shape": list(clean[site].shape),
                "source_mean_vs_stage1_max_abs": source_mean_max_abs,
                "hook_calls": calls,
            }
            record_mps_memory(memory, f"fit_{i}_{site}_alpha_{ALPHA}")
        records.append({"case_id": case["case_id"], "pair_id": case["pair_id"],
                        "audio_sha256": case["audio_sha256"],
                        "baseline_v2_score": baseline_v2,
                        "baseline_v1_score": baseline_v1,
                        "clean_norm_head_vs_forward_logit_max_abs": clean_logit_max_abs,
                        "sites": site_results})
        if args.smoke:
            atomic_torch_save(smoke_raw, {
                "status": "fit_only_smoke_not_calibration",
                "metadata": metadata, "record": records[0],
            })
            atomic_json_save(smoke_report, {
                "status": "fit_only_smoke_not_calibration",
                "metadata": metadata,
                "case_id": case["case_id"],
                "clean_norm_head_vs_forward_logit_max_abs": clean_logit_max_abs,
                "source_mean_vs_stage1_max_abs": {
                    site: site_results[site]["source_mean_vs_stage1_max_abs"]
                    for site in SITES
                },
                "max_zero_score_departure": {
                    site: max(abs(v - baseline_v2)
                              for v in site_results[site]["zero_v2_scores"])
                    for site in SITES
                },
                "smoke_raw_sha256": sha(smoke_raw),
                "memory_sha256": sha(memory),
            })
            return
        atomic_torch_save(checkpoint, {"metadata": metadata, "records": records})
        if (i + 1) % 8 == 0:
            print(f"calibrated {i+1}/48 FIT contexts", flush=True)
    summary = finish_summary(records)
    atomic_torch_save(raw, {"metadata": metadata, "records": records,
                            "summary": summary})
    atomic_json_save(report, {"metadata": metadata, "summary": summary,
                              "raw_sha256": sha(raw), "memory_sha256": sha(memory)})


if __name__ == "__main__":
    main()
