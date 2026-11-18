"""Frozen DEVELOPMENT-only FP32 audio Jacobian evaluation.

Reads eval/heldout_flow/stimuli_v1 only. It is bound to the passed fit result,
fit progress, norm baseline, and committed FP32 plan. No locked-test manifest
or waveform is opened by this program.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from fit_fp32_diagnostic_v1 import mps_cap_fraction
from fit_fp32_phase1_v1 import (
    ALPHAS, CACHE_PARITY_MAX_ABS, FIT_BASELINE_ALPHAS,
    SITES_TO_CHECK, ZERO_EDIT_MAX_SCORE_DRIFT,
    controls_rows, make_controls, predict_row_contributions,
)
from fit_gate_v1 import (
    MAX_MPS_DRIVER_BYTES, MODEL_ID, MODEL_REVISION, SITES,
    TRANSFORMERS_REVISION, ROOT, build_row_index, capture_state,
    generated_answer, load_pinned_thinker, patch_tensor,
    prepare_audio_batch, record_mps_memory, resolve_hook_points,
    score_and_gradient, score_with_patch, scored_forward, sha,
)


HERE = Path(__file__).resolve().parent
DEV_MANIFEST = ROOT / "eval" / "heldout_flow" / "stimuli_v1" / "manifest.json"
DEV_MANIFEST_SHA = "ae7cfd2e08a09e44847bd7a2e0d5775b8017a6f5e072f3f5d3dc007b9f759a01"
PLAN = HERE / "PLAN_FP32_V1.md"
PLAN_SHA = "c5cc77a37ebaa69c0d231c83cd19e0956c7a5c8ab87e1aae067f3812b6a5b34b"
FIT_RESULT = HERE / "fit_fp32_phase1_v1_results.json"
FIT_RESULT_SHA = "3cf12f5e85891d1f804111384c8fab3d09242980d44c2e3854c875310b98d520"
FIT_PROGRESS = HERE / "fit_fp32_phase1_v1_results.progress.jsonl"
FIT_PROGRESS_SHA = "2de5082c6e4b954938f02615469e096e48ed3d9d1a6ae87cc78aba6e1a66b538"
FIT_BASELINE = HERE / "fit_fp32_phase1_v1_results.baseline.json"
FIT_BASELINE_SHA = "076346729817170d488cd81ad549b0ac7f9e4daf5a402c5c21bdb244a315727c"
SYSTEM_PROMPT = "Answer the spoken question with the one word the speaker gave. Do not add anything else."
ANSWER_SURFACES = {"cat": "Cat", "dog": "Dog", "Monday": "Monday", "Friday": "Friday"}
ANSWER_IDS = {
    "cat": [26801, 151645], "dog": [48940, 151645],
    "Monday": [35239, 151645], "Friday": [34620, 151645],
}
EXPECTED_PAIRS = {
    (engine, category, phrasing)
    for engine in ("piper", "kokoro")
    for category in ("animal", "weekday")
    for phrasing in ("chosen", "remember")
}
EXPECTED_VOICES = {"piper": "en_US-amy-medium", "kokoro": "af_bella"}
EXPECTED_WORDS = {"animal": ["cat", "dog"], "weekday": ["Monday", "Friday"]}
CONTROL_PHRASING = "chosen"
ALL_PLANNED_DIRECTIONS_PER_SITE = 16
CONTROL_DIRECTIONS_PER_SITE = 8


def verify_fit_artifacts() -> tuple[dict, dict]:
    for path, expected in (
        (PLAN, PLAN_SHA), (FIT_RESULT, FIT_RESULT_SHA),
        (FIT_PROGRESS, FIT_PROGRESS_SHA), (FIT_BASELINE, FIT_BASELINE_SHA),
    ):
        if sha(path) != expected:
            raise ValueError(f"Frozen fit/plan artifact changed: {path}")
    fit = json.loads(FIT_RESULT.read_text())
    baseline = json.loads(FIT_BASELINE.read_text())
    if fit["status"] != "fit_only_gate_passed" or not fit["development_permitted_by_fit_gate"]:
        raise ValueError("Frozen fit gate did not permit development")
    if fit["fit_progress_sha256"] != FIT_PROGRESS_SHA or baseline["fit_progress_sha256"] != FIT_PROGRESS_SHA:
        raise ValueError("Fit result and baseline do not bind the frozen progress")
    if fit["baseline_sha256"] != FIT_BASELINE_SHA or fit["fp32_plan_sha256"] != PLAN_SHA:
        raise ValueError("Fit result did not bind frozen baseline/plan")
    if baseline["fp32_plan_sha256"] != PLAN_SHA or not baseline["fit_gate_pass"]:
        raise ValueError("Fit baseline did not bind a passed frozen plan")
    for path, expected in (
        (HERE / "fit_fp32_phase1_v1.py", fit["script_sha256"]),
        (HERE / "fit_gate_v1.py", fit["fit_gate_helpers_sha256"]),
        (HERE / "fit_fp32_diagnostic_v1.py", fit["fp32_cap_helper_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError(f"Fit helper code changed after frozen fit: {path}")
    for site in SITES_TO_CHECK:
        if site not in baseline["coefficients"] or site not in baseline["fit_numerical_floors"]:
            raise ValueError(f"Fit baseline missing primary site {site}")
        if baseline["fit_numerical_floors"][site] != fit["fit_gate"]["sites"][site]["numerical_floor"]:
            raise ValueError("Fit numerical floor changed")
    return fit, baseline


def verify_dev_manifest() -> dict:
    if sha(DEV_MANIFEST) != DEV_MANIFEST_SHA:
        raise ValueError("Frozen development manifest SHA-256 changed")
    manifest = json.loads(DEV_MANIFEST.read_text())
    if manifest["system_prompt"] != SYSTEM_PROMPT:
        raise ValueError("Development system prompt changed")
    pairs = manifest["pairs"]
    keys = {(p["engine"], p["pair_id"], p["phrasing_id"]) for p in pairs}
    if len(pairs) != 8 or keys != EXPECTED_PAIRS:
        raise ValueError("Development pair allocation changed")
    for pair in pairs:
        engine, category = pair["engine"], pair["pair_id"]
        if pair["voice"] != EXPECTED_VOICES[engine] or pair["words"] != EXPECTED_WORDS[category]:
            raise ValueError("Development voice/word allocation changed")
        if not pair["outside_slot_pcm_identical"] or not pair["inside_slot_pcm_differs"]:
            raise ValueError("Development exact-PCM pair invariant failed")
        for word in pair["words"]:
            info = pair["outputs"][word]
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError(f"Development WAV changed: {path}")
    return manifest


def verify_candidate_tokens(processor) -> None:
    if processor.tokenizer.eos_token_id != 151645:
        raise ValueError("Pinned end-of-turn token ID changed")
    for word, surface in ANSWER_SURFACES.items():
        actual = processor.tokenizer.encode(surface, add_special_tokens=False)
        if actual + [151645] != ANSWER_IDS[word]:
            raise ValueError(f"Candidate token IDs changed for {word}: {actual}")


def prepare_pair_inputs(model, processor, pair: dict) -> dict:
    inputs = {}
    for word in pair["words"]:
        info = pair["outputs"][word]
        batch, digest = prepare_audio_batch(
            processor, ROOT / info["path"], system_prompt=SYSTEM_PROMPT
        )
        if digest != info["file_sha256"]:
            raise ValueError("Processor development WAV identity mismatch")
        mel_frames = int(batch["feature_attention_mask"].sum())
        positions = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
        row_map = build_row_index(
            mel_frames=mel_frames, audio_token_positions=positions,
            n_window=model.config.audio_config.n_window,
            hop_length=processor.feature_extractor.hop_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )
        inputs[word] = {
            "batch": batch.to(model.device, dtype=model.dtype),
            "sha256": digest, "row_map": row_map,
        }
    a, b = pair["words"]
    if inputs[a]["row_map"] != inputs[b]["row_map"]:
        raise ValueError("Matched development row maps differ")
    if not torch.equal(inputs[a]["batch"]["input_ids"], inputs[b]["batch"]["input_ids"]):
        raise ValueError("Matched development prompt token IDs differ")
    return inputs


def score_contrast(model, batch: dict, module, replacement: torch.Tensor,
                   rows: list[int], recipient: str, donor: str) -> dict:
    donor_score = score_with_patch(
        model, batch, ANSWER_IDS[donor], module, replacement, rows
    )
    recipient_score = score_with_patch(
        model, batch, ANSWER_IDS[recipient], module, replacement, rows
    )
    return {
        "donor_answer_logprob": donor_score,
        "recipient_answer_logprob": recipient_score,
        "contrast": donor_score - recipient_score,
    }


def run_case(model, processor, pair: dict, inputs: dict, clean: dict,
             recipient: str, donor: str, site: str, beta: float,
             memory_log: Path) -> dict:
    if site not in SITES_TO_CHECK or recipient not in pair["words"] or donor not in pair["words"]:
        raise ValueError("Unexpected development site or pair direction")
    audio_layers, point_name = SITES[site]
    point = next(p for p in resolve_hook_points(
        model, audio_layers=audio_layers, text_layers=()
    ) if p.name == point_name)
    batch = inputs[recipient]["batch"]
    donor_batch = inputs[donor]["batch"]
    rows, offword_rows, offword_rule = controls_rows(
        inputs[recipient]["row_map"], site,
        pair["word_slot_start_seconds"], pair["word_slot_end_seconds"],
    )
    donor_rows, _, _ = controls_rows(
        inputs[donor]["row_map"], site,
        pair["word_slot_start_seconds"], pair["word_slot_end_seconds"],
    )
    if rows != donor_rows:
        raise ValueError("Development donor/recipient word rows differ")
    case_id = (
        f"{pair['engine']}/{pair['pair_id']}/{pair['phrasing_id']}/"
        f"{recipient}_to_{donor}/{site}"
    )
    donor_state = capture_state(model, donor_batch, point.module)
    donor_vjp_score, base_a, donor_grad = score_and_gradient(
        model, batch, ANSWER_IDS[donor], point.module
    )
    record_mps_memory(memory_log, f"{case_id}/donor_vjp")
    recipient_vjp_score, base_b, recipient_grad = score_and_gradient(
        model, batch, ANSWER_IDS[recipient], point.module
    )
    record_mps_memory(memory_log, f"{case_id}/recipient_vjp")
    if not torch.equal(base_a, base_b) or base_a.shape != donor_state.shape:
        raise ValueError("Development source anchors or donor shape differ")
    base = base_a
    if base.dtype != torch.float32 or donor_state.dtype != torch.float32:
        raise ValueError("Development source states are not FP32")
    cached_base = capture_state(model, batch, point.module, use_cache=True)
    record_mps_memory(memory_log, f"{case_id}/cached_prefill")
    cache_gap = float((cached_base - base).abs().max())
    if cache_gap > CACHE_PARITY_MAX_ABS:
        raise ValueError(f"Cached/uncached source mismatch {cache_gap} at {case_id}")
    grad = donor_grad.float() - recipient_grad.float()
    if not torch.isfinite(grad).all() or float(grad[rows].abs().max()) == 0:
        raise ValueError("Disconnected or nonfinite development contrast gradient")

    with torch.no_grad():
        clean_donor_score = float(scored_forward(model, batch, ANSWER_IDS[donor]))
        clean_recipient_score = float(scored_forward(model, batch, ANSWER_IDS[recipient]))
    no_grad_intercept = clean_donor_score - clean_recipient_score
    grad_intercept = donor_vjp_score - recipient_vjp_score
    outside = torch.ones(base.shape[0], dtype=torch.bool, device=base.device)
    outside[rows] = False
    map_key = "projected_rows" if site == "connector" else "encoder_rows"
    chunk_ids = sorted({
        inputs[recipient]["row_map"][map_key][row]["chunk_index"] for row in rows
    })
    pair_clean_eligible = all(
        clean[word]["answer"].casefold() == word.casefold()
        and clean[word]["stopped_on_eos"]
        for word in pair["words"]
    )
    conditions = []
    for alpha in ALPHAS:
        replacement, effective = patch_tensor(base, donor_state, rows, alpha)
        per_row = predict_row_contributions(grad, effective, rows)
        predicted = float(torch.stack(per_row).sum())
        scores = score_contrast(
            model, batch, point.module, replacement, rows, recipient, donor
        )
        measured = scores["contrast"] - no_grad_intercept
        generated = generated_answer(
            model, processor, batch, module=point.module,
            replacement=replacement, rows=rows,
        )
        record_mps_memory(memory_log, f"{case_id}/alpha={alpha}")
        if alpha == 0 and (
            float(effective.norm()) != 0
            or abs(measured) > ZERO_EDIT_MAX_SCORE_DRIFT
            or generated["token_ids"] != clean[recipient]["token_ids"]
        ):
            raise ValueError(f"Development alpha-zero invariant failed at {case_id}")
        norm = float(effective.norm())
        conditions.append({
            "alpha": alpha,
            "effective_fp32_edit_norm": norm,
            "effective_fp32_edit_nonzero_elements": int(torch.count_nonzero(effective)),
            "predicted_delta_score": predicted,
            "zero_baseline_predicted_delta_score": 0.0,
            "fit_norm_baseline_predicted_delta_score": beta * norm,
            "per_row_predicted_contributions": [float(x) for x in per_row],
            "measured_delta_score": measured,
            "prediction_error": measured - predicted,
            "scores": scores,
            "generated": generated,
        })

    control_results = []
    if pair["phrasing_id"] == CONTROL_PHRASING:
        for control in make_controls(base, donor_state, rows, offword_rows):
            if not control["eligible"]:
                control_results.append({
                    "name": control["name"], "eligible": False,
                    "ineligible_reason": control["note"],
                    "rows": control["rows"], "seed": control["seed"],
                    "natural_offword_edit_norm": control["natural_offword_edit_norm"],
                    "full_word_donor_edit_norm": control["full_word_donor_edit_norm"],
                })
                continue
            control_rows = control["rows"]
            effective = control["effective"]
            scores = score_contrast(
                model, batch, point.module, control["replacement"],
                control_rows, recipient, donor,
            )
            generated = generated_answer(
                model, processor, batch, module=point.module,
                replacement=control["replacement"], rows=control_rows,
            )
            record_mps_memory(memory_log, f"{case_id}/control={control['name']}")
            measured = scores["contrast"] - no_grad_intercept
            if control["name"] == "self_patch" and (
                abs(measured) > ZERO_EDIT_MAX_SCORE_DRIFT
                or generated["token_ids"] != clean[recipient]["token_ids"]
            ):
                raise ValueError(f"Development self-patch invariant failed at {case_id}")
            control_results.append({
                "name": control["name"], "eligible": True,
                "rows": control_rows, "seed": control["seed"],
                "note": control["note"],
                "natural_offword_edit_norm": control["natural_offword_edit_norm"],
                "full_word_donor_edit_norm": control["full_word_donor_edit_norm"],
                "effective_fp32_edit_norm": float(effective.norm()),
                "predicted_delta_score": float((grad[control_rows] * effective).sum()),
                "measured_delta_score": measured,
                "scores": scores,
                "generated": generated,
            })
    return {
        "case_id": case_id,
        "engine": pair["engine"], "voice": pair["voice"],
        "category": pair["pair_id"], "phrasing": pair["phrasing_id"],
        "recipient": recipient, "donor": donor,
        "site": site, "source_shape": list(base.shape),
        "word_rows": rows, "offword_control_rows": offword_rows,
        "offword_selection": offword_rule,
        "word_row_attention_chunk_ids": chunk_ids,
        "source_audio_sha256": inputs[recipient]["sha256"],
        "donor_audio_sha256": inputs[donor]["sha256"],
        "candidate_answer_token_ids": {
            "recipient": ANSWER_IDS[recipient], "donor": ANSWER_IDS[donor],
        },
        "pair_clean_eligible": pair_clean_eligible,
        "clean_recipient": clean[recipient],
        "clean_donor": clean[donor],
        "full_word_donor_edit_norm": float((donor_state[rows] - base[rows]).norm()),
        "outside_word_rows_state_delta_norm": float((donor_state[outside] - base[outside]).norm()),
        "gradient_word_rows_norm": float(grad[rows].norm()),
        "cache_prefill_max_abs_difference": cache_gap,
        "grad_intercept": grad_intercept,
        "no_grad_intercept": no_grad_intercept,
        "grad_minus_no_grad_intercept": grad_intercept - no_grad_intercept,
        "conditions": conditions,
        "controls_planned": pair["phrasing_id"] == CONTROL_PHRASING,
        "controls": control_results,
    }


def evaluate_development_gate(results: list[dict], baseline: dict) -> dict:
    """Apply frozen site floors and zero/norm comparator gates to all cases."""
    if len(results) != 2 * ALL_PLANNED_DIRECTIONS_PER_SITE:
        raise ValueError("Development results do not contain 32 site/direction cases")
    site_summaries = {}
    for site in SITES_TO_CHECK:
        cases = [case for case in results if case["site"] == site]
        if len(cases) != ALL_PLANNED_DIRECTIONS_PER_SITE or len({c["case_id"] for c in cases}) != len(cases):
            raise ValueError(f"Development site {site} lacks 16 distinct directions")
        controls_planned = sum(case["controls_planned"] for case in cases)
        if controls_planned != CONTROL_DIRECTIONS_PER_SITE:
            raise ValueError(f"Development controls allocation changed at {site}")
        floor = baseline["fit_numerical_floors"][site]
        beta = baseline["coefficients"][site]["beta"]
        if floor <= 0 or beta < 0:
            raise ValueError("Frozen fit floor or norm coefficient is invalid")
        rows = []
        flips = []
        for case in cases:
            by_alpha = {condition["alpha"]: condition for condition in case["conditions"]}
            if set(by_alpha) != set(ALPHAS) or len(case["conditions"]) != len(ALPHAS):
                raise ValueError(f"Development alpha grid incomplete: {case['case_id']}")
            full = by_alpha[1.0]["generated"]
            donor_flip = full["answer"].casefold() == case["donor"].casefold() and full["stopped_on_eos"]
            flips.append({
                "case_id": case["case_id"], "clean_eligible": case["pair_clean_eligible"],
                "complete_donor_flip": donor_flip, "generated": full,
            })
            for alpha in FIT_BASELINE_ALPHAS:
                condition = by_alpha[alpha]
                measured = condition["measured_delta_score"]
                predicted = condition["predicted_delta_score"]
                norm_prediction = beta * condition["effective_fp32_edit_norm"]
                determinate = abs(measured) > floor
                rows.append({
                    "case_id": case["case_id"], "category": case["category"],
                    "phrasing": case["phrasing"], "engine": case["engine"],
                    "recipient": case["recipient"], "donor": case["donor"],
                    "alpha": alpha, "clean_eligible": case["pair_clean_eligible"],
                    "measured_delta_score": measured,
                    "predicted_delta_score": predicted,
                    "zero_predicted_delta_score": 0.0,
                    "fit_norm_predicted_delta_score": norm_prediction,
                    "determinate": determinate,
                    "sign_match": determinate and predicted * measured > 0,
                    "jac_abs_error": abs(predicted - measured),
                    "zero_abs_error": abs(measured),
                    "norm_abs_error": abs(norm_prediction - measured),
                    "jac_relative_error": abs(predicted - measured) / max(abs(measured), floor),
                })
        eligible_cases = sum(case["pair_clean_eligible"] for case in cases)
        eligible_rows = [row for row in rows if row["clean_eligible"]]
        determinate_rows = [row for row in eligible_rows if row["determinate"]]
        sign_matches = sum(row["sign_match"] for row in determinate_rows)
        sign_rate = sign_matches / len(determinate_rows) if determinate_rows else None
        jac_mae = statistics.fmean(row["jac_abs_error"] for row in eligible_rows) if eligible_rows else None
        zero_mae = statistics.fmean(row["zero_abs_error"] for row in eligible_rows) if eligible_rows else None
        norm_mae = statistics.fmean(row["norm_abs_error"] for row in eligible_rows) if eligible_rows else None
        median_abs_error = statistics.median(row["jac_abs_error"] for row in eligible_rows) if eligible_rows else None
        median_relative_error = statistics.median(row["jac_relative_error"] for row in eligible_rows) if eligible_rows else None
        reasons = []
        if eligible_cases == 0:
            reasons.append("no_clean_eligible_directions")
        if not determinate_rows:
            reasons.append("no_determinate_small_step_effects")
        elif sign_rate < 0.75:
            reasons.append("small_step_sign_agreement_below_75_percent")
        if zero_mae is None or zero_mae == 0:
            reasons.append("zero_baseline_mae_zero_or_unavailable")
        elif jac_mae > 0.75 * zero_mae:
            reasons.append("jac_mae_not_25_percent_better_than_zero")
        if norm_mae is None or norm_mae == 0:
            reasons.append("norm_baseline_mae_zero_or_unavailable")
        elif jac_mae > 0.75 * norm_mae:
            reasons.append("jac_mae_not_25_percent_better_than_fit_norm")
        breakdown = []
        for category in EXPECTED_WORDS:
            for engine in EXPECTED_VOICES:
                group = [case for case in cases if case["category"] == category and case["engine"] == engine]
                group_eligible = sum(case["pair_clean_eligible"] for case in group)
                group_flips = sum(
                    flip["complete_donor_flip"] for flip in flips
                    if flip["case_id"] in {case["case_id"] for case in group}
                )
                breakdown.append({
                    "category": category, "engine": engine,
                    "all_planned_directions": len(group),
                    "clean_eligible_directions": group_eligible,
                    "alpha_one_complete_donor_flips_all_planned": group_flips,
                })
        phrasing_breakdown = []
        for category in EXPECTED_WORDS:
            for phrasing in ("chosen", "remember"):
                for engine in EXPECTED_VOICES:
                    group = [
                        case for case in cases
                        if case["category"] == category
                        and case["phrasing"] == phrasing
                        and case["engine"] == engine
                    ]
                    group_ids = {case["case_id"] for case in group}
                    phrasing_breakdown.append({
                        "category": category, "phrasing": phrasing,
                        "engine": engine,
                        "all_planned_directions": len(group),
                        "clean_eligible_directions": sum(
                            case["pair_clean_eligible"] for case in group
                        ),
                        "alpha_one_complete_donor_flips_all_planned": sum(
                            flip["complete_donor_flip"] for flip in flips
                            if flip["case_id"] in group_ids
                        ),
                    })
        all_flips = sum(f["complete_donor_flip"] for f in flips)
        eligible_flips = sum(
            f["complete_donor_flip"] for f in flips if f["clean_eligible"]
        )
        site_summaries[site] = {
            "pass": not reasons,
            "failure_reasons": reasons,
            "numerical_floor_from_fit": floor,
            "beta_from_fit": beta,
            "all_planned_directions": len(cases),
            "clean_eligible_directions": eligible_cases,
            "all_planned_small_step_conditions": len(rows),
            "clean_eligible_small_step_conditions": len(eligible_rows),
            "determinate_clean_small_step_conditions": len(determinate_rows),
            "indeterminate_clean_small_step_conditions": len(eligible_rows) - len(determinate_rows),
            "determinate_sign_matches": sign_matches,
            "determinate_sign_rate": sign_rate,
            "jac_mean_abs_error": jac_mae,
            "zero_mean_abs_error": zero_mae,
            "fit_norm_mean_abs_error": norm_mae,
            "jac_median_abs_error": median_abs_error,
            "jac_median_relative_error": median_relative_error,
            "relative_error_denominator": "max(abs(measured_delta_score), frozen_fit_site_floor)",
            "alpha_one_complete_donor_flips_all_planned": all_flips,
            "alpha_one_complete_donor_flips_clean_eligible": eligible_flips,
            "alpha_one_all_planned_denominator": len(flips),
            "alpha_one_clean_eligible_denominator": eligible_cases,
            "alpha_one_complete_donor_flip_rate_all_planned": all_flips / len(flips),
            "alpha_one_complete_donor_flip_rate_clean_eligible": (
                eligible_flips / eligible_cases if eligible_cases else None
            ),
            "controls_planned_directions": controls_planned,
            "category_voice_breakdown": breakdown,
            "category_phrasing_voice_breakdown": phrasing_breakdown,
            "small_step_rows": rows,
            "alpha_one_rows": flips,
        }
    passed = all(summary["pass"] for summary in site_summaries.values())
    return {"pass": passed, "locked_test_permitted_by_development_gate": passed,
            "sites": site_summaries}


def hash_complete_dev_progress(progress_log: Path, results: list[dict]) -> str:
    if len(results) != 2 * ALL_PLANNED_DIRECTIONS_PER_SITE:
        raise ValueError("Development progress missing planned cases")
    saved = [json.loads(line) for line in progress_log.read_text().splitlines()]
    if saved != results:
        raise ValueError("Development progress JSONL differs from completed cases")
    return sha(progress_log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "development_fp32_v1_results.json",
    )
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    fit, baseline = verify_fit_artifacts()
    manifest = verify_dev_manifest()
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision differs from pin")
    # Verify exact canonical answer and EOS tokens before loading/running weights.
    from transformers import Qwen2_5OmniProcessor
    preflight_processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    verify_candidate_tokens(preflight_processor)
    if args.validate_inputs_only:
        print(json.dumps({
            "status": "development_inputs_verified_without_model",
            "dev_manifest_sha256": DEV_MANIFEST_SHA,
            "fit_result_sha256": FIT_RESULT_SHA,
            "fit_progress_sha256": FIT_PROGRESS_SHA,
            "fit_baseline_sha256": FIT_BASELINE_SHA,
            "fp32_plan_sha256": PLAN_SHA,
            "answer_token_ids": ANSWER_IDS,
            "planned_pairs": len(manifest["pairs"]),
            "planned_directions_per_site": ALL_PLANNED_DIRECTIONS_PER_SITE,
        }, indent=2))
        return

    memory_log = args.output.with_suffix(".memory.jsonl")
    progress_log = args.output.with_suffix(".progress.jsonl")
    if any(path.exists() for path in (args.output, memory_log, progress_log)):
        raise FileExistsError("Refusing to overwrite development result or sidecar")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Development candidate requires MPS; CPU checks do not")
    recommended = int(torch.mps.recommended_max_memory())
    fraction = mps_cap_fraction(recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    memory_log.write_text("")
    progress_log.write_text("")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32:
        raise ValueError(f"Expected FP32 model, got {model.dtype}")
    record_mps_memory(memory_log, "after_model_load")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    verify_candidate_tokens(processor)
    report = {
        "status": "development_running",
        "purpose": "FP32 audio-to-answer VJP transfer on frozen development clips",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "development_manifest_sha256": DEV_MANIFEST_SHA,
        "fit_result_sha256": FIT_RESULT_SHA,
        "fit_progress_sha256": FIT_PROGRESS_SHA,
        "fit_baseline_sha256": FIT_BASELINE_SHA,
        "fp32_plan_sha256": PLAN_SHA,
        "development_script_sha256": sha(Path(__file__)),
        "fit_gate_helper_sha256": sha(HERE / "fit_gate_v1.py"),
        "fit_phase1_helper_sha256": sha(HERE / "fit_fp32_phase1_v1.py"),
        "system_prompt": SYSTEM_PROMPT,
        "answer_surfaces": ANSWER_SURFACES,
        "answer_token_ids": ANSWER_IDS,
        "primary_sites": list(SITES_TO_CHECK),
        "alpha_grid": list(ALPHAS),
        "locality_alpha_grid": list(FIT_BASELINE_ALPHAS),
        "fit_site_numerical_floors": baseline["fit_numerical_floors"],
        "fit_site_norm_coefficients": {
            site: baseline["coefficients"][site]["beta"] for site in SITES_TO_CHECK
        },
        "controls_phrasing": CONTROL_PHRASING,
        "all_planned_directions_per_site": ALL_PLANNED_DIRECTIONS_PER_SITE,
        "control_directions_per_site": CONTROL_DIRECTIONS_PER_SITE,
        "memory_log_path": str(memory_log),
        "progress_log_path": str(progress_log),
        "mps_recommended_max_bytes": recommended,
        "mps_hard_cap_fraction": fraction,
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "device": "mps", "dtype": "float32", "torch_version": torch.__version__,
        "pair_clean_answers": [], "results": [],
    }
    for pair in manifest["pairs"]:
        inputs = prepare_pair_inputs(model, processor, pair)
        clean = {}
        for word in pair["words"]:
            clean[word] = generated_answer(model, processor, inputs[word]["batch"])
            record_mps_memory(
                memory_log,
                f"{pair['engine']}/{pair['pair_id']}/{pair['phrasing_id']}/{word}/clean_generation",
            )
        pair_eligible = all(
            clean[word]["answer"].casefold() == word.casefold()
            and clean[word]["stopped_on_eos"]
            for word in pair["words"]
        )
        report["pair_clean_answers"].append({
            "engine": pair["engine"], "voice": pair["voice"],
            "category": pair["pair_id"], "phrasing": pair["phrasing_id"],
            "words": pair["words"], "clean": clean,
            "pair_clean_eligible": pair_eligible,
            "audio_sha256": {word: inputs[word]["sha256"] for word in pair["words"]},
        })
        a, b = pair["words"]
        for recipient, donor in ((a, b), (b, a)):
            for site in SITES_TO_CHECK:
                result = run_case(
                    model, processor, pair, inputs, clean,
                    recipient, donor, site,
                    baseline["coefficients"][site]["beta"], memory_log,
                )
                report["results"].append(result)
                with progress_log.open("a") as handle:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
                print(f"completed {result['case_id']}", flush=True)
    report["development_progress_sha256"] = hash_complete_dev_progress(
        progress_log, report["results"]
    )
    gate = evaluate_development_gate(report["results"], baseline)
    report["development_gate"] = gate
    report["locked_test_permitted_by_development_gate"] = gate["pass"]
    report["status"] = "development_gate_passed" if gate["pass"] else "development_gate_failed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}; gate pass={gate['pass']}")


if __name__ == "__main__":
    main()
