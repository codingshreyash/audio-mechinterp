"""Phase-1 FIT-only FP32 audio Jacobian evaluation for Qwen2.5-Omni-3B.

Runs the original exact-PCM red/blue pair in Piper and Kokoro, both directions,
at connector and encoder31. It opens no development or locked-test audio.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from fit_fp32_diagnostic_v1 import mps_cap_fraction
from fit_gate_v1 import (
    ANSWER_IDS,
    FIT_MANIFEST_SHA,
    MAX_MPS_DRIVER_BYTES,
    MODEL_ID,
    MODEL_REVISION,
    SITES,
    SYSTEM_PROMPT,
    TRANSFORMERS_REVISION,
    build_inputs,
    capture_state,
    generated_answer,
    load_pinned_thinker,
    patch_tensor,
    record_mps_memory,
    resolve_hook_points,
    score_and_gradient,
    score_with_patch,
    scored_forward,
    sha,
    verify_fit_files,
    verify_tokenizer,
)


HERE = Path(__file__).resolve().parent
PLAN = HERE / "PLAN_FP32_V1.md"
PLAN_SHA = "c5cc77a37ebaa69c0d231c83cd19e0956c7a5c8ab87e1aae067f3812b6a5b34b"
ENGINES = ("piper", "kokoro")
DIRECTIONS = (("red", "blue"), ("blue", "red"))
SITES_TO_CHECK = ("connector", "encoder31")
ALPHAS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.10, 1.0)
FIT_BASELINE_ALPHAS = (0.005, 0.01, 0.025)
ZERO_REPEATS = 3
CACHE_PARITY_MAX_ABS = 1e-4
ZERO_EDIT_MAX_SCORE_DRIFT = 1e-4
RANDOM_SEEDS = (0, 1, 2)
ROW_SHUFFLE_SEED = 0


def normalised_random(shape: tuple[int, int], target_norm: float,
                      seed: int, device: torch.device) -> torch.Tensor:
    """Fixed CPU Gaussian direction, transported to the site device in FP32."""
    if target_norm < 0 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Invalid random edit shape/norm")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(shape, generator=generator, dtype=torch.float32)
    norm = float(direction.norm())
    if norm == 0:
        raise ValueError("Random direction has zero norm")
    return (direction * (target_norm / norm)).to(device)


def replacement_from_delta(base: torch.Tensor, rows: list[int],
                           delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if base.dtype != torch.float32 or base.ndim != 2:
        raise ValueError("Control source must be a two-dimensional FP32 tensor")
    if not rows or len(set(rows)) != len(rows) or min(rows) < 0 or max(rows) >= base.shape[0]:
        raise ValueError("Invalid control rows")
    if delta.shape != (len(rows), base.shape[1]):
        raise ValueError("Control delta shape mismatch")
    replacement = base.clone()
    replacement[rows] = base[rows] + delta.to(base.device, dtype=base.dtype)
    effective = replacement[rows] - base[rows]
    return replacement, effective


def make_controls(base: torch.Tensor, donor: torch.Tensor,
                  word_rows: list[int], offword_rows: list[int] | None) -> list[dict]:
    """Construct prespecified controls with explicit natural and applied norms."""
    if base.shape != donor.shape or base.dtype != torch.float32 or donor.dtype != torch.float32:
        raise ValueError("Controls require matched FP32 source/donor tensors")
    if len(word_rows) < 2:
        raise ValueError("Word window must allow a row shuffle")
    if offword_rows is not None and len(word_rows) != len(offword_rows):
        raise ValueError("Control windows must be equal-size")
    if offword_rows is not None and set(word_rows) & set(offword_rows):
        raise ValueError("Off-word control overlaps word rows")
    word_delta = donor[word_rows] - base[word_rows]
    word_norm = float(word_delta.norm())
    if word_norm <= 0:
        raise ValueError("Word donor edit has zero norm")
    offword_delta = None if offword_rows is None else donor[offword_rows] - base[offword_rows]
    offword_norm = None if offword_delta is None else float(offword_delta.norm())
    specs: list[tuple[str, list[int], torch.Tensor, int | None, str]] = [
        ("self_patch", word_rows, torch.zeros_like(word_delta), None, "exact source self-edit"),
    ]
    if offword_rows is not None and offword_delta is not None:
        specs.append((
            "wrong_time_natural", offword_rows, offword_delta, None,
            "natural donor edit at prespecified off-word rows",
        ))
    for seed in RANDOM_SEEDS:
        specs.append((
            f"random_same_rows_seed_{seed}", word_rows,
            normalised_random(tuple(word_delta.shape), word_norm, seed, base.device),
            seed, "Gaussian direction on word rows, norm-matched to donor word edit",
        ))
    permutation = np.random.default_rng(ROW_SHUFFLE_SEED).permutation(len(word_rows)).tolist()
    specs.append((
        "row_shuffled_donor", word_rows, word_delta[permutation],
        ROW_SHUFFLE_SEED, f"donor word-row edits permuted by {permutation}",
    ))
    if offword_rows is not None and offword_delta is not None and offword_norm > 0:
        synthetic_delta = offword_delta * (word_norm / offword_norm)
        synthetic_note = "natural off-word edit rescaled to donor word-edit norm"
        specs.append((
            "norm_scaled_offword_synthetic", offword_rows, synthetic_delta,
            None, synthetic_note,
        ))
    controls = []
    for name, rows, delta, seed, note in specs:
        replacement, effective = replacement_from_delta(base, rows, delta)
        controls.append({
            "name": name, "rows": rows, "replacement": replacement,
            "effective": effective, "seed": seed, "note": note,
            "eligible": True,
            "natural_offword_edit_norm": offword_norm,
            "full_word_donor_edit_norm": word_norm,
        })
    if offword_rows is None:
        for name in ("wrong_time_natural", "norm_scaled_offword_synthetic"):
            controls.append({
                "name": name, "rows": None, "replacement": None,
                "effective": None, "seed": None,
                "note": "ineligible_no_offword_window", "eligible": False,
                "natural_offword_edit_norm": None,
                "full_word_donor_edit_norm": word_norm,
            })
    elif offword_norm == 0:
        controls.append({
            "name": "norm_scaled_offword_synthetic", "rows": offword_rows,
            "replacement": None, "effective": None, "seed": None,
            "note": "ineligible_zero_offword_delta", "eligible": False,
            "natural_offword_edit_norm": 0.0,
            "full_word_donor_edit_norm": word_norm,
        })
    return controls


def controls_rows(row_map: dict, site: str, start: float,
                  end: float) -> tuple[list[int], list[int] | None, str]:
    """Lowest contiguous nonoverlap, preferring chunks disjoint from word chunks."""
    if site not in SITES_TO_CHECK:
        raise ValueError(f"Unexpected primary site {site}")
    entries = row_map["projected_rows" if site == "connector" else "encoder_rows"]
    if site == "connector":
        period = row_map["nominal_mel_frame_period_seconds"]
        word_rows = [
            entry["row_index"] for entry in entries
            if start <= entry["nominal_center_mel_frame"] * period < end
        ]
    else:
        word_rows = [
            entry["row_index"] for entry in entries
            if start <= entry["nominal_center_seconds"] < end
        ]
    if not word_rows:
        raise ValueError("No source rows fall in the nominal word slot")
    word_set = set(word_rows)
    word_chunks = {entries[i]["chunk_index"] for i in word_rows}
    width = len(word_rows)
    candidates = []
    for first in range(len(entries) - width + 1):
        candidate = list(range(first, first + width))
        if word_set.intersection(candidate):
            continue
        chunk_set = {entries[i]["chunk_index"] for i in candidate}
        candidates.append((candidate, chunk_set))
    if not candidates:
        return word_rows, None, "ineligible_no_offword_window"
    disjoint = [c for c in candidates if c[1].isdisjoint(word_chunks) and len(c[1]) == 1]
    if disjoint:
        return word_rows, disjoint[0][0], "disjoint_attention_chunk_lowest_start"
    return word_rows, candidates[0][0], "lowest_nonoverlapping_fallback"


def score_contrast(model, batch: dict, module, replacement: torch.Tensor,
                   rows: list[int], recipient: str, donor_word: str) -> dict:
    donor_score = score_with_patch(
        model, batch, ANSWER_IDS[donor_word], module, replacement, rows
    )
    recipient_score = score_with_patch(
        model, batch, ANSWER_IDS[recipient], module, replacement, rows
    )
    return {
        "donor_answer_logprob": donor_score,
        "recipient_answer_logprob": recipient_score,
        "contrast": donor_score - recipient_score,
    }


def predict_row_contributions(grad: torch.Tensor, effective: torch.Tensor,
                              rows: list[int]) -> list[torch.Tensor]:
    if grad.ndim != 2 or effective.shape != (len(rows), grad.shape[1]):
        raise ValueError("Gradient/edit row shapes differ")
    if not rows or min(rows) < 0 or max(rows) >= grad.shape[0]:
        raise ValueError("Prediction rows outside source tensor")
    return [(grad[row] * effective[i]).sum() for i, row in enumerate(rows)]


def run_case(model, processor, inputs: dict, group: dict, engine: str,
             recipient: str, donor_word: str, site: str,
             memory_log: Path, clean_answer: dict) -> dict:
    audio_layers, point_name = SITES[site]
    point = next(p for p in resolve_hook_points(
        model, audio_layers=audio_layers, text_layers=()
    ) if p.name == point_name)
    batch = inputs[recipient]["batch"]
    donor_batch = inputs[donor_word]["batch"]
    start = group["word_slot_start_seconds"]
    end = group["word_slot_end_seconds"]
    rows, offword_rows, offword_selection = controls_rows(
        inputs[recipient]["row_map"], site, start, end
    )
    donor_rows, _, _ = controls_rows(inputs[donor_word]["row_map"], site, start, end)
    if rows != donor_rows:
        raise ValueError("Donor/recipient nominal word rows differ")
    case_id = f"{engine}/{recipient}_to_{donor_word}/{site}"
    donor_state = capture_state(model, donor_batch, point.module)
    donor_vjp_score, source_a, donor_grad = score_and_gradient(
        model, batch, ANSWER_IDS[donor_word], point.module
    )
    record_mps_memory(memory_log, f"{case_id}/donor_vjp")
    recipient_vjp_score, source_b, recipient_grad = score_and_gradient(
        model, batch, ANSWER_IDS[recipient], point.module
    )
    record_mps_memory(memory_log, f"{case_id}/recipient_vjp")
    if not torch.equal(source_a, source_b) or donor_state.shape != source_a.shape:
        raise ValueError("Source anchor or donor tensor shape mismatch")
    base = source_a
    if base.dtype != torch.float32 or donor_state.dtype != torch.float32:
        raise ValueError("Primary source tensors must be FP32")
    cached_base = capture_state(model, batch, point.module, use_cache=True)
    record_mps_memory(memory_log, f"{case_id}/cached_prefill")
    cache_gap = float((cached_base - base).abs().max())
    if cache_gap > CACHE_PARITY_MAX_ABS:
        raise ValueError(f"Cached/uncached source mismatch {cache_gap} at {case_id}")
    grad = donor_grad.float() - recipient_grad.float()
    if not torch.isfinite(grad).all() or float(grad[rows].abs().max()) == 0:
        raise ValueError("Disconnected or nonfinite word-row contrast gradient")

    with torch.no_grad():
        clean_donor_score = float(scored_forward(model, batch, ANSWER_IDS[donor_word]))
        clean_recipient_score = float(scored_forward(model, batch, ANSWER_IDS[recipient]))
    no_grad_intercept = clean_donor_score - clean_recipient_score
    grad_intercept = donor_vjp_score - recipient_vjp_score
    zero_patch, _ = patch_tensor(base, donor_state, rows, 0.0)
    zero_repeats = []
    for repeat in range(ZERO_REPEATS):
        scores = score_contrast(
            model, batch, point.module, zero_patch, rows, recipient, donor_word
        )
        zero_repeats.append(scores)
        record_mps_memory(memory_log, f"{case_id}/zero_repeat={repeat}")
    zero_deviations = [abs(x["contrast"] - no_grad_intercept) for x in zero_repeats]
    zero_repeat_mean = sum(x["contrast"] for x in zero_repeats) / ZERO_REPEATS
    zero_max_from_mean = max(abs(x["contrast"] - zero_repeat_mean) for x in zero_repeats)
    if max(zero_deviations) > ZERO_EDIT_MAX_SCORE_DRIFT:
        raise ValueError(f"Zero patch changed score at {case_id}: {zero_deviations}")

    outside = torch.ones(base.shape[0], dtype=torch.bool, device=base.device)
    outside[rows] = False
    map_key = "projected_rows" if site == "connector" else "encoder_rows"
    chunk_ids = sorted({
        inputs[recipient]["row_map"][map_key][row]["chunk_index"] for row in rows
    })
    conditions = []
    for alpha in ALPHAS:
        replacement, effective = patch_tensor(base, donor_state, rows, alpha)
        row_terms = predict_row_contributions(grad, effective, rows)
        predicted = float(torch.stack(row_terms).sum())
        scores = score_contrast(
            model, batch, point.module, replacement, rows, recipient, donor_word
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
            or generated["token_ids"] != clean_answer["token_ids"]
        ):
            raise ValueError(f"Alpha-zero score/answer invariant failed at {case_id}")
        conditions.append({
            "alpha": alpha,
            "effective_fp32_edit_norm": float(effective.norm()),
            "effective_fp32_edit_nonzero_elements": int(torch.count_nonzero(effective)),
            "predicted_delta_score": predicted,
            "per_row_predicted_contributions": [float(x) for x in row_terms],
            "measured_delta_score": measured,
            "prediction_error": measured - predicted,
            "scores": scores,
            "generated": generated,
        })

    control_results = []
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
            control_rows, recipient, donor_word,
        )
        generated = generated_answer(
            model, processor, batch, module=point.module,
            replacement=control["replacement"], rows=control_rows,
        )
        record_mps_memory(memory_log, f"{case_id}/control={control['name']}")
        measured = scores["contrast"] - no_grad_intercept
        if control["name"] == "self_patch" and (
            abs(measured) > ZERO_EDIT_MAX_SCORE_DRIFT
            or generated["token_ids"] != clean_answer["token_ids"]
        ):
            raise ValueError(f"Self-patch invariant failed at {case_id}")
        control_results.append({
            "name": control["name"], "eligible": True,
            "rows": control_rows,
            "seed": control["seed"], "note": control["note"],
            "natural_offword_edit_norm": control["natural_offword_edit_norm"],
            "full_word_donor_edit_norm": control["full_word_donor_edit_norm"],
            "effective_fp32_edit_norm": float(effective.norm()),
            "effective_fp32_edit_nonzero_elements": int(torch.count_nonzero(effective)),
            "predicted_delta_score": float((grad[control_rows] * effective).sum()),
            "measured_delta_score": measured,
            "scores": scores,
            "generated": generated,
        })
    return {
        "case_id": case_id,
        "engine": engine, "recipient": recipient, "donor": donor_word,
        "site": site, "source_shape": list(base.shape),
        "word_rows": rows, "offword_control_rows": offword_rows,
        "offword_selection": offword_selection,
        "word_row_attention_chunk_ids": chunk_ids,
        "source_audio_sha256": inputs[recipient]["sha256"],
        "donor_audio_sha256": inputs[donor_word]["sha256"],
        "candidate_answer_token_ids": {
            "recipient": ANSWER_IDS[recipient], "donor": ANSWER_IDS[donor_word],
        },
        "full_word_donor_edit_norm": float((donor_state[rows] - base[rows]).norm()),
        "outside_word_rows_state_delta_norm": float((donor_state[outside] - base[outside]).norm()),
        "gradient_word_rows_norm": float(grad[rows].norm()),
        "cache_prefill_max_abs_difference": cache_gap,
        "grad_intercept": grad_intercept,
        "no_grad_intercept": no_grad_intercept,
        "grad_minus_no_grad_intercept": grad_intercept - no_grad_intercept,
        "clean_generated": clean_answer,
        "repeated_zero_scores": zero_repeats,
        "zero_contrast_max_abs_deviation_from_mean": zero_max_from_mean,
        "zero_patch_max_abs_difference_from_unpatched": max(zero_deviations),
        "conditions": conditions,
        "controls": control_results,
    }


def fit_norm_baseline(results: list[dict]) -> dict:
    coefficients = {}
    for site in SITES_TO_CHECK:
        rows = []
        for case in results:
            if case["site"] != site:
                continue
            for condition in case["conditions"]:
                if condition["alpha"] in FIT_BASELINE_ALPHAS:
                    rows.append({
                        "case_id": case["case_id"],
                        "audio_sha256": case["source_audio_sha256"],
                        "donor_audio_sha256": case["donor_audio_sha256"],
                        "word_rows": case["word_rows"],
                        "alpha": condition["alpha"],
                        "n": condition["effective_fp32_edit_norm"],
                        "z": condition["measured_delta_score"],
                    })
        if len(rows) != len(ENGINES) * len(DIRECTIONS) * len(FIT_BASELINE_ALPHAS):
            raise ValueError(f"Norm baseline missing fit rows at {site}")
        numerator = sum(r["n"] * r["z"] for r in rows)
        denominator = sum(r["n"] ** 2 for r in rows)
        coefficients[site] = {
            "beta": max(0.0, numerator / denominator) if denominator > 0 else 0.0,
            "numerator": numerator, "denominator": denominator,
            "fit_rows": rows,
        }
    return coefficients


def hash_complete_fit_progress(progress_log: Path, results: list[dict]) -> str:
    expected_count = len(ENGINES) * len(DIRECTIONS) * len(SITES_TO_CHECK)
    if len(results) != expected_count:
        raise ValueError("Fit progress does not contain all eight planned cases")
    saved = [json.loads(line) for line in progress_log.read_text().splitlines()]
    if saved != results:
        raise ValueError("Fit progress JSONL differs from completed in-memory cases")
    return sha(progress_log)


def fit_gate_summary(results: list[dict], clean_answers: list[dict]) -> dict:
    """Apply the frozen plan's fit stop rule without selecting cases afterward."""
    expected_routes = {
        (engine, recipient, donor)
        for engine in ENGINES for recipient, donor in DIRECTIONS
    }
    clean_eligible = sum(bool(item["eligible"]) for item in clean_answers)
    clean_pass = len(clean_answers) == 4 and clean_eligible == 4
    site_summaries = {}
    for site in SITES_TO_CHECK:
        cases = [case for case in results if case["site"] == site]
        routes = {(c["engine"], c["recipient"], c["donor"]) for c in cases}
        complete_routes = len(cases) == 4 and routes == expected_routes
        if not complete_routes:
            raise ValueError(f"Fit gate lacks four distinct routes at {site}")
        max_zero_departure = max(
            case["zero_contrast_max_abs_deviation_from_mean"] for case in cases
        )
        floor = max(1e-4, 5 * max_zero_departure)
        flip_rows = []
        locality_rows = []
        for case in cases:
            by_alpha = {condition["alpha"]: condition for condition in case["conditions"]}
            if set(ALPHAS) != set(by_alpha) or len(case["conditions"]) != len(ALPHAS):
                raise ValueError(f"Fit gate alpha grid incomplete at {case['case_id']}")
            generated = by_alpha[1.0]["generated"]
            flipped = (
                generated["answer"].casefold() == case["donor"]
                and generated["stopped_on_eos"]
            )
            flip_rows.append({"case_id": case["case_id"], "flipped_to_donor": flipped,
                              "generated": generated})
            for alpha in FIT_BASELINE_ALPHAS:
                condition = by_alpha[alpha]
                measured = condition["measured_delta_score"]
                predicted = condition["predicted_delta_score"]
                determinate = abs(measured) > floor
                sign_match = determinate and predicted * measured > 0
                locality_rows.append({
                    "case_id": case["case_id"], "alpha": alpha,
                    "measured_delta_score": measured,
                    "predicted_delta_score": predicted,
                    "determinate": determinate, "sign_match": sign_match,
                })
        flips = sum(row["flipped_to_donor"] for row in flip_rows)
        determinate_count = sum(row["determinate"] for row in locality_rows)
        sign_matches = sum(row["sign_match"] for row in locality_rows)
        sign_rate = sign_matches / determinate_count if determinate_count else None
        reasons = []
        if flips != 4:
            reasons.append("fewer_than_four_alpha_one_complete_donor_flips")
        if determinate_count == 0:
            reasons.append("no_determinate_small_step_effects")
        elif sign_rate < 0.75:
            reasons.append("small_step_sign_agreement_below_75_percent")
        site_summaries[site] = {
            "pass": not reasons,
            "failure_reasons": reasons,
            "numerical_floor": floor,
            "max_repeated_zero_contrast_departure": max_zero_departure,
            "complete_donor_flips": flips,
            "planned_donor_flips": 4,
            "small_step_planned": len(locality_rows),
            "small_step_determinate": determinate_count,
            "small_step_indeterminate": len(locality_rows) - determinate_count,
            "small_step_sign_matches": sign_matches,
            "small_step_sign_rate": sign_rate,
            "flip_rows": flip_rows,
            "locality_rows": locality_rows,
        }
    overall_pass = clean_pass and all(x["pass"] for x in site_summaries.values())
    return {
        "pass": overall_pass,
        "development_permitted_by_fit_gate": overall_pass,
        "clean_complete_eligible": clean_eligible,
        "clean_complete_planned": 4,
        "clean_gate_pass": clean_pass,
        "sites": site_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "fit_fp32_phase1_v1_results.json",
    )
    args = parser.parse_args()
    memory_log = args.output.with_suffix(".memory.jsonl")
    progress_log = args.output.with_suffix(".progress.jsonl")
    baseline_path = args.output.with_suffix(".baseline.json")
    if any(path.exists() for path in (args.output, memory_log, progress_log, baseline_path)):
        raise FileExistsError("Refusing to overwrite phase-1 result or sidecar")
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 Qwen phase-1 requires MPS; CPU tests do not")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision differs from pin")
    manifest = verify_fit_files(list(ENGINES))
    if sha(PLAN) != PLAN_SHA:
        raise ValueError("FP32 plan SHA-256 changed after runner was frozen")
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
    verify_tokenizer(processor)
    report = {
        "status": "fit_only_running",
        "purpose": "FP32 local audio-to-answer VJP, four original fit routes",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "split_sha256": sha(HERE / "SPLIT_V1.json"),
        "fp32_plan_sha256": PLAN_SHA,
        "script_sha256": sha(Path(__file__)),
        "fit_gate_helpers_sha256": sha(HERE / "fit_gate_v1.py"),
        "fp32_cap_helper_sha256": sha(HERE / "fit_fp32_diagnostic_v1.py"),
        "system_prompt": SYSTEM_PROMPT,
        "answer_token_ids": ANSWER_IDS,
        "engines": list(ENGINES),
        "directions": [f"{r}_to_{d}" for r, d in DIRECTIONS],
        "primary_sites": list(SITES_TO_CHECK),
        "alpha_grid": list(ALPHAS),
        "fit_baseline_alphas": list(FIT_BASELINE_ALPHAS),
        "zero_repeats": ZERO_REPEATS,
        "cache_parity_max_abs": CACHE_PARITY_MAX_ABS,
        "zero_edit_max_score_drift": ZERO_EDIT_MAX_SCORE_DRIFT,
        "random_seeds": list(RANDOM_SEEDS),
        "row_shuffle_seed": ROW_SHUFFLE_SEED,
        "offword_window_rule": "lowest contiguous nonoverlap; prefer one wholly disjoint attention chunk",
        "memory_log_path": str(memory_log),
        "progress_log_path": str(progress_log),
        "baseline_path": str(baseline_path),
        "mps_recommended_max_bytes": recommended,
        "mps_hard_cap_fraction": fraction,
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "device": "mps", "dtype": "float32", "torch_version": torch.__version__,
        "clean_answers": [], "results": [],
    }
    prepared = {}
    for engine in ENGINES:
        inputs, group = build_inputs(model, processor, manifest, engine)
        prepared[engine] = (inputs, group)
        for word in ANSWER_IDS:
            clean = generated_answer(model, processor, inputs[word]["batch"])
            record_mps_memory(memory_log, f"{engine}/{word}/clean_generation")
            report["clean_answers"].append({
                "engine": engine, "word": word,
                "audio_sha256": inputs[word]["sha256"],
                "generated": clean,
                "eligible": clean["answer"].casefold() == word and clean["stopped_on_eos"],
            })
    if not all(item["eligible"] for item in report["clean_answers"]):
        report["status"] = "fit_clean_gate_failed"
        report["development_permitted_by_fit_gate"] = False
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    clean_lookup = {
        (item["engine"], item["word"]): item["generated"]
        for item in report["clean_answers"]
    }
    for engine in ENGINES:
        inputs, group = prepared[engine]
        for recipient, donor_word in DIRECTIONS:
            for site in SITES_TO_CHECK:
                result = run_case(
                    model, processor, inputs, group, engine,
                    recipient, donor_word, site, memory_log,
                    clean_lookup[(engine, recipient)],
                )
                report["results"].append(result)
                with progress_log.open("a") as handle:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
                print(f"completed {result['case_id']}", flush=True)
    fit_progress_sha256 = hash_complete_fit_progress(progress_log, report["results"])
    report["fit_progress_sha256"] = fit_progress_sha256
    fit_gate = fit_gate_summary(report["results"], report["clean_answers"])
    report["fit_gate"] = fit_gate
    report["development_permitted_by_fit_gate"] = fit_gate["pass"]
    baseline = {
        "status": "fit_only_frozen_candidate",
        "formula": "beta=max(0,sum(n*z)/sum(n*n)); prediction=beta*n",
        "source_result_script_sha256": report["script_sha256"],
        "fit_progress_sha256": fit_progress_sha256,
        "fit_gate_pass": fit_gate["pass"],
        "fit_numerical_floors": {
            site: summary["numerical_floor"]
            for site, summary in fit_gate["sites"].items()
        },
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "fp32_plan_sha256": report["fp32_plan_sha256"],
        "coefficients": fit_norm_baseline(report["results"]),
    }
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    report["baseline_sha256"] = sha(baseline_path)
    report["status"] = (
        "fit_only_gate_passed" if fit_gate["pass"] else "fit_only_gate_failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
