"""Fit-only FP32 microstep VJP diagnostic at audio encoder layers 15 and 31.

Only the original Piper red-to-blue fit pair is opened. This records actual
FP32 source/donor states and edits, plus a separately labeled hypothetical
BF16 cast of those FP32 states. It does not run the BF16 model.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from fit_fd_diagnostic_v1 import score_grad_mode_patch
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
    load_pinned_thinker,
    patch_tensor,
    record_mps_memory,
    resolve_hook_points,
    score_and_gradient,
    score_with_patch,
    sha,
    source_rows,
    verify_fit_files,
    verify_tokenizer,
)


HERE = Path(__file__).resolve().parent
ALPHAS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.025)
ZERO_REPEATS = 3
ENGINE = "piper"
RECIPIENT = "red"
DONOR = "blue"
SITES_TO_CHECK = ("encoder15", "encoder31")


def contrast(scores: dict[str, float]) -> float:
    return scores[DONOR] - scores[RECIPIENT]


def score_both(model, batch: dict, module, replacement: torch.Tensor,
               rows: list[int], base: torch.Tensor, mode: str) -> dict[str, float]:
    if mode == "grad":
        score = lambda ids: score_grad_mode_patch(
            model, batch, ids, module, replacement, rows, base
        )
    elif mode == "no_grad":
        score = lambda ids: score_with_patch(
            model, batch, ids, module, replacement, rows
        )
    else:
        raise ValueError(f"Unknown score mode {mode}")
    return {word: score(ANSWER_IDS[word]) for word in (DONOR, RECIPIENT)}


def edit_pair_with_bf16_cast(base: torch.Tensor, donor: torch.Tensor,
                             rows: list[int], alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Actual FP32 edit and hypothetical BF16-rounded edit of FP32 anchors."""
    if base.dtype != torch.float32 or donor.dtype != torch.float32:
        raise ValueError("This diagnostic requires actual FP32 source and donor states")
    _patched_fp32, effective_fp32 = patch_tensor(base, donor, rows, alpha)
    _patched_cast, effective_bf16_cast = patch_tensor(
        base.to(torch.bfloat16), donor.to(torch.bfloat16), rows, alpha
    )
    return effective_fp32, effective_bf16_cast


def diagnose_site(model, inputs: dict, group: dict, site: str,
                  memory_log: Path) -> tuple[dict, dict]:
    if site not in SITES_TO_CHECK:
        raise ValueError("Microstep is restricted to encoder15 and encoder31")
    audio_layers, point_name = SITES[site]
    point = next(
        p for p in resolve_hook_points(model, audio_layers=audio_layers, text_layers=())
        if p.name == point_name
    )
    batch = inputs[RECIPIENT]["batch"]
    donor_batch = inputs[DONOR]["batch"]
    rows = source_rows(
        inputs[RECIPIENT]["row_map"], site,
        group["word_slot_start_seconds"], group["word_slot_end_seconds"],
    )
    if rows != source_rows(
        inputs[DONOR]["row_map"], site,
        group["word_slot_start_seconds"], group["word_slot_end_seconds"],
    ):
        raise ValueError("Fit donor/recipient row mismatch")

    donor_state = capture_state(model, donor_batch, point.module)
    donor_vjp_score, base_a, donor_grad = score_and_gradient(
        model, batch, ANSWER_IDS[DONOR], point.module
    )
    record_mps_memory(memory_log, f"{site}/donor_vjp")
    recipient_vjp_score, base_b, recipient_grad = score_and_gradient(
        model, batch, ANSWER_IDS[RECIPIENT], point.module
    )
    record_mps_memory(memory_log, f"{site}/recipient_vjp")
    if not torch.equal(base_a, base_b) or base_a.shape != donor_state.shape:
        raise ValueError("Fit anchors or donor shape differ")
    base = base_a
    if base.dtype != torch.float32 or donor_state.dtype != torch.float32:
        raise ValueError("Actual encoder states are not FP32")
    grad = donor_grad.float() - recipient_grad.float()
    if not torch.isfinite(grad).all():
        raise ValueError("Nonfinite contrast gradient")

    zero_patch, _ = patch_tensor(base, donor_state, rows, 0.0)
    repeated_zero = {"grad": [], "no_grad": []}
    for repeat in range(ZERO_REPEATS):
        for mode in repeated_zero:
            repeated_zero[mode].append(
                score_both(model, batch, point.module, zero_patch, rows, base, mode)
            )
        record_mps_memory(memory_log, f"{site}/zero_repeat={repeat}")
    zero_means = {
        mode: sum(contrast(scores) for scores in values) / len(values)
        for mode, values in repeated_zero.items()
    }
    zero_spreads = {
        mode: max(abs(contrast(scores) - zero_means[mode]) for scores in values)
        for mode, values in repeated_zero.items()
    }

    snapshot = {
        "recipient_state_fp32": base.detach().cpu(),
        "donor_state_fp32": donor_state.detach().cpu(),
        "contrast_gradient_fp32": grad.detach().cpu(),
        "word_rows": torch.tensor(rows, dtype=torch.int64),
        "recipient_state_bf16_cast_of_fp32": base.detach().to(torch.bfloat16).cpu(),
        "donor_state_bf16_cast_of_fp32": donor_state.detach().to(torch.bfloat16).cpu(),
        "effective_fp32_edits": {},
        "hypothetical_bf16_cast_edits": {},
    }
    conditions = []
    for alpha in ALPHAS:
        replacement, effective = patch_tensor(base, donor_state, rows, alpha)
        actual_edit, hypothetical_bf16_edit = edit_pair_with_bf16_cast(
            base, donor_state, rows, alpha
        )
        if not torch.equal(effective, actual_edit):
            raise ValueError("FP32 edit computation is inconsistent")
        predicted = float((grad[rows] * effective).sum())
        scores = {
            mode: score_both(model, batch, point.module, replacement, rows, base, mode)
            for mode in ("grad", "no_grad")
        }
        record_mps_memory(memory_log, f"{site}/alpha={alpha}")
        key = f"alpha_{alpha:.4f}"
        snapshot["effective_fp32_edits"][key] = effective.detach().cpu()
        snapshot["hypothetical_bf16_cast_edits"][key] = hypothetical_bf16_edit.detach().cpu()
        conditions.append({
            "alpha": alpha,
            "actual_edit_dtype": str(base.dtype),
            "effective_fp32_delta_norm": float(effective.norm()),
            "effective_fp32_nonzero_elements": int(torch.count_nonzero(effective)),
            "hypothetical_bf16_cast_delta_norm": float(hypothetical_bf16_edit.norm()),
            "hypothetical_bf16_cast_nonzero_elements": int(torch.count_nonzero(hypothetical_bf16_edit)),
            "fp32_minus_bf16_cast_edit_norm": float((effective - hypothetical_bf16_edit).norm()),
            "predicted_fp32_delta_score": predicted,
            "scores": scores,
            "finite_differences": {
                mode: contrast(value) - zero_means[mode]
                for mode, value in scores.items()
            },
        })

    report = {
        "site": site,
        "word_rows": rows,
        "source_shape": list(base.shape),
        "actual_state_dtype": str(base.dtype),
        "hypothetical_cast_dtype": "torch.bfloat16",
        "bf16_cast_caveat": (
            "These are BF16 casts of FP32-model states, not activations from a BF16 model."
        ),
        "source_audio_sha256": inputs[RECIPIENT]["sha256"],
        "donor_audio_sha256": inputs[DONOR]["sha256"],
        "full_fp32_donor_minus_source_word_norm": float(
            (donor_state[rows] - base[rows]).norm()
        ),
        "full_bf16_cast_donor_minus_source_word_norm": float(
            (snapshot["donor_state_bf16_cast_of_fp32"][rows].float()
             - snapshot["recipient_state_bf16_cast_of_fp32"][rows].float()).norm()
        ),
        "contrast_gradient_word_norm": float(grad[rows].norm()),
        "vjp_path_scores": {DONOR: donor_vjp_score, RECIPIENT: recipient_vjp_score},
        "vjp_path_contrast": donor_vjp_score - recipient_vjp_score,
        "repeated_zero_scores": repeated_zero,
        "zero_contrast_means": zero_means,
        "zero_contrast_max_abs_deviation": zero_spreads,
        "conditions": conditions,
    }
    return report, snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "fit_fp32_microstep_v1_results.json",
    )
    args = parser.parse_args()
    memory_log = args.output.with_suffix(".memory.jsonl")
    snapshot_path = args.output.with_suffix(".states.pt")
    if args.output.exists() or memory_log.exists() or snapshot_path.exists():
        raise FileExistsError("Refusing to overwrite microstep result, memory log, or states")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Qwen FP32 microstep requires MPS; CPU tests do not")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers revision differs from pin")
    manifest = verify_fit_files([ENGINE])
    recommended = int(torch.mps.recommended_max_memory())
    fraction = mps_cap_fraction(recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    memory_log.write_text("")
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
    inputs, group = build_inputs(model, processor, manifest, ENGINE)
    report = {
        "status": "fit_only_complete",
        "purpose": "FP32 encoder15/31 microstep VJP convergence diagnostic",
        "engine": ENGINE, "recipient": RECIPIENT, "donor": DONOR,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "split_sha256": sha(HERE / "SPLIT_V1.json"),
        "fit_gate_script_sha256": sha(HERE / "fit_gate_v1.py"),
        "score_path_script_sha256": sha(HERE / "fit_fd_diagnostic_v1.py"),
        "fp32_microstep_script_sha256": sha(Path(__file__)),
        "prior_bf16_fit_result_sha256": sha(HERE / "fit_fd_diagnostic_v1_results.json"),
        "prior_fp32_fit_result_sha256": sha(HERE / "fit_fp32_diagnostic_v1_results.json"),
        "system_prompt": SYSTEM_PROMPT,
        "answer_token_ids": ANSWER_IDS,
        "alpha_grid": list(ALPHAS),
        "zero_repeats": ZERO_REPEATS,
        "memory_log_path": str(memory_log),
        "states_path": str(snapshot_path),
        "mps_recommended_max_bytes": recommended,
        "mps_hard_cap_fraction": fraction,
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "device": "mps", "model_dtype": "float32",
        "bf16_cast_is_hypothetical": True,
        "torch_version": torch.__version__,
        "results": [],
    }
    snapshots = {}
    for site in SITES_TO_CHECK:
        result, snapshot = diagnose_site(model, inputs, group, site, memory_log)
        report["results"].append(result)
        snapshots[site] = snapshot
    torch.save(snapshots, snapshot_path)
    report["states_sha256"] = sha(snapshot_path)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output} and {snapshot_path}")


if __name__ == "__main__":
    main()
