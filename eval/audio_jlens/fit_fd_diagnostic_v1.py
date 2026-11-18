"""Fit-only directional finite differences under two Qwen execution modes.

Compares a grad-enabled source-leaf score path with the existing no-grad patch
path on the original Piper red-to-blue pair. No held-out files are opened.
This script is diagnostic; it does not replace the frozen fit gate or its data.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from fit_gate_v1 import (
    ALPHAS as GATE_ALPHAS,
    ANSWER_IDS,
    FIT_MANIFEST_SHA,
    MODEL_ID,
    MODEL_REVISION,
    SITES,
    SYSTEM_PROMPT,
    TRANSFORMERS_REVISION,
    _output_tensor,
    _replace_output,
    build_inputs,
    capture_state,
    load_pinned_thinker,
    patch_tensor,
    record_mps_memory,
    resolve_hook_points,
    score_and_gradient,
    score_with_patch,
    scored_forward,
    sha,
    source_rows,
    verify_fit_files,
    verify_tokenizer,
)


HERE = Path(__file__).resolve().parent
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10)
ZERO_REPEATS = 3
ENGINE = "piper"
RECIPIENT = "red"
DONOR = "blue"
SITES_TO_CHECK = ("connector", "encoder15")


def score_grad_mode_patch(model, batch: dict, candidate_ids: list[int], module,
                          replacement: torch.Tensor, rows: list[int],
                          expected_state: torch.Tensor) -> float:
    """Score with the VJP's source leaf and grad-enabled downstream kernels.

    No backward is taken. A clone plus row assignment is used at every alpha,
    including zero, so all finite-difference scores have the same hook path.
    """
    calls = []

    def hook(_module, _inputs, output):
        state = _output_tensor(output).detach().requires_grad_(True)
        if state.shape != replacement.shape or not torch.equal(state, expected_state):
            raise ValueError("Grad-path source state differs from fit anchor")
        changed = state.clone()
        changed[rows] = replacement[rows].to(state.device, dtype=state.dtype)
        calls.append(True)
        return _replace_output(output, changed)

    handle = module.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            result = float(scored_forward(model, batch, candidate_ids).detach())
    finally:
        handle.remove()
    if len(calls) != 1:
        raise ValueError(f"Expected one grad-mode hook call, got {len(calls)}")
    return result


def _contrast(scores: dict[str, float]) -> float:
    return scores[DONOR] - scores[RECIPIENT]


def _score_both(model, batch: dict, module, replacement: torch.Tensor,
                rows: list[int], base: torch.Tensor, mode: str) -> dict[str, float]:
    score = (
        (lambda ids: score_grad_mode_patch(
            model, batch, ids, module, replacement, rows, base
        )) if mode == "grad" else
        (lambda ids: score_with_patch(model, batch, ids, module, replacement, rows))
    )
    return {word: score(ANSWER_IDS[word]) for word in (DONOR, RECIPIENT)}


def diagnose_site(model, processor, inputs: dict, group: dict, site: str,
                  memory_log: Path) -> dict:
    audio_layers, point_name = SITES[site]
    points = resolve_hook_points(model, audio_layers=audio_layers, text_layers=())
    point = next(p for p in points if p.name == point_name)
    batch = inputs[RECIPIENT]["batch"]
    donor_batch = inputs[DONOR]["batch"]
    rows = source_rows(
        inputs[RECIPIENT]["row_map"], site,
        group["word_slot_start_seconds"], group["word_slot_end_seconds"],
    )
    donor_rows = source_rows(
        inputs[DONOR]["row_map"], site,
        group["word_slot_start_seconds"], group["word_slot_end_seconds"],
    )
    if rows != donor_rows:
        raise ValueError("Fit donor/recipient rows differ")

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
    grad = donor_grad.float() - recipient_grad.float()
    if not torch.isfinite(grad).all():
        raise ValueError("Nonfinite contrast gradient")

    zero_patch, zero_effective = patch_tensor(base, donor_state, rows, 0.0)
    if float(zero_effective.norm()) != 0:
        raise ValueError("Alpha-zero edit is not exactly zero")
    zero_scores: dict[str, list[dict[str, float]]] = {"grad": [], "no_grad": []}
    for repeat in range(ZERO_REPEATS):
        for mode in zero_scores:
            zero_scores[mode].append(
                _score_both(model, batch, point.module, zero_patch, rows, base, mode)
            )
        record_mps_memory(memory_log, f"{site}/zero_repeat={repeat}")
    zero_baselines = {
        mode: sum(_contrast(x) for x in values) / len(values)
        for mode, values in zero_scores.items()
    }
    zero_max_abs_deviation = {
        mode: max(abs(_contrast(x) - zero_baselines[mode]) for x in values)
        for mode, values in zero_scores.items()
    }

    conditions = []
    for alpha in ALPHAS:
        replacement, effective = patch_tensor(base, donor_state, rows, alpha)
        pred = float((grad[rows] * effective).sum())
        scores = {
            mode: _score_both(model, batch, point.module, replacement, rows, base, mode)
            for mode in ("grad", "no_grad")
        }
        record_mps_memory(memory_log, f"{site}/alpha={alpha}")
        conditions.append({
            "alpha": alpha,
            "effective_bf16_delta_norm": float(effective.norm()),
            "effective_nonzero_elements": int(torch.count_nonzero(effective)),
            "predicted_delta_score": pred,
            "scores": scores,
            "contrasts": {mode: _contrast(value) for mode, value in scores.items()},
            "finite_differences": {
                mode: _contrast(value) - zero_baselines[mode]
                for mode, value in scores.items()
            },
        })

    return {
        "site": site,
        "rows": rows,
        "source_shape": list(base.shape),
        "source_audio_sha256": inputs[RECIPIENT]["sha256"],
        "donor_audio_sha256": inputs[DONOR]["sha256"],
        "full_donor_minus_source_norm": float((donor_state[rows].float() - base[rows].float()).norm()),
        "contrast_gradient_norm": float(grad[rows].norm()),
        "vjp_path_scores": {DONOR: donor_vjp_score, RECIPIENT: recipient_vjp_score},
        "vjp_path_contrast": donor_vjp_score - recipient_vjp_score,
        "zero_scores_repeated": zero_scores,
        "zero_contrast_means": zero_baselines,
        "zero_contrast_max_abs_deviation": zero_max_abs_deviation,
        "vjp_minus_grad_patch_zero_contrast": (
            donor_vjp_score - recipient_vjp_score - zero_baselines["grad"]
        ),
        "grad_minus_no_grad_zero_contrast": (
            zero_baselines["grad"] - zero_baselines["no_grad"]
        ),
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "fit_fd_diagnostic_v1_results.json",
    )
    args = parser.parse_args()
    memory_log = args.output.with_suffix(".memory.jsonl")
    if args.output.exists() or memory_log.exists():
        raise FileExistsError("Refusing to overwrite diagnostic result or memory log")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Qwen diagnostic requires MPS; CPU tests do not")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers revision differs from pin")
    manifest = verify_fit_files([ENGINE])
    memory_log.write_text("")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    record_mps_memory(memory_log, "after_model_load")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    verify_tokenizer(processor)
    inputs, group = build_inputs(model, processor, manifest, ENGINE)
    report = {
        "status": "fit_only_complete",
        "purpose": "same-path grad versus no-grad directional finite differences",
        "engine": ENGINE, "recipient": RECIPIENT, "donor": DONOR,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "split_sha256": sha(HERE / "SPLIT_V1.json"),
        "fit_gate_script_sha256": sha(HERE / "fit_gate_v1.py"),
        "diagnostic_script_sha256": sha(Path(__file__)),
        "system_prompt": SYSTEM_PROMPT,
        "answer_token_ids": ANSWER_IDS,
        "alpha_grid": list(ALPHAS),
        "original_gate_alpha_grid": list(GATE_ALPHAS),
        "zero_repeats": ZERO_REPEATS,
        "memory_log_path": str(memory_log),
        "device": "mps", "dtype": "bfloat16",
        "results": [],
    }
    for site in SITES_TO_CHECK:
        report["results"].append(
            diagnose_site(model, processor, inputs, group, site, memory_log)
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
