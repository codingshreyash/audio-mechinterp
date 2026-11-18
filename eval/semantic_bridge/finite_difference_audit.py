"""Fit-only finite-difference audit of the saved terminal-position Jacobian.

Patches the pinned Piper anchor at one text decoder layer's terminal residual.
Compares actual final-residual changes with the saved J @ *effective BF16 patch*.
No development/test data, generated answer, or waveform optimization is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "models" / "qwen_omni"))

from adapter import (  # noqa: E402
    ActivationCapture,
    load_pinned_thinker,
    prepare_audio_batch,
    resolve_hook_points,
)


CLIP = ROOT / "eval" / "tts" / "outputs" / "semantic_fit_smoke_12" / "processed" / "piper" / "0000_00a0a52903f8f5a9.wav"
CHECKPOINT = HERE / "artifacts" / "terminal_interview_piper.pt"
REPORT = HERE / "FINITE_DIFFERENCE_AUDIT.json"
SEED = 20261118
N_DIRECTIONS = 3
EPS_FRACTIONS = (0.01, 0.03)
MAX_DRIVER_GIB = 30.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(predicted: torch.Tensor, observed: torch.Tensor) -> dict[str, float | None]:
    predicted, observed = predicted.float(), observed.float()
    pred_norm = float(predicted.norm())
    obs_norm = float(observed.norm())
    difference = float((observed - predicted).norm())
    return {
        "predicted_l2": pred_norm,
        "observed_l2": obs_norm,
        "difference_l2": difference,
        "relative_error_to_observed": difference / obs_norm if obs_norm else None,
        "cosine": float(torch.dot(predicted, observed) / (pred_norm * obs_norm))
        if pred_norm and obs_norm else None,
    }


def main() -> None:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    meta = state["meta"]
    if not bool(state["completed"].all()):
        raise ValueError("Checkpoint is incomplete")
    if digest(CLIP) != meta["audio_sha256"]:
        raise ValueError("Anchor clip does not match fit checkpoint")
    if meta["position_policy"] != "same terminal prompt position for source and target":
        raise ValueError("Saved matrix has wrong position policy")
    if meta["source_layers"] != [0, 17] or meta["target_layer"] != 35:
        raise ValueError("Saved matrix has unexpected source/target layers")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Audit requires local MPS")
    limit = int(MAX_DRIVER_GIB * 2**30)
    if limit > int(0.9 * torch.mps.recommended_max_memory()):
        raise ValueError("Memory cap exceeds 90% of MPS recommendation")

    def guard(stage: str) -> dict[str, int]:
        memory = {
            "allocated": torch.mps.current_allocated_memory(),
            "driver": torch.mps.driver_allocated_memory(),
            "recommended_max": torch.mps.recommended_max_memory(),
        }
        if memory["driver"] > limit:
            raise MemoryError(f"MPS driver allocation exceeded after {stage}: {memory}")
        return memory

    model, processor = load_pinned_thinker(
        allow_download=False,
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    model.eval()
    guard("load")
    batch, clip_sha256 = prepare_audio_batch(
        processor, CLIP, system_prompt=meta["system_prompt"]
    )
    if int(batch["input_ids"].shape[1]) != meta["prompt_positions"]:
        raise ValueError("Prompt length differs from fit checkpoint")
    batch = batch.to(model.device, dtype=model.dtype)
    points = resolve_hook_points(model, audio_layers=(), text_layers=(0, 17, 35))

    def ordinary_forward() -> dict[int, torch.Tensor]:
        with torch.inference_mode(), ActivationCapture(points, detach=True) as capture:
            model.model(**batch, use_cache=False, return_dict=True)
        return {
            layer: capture.activations[f"text.residual.{layer}"][0][0, -1].float()
            for layer in (0, 17, 35)
        }

    baseline = ordinary_forward()
    repeat = ordinary_forward()
    baseline_repeat = {
        str(layer): float((baseline[layer] - repeat[layer]).abs().max())
        for layer in (0, 17, 35)
    }
    if max(baseline_repeat.values()) > 0:
        raise ValueError(f"Repeated anchor forward is not bitwise stable: {baseline_repeat}")
    guard("baseline")

    report = {
        "scope": "fit-anchor only; local terminal text-residual finite difference",
        "model_revision": meta["model_revision"],
        "transformers_revision": meta["transformers_revision"],
        "clip_sha256": clip_sha256,
        "checkpoint_sha256": digest(CHECKPOINT),
        "script_sha256": digest(Path(__file__)),
        "system_prompt": meta["system_prompt"],
        "prompt_positions": meta["prompt_positions"],
        "source_layers": meta["source_layers"],
        "target_layer": meta["target_layer"],
        "dtype": str(model.dtype),
        "device": str(model.device),
        "seed": SEED,
        "directions_per_layer": N_DIRECTIONS,
        "epsilon_fractions_of_source_l2": list(EPS_FRACTIONS),
        "baseline_repeat_max_abs": baseline_repeat,
        "baseline_terminal_target_l2": float(baseline[35].norm()),
        "baseline_terminal_target_max_abs": float(baseline[35].abs().max()),
        "records": [],
    }
    rng = torch.Generator(device="cpu").manual_seed(SEED)

    def patched_forward(layer: int, direction: torch.Tensor, signed_epsilon: float):
        observed_patch = {}

        def patch(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Pinned text block output is not a tensor")
            source = output[0, -1, :]
            desired = (source.float() + signed_epsilon * direction.to(source.device)).to(source.dtype)
            modified = output.clone()
            modified[0, -1, :] = desired
            observed_patch["effective_delta"] = (desired.float() - source.float()).cpu()
            observed_patch["base_source_max_abs_delta"] = float(
                (source.float().cpu() - baseline[layer]).abs().max()
            )
            return modified

        handle = model.model.language_model.layers[layer].register_forward_hook(patch)
        try:
            with torch.inference_mode(), ActivationCapture(
                resolve_hook_points(model, audio_layers=(), text_layers=(35,)), detach=True
            ) as capture:
                model.model(**batch, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        target = capture.activations["text.residual.35"][0][0, -1].float()
        return target, observed_patch

    zero_direction = torch.zeros(2048)
    report["zero_patch_control"] = {}
    for layer in meta["source_layers"]:
        zero_target, zero = patched_forward(layer, zero_direction, 0.0)
        report["zero_patch_control"][str(layer)] = {
            "effective_delta_l2": float(zero["effective_delta"].norm()),
            "target_max_abs_delta": float((zero_target - baseline[35]).abs().max()),
            "target_l2_delta": float((zero_target - baseline[35]).norm()),
        }
    started = time.perf_counter()
    for layer in meta["source_layers"]:
        jacobian = state["rows"][str(layer)]
        if jacobian.shape != (2048, 2048) or not bool(torch.isfinite(jacobian).all()):
            raise ValueError(f"Invalid Jacobian matrix for layer {layer}")
        source_l2 = float(baseline[layer].norm())
        for direction_index in range(N_DIRECTIONS):
            direction = torch.randn(2048, generator=rng)
            direction /= direction.norm()
            for fraction in EPS_FRACTIONS:
                epsilon = source_l2 * fraction
                plus_target, plus = patched_forward(layer, direction, epsilon)
                minus_target, minus = patched_forward(layer, direction, -epsilon)
                if plus["base_source_max_abs_delta"] or minus["base_source_max_abs_delta"]:
                    raise ValueError("Patch source differs from the saved baseline")
                delta_plus = plus["effective_delta"]
                delta_minus = minus["effective_delta"]
                observed_plus = plus_target - baseline[35]
                observed_minus = minus_target - baseline[35]
                observed_central = plus_target - minus_target
                record = {
                    "layer": layer,
                    "direction_index": direction_index,
                    "epsilon_fraction": fraction,
                    "nominal_epsilon_l2": epsilon,
                    "source_l2": source_l2,
                    "effective_plus_delta_l2": float(delta_plus.norm()),
                    "effective_minus_delta_l2": float(delta_minus.norm()),
                    "plus": metrics(jacobian @ delta_plus, observed_plus),
                    "minus": metrics(jacobian @ delta_minus, observed_minus),
                    "central": metrics(
                        jacobian @ (delta_plus - delta_minus), observed_central
                    ),
                    "memory_after_pair_bytes": guard("patch pair"),
                }
                report["records"].append(record)
                print(json.dumps({
                    "layer": layer,
                    "direction": direction_index,
                    "fraction": fraction,
                    "central": record["central"],
                }), flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {REPORT}")


if __name__ == "__main__":
    main()
