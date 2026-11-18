"""Old-FIT Voxtral numerical follow-up, v2 BF16 load-policy amendment.

--validate-only is CPU-only. --run is one fresh, bounded MPS attempt. Never
point this runner at fresh v2 data or use its output to tune a v2 evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import voxtral_fit_pilot_capability_v1 as base
import voxtral_fit_v2_causal_first_record as first
import voxtral_v1_fit_backward_feasibility as old


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_FOLLOWUP_PROTOCOL_V1.md"
AMENDMENT = HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_FOLLOWUP_AMENDMENT_V2.md"
OLD_RESULT = HERE / "runs/voxtral_v1_fit_backward_feasibility/20261118T1122Z/result.json"
OLD_DETAILS = OLD_RESULT.with_name("details.pt")
RUNS = HERE / "runs/voxtral_v1_fit_backward_numerical_followup_v2"
PROTOCOL_SHA = "5778a4b84e535f7182a3daa2b973637a1c9127118ce2434ce82ba80fee467a82"
AMENDMENT_SHA = "8c1515c7f967392213189aefea8d84df814e86c4a4ce96ad6bcc62e068e33ffa"
MODELING_UTILS_SHA = "553412f58334f65abba617f3b344828698dbf63615e109159fd03dafd3422d38"
POSITION_NAME = "model.audio_tower.embed_positions.weight"
POSITION_SHAPE = [1500, 1280]
OLD_RUNNER_SHA = "cc48437b96d47c766f5ba9f86eb5789481fecd7d1469c09a503d225a42ed3be1"
OLD_RESULT_SHA = "d4ccfd213cf3a5ebebd3a7c0ecebab691dc15632f61076669d7535bd58154ebe"
OLD_DETAILS_SHA = "1bc49aa650b96c7ac6cb2f545da98e198edfa18998ebae8dc263c70ad5ad7b3c"
SOURCE_SHAS = {
    "modeling_voxtral": "2d3ec9d78c469b77c42a8d5ba258185fb1c58f4c2c7bf6007ec724436dfc7056",
    "processing_voxtral": "239e84657d1a0287ab27a540c946a787a69b4b0bf45727d6b24bee21ccec0296",
    "feature_extraction_whisper": "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb",
    "modeling_llama": "affc6a4ab36b562a6b4c90b84c4e7939c5417a69ad7bd37e858be5583f189830",
}
SITES = ("encoder31", "projector")
COORDS = (0, 1, 2, 3)
ORDER = (0.00625, -0.00625, 0.025, -0.025, 0.1, -0.1)
CYCLES = (ORDER, tuple(reversed(ORDER)), ORDER)
ARMS = {"bf16": ("bfloat16", 900), "fp32": ("float32", 1200)}
MAX_DRIVER = 30 * 1024**3


def preflight() -> dict:
    """Verify only old FIT inputs and pinned sources; never load weights."""
    import torch
    import transformers.modeling_utils as modeling_utils

    if (base.sha(PROTOCOL) != PROTOCOL_SHA or base.sha(AMENDMENT) != AMENDMENT_SHA or
            base.sha(Path(modeling_utils.__file__)) != MODELING_UTILS_SHA or
            base.sha(Path(old.__file__)) != OLD_RUNNER_SHA or
            base.sha(OLD_RESULT) != OLD_RESULT_SHA or base.sha(OLD_DETAILS) != OLD_DETAILS_SHA):
        raise ValueError("Follow-up protocol or old-FIT provenance changed")
    prepared = old.preflight()
    if prepared["sources"] != SOURCE_SHAS:
        raise ValueError("Installed Voxtral/Whisper/Llama source hashes changed")
    if prepared["record"]["words"] != ["beaver", "monkey"]:
        raise ValueError("Frozen old-FIT word orientation changed")
    rows = {s: prepared["record"]["site_maps"][s]["bins"]["word_0"] for s in SITES}
    if rows != {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]}:
        raise ValueError("Frozen old-FIT source rows changed")
    old_result = json.loads(OLD_RESULT.read_text())
    if (old_result["status"] != "completed_numerical_feasibility_only" or
            old_result["details_sha256"] != OLD_DETAILS_SHA or
            old_result["natural_suffix_ids"] != prepared["saved"]["generated_token_ids"]):
        raise ValueError("Old FIT result provenance changed")
    sidecar = torch.load(OLD_DETAILS, map_location="cpu", weights_only=True)
    directions = {}
    for site in SITES:
        recipient = sidecar["source_states"][site]
        donor = sidecar["donor_states"][site]
        if (first.tensor_sha(recipient) != old_result["source_hashes"][site] or
                first.tensor_sha(donor) != old_result["donor_hashes"][site] or
                recipient.dtype != torch.bfloat16 or donor.dtype != torch.bfloat16 or
                tuple(recipient.shape) != first.expected_hook_shape(site)[-2:]):
            raise ValueError(f"Old FIT {site} source sidecar changed")
        direction = donor[rows[site]].float().mean(0) - recipient[rows[site]].float().mean(0)
        if not torch.isfinite(direction).all() or float(torch.linalg.vector_norm(direction)) <= 0:
            raise ValueError(f"Invalid frozen {site} direction")
        directions[site] = direction.contiguous()
    return {**prepared, "rows": rows, "directions": directions,
            "old_result": old_result, "old_source_hashes": old_result["source_hashes"]}


def parameter_layout(model, arm: str) -> dict:
    """Record every loaded parameter and reject any unplanned dtype or device."""
    import torch

    if arm not in ARMS:
        raise ValueError("Unknown arithmetic arm")
    parameters = []
    errors = []
    counts = {}
    seen = set()
    for name, param in model.named_parameters():
        if name in seen:
            errors.append(f"duplicate parameter name: {name}")
        seen.add(name)
        expected = torch.float32 if arm == "fp32" or name == POSITION_NAME else torch.bfloat16
        if param.dtype != expected or param.device.type != "mps":
            errors.append(f"unexpected dtype/device: {name}: {param.dtype}/{param.device}")
        shape = list(param.shape)
        if name == POSITION_NAME and shape != POSITION_SHAPE:
            errors.append(f"positional embedding shape changed: {shape}")
        descriptor = f"{param.dtype}/{param.device.type}"
        counts[descriptor] = counts.get(descriptor, 0) + 1
        parameters.append({"name": name, "dtype": str(param.dtype),
                           "device": str(param.device), "shape": shape, "numel": param.numel()})
    if POSITION_NAME not in seen:
        errors.append("required positional embedding absent")
    if len(parameters) != 762:
        errors.append(f"named parameter count changed: {len(parameters)}")
    return {"arm": arm, "valid": not errors, "errors": errors,
            "named_parameter_count": len(parameters),
            "parameter_count": sum(item["numel"] for item in parameters),
            "dtype_device_counts": counts,
            "expected_fp32_exception": POSITION_NAME if arm == "bf16" else None,
            "parameters": parameters}


def realized_edit(source_rows, direction, alpha: float, dtype):
    """Return native-dtype patched rows and the actual FP32 difference."""
    import torch

    if source_rows.dtype != dtype or direction.dtype != torch.float32 or not math.isfinite(alpha):
        raise ValueError("Edit dtype or alpha invalid")
    source_fp32 = source_rows.float()
    intended = alpha * direction
    edited = (source_fp32 + intended).to(dtype)
    effective = edited.float() - source_fp32
    if alpha == 0 and not torch.equal(edited, source_rows):
        raise ValueError("Zero source edit was not exact")
    mean = effective.mean(0)
    mean_norm = float(torch.linalg.vector_norm(mean))
    intended_norm = float(torch.linalg.vector_norm(intended))
    return {"edited": edited, "effective": effective, "intended": intended,
            "mean": mean, "effective_norm": float(torch.linalg.vector_norm(effective)),
            "mean_fidelity_ratio": float(torch.linalg.vector_norm(mean - intended)) / max(intended_norm, 1e-6),
            "row_nonuniformity_ratio": float(torch.linalg.vector_norm(effective - mean)) /
            math.sqrt(source_rows.shape[0]) / max(mean_norm, 1e-6),
            "unchanged_fraction": float((effective == 0).float().mean())}


def contract(grad_rows, edit) -> dict[str, list[float]]:
    """Four FP32 contractions of BF16 or FP32 source VJPs."""
    import torch

    g = grad_rows.float()
    e = edit["effective"].float()
    if g.ndim != 3 or g.shape[0] != 4 or tuple(g.shape[1:]) != tuple(e.shape):
        raise ValueError("Selected gradient/edit shapes differ")
    summed = g.sum(1)
    return {"rowwise": (g * e.unsqueeze(0)).sum((1, 2)).tolist(),
            "uniform": (summed * edit["mean"]).sum(1).tolist(),
            "ideal": (summed * edit["intended"]).sum(1).tolist()}


def _norm(values) -> float:
    import torch

    return float(torch.linalg.vector_norm(torch.as_tensor(values, dtype=torch.float64)))


def ulp_spacing(native_values) -> list[float]:
    """Largest adjacent representable spacing at each native-dtype value."""
    import torch

    values = native_values.detach().cpu()
    if values.shape != (4,) or values.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("ULP values require four native FP32 or BF16 coordinates")
    up = torch.nextafter(values, torch.full_like(values, float("inf")))
    down = torch.nextafter(values, torch.full_like(values, float("-inf")))
    return torch.maximum((up.float() - values.float()).abs(),
                         (values.float() - down.float()).abs()).tolist()


def analyze_site(records: list[dict], source_grad) -> dict:
    """Per-repeat and median metrics with one fixed clean-state source VJP."""
    import torch

    if len(records) != 18 or source_grad.shape[0] != 4:
        raise ValueError("Expected eighteen paired observations")
    zeros = torch.stack([r["zero_terminal"] for r in records]).double()
    z4 = zeros[:, list(COORDS)]
    noise = float(torch.cdist(z4, z4).max())
    noise_full = float(torch.cdist(zeros, zeros).max())
    groups = {}
    for alpha in ORDER:
        selected = [r for r in records if r["alpha"] == alpha]
        if len(selected) != 3 or sorted(r["cycle"] for r in selected) != [0, 1, 2]:
            raise ValueError("Signed-alpha repeat group incomplete")
        predicted = {name: torch.tensor([r["prediction"][name] for r in selected], dtype=torch.float64)
                     for name in ("rowwise", "uniform", "ideal")}
        measured = torch.stack([(r["alpha_terminal"] - r["zero_terminal"])[list(COORDS)]
                                for r in selected]).double()
        median_predictions = {name: values.median(dim=0).values for name, values in predicted.items()}
        p_median = median_predictions["rowwise"]
        d_median = measured.median(dim=0).values
        p_norm, d_norm = _norm(p_median), _norm(d_median)
        resolution = max(3 * noise, 1e-6)
        cosine = None
        if p_norm > resolution and d_norm > resolution:
            cosine = float(torch.dot(p_median, d_median) / (p_norm * d_norm))
        median_error = _norm(p_median - d_median)
        repeats = []
        for i, r in enumerate(selected):
            d = measured[i]
            p = predicted["rowwise"][i]
            repeats.append({"cycle": r["cycle"], "paired_delta_four": d.tolist(),
                            "rowwise_prediction_four": p.tolist(),
                            "uniform_prediction_four": r["prediction"]["uniform"],
                            "ideal_prediction_four": r["prediction"]["ideal"],
                            "absolute_l2_error": _norm(p - d), "measured_l2": _norm(d),
                            "uniform_absolute_l2_error": _norm(predicted["uniform"][i] - d),
                            "ideal_absolute_l2_error": _norm(predicted["ideal"][i] - d),
                            "edit_metrics": r["edit_metrics"],
                            "zero_terminal_sha256": r["zero_sha"], "alpha_terminal_sha256": r["alpha_sha"],
                            "zero_source_sha256": r["zero_source_sha"],
                            "alpha_source_sha256": r["alpha_source_sha"],
                            "zero_terminal_ulp_four": r["zero_ulp_four"],
                            "measured_change_in_ulp_four": [float(x / u) if u > 0 else None
                                                            for x, u in zip(d.tolist(), r["zero_ulp_four"])]})
        groups[str(alpha)] = {"per_repeat": repeats,
                              "median_prediction_four": p_median.tolist(),
                              "median_uniform_prediction_four": median_predictions["uniform"].tolist(),
                              "median_ideal_prediction_four": median_predictions["ideal"].tolist(),
                              "median_measured_four": d_median.tolist(),
                              "median_absolute_l2_error": median_error,
                              "median_relative_l2_error": median_error / max(d_norm, noise, 1e-6),
                              "median_uniform_absolute_l2_error": _norm(median_predictions["uniform"] - d_median),
                              "median_uniform_relative_l2_error":
                              _norm(median_predictions["uniform"] - d_median) / max(d_norm, noise, 1e-6),
                              "median_ideal_absolute_l2_error": _norm(median_predictions["ideal"] - d_median),
                              "median_ideal_relative_l2_error":
                              _norm(median_predictions["ideal"] - d_median) / max(d_norm, noise, 1e-6),
                              "median_measured_l2": d_norm,
                              "below_repeatability_resolution": d_norm < 3 * noise,
                              "zero_measured_signal": d_norm == 0,
                              "cosine_if_resolved": cosine}
    odd_even = {}
    for magnitude in (0.00625, 0.025, 0.1):
        positive, negative = groups[str(magnitude)], groups[str(-magnitude)]
        dp = torch.tensor(positive["median_measured_four"])
        dn = torch.tensor(negative["median_measured_four"])
        pp = torch.tensor(positive["median_prediction_four"])
        pn = torch.tensor(negative["median_prediction_four"])
        odd_even[str(magnitude)] = {"measured_odd_four": ((dp - dn) / 2).tolist(),
                                    "measured_even_four": ((dp + dn) / 2).tolist(),
                                    "predicted_rowwise_odd_four": ((pp - pn) / 2).tolist()}
    return {"zero_repeatability_l2_four": noise, "zero_repeatability_l2_full": noise_full,
            "alpha_groups": groups, "odd_even": odd_even}


def check_limit(torch, memory, started: float, limit_seconds: int) -> dict:
    torch.mps.synchronize()
    elapsed = time.monotonic() - started
    driver = int(torch.mps.driver_allocated_memory())
    peak = max(driver, memory.driver_max)
    if peak >= MAX_DRIVER:
        raise MemoryError("30 GiB MPS driver allocation cap reached")
    if elapsed >= limit_seconds:
        raise TimeoutError(f"{limit_seconds}-second arm cap reached")
    return {"elapsed_seconds": elapsed, "driver_bytes": driver,
            "current_bytes": int(torch.mps.current_allocated_memory()), "peak_driver_bytes": peak}


def capture(model, batch, dtype, *, grad: bool, patch_site: str | None = None,
            rows: list[int] | None = None, direction=None, alpha: float = 0.0) -> dict:
    """One no-cache prompt forward, preserving encoder→projector→LM graph."""
    import torch

    hits = {"encoder31": [], "projector": [], "lm29": []}
    prepatch_rows = {}
    edit = None
    prompt_len = int(batch["input_ids"].shape[1])
    if prompt_len != 396:
        raise ValueError("Prompt length changed")
    if patch_site is not None and (patch_site not in SITES or not rows or direction is None):
        raise ValueError("Patch specification incomplete")

    def source_hook(site):
        def hook(_module, _inputs, output):
            nonlocal edit
            if not isinstance(output, torch.Tensor) or output.dtype != dtype or tuple(output.shape) != first.expected_hook_shape(site):
                raise ValueError(f"Unexpected {site} source hook shape/dtype")
            value = output.detach().requires_grad_(True) if grad and site == "encoder31" else output
            hits[site].append(value)
            if patch_site != site:
                return value
            original = value[0, rows] if site == "encoder31" else value[rows]
            prepatch_rows[site] = original.detach().cpu().clone()
            edit = realized_edit(original, direction.to(original.device), alpha, dtype)
            edited = edit["edited"].to(value.device)
            patched = value.clone()
            if site == "encoder31":
                patched[0, rows] = edited
            else:
                patched[rows] = edited
            edit = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in edit.items()}
            return patched
        return hook

    def terminal_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.dtype != dtype or tuple(output.shape) != (1, 396, 3072):
            raise ValueError("Unexpected LM29 hook shape/dtype")
        hits["lm29"].append(output)

    handles = [model.model.audio_tower.layers[31].register_forward_hook(source_hook("encoder31")),
               model.model.multi_modal_projector.register_forward_hook(source_hook("projector")),
               model.model.language_model.layers[29].register_forward_hook(terminal_hook)]
    try:
        gpu_batch = {k: v.to("mps") for k, v in batch.items()}
        with torch.enable_grad() if grad else torch.no_grad():
            model(**gpu_batch, use_cache=False, logits_to_keep=1)
        torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != 1 for values in hits.values()):
        raise ValueError("Prompt hook count changed")
    terminal = hits["lm29"][0][0, -1]
    if any(not torch.isfinite(values[0]).all() for values in hits.values()):
        raise ValueError("Nonfinite hook state")
    if grad and not all(hits[s][0].requires_grad for s in ("encoder31", "projector")):
        raise ValueError("Shared source graph disconnected")
    return {"encoder31": hits["encoder31"][0], "projector": hits["projector"][0],
            "terminal": terminal, "terminal_native_cpu": terminal.detach().cpu().clone(),
            "terminal_fp32_cpu": terminal.detach().float().cpu().clone(),
            "prepatch_rows": prepatch_rows, "edit": edit,
            "events": {s: {"calls": 1, "shape": list(hits[s][0].shape), "dtype": str(dtype)}
                       for s in hits}}


def generate(model, batch, dtype, *, passive: bool) -> dict:
    import torch

    hits = {"encoder31": [], "projector": [], "lm29": []}
    handles = []
    if passive:
        def hook(site):
            def record(_module, _inputs, output):
                if not isinstance(output, torch.Tensor) or output.dtype != dtype:
                    raise ValueError(f"Unexpected passive {site} output dtype")
                hits[site].append(list(output.shape))
                return output
            return record
        handles = [model.model.audio_tower.layers[31].register_forward_hook(hook("encoder31")),
                   model.model.multi_modal_projector.register_forward_hook(hook("projector")),
                   model.model.language_model.layers[29].register_forward_hook(hook("lm29"))]
    try:
        with torch.inference_mode():
            output = model.generate(**{k: v.to("mps") for k, v in batch.items()},
                                    do_sample=False, use_cache=True, max_new_tokens=base.MAX_NEW_TOKENS,
                                    eos_token_id=2, pad_token_id=int(model.generation_config.pad_token_id))
        torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if passive and (hits["encoder31"] != [[1, 1500, 1280]] or
                    hits["projector"] != [[375, 3072]] or
                    not hits["lm29"] or hits["lm29"][0] != [1, 396, 3072]):
        raise ValueError("Passive generation hook parity failed")
    return {"suffix_ids": output[0, int(batch["input_ids"].shape[1]):].tolist(),
            "passive_hook_shapes": hits}


def run(arm: str, output: Path) -> None:
    if arm not in ARMS:
        raise ValueError("Unknown arithmetic arm")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.jsonl"
    progress.touch(exist_ok=False)
    base.append_line(progress, {"phase": "attempt_started", "arm": arm,
                                "wall_utc": datetime.now(timezone.utc).isoformat(),
                                "script_sha256": base.sha(Path(__file__)), "protocol_sha256": PROTOCOL_SHA,
                                "amendment_sha256": AMENDMENT_SHA,
                                "elapsed_seconds": time.monotonic() - started})
    max_seconds = ARMS[arm][1]

    def wall_check():
        elapsed = time.monotonic() - started
        if elapsed >= max_seconds:
            raise TimeoutError(f"{max_seconds}-second arm cap reached")
        return elapsed

    try:
        prepared = preflight()
        base.append_line(progress, {"phase": "preflight_validated", "elapsed_seconds": wall_check()})
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        recommended = int(torch.mps.recommended_max_memory())
        if recommended <= 0:
            raise RuntimeError("MPS recommended memory unavailable")
        fraction = min(1.0, MAX_DRIVER / recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        base.append_line(progress, {"phase": "mps_ready", "elapsed_seconds": wall_check(),
                                    "recommended_bytes": recommended, "fraction_limit": fraction})
        from transformers import VoxtralForConditionalGeneration
        dtype = torch.bfloat16 if arm == "bf16" else torch.float32
        metadata = {"arm": arm, "purpose": "old_fit_numerical_diagnostic_only",
                    "script_sha256": base.sha(Path(__file__)), "protocol_sha256": PROTOCOL_SHA,
                    "amendment_sha256": AMENDMENT_SHA,
                    "modeling_utils_sha256": MODELING_UTILS_SHA,
                    "expected_parameter_layout": {
                        "named_parameter_count": 762,
                        "fp32_exception": POSITION_NAME if arm == "bf16" else None,
                        "fp32_exception_shape": POSITION_SHAPE if arm == "bf16" else None,
                        "all_parameters_on": "mps"},
                    "old_result_sha256": OLD_RESULT_SHA, "old_details_sha256": OLD_DETAILS_SHA,
                    "model_snapshot": prepared["snapshot"],
                    "source_hashes": prepared["sources"], "rows": prepared["rows"],
                    "wav_hashes": {clip["word"]: clip["wav_sha256"] for clip in prepared["record"]["clips"]},
                    "direction_hashes": {s: first.tensor_sha(prepared["directions"][s]) for s in SITES},
                    "max_seconds": max_seconds, "max_mps_driver_bytes": MAX_DRIVER,
                    "mps_fraction_limit": fraction, "fresh_v2_corpus_opened": False,
                    "planned_patch_forwards": 72, "planned_pairs": 36,
                    "planned_other_prompt_forwards": 5, "planned_vjps": 4,
                    "planned_generations": 2}
        base.exclusive_json(output / "meta.json", metadata)

        with base.MemorySampler(torch) as memory:
            def receipt(phase, **extra):
                base.append_line(progress, {"phase": phase, **check_limit(torch, memory, started, max_seconds),
                                            **extra})

            receipt("before_model_load")
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(base.SNAPSHOT), local_files_only=True, dtype=dtype,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
            model.requires_grad_(False)
            layout = parameter_layout(model, arm)
            base.exclusive_json(output / "model_layout.json", layout)
            if not layout["valid"]:
                raise ValueError("Full-pipeline parameter dtype/device changed: " +
                                 "; ".join(layout["errors"][:5]))
            receipt("model_loaded", parameter_count=layout["parameter_count"],
                    named_parameter_count=layout["named_parameter_count"],
                    model_layout_sha256=base.sha(output / "model_layout.json"))
            batch = prepared["batches"]["beaver"]
            natural = generate(model, batch, dtype, passive=False)
            receipt("natural_generation")
            passive = generate(model, batch, dtype, passive=True)
            receipt("passive_generation")
            if natural["suffix_ids"] != passive["suffix_ids"]:
                raise ValueError("Within-arm natural/passive suffix mismatch")
            if arm == "bf16" and natural["suffix_ids"] != prepared["saved"]["generated_token_ids"]:
                raise ValueError("BF16 saved natural suffix parity failed")
            donor = capture(model, prepared["batches"]["monkey"], dtype, grad=False)
            donor_terminal = donor["terminal_fp32_cpu"]
            donor_sources = {s: first.tensor_sha((donor[s][0] if s == "encoder31" else donor[s]).detach().cpu())
                             for s in SITES}
            del donor
            receipt("donor_capture")
            no_grad_terminals = []
            for i in range(3):
                no_grad = capture(model, batch, dtype, grad=False)
                no_grad_terminals.append(no_grad["terminal_fp32_cpu"])
                del no_grad
                receipt(f"no_grad_clean_{i}")
            clean = capture(model, batch, dtype, grad=True)
            clean_terminal = clean["terminal_fp32_cpu"]
            clean_source_hashes = {s: first.tensor_sha((clean[s][0] if s == "encoder31" else clean[s]).detach().cpu())
                                   for s in SITES}
            clean_selected_source_rows = {
                s: (clean[s][0] if s == "encoder31" else clean[s]).detach().cpu()[prepared["rows"][s]].clone()
                for s in SITES}
            clean_events = clean["events"]
            receipt("clean_shared_graph")
            gradient_rows = {s: [] for s in SITES}
            raw_vjp_dtypes = {s: [] for s in SITES}
            for i, coord in enumerate(COORDS):
                receipt(f"before_vjp_{coord}")
                ge, gp = torch.autograd.grad(clean["terminal"].float()[coord],
                                              (clean["encoder31"], clean["projector"]),
                                              retain_graph=i < 3, allow_unused=False)
                for site, grad in (("encoder31", ge[0]), ("projector", gp)):
                    raw_vjp_dtypes[site].append(str(grad.dtype))
                    if grad.dtype != dtype or not torch.isfinite(grad).all() or float(torch.linalg.vector_norm(grad.float())) <= 0:
                        raise ValueError(f"Disconnected/nonfinite/wrong-dtype {site} VJP")
                    gradient_rows[site].append(grad.detach().float().cpu()[prepared["rows"][site]].contiguous())
                receipt(f"after_vjp_{coord}")
                del ge, gp
            del clean
            gradient_rows = {s: torch.stack(gradient_rows[s]) for s in SITES}
            observations = {s: [] for s in SITES}
            patch_count = 0
            for site in SITES:
                rows = prepared["rows"][site]
                direction = prepared["directions"][site]
                for cycle, order in enumerate(CYCLES):
                    for alpha in order:
                        pair = {}
                        for label, a in (("zero", 0.0), ("alpha", alpha)):
                            receipt(f"before_{site}_{cycle}_{label}_{alpha}")
                            captured = capture(model, batch, dtype, grad=True, patch_site=site,
                                               rows=rows, direction=direction, alpha=a)
                            source_rows = captured["prepatch_rows"][site]
                            edit = captured["edit"]
                            if a == 0 and float(edit["effective"].abs().sum()) != 0:
                                raise ValueError("Zero realized source edit not exact")
                            pair[label] = {"terminal": captured["terminal_fp32_cpu"],
                                           "terminal_native_four": captured["terminal_native_cpu"][list(COORDS)],
                                           "source_rows": source_rows, "source_sha": first.tensor_sha(source_rows),
                                           "terminal_sha": first.tensor_sha(captured["terminal_native_cpu"]),
                                           "edit": edit, "hook_events": captured["events"]}
                            del captured
                            patch_count += 1
                            receipt(f"after_{site}_{cycle}_{label}_{alpha}", patch_count=patch_count)
                        pred = contract(gradient_rows[site], pair["alpha"]["edit"])
                        edit = pair["alpha"]["edit"]
                        observations[site].append({"site": site, "cycle": cycle, "alpha": alpha,
                                                   "zero_terminal": pair["zero"]["terminal"],
                                                   "alpha_terminal": pair["alpha"]["terminal"],
                                                   "zero_source_rows": pair["zero"]["source_rows"],
                                                   "alpha_source_rows": pair["alpha"]["source_rows"],
                                                   "effective_edit": edit["effective"],
                                                   "zero_sha": pair["zero"]["terminal_sha"],
                                                   "alpha_sha": pair["alpha"]["terminal_sha"],
                                                   "zero_source_sha": pair["zero"]["source_sha"],
                                                   "alpha_source_sha": pair["alpha"]["source_sha"],
                                                   "zero_ulp_four": ulp_spacing(pair["zero"]["terminal_native_four"]),
                                                   "zero_hook_events": pair["zero"]["hook_events"],
                                                   "alpha_hook_events": pair["alpha"]["hook_events"],
                                                   "prediction": pred,
                                                   "edit_metrics": {"effective_norm": edit["effective_norm"],
                                                                    "mean_fidelity_ratio": edit["mean_fidelity_ratio"],
                                                                    "row_nonuniformity_ratio": edit["row_nonuniformity_ratio"],
                                                                    "unchanged_fraction": edit["unchanged_fraction"]}})
            if patch_count != 72 or any(len(observations[s]) != 18 for s in SITES):
                raise ValueError("Fixed paired workload incomplete")
            analysis = {s: analyze_site(observations[s], gradient_rows[s]) for s in SITES}
            details = {"directions": prepared["directions"], "selected_gradients_fp32": gradient_rows,
                       "clean_terminal_fp32": clean_terminal,
                       "clean_selected_source_rows": clean_selected_source_rows,
                       "donor_terminal_fp32": donor_terminal,
                       "no_grad_terminals_fp32": torch.stack(no_grad_terminals),
                       "observations": observations}
            with (output / "details.pt").open("xb") as file:
                torch.save(details, file)
                file.flush()
                os.fsync(file.fileno())
            receipt("details_saved")
            result = {"status": "completed_old_fit_numerical_diagnostic_only", "arm": arm,
                      "meta_sha256": base.sha(output / "meta.json"),
                      "progress_sha256": base.sha(progress),
                      "details_sha256": base.sha(output / "details.pt"),
                      "amendment_sha256": AMENDMENT_SHA,
                      "model_layout_sha256": base.sha(output / "model_layout.json"),
                      "model_layout_counts": layout["dtype_device_counts"],
                      "natural_suffix_ids": natural["suffix_ids"], "passive_suffix_ids": passive["suffix_ids"],
                      "passive_hook_shapes": passive["passive_hook_shapes"],
                      "clean_hook_events": clean_events, "clean_terminal_sha256": first.tensor_sha(clean_terminal),
                      "clean_selected_source_row_hashes": {
                          s: first.tensor_sha(clean_selected_source_rows[s]) for s in SITES},
                      "clean_source_hashes": clean_source_hashes, "donor_source_hashes": donor_sources,
                      "raw_vjp_dtypes": raw_vjp_dtypes, "selected_gradient_storage_dtype": "torch.float32",
                      "gradient_hashes": {s: first.tensor_sha(gradient_rows[s]) for s in SITES},
                      "patch_forward_count": patch_count, "zero_nonzero_pair_count": 36,
                      "paired_observations_per_site": 18,
                      "source_element_unchanged_label": ("unchanged_by_bf16_cast" if arm == "bf16"
                                                         else "unchanged_by_fp32_addition_rounding"),
                      "analysis": analysis,
                      "peak_mps_driver_bytes": max(memory.driver_max, int(torch.mps.driver_allocated_memory())),
                      "peak_mps_current_bytes": memory.current_max,
                      "elapsed_seconds": time.monotonic() - started,
                      "scope": "one old v1 FIT case; no v2 transfer or efficacy"}
            if result["elapsed_seconds"] >= max_seconds or result["peak_mps_driver_bytes"] >= MAX_DRIVER:
                raise RuntimeError("Arm cap reached before final result")
            json.dumps(result, allow_nan=False)
            base.exclusive_json(output / "result.json", result)
    except Exception as exc:
        resource_limited = isinstance(exc, (MemoryError, TimeoutError)) or "out of memory" in str(exc).lower()
        base.exclusive_json(output / "error.json", {"status": ("resource_limited_arm" if resource_limited
                                                                 else "invalid_old_fit_arm"), "arm": arm,
                                                      "error_type": type(exc).__name__, "error": str(exc),
                                                      "elapsed_seconds": time.monotonic() - started,
                                                      "progress_sha256": base.sha(progress)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validate_only and args.output_dir is not None:
        parser.error("--output-dir requires --run")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.validate_only:
        prepared = preflight()
        print(json.dumps({"status": "validated_cpu_only", "arm": args.arm,
                          "record_key": "animal_08|piper|fit_chosen",
                          "rows": prepared["rows"],
                          "direction_hashes": {s: first.tensor_sha(prepared["directions"][s]) for s in SITES},
                          "snapshot_revision": prepared["snapshot"]["revision"]}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run(args.arm, args.output_dir or RUNS / args.arm / stamp)


if __name__ == "__main__":
    main()
