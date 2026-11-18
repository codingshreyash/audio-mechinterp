"""FIT-only FP32 feasibility probe for a shared audio-to-terminal VJP graph.

Uses the original Piper red clip only, 16 fixed target coordinates, and both
Qwen audio sites in one forward. This measures cost and connectivity, not a
corpus-fitted transport or model efficacy. No development/test path is opened.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch

from fit_fp32_diagnostic_v1 import mps_cap_fraction
from fit_gate_v1 import (
    FIT_MANIFEST_SHA, MAX_MPS_DRIVER_BYTES, MODEL_ID, MODEL_REVISION, ROOT,
    SITES, SYSTEM_PROMPT, TRANSFORMERS_REVISION, _output_tensor,
    _replace_output, build_inputs, load_pinned_thinker, record_mps_memory,
    resolve_hook_points, sha, verify_fit_files,
)


HERE = Path(__file__).resolve().parent
PROPOSAL = HERE / "NEXT_PHASE_CORPUS_LENS_PROPOSAL.md"
FIT_ENGINE = "piper"
FIT_WORD = "red"
N_TARGET_COTANGENTS = 16
SOURCE_SITES = ("encoder31", "connector")
TARGET_SITE = "text.residual.35"


def six_intervals(start: float, end: float) -> list[tuple[float, float]]:
    """Pre, four word quarters, post; all half-open except last endpoint."""
    if not 0 < start < end:
        raise ValueError("Invalid word slot")
    quarter = (end - start) / 4
    if start - quarter < 0:
        raise ValueError("No full preword bin")
    return [(start - quarter, start)] + [
        (start + j * quarter, start + (j + 1) * quarter) for j in range(4)
    ] + [(end, end + quarter)]


def bin_rows(row_map: dict, site: str, intervals: list[tuple[float, float]],
             n_rows: int) -> list[list[int]]:
    """Assign nominal row centers to disjoint intervals; exact ties go earlier."""
    if site == "connector":
        entries = row_map["projected_rows"]
        period = row_map["nominal_mel_frame_period_seconds"]
        centers = [e["nominal_center_mel_frame"] * period for e in entries]
    elif site == "encoder31":
        entries = row_map["encoder_rows"]
        centers = [e["nominal_center_seconds"] for e in entries]
    else:
        raise ValueError("Unknown site")
    if len(centers) != n_rows:
        raise ValueError("Row-map/source shape mismatch")
    result = [[] for _ in intervals]
    for row, center in enumerate(centers):
        for index, (lo, hi) in enumerate(intervals):
            if (lo < center <= hi) or (index == 0 and center == lo):
                result[index].append(row)
                break
    if any(not rows for rows in result):
        raise ValueError("At least one of six bins has no source rows")
    if len(set(row for rows in result for row in rows)) != sum(map(len, result)):
        raise ValueError("A source row was assigned twice")
    return result


def vjp_rows(target: torch.Tensor, sources: dict[str, torch.Tensor],
             coordinates: tuple[int, ...], synchronize=lambda: None):
    """Yield one terminal-coordinate VJP at both source sites per backward."""
    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError("Expected one prompt and 3D terminal residual")
    if not all(value.ndim == 2 and value.requires_grad for value in sources.values()):
        raise ValueError("Both source sites must be differentiable 2D tensors")
    if not coordinates or max(coordinates) >= target.shape[-1]:
        raise ValueError("Target coordinate outside residual width")
    for number, coord in enumerate(coordinates):
        gradient_output = torch.zeros_like(target)
        gradient_output[0, -1, coord] = 1
        synchronize()
        tick = time.perf_counter()
        grads = torch.autograd.grad(
            target, [sources[site] for site in SOURCE_SITES],
            grad_outputs=gradient_output,
            retain_graph=number < len(coordinates) - 1,
            allow_unused=False,
        )
        synchronize()
        seconds = time.perf_counter() - tick
        yield coord, dict(zip(SOURCE_SITES, grads, strict=True)), seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "fit_corpus_cost_probe_v1_results.json")
    args = parser.parse_args()
    output = args.output.resolve()
    memory_log = output.with_suffix(".memory.jsonl")
    if output.exists() or memory_log.exists():
        raise FileExistsError("Cost probe refuses to overwrite outputs")
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 cost probe requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision differs from pin")
    manifest = verify_fit_files([FIT_ENGINE])
    recommended = int(torch.mps.recommended_max_memory())
    fraction = mps_cap_fraction(recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    memory_log.write_text("")
    report = {
        "status": "running", "purpose": "fit-only shared-forward FP32 cost probe",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "proposal_sha256": sha(PROPOSAL),
        "script_sha256": sha(Path(__file__)),
        "helper_sha256": sha(HERE / "fit_gate_v1.py"),
        "adapter_sha256": sha(ROOT / "models" / "qwen_omni" / "adapter.py"),
        "time_map_sha256": sha(ROOT / "models" / "qwen_omni" / "time_map" / "index.py"),
        "engine": FIT_ENGINE, "word": FIT_WORD,
        "system_prompt": SYSTEM_PROMPT,
        "target_site": TARGET_SITE, "source_sites": SOURCE_SITES,
        "target_coordinates": list(range(N_TARGET_COTANGENTS)),
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "mps_hard_cap_fraction": fraction,
        "forward_seconds": None, "backward_seconds": [],
        "per_coordinate": [], "memory_log": str(memory_log),
    }
    try:
        tick = time.perf_counter()
        model, processor = load_pinned_thinker(
            allow_download=False, dtype=torch.float32,
            device_map="mps", low_cpu_mem_usage=True,
        )
        report["load_seconds"] = time.perf_counter() - tick
        if model.dtype != torch.float32:
            raise ValueError("Model is not FP32")
        record_mps_memory(memory_log, "after_model_load")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        inputs, group = build_inputs(model, processor, manifest, FIT_ENGINE)
        batch = inputs[FIT_WORD]["batch"]
        report["audio_sha256"] = inputs[FIT_WORD]["sha256"]
        report["prompt_tokens"] = int(batch["input_ids"].shape[1])
        intervals = six_intervals(
            group["word_slot_start_seconds"], group["word_slot_end_seconds"]
        )
        report["intervals_seconds"] = intervals
        points = {point.name: point.module for point in resolve_hook_points(
            model, audio_layers=(31,), text_layers=(35,)
        )}
        source_modules = {
            "encoder31": points[SITES["encoder31"][1]],
            "connector": points[SITES["connector"][1]],
        }
        target_module = points[TARGET_SITE]
        captures: dict[str, torch.Tensor] = {}
        calls: dict[str, int] = {name: 0 for name in (*SOURCE_SITES, TARGET_SITE)}

        def encoder_hook(_module, _inputs, output):
            calls["encoder31"] += 1
            state = _output_tensor(output).detach().requires_grad_(True)
            captures["encoder31"] = state
            return _replace_output(output, state)

        def connector_hook(_module, _inputs, output):
            calls["connector"] += 1
            state = _output_tensor(output)
            captures["connector"] = state
            return output

        def target_hook(_module, _inputs, output):
            calls[TARGET_SITE] += 1
            state = output[0] if isinstance(output, tuple) else output
            if not isinstance(state, torch.Tensor) or state.ndim != 3:
                raise ValueError("Target residual shape changed")
            captures[TARGET_SITE] = state
            return output

        handles = [
            source_modules["encoder31"].register_forward_hook(encoder_hook),
            source_modules["connector"].register_forward_hook(connector_hook),
            target_module.register_forward_hook(target_hook),
        ]
        try:
            example_tick = time.perf_counter()
            torch.mps.synchronize()
            tick = time.perf_counter()
            with torch.enable_grad():
                model(**batch, use_cache=False, return_dict=True)
            torch.mps.synchronize()
            report["forward_seconds"] = time.perf_counter() - tick
        finally:
            for handle in handles:
                handle.remove()
        if any(count != 1 for count in calls.values()):
            raise ValueError(f"Hook calls not exactly once: {calls}")
        report["hook_calls"] = calls
        record_mps_memory(memory_log, "after_shared_forward")
        sources = {site: captures[site] for site in SOURCE_SITES}
        target = captures[TARGET_SITE]
        report["source_shapes"] = {site: list(value.shape) for site, value in sources.items()}
        report["target_shape"] = list(target.shape)
        report["bin_rows"] = {
            site: bin_rows(inputs[FIT_WORD]["row_map"], site, intervals, sources[site].shape[0])
            for site in SOURCE_SITES
        }
        for coord, grads, backward_seconds in vjp_rows(
            target, sources, tuple(range(N_TARGET_COTANGENTS)), torch.mps.synchronize
        ):
            norms = {}
            for site, grad in grads.items():
                if grad.shape != sources[site].shape or not bool(torch.isfinite(grad).all()):
                    raise ValueError(f"Invalid {site} VJP")
                norms[site] = [float(grad[rows].sum(dim=0).norm())
                               for rows in report["bin_rows"][site]]
            report["backward_seconds"].append(backward_seconds)
            report["per_coordinate"].append({"coordinate": coord, "six_bin_gradient_norms": norms})
            record_mps_memory(memory_log, f"after_vjp_{coord}")
        report["status"] = "complete"
        report["n_backward_passes"] = N_TARGET_COTANGENTS
        report["one_example_model_work_seconds"] = time.perf_counter() - example_tick
        report["projected_48_example_model_work_seconds"] = (
            48 * report["one_example_model_work_seconds"]
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
