"""Resume-safe one-prompt terminal-residual Jacobian for pinned Omni Thinker.

This computes rows of d h_target[last] / d h_source[last] for each selected
source layer. It reuses one forward graph for all coordinate batches in each
process, and checkpoints completed rows after every backward pass. A resume
rebuilds the graph once and skips saved batches. No corpus averaging is done.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import transformers

from adapter import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_REVISION,
    load_pinned_thinker,
    prepare_audio_batch,
)


SOURCE_LAYERS = (0, 17)
TARGET_LAYER = 35
WIDTH = 2048


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_state(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    width = int(meta["width"])
    layers = tuple(int(index) for index in meta["source_layers"])
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("meta") != meta:
            raise ValueError("Checkpoint metadata differs from requested fit")
        completed = state.get("completed")
        if not isinstance(completed, torch.Tensor) or completed.shape != (width,):
            raise ValueError("Checkpoint completion mask has wrong shape")
        if completed.dtype != torch.bool:
            raise ValueError("Checkpoint completion mask must be bool")
        if set(state.get("rows", {})) != {str(index) for index in layers}:
            raise ValueError("Checkpoint source-layer keys differ")
        for index in layers:
            rows = state["rows"][str(index)]
            if rows.shape != (width, width) or rows.dtype != torch.float32:
                raise ValueError(f"Checkpoint rows for source {index} have wrong shape/type")
            if not torch.isfinite(rows[completed]).all():
                raise ValueError(f"Checkpoint completed rows for source {index} are nonfinite")
        return state
    return {
        "meta": meta,
        "completed": torch.zeros(width, dtype=torch.bool),
        "rows": {
            str(index): torch.full((width, width), float("nan"), dtype=torch.float32)
            for index in layers
        },
        "batch_records": [],
    }


def coordinate_batches(width: int, dim_batch: int) -> list[tuple[int, int]]:
    if width < 1 or dim_batch < 1:
        raise ValueError("width and dim_batch must be positive")
    return [(start, min(start + dim_batch, width)) for start in range(0, width, dim_batch)]


def fit_rows_from_graph(
    target: torch.Tensor,
    sources: dict[int, torch.Tensor],
    state: dict[str, Any],
    *,
    dim_batch: int,
    checkpoint: Path,
    synchronize: Callable[[], None] = lambda: None,
    check_memory: Callable[[str], dict[str, int]] = lambda stage: {},
    on_batch: Callable[[dict[str, Any]], None] = lambda record: None,
) -> None:
    """Complete the row matrix from a retained graph, with atomic batch saves.

    Target/source tensors must all have shape [dim_batch, seq, width]. Each
    batch example carries a different target coordinate at terminal position.
    """
    width = int(state["meta"]["width"])
    if target.ndim != 3 or target.shape[0] != dim_batch or target.shape[2] != width:
        raise ValueError("Target tensor shape does not match fit metadata")
    if target.shape[1] < 1:
        raise ValueError("No terminal prompt position")
    layer_ids = tuple(int(index) for index in state["meta"]["source_layers"])
    for layer in layer_ids:
        if sources[layer].shape != target.shape or not sources[layer].requires_grad:
            raise ValueError(f"Source {layer} is not a differentiable matching residual")
    batches = coordinate_batches(width, dim_batch)
    missing = [
        (start, stop) for start, stop in batches
        if not bool(state["completed"][start:stop].all())
    ]
    for start, stop in batches:
        segment = state["completed"][start:stop]
        if bool(segment.any()) and not bool(segment.all()):
            raise ValueError("Checkpoint has a partially complete coordinate batch")
    cotangent = torch.zeros_like(target)
    batch_indices = torch.arange(dim_batch, device=target.device)
    for batch_number, (start, stop) in enumerate(missing):
        count = stop - start
        cotangent.zero_()
        cotangent[batch_indices[:count], -1, start + batch_indices[:count]] = 1
        synchronize()
        tick = time.perf_counter()
        grads = torch.autograd.grad(
            target,
            [sources[layer] for layer in layer_ids],
            grad_outputs=cotangent,
            retain_graph=batch_number < len(missing) - 1,
            allow_unused=False,
        )
        synchronize()
        seconds = time.perf_counter() - tick
        memory = check_memory("backward")
        row_norms = {}
        for layer, grad in zip(layer_ids, grads, strict=True):
            rows = grad[:count, -1, :].float().cpu()
            if not torch.isfinite(rows).all():
                raise FloatingPointError(f"Nonfinite rows for source {layer}")
            state["rows"][str(layer)][start:stop] = rows
            row_norms[str(layer)] = [float(x) for x in rows.norm(dim=1)]
        state["completed"][start:stop] = True
        record = {
            "coordinates": list(range(start, stop)),
            "backward_seconds": seconds,
            "memory_bytes": memory,
            "row_norms": row_norms,
        }
        state["batch_records"].append(record)
        atomic_torch_save(checkpoint, state)
        on_batch(record)
        del grads


def mps_memory_bytes() -> dict[str, int]:
    return {
        "allocated": torch.mps.current_allocated_memory(),
        "driver": torch.mps.driver_allocated_memory(),
        "recommended_max": torch.mps.recommended_max_memory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("terminal_jacobian.pt"))
    parser.add_argument("--report", type=Path, default=Path("terminal_jacobian.json"))
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-driver-gib", type=float, default=30.0)
    parser.add_argument(
        "--system-prompt",
        default="Answer the spoken request briefly.",
        help="Pinned in checkpoint metadata; use the same instruction at evaluation.",
    )
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("This fit script currently requires MPS")
    if not 1 <= args.dim_batch <= 8:
        raise ValueError("dim-batch must be between 1 and 8")
    limit = int(args.max_driver_gib * 1024**3)
    if limit > int(torch.mps.recommended_max_memory() * 0.9):
        raise ValueError("max-driver-gib exceeds 90% of MPS recommended maximum")

    def guard(stage: str) -> dict[str, int]:
        result = mps_memory_bytes()
        if result["driver"] > limit:
            raise MemoryError(f"MPS driver allocation exceeds cap after {stage}: {result}")
        return result

    tick = time.perf_counter()
    model, processor = load_pinned_thinker(
        allow_download=False,
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = time.perf_counter() - tick
    guard("load")
    batch, audio_sha256 = prepare_audio_batch(
        processor, args.audio, system_prompt=args.system_prompt
    )
    batch = batch.to("mps", dtype=torch.bfloat16)
    if int(batch["attention_mask"][0].sum()) != batch["input_ids"].shape[1]:
        raise ValueError("Single-prompt benchmark expects no padding")
    meta = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "audio_sha256": audio_sha256,
        "system_prompt": args.system_prompt,
        "source_layers": list(SOURCE_LAYERS),
        "target_layer": TARGET_LAYER,
        "width": WIDTH,
        "dim_batch": args.dim_batch,
        "position_policy": "same terminal prompt position for source and target",
        "target_quantity": "post-final-block residual before final norm",
        "prompt_positions": int(batch["input_ids"].shape[1]),
    }
    state = initialize_state(args.checkpoint, meta)
    report = {
        **meta,
        "status": "running",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "mps",
        "dtype": "bfloat16",
        "model_load_seconds": load_seconds,
        "driver_limit_bytes": limit,
        "completed_coordinates": int(state["completed"].sum()),
        "batch_records": state["batch_records"],
    }
    atomic_json_save(args.report, report)
    if bool(state["completed"].all()):
        report["status"] = "complete"
        atomic_json_save(args.report, report)
        return

    repeated = {
        key: value.repeat((args.dim_batch,) + (1,) * (value.ndim - 1))
        for key, value in batch.items() if isinstance(value, torch.Tensor)
    }
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def make_capture(index: int):
        def capture(_module, _inputs, output):
            if isinstance(output, tuple):
                raise TypeError("Expected tensor decoder block output")
            if index == SOURCE_LAYERS[0]:
                # This is the same tensor value, but becomes the gradient root;
                # no audio-encoder graph need be retained across 256 backwards.
                output = output.detach().requires_grad_(True)
            captures[index] = output
            return output
        return capture

    try:
        for index in (*SOURCE_LAYERS, TARGET_LAYER):
            block = model.model.language_model.layers[index]
            handles.append(block.register_forward_hook(make_capture(index)))
        torch.mps.synchronize()
        tick = time.perf_counter()
        with torch.enable_grad():
            model.model(**repeated, use_cache=False, return_dict=True)
        torch.mps.synchronize()
        report["forward_seconds"] = time.perf_counter() - tick
        report["memory_after_forward_bytes"] = guard("forward")
    finally:
        for handle in handles:
            handle.remove()

    try:
        fit_rows_from_graph(
            captures[TARGET_LAYER],
            {index: captures[index] for index in SOURCE_LAYERS},
            state,
            dim_batch=args.dim_batch,
            checkpoint=args.checkpoint,
            synchronize=torch.mps.synchronize,
            check_memory=guard,
            on_batch=lambda record: (
                report.update({
                    "completed_coordinates": int(state["completed"].sum()),
                    "batch_records": state["batch_records"],
                }),
                atomic_json_save(args.report, report),
                print(f"saved {int(state['completed'].sum())}/{WIDTH} rows", flush=True),
            ),
        )
        report["status"] = "complete"
        atomic_json_save(args.report, report)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["memory_at_failure_bytes"] = mps_memory_bytes()
        atomic_json_save(args.report, report)
        raise
    finally:
        captures.clear()
        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
