"""Bounded audio-conditioned Jacobian cost pilot on the pinned Omni-3B Thinker.

Only 16 of 2,048 target residual coordinates are computed. Each eight-coordinate
batch receives a separate replicated audio prompt, matching the reference
J-lens estimator's batched coordinate-VJP approach. This is a cost pilot, not
a fitted lens or an evaluation of interpretability quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers

from adapter import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_REVISION,
    ActivationCapture,
    load_pinned_thinker,
    prepare_audio_batch,
    resolve_hook_points,
)


SOURCE_LAYERS = (0, 17)
TARGET_LAYER = 35
WIDTH = 2048
COORDINATES = tuple(range(0, WIDTH, 128))  # 16 deterministic, spread-out rows


def synchronize() -> None:
    torch.mps.synchronize()


def memory_bytes() -> dict[str, int]:
    return {
        "allocated": torch.mps.current_allocated_memory(),
        "driver": torch.mps.driver_allocated_memory(),
        "recommended_max": torch.mps.recommended_max_memory(),
    }


def check_memory(limit_bytes: int, stage: str) -> dict[str, int]:
    memory = memory_bytes()
    if memory["driver"] > limit_bytes:
        raise MemoryError(
            f"MPS driver allocation {memory['driver']} exceeds {limit_bytes} "
            f"after {stage}"
        )
    return memory


def metadata(audio_sha256: str, batch_size: int) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "audio_sha256": audio_sha256,
        "system_prompt": "Answer the spoken request briefly.",
        "source_layers": list(SOURCE_LAYERS),
        "target_layer": TARGET_LAYER,
        "coordinates": list(COORDINATES),
        "dim_batch": batch_size,
        "source_positions": "input_ids == processor.audio_token_id",
        "target_positions": "all non-padding prompt positions",
        "target_quantity": "post-final-block residual, not logits",
        "target_reduction": "sum over current-and-future positions (causal model)",
        "source_reduction": "mean over audio placeholder positions",
    }


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_or_initialize(path: Path, expected_meta: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["meta"] != expected_meta:
            raise ValueError("Checkpoint metadata differs from this pinned benchmark")
        if set(saved["rows"]) != {f"text.residual.{i}" for i in SOURCE_LAYERS}:
            raise ValueError("Checkpoint layer keys are incomplete")
        return saved
    return {
        "meta": expected_meta,
        "completed": torch.zeros(len(COORDINATES), dtype=torch.bool),
        "rows": {
            f"text.residual.{i}": torch.full(
                (len(COORDINATES), WIDTH), float("nan"), dtype=torch.float32
            )
            for i in SOURCE_LAYERS
        },
        "batch_records": [],
    }


def repeated_batch(batch: Any, count: int) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
            raise ValueError(f"Unexpected processor field {key!r}")
        result[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
    result["input_features"] = result["input_features"].detach().requires_grad_(True)
    return result


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("jacobian_16coords.pt"))
    parser.add_argument("--report", type=Path, default=Path("jacobian_16coords.json"))
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-driver-gib", type=float, default=30.0)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("This bounded pilot requires local MPS")
    if not 1 <= args.dim_batch <= 8 or len(COORDINATES) % args.dim_batch:
        raise ValueError("dim-batch must divide 16 and be between 1 and 8")
    limit_bytes = int(args.max_driver_gib * 1024**3)
    if limit_bytes > int(torch.mps.recommended_max_memory() * 0.9):
        raise ValueError("max-driver-gib exceeds 90% of MPS recommended maximum")

    started = time.perf_counter()
    model, processor = load_pinned_thinker(
        allow_download=False,
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    load_seconds = time.perf_counter() - started
    after_load = check_memory(limit_bytes, "load")
    batch, audio_sha256 = prepare_audio_batch(
        processor, args.audio, system_prompt="Answer the spoken request briefly."
    )
    batch = batch.to("mps", dtype=torch.bfloat16)
    audio_mask = (batch["input_ids"][0] == processor.audio_token_id)
    target_mask = batch["attention_mask"][0].bool()
    if not audio_mask.any():
        raise ValueError("Pinned processor produced no audio placeholder positions")
    meta = metadata(audio_sha256, args.dim_batch)
    saved = load_or_initialize(args.checkpoint, meta)
    report = {
        **meta,
        "status": "running",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "mps",
        "dtype": "bfloat16",
        "model_load_seconds": load_seconds,
        "memory_after_load_bytes": after_load,
        "driver_limit_bytes": limit_bytes,
        "prompt_positions": int(batch["input_ids"].shape[1]),
        "audio_positions": int(audio_mask.sum()),
        "completed_coordinates": int(saved["completed"].sum()),
        "batch_records": saved["batch_records"],
    }
    write_report(args.report, report)
    points = resolve_hook_points(
        model, audio_layers=(), text_layers=(*SOURCE_LAYERS, TARGET_LAYER)
    )

    try:
        for offset in range(0, len(COORDINATES), args.dim_batch):
            positions = list(range(offset, offset + args.dim_batch))
            if all(bool(saved["completed"][index]) for index in positions):
                continue
            if any(bool(saved["completed"][index]) for index in positions):
                raise ValueError("Partially completed coordinate batch in checkpoint")
            coordinates = [COORDINATES[index] for index in positions]
            repeated = repeated_batch(batch, args.dim_batch)
            synchronize()
            with ActivationCapture(points, detach=False) as capture:
                started = time.perf_counter()
                # We hook the final residual directly, so computing vocabulary
                # logits is unnecessary for this Jacobian-cost measurement.
                model.model(**repeated, use_cache=False, return_dict=True)
                synchronize()
                forward_seconds = time.perf_counter() - started
                sources = [
                    capture.activations[f"text.residual.{i}"][0]
                    for i in SOURCE_LAYERS
                ]
                target = capture.activations[f"text.residual.{TARGET_LAYER}"][0]
                if tuple(target.shape) != (args.dim_batch, batch["input_ids"].shape[1], WIDTH):
                    raise ValueError(f"Unexpected final residual shape {tuple(target.shape)}")
                check_memory(limit_bytes, "forward")
                cotangent = torch.zeros_like(target)
                for batch_index, coordinate in enumerate(coordinates):
                    cotangent[batch_index, target_mask, coordinate] = 1
                synchronize()
                started = time.perf_counter()
                gradients = torch.autograd.grad(
                    target, sources, grad_outputs=cotangent, allow_unused=False
                )
                synchronize()
                backward_seconds = time.perf_counter() - started
                memory = check_memory(limit_bytes, "backward")
            row_norms = {}
            for source_index, gradient in zip(SOURCE_LAYERS, gradients):
                rows = gradient[:, audio_mask, :].float().mean(dim=1).cpu()
                if not torch.isfinite(rows).all():
                    raise FloatingPointError(f"Nonfinite Jacobian rows for layer {source_index}")
                key = f"text.residual.{source_index}"
                saved["rows"][key][positions] = rows
                row_norms[key] = [float(value) for value in rows.norm(dim=1)]
            saved["completed"][positions] = True
            record = {
                "coordinates": coordinates,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "driver_memory_bytes": memory,
                "row_norms": row_norms,
            }
            saved["batch_records"].append(record)
            atomic_save_checkpoint(args.checkpoint, saved)
            report["completed_coordinates"] = int(saved["completed"].sum())
            report["batch_records"] = saved["batch_records"]
            write_report(args.report, report)
            print(f"saved {report['completed_coordinates']}/16 coordinates", flush=True)
            del repeated, sources, target, cotangent, gradients
            gc.collect()
            torch.mps.empty_cache()
        report["status"] = "complete"
        sample_seconds = sum(
            item["forward_seconds"] + item["backward_seconds"]
            for item in saved["batch_records"]
        )
        report["measured_16_coordinate_seconds"] = sample_seconds
        report["naive_full_2048_coordinate_seconds"] = sample_seconds * (WIDTH / len(COORDINATES))
        report["estimate_note"] = (
            "Upper-bound-ish arithmetic scale check from two selected-layer batches on one audio prompt: "
            "this pilot repeats forwards per batch, unlike an upstream-style graph-reuse fit. "
            "Not a measured full-rank fit, a runtime prediction, or a semantic readout."
        )
        write_report(args.report, report)
        print(json.dumps({
            "status": report["status"],
            "measured_16_coordinate_seconds": sample_seconds,
            "naive_full_2048_coordinate_seconds": report["naive_full_2048_coordinate_seconds"],
        }, indent=2))
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["memory_at_failure_bytes"] = memory_bytes()
        write_report(args.report, report)
        raise


if __name__ == "__main__":
    main()
