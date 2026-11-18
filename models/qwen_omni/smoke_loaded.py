"""Bounded loaded-model gate: one pinned audio forward and one backward probe.

Run from this directory with HF_HOME pointing to its local .hf_cache. The
checkpoint is downloaded only with --allow-download. This does not fit a lens.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

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


def tensor_shape(tensor):
    return list(tensor.shape) if hasattr(tensor, "shape") else None


def mps_memory():
    if not torch.backends.mps.is_available():
        return None
    return {
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_bytes": torch.mps.driver_allocated_memory(),
    }


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_source_revision": TRANSFORMERS_REVISION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": args.device,
        "dtype": "bfloat16" if args.device == "mps" else "float32",
        "purpose": "loaded model interface and single backward reachability, not a lens fit",
    }
    started = time.perf_counter()
    dtype = torch.bfloat16 if args.device == "mps" else torch.float32
    model, processor = load_pinned_thinker(
        allow_download=args.allow_download,
        dtype=dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    report["load_seconds"] = time.perf_counter() - started
    report["memory_after_load"] = mps_memory()
    report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    report["parameter_devices"] = dict(
        Counter(str(parameter.device) for parameter in model.parameters())
    )
    batch, report["audio_sha256"] = prepare_audio_batch(
        processor,
        args.audio,
        system_prompt="Answer the spoken request briefly.",
    )
    report["batch_shapes"] = {key: tensor_shape(value) for key, value in batch.items()}
    batch = batch.to(model.device, dtype=model.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    points = resolve_hook_points(
        model, audio_layers=(0, 31), text_layers=(0, 17, 35)
    )

    with torch.no_grad():
        synchronize(args.device)
        started = time.perf_counter()
        baseline = model(**batch, use_cache=False, return_dict=True).logits
        synchronize(args.device)
        report["forward_seconds"] = time.perf_counter() - started
        with ActivationCapture(points, detach=True) as capture:
            with_hooks = model(**batch, use_cache=False, return_dict=True).logits
    report["logits_shape"] = tensor_shape(baseline)
    report["max_abs_logit_delta"] = float((baseline.float() - with_hooks.float()).abs().max())
    report["hook_shapes"] = {
        key: [tensor_shape(value) for value in values]
        for key, values in capture.activations.items()
    }
    report["memory_after_forward"] = mps_memory()

    # Free both full-logit tensors before retaining an autograd graph.
    del baseline, with_hooks, capture
    batch["input_features"] = batch["input_features"].detach().requires_grad_(True)
    with ActivationCapture(points, detach=False) as capture:
        synchronize(args.device)
        started = time.perf_counter()
        logits = model(**batch, use_cache=False, return_dict=True).logits
        synchronize(args.device)
        report["gradient_forward_seconds"] = time.perf_counter() - started
        token_id = int(logits[0, -1].argmax())
        score = logits[0, -1, token_id].float()
        activation_names = (
            "audio.encoder.0",
            "audio.to_language",
            "text.residual.17",
        )
        activations = [capture.activations[name][0] for name in activation_names]
        synchronize(args.device)
        started = time.perf_counter()
        gradients = torch.autograd.grad(
            score, activations, retain_graph=False, allow_unused=True
        )
        synchronize(args.device)
        report["backward_probe_seconds"] = time.perf_counter() - started
    report["selected_token_id"] = token_id
    report["gradient_norms"] = {
        name: (None if gradient is None else float(gradient.float().norm()))
        for name, gradient in zip(activation_names, gradients)
    }
    report["memory_after_backward"] = mps_memory()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
