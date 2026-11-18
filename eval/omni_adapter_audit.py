"""Independent loaded-model check of the Qwen Omni adapter's narrow smoke claim.

Uses the adapter without changing it. The zeroed-feature control is deliberately
out of distribution; it only checks that audio features affect the chosen logit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "qwen_omni"
sys.path.insert(0, str(MODEL_DIR))

from adapter import (  # noqa: E402
    ActivationCapture,
    load_pinned_thinker,
    prepare_audio_batch,
    resolve_hook_points,
)


def main() -> None:
    model, processor = load_pinned_thinker(
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    batch, digest = prepare_audio_batch(
        processor,
        MODEL_DIR / "sample_apple_5p5s.wav",
        system_prompt="Answer the spoken request briefly.",
    )
    batch = batch.to(model.device, dtype=model.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    points = resolve_hook_points(model, audio_layers=(0,), text_layers=(17,))

    with torch.no_grad():
        baseline = model(**batch, use_cache=False, return_dict=True).logits
        with ActivationCapture(points, detach=True) as capture:
            hooked = model(**batch, use_cache=False, return_dict=True).logits
        zero_batch = {key: value.clone() for key, value in batch.items()}
        zero_batch["input_features"].zero_()
        zero = model(**zero_batch, use_cache=False, return_dict=True).logits
        torch.mps.synchronize()
        token_id = int(baseline[0, -1].argmax())
        report = {
            "evidence": "independent one-prompt loaded-model interface audit; not a lens fit",
            "audio_sha256": digest,
            "audio_placeholder_count": int((batch["input_ids"] == processor.audio_token_id).sum()),
            "total_input_positions": int(batch["input_ids"].numel()),
            "valid_audio_feature_frames": int(batch["feature_attention_mask"].sum()),
            "hook_calls": {key: len(values) for key, values in capture.activations.items()},
            "baseline_hook_max_abs_logit_delta": float(
                (baseline.float() - hooked.float()).abs().max()
            ),
            "audio_zero_full_logit_max_abs_delta": float(
                (baseline.float() - zero.float()).abs().max()
            ),
            "selected_token_id": token_id,
            "selected_token_text": processor.tokenizer.decode([token_id]),
            "selected_token_audio_zero_abs_delta": float(
                (baseline[0, -1, token_id].float() - zero[0, -1, token_id].float()).abs()
            ),
            "all_forward_hooks_removed": all(not point.module._forward_hooks for point in points),
        }
    del baseline, hooked, zero, zero_batch, capture

    batch["input_features"] = batch["input_features"].detach().requires_grad_(True)
    with ActivationCapture(points, detach=False) as capture:
        logits = model(**batch, use_cache=False, return_dict=True).logits
        score = logits[0, -1, token_id].float()
        names = ("audio.encoder.0", "audio.to_language", "text.residual.17")
        tensors = [batch["input_features"]] + [capture.activations[name][0] for name in names]
        gradients = torch.autograd.grad(score, tensors, allow_unused=True)
        torch.mps.synchronize()
        report["gradient_norms"] = {
            name: None if gradient is None else float(gradient.float().norm())
            for name, gradient in zip(("input_features",) + names, gradients)
        }

    output = ROOT / "eval" / "omni_adapter_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
