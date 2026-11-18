"""Apply one audio-fitted terminal J-lens to fit-only paired bridge inputs.

This is an exploratory plumbing/conditioning check, not held-out evaluation:
the first Piper clip was used to fit the supplied map. The caller must not
pass a dev or test render manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
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
from scoring import score_item, semantic_token_ids  # noqa: E402
from task_framing_gate import SYSTEM_PROMPT, typed_batch  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> dict[str, dict]:
    return {item["id"]: item for item in (
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_manifest", type=Path)
    parser.add_argument("lens_checkpoint", type=Path)
    parser.add_argument("--anchor-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    render = json.loads(args.render_manifest.read_text())
    if render["selection"]["split"] != "fit":
        raise ValueError("This pilot accepts fit items only")
    frozen = read_jsonl(HERE / "scorable_manifest.jsonl")
    state = torch.load(args.lens_checkpoint, map_location="cpu", weights_only=True)
    meta = state["meta"]
    if not bool(state["completed"].all()):
        raise ValueError("Lens checkpoint is incomplete")
    if meta["system_prompt"] != SYSTEM_PROMPT:
        raise ValueError("Lens prompt differs from paired task frame")
    layers = tuple(meta["source_layers"])
    if layers != (0, 17) or meta["target_layer"] != 35:
        raise ValueError("Lens layers differ from frozen pilot")
    maps = {layer: state["rows"][str(layer)] for layer in layers}
    if not all(torch.isfinite(matrix).all() for matrix in maps.values()):
        raise ValueError("Lens matrix contains a nonfinite row")
    model, processor = load_pinned_thinker(
        allow_download=False,
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    if meta["model_revision"] != "f75b40e3da2003cdd6e1829b1f420ca70797c34e":
        raise ValueError("Lens checkpoint/model revision mismatch")
    vocab = int(model.lm_head.out_features)
    allowed_ids = semantic_token_ids(processor.tokenizer, vocab)
    semantic_hash = hashlib.sha256(
        struct.pack(f"<{len(allowed_ids)}I", *allowed_ids)
    ).hexdigest()
    expected_semantic_hash = json.loads(
        (HERE / "scoring_metadata.json").read_text()
    )["semantic_ids_uint32le_sha256"]
    if semantic_hash != expected_semantic_hash:
        raise ValueError("Semantic vocabulary differs from frozen scoring metadata")
    points = resolve_hook_points(model, audio_layers=(), text_layers=(*layers, 35))
    norm = model.model.language_model.norm
    matrices = {layer: matrix.to(model.device, dtype=torch.float32) for layer, matrix in maps.items()}
    anchor_batch, anchor_hash = prepare_audio_batch(
        processor, args.anchor_audio, system_prompt=SYSTEM_PROMPT
    )
    if anchor_hash != meta["audio_sha256"]:
        raise ValueError("Anchor audio differs from lens fit input")
    with torch.inference_mode(), ActivationCapture(points, detach=True) as anchor_capture:
        model.model(**anchor_batch.to(model.device, dtype=model.dtype),
                    use_cache=False, return_dict=True)
    anchor = {
        layer: anchor_capture.activations[f"text.residual.{layer}"][0][0, -1]
        .to(model.device, dtype=torch.float32)
        for layer in layers
    }
    anchor_target = anchor_capture.activations["text.residual.35"][0][0, -1].to(
        model.device, dtype=torch.float32
    )

    def one(batch, item, modality, engine=None):
        batch = batch.to(model.device, dtype=model.dtype)
        with torch.inference_mode(), ActivationCapture(points, detach=True) as capture:
            final_logits = model(**batch, use_cache=False, return_dict=True).logits[0, -1]
        final_state = capture.activations["text.residual.35"][0][0, -1].to(
            model.device, dtype=model.dtype
        )
        with torch.inference_mode():
            reconstructed = model.lm_head(norm(final_state))
            reconstruction_delta = float((reconstructed.float() - final_logits.float()).abs().max())
            raw, jacobian, affine, constant = {}, {}, {}, {}
            for layer in layers:
                source = capture.activations[f"text.residual.{layer}"][0][0, -1].to(
                    model.device, dtype=model.dtype
                )
                raw[layer] = model.lm_head(norm(source)).float().cpu()
                source32 = source.float()
                transported = matrices[layer] @ source32
                predicted = anchor_target + matrices[layer] @ (source32 - anchor[layer])
                jacobian[layer] = model.lm_head(norm(transported.to(model.dtype))).float().cpu()
                affine[layer] = model.lm_head(norm(predicted.to(model.dtype))).float().cpu()
                constant[layer] = model.lm_head(norm(anchor_target.to(model.dtype))).float().cpu()
        # The capture is copied to CPU and back in BF16. Quantized reductions
        # can differ by one or two output ulps after that round trip.
        if reconstruction_delta > 0.125:
            raise ValueError(f"Final state does not reconstruct full logits: {reconstruction_delta}")
        return {
            "modality": modality,
            "engine": engine,
            "prompt_tokens": int(batch["input_ids"].shape[1]),
            "final_logit_reconstruction_max_abs": reconstruction_delta,
            "scores": {
                "raw_logit": score_item(item, raw, modality=modality,
                                        method="raw_logit", head_vocab_size=vocab,
                                        audio_source=engine, allowed_ids=allowed_ids),
                "audio_fitted_jacobian": score_item(item, jacobian, modality=modality,
                                                    method="audio_fitted_jacobian",
                                                    head_vocab_size=vocab,
                                                    audio_source=engine, allowed_ids=allowed_ids),
                "local_affine_jacobian": score_item(item, affine, modality=modality,
                                                   method="local_affine_jacobian",
                                                   head_vocab_size=vocab,
                                                   audio_source=engine, allowed_ids=allowed_ids),
                "anchor_constant": score_item(item, constant, modality=modality,
                                              method="anchor_constant",
                                              head_vocab_size=vocab,
                                              audio_source=engine, allowed_ids=allowed_ids),
            },
        }

    report = {
        "purpose": "fit-only exploratory readout plumbing check; no held-out claim",
        "render_manifest_sha256": sha(args.render_manifest),
        "lens_checkpoint_sha256": sha(args.lens_checkpoint),
        "anchor_audio_sha256": anchor_hash,
        "semantic_ids_uint32le_sha256": semantic_hash,
        "scorable_manifest_sha256": sha(HERE / "scorable_manifest.jsonl"),
        "system_prompt": SYSTEM_PROMPT,
        "source_layers": list(layers),
        "target_layer": meta["target_layer"],
        "records": [],
    }
    for record in render["records"][:args.limit]:
        source = record["source_item"]
        item = frozen[source["id"]]
        if source["split"] != "fit" or item["split"] != "fit":
            raise ValueError("Non-fit item in pilot")
        if item["scoring_status"] != "scorable":
            continue
        result = {"item_id": source["id"], "modalities": []}
        result["modalities"].append(one(
            typed_batch(processor, source["typed_input"]["prompt"]), item, "typed"
        ))
        for engine in ("piper", "kokoro"):
            info = record["outputs"][engine]["processed"]
            path = (args.render_manifest.parent / info["path"]).resolve()
            if sha(path) != info["file_sha256"]:
                raise ValueError(f"Audio hash mismatch: {path}")
            batch, digest = prepare_audio_batch(processor, path, system_prompt=SYSTEM_PROMPT)
            if digest != info["file_sha256"]:
                raise ValueError(f"Processor audio hash mismatch: {path}")
            result["modalities"].append(one(batch, item, "audio", engine))
        report["records"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"scored fit-only {source['id']}", flush=True)


if __name__ == "__main__":
    main()
