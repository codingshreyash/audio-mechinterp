"""Fit-only complete-answer influence curve for all 36 Thinker layers."""

from __future__ import annotations

import json

import torch

from patch_encoder_v1 import (  # noqa: E402
    HERE, ROOT, complete_answer, sha, load_pinned_thinker,
    prepare_audio_batch, resolve_hook_points, ActivationCapture,
    build_row_index,
)
from patch_connector_v1 import projected_windows  # noqa: E402
from patch_thinker_residual_v1 import answer_with_residual_patch  # noqa: E402


def main() -> None:
    plan_path = HERE / "thinker_layer_sweep_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    manifest_path = (HERE / plan["input_manifest"]).resolve()
    if sha(manifest_path) != plan["input_manifest_sha256"]:
        raise ValueError("Controlled-pair manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["system_prompt"] != plan["system_prompt"]:
        raise ValueError("Task-frame mismatch")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    points = resolve_hook_points(model, audio_layers=(),
                                 text_layers=tuple(plan["layers"]))
    text_points = {int(p.name.split(".")[-1]): p for p in points[1:]}
    audio_config = model.config.audio_config
    extractor = processor.feature_extractor
    report = {
        "purpose": "fit-only 36-layer audio-placeholder donor-patch influence curve",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "results": [],
    }
    output_path = HERE / "thinker_layer_sweep_results_v1.json"
    for engine in plan["engines"]:
        group = manifest["engines"][engine]
        inputs = {}
        for word in ("red", "blue"):
            info = group["outputs"][word]
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError(f"WAV hash mismatch: {path}")
            batch, digest = prepare_audio_batch(
                processor, path, system_prompt=plan["system_prompt"]
            )
            if digest != info["file_sha256"]:
                raise ValueError("Processor audio hash mismatch")
            mel_frames = int(batch["feature_attention_mask"].sum())
            placeholders = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
            row_map = build_row_index(
                mel_frames=mel_frames, audio_token_positions=placeholders,
                n_window=audio_config.n_window, hop_length=extractor.hop_length,
                sample_rate=extractor.sampling_rate,
            )
            inputs[word] = {"batch": batch.to(model.device, dtype=model.dtype),
                            "row_map": row_map}
        if inputs["red"]["row_map"] != inputs["blue"]["row_map"]:
            raise ValueError("Matched pair row maps differ")
        if not torch.equal(inputs["red"]["batch"]["input_ids"],
                           inputs["blue"]["batch"]["input_ids"]):
            raise ValueError("Matched pair input IDs differ")
        row_map = inputs["red"]["row_map"]
        windows = projected_windows(row_map, group["word_slot_start_seconds"],
                                    group["word_slot_end_seconds"])
        projected = row_map["projected_rows"]
        positions = [projected[i]["audio_placeholder_token_position"]
                     for i in windows["word_local"]]
        clean = {word: complete_answer(model, processor, inputs[word]["batch"])
                 for word in ("red", "blue")}
        if not all(clean[word]["answer"].lower() == word
                   and clean[word]["stopped_on_eos"] for word in clean):
            raise ValueError("Clean answer gate failed")
        captures = {}
        for word in ("red", "blue"):
            with torch.no_grad(), ActivationCapture(points[1:], detach=True) as capture:
                model(**inputs[word]["batch"], use_cache=False, return_dict=True)
            captures[word] = {
                layer: capture.activations[text_points[layer].name][0]
                for layer in plan["layers"]
            }
        for layer in plan["layers"]:
            point = text_points[layer]
            for recipient, donor_word in (("red", "blue"), ("blue", "red")):
                source = captures[recipient][layer]
                donor = captures[donor_word][layer]
                if source.shape != donor.shape:
                    raise ValueError("Matched residual shapes differ")
                result = answer_with_residual_patch(
                    model, processor, inputs[recipient]["batch"],
                    point.module, donor, positions,
                )
                case = {
                    "engine": engine, "layer": layer,
                    "recipient": recipient, "donor": donor_word,
                    "word_audio_positions": positions,
                    "donor_minus_recipient_norm": float(
                        (donor[:, positions, :].float() - source[:, positions, :].float()).norm()
                    ),
                    "clean": clean[recipient], "patched": result,
                }
                report["results"].append(case)
                output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(engine, "layer", layer,
                  [(r["recipient"], r["patched"]["answer"])
                   for r in report["results"][-2:]], flush=True)


if __name__ == "__main__":
    main()
