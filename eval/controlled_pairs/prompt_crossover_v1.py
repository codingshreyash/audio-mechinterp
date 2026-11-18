"""Posthoc system-prompt crossover on the original red/blue WAVs."""

from __future__ import annotations

import json

import torch

from patch_encoder_v1 import (
    HERE, ROOT, complete_answer, sha, load_pinned_thinker,
    prepare_audio_batch, resolve_hook_points, ActivationCapture, build_row_index,
)
from patch_connector_v1 import projected_windows
from patch_thinker_residual_v1 import answer_with_residual_patch


def main() -> None:
    plan_path = HERE / "prompt_crossover_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    manifest_path = (HERE / plan["old_input_manifest"]).resolve()
    if sha(manifest_path) != plan["old_input_manifest_sha256"]:
        raise ValueError("Old input manifest changed")
    if sha(ROOT / "eval" / "heldout_flow" / "plan_v1.json") != plan["heldout_plan_sha256"]:
        raise ValueError("Heldout plan changed")
    if sha(ROOT / "eval" / "heldout_flow" / "results_v1.json") != plan["heldout_results_sha256"]:
        raise ValueError("Heldout result changed")
    manifest = json.loads(manifest_path.read_text())
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    points = resolve_hook_points(model, audio_layers=(), text_layers=plan["layers"])
    by_layer = {int(p.name.split(".")[-1]): p for p in points[1:]}
    report = {
        "purpose": "posthoc fit-only prompt crossover; no held-out/security claim",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "heldout_results_sha256": plan["heldout_results_sha256"],
        "results": [],
    }
    output_path = HERE / "prompt_crossover_results_v1.json"
    for engine in plan["engines"]:
        group = manifest["engines"][engine]
        inputs = {}
        for word in ("red", "blue"):
            info = group["outputs"][word]
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError("WAV changed")
            batch, digest = prepare_audio_batch(
                processor, path, system_prompt=plan["system_prompt"]
            )
            if digest != info["file_sha256"]:
                raise ValueError("Processor WAV digest mismatch")
            mel_frames = int(batch["feature_attention_mask"].sum())
            placeholders = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
            row_map = build_row_index(
                mel_frames=mel_frames, audio_token_positions=placeholders,
                n_window=model.config.audio_config.n_window,
                hop_length=processor.feature_extractor.hop_length,
                sample_rate=processor.feature_extractor.sampling_rate,
            )
            windows = projected_windows(
                row_map, group["word_slot_start_seconds"], group["word_slot_end_seconds"]
            )
            word_positions = [row_map["projected_rows"][i]["audio_placeholder_token_position"]
                              for i in windows["word_local"]]
            inputs[word] = {"batch": batch.to(model.device, dtype=model.dtype),
                            "word_positions": word_positions}
        if not torch.equal(inputs["red"]["batch"]["input_ids"], inputs["blue"]["batch"]["input_ids"]):
            raise ValueError("Matched input IDs differ")
        if inputs["red"]["word_positions"] != inputs["blue"]["word_positions"]:
            raise ValueError("Matched word positions differ")
        clean = {w: complete_answer(model, processor, inputs[w]["batch"])
                 for w in ("red", "blue")}
        if not all(clean[w]["answer"].lower() == w and clean[w]["stopped_on_eos"] for w in clean):
            raise ValueError("Clean prompt-crossover gate failed")
        captures = {}
        for word in ("red", "blue"):
            with torch.no_grad(), ActivationCapture(points[1:], detach=True) as capture:
                model(**inputs[word]["batch"], use_cache=False, return_dict=True)
            captures[word] = {layer: capture.activations[by_layer[layer].name][0]
                              for layer in plan["layers"]}
        for layer in plan["layers"]:
            module = by_layer[layer].module
            for recipient, donor_word in (("red", "blue"), ("blue", "red")):
                batch = inputs[recipient]["batch"]
                terminal = int(batch["input_ids"].shape[1]) - 1
                positions = inputs[recipient]["word_positions"]
                donor, source = captures[donor_word][layer], captures[recipient][layer]
                if donor.shape != source.shape:
                    raise ValueError("Donor/recipient residual shape mismatch")
                outcomes = {
                    "word_donor": answer_with_residual_patch(
                        model, processor, batch, module, donor, positions,
                    ),
                    "terminal_donor": answer_with_residual_patch(
                        model, processor, batch, module, donor, [terminal],
                    ),
                    "self_word": answer_with_residual_patch(
                        model, processor, batch, module, source, positions,
                    ),
                }
                if outcomes["self_word"]["new_token_ids"] != clean[recipient]["new_token_ids"]:
                    raise ValueError("Self patch changed complete answer")
                report["results"].append({
                    "engine": engine, "layer": layer,
                    "recipient": recipient, "donor": donor_word,
                    "word_positions": positions, "terminal_position": terminal,
                    "clean": clean[recipient], "outcomes": outcomes,
                })
                output_path.write_text(json.dumps(report, indent=2) + "\n")
                print(engine, layer, recipient, "->", donor_word,
                      {k: v["answer"] for k, v in outcomes.items()}, flush=True)


if __name__ == "__main__":
    main()
