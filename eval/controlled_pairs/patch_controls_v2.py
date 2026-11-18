"""Post-v1, fixed equal-magnitude and wrong-location patch controls."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from patch_encoder_v1 import (  # noqa: E402
    HERE, ROOT, complete_answer, row_windows, sha,
    load_pinned_thinker, prepare_audio_batch, resolve_hook_points,
    ActivationCapture, build_row_index,
)


def main() -> None:
    plan_path = HERE / "patch_plan_v2_controls.json"
    plan = json.loads(plan_path.read_text())
    if sha(HERE / "patch_plan_v1.json") != plan["v1_patch_plan_sha256"]:
        raise ValueError("v1 patch plan changed")
    if sha(HERE / "patch_results_v1.json") != plan["v1_patch_results_sha256"]:
        raise ValueError("v1 patch results changed")
    manifest_path = (HERE / plan["input_manifest"]).resolve()
    if sha(manifest_path) != plan["input_manifest_sha256"]:
        raise ValueError("Controlled-pair input manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["system_prompt"] != plan["system_prompt"]:
        raise ValueError("Task-frame mismatch")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    audio_config = model.config.audio_config
    extractor = processor.feature_extractor
    report = {
        "purpose": "fit-only post-v1 equal-magnitude causal patch controls",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "v1_results_sha256": plan["v1_patch_results_sha256"],
        "results": [],
    }
    output_path = HERE / "patch_controls_v2.json"
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
        windows = row_windows(inputs["red"]["row_map"],
                              group["word_slot_start_seconds"],
                              group["word_slot_end_seconds"])
        word_rows, wrong_rows = windows["word_local"], windows["wrong_chunk"]
        if len(word_rows) != len(wrong_rows):
            raise ValueError("Wrong-chunk rows must match word-row count")
        for layer in plan["layers"]:
            point = resolve_hook_points(model, audio_layers=(layer,), text_layers=())[0]
            activations = {}
            for word in ("red", "blue"):
                with torch.no_grad(), ActivationCapture((point,), detach=True) as capture:
                    model(**inputs[word]["batch"], use_cache=False, return_dict=True)
                activations[word] = capture.activations[point.name][0]
            for recipient, donor_word in (("red", "blue"), ("blue", "red")):
                source = activations[recipient]
                donor = activations[donor_word]
                delta = donor[word_rows].float() - source[word_rows].float()
                target_norm = float(delta.norm())
                if target_norm <= 0:
                    raise ValueError("Donor word delta has zero norm")
                case = {
                    "engine": engine, "layer": layer,
                    "recipient": recipient, "donor": donor_word,
                    "word_rows": word_rows, "wrong_rows": wrong_rows,
                    "donor_word_delta_norm": target_norm,
                    "conditions": {},
                }

                def run_control(name: str, rows: list[int], added: torch.Tensor) -> None:
                    synthetic = source.clone()
                    synthetic[rows] = (source[rows].float() + added).to(source.dtype)
                    effective = synthetic[rows].float() - source[rows].float()
                    case["conditions"][name] = {
                        "rows": rows,
                        "requested_delta_norm": float(added.norm()),
                        "effective_bf16_delta_norm": float(effective.norm()),
                        "outcome": complete_answer(
                            model, processor, inputs[recipient]["batch"],
                            module=point.module, donor=synthetic, rows=rows,
                        ),
                    }

                for seed in plan["random_seeds"]:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    noise = torch.randn(delta.shape, generator=generator,
                                        dtype=torch.float32)
                    noise *= target_norm / noise.norm()
                    run_control(f"random_word_rows_seed_{seed}", word_rows, noise)
                run_control("donor_delta_wrong_chunk", wrong_rows, delta)
                generator = torch.Generator(device="cpu").manual_seed(plan["shuffle_seed"])
                order = torch.randperm(len(word_rows), generator=generator)
                run_control("shuffled_word_rows", word_rows, delta[order])
                report["results"].append(case)
                output_path.write_text(json.dumps(report, indent=2) + "\n")
                print(engine, layer, recipient, "->", donor_word,
                      {name: x["outcome"]["answer"]
                       for name, x in case["conditions"].items()}, flush=True)


if __name__ == "__main__":
    main()
