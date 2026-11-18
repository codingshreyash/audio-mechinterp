"""Fit-only causal patch at Omni's audio-to-language projection boundary."""

from __future__ import annotations

import json

import torch

from patch_encoder_v1 import (  # noqa: E402
    HERE, ROOT, complete_answer, sha,
    load_pinned_thinker, prepare_audio_batch,
    resolve_hook_points, ActivationCapture, build_row_index,
)


def projected_windows(row_map: dict, start: float, end: float) -> dict[str, list[int]]:
    period = row_map["nominal_mel_frame_period_seconds"]
    projected = row_map["projected_rows"]
    word = [r["row_index"] for r in projected
            if start <= r["nominal_center_mel_frame"] * period < end]
    if not word:
        raise ValueError("No projected row in word slot")
    center = (start + end) / 2
    changed = next(c for c in row_map["chunks"]
                   if c["nominal_start_seconds"] <= center < c["nominal_end_seconds"])
    if not all(projected[i]["chunk_index"] == changed["chunk_index"] for i in word):
        raise ValueError("Word projected rows span multiple chunks")
    whole = [r["row_index"] for r in projected
             if r["chunk_index"] == changed["chunk_index"]]
    others = {}
    for r in projected:
        if r["chunk_index"] != changed["chunk_index"]:
            others.setdefault(r["chunk_index"], []).append(r["row_index"])
    wrong = next(indices[:len(word)] for _, indices in sorted(others.items())
                 if len(indices) >= len(word))
    return {"word_local": word, "changed_chunk": whole, "wrong_chunk": wrong}


def main() -> None:
    plan_path = HERE / "connector_patch_plan_v1.json"
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
    point = resolve_hook_points(model, audio_layers=(), text_layers=())[0]
    audio_config = model.config.audio_config
    extractor = processor.feature_extractor
    report = {
        "purpose": "fit-only audio-to-language connector causal patch",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "results": [],
    }
    output_path = HERE / "connector_patch_results_v1.json"
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
        row_map = inputs["red"]["row_map"]
        windows = projected_windows(row_map, group["word_slot_start_seconds"],
                                    group["word_slot_end_seconds"])
        clean = {word: complete_answer(model, processor, inputs[word]["batch"])
                 for word in ("red", "blue")}
        if not all(clean[word]["answer"].lower() == word
                   and clean[word]["stopped_on_eos"] for word in clean):
            raise ValueError("Clean behavior gate failed")
        activations = {}
        for word in ("red", "blue"):
            with torch.no_grad(), ActivationCapture((point,), detach=True) as capture:
                model(**inputs[word]["batch"], use_cache=False, return_dict=True)
            activations[word] = capture.activations[point.name][0]
        if activations["red"].shape != activations["blue"].shape:
            raise ValueError("Matched connector shapes differ")
        if activations["red"].shape[0] != row_map["projected_row_count"]:
            raise ValueError("Connector row count differs from source map")
        for recipient, donor_word in (("red", "blue"), ("blue", "red")):
            source, donor = activations[recipient], activations[donor_word]
            case = {
                "engine": engine, "recipient": recipient, "donor": donor_word,
                "projected_rows": row_map["projected_row_count"],
                "clean": clean[recipient], "conditions": {},
            }
            for name in ("word_local", "changed_chunk", "wrong_chunk"):
                rows = windows[name]
                case["conditions"][name] = {
                    "rows": rows,
                    "donor_minus_recipient_norm": float(
                        (donor[rows].float() - source[rows].float()).norm()
                    ),
                    "outcome": complete_answer(
                        model, processor, inputs[recipient]["batch"],
                        module=point.module, donor=donor, rows=rows,
                    ),
                }
            word_rows = windows["word_local"]
            case["conditions"]["self_patch"] = {
                "rows": word_rows,
                "outcome": complete_answer(
                    model, processor, inputs[recipient]["batch"],
                    module=point.module, donor=source, rows=word_rows,
                ),
            }
            target_norm = float((donor[word_rows].float() - source[word_rows].float()).norm())
            for seed in plan["random_seeds"]:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn((len(word_rows), source.shape[1]),
                                    generator=generator, dtype=torch.float32)
                noise *= target_norm / noise.norm()
                synthetic = source.clone()
                synthetic[word_rows] = (source[word_rows].float() + noise).to(source.dtype)
                effective = synthetic[word_rows].float() - source[word_rows].float()
                case["conditions"][f"random_seed_{seed}"] = {
                    "rows": word_rows,
                    "requested_delta_norm": target_norm,
                    "effective_bf16_delta_norm": float(effective.norm()),
                    "outcome": complete_answer(
                        model, processor, inputs[recipient]["batch"],
                        module=point.module, donor=synthetic, rows=word_rows,
                    ),
                }
            if case["conditions"]["self_patch"]["outcome"]["new_token_ids"] != clean[recipient]["new_token_ids"]:
                raise ValueError("Self patch changed complete answer")
            report["results"].append(case)
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(engine, recipient, "->", donor_word,
                  {name: x["outcome"]["answer"] for name, x in case["conditions"].items()},
                  flush=True)


if __name__ == "__main__":
    main()
