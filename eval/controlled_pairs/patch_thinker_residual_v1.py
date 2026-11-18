"""Fit-only audio-placeholder and terminal residual patching in Omni Thinker."""

from __future__ import annotations

import json

import torch

from patch_encoder_v1 import (  # noqa: E402
    HERE, ROOT, complete_answer, sha, load_pinned_thinker,
    prepare_audio_batch, resolve_hook_points, ActivationCapture,
    build_row_index,
)
from patch_connector_v1 import projected_windows  # noqa: E402


RED_ID = 6033
BLUE_ID = 10331


def answer_with_residual_patch(model, processor, batch, module,
                               donor: torch.Tensor, positions: list[int]) -> dict:
    prompt_len = int(batch["input_ids"].shape[1])
    if donor.ndim != 3 or donor.shape[0] != 1 or donor.shape[1] != prompt_len:
        raise ValueError("Donor residual shape differs from prompt")
    if not positions or min(positions) < 0 or max(positions) >= prompt_len:
        raise ValueError("Invalid patch positions")
    prefill_calls = 0
    decode_calls = 0

    def patch(_module, _inputs, output):
        nonlocal prefill_calls, decode_calls
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != donor.shape[2]:
            raise ValueError("Unexpected Thinker block output")
        if tensor.shape[1] != prompt_len:
            decode_calls += 1
            return None
        if tuple(tensor.shape) != tuple(donor.shape):
            raise ValueError("Donor and recipient residual shapes differ")
        changed = tensor.clone()
        changed[:, positions, :] = donor[:, positions, :].to(
            tensor.device, dtype=tensor.dtype
        )
        prefill_calls += 1
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = module.register_forward_hook(patch)
    try:
        with torch.inference_mode():
            result = model.generate(
                **batch, do_sample=False, use_cache=True, max_new_tokens=16,
                eos_token_id=int(processor.tokenizer.eos_token_id),
                pad_token_id=int(processor.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        handle.remove()
    if prefill_calls != 1:
        raise ValueError(f"Expected one prefill hook call, got {prefill_calls}")
    tokens = result.sequences[0, prompt_len:].tolist()
    first = result.scores[0][0].float()
    return {
        "new_token_ids": tokens,
        "answer": processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "stopped_on_eos": bool(tokens and tokens[-1] == processor.tokenizer.eos_token_id),
        "first_reply_blue_minus_red_logit": float(first[BLUE_ID] - first[RED_ID]),
        "prefill_hook_calls": prefill_calls,
        "decode_hook_calls_unmodified": decode_calls,
    }


def main() -> None:
    plan_path = HERE / "thinker_patch_plan_v1.json"
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
    audio_config = model.config.audio_config
    extractor = processor.feature_extractor
    points = resolve_hook_points(model, audio_layers=(),
                                 text_layers=tuple(plan["layers"]))
    point_by_layer = {int(p.name.split(".")[-1]): p for p in points[1:]}
    report = {
        "purpose": "fit-only residual pathway intervention; no held-out or security claim",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "results": [],
    }
    output_path = HERE / "thinker_patch_results_v1.json"
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
            raise ValueError("Matched pair text/audio placeholder IDs differ")
        row_map = inputs["red"]["row_map"]
        windows = projected_windows(row_map, group["word_slot_start_seconds"],
                                    group["word_slot_end_seconds"])
        projected = row_map["projected_rows"]
        word_positions = [projected[i]["audio_placeholder_token_position"]
                          for i in windows["word_local"]]
        wrong_positions = [projected[i]["audio_placeholder_token_position"]
                           for i in windows["wrong_chunk"]]
        all_audio_positions = [r["audio_placeholder_token_position"] for r in projected]
        terminal = int(inputs["red"]["batch"]["input_ids"].shape[1]) - 1
        clean = {word: complete_answer(model, processor, inputs[word]["batch"])
                 for word in ("red", "blue")}
        if not all(clean[word]["answer"].lower() == word
                   and clean[word]["stopped_on_eos"] for word in clean):
            raise ValueError("Clean behavior gate failed")
        captures = {}
        for word in ("red", "blue"):
            with torch.no_grad(), ActivationCapture(points[1:], detach=True) as capture:
                model(**inputs[word]["batch"], use_cache=False, return_dict=True)
            captures[word] = {
                layer: capture.activations[point_by_layer[layer].name][0]
                for layer in plan["layers"]
            }
        for layer in plan["layers"]:
            point = point_by_layer[layer]
            source_red = captures["red"][layer]
            source_blue = captures["blue"][layer]
            if tuple(source_red.shape) != tuple(source_blue.shape):
                raise ValueError("Matched residual shapes differ")
            for recipient, donor_word in (("red", "blue"), ("blue", "red")):
                source = captures[recipient][layer]
                donor = captures[donor_word][layer]
                batch = inputs[recipient]["batch"]
                delta = donor[:, word_positions, :].float() - source[:, word_positions, :].float()
                target_norm = float(delta.norm())
                case = {
                    "engine": engine, "layer": layer,
                    "recipient": recipient, "donor": donor_word,
                    "prompt_positions": terminal + 1,
                    "word_audio_positions": word_positions,
                    "wrong_audio_positions": wrong_positions,
                    "clean": clean[recipient],
                    "conditions": {},
                }
                for name, positions in (
                    ("word_audio_placeholders", word_positions),
                    ("all_audio_placeholders", all_audio_positions),
                    ("wrong_chunk_audio_placeholders", wrong_positions),
                    ("terminal_position", [terminal]),
                ):
                    case["conditions"][name] = {
                        "positions": positions,
                        "donor_minus_recipient_norm": float(
                            (donor[:, positions, :].float() - source[:, positions, :].float()).norm()
                        ),
                        "outcome": answer_with_residual_patch(
                            model, processor, batch, point.module, donor, positions,
                        ),
                    }
                case["conditions"]["self_word"] = {
                    "positions": word_positions,
                    "outcome": answer_with_residual_patch(
                        model, processor, batch, point.module, source, word_positions,
                    ),
                }
                if case["conditions"]["self_word"]["outcome"]["new_token_ids"] != clean[recipient]["new_token_ids"]:
                    raise ValueError("Self residual patch changed answer")
                if target_norm <= 0:
                    raise ValueError("Word-position donor difference has zero norm")
                for seed in plan["random_seeds"]:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
                    noise *= target_norm / noise.norm()
                    synthetic = source.clone()
                    synthetic[:, word_positions, :] = (
                        source[:, word_positions, :].float() + noise
                    ).to(source.dtype)
                    effective = synthetic[:, word_positions, :].float() - source[:, word_positions, :].float()
                    case["conditions"][f"random_seed_{seed}"] = {
                        "positions": word_positions,
                        "requested_delta_norm": target_norm,
                        "effective_bf16_delta_norm": float(effective.norm()),
                        "outcome": answer_with_residual_patch(
                            model, processor, batch, point.module, synthetic, word_positions,
                        ),
                    }
                report["results"].append(case)
                output_path.write_text(json.dumps(report, indent=2) + "\n")
                print(engine, layer, recipient, "->", donor_word,
                      {name: x["outcome"]["answer"] for name, x in case["conditions"].items()},
                      flush=True)


if __name__ == "__main__":
    main()
