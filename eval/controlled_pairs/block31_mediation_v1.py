"""Fit-only, preplanned sublayer interventions at Omni Thinker block 31."""

from __future__ import annotations

import json

import torch

from patch_encoder_v1 import (
    HERE, ROOT, BLUE_ID, RED_ID, complete_answer, load_pinned_thinker,
    prepare_audio_batch, sha, build_row_index,
)
from patch_connector_v1 import projected_windows


def run_with_hooks(model, processor, batch, edits: list[tuple[str, object, torch.Tensor, list[int]]]) -> dict:
    prompt_len = int(batch["input_ids"].shape[1])
    handles = []
    counts = {name: {"prefill": 0, "decode_unmodified": 0} for name, *_ in edits}
    if len(counts) != len(edits):
        raise ValueError("Duplicate hook name")

    def make_hook(name, donor, positions):
        if not positions or min(positions) < 0 or max(positions) >= prompt_len:
            raise ValueError(f"Invalid positions for {name}")

        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != donor.shape[2]:
                raise ValueError(f"Unexpected activation shape at {name}: {value.shape}")
            if value.shape[1] != prompt_len:
                counts[name]["decode_unmodified"] += 1
                return None
            if value.shape != donor.shape:
                raise ValueError(f"Donor shape mismatch at {name}")
            changed = value.clone()
            changed[:, positions, :] = donor[:, positions, :].to(value.device, dtype=value.dtype)
            counts[name]["prefill"] += 1
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        return hook

    for name, module, donor, positions in edits:
        handles.append(module.register_forward_hook(make_hook(name, donor, positions)))
    try:
        with torch.inference_mode():
            result = model.generate(
                **batch, do_sample=False, use_cache=True, max_new_tokens=16,
                eos_token_id=int(processor.tokenizer.eos_token_id),
                pad_token_id=int(processor.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        for handle in reversed(handles):
            handle.remove()
    if any(c["prefill"] != 1 for c in counts.values()):
        raise ValueError(f"Expected one prefill call per hook: {counts}")
    tokens = result.sequences[0, prompt_len:].tolist()
    first = result.scores[0][0].float()
    return {
        "new_token_ids": tokens,
        "answer": processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "stopped_on_eos": bool(tokens and tokens[-1] == processor.tokenizer.eos_token_id),
        "first_reply_blue_minus_red_logit": float(first[BLUE_ID] - first[RED_ID]),
        "hook_calls": counts,
    }


def capture_sublayers(model, batch, modules: dict[str, object]) -> dict[str, torch.Tensor]:
    activations = {}
    handles = []
    for name, module in modules.items():
        def record(_module, _inputs, output, key=name):
            tensor = output[0] if isinstance(output, tuple) else output
            if key in activations:
                raise ValueError(f"Repeated capture of {key}")
            activations[key] = tensor.detach().cpu().clone()
        handles.append(module.register_forward_hook(record))
    try:
        with torch.no_grad():
            model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in reversed(handles):
            handle.remove()
    if set(activations) != set(modules):
        raise ValueError(f"Missing captures: {set(modules) - set(activations)}")
    return activations


def main() -> None:
    plan_path = HERE / "block31_mediation_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    if sha(HERE / "thinker_handoff_results_v1.json") != plan["handoff_results_sha256"]:
        raise ValueError("Handoff results changed")
    manifest_path = (HERE / plan["input_manifest"]).resolve()
    if sha(manifest_path) != plan["input_manifest_sha256"]:
        raise ValueError("Manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["system_prompt"] != plan["system_prompt"]:
        raise ValueError("System prompt mismatch")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    blocks = model.model.language_model.layers
    modules = {
        "layer30": blocks[30],
        "attention31": blocks[31].self_attn,
        "mlp31": blocks[31].mlp,
    }
    output_path = HERE / "block31_mediation_results_v1.json"
    report = {
        "purpose": "posthoc fit-only sublayer causal diagnostic; no held-out/security claim",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "model_revision": plan["model_revision"],
        "results": [],
    }
    for engine in plan["engines"]:
        group = manifest["engines"][engine]
        inputs = {}
        for word in ("red", "blue"):
            info = group["outputs"][word]
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError(f"WAV hash changed: {path}")
            batch, digest = prepare_audio_batch(
                processor, path, system_prompt=plan["system_prompt"]
            )
            if digest != info["file_sha256"]:
                raise ValueError("Processor WAV digest mismatch")
            mel_frames = int(batch["feature_attention_mask"].sum())
            audio_positions = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
            row_map = build_row_index(
                mel_frames=mel_frames, audio_token_positions=audio_positions,
                n_window=model.config.audio_config.n_window,
                hop_length=processor.feature_extractor.hop_length,
                sample_rate=processor.feature_extractor.sampling_rate,
            )
            windows = projected_windows(
                row_map, group["word_slot_start_seconds"], group["word_slot_end_seconds"]
            )
            word_positions = [
                row_map["projected_rows"][i]["audio_placeholder_token_position"]
                for i in windows["word_local"]
            ]
            inputs[word] = {
                "batch": batch.to(model.device, dtype=model.dtype),
                "word_positions": word_positions,
                "wav_sha256": digest,
            }
        if not torch.equal(inputs["red"]["batch"]["input_ids"], inputs["blue"]["batch"]["input_ids"]):
            raise ValueError("Matched pair token IDs differ")
        if inputs["red"]["word_positions"] != inputs["blue"]["word_positions"]:
            raise ValueError("Matched pair word positions differ")
        clean = {word: complete_answer(model, processor, inputs[word]["batch"])
                 for word in ("red", "blue")}
        if not all(clean[word]["answer"].lower() == word and clean[word]["stopped_on_eos"]
                   for word in clean):
            raise ValueError("Clean behavior gate failed")
        captures = {
            word: capture_sublayers(model, inputs[word]["batch"], modules)
            for word in ("red", "blue")
        }
        terminal = int(inputs["red"]["batch"]["input_ids"].shape[1]) - 1
        for recipient, donor_word in (("red", "blue"), ("blue", "red")):
            batch = inputs[recipient]["batch"]
            source = captures[recipient]
            donor = captures[donor_word]
            word_positions = inputs[recipient]["word_positions"]
            if any(source[k].shape != donor[k].shape for k in modules):
                raise ValueError("Matched activation shapes differ")
            for k in modules:
                if source[k].shape[1] != terminal + 1:
                    raise ValueError("Activation length differs from prompt")
            upstream = ("layer30_word", modules["layer30"], donor["layer30"], word_positions)
            case = {
                "engine": engine, "recipient": recipient, "donor": donor_word,
                "recipient_wav_sha256": inputs[recipient]["wav_sha256"],
                "donor_wav_sha256": inputs[donor_word]["wav_sha256"],
                "word_audio_positions": word_positions,
                "terminal_position": terminal,
                "clean": clean[recipient],
                "conditions": {},
            }

            def run(name, edits, **metadata):
                case["conditions"][name] = {
                    **metadata,
                    "outcome": run_with_hooks(model, processor, batch, edits),
                }

            run("layer30_word_patch", [upstream])
            run("donor_attention_terminal_only", [
                ("attention31", modules["attention31"], donor["attention31"], [terminal])
            ])
            run("donor_mlp_terminal_only", [
                ("mlp31", modules["mlp31"], donor["mlp31"], [terminal])
            ])
            run("layer30_word_patch_plus_restore_clean_attention_terminal", [
                upstream,
                ("attention31", modules["attention31"], source["attention31"], [terminal]),
            ])
            run("layer30_word_patch_plus_restore_clean_mlp_terminal", [
                upstream,
                ("mlp31", modules["mlp31"], source["mlp31"], [terminal]),
            ])
            for key in ("attention31", "mlp31"):
                run(f"self_{key}_terminal", [
                    (key, modules[key], source[key], [terminal])
                ])
            delta = donor["attention31"][:, terminal, :].float() - source["attention31"][:, terminal, :].float()
            target_norm = float(delta.norm())
            if target_norm <= 0:
                raise ValueError("Zero attention donor delta")
            for seed in plan["random_seeds"]:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
                noise *= target_norm / noise.norm()
                synthetic = source["attention31"].clone()
                synthetic[:, terminal, :] = (
                    source["attention31"][:, terminal, :].float() + noise
                ).to(synthetic.dtype)
                effective = synthetic[:, terminal, :].float() - source["attention31"][:, terminal, :].float()
                run(f"random_attention_terminal_seed_{seed}", [
                    ("attention31", modules["attention31"], synthetic, [terminal])
                ], requested_delta_norm=target_norm, effective_bf16_delta_norm=float(effective.norm()))
            for key in ("self_attention31_terminal", "self_mlp31_terminal"):
                if case["conditions"][key]["outcome"]["new_token_ids"] != clean[recipient]["new_token_ids"]:
                    raise ValueError(f"Self patch changed complete answer: {key}")
            report["results"].append(case)
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(engine, recipient, "->", donor_word,
                  {name: c["outcome"]["answer"] for name, c in case["conditions"].items()},
                  flush=True)


if __name__ == "__main__":
    main()
