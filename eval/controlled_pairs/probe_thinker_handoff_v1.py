"""Posthoc fit-only audio-position to answer-position causal handoff probe."""

from __future__ import annotations

import json

import torch

from patch_encoder_v1 import (
    HERE, ROOT, complete_answer, sha, load_pinned_thinker,
    prepare_audio_batch, resolve_hook_points, ActivationCapture,
)
from patch_thinker_residual_v1 import answer_with_residual_patch


def main() -> None:
    plan_path = HERE / "thinker_handoff_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    if sha(HERE / "thinker_layer_sweep_results_v1.json") != plan["word_layer_sweep_results_sha256"]:
        raise ValueError("Prior word-position layer sweep changed")
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
    points = resolve_hook_points(
        model, audio_layers=(), text_layers=tuple(plan["terminal_swap_layers"])
    )
    text_points = {int(p.name.split(".")[-1]): p for p in points[1:]}
    report = {
        "purpose": "posthoc fit-only full-answer handoff diagnostic",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "word_sweep_sha256": plan["word_layer_sweep_results_sha256"],
        "results": [],
    }
    output_path = HERE / "thinker_handoff_results_v1.json"
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
            inputs[word] = batch.to(model.device, dtype=model.dtype)
        if not torch.equal(inputs["red"]["input_ids"], inputs["blue"]["input_ids"]):
            raise ValueError("Matched pair input IDs differ")
        prompt_len = int(inputs["red"]["input_ids"].shape[1])
        terminal = prompt_len - 1
        audio_positions = (
            inputs["red"]["input_ids"][0] == processor.audio_token_id
        ).nonzero().flatten().tolist()
        clean = {
            word: complete_answer(model, processor, inputs[word])
            for word in ("red", "blue")
        }
        if not all(
            clean[word]["answer"].lower() == word and clean[word]["stopped_on_eos"]
            for word in clean
        ):
            raise ValueError("Clean behavior gate failed")
        captures = {}
        for word in ("red", "blue"):
            with torch.no_grad(), ActivationCapture(points[1:], detach=True) as capture:
                model(**inputs[word], use_cache=False, return_dict=True)
            captures[word] = {
                layer: capture.activations[text_points[layer].name][0]
                for layer in plan["terminal_swap_layers"]
            }
        for layer in plan["terminal_swap_layers"]:
            point = text_points[layer]
            for recipient, donor_word in (("red", "blue"), ("blue", "red")):
                source = captures[recipient][layer]
                donor = captures[donor_word][layer]
                if source.shape != donor.shape or source.shape[1] != prompt_len:
                    raise ValueError("Matched residual shapes differ")
                batch = inputs[recipient]
                case = {
                    "engine": engine, "layer": layer,
                    "recipient": recipient, "donor": donor_word,
                    "clean": clean[recipient], "conditions": {},
                }

                def run(name: str, donor_tensor: torch.Tensor, positions: list[int]) -> None:
                    case["conditions"][name] = {
                        "positions": positions,
                        "donor_minus_recipient_norm": float(
                            (donor_tensor[:, positions, :].float()
                             - source[:, positions, :].float()).norm()
                        ),
                        "outcome": answer_with_residual_patch(
                            model, processor, batch, point.module,
                            donor_tensor, positions,
                        ),
                    }

                run("terminal_donor", donor, [terminal])
                if layer in plan["all_audio_swap_layers"]:
                    run("all_audio_donor", donor, audio_positions)
                if layer in plan["boundary_control_layers"]:
                    run("self_terminal", source, [terminal])
                    if (case["conditions"]["self_terminal"]["outcome"]["new_token_ids"]
                            != clean[recipient]["new_token_ids"]):
                        raise ValueError("Self terminal patch changed complete answer")
                    delta = donor[:, terminal, :].float() - source[:, terminal, :].float()
                    target_norm = float(delta.norm())
                    if target_norm <= 0:
                        raise ValueError("Terminal donor delta has zero norm")
                    for seed in plan["random_seeds"]:
                        generator = torch.Generator(device="cpu").manual_seed(seed)
                        noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
                        noise *= target_norm / noise.norm()
                        synthetic = source.clone()
                        synthetic[:, terminal, :] = (
                            source[:, terminal, :].float() + noise
                        ).to(source.dtype)
                        run(f"random_terminal_seed_{seed}", synthetic, [terminal])
                        case["conditions"][f"random_terminal_seed_{seed}"]["requested_delta_norm"] = target_norm
                report["results"].append(case)
                output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                engine, "layer", layer,
                [(r["recipient"], r["conditions"]["terminal_donor"]["outcome"]["answer"])
                 for r in report["results"][-2:]],
                flush=True,
            )


if __name__ == "__main__":
    main()
