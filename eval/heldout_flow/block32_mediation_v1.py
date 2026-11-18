"""Posthoc block-32 sublayer diagnostic on frozen spoken-word transfer clips."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval" / "controlled_pairs"))
from patch_encoder_v1 import (  # noqa: E402
    load_pinned_thinker, prepare_audio_batch, sha, build_row_index,
)
from block31_mediation_v1 import capture_sublayers  # noqa: E402
from evaluate_v1 import normalized_answer, single_token_id, select_positions  # noqa: E402


def run(model, processor, batch, *, recipient: str, donor_word: str,
        candidate_ids: dict[str, int | None], edits: list[tuple[str, object, torch.Tensor, list[int]]]) -> dict:
    prompt_len = int(batch["input_ids"].shape[1])
    counts = {name: {"prefill": 0, "decode_unmodified": 0} for name, *_ in edits}
    if len(counts) != len(edits):
        raise ValueError("Duplicate hook names")
    handles = []

    def make_hook(name, donor_state, positions):
        if donor_state.ndim != 3 or donor_state.shape[0] != 1 or donor_state.shape[1] != prompt_len:
            raise ValueError(f"Bad donor state at {name}")
        if not positions or min(positions) < 0 or max(positions) >= prompt_len:
            raise ValueError(f"Bad patch positions at {name}")

        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != donor_state.shape[2]:
                raise ValueError(f"Unexpected activation shape at {name}")
            if value.shape[1] != prompt_len:
                counts[name]["decode_unmodified"] += 1
                return None
            if value.shape != donor_state.shape:
                raise ValueError(f"Donor shape mismatch at {name}")
            changed = value.clone()
            changed[:, positions, :] = donor_state[:, positions, :].to(value.device, dtype=value.dtype)
            counts[name]["prefill"] += 1
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        return hook

    for name, module, donor_state, positions in edits:
        handles.append(module.register_forward_hook(make_hook(name, donor_state, positions)))
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
    if any(v["prefill"] != 1 for v in counts.values()):
        raise ValueError(f"Expected one prefill per hook: {counts}")
    tokens = result.sequences[0, prompt_len:].tolist()
    answer = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    eos = bool(tokens and tokens[-1] == processor.tokenizer.eos_token_id)
    margin = None
    if candidate_ids[recipient] is not None and candidate_ids[donor_word] is not None:
        scores = result.scores[0][0].float()
        margin = float(scores[candidate_ids[donor_word]] - scores[candidate_ids[recipient]])
    return {
        "new_token_ids": tokens, "answer": answer,
        "normalized_answer": normalized_answer(answer),
        "stopped_on_eos": eos,
        "recipient_match": bool(eos and normalized_answer(answer) == recipient.casefold()),
        "donor_match": bool(eos and normalized_answer(answer) == donor_word.casefold()),
        "first_token_donor_minus_recipient_logit": margin,
        "hook_calls": counts,
    }


def main() -> None:
    plan_path = HERE / "block32_mediation_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    manifest_path = HERE / "stimuli_v1" / "manifest.json"
    prior_plan_path = HERE / "plan_v1.json"
    prior_result_path = HERE / "results_v1.json"
    if (sha(manifest_path) != plan["stimulus_manifest_sha256"]
            or sha(prior_plan_path) != plan["heldout_plan_sha256"]
            or sha(prior_result_path) != plan["heldout_results_sha256"]):
        raise ValueError("Input, plan, or prior result changed")
    manifest = json.loads(manifest_path.read_text())
    prior = json.loads(prior_result_path.read_text())
    if manifest["system_prompt"] != plan["system_prompt"]:
        raise ValueError("System prompt changed")
    prior_pairs = {(p["engine"], p["pair_id"], p["phrasing_id"]): p for p in prior["pairs"]}
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    blocks = model.model.language_model.layers
    modules = {
        "upstream31": blocks[31], "postblock32": blocks[32],
        "attention32": blocks[32].self_attn, "mlp32": blocks[32].mlp,
    }
    result_path = HERE / "block32_mediation_results_v1.json"
    report = {
        "purpose": "posthoc held-out-input block32 diagnostic; no confirmatory or security claim",
        "plan_sha256": sha(plan_path),
        "stimulus_manifest_sha256": sha(manifest_path),
        "heldout_results_sha256": sha(prior_result_path),
        "results": [],
    }
    for pair in manifest["pairs"]:
        key = (pair["engine"], pair["pair_id"], pair["phrasing_id"])
        earlier = prior_pairs[key]
        words = pair["words"]
        candidate_ids = {word: single_token_id(processor, word) for word in words}
        inputs = {}
        for word in words:
            info = pair["outputs"][word]
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
            word_positions, _ = select_positions(
                row_map, pair["word_slot_start_seconds"], pair["word_slot_end_seconds"]
            )
            if word_positions != earlier["word_positions"]:
                raise ValueError("Word positions differ from held-out run")
            inputs[word] = {"batch": batch.to(model.device, dtype=model.dtype),
                            "word_positions": word_positions}
        if not torch.equal(inputs[words[0]]["batch"]["input_ids"], inputs[words[1]]["batch"]["input_ids"]):
            raise ValueError("Matched input IDs differ")
        captures = {word: capture_sublayers(model, inputs[word]["batch"], modules)
                    for word in words}
        for recipient, donor_word in ((words[0], words[1]), (words[1], words[0])):
            source, donor = captures[recipient], captures[donor_word]
            batch = inputs[recipient]["batch"]
            terminal = int(batch["input_ids"].shape[1]) - 1
            word_positions = inputs[recipient]["word_positions"]
            if any(source[k].shape != donor[k].shape or source[k].shape[1] != terminal + 1
                   for k in modules):
                raise ValueError("Donor/recipient shapes differ")
            upstream = ("upstream31", modules["upstream31"], donor["upstream31"], word_positions)
            case = {
                "engine": pair["engine"], "pair_id": pair["pair_id"],
                "phrasing_id": pair["phrasing_id"], "recipient": recipient,
                "donor": donor_word, "word_positions": word_positions,
                "terminal_position": terminal, "conditions": {},
            }

            def execute(name, edits, **metadata):
                case["conditions"][name] = {
                    **metadata,
                    "outcome": run(model, processor, batch, recipient=recipient,
                                   donor_word=donor_word, candidate_ids=candidate_ids,
                                   edits=edits),
                }

            execute("upstream31_word_donor", [upstream])
            execute("postblock32_terminal_donor", [
                ("postblock32", modules["postblock32"], donor["postblock32"], [terminal])
            ])
            execute("donor_attention32_terminal_only", [
                ("attention32", modules["attention32"], donor["attention32"], [terminal])
            ])
            execute("donor_mlp32_terminal_only", [
                ("mlp32", modules["mlp32"], donor["mlp32"], [terminal])
            ])
            execute("upstream31_plus_restore_clean_attention32_terminal", [
                upstream,
                ("attention32", modules["attention32"], source["attention32"], [terminal]),
            ])
            execute("upstream31_plus_restore_clean_mlp32_terminal", [
                upstream,
                ("mlp32", modules["mlp32"], source["mlp32"], [terminal]),
            ])
            for sublayer in ("attention32", "mlp32"):
                execute(f"self_{sublayer}_terminal", [
                    (sublayer, modules[sublayer], source[sublayer], [terminal])
                ])
            delta = donor["attention32"][:, terminal, :].float() - source["attention32"][:, terminal, :].float()
            norm = float(delta.norm())
            if norm <= 0:
                raise ValueError("Zero attention donor difference")
            for seed in plan["random_seeds"]:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
                noise *= norm / noise.norm()
                synthetic = source["attention32"].clone()
                synthetic[:, terminal, :] = (source["attention32"][:, terminal, :].float()
                                             + noise).to(synthetic.dtype)
                effective = synthetic[:, terminal, :].float() - source["attention32"][:, terminal, :].float()
                execute(f"random_attention32_seed_{seed}", [
                    ("attention32", modules["attention32"], synthetic, [terminal])
                ], requested_delta_norm=norm, effective_bf16_delta_norm=float(effective.norm()))
            previous_case = next(c for c in earlier["cases"]
                                 if c["recipient"] == recipient and c["layer"] == 31)
            if case["conditions"]["upstream31_word_donor"]["outcome"]["new_token_ids"] != previous_case["conditions"]["word_donor"]["outcome"]["new_token_ids"]:
                raise ValueError("Upstream positive control disagrees with held-out result")
            if any(case["conditions"][f"self_{k}_terminal"]["outcome"]["new_token_ids"]
                   != earlier["clean"][recipient]["new_token_ids"] for k in ("attention32", "mlp32")):
                raise ValueError("Self patch changed complete answer")
            if not all(c["outcome"]["stopped_on_eos"] for c in case["conditions"].values()):
                raise ValueError("A continuation did not reach EOS")
            report["results"].append(case)
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            print(*key, recipient, "->", donor_word,
                  {name: c["outcome"]["answer"] for name, c in case["conditions"].items()
                   if not name.startswith("random_")}, flush=True)


if __name__ == "__main__":
    main()
