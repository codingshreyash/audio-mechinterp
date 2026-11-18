"""Posthoc fit-only scan of block-31 attention-head output slices."""

from __future__ import annotations

import json

import torch

from patch_encoder_v1 import (
    HERE, ROOT, BLUE_ID, RED_ID, complete_answer, load_pinned_thinker,
    prepare_audio_batch, sha, build_row_index,
)
from patch_connector_v1 import projected_windows
from patch_thinker_residual_v1 import answer_with_residual_patch


def capture_heads(model, batch, o_proj) -> torch.Tensor:
    seen = []

    def capture(_module, inputs):
        if len(inputs) != 1 or len(seen):
            raise ValueError("Unexpected o_proj input or repeated prehook")
        seen.append(inputs[0].detach().cpu().clone())

    handle = o_proj.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model(**batch, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(seen) != 1:
        raise ValueError("Missing o_proj capture")
    return seen[0]


def generate_with_head_patch(model, processor, batch, *, o_proj, head_source,
                             heads, head_dim, layer30=None, layer30_donor=None,
                             word_positions=None) -> dict:
    prompt_len = int(batch["input_ids"].shape[1])
    if head_source.ndim != 3 or head_source.shape[0] != 1 or head_source.shape[1] != prompt_len:
        raise ValueError("Head source length differs from prompt")
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("Head indices must be unique and nonempty")
    if min(heads) < 0 or max(heads) >= head_source.shape[2] // head_dim:
        raise ValueError("Invalid head index")
    if (layer30 is None) != (layer30_donor is None):
        raise ValueError("Incomplete upstream patch")
    if layer30 is not None and (not word_positions or layer30_donor.shape[1] != prompt_len):
        raise ValueError("Invalid upstream word patch")
    calls = {"head_prefill": 0, "head_decode_unmodified": 0,
             "layer30_prefill": 0, "layer30_decode_unmodified": 0}
    handles = []

    def patch_layer30(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.shape[1] != prompt_len:
            calls["layer30_decode_unmodified"] += 1
            return None
        if tensor.shape != layer30_donor.shape:
            raise ValueError("Layer30 donor shape mismatch")
        changed = tensor.clone()
        changed[:, word_positions, :] = layer30_donor[:, word_positions, :].to(
            tensor.device, dtype=tensor.dtype
        )
        calls["layer30_prefill"] += 1
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def patch_heads(_module, inputs):
        if len(inputs) != 1:
            raise ValueError("Unexpected o_proj arguments")
        tensor = inputs[0]
        if tensor.shape[1] != prompt_len:
            calls["head_decode_unmodified"] += 1
            return None
        if tensor.shape != head_source.shape:
            raise ValueError("o_proj donor shape mismatch")
        changed = tensor.clone()
        for head in heads:
            sl = slice(head * head_dim, (head + 1) * head_dim)
            changed[:, -1, sl] = head_source[:, -1, sl].to(tensor.device, dtype=tensor.dtype)
        calls["head_prefill"] += 1
        return (changed,)

    if layer30 is not None:
        handles.append(layer30.register_forward_hook(patch_layer30))
    handles.append(o_proj.register_forward_pre_hook(patch_heads))
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
    if calls["head_prefill"] != 1 or calls["layer30_prefill"] != int(layer30 is not None):
        raise ValueError(f"Unexpected prefill hook counts: {calls}")
    tokens = result.sequences[0, prompt_len:].tolist()
    first = result.scores[0][0].float()
    return {
        "new_token_ids": tokens,
        "answer": processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "stopped_on_eos": bool(tokens and tokens[-1] == processor.tokenizer.eos_token_id),
        "first_reply_blue_minus_red_logit": float(first[BLUE_ID] - first[RED_ID]),
        "hook_calls": calls,
    }


def main() -> None:
    plan_path = HERE / "block31_head_scan_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    prior_path = HERE / "block31_mediation_results_v1.json"
    if sha(prior_path) != plan["block31_mediation_results_sha256"]:
        raise ValueError("Prior mediation output changed")
    prior = json.loads(prior_path.read_text())
    manifest_path = (HERE / plan["input_manifest"]).resolve()
    if sha(manifest_path) != plan["input_manifest_sha256"]:
        raise ValueError("Manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["system_prompt"] != plan["system_prompt"]:
        raise ValueError("Task frame mismatch")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    blocks = model.model.language_model.layers
    attention = blocks[plan["layer"]].self_attn
    heads = plan["attention_head_indices"]
    head_dim = attention.head_dim
    if heads != list(range(attention.num_heads)) or attention.o_proj.in_features != len(heads) * head_dim:
        raise ValueError("Pinned attention head layout changed")
    result_path = HERE / "block31_head_scan_results_v1.json"
    report = {
        "purpose": "posthoc fit-only all-head scan; no selected-head, held-out, or security claim",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "prior_result_sha256": sha(prior_path),
        "num_heads": len(heads), "head_dim": head_dim,
        "results": [],
    }
    prior_cases = {(r["engine"], r["recipient"], r["donor"]): r for r in prior["results"]}
    for engine in plan["engines"]:
        group = manifest["engines"][engine]
        inputs = {}
        for word in ("red", "blue"):
            info = group["outputs"][word]
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError("WAV hash changed")
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
            window = projected_windows(
                row_map, group["word_slot_start_seconds"], group["word_slot_end_seconds"]
            )
            word_positions = [
                row_map["projected_rows"][i]["audio_placeholder_token_position"]
                for i in window["word_local"]
            ]
            inputs[word] = {"batch": batch.to(model.device, dtype=model.dtype),
                            "word_positions": word_positions, "wav_sha256": digest}
        if not torch.equal(inputs["red"]["batch"]["input_ids"], inputs["blue"]["batch"]["input_ids"]):
            raise ValueError("Matched pair token IDs differ")
        if inputs["red"]["word_positions"] != inputs["blue"]["word_positions"]:
            raise ValueError("Matched pair word windows differ")
        clean = {w: complete_answer(model, processor, inputs[w]["batch"]) for w in inputs}
        if not all(clean[w]["answer"].lower() == w and clean[w]["stopped_on_eos"] for w in clean):
            raise ValueError("Clean answer gate failed")
        captures = {}
        for word in inputs:
            batch = inputs[word]["batch"]
            captures[word] = {
                "head_input": capture_heads(model, batch, attention.o_proj),
                "layer30": None,
            }
            seen = []
            def capture_layer30(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                seen.append(tensor.detach().cpu().clone())
            handle = blocks[30].register_forward_hook(capture_layer30)
            try:
                with torch.no_grad():
                    model(**batch, use_cache=False, return_dict=True)
            finally:
                handle.remove()
            if len(seen) != 1:
                raise ValueError("Missing layer30 capture")
            captures[word]["layer30"] = seen[0]
        for recipient, donor_word in (("red", "blue"), ("blue", "red")):
            batch = inputs[recipient]["batch"]
            source, donor = captures[recipient], captures[donor_word]
            if any(source[k].shape != donor[k].shape for k in source):
                raise ValueError("Donor/recipient state shape mismatch")
            case = {
                "engine": engine, "recipient": recipient, "donor": donor_word,
                "recipient_wav_sha256": inputs[recipient]["wav_sha256"],
                "donor_wav_sha256": inputs[donor_word]["wav_sha256"],
                "word_audio_positions": inputs[recipient]["word_positions"],
                "terminal_position": int(batch["input_ids"].shape[1]) - 1,
                "clean": clean[recipient], "conditions": {},
            }
            def run(name, head_source, head_indices, with_upstream=False):
                case["conditions"][name] = generate_with_head_patch(
                    model, processor, batch, o_proj=attention.o_proj,
                    head_source=head_source, heads=head_indices, head_dim=head_dim,
                    layer30=blocks[30] if with_upstream else None,
                    layer30_donor=donor["layer30"] if with_upstream else None,
                    word_positions=inputs[recipient]["word_positions"] if with_upstream else None,
                )
            case["conditions"]["layer30_word_patch"] = answer_with_residual_patch(
                model, processor, batch, blocks[30], donor["layer30"],
                inputs[recipient]["word_positions"],
            )
            run("all_heads_donor_terminal_only", donor["head_input"], heads)
            run("all_heads_clean_restore_after_layer30_word_patch", source["head_input"], heads, True)
            run("self_all_heads_terminal", source["head_input"], heads)
            for head in heads:
                run(f"head_{head:02d}_donor_terminal_only", donor["head_input"], [head])
                run(f"head_{head:02d}_clean_restore_after_layer30_word_patch",
                    source["head_input"], [head], True)
            expected = prior_cases[(engine, recipient, donor_word)]["conditions"]
            if case["conditions"]["layer30_word_patch"]["new_token_ids"] != expected["layer30_word_patch"]["outcome"]["new_token_ids"]:
                raise ValueError("Upstream word control disagrees with prior result")
            if case["conditions"]["all_heads_donor_terminal_only"]["new_token_ids"] != expected["donor_attention_terminal_only"]["outcome"]["new_token_ids"]:
                raise ValueError("All-head donor control disagrees with attention-output patch")
            if case["conditions"]["all_heads_clean_restore_after_layer30_word_patch"]["new_token_ids"] != expected["layer30_word_patch_plus_restore_clean_attention_terminal"]["outcome"]["new_token_ids"]:
                raise ValueError("All-head restoration disagrees with attention-output patch")
            if case["conditions"]["self_all_heads_terminal"]["new_token_ids"] != clean[recipient]["new_token_ids"]:
                raise ValueError("Self all-head patch changed complete answer")
            if not all(c["stopped_on_eos"] for c in case["conditions"].values()):
                raise ValueError("One head-scan generation did not stop on EOS")
            report["results"].append(case)
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            donor_only = [h for h in heads if case["conditions"][f"head_{h:02d}_donor_terminal_only"]["answer"].lower() == donor_word]
            restore = [h for h in heads if case["conditions"][f"head_{h:02d}_clean_restore_after_layer30_word_patch"]["answer"].lower() == recipient]
            print(engine, recipient, "->", donor_word, "donor_head_flips", donor_only,
                  "single_head_restores", restore, flush=True)


if __name__ == "__main__":
    main()
