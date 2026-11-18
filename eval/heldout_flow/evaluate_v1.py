"""Run the frozen held-out Qwen Omni spoken-word information-flow transfer test."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval" / "controlled_pairs"))
from patch_encoder_v1 import (  # noqa: E402
    ActivationCapture, build_row_index, load_pinned_thinker,
    prepare_audio_batch, resolve_hook_points,
)
from adapter import MODEL_REVISION  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_answer(answer: str) -> str:
    value = answer.strip()
    if value and value[-1] in ".!?":
        value = value[:-1].strip()
    return value.casefold()


def single_token_id(processor, word: str) -> int | None:
    surface = word[0].upper() + word[1:]
    ids = processor.tokenizer.encode(surface, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def generate_answer(model, processor, batch, *, recipient: str, donor_word: str,
                    candidate_ids: dict[str, int | None], module=None,
                    donor_state=None, positions=None) -> dict:
    prompt_len = int(batch["input_ids"].shape[1])
    calls = {"prefill_modified": 0, "decode_unmodified": 0}
    handle = None
    if module is not None:
        if donor_state is None or not positions or min(positions) < 0 or max(positions) >= prompt_len:
            raise ValueError("Invalid patch state or positions")
        if donor_state.ndim != 3 or donor_state.shape[0] != 1 or donor_state.shape[1] != prompt_len:
            raise ValueError("Donor state differs from full prompt")

        def patch(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != donor_state.shape[2]:
                raise ValueError("Unexpected decoder residual shape")
            if tensor.shape[1] != prompt_len:
                calls["decode_unmodified"] += 1
                return None
            if tensor.shape != donor_state.shape:
                raise ValueError("Donor/recipient prompt state shape differs")
            changed = tensor.clone()
            changed[:, positions, :] = donor_state[:, positions, :].to(tensor.device, dtype=tensor.dtype)
            calls["prefill_modified"] += 1
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
        if handle is not None:
            handle.remove()
    if module is not None and calls["prefill_modified"] != 1:
        raise ValueError(f"Expected one prefill patch, saw {calls}")
    tokens = result.sequences[0, prompt_len:].tolist()
    answer = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    eos = bool(tokens and tokens[-1] == processor.tokenizer.eos_token_id)
    candidate_margin = None
    if candidate_ids[recipient] is not None and candidate_ids[donor_word] is not None:
        scores = result.scores[0][0].float()
        candidate_margin = float(scores[candidate_ids[donor_word]] - scores[candidate_ids[recipient]])
    return {
        "new_token_ids": tokens,
        "answer": answer,
        "normalized_answer": normalized_answer(answer),
        "stopped_on_eos": eos,
        "recipient_match": bool(eos and normalized_answer(answer) == recipient.casefold()),
        "donor_match": bool(eos and normalized_answer(answer) == donor_word.casefold()),
        "first_token_donor_minus_recipient_logit": candidate_margin,
        "hook_calls": calls,
    }


def select_positions(row_map: dict, start: float, end: float) -> tuple[list[int], list[int] | None]:
    period = row_map["nominal_mel_frame_period_seconds"]
    projected = row_map["projected_rows"]
    times = [r["nominal_center_mel_frame"] * period for r in projected]
    word_rows = [i for i, time in enumerate(times) if start <= time < end]
    if not word_rows:
        raise ValueError("No projected rows in word interval")
    width = len(word_rows)
    after = next((i for i in range(len(times) - width + 1)
                  if times[i] >= end + 0.20), None)
    if after is not None:
        wrong_rows = list(range(after, after + width))
    else:
        before = next((i for i in range(len(times) - width + 1)
                       if times[i + width - 1] <= start - 0.20), None)
        wrong_rows = list(range(before, before + width)) if before is not None else None
    word_positions = [projected[i]["audio_placeholder_token_position"] for i in word_rows]
    wrong_positions = ([projected[i]["audio_placeholder_token_position"] for i in wrong_rows]
                       if wrong_rows is not None else None)
    if wrong_positions and set(word_positions) & set(wrong_positions):
        raise ValueError("Wrong-time control overlaps target word")
    return word_positions, wrong_positions


def main() -> None:
    plan_path = HERE / "plan_v1.json"
    rule_path = HERE / "evaluator_rule_addendum_v1.json"
    manifest_path = HERE / "stimuli_v1" / "manifest.json"
    plan = json.loads(plan_path.read_text())
    rule = json.loads(rule_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha(plan_path) != rule["plan_v1_sha256"] or sha(manifest_path) != rule["stimulus_manifest_sha256"]:
        raise ValueError("Frozen plan or stimulus manifest changed")
    if manifest["plan_sha256"] != sha(plan_path) or manifest["system_prompt"] != plan["system_prompt"]:
        raise ValueError("Stimulus render does not match frozen plan")
    pairs = manifest["pairs"]
    if len(pairs) != plan["expected"]["matched_pairs"]:
        raise ValueError("Held-out pair count changed")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if MODEL_REVISION != plan["model"]["revision"]:
        raise ValueError("Checkpoint revision changed")
    layers = plan["evaluation"]["layers"]
    points = resolve_hook_points(model, audio_layers=(), text_layers=layers)
    by_layer = {int(p.name.split(".")[-1]): p for p in points[1:]}
    if set(by_layer) != set(layers):
        raise ValueError("Decoder layer hook layout changed")
    output_path = HERE / "results_v1.json"
    report = {
        "purpose": "frozen small held-out word/voice/phrasing transfer test; no security claim",
        "plan_sha256": sha(plan_path), "rule_addendum_sha256": sha(rule_path),
        "stimulus_manifest_sha256": sha(manifest_path),
        "model_revision": plan["model"]["revision"],
        "pairs": [],
    }
    for pair in pairs:
        words = pair["words"]
        if len(words) != 2 or not pair["outside_slot_pcm_identical"] or not pair["inside_slot_pcm_differs"]:
            raise ValueError("Invalid matched-pair manifest")
        candidate_ids = {word: single_token_id(processor, word) for word in words}
        inputs = {}
        for word in words:
            info = pair["outputs"][word]
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError(f"Frozen WAV changed: {path}")
            batch, digest = prepare_audio_batch(
                processor, path, system_prompt=plan["system_prompt"]
            )
            if digest != info["file_sha256"]:
                raise ValueError("Processed WAV digest mismatch")
            mel_frames = int(batch["feature_attention_mask"].sum())
            audio_positions = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
            row_map = build_row_index(
                mel_frames=mel_frames, audio_token_positions=audio_positions,
                n_window=model.config.audio_config.n_window,
                hop_length=processor.feature_extractor.hop_length,
                sample_rate=processor.feature_extractor.sampling_rate,
            )
            word_positions, wrong_positions = select_positions(
                row_map, pair["word_slot_start_seconds"], pair["word_slot_end_seconds"]
            )
            inputs[word] = {
                "batch": batch.to(model.device, dtype=model.dtype),
                "sha256": digest, "mel_frames": mel_frames,
                "row_map": row_map, "word_positions": word_positions,
                "wrong_positions": wrong_positions,
            }
        if not torch.equal(inputs[words[0]]["batch"]["input_ids"], inputs[words[1]]["batch"]["input_ids"]):
            raise ValueError("Matched pair input token IDs differ")
        if inputs[words[0]]["row_map"] != inputs[words[1]]["row_map"]:
            raise ValueError("Matched pair time maps differ")
        if inputs[words[0]]["word_positions"] != inputs[words[1]]["word_positions"]:
            raise ValueError("Matched pair word positions differ")
        if inputs[words[0]]["wrong_positions"] != inputs[words[1]]["wrong_positions"]:
            raise ValueError("Matched pair wrong-time positions differ")
        terminal = int(inputs[words[0]]["batch"]["input_ids"].shape[1]) - 1
        pair_result = {
            "engine": pair["engine"], "voice": pair["voice"],
            "pair_id": pair["pair_id"], "phrasing_id": pair["phrasing_id"],
            "words": words, "candidate_token_ids": candidate_ids,
            "word_slot_seconds": [pair["word_slot_start_seconds"], pair["word_slot_end_seconds"]],
            "word_positions": inputs[words[0]]["word_positions"],
            "wrong_time_positions": inputs[words[0]]["wrong_positions"],
            "terminal_position": terminal,
            "clean": {}, "cache_checks": {}, "cases": [],
        }
        report["pairs"].append(pair_result)
        for word in words:
            pair_result["clean"][word] = generate_answer(
                model, processor, inputs[word]["batch"],
                recipient=word, donor_word=words[1 - words.index(word)],
                candidate_ids=candidate_ids,
            )
        pair_result["clean_eligible"] = all(
            pair_result["clean"][word]["recipient_match"] for word in words
        )
        captures = {}
        for word in words:
            batch = inputs[word]["batch"]
            captures[word] = {}
            for use_cache in (False, True):
                with torch.no_grad(), ActivationCapture(points[1:], detach=True) as capture:
                    model(**batch, use_cache=use_cache, return_dict=True)
                captures[word][use_cache] = {
                    layer: capture.activations[by_layer[layer].name][0]
                    for layer in layers
                }
                if any(len(capture.activations[by_layer[layer].name]) != 1 for layer in layers):
                    raise ValueError("Unexpected decoder capture calls")
            pair_result["cache_checks"][word] = {}
            for layer in layers:
                uncached, cached = captures[word][False][layer], captures[word][True][layer]
                if uncached.shape != cached.shape or uncached.shape[1] != terminal + 1:
                    raise ValueError("Cached/uncached prefill shape differs")
                diff = (cached.float() - uncached.float()).abs()
                pair_result["cache_checks"][word][str(layer)] = {
                    "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                    "changed_elements": int((diff != 0).sum()),
                }
        for recipient, donor_word in ((words[0], words[1]), (words[1], words[0])):
            batch = inputs[recipient]["batch"]
            word_positions = inputs[recipient]["word_positions"]
            wrong_positions = inputs[recipient]["wrong_positions"]
            for layer in layers:
                module = by_layer[layer].module
                source = captures[recipient][False][layer]
                donor = captures[donor_word][False][layer]
                if source.shape != donor.shape:
                    raise ValueError("Donor/recipient residual shapes differ")
                delta = donor[:, word_positions, :].float() - source[:, word_positions, :].float()
                norm = float(delta.norm())
                case = {
                    "recipient": recipient, "donor": donor_word, "layer": layer,
                    "clean_eligible": pair_result["clean_eligible"],
                    "word_delta_norm": norm, "conditions": {},
                }

                def run(name, state, positions, **metadata):
                    case["conditions"][name] = {
                        **metadata,
                        "outcome": generate_answer(
                            model, processor, batch, recipient=recipient,
                            donor_word=donor_word, candidate_ids=candidate_ids,
                            module=module, donor_state=state, positions=positions,
                        ),
                    }

                run("word_donor", donor, word_positions)
                run("self_word", source, word_positions)
                if wrong_positions is None:
                    case["conditions"]["wrong_time_donor"] = {
                        "ineligible_reason": "no same-width run at required temporal distance"
                    }
                else:
                    run("wrong_time_donor", donor, wrong_positions,
                        wrong_time_delta_norm=float((donor[:, wrong_positions, :].float()
                                                     - source[:, wrong_positions, :].float()).norm()))
                if norm <= 0:
                    raise ValueError("Donor word-position edit has zero norm")
                for seed in (0, 1, 2):
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
                    noise *= norm / noise.norm()
                    synthetic = source.clone()
                    synthetic[:, word_positions, :] = (
                        source[:, word_positions, :].float() + noise
                    ).to(source.dtype)
                    effective = synthetic[:, word_positions, :].float() - source[:, word_positions, :].float()
                    run(f"random_seed_{seed}", synthetic, word_positions,
                        requested_norm=norm, effective_bf16_norm=float(effective.norm()))
                if layer in (31, 35):
                    run("terminal_donor", donor, [terminal],
                        terminal_delta_norm=float((donor[:, terminal, :].float()
                                                   - source[:, terminal, :].float()).norm()))
                if (case["conditions"]["self_word"]["outcome"]["new_token_ids"]
                        != pair_result["clean"][recipient]["new_token_ids"]):
                    raise ValueError("Self patch changed complete answer")
                pair_result["cases"].append(case)
                output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(pair["engine"], pair["pair_id"], pair["phrasing_id"], recipient,
                  "clean", pair_result["clean"][recipient]["answer"],
                  "layer30", next(c["conditions"]["word_donor"]["outcome"]["answer"]
                                  for c in pair_result["cases"] if c["recipient"] == recipient and c["layer"] == 30),
                  flush=True)
    if len(report["pairs"]) != len(pairs):
        raise ValueError("Result pair count differs from plan")


if __name__ == "__main__":
    main()
