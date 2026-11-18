"""Fit-only reciprocal audio-encoder activation patch on matched color speech.

The patch plan, word-slot waveform samples and model pins are fixed in adjacent
artifacts. This script records full answers and first-token margins for every
prespecified layer/window/control. It makes no security or generalization claim.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "models" / "qwen_omni"))
sys.path.insert(0, str(ROOT / "models" / "qwen_omni" / "time_map"))
from adapter import (  # noqa: E402
    ActivationCapture, load_pinned_thinker, prepare_audio_batch,
    resolve_hook_points,
)
from index import build_row_index  # noqa: E402


RED_ID = 6033
BLUE_ID = 10331


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_answer(model, processor, batch, *, module=None, donor=None,
                    rows=None) -> dict:
    handles = []
    calls = []
    if module is not None:
        if donor is None or rows is None or not rows:
            raise ValueError("A patch needs donor states and nonempty rows")

        def patch(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if tensor.ndim != 2 or tuple(tensor.shape) != tuple(donor.shape):
                raise ValueError("Donor/recipient audio encoder shapes differ")
            changed = tensor.clone()
            changed[rows] = donor[rows].to(tensor.device, dtype=tensor.dtype)
            calls.append(len(rows))
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handles.append(module.register_forward_hook(patch))
    try:
        with torch.inference_mode():
            output = model.generate(
                **batch, do_sample=False, use_cache=True,
                max_new_tokens=16,
                eos_token_id=int(processor.tokenizer.eos_token_id),
                pad_token_id=int(processor.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if module is not None and calls != [len(rows)]:
        raise ValueError(f"Audio hook was called {len(calls)} times, expected once")
    prompt_len = int(batch["input_ids"].shape[1])
    tokens = output.sequences[0, prompt_len:].tolist()
    first = output.scores[0][0].float()
    return {
        "new_token_ids": tokens,
        "answer": processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "stopped_on_eos": bool(tokens and tokens[-1] == processor.tokenizer.eos_token_id),
        "first_reply_blue_minus_red_logit": float(first[BLUE_ID] - first[RED_ID]),
        "hook_calls": len(calls),
    }


def row_windows(row_map: dict, slot_start: float, slot_end: float) -> dict[str, list[int]]:
    rows = row_map["encoder_rows"]
    word = [r["row_index"] for r in rows
            if slot_start <= r["nominal_center_seconds"] < slot_end]
    if not word:
        raise ValueError("No encoder rows map nominally into word slot")
    center = (slot_start + slot_end) / 2
    changed_chunk = next(c for c in row_map["chunks"]
                         if c["nominal_start_seconds"] <= center < c["nominal_end_seconds"])
    if not all(rows[i]["chunk_index"] == changed_chunk["chunk_index"] for i in word):
        raise ValueError("Word slot spans multiple attention chunks")
    chunk = list(range(changed_chunk["encoder_row_start"],
                       changed_chunk["encoder_row_end_exclusive"]))
    alternative = next(c for c in row_map["chunks"]
                       if c["chunk_index"] != changed_chunk["chunk_index"]
                       and c["encoder_row_end_exclusive"] - c["encoder_row_start"] >= len(word))
    wrong = list(range(alternative["encoder_row_start"],
                       alternative["encoder_row_start"] + len(word)))
    return {"word_local": word, "changed_chunk": chunk, "wrong_chunk": wrong}


def main() -> None:
    plan_path = HERE / "patch_plan_v1.json"
    plan = json.loads(plan_path.read_text())
    manifest_path = (HERE / plan["input_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["system_prompt"] != plan["system_prompt"]:
        raise ValueError("Task frame mismatch")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    report = {
        "purpose": "fit-only paired causal patch pilot, no held-out or security claim",
        "patch_plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "model_revision": plan["model_revision"],
        "system_prompt": plan["system_prompt"],
        "results": [],
    }
    output_path = HERE / "patch_results_v1.json"
    audio_config = model.config.audio_config
    extractor = processor.feature_extractor
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
                            "row_map": row_map, "audio_sha256": digest}
        red_map, blue_map = inputs["red"]["row_map"], inputs["blue"]["row_map"]
        if (red_map["encoder_row_count"] != blue_map["encoder_row_count"]
                or red_map["mel_frames"] != blue_map["mel_frames"]):
            raise ValueError("Matched pair has different model frame counts")
        windows = row_windows(red_map, group["word_slot_start_seconds"],
                              group["word_slot_end_seconds"])
        if windows != row_windows(blue_map, group["word_slot_start_seconds"],
                                  group["word_slot_end_seconds"]):
            raise ValueError("Matched pair row windows differ")
        baseline = {word: complete_answer(model, processor, inputs[word]["batch"])
                    for word in ("red", "blue")}
        if not all(baseline[word]["answer"].lower() == word
                   and baseline[word]["stopped_on_eos"] for word in baseline):
            raise ValueError(f"Clean answer positive control failed for {engine}")
        for layer in plan["layers"]:
            point = resolve_hook_points(model, audio_layers=(layer,), text_layers=())[0]
            activations = {}
            for word in ("red", "blue"):
                with torch.no_grad(), ActivationCapture((point,), detach=True) as capture:
                    model(**inputs[word]["batch"], use_cache=False, return_dict=True)
                activations[word] = capture.activations[point.name][0]
            if activations["red"].shape != activations["blue"].shape:
                raise ValueError("Matched encoder activation shapes differ")
            for recipient, donor_word in (("red", "blue"), ("blue", "red")):
                delta = activations[donor_word].float() - activations[recipient].float()
                case = {
                    "engine": engine, "layer": layer,
                    "recipient": recipient, "donor": donor_word,
                    "recipient_audio_sha256": inputs[recipient]["audio_sha256"],
                    "donor_audio_sha256": inputs[donor_word]["audio_sha256"],
                    "encoder_rows": red_map["encoder_row_count"],
                    "mel_frames": red_map["mel_frames"],
                    "word_slot_seconds": [group["word_slot_start_seconds"],
                                          group["word_slot_end_seconds"]],
                    "clean": baseline[recipient],
                    "conditions": {},
                }
                for name in ("word_local", "changed_chunk", "wrong_chunk"):
                    rows = windows[name]
                    case["conditions"][name] = {
                        "rows": rows,
                        "donor_minus_recipient_frobenius_norm": float(delta[rows].norm()),
                        "outcome": complete_answer(
                            model, processor, inputs[recipient]["batch"],
                            module=point.module, donor=activations[donor_word], rows=rows,
                        ),
                    }
                rows = windows["word_local"]
                case["conditions"]["self_patch"] = {
                    "rows": rows,
                    "outcome": complete_answer(
                        model, processor, inputs[recipient]["batch"],
                        module=point.module, donor=activations[recipient], rows=rows,
                    ),
                }
                if case["conditions"]["self_patch"]["outcome"]["new_token_ids"] != baseline[recipient]["new_token_ids"]:
                    raise ValueError("Self-patch changed the complete answer")
                report["results"].append(case)
                output_path.write_text(json.dumps(report, indent=2) + "\n")
                print(engine, layer, recipient, "->", donor_word,
                      {name: x["outcome"]["answer"] for name, x in case["conditions"].items()},
                      flush=True)


if __name__ == "__main__":
    main()
