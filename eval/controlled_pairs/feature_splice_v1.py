"""Causal input-feature splice for the fixed red/blue audio pair."""

from __future__ import annotations

import json
import math

import torch

from patch_encoder_v1 import (  # noqa: E402
    HERE, ROOT, complete_answer, load_pinned_thinker,
    prepare_audio_batch, sha,
)


def main() -> None:
    plan_path = HERE / "feature_splice_plan_v1.json"
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
    report = {
        "purpose": "fit-only input-feature splice, not a replayable waveform",
        "plan_sha256": sha(plan_path),
        "input_manifest_sha256": sha(manifest_path),
        "results": [],
    }
    output_path = HERE / "feature_splice_results_v1.json"
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
            inputs[word] = {
                k: (v.to(model.device, dtype=model.dtype)
                    if v.is_floating_point() else v.to(model.device))
                for k, v in batch.items()
            }
        red, blue = inputs["red"], inputs["blue"]
        for key in ("input_ids", "attention_mask", "feature_attention_mask"):
            if not torch.equal(red[key], blue[key]):
                raise ValueError(f"Matched pair differs outside mel features: {key}")
        valid_frames = int(red["feature_attention_mask"].sum())
        start = max(0, math.floor(group["word_slot_start_seconds"] * 100)
                    - plan["margin_frames"])
        stop = min(valid_frames, math.ceil(group["word_slot_end_seconds"] * 100)
                   + plan["margin_frames"] + 1)
        if start >= stop:
            raise ValueError("Empty word-adjacent mel interval")
        first = red["input_features"][..., :valid_frames]
        second = blue["input_features"][..., :valid_frames]
        diff = (first.float() - second.float()).abs().amax(dim=1)[0]
        outside = torch.cat((diff[:start], diff[stop:]))
        clean = {word: complete_answer(model, processor, inputs[word])
                 for word in ("red", "blue")}
        if not all(clean[word]["answer"].lower() == word
                   and clean[word]["stopped_on_eos"] for word in clean):
            raise ValueError("Clean behavior gate failed")
        for base_word, donor_word in (("red", "blue"), ("blue", "red")):
            base, donor = inputs[base_word], inputs[donor_word]
            word_only = {key: value.clone() for key, value in base.items()}
            word_only["input_features"][..., start:stop] = donor["input_features"][..., start:stop]
            outside_only = {key: value.clone() for key, value in donor.items()}
            outside_only["input_features"][..., start:stop] = base["input_features"][..., start:stop]
            case = {
                "engine": engine, "base": base_word, "donor": donor_word,
                "base_audio_sha256": group["outputs"][base_word]["file_sha256"],
                "donor_audio_sha256": group["outputs"][donor_word]["file_sha256"],
                "valid_mel_frames": valid_frames,
                "word_adjacent_mel_start": start,
                "word_adjacent_mel_end_exclusive": stop,
                "changed_frames_outside_interval_bf16": int((outside > 0).sum()),
                "max_abs_difference_outside_interval_bf16": float(outside.max()),
                "clean_base": clean[base_word],
                "clean_donor": clean[donor_word],
                "word_only": complete_answer(model, processor, word_only),
                "outside_only": complete_answer(model, processor, outside_only),
            }
            report["results"].append(case)
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(engine, base_word, "->", donor_word,
                  "word_only", case["word_only"]["answer"],
                  "outside_only", case["outside_only"]["answer"], flush=True)


if __name__ == "__main__":
    main()
