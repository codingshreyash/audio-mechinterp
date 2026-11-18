"""Independent fit-only Piper layer-15 replay, without patch_encoder_v1 helpers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "models" / "qwen_omni"))
from adapter import load_pinned_thinker, prepare_audio_batch  # noqa: E402

MANIFEST = ROOT / "eval" / "tts" / "outputs" / "controlled_pair_v1" / "controlled_pair_manifest.json"
OUTPUT = HERE / "independent_piper_layer15.json"
LAYER = 15
RED_ID = 6033
BLUE_ID = 10331


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    piper = manifest["engines"]["piper"]
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16, device_map="mps", low_cpu_mem_usage=True
    )
    block = model.model.audio_tower.layers[LAYER]
    batches, activations, audio_hashes = {}, {}, {}
    for word in ("red", "blue"):
        info = piper["outputs"][word]
        path = ROOT / info["path"]
        if sha(path) != info["file_sha256"]:
            raise ValueError(f"Audio hash mismatch: {word}")
        batch, digest = prepare_audio_batch(
            processor, path, system_prompt=manifest["system_prompt"]
        )
        audio_hashes[word] = digest
        batches[word] = batch.to(model.device, dtype=model.dtype)
        captured = []

        def observe(_module, _inputs, output):
            captured.append((output[0] if isinstance(output, tuple) else output).detach().cpu())

        handle = block.register_forward_hook(observe)
        try:
            with torch.inference_mode():
                model(**batches[word], use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("Encoder layer did not run exactly once")
        activations[word] = captured[0]
    if activations["red"].shape != activations["blue"].shape:
        raise ValueError("Matched audio paths produced different encoder shapes")
    rows_total = int(activations["red"].shape[0])
    start, stop = piper["word_slot_start_seconds"], piper["word_slot_end_seconds"]
    hop_s = 160 / 16000
    # 200 mel frames per attention chunk, two mel frames per encoder row.
    word_rows = [i for i in range(rows_total)
                 if start <= (i * 2 * hop_s) < stop]
    if not word_rows:
        raise ValueError("No word rows")
    changed_chunk_index = math.floor(((start + stop) / 2) / (200 * hop_s))
    changed_chunk_start = changed_chunk_index * 100
    wrong_chunk_start = 100 if changed_chunk_index == 0 else 0
    wrong_rows = list(range(wrong_chunk_start, wrong_chunk_start + len(word_rows)))
    if word_rows[-1] >= (changed_chunk_index + 1) * 100:
        raise ValueError("Word rows span chunks")

    def answer(recipient: str, donor: str | None = None, rows: list[int] | None = None):
        calls = []
        handles = []
        if donor is not None:
            if rows is None:
                raise ValueError("Rows required for a patch")
            def patch(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                modified = tensor.clone()
                modified[rows] = activations[donor][rows].to(tensor.device, dtype=tensor.dtype)
                calls.append(1)
                return (modified, *output[1:]) if isinstance(output, tuple) else modified
            handles.append(block.register_forward_hook(patch))
        try:
            with torch.inference_mode():
                out = model.generate(
                    **batches[recipient], do_sample=False, use_cache=True,
                    max_new_tokens=16,
                    eos_token_id=int(processor.tokenizer.eos_token_id),
                    pad_token_id=int(processor.tokenizer.eos_token_id),
                    return_dict_in_generate=True, output_scores=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if donor is not None and calls != [1]:
            raise ValueError(f"Expected one hook call, got {calls}")
        prompt_tokens = int(batches[recipient]["input_ids"].shape[1])
        ids = out.sequences[0, prompt_tokens:].tolist()
        first = out.scores[0][0].float()
        return {
            "ids": ids,
            "answer": processor.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "eos": bool(ids and ids[-1] == processor.tokenizer.eos_token_id),
            "blue_minus_red_logit": float(first[BLUE_ID] - first[RED_ID]),
            "hook_calls": len(calls),
        }

    results = []
    for recipient, donor in (("red", "blue"), ("blue", "red")):
        results.append({
            "recipient": recipient,
            "donor": donor,
            "clean": answer(recipient),
            "word_local": answer(recipient, donor, word_rows),
            "self_patch": answer(recipient, recipient, word_rows),
            "wrong_chunk": answer(recipient, donor, wrong_rows),
        })
    report = {
        "purpose": "independent fit-only Piper layer-15 replay",
        "manifest_sha256": sha(MANIFEST),
        "audio_sha256": audio_hashes,
        "layer": LAYER,
        "encoder_shape": list(activations["red"].shape),
        "word_rows": word_rows,
        "wrong_rows": wrong_rows,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": [word_rows[0], word_rows[-1]],
                      "wrong": [wrong_rows[0], wrong_rows[-1]],
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
