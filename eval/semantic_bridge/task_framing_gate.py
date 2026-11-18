"""Fit-only paired typed/audio generation gate for the frozen bridge manifest.

This checks whether the chat-wrapped task elicits meaningful continuations.
It does not score held-out concepts or fit/read out a lens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "qwen_omni"))

from adapter import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_REVISION,
    load_pinned_thinker,
    prepare_audio_batch,
)


SYSTEM_PROMPT = "Continue the user's text naturally. Do not explain your continuation."


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed_batch(processor, prompt: str):
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    return processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def generate(model, processor, batch, max_new_tokens: int) -> dict:
    prompt_tokens = int(batch["input_ids"].shape[1])
    batch = batch.to(model.device, dtype=model.dtype)
    eos_id = int(processor.tokenizer.eos_token_id)
    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    token_ids = output[0, prompt_tokens:].tolist()
    return {
        "prompt_tokens": prompt_tokens,
        "new_token_ids": token_ids,
        "stopped_on_eos": bool(token_ids and token_ids[-1] == eos_id),
        "decoded_plain": processor.tokenizer.decode(
            token_ids, skip_special_tokens=True
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        raise ValueError("max-new-tokens must be in 1..256")
    render = json.loads(args.render_manifest.read_text())
    if render["selection"]["split"] != "fit":
        raise ValueError("Task-framing gate is fit-only")
    model, processor = load_pinned_thinker(
        allow_download=False,
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    report = {
        "purpose": "fit-only paired task-framing gate; no held-out metric",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "render_manifest_sha256": file_hash(args.render_manifest),
        "source_manifest_sha256": render["input_manifest_sha256"],
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": int(processor.tokenizer.eos_token_id),
        "items": [],
    }
    for record in render["records"]:
        item = record["source_item"]
        if item["split"] != "fit":
            raise ValueError(f"Non-fit item present: {item['id']}")
        result = {
            "item_id": item["id"],
            "expected_bridge_labels": item["expected_bridge_labels"],
            "typed_prompt_sha256": hashlib.sha256(
                item["typed_input"]["prompt"].encode()
            ).hexdigest(),
            "typed": generate(
                model,
                processor,
                typed_batch(processor, item["typed_input"]["prompt"]),
                args.max_new_tokens,
            ),
            "audio": {},
        }
        for engine in ("piper", "kokoro"):
            info = record["outputs"][engine]["processed"]
            path = (args.render_manifest.parent / info["path"]).resolve()
            if file_hash(path) != info["file_sha256"]:
                raise ValueError(f"Audio hash mismatch: {path}")
            batch, digest = prepare_audio_batch(
                processor, path, system_prompt=SYSTEM_PROMPT
            )
            result["audio"][engine] = {
                "audio_sha256": digest,
                **generate(model, processor, batch, args.max_new_tokens),
            }
        report["items"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(item["id"], json.dumps({
            "typed": result["typed"]["decoded_plain"],
            "piper": result["audio"]["piper"]["decoded_plain"],
            "kokoro": result["audio"]["kokoro"]["decoded_plain"],
        }), flush=True)


if __name__ == "__main__":
    main()
