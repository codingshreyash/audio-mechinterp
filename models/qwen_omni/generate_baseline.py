"""Record an unedited, complete Thinker answer to a pinned WAV.

This is a serving-path sanity check. It does not exercise the Omni Talker or
establish a mechanistic claim. Use fit-only stimuli until evaluation is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adapter import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_REVISION,
    load_pinned_thinker,
    prepare_audio_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-prompt", default="Answer the spoken request briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_new_tokens > 256:
        raise ValueError("max-new-tokens must be between 1 and 256")
    model, processor = load_pinned_thinker(
        allow_download=False,
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    batch, digest = prepare_audio_batch(
        processor, args.audio, system_prompt=args.system_prompt
    )
    batch = batch.to(model.device, dtype=model.dtype)
    eos_id = int(processor.tokenizer.eos_token_id)
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    prompt_len = int(batch["input_ids"].shape[1])
    continuation = generated[0, prompt_len:].tolist()
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "audio_sha256": digest,
        "system_prompt": args.system_prompt,
        "prompt_tokens": prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": eos_id,
        "stopped_on_eos": bool(continuation and continuation[-1] == eos_id),
        "new_token_ids": continuation,
        "decoded_special": processor.tokenizer.decode(
            continuation, skip_special_tokens=False
        ),
        "decoded_plain": processor.tokenizer.decode(
            continuation, skip_special_tokens=True
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
