"""Complete-answer positive control for the four fit-only color recordings."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "qwen_omni"))
from adapter import load_pinned_thinker, prepare_audio_batch  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = ROOT / "eval" / "tts" / "outputs" / "controlled_pair_v1"
    manifest_path = base / "controlled_pair_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    eos_id = int(processor.tokenizer.eos_token_id)
    report = {
        "purpose": "fit-only complete-answer positive control, no intervention",
        "model_id": "Qwen/Qwen2.5-Omni-3B",
        "model_revision": "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        "input_manifest_sha256": sha(manifest_path),
        "system_prompt": manifest["system_prompt"],
        "generation": {"do_sample": False, "max_new_tokens": 16,
                       "eos_token_id": eos_id, "repetitions": 2},
        "records": [],
    }
    for engine, group in manifest["engines"].items():
        for word, info in group["outputs"].items():
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError(f"WAV hash mismatch: {path}")
            batch, digest = prepare_audio_batch(
                processor, path, system_prompt=manifest["system_prompt"]
            )
            batch = batch.to(model.device, dtype=model.dtype)
            item = {"engine": engine, "word": word, "audio_sha256": digest,
                    "prompt_tokens": int(batch["input_ids"].shape[1]),
                    "repeats": []}
            for _ in range(2):
                with torch.inference_mode():
                    output = model.generate(
                        **batch, do_sample=False, use_cache=True,
                        max_new_tokens=16, eos_token_id=eos_id,
                        pad_token_id=eos_id,
                    )
                token_ids = output[0, item["prompt_tokens"]:].tolist()
                item["repeats"].append({
                    "token_ids": token_ids,
                    "stopped_on_eos": bool(token_ids and token_ids[-1] == eos_id),
                    "answer": processor.tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    ).strip(),
                })
            report["records"].append(item)
            print(engine, word, [x["answer"] for x in item["repeats"]], flush=True)
    (base / "thinker_answers.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
