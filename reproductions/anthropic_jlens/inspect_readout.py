"""Run the pinned model and Neuronpedia Jacobian lens on released probe-swap prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jlens
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from download_assets import LENS_FILENAME, LENS_REVISION, MODEL_REVISION, ROOT


MODEL_PATH = (
    ROOT / "hf_cache/models--Qwen--Qwen3-1.7B/snapshots" / MODEL_REVISION
)
LENS_PATH = (
    ROOT / "hf_cache/models--neuronpedia--jacobian-lens/snapshots"
    / LENS_REVISION
    / LENS_FILENAME
)
PROBE_ITEMS = (
    ROOT / "jacobian-lens/data/experiments/probe-swap.json"
)


def top_tokens(logits: torch.Tensor, tokenizer, k: int = 10) -> list[dict]:
    values, indices = logits.float().topk(k)
    return [
        {"token_id": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
        for v, i in zip(values, indices, strict=True)
    ]


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    lm = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    assert lm.d_model == lens.d_model
    assert lens.source_layers[-1] < lm.n_layers
    layers = sorted(set(lens.source_layers[:: max(1, len(lens.source_layers) // 8)]))
    items = json.loads(PROBE_ITEMS.read_text())["items"]

    output = {
        "model_revision": MODEL_REVISION,
        "lens_revision": LENS_REVISION,
        "lens_sha256": hashlib.sha256(LENS_PATH.read_bytes()).hexdigest(),
        "reference_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "device": device,
        "dtype": str(dtype),
        "layers": layers,
        "items": [],
    }
    for item in items:
        lens_logits, model_logits, ids = lens.apply(
            lm, item["prompt"], layers=layers, positions=[-1]
        )
        output["items"].append(
            {
                "name": item["name"],
                "prompt": item["prompt"],
                "intermediate": item["intermediate"],
                "answer": item["answer"],
                "swap_to": item["swap_to"],
                "swap_answer": item["swap_answer"],
                "input_ids": ids[0].tolist(),
                "model_top": top_tokens(model_logits[0], tok),
                "lens_top_by_layer": {
                    str(layer): top_tokens(logits[0], tok) for layer, logits in lens_logits.items()
                },
            }
        )
        print(item["name"], "model", output["items"][-1]["model_top"][:3])
        for layer in layers:
            print("  layer", layer, output["items"][-1]["lens_top_by_layer"][str(layer)][:3])

    out = ROOT / "readout_results.json"
    out.write_text(json.dumps(output, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
