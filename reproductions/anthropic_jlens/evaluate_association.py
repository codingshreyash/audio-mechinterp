"""Full released association evaluation, with J-lens and raw logit-lens.

Scores the min full-vocabulary rank of each single-token concept surface
across all fitted layers at the final prompt token. No post-hoc filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

import jlens
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inspect_readout import LENS_PATH, MODEL_PATH, ROOT


def candidate_ids(tok, word: str) -> list[int]:
    forms = {word, word.lower(), word.capitalize()}
    ids = set()
    for form in forms:
        for prefix in ("", " "):
            encoded = tok(prefix + form, add_special_tokens=False).input_ids
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    return sorted(ids)


def rank(logits: torch.Tensor, candidates: list[int]) -> int:
    best = logits[candidates].max()
    return int((logits > best).sum().item()) + 1


def main() -> None:
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    assert model.d_model == lens.d_model
    path = ROOT / "jacobian-lens/data/evaluations/lens-eval-association.json"
    items = json.loads(path.read_text())["items"]
    result = {"dataset": str(path), "count": len(items), "layers": lens.source_layers,
              "items": [], "dropped": []}
    for index, item in enumerate(items, 1):
        targets = {word: candidate_ids(tok, word) for word in item["intermediates"]}
        if any(not ids for ids in targets.values()):
            result["dropped"].append({"name": item["name"], "targets": targets})
            continue
        j_logits, final_logits, _ = lens.apply(model, item["prompt"],
                                               layers=lens.source_layers, positions=[-1])
        raw_logits, _, _ = lens.apply(model, item["prompt"],
                                      layers=lens.source_layers, positions=[-1],
                                      use_jacobian=False)
        per_target = {}
        for word, ids in targets.items():
            per_target[word] = {
                "ids": ids,
                "jacobian_min_rank": min(rank(logits[0], ids) for logits in j_logits.values()),
                "raw_min_rank": min(rank(logits[0], ids) for logits in raw_logits.values()),
                "final_rank": rank(final_logits[0], ids),
            }
        result["items"].append({"name": item["name"], "targets": per_target})
        if index % 10 == 0:
            print(f"scored {index}/{len(items)}", flush=True)
    for k in (1, 5, 10, 50, 100):
        result[f"pass@{k}"] = {
            method: sum(
                sum(x[f"{method}_min_rank"] <= k for x in item["targets"].values())
                / len(item["targets"])
                for item in result["items"]
            ) / len(result["items"])
            for method in ("jacobian", "raw")
        }
    out = ROOT / "association_eval.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"scored": len(result["items"]), "dropped": len(result["dropped"]),
                      **{f"pass@{k}": result[f"pass@{k}"] for k in (1, 5, 10, 50, 100)}}, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
