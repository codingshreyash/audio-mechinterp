"""Small released probe-swap replication, adapted from J++ causal runner.

This is a gated qualitative reproduction, not the full 90-item published eval.
The same released prompts, source/target concepts, J-lens vectors, clamp
coordinates, and scale grid are used. Models and lens are pinned elsewhere.
"""

from __future__ import annotations

import json

import jlens
import torch
from jlens.hooks import ActivationRecorder
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_association import candidate_ids, rank
from inspect_readout import LENS_PATH, MODEL_PATH, ROOT


NAMES = ("animal-nose-elephant", "colosseum-currency",
         "person-firstname-darwin", "person-firstname-newton")
SCALES = (0.5, 1.0, 2.0, 4.0)
RANDOM_NULL = "pencil"
# A contiguous mid-network band, fixed before this intervention run.
BAND = (12, 13, 14, 15, 16)


def resolve_token(tok, concept):
    for surface in (" " + concept, concept, " " + concept.lower(), concept.lower(),
                    " " + concept.capitalize(), concept.capitalize()):
        ids = tok(surface, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0], surface
    raise ValueError(f"No single-token surface: {concept}")


def clean_forward(model, ids, band):
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[*band, final]) as rec:
        model.forward(ids)
        residuals = {l: rec.activations[l][0].detach().float().cpu() for l in band}
        logits = model.unembed(rec.activations[final][0, -1].detach().float()).float().cpu()
    return logits, residuals


def make_hooks(model, lens, clean, src_id, target_id, scale, band, prompt_len):
    W = model._lm_head.weight.detach().float().cpu()
    handles = []
    diagnostics = []
    for layer in band:
        J = lens.jacobians[layer]
        source = J.T @ W[src_id]
        target = J.T @ W[target_id]
        source /= source.norm()
        target /= target.norm()
        cosine = float(torch.dot(source, target))
        if abs(cosine) > .999:
            diagnostics.append({"layer": layer, "skipped_near_parallel": cosine})
            continue
        V = torch.stack((source, target), dim=1)
        Vpinv = torch.linalg.pinv(V)
        coords = clean[layer][1:prompt_len] @ Vpinv.T
        goal = coords.clone()
        goal[:, 0] = scale * coords[:, 1]
        goal[:, 1] = scale * coords[:, 0]
        diagnostics.append({"layer": layer, "cosine": cosine,
                            "median_clean_coordinate": coords.abs().median().item()})

        def hook(_mod, _inputs, output, V=V, Vpinv=Vpinv, goal=goal, prompt_len=prompt_len):
            # Qwen3 block output is [B,S,D]. We edit original prompt tokens
            # except attention sink; new continuation tokens remain untouched.
            h = output.clone()
            stop = min(prompt_len, h.shape[1])
            if stop <= 1:
                return h
            original = h[:, 1:stop].float()
            basis = V.to(original.device)
            inv = Vpinv.to(original.device)
            desired = goal[:stop - 1].to(original.device)
            current = original @ inv.T
            changed = original + (desired[None] - current) @ basis.T
            h[:, 1:stop] = changed.to(h.dtype)
            return h

        handles.append(model.layers[layer].register_forward_hook(hook))
    return handles, diagnostics


def edited_forward(model, ids, handles):
    try:
        with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
            model.forward(ids)
            return model.unembed(rec.activations[model.n_layers - 1][0, -1].detach().float()).float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def generate(model, tok, lens, clean, ids, src_id, target_id, scale, band, max_new=8):
    sequence = ids.clone()
    out = []
    for _ in range(max_new):
        handles, _ = make_hooks(model, lens, clean, src_id, target_id, scale, band, ids.shape[1])
        logits = edited_forward(model, sequence, handles)
        next_id = int(logits.argmax())
        out.append(next_id)
        sequence = torch.cat((sequence, torch.tensor([[next_id]], device=sequence.device)), 1)
        if next_id == tok.eos_token_id:
            break
    return tok.decode(out, skip_special_tokens=False)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    assert set(BAND) <= set(lens.source_layers)
    all_items = json.loads((ROOT / "jacobian-lens/data/experiments/probe-swap.json").read_text())["items"]
    items = [next(i for i in all_items if i["name"] == name) for name in NAMES]
    results = {"source": "released probe-swap", "subset_selection": "four clean-top1 items from readout probe",
               "band": BAND, "scales": SCALES, "null": RANDOM_NULL, "items": []}
    for item in items:
        ids = model.encode(item["prompt"])
        clean_logits, clean = clean_forward(model, ids, (*BAND, 0))
        answer_ids = candidate_ids(tok, item["answer"])
        new_ids = candidate_ids(tok, item["swap_answer"])
        if not answer_ids or not new_ids:
            raise ValueError(f"unscorable answer: {item['name']}")
        baseline_rank = rank(clean_logits, answer_ids)
        entry = {k: item[k] for k in ("name", "prompt", "intermediate", "answer", "swap_to", "swap_answer")}
        entry["clean_top"] = tok.decode([int(clean_logits.argmax())])
        entry["clean_answer_rank"] = baseline_rank
        entry["clean_swap_answer_rank"] = rank(clean_logits, new_ids)
        clean_id, _ = resolve_token(tok, item["intermediate"])
        # Identical concept directions make the clamp skip every layer,
        # yielding the unedited full greedy continuation through this path.
        entry["clean_continuation"] = generate(model, tok, lens, clean, ids,
                                               clean_id, clean_id, 1.0, BAND)
        entry["trials"] = []
        for variant, src, dest, band in (
            ("main", item["intermediate"], item["swap_to"], BAND),
            ("answer_swap", item["answer"], item["swap_answer"], BAND),
            ("random_null", item["intermediate"], RANDOM_NULL, BAND),
            ("self_swap", item["intermediate"], item["intermediate"], BAND),
            ("early_layer", item["intermediate"], item["swap_to"], (0,)),
        ):
            src_id, src_surface = resolve_token(tok, src)
            dest_id, dest_surface = resolve_token(tok, dest)
            for scale in SCALES:
                handles, diagnostics = make_hooks(model, lens, clean, src_id, dest_id,
                                                  scale, band, ids.shape[1])
                logits = edited_forward(model, ids, handles)
                trial = {"variant": variant, "scale": scale, "band": band,
                         "source_surface": src_surface, "target_surface": dest_surface,
                         "next_token": tok.decode([int(logits.argmax())]),
                         "answer_rank": rank(logits, answer_ids),
                         "swap_answer_rank": rank(logits, new_ids),
                         "diagnostics": diagnostics}
                # Full greedy continuation for both a moderate and strong edit.
                if scale in (1.0, 4.0):
                    trial["continuation"] = generate(model, tok, lens, clean, ids,
                                                     src_id, dest_id, scale, band)
                entry["trials"].append(trial)
        results["items"].append(entry)
        print(item["name"], "clean", entry["clean_top"], "rank", baseline_rank,
              "main", [(x["scale"], x["next_token"], x["swap_answer_rank"])
                       for x in entry["trials"] if x["variant"] == "main"], flush=True)
    out = ROOT / "swap_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
