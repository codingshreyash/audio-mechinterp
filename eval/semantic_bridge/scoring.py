"""Frozen Qwen2.5-Omni semantic-bridge token/rank scoring.

Pure scoring code: no model load, no device access, no fitting. A caller hands
in final-vocabulary logits for each prespecified model layer and modality.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean


K_VALUES = (1, 5, 10, 50, 100)
PRIMARY_LAYERS = (0, 17)
BOOTSTRAP_SEED = 0
BOOTSTRAP_REPS = 10_000


def label_token_surfaces(tokenizer, label: str, head_vocab_size: int) -> list[dict]:
    """Single-token original/lower/title forms, with/without leading space.

    Exact token IDs are returned in the frozen manifest. No synonyms are
    generated; case and spacing variants are not different concepts.
    """
    if not label or not label.strip():
        raise ValueError("empty bridge label")
    special = {int(i) for i in tokenizer.all_special_ids}
    added = {int(i) for i in tokenizer.get_added_vocab().values()}
    accepted = {}
    for form in sorted({label, label.lower(), label.capitalize()}):
        for surface in (form, " " + form):
            ids = tokenizer(surface, add_special_tokens=False).input_ids
            if len(ids) != 1:
                continue
            token_id = int(ids[0])
            if token_id >= head_vocab_size or token_id in special or token_id in added:
                continue
            decoded = tokenizer.decode([token_id])
            if not any(char.isalnum() for char in decoded):
                continue
            accepted[(token_id, surface)] = {"token_id": token_id, "surface": surface}
    return [accepted[k] for k in sorted(accepted)]


def semantic_token_ids(tokenizer, head_vocab_size: int) -> list[int]:
    """Full head vocabulary less special, added, punctuation, blank/padding."""
    special = {int(i) for i in tokenizer.all_special_ids}
    added = {int(i) for i in tokenizer.get_added_vocab().values()}
    decoded = tokenizer.batch_decode([[i] for i in range(head_vocab_size)])
    return [i for i, surface in enumerate(decoded)
            if i not in special and i not in added
            and any(char.isalnum() for char in surface)]


def rank_candidates(
    logits: Sequence[float], candidate_ids: Sequence[int],
    *, allowed_ids: Sequence[int] | None = None,
) -> int:
    """Competition rank 1 + count(score > best candidate score), ties share rank."""
    if not candidate_ids:
        raise ValueError("empty candidate set")
    if not all(math.isfinite(float(logits[i])) for i in candidate_ids):
        raise ValueError("nonfinite candidate logit")
    best = max(float(logits[i]) for i in candidate_ids)
    positions = range(len(logits)) if allowed_ids is None else allowed_ids
    return 1 + sum(math.isfinite(float(logits[i])) and float(logits[i]) > best
                   for i in positions)


def score_item(
    frozen_item: Mapping, logits_by_layer: Mapping[int, Sequence[float]],
    *, modality: str, method: str, head_vocab_size: int,
    audio_source: str | None = None,
    allowed_ids: Sequence[int] | None = None,
    expected_layers: Sequence[int] = PRIMARY_LAYERS,
) -> dict:
    """Per-label/per-layer ranks; caller must pass every predeclared layer."""
    if frozen_item["scoring_status"] != "scorable":
        raise ValueError(f"excluded item {frozen_item['id']}")
    if modality not in ("audio", "typed") or not method:
        raise ValueError("require audio/typed modality and a readout method")
    if (modality == "audio" and not audio_source) or (modality == "typed" and audio_source is not None):
        raise ValueError("audio requires an audio_source; typed requires None")
    if not logits_by_layer:
        raise ValueError("no layers")
    layers = sorted(logits_by_layer)
    if layers != sorted(expected_layers):
        raise ValueError(f"got layers {layers}, frozen layer set is {sorted(expected_layers)}")
    # Convert a future torch tensor to CPU scalars once per layer, avoiding
    # one device synchronization per vocabulary competitor during ranking.
    scalar_logits = {
        layer: logits.detach().float().cpu().tolist()
        if hasattr(logits, "detach") else logits
        for layer, logits in logits_by_layer.items()
    }
    for layer, logits in scalar_logits.items():
        if len(logits) != head_vocab_size:
            raise ValueError(f"layer {layer} has {len(logits)} logits, expected {head_vocab_size}")
    labels = {}
    for label in frozen_item["scorable_labels"]:
        ids = [x["token_id"] for x in label["token_surfaces"]]
        layer_ranks = {
            str(layer): {
                "full_vocab_rank": rank_candidates(scalar_logits[layer], ids),
                "semantic_rank": rank_candidates(scalar_logits[layer], ids,
                                                 allowed_ids=allowed_ids)
                if allowed_ids is not None else None,
            }
            for layer in layers
        }
        labels[label["label"]] = {
            "candidate_ids": sorted(set(ids)),
            "per_layer": layer_ranks,
            "full_min_rank": min(v["full_vocab_rank"] for v in layer_ranks.values()),
            "semantic_min_rank": min(v["semantic_rank"] for v in layer_ranks.values())
            if allowed_ids is not None else None,
        }
    return {"item_id": frozen_item["id"], "category": frozen_item["category"],
            "split": frozen_item["split"], "modality": modality, "method": method,
            "audio_source": audio_source,
            "layers": layers, "labels": labels}


def item_pass(score: Mapping, k: int, metric: str = "full_min_rank") -> float:
    values = [int(label[metric]) for label in score["labels"].values()]
    return sum(rank <= k for rank in values) / len(values)


def category_pass(rows: Sequence[Mapping], k: int,
                  metric: str = "full_min_rank") -> dict[str, float]:
    """Anthropic-style mean per-item fraction, separately by category."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(item_pass(row, k, metric))
    return {category: mean(values) for category, values in grouped.items()}


def aggregate_scores(rows: Sequence[Mapping],
                     metric: str = "full_min_rank") -> dict[tuple[str, str | None, str, str, str], dict]:
    """Pass@k by modality, audio source, method, category, split."""
    grouped: dict[tuple[str, str | None, str, str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        key = (row["modality"], row["audio_source"], row["method"], row["category"], row["split"])
        grouped[key].append(row)
    return {key: {"n_items": len(group),
                  **{f"pass@{k}": mean(item_pass(row, k, metric) for row in group)
                     for k in K_VALUES}}
            for key, group in grouped.items()}


def _bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    if not n:
        raise ValueError("no paired items")
    draws = sorted(mean(values[rng.randrange(n)] for _ in range(n))
                   for _ in range(BOOTSTRAP_REPS))
    return draws[int(.025 * BOOTSTRAP_REPS)], draws[int(.975 * BOOTSTRAP_REPS) - 1]


def paired_audio_typed(
    audio_rows: Sequence[Mapping], typed_rows: Sequence[Mapping],
    *, k: int, metric: str = "full_min_rank",
) -> dict[tuple[str, str], dict]:
    """Paired audio-minus-typed pass@k, clustered at item (not token) level."""
    audio = {(row["item_id"], row["audio_source"]): row for row in audio_rows}
    typed = {row["item_id"]: row for row in typed_rows}
    if len(audio) != len(audio_rows) or len(typed) != len(typed_rows):
        raise ValueError("duplicate item in one modality")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item_id, source in sorted(audio):
        if item_id not in typed:
            continue
        a, t = audio[(item_id, source)], typed[item_id]
        if a["modality"] != "audio" or t["modality"] != "typed":
            raise ValueError(f"wrong paired modalities: {item_id}")
        if (a["category"], a["split"], a["method"], a["layers"], set(a["labels"])) != (
            t["category"], t["split"], t["method"], t["layers"], set(t["labels"])
        ):
            raise ValueError(f"mismatched audio/typed score schema: {item_id}")
        grouped[(a["category"], source)].append(item_pass(a, k, metric) - item_pass(t, k, metric))
    return {category: {"n_items": len(values), "mean_audio_minus_typed": mean(values),
                       "item_bootstrap_95pct": _bootstrap_interval(values)}
            for category, values in grouped.items()}
