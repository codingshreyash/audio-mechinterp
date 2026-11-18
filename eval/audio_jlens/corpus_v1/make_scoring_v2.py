"""Freeze four canonical lexical surfaces from SPEC_V1; tokenizer-only.

V2 is a documented post-FIT adaptation. It never edits the original selection,
single-token aliases, original score result, or any development/test outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
SPEC = HERE / "SPEC_V1.json"
OUTPUT = HERE / "SCORING_V2.json"
SPEC_SHA = "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265"
FIT_CAPABILITY_SHA = "d0e185bcc778f2fc681a243b59417858ca91f4bb04fa3f715adaea64d2a3dd23"
V1_SCORE_SHA = "34128f325b1e08818be0369fb91754cedda6331ac42c07439326cb1b09608c45"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_surfaces(word: str) -> tuple[str, ...]:
    """The same four strings already used to make SPEC_V1's one-token aliases."""
    return (word, word.title(), " " + word, " " + word.title())


def expanded_word(tokenizer, word: str, eos_id: int) -> dict:
    variants = []
    seen_sequences = set()
    for surface in canonical_surfaces(word):
        ids = tuple(tokenizer.encode(surface, add_special_tokens=False))
        if not ids or tokenizer.decode(ids, skip_special_tokens=False).strip().casefold() != word.casefold():
            raise ValueError(f"Canonical surface tokenizes unexpectedly: {word} {surface!r}")
        if ids in seen_sequences:
            continue
        seen_sequences.add(ids)
        variants.append({
            "surface": surface,
            "answer_token_ids": list(ids),
            "answer_plus_eos_ids": [*ids, eos_id],
            "first_token_id": ids[0],
        })
    if not variants:
        raise ValueError(f"No canonical variants: {word}")
    return {
        "variants": variants,
        "first_token_ids": sorted({v["first_token_id"] for v in variants}),
        "complete_answer_sequences_plus_eos": [
            v["answer_plus_eos_ids"] for v in variants
        ],
    }


def build() -> dict:
    if sha(SPEC) != SPEC_SHA:
        raise ValueError("Frozen SPEC_V1 changed")
    spec = json.loads(SPEC.read_text())
    model_id = spec["model_id_for_tokenizer_only"]
    revision = spec["model_revision_for_tokenizer_only"]
    snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    if sha(snapshot / "tokenizer.json") != spec["tokenizer_json_sha256"]:
        raise ValueError("Tokenizer JSON changed")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True, trust_remote_code=True
    )
    eos = spec["tokenizer_eos_id"]
    if tokenizer.eos_token_id != eos:
        raise ValueError("EOS changed")
    words = {word: expanded_word(tokenizer, word, eos) for word in spec["aliases"]}
    for word, item in words.items():
        v1 = {a["token_id"] for a in spec["aliases"][word]}
        if not v1.issubset(item["first_token_ids"]):
            raise ValueError(f"V1 aliases not preserved for {word}")
        item["v1_single_token_ids"] = sorted(v1)
    pairs = []
    for split, entries in spec["word_pairs"].items():
        for pair in entries:
            a, b = pair["words"]
            overlap = set(words[a]["first_token_ids"]) & set(words[b]["first_token_ids"])
            if overlap:
                raise ValueError(f"V2 first-token alias collision in {pair['pair_id']}: {overlap}")
            pairs.append({"split": split, "pair_id": pair["pair_id"],
                          "words": [a, b], "first_token_sets_disjoint": True})
    return {
        "schema_version": 2,
        "status": "post_fit_capability_adaptation_before_any_dev_or_test_candidate_answers",
        "source_spec_sha256": SPEC_SHA,
        "adaptation_informed_by_fit_capability_sha256": FIT_CAPABILITY_SHA,
        "v1_original_score_result_sha256": V1_SCORE_SHA,
        "generator_sha256": sha(Path(__file__)),
        "model_id_for_tokenizer_only": model_id,
        "model_revision_for_tokenizer_only": revision,
        "tokenizer_json_sha256": spec["tokenizer_json_sha256"],
        "eos_token_id": eos,
        "surface_policy": "the tokenizer's deterministic encoding of each of the same four lower/title and optional leading-space strings listed in make_spec.py; deduplicate identical ID sequences; no claim to enumerate all possible segmentations",
        "first_step_policy": "secondary next-token logsumexp over unique first token IDs of canonical variants; paired sets are disjoint but a prefix alone does not imply a complete word",
        "complete_sequence_policy": "secondary logsumexp of teacher-forced log P of every unique canonical answer sequence plus EOS; no length normalization or answer-based variant selection",
        "words": words,
        "pairs": pairs,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUTPUT}, sha256={sha(OUTPUT)}")


if __name__ == "__main__":
    main()
