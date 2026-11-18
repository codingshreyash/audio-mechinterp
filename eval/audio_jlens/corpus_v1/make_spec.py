"""CPU/tokenizer-only deterministic corpus selection; never loads model weights."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_PLAN = ROOT / "eval" / "heldout_flow" / "plan_v1.json"
PROPOSAL = ROOT / "eval" / "audio_jlens" / "NEXT_PHASE_CORPUS_LENS_PROPOSAL.md"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
SEED = 20261118
EXCLUDED = frozenset("red blue cat dog monday friday north south seven eight".split())

# Pair neighbors are fixed *before* the seeded split. No Qwen answers or ASR
# outputs were used to select these words.
WORDS = {
    "animal": "bear wolf lion tiger horse sheep goat duck goose mouse rabbit fox deer moose panda camel beaver monkey turtle cow pig chicken snake frog".split(),
    "color": "green yellow orange purple pink brown black white gray gold silver teal cyan violet beige coral olive navy lavender turquoise crimson cream tan lime".split(),
    "object": "chair table spoon fork plate cup bowl knife lamp clock phone book pen pencil window door mirror blanket pillow carpet candle basket hammer ladder".split(),
    "nature": "river lake ocean forest desert mountain valley island beach canyon glacier volcano meadow swamp marsh cliff cave stream waterfall garden grove pond reef rain".split(),
}

TEMPLATES = {
    "fit": [
        {"id": "fit_chosen", "prefix": "The word I chose is", "suffix": "What word did I choose?"},
        {"id": "fit_remember", "prefix": "Please remember this word:", "suffix": "Repeat that word."},
    ],
    "development": [
        {"id": "dev_listen", "prefix": "Listen for my chosen word:", "suffix": "Tell me the chosen word."},
        {"id": "dev_hear", "prefix": "I will say one word now:", "suffix": "Which word did you hear?"},
    ],
    "locked_test": [
        {"id": "test_answer", "prefix": "Here is my answer word:", "suffix": "Reply with only that word."},
        {"id": "test_attention", "prefix": "Pay attention to this word:", "suffix": "What word was spoken?"},
    ],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_pairs() -> dict[str, list[dict]]:
    all_words = [word.casefold() for group in WORDS.values() for word in group]
    if len(all_words) != 96 or len(set(all_words)) != 96 or set(all_words) & EXCLUDED:
        raise ValueError("Corpus has duplicate, excluded, or missing words")
    result = {name: [] for name in ("fit_pool", "development", "locked_test")}
    for category, words in WORDS.items():
        if len(words) != 24:
            raise ValueError(f"Category {category} lacks 24 words")
        pairs = [
            {"pair_id": f"{category}_{i:02d}", "category": category,
             "words": words[2*i:2*i+2]}
            for i in range(12)
        ]
        random.Random(f"{SEED}/{category}").shuffle(pairs)
        result["fit_pool"].extend(pairs[:6])
        result["development"].extend(pairs[6:9])
        result["locked_test"].extend(pairs[9:12])
    # Exactly three per category enter the first 48-example map. The other
    # fit-pool pairs are fit-only diagnostics and never leak into dev/test.
    selected_ids = {pair["pair_id"] for category in WORDS
                    for pair in [p for p in result["fit_pool"] if p["category"] == category][:3]}
    for pair in result["fit_pool"]:
        pair["map_fit"] = pair["pair_id"] in selected_ids
    return result


def single_token_aliases(tokenizer, word: str) -> list[dict]:
    aliases = []
    ids_seen = set()
    for surface in (word, word.title(), " " + word, " " + word.title()):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False).strip().casefold() != word.casefold():
            continue
        if ids[0] not in ids_seen:
            aliases.append({"surface": surface, "token_id": ids[0]})
            ids_seen.add(ids[0])
    if not aliases:
        raise ValueError(f"No exact one-token answer surface for {word}")
    return aliases


def build_spec() -> dict:
    source = json.loads(SOURCE_PLAN.read_text())
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                      local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        trust_remote_code=True,
    )
    selection = select_pairs()
    aliases = {word: single_token_aliases(tokenizer, word)
               for words in WORDS.values() for word in words}
    for split_pairs in selection.values():
        for pair in split_pairs:
            a, b = pair["words"]
            if {x["token_id"] for x in aliases[a]} & {x["token_id"] for x in aliases[b]}:
                raise ValueError(f"Pair aliases overlap: {pair['pair_id']}")
    voices = source["render"]["voices"]
    if set(voices) != {"piper", "kokoro"}:
        raise ValueError("Expected two pinned TTS families")
    return {
        "schema_version": 1,
        "status": "selection_frozen_before_cpu_render_and_candidate_answers",
        "purpose": "new corpus for reusable audio-time transport; stimulus selection only",
        "seed": SEED,
        "selection_rule": "adjacent word pairs in fixed category lists, then seeded category shuffle; first 6 fit, next 3 dev, final 3 locked test; first 3 fit pairs/category map fit",
        "historical_excluded_words": sorted(EXCLUDED),
        "model_id_for_tokenizer_only": MODEL_ID,
        "model_revision_for_tokenizer_only": MODEL_REVISION,
        "tokenizer_json_sha256": sha(snapshot / "tokenizer.json"),
        "tokenizer_config_sha256": sha(snapshot / "tokenizer_config.json"),
        "tokenizer_eos_id": tokenizer.eos_token_id,
        "source_tts_plan_sha256": sha(SOURCE_PLAN),
        "source_tts_render_helper_sha256": source["render"]["tts_render_source_sha256"],
        "proposal_sha256_at_selection": sha(PROPOSAL),
        "spec_generator_sha256": sha(Path(__file__)),
        "system_prompt": "Repeat only the single word explicitly introduced before the final question in the spoken request.",
        "render": {
            "sample_rate": 16000, "channels": 1, "subtype": "PCM_16",
            "gap_samples": source["render"]["gap_samples"],
            "word_slot_padding": source["render"]["word_slot_padding"],
            "voices": voices,
        },
        "word_pairs": selection,
        "templates": TEMPLATES,
        "aliases": aliases,
        "expected": {
            "word_pairs": 48, "unique_words": 96,
            "fit_pool_pairs": 24, "map_fit_pairs": 12,
            "development_pairs": 12, "locked_test_pairs": 12,
            "matched_audio_pairs": 192, "wav_clips": 384,
            "map_fit_recipient_examples": 48,
            "development_directions": 96, "locked_test_directions": 96,
        },
    }


def main() -> None:
    output = HERE / "SPEC_V1.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(build_spec(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}; sha256={sha(output)}")


if __name__ == "__main__":
    main()
