"""CPU/tokenizer-only audit of the UNFROZEN v2 candidate lexicon.

Reads only prior corpus word-selection metadata. It never opens prior WAVs,
model outputs, activations, scores, or model weights. It does not render audio.
The CMU dictionary must be supplied from the pinned upstream checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROPOSAL = HERE / "CANDIDATES_UNFROZEN.json"
V1_SPEC = ROOT / "eval/audio_jlens/corpus_v1/SPEC_V1.json"
OLD_PLAN = ROOT / "eval/heldout_flow/plan_v1.json"
OLD_TEST_METADATA = ROOT / "eval/audio_jlens/test_stimuli_v1/manifest.json"
MODEL = "Qwen/Qwen2.5-Omni-3B"
REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
CMU_COMMIT = "74790861f652b15e4ac49015a90074ad62a27690"
CMU_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
SOURCE_SHA256 = {
    "v1_spec": "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265",
    "old_plan": "13cea31e2e4843d70a6ab26b060812f43691147435c1e10ea9ebaaa0b24efc78",
    "old_test_metadata": "88bb12c02bc000ffee7dee8329ac3aeddd57d5b44898786f8e6c88d63454f845",
    "tokenizer_json": "8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99",
    "tokenizer_config": "569aa7a9171e36dfff80f0a7550ab0c9c09e46ac840fea5030641417394fb0d2",
}
CATEGORIES = ("animal", "color", "nature", "object")
SPLITS = ("fit_pool", "development", "locked_test")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def historical_words() -> tuple[set[str], dict[str, str]]:
    """Read only frozen selection fields, including old locked word metadata."""
    paths = {"v1_spec": V1_SPEC, "old_plan": OLD_PLAN, "old_test_metadata": OLD_TEST_METADATA}
    for name, path in paths.items():
        require(sha(path) == SOURCE_SHA256[name], f"prior metadata drift: {name}")
    spec = json.loads(V1_SPEC.read_text())
    plan = json.loads(OLD_PLAN.read_text())
    test_meta = json.loads(OLD_TEST_METADATA.read_text())
    require(set(spec["word_pairs"]) == set(SPLITS), "old split schema changed")
    require(len(spec["word_pairs"]["locked_test"]) == 12, "old locked word metadata changed")
    words = {w.casefold() for rows in spec["word_pairs"].values() for pair in rows for w in pair["words"]}
    words |= {w.casefold() for w in spec["historical_excluded_words"]}
    words |= {w.casefold() for pair in plan["word_pairs"] for w in pair["words"]}
    words |= {w.casefold() for pair in test_meta["selection"]["pairs"] for w in pair["words"]}
    require(len(words) == 106, "historical word universe changed")
    return words, {key: sha(path) for key, path in paths.items()}


def inflection_roots(word: str) -> set[str]:
    """A conservative, explicit suffix screen; not a linguistic parser."""
    roots = {word}
    if word.endswith("ies") and len(word) > 4:
        roots.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 4:
        roots.add(word[:-2])
    if word.endswith("s") and len(word) > 3:
        roots.add(word[:-1])
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[: -len(suffix)]
            roots.update((stem, stem + "e"))
            if len(stem) > 2 and stem[-1] == stem[-2]:
                roots.add(stem[:-1])
    return roots


def assign_pairs(spec: dict) -> dict[str, list[dict]]:
    """Stable hash sort, balanced whole-pair split, then independent FIT roles."""
    seed = spec["seed_text"]
    out: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for category in CATEGORIES:
        pairs = spec["pairs_by_category"][category]
        named = [
            {"pair_id": f"v2_{category}_{i:02d}", "category": category, "words": words}
            for i, words in enumerate(pairs, start=1)
        ]
        ordered = sorted(named, key=lambda row: hashlib.sha256(
            f"{seed}/{category}/{row['pair_id']}".encode()
        ).hexdigest())
        for i, row in enumerate(ordered):
            if i < 9:
                out["fit_pool"].append({**row, "fit_role": "basis" if i < 4 else "map"})
            elif i < 12:
                out["development"].append(row)
            else:
                out["locked_test"].append(row)
    return out


def validate_structure(spec: dict, old: set[str]) -> tuple[list[str], dict[str, list[dict]]]:
    require(spec["schema_version"] == 0 and spec["status"] ==
            "unfrozen_cpu_only_candidate_proposal_no_wavs_or_model_answers", "proposal status changed")
    require(set(spec["pairs_by_category"]) == set(CATEGORIES) and
            set(spec["ordered_reserve_pairs_by_category"]) == set(CATEGORIES),
            "category balance changed")
    require(set(spec["templates_by_split"]) == set(SPLITS), "split templates changed")
    words: list[str] = []
    for category in CATEGORIES:
        primary = spec["pairs_by_category"][category]
        reserve = spec["ordered_reserve_pairs_by_category"][category]
        require(len(primary) == 15 and len(reserve) == 3, f"{category} pair/reserve count")
        for pair in (*primary, *reserve):
            require(len(pair) == 2 and pair[0] != pair[1], f"bad pair in {category}")
            for word in pair:
                require(isinstance(word, str) and re.fullmatch(r"[a-z]+", word) is not None,
                        f"nonlowercase/nonalphabetic word: {word!r}")
                words.append(word)
    require(len(words) == 144 and len(set(words)) == 144, "candidate/reserve word reuse")
    require(not set(words) & old, "candidate overlaps prior word metadata")
    old_roots = set().union(*(inflection_roots(w) for w in old))
    new_roots = {word: inflection_roots(word) for word in words}
    near_old = [word for word, roots in new_roots.items() if roots & old_roots]
    require(not near_old, f"candidate close inflection of prior word: {near_old}")
    ids = []
    for split in SPLITS:
        templates = spec["templates_by_split"][split]
        require(len(templates) == 2, f"{split} must have two templates")
        for template in templates:
            require(template["id"].startswith(f"v2_{'fit' if split == 'fit_pool' else 'dev' if split == 'development' else 'test'}_")
                    and template["prefix"].strip() and template["suffix"].strip(),
                    f"bad template in {split}")
            ids.append(template["id"])
    require(len(set(ids)) == 6, "template IDs reused across splits")
    render = spec["render_intent"]
    require(render["voices"] == ["piper", "kokoro"] and render["sample_rate"] == 16000
            and render["channels"] == 1 and render["subtype"] == "PCM_16"
            and render["gap_samples"] == 1920, "rendering intent changed")
    require(render["expected_pairs"] == 60 and render["expected_matched_pair_records"] == 240
            and render["expected_wavs"] == 480, "render workload changed")
    assigned = assign_pairs(spec)
    require({k: len(v) for k, v in assigned.items()} ==
            {"fit_pool": 36, "development": 12, "locked_test": 12}, "split counts")
    require(sum(r.get("fit_role") == "basis" for r in assigned["fit_pool"]) == 16
            and sum(r.get("fit_role") == "map" for r in assigned["fit_pool"]) == 20,
            "FIT roles changed")
    return words, assigned


def pronunciations(path: Path) -> dict[str, set[tuple[str, ...]]]:
    require(sha(path) == CMU_SHA256, "CMU dictionary bytes changed")
    entries: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for line in path.read_text().splitlines():
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        word = re.sub(r"\(\d+\)$", "", parts[0].casefold())
        entries[word].add(tuple(re.sub(r"\d", "", phone) for phone in parts[1:]))
    return entries


def check_phonemes(words: list[str], old: set[str], entries: dict[str, set[tuple[str, ...]]]) -> None:
    missing = sorted((set(words) | old) - set(entries))
    require(not missing, f"CMU pronunciation missing: {missing}")
    reverse: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for word in words:
        for phones in entries[word]:
            reverse[phones].add(word)
    collisions = [sorted(group) for group in reverse.values() if len(group) > 1]
    require(not collisions, f"candidate homophones: {collisions}")
    old_phones = set().union(*(entries[word] for word in old))
    overlap = [word for word in words if entries[word] & old_phones]
    require(not overlap, f"candidate homophones of old word: {overlap}")


def first_token_ids(tokenizer, word: str) -> set[int]:
    ids: set[int] = set()
    for surface in (word, word.title(), " " + word, " " + word.title()):
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        require(bool(token_ids) and tokenizer.decode(token_ids, skip_special_tokens=False).strip().casefold() == word,
                f"tokenizer round trip changed: {word}/{surface}")
        ids.add(int(token_ids[0]))
    return ids


def check_prefixes(spec: dict, tokenizer) -> dict[str, dict[str, list[int]]]:
    result = {}
    for category in CATEGORIES:
        for i, pair in enumerate(spec["pairs_by_category"][category], start=1):
            a, b = pair
            aa, bb = first_token_ids(tokenizer, a), first_token_ids(tokenizer, b)
            require(not aa & bb, f"first-token class collision: {category}_{i:02d} {a}/{b}")
            result[f"v2_{category}_{i:02d}"] = {a: sorted(aa), b: sorted(bb)}
        for i, pair in enumerate(spec["ordered_reserve_pairs_by_category"][category], start=1):
            a, b = pair
            aa, bb = first_token_ids(tokenizer, a), first_token_ids(tokenizer, b)
            require(not aa & bb, f"reserve prefix collision: {category}_{i:02d} {a}/{b}")
    return result


def audit(cmudict_path: Path) -> dict:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    spec = json.loads(PROPOSAL.read_text())
    old, old_hashes = historical_words()
    words, assigned = validate_structure(spec, old)
    entries = pronunciations(cmudict_path)
    check_phonemes(words, old, entries)
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    require(sha(snapshot / "tokenizer.json") == SOURCE_SHA256["tokenizer_json"] and
            sha(snapshot / "tokenizer_config.json") == SOURCE_SHA256["tokenizer_config"],
            "pinned tokenizer bytes changed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                              local_files_only=True, trust_remote_code=True)
    require(tokenizer.eos_token_id == 151645, "Qwen EOS ID changed")
    prefix_ids = check_prefixes(spec, tokenizer)
    return {
        "status": "unfrozen_cpu_candidate_check_passed_no_wavs_or_model_answers",
        "proposal_sha256": sha(PROPOSAL), "validator_sha256": sha(Path(__file__)),
        "prior_metadata_sha256": old_hashes, "prior_word_count": len(old),
        "cmudict_commit": CMU_COMMIT, "cmudict_dict_sha256": sha(cmudict_path),
        "model_for_tokenizer_only": MODEL, "revision": REVISION,
        "tokenizer_json_sha256": SOURCE_SHA256["tokenizer_json"],
        "candidate_pairs": 60, "reserve_pairs": 12, "distinct_candidate_and_reserve_words": 144,
        "split_counts": {split: len(rows) for split, rows in assigned.items()},
        "fit_basis_pairs": 16, "fit_map_pairs": 20,
        "rendered_pair_records_planned": 240, "wavs_planned": 480,
        "assigned_pairs": assigned, "first_token_prefix_ids": prefix_ids,
        "limits": "CMU dictionary is an accent-dependent homophone screen, not a human or ASR intelligibility check; proposal remains unfrozen"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmudict", type=Path, required=True, help="pinned cmudict.dict checkout")
    parser.add_argument("--report-out", type=Path, help="exclusive create; omit for stdout only")
    args = parser.parse_args()
    report = audit(args.cmudict)
    output = json.dumps(report, indent=2) + "\n"
    if args.report_out:
        with args.report_out.open("x") as stream:
            stream.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
