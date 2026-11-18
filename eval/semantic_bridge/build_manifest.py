"""Build a frozen 60-item audio/text semantic-bridge evaluation manifest.

Only the Python standard library is used. This script never loads or fits a
model and never renders audio. Re-running checks existing artifacts bytewise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
SOURCE_SHA256 = {
    "association": "d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c",
    "multihop": "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
}
SOURCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "reproductions/anthropic_jlens/jacobian-lens/data/evaluations"
)
OUT = Path(__file__).resolve().parent
NON_ENGLISH_NAME_PREFIXES = ("fr-", "es-", "de-", "pt-")
SPLIT_CAPACITY = {"fit": 18, "dev": 6, "test": 6}  # per source category


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def lines(items: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for item in items
    )


def sortkey(name: str) -> str:
    return digest(name.encode("utf-8"))


def groups_for(category: str, selected: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in selected:
        # Multihop has several formulaic families (nhop, rhyme, chem, ...).
        # Keep a family out of the other splits to avoid obvious template reuse.
        group = item["name"].split("-", 1)[0] if category == "multihop" else item["name"]
        groups[group].append(item)
    return sorted(groups.items(), key=lambda entry: (sortkey(entry[0]), entry[0]))


def assign_groups(groups: list[tuple[str, list[dict]]]) -> dict[str, str]:
    """Deterministic exact-capacity backtracking, holding groups intact."""
    capacities = SPLIT_CAPACITY.copy()
    assignment: dict[str, str] = {}
    # Large groups first reduces the search; hash breaks size ties.
    ordered = sorted(groups, key=lambda e: (-len(e[1]), sortkey(e[0])))

    def search(index: int) -> bool:
        if index == len(ordered):
            return all(n == 0 for n in capacities.values())
        group, members = ordered[index]
        n = len(members)
        options = sorted(
            capacities,
            key=lambda split: (sortkey(group + ":" + split), split),
        )
        for split in options:
            if capacities[split] < n:
                continue
            capacities[split] -= n
            assignment[group] = split
            if search(index + 1):
                return True
            capacities[split] += n
            del assignment[group]
        return False

    if not search(0):
        raise ValueError("Could not make disjoint group split at exact capacities")
    return assignment


def source_items(category: str, source_root: Path) -> tuple[list[dict], int]:
    path = source_root / f"lens-eval-{category}.json"
    raw = path.read_bytes()
    if digest(raw) != SOURCE_SHA256[category]:
        raise ValueError(f"Source hash mismatch: {path}")
    original = json.loads(raw)["items"]
    if category == "association":
        pool = [x for x in original if not x["name"].startswith(NON_ENGLISH_NAME_PREFIXES)]
    else:
        pool = original
    if len(pool) < 30:
        raise ValueError(f"Fewer than 30 eligible {category} items")
    if len({x["name"] for x in pool}) != len(pool):
        raise ValueError(f"Duplicate {category} names")
    selected = sorted(pool, key=lambda item: (sortkey(item["name"]), item["name"]))[:30]
    return selected, len(original) - len(pool)


def build(source_root: Path) -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    exclusion_counts = {}
    assignments = {}
    for category in ("association", "multihop"):
        selected, excluded = source_items(category, source_root)
        exclusion_counts[category] = excluded
        group_assignment = assign_groups(groups_for(category, selected))
        assignments[category] = group_assignment
        for item in selected:
            name = item["name"]
            group = name.split("-", 1)[0] if category == "multihop" else name
            prompt = item["prompt"]
            if not isinstance(prompt, str) or not prompt or not item.get("intermediates"):
                raise ValueError(f"Malformed source item: {category}/{name}")
            rows.append(
                {
                    "id": f"{category}:{name}",
                    "category": category,
                    "source_name": name,
                    "source_sha256": SOURCE_SHA256[category],
                    "split": group_assignment[group],
                    "template_group": f"{category}:{group}",
                    "typed_input": {"format": "raw_completion", "prompt": prompt},
                    "audio_input": {
                        "format": "spoken_script_for_future_render",
                        "script": prompt.rstrip(),
                        "rendered_audio_sha256": None,
                    },
                    "expected_bridge_labels": item["intermediates"],
                    "expected_next_text": item.get("target"),
                    "readout_position": "final_prompt_token",
                    "source_prompt_sha256": digest(prompt.encode("utf-8")),
                }
            )

    order = {"fit": 0, "dev": 1, "test": 2}
    rows.sort(key=lambda x: (order[x["split"]], x["category"], sortkey(x["source_name"])))
    ids = [x["id"] for x in rows]
    if len(rows) != 60 or len(set(ids)) != 60:
        raise ValueError("Expected 60 unique manifest items")
    counts = Counter((x["category"], x["split"]) for x in rows)
    for category in ("association", "multihop"):
        for split, count in SPLIT_CAPACITY.items():
            if counts[(category, split)] != count:
                raise ValueError(f"Wrong {category}/{split} count")
    # Smoke uses only fit items, six per category, ordered by the same hash.
    smoke = []
    for category in ("association", "multihop"):
        eligible = [x for x in rows if x["category"] == category and x["split"] == "fit"]
        smoke.extend(sorted(eligible, key=lambda x: sortkey(x["source_name"]))[:6])
    smoke.sort(key=lambda x: (x["category"], sortkey(x["source_name"])))

    metadata = {
        "schema_version": 1,
        "source_repository": "https://github.com/anthropics/jacobian-lens",
        "source_commit": SOURCE_COMMIT,
        "source_file_sha256": SOURCE_SHA256,
        "source_excluded_non_english_counts": exclusion_counts,
        "selection": "For each category: exclude association names prefixed fr-/es-/de-/pt-; sort SHA256(UTF-8 source name), take first 30.",
        "split": "Exact 18 fit / 6 dev / 6 test per category. Multihop name-prefix template groups kept intact; deterministic SHA-based backtracking. Association items grouped individually.",
        "smoke": "Six fit items per category, sorted by SHA256(source name); no dev/test item in smoke.",
        "source_language": "English pilot; language exclusion uses explicit source name prefixes, not automated language identification.",
        "audio_status": "scripts only, no waveforms rendered or scored",
        "counts": {category: {split: counts[(category, split)] for split in SPLIT_CAPACITY}
                   for category in ("association", "multihop")},
        "item_ids_by_split": {split: [x["id"] for x in rows if x["split"] == split]
                              for split in SPLIT_CAPACITY},
        "smoke_ids": [x["id"] for x in smoke],
        "template_assignment": assignments,
        "manifest_sha256": digest(lines(rows)),
        "smoke_sha256": digest(lines(smoke)),
    }
    return rows, smoke, metadata


def write_or_check(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen artifact differs: {path}; review before versioning")
    else:
        path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    rows, smoke, metadata = build(args.source_root)
    OUT.mkdir(parents=True, exist_ok=True)
    write_or_check(OUT / "manifest.jsonl", lines(rows))
    write_or_check(OUT / "smoke.jsonl", lines(smoke))
    write_or_check(OUT / "metadata.json", canonical(metadata))
    print(f"manifest={len(rows)} smoke={len(smoke)} sha256={metadata['manifest_sha256']}")
    print(json.dumps(metadata["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
