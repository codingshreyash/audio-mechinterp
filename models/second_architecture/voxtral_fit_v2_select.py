"""CPU-only, fit-informed v2 selection of unopened Voxtral FIT items.

The v1 outcome motivates this version. It never reads development/test WAVs or
candidate-model outputs for unopened items.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = HERE / "voxtral_corpus_v1_index.json"
GATE_V1 = HERE / "VOXTRAL_FIT_STAGE2_GATE_V1.json"
OUTPUT = HERE / "VOXTRAL_FIT_V2_SELECTION.json"
INDEX_SHA256 = "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971"
GATE_V1_SHA256 = "bee8741a9560a2eb228b5d7b106cb9befe1ac9b5913f53f370e752b8b19ba156"
OLD_PILOT_IDS = {"animal_06", "color_03", "nature_00", "object_02"}
CATEGORIES = ("animal", "color", "nature", "object")
# Fixed common homophone families. This is an exclusion list, not a complete
# English homophone detector. Its inclusion of deer/dear is fit-informed.
HOMOPHONE_FAMILIES = {
    "deer": ("dear",), "bear": ("bare",), "hare": ("hair",),
    "sea": ("see",), "sun": ("son",), "rain": ("reign", "rein"),
    "one": ("won",), "two": ("to", "too"), "four": ("for", "fore"),
    "blue": ("blew",), "red": ("read",),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select(index: dict) -> tuple[list[str], list[dict]]:
    if index["source_manifest_sha256"] != "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f":
        raise ValueError("Source manifest binding changed")
    fit_rows = [r for r in index["rows"] if r["split"] == "fit_pool" and r["map_fit"]]
    grouped = defaultdict(list)
    for row in fit_rows:
        grouped[row["pair_id"]].append(row)
    eligible = {}
    for pair_id, rows in grouped.items():
        if pair_id in OLD_PILOT_IDS:
            continue
        words = rows[0]["words"]
        if any(row["words"] != words for row in rows):
            raise ValueError(f"Inconsistent words for {pair_id}")
        if all(word.isascii() and word.isalpha() and len(word) >= 4 and word.casefold() not in HOMOPHONE_FAMILIES
               for word in words):
            eligible[pair_id] = rows
    selected_ids = []
    for category in CATEGORIES:
        candidates = sorted(p for p in eligible if p.startswith(category + "_"))
        if not candidates:
            raise ValueError(f"No unopened eligible fit pair for {category}")
        selected_ids.append(candidates[0])
    selected_rows = [r for pair_id in selected_ids for r in sorted(
        eligible[pair_id], key=lambda r: (r["engine"], r["phrasing_id"]))]
    if len(selected_rows) != 16 or Counter(r["engine"] for r in selected_rows) != {"piper": 8, "kokoro": 8}:
        raise ValueError("Expected 4 pairs × 2 voices × 2 templates")
    if any(r["pair_id"] in OLD_PILOT_IDS or r["split"] != "fit_pool" for r in selected_rows):
        raise ValueError("Selected an opened pilot or non-fit row")
    return selected_ids, selected_rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if sha(INDEX) != INDEX_SHA256 or sha(GATE_V1) != GATE_V1_SHA256:
        raise ValueError("Frozen index or v1 gate report hash changed")
    gate = json.loads(GATE_V1.read_text())
    if gate["status"] != "gate_failed":
        raise ValueError("v2 must document an actual v1 gate failure")
    index = json.loads(INDEX.read_text())
    ids, rows = select(index)
    clips = []
    for row in rows:
        for word in row["words"]:
            clip = row["clips"][word]
            wav = (ROOT / clip["path"]).resolve()
            if not wav.is_relative_to((ROOT / "eval/audio_jlens/corpus_v1/rendered/clips").resolve()):
                raise ValueError("Selected clip path escaped corpus")
            if sha(wav) != clip["wav_sha256"]:
                raise ValueError(f"Selected clip changed: {wav}")
            clips.append({
                "pair_id": row["pair_id"], "split": row["split"], "engine": row["engine"],
                "voice": row["voice"], "phrasing_id": row["phrasing_id"],
                "word": word, "wav_path": clip["path"], "wav_sha256": clip["wav_sha256"],
            })
    if len(clips) != 32 or len({c["wav_path"] for c in clips}) != 32:
        raise ValueError("Expected 32 unique unopened fit clips")
    report = {
        "status": "fit_informed_v2_selection_only_no_model_inference",
        "adaptation_disclosure": "Selection rule was written after v1 deer-to-dear strict-surface failures; v1 remains failed.",
        "source_index_sha256": INDEX_SHA256, "v1_gate_report_sha256": GATE_V1_SHA256,
        "selector_script_sha256": sha(Path(__file__)),
        "selection_rule": "exclude all v1 pilot pair IDs; among remaining map_fit pairs reject non-ASCII, nonalphabetic, <4-letter, or fixed common-homophone-list words; select lexicographically first pair ID per category",
        "homophone_families_excluded": HOMOPHONE_FAMILIES,
        "selected_pair_ids": ids, "planned_clips": len(clips), "clips": clips,
        "candidate_model_inference_run": False,
    }
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Selected {ids}; 32 fit clips; {OUTPUT}; SHA-256 {sha(OUTPUT)}")


if __name__ == "__main__":
    main()
