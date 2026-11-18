"""Freeze exact CPU-only v2 FIT audio-time masks; no model weights or inference."""

from __future__ import annotations

import hashlib
import json
import wave
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = HERE / "voxtral_corpus_v1_index.json"
SELECTION = HERE / "VOXTRAL_FIT_V2_SELECTION.json"
OUTPUT = HERE / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json"
INDEX_SHA256 = "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971"
SELECTION_SHA256 = "f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b"
PAIR_IDS = ("animal_08", "color_05", "nature_09", "object_05")
SAMPLE_RATE = 16000
MEL_HOP = 160
MEL_HALF_WINDOW = 200


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounds(start: int, end: int) -> dict[str, tuple[int, int]]:
    width = end - start
    if width <= 0:
        raise ValueError("Empty word interval")
    edges = [start + (width * k) // 4 for k in range(5)]
    return {"pre": (start - width, start),
            **{f"word_{k}": (edges[k], edges[k + 1]) for k in range(4)},
            "post": (end, end + width)}


def center_rows(lo: int, hi: int, site: str) -> list[int]:
    if site == "encoder31":
        return [j for j in range(1500) if lo <= 320 * j < hi]
    if site == "projector":
        return [r for r in range(375) if lo <= 160 * (8 * r + 3) < hi]
    raise ValueError(site)


def padding_rows(frames: int, n_word: int, site: str) -> list[int]:
    if site == "encoder31":
        choices = [j for j in range(1500) if 320 * j >= frames + MEL_HALF_WINDOW]
    elif site == "projector":
        choices = [r for r in range(375) if 160 * (8 * r + 3) >= frames + MEL_HALF_WINDOW]
    else:
        raise ValueError(site)
    if len(choices) < n_word:
        raise ValueError("Insufficient nominal padding rows")
    return choices[:n_word]


def make_manifest(index: dict, selection: dict) -> dict:
    if index["source_manifest_sha256"] != "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f":
        raise ValueError("Wrong source corpus binding")
    if tuple(selection["selected_pair_ids"]) != PAIR_IDS or len(selection["clips"]) != 32:
        raise ValueError("Wrong v2 selection")
    selected = {(c["pair_id"], c["engine"], c["phrasing_id"], c["word"]): c for c in selection["clips"]}
    if len(selected) != 32:
        raise ValueError("Duplicate v2 clip in selection")
    by_group = defaultdict(list)
    for c in selection["clips"]:
        if c["split"] != "fit_pool" or c["pair_id"] not in PAIR_IDS:
            raise ValueError("Non-fit v2 clip")
        by_group[(c["pair_id"], c["engine"])].append(c)
    if set(by_group) != {(p, e) for p in PAIR_IDS for e in ("piper", "kokoro")}:
        raise ValueError("Wrong pair/voice groups")
    first_template = {}
    for key, clips in by_group.items():
        templates = sorted({c["phrasing_id"] for c in clips})
        if len(templates) != 2 or templates[0] != "fit_chosen":
            raise ValueError(f"Unexpected fit templates for {key}: {templates}")
        first_template[key] = templates[0]
    index_rows = {(r["pair_id"], r["engine"], r["phrasing_id"]): r for r in index["rows"]}
    records = []
    for pair_id in PAIR_IDS:
        for engine in ("piper", "kokoro"):
            phrasing = first_template[(pair_id, engine)]
            row = index_rows[(pair_id, engine, phrasing)]
            if row["split"] != "fit_pool" or not row["map_fit"] or row["pilot_fit"]:
                raise ValueError("Selected a non-fit or old pilot row")
            start, end = row["word_slot_start_sample"], row["word_slot_end_sample_exclusive"]
            intervals = bounds(start, end)
            clips = []
            for word in row["words"]:
                c = selected[(pair_id, engine, phrasing, word)]
                source = row["clips"][word]
                if c["wav_sha256"] != source["wav_sha256"] or c["wav_path"] != source["path"]:
                    raise ValueError("Selection/index WAV binding mismatch")
                path = (ROOT / c["wav_path"]).resolve()
                if not path.is_relative_to((ROOT / "eval/audio_jlens/corpus_v1/rendered/clips").resolve()):
                    raise ValueError("WAV outside frozen FIT corpus")
                if sha(path) != c["wav_sha256"]:
                    raise ValueError("Selected FIT WAV bytes changed")
                with wave.open(str(path), "rb") as w:
                    if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2 or w.getnframes() != source["frames"]:
                        raise ValueError("Selected FIT WAV format changed")
                if 2999 * MEL_HOP - MEL_HALF_WINDOW < source["frames"]:
                    raise ValueError("Mel frame 2999 is not nominally padded")
                if intervals["pre"][0] < 0 or intervals["post"][1] > source["frames"]:
                    raise ValueError("Pre/post interval outside clip")
                clips.append({"word": word, "wav_path": c["wav_path"],
                              "wav_sha256": c["wav_sha256"], "frames": source["frames"]})
            if clips[0]["frames"] != clips[1]["frames"]:
                raise ValueError("Matched FIT clips have different lengths")
            word_mel = [i for i in range(3000)
                        if i * MEL_HOP - MEL_HALF_WINDOW < end
                        and i * MEL_HOP + MEL_HALF_WINDOW > start]
            if not word_mel or 2999 in word_mel:
                raise ValueError("Invalid direct-overlap word mel mask")
            site_maps = {}
            for site in ("encoder31", "projector"):
                bins = {name: center_rows(lo, hi, site) for name, (lo, hi) in intervals.items()}
                if any(not rows for rows in bins.values()):
                    raise ValueError(f"Empty source bin for {pair_id}/{engine}/{site}")
                all_rows = [r for rows in bins.values() for r in rows]
                if len(all_rows) != len(set(all_rows)):
                    raise ValueError("Center-assigned bins overlap")
                whole = sorted(r for k in ("word_0", "word_1", "word_2", "word_3") for r in bins[k])
                pad = padding_rows(clips[0]["frames"], len(whole), site)
                if set(pad) & set(all_rows):
                    raise ValueError("Padding control overlaps six bins")
                site_maps[site] = {"bins": bins, "whole_word_rows": whole,
                                   "padding_control_rows": pad}
            records.append({
                "pair_id": pair_id, "engine": engine, "phrasing_id": phrasing,
                "words": row["words"], "clips": clips,
                "word_slot_start_sample": start, "word_slot_end_sample_exclusive": end,
                "intervals_half_open_samples": {k: list(v) for k, v in intervals.items()},
                "word_mel_frames_direct_overlap": word_mel,
                "blank_fill_mel_frame": 2999,
                "site_maps": site_maps,
            })
    return {
        "schema_version": "voxtral_fit_v2_causal_masks_v1",
        "status": "CPU-only FIT mask freeze, no model inference",
        "source_index_sha256": INDEX_SHA256,
        "v2_selection_sha256": SELECTION_SHA256,
        "builder_sha256": sha(Path(__file__)),
        "site_tensor_shapes": {"encoder31": [1, 1500, 1280], "projector": [375, 3072]},
        "row_centers_samples": {"encoder31": "320*j", "projector": "160*(8*r+3)"},
        "bin_rule": "half-open [lo,hi), pre/post each one full word-slot width; four word-quarter edges start+floor(width*k/4)",
        "records": records,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDEX) != INDEX_SHA256 or sha(SELECTION) != SELECTION_SHA256:
        raise ValueError("Frozen index or selection hash changed")
    data = make_manifest(json.loads(INDEX.read_text()), json.loads(SELECTION.read_text()))
    with OUTPUT.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Wrote {OUTPUT}; SHA-256 {sha(OUTPUT)}; records {len(data['records'])}")


if __name__ == "__main__":
    main()
