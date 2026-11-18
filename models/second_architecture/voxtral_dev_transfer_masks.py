"""Build development-only Voxtral word-time masks without model inference.

The 48 records come from the pinned corpus index. This module deliberately
reads no development model output and no locked-test audio bytes.
"""

from __future__ import annotations

import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

import voxtral_fit_v2_causal_masks as fit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = HERE / "voxtral_corpus_v1_index.json"
PROTOCOL = HERE / "PROSPECTIVE_VOXTRAL_DEV_TRANSFER_V1.md"
FIT_BUILDER = HERE / "voxtral_fit_v2_causal_masks.py"
OUTPUT = HERE / "VOXTRAL_DEV_TRANSFER_MASKS_V1.json"
PIN = {
    "index": "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971",
    "protocol": "779fb190e93ab233f386cb1c33bd01d69fd9dc7e8a5d8c4276549bba55ec64b9",
    "fit_builder": "ab8bdcf0374794c32b904cbdee37cba48c87296cef2970fbc0af896e0d21c2fc",
}
ENGINES = ("piper", "kokoro")
TEMPLATES = ("dev_listen", "dev_hear")
CATEGORIES = ("animal", "color", "nature", "object")
SITES = ("encoder31", "projector")
CLIP_ROOT = (ROOT / "eval/audio_jlens/corpus_v1/rendered/clips").resolve()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources() -> None:
    for name, path in (("index", INDEX), ("protocol", PROTOCOL), ("fit_builder", FIT_BUILDER)):
        if sha(path) != PIN[name]:
            raise ValueError(f"Pinned {name} changed")


def read_wav(clip: dict) -> tuple[bytes, int]:
    path = (ROOT / clip["path"]).resolve()
    if not path.is_relative_to(CLIP_ROOT) or sha(path) != clip["wav_sha256"]:
        raise ValueError("Development WAV path/hash drift")
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(),
                wav.getcomptype(), wav.getnframes()) != (1, 16000, 2, "NONE", clip["frames"]):
            raise ValueError("Development WAV format drift")
        pcm = wav.readframes(wav.getnframes())
    if hashlib.sha256(pcm).hexdigest() != clip["pcm_int16_sha256"]:
        raise ValueError("Development PCM hash drift")
    return pcm, clip["frames"]


def make_manifest(index: dict) -> dict:
    if index["source_manifest_sha256"] != "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f":
        raise ValueError("Wrong source corpus")
    dev = [r for r in index["rows"] if r["split"] == "development"]
    dev.sort(key=lambda r: (r["pair_id"].split("_", 1)[0], r["pair_id"],
                            ENGINES.index(r["engine"]), TEMPLATES.index(r["phrasing_id"])))
    if len(dev) != 48:
        raise ValueError("Expected 48 development records")
    keys = [(r["pair_id"], r["engine"], r["phrasing_id"]) for r in dev]
    if len(set(keys)) != 48:
        raise ValueError("Duplicate development record")
    pairs = sorted({r["pair_id"] for r in dev})
    if len(pairs) != 12 or Counter(p.split("_", 1)[0] for p in pairs) != {c: 3 for c in CATEGORIES}:
        raise ValueError("Wrong development pair/category balance")
    if set(keys) != {(p, e, t) for p in pairs for e in ENGINES for t in TEMPLATES}:
        raise ValueError("Missing pair/engine/template development record")
    words_by_split: dict[str, set[str]] = {s: set() for s in ("fit_pool", "development", "locked_test")}
    for row in index["rows"]:
        words_by_split[row["split"]].update(row["words"])
    if (len(words_by_split["development"]) != 24 or
            any(words_by_split[a] & words_by_split[b]
                for a, b in (("development", "fit_pool"), ("development", "locked_test")))):
        raise ValueError("Development words overlap another split")

    records = []
    for row in dev:
        if row["map_fit"] or row["pilot_fit"] or len(row["words"]) != 2:
            raise ValueError("Wrong development flags or word count")
        start, end = row["word_slot_start_sample"], row["word_slot_end_sample_exclusive"]
        intervals = fit.bounds(start, end)
        clips = []
        pcms = []
        for word in row["words"]:
            source = row["clips"][word]
            pcm, frames = read_wav(source)
            if 2999 * fit.MEL_HOP - fit.MEL_HALF_WINDOW < frames:
                raise ValueError("Mel fill frame 2999 is not padded")
            if intervals["pre"][0] < 0 or intervals["post"][1] > frames:
                raise ValueError("Pre/post interval outside development clip")
            clips.append({"word": word, "wav_path": source["path"],
                          "wav_sha256": source["wav_sha256"],
                          "pcm_int16_sha256": source["pcm_int16_sha256"], "frames": frames})
            pcms.append(pcm)
        if clips[0]["frames"] != clips[1]["frames"]:
            raise ValueError("Matched development clips have unequal length")
        if (pcms[0][:2 * start] != pcms[1][:2 * start] or
                pcms[0][2 * end:] != pcms[1][2 * end:] or
                pcms[0][2 * start:2 * end] == pcms[1][2 * start:2 * end]):
            raise ValueError("Matched development PCM outside word slot changed")
        mel = [i for i in range(3000)
               if i * fit.MEL_HOP - fit.MEL_HALF_WINDOW < end
               and i * fit.MEL_HOP + fit.MEL_HALF_WINDOW > start]
        if not mel or 2999 in mel:
            raise ValueError("Invalid word-overlap mel mask")
        site_maps = {}
        for site in SITES:
            bins = {name: fit.center_rows(lo, hi, site) for name, (lo, hi) in intervals.items()}
            if any(not rows for rows in bins.values()):
                raise ValueError("Empty development source bin")
            all_rows = [j for group in bins.values() for j in group]
            if len(all_rows) != len(set(all_rows)):
                raise ValueError("Overlapping development row bins")
            whole = sorted(j for k in range(4) for j in bins[f"word_{k}"])
            padding = fit.padding_rows(clips[0]["frames"], len(whole), site)
            if set(padding) & set(all_rows):
                raise ValueError("Padding control overlaps word/pre/post")
            site_maps[site] = {"bins": bins, "whole_word_rows": whole,
                               "padding_control_rows": padding}
        records.append({"pair_id": row["pair_id"], "category": row["pair_id"].split("_", 1)[0],
                        "engine": row["engine"], "phrasing_id": row["phrasing_id"],
                        "words": row["words"], "clips": clips,
                        "word_slot_start_sample": start, "word_slot_end_sample_exclusive": end,
                        "intervals_half_open_samples": {k: list(v) for k, v in intervals.items()},
                        "word_mel_frames_direct_overlap": mel,
                        "blank_fill_mel_frame": 2999, "site_maps": site_maps})
    return {"schema_version": "voxtral_dev_transfer_masks_v1",
            "status": "CPU-only development mask proposal; no model inference",
            "source_index_sha256": PIN["index"], "protocol_sha256": PIN["protocol"],
            "fit_mask_builder_sha256": PIN["fit_builder"], "builder_sha256": sha(Path(__file__)),
            "site_tensor_shapes": {"encoder31": [1, 1500, 1280], "projector": [375, 3072]},
            "row_centers_samples": {"encoder31": "320*j", "projector": "160*(8*r+3)"},
            "records": records}


def main() -> None:
    verify_sources()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = make_manifest(json.loads(INDEX.read_text()))
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {OUTPUT}; SHA-256 {sha(OUTPUT)}; records {len(data['records'])}")


if __name__ == "__main__":
    main()
