"""Build a Voxtral-specific, CPU-only index for the frozen 48-pair corpus.

This reads waveform bytes and the pinned processor/tokenizer only. It never
loads Voxtral model weights or runs Qwen/Voxtral inference.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import AutoProcessor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
OUTPUT = HERE / "voxtral_corpus_v1_index.json"
SNAPSHOT = HERE / "hf_cache/voxtral"
SOURCE_SHA256 = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
CONFIG_SHA256 = "368124c9a9171b3a6c0cdb35cd7fedeff465eefecf315e2a621684bbed7dcd7e"
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
REVISION = "3060fe34b35ba5d44202ce9ff3c097642914f8f3"
SAMPLE_RATE = 16000
HOP = 160
HALF_WINDOW = 200
MEL_FRAMES = 3000
ENCODER_ROWS = 1500
PROJECTOR_ROWS = 375


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def intersecting_mel_rows(start: int, end: int) -> list[int]:
    return [i for i in range(MEL_FRAMES)
            if i * HOP - HALF_WINDOW < end and i * HOP + HALF_WINDOW > start]


def intersecting_encoder_rows(start: int, end: int) -> list[int]:
    # Conv1 k3/s1/p1; Conv2 k3/s2/p1: row j receives mel 2j-2..2j+2.
    return [j for j in range(ENCODER_ROWS)
            if (2 * j - 2) * HOP - HALF_WINDOW < end
            and (2 * j + 2) * HOP + HALF_WINDOW > start]


def intersecting_projector_rows(start: int, end: int) -> list[int]:
    # Four adjacent encoder rows are grouped before the projector.
    return [r for r in range(PROJECTOR_ROWS)
            if (8 * r - 2) * HOP - HALF_WINDOW < end
            and (8 * r + 8) * HOP + HALF_WINDOW > start]


def intervals(start: int, end: int) -> dict[str, tuple[int, int]]:
    width = end - start
    edges = [start + (width * k) // 4 for k in range(5)]
    return {
        "pre": (start - width, start),
        **{f"word_{k}": (edges[k], edges[k + 1]) for k in range(4)},
        "post": (end, end + width),
    }


def candidate_sequences(tokenizer, word: str) -> list[dict]:
    surfaces = (word, word.capitalize(), " " + word, " " + word.capitalize())
    seen = set()
    out = []
    for surface in surfaces:
        ids = tuple(int(x) for x in tokenizer.encode(surface, add_special_tokens=False))
        if not ids or tokenizer.decode(list(ids), skip_special_tokens=False) != surface:
            raise ValueError(f"Candidate tokenization failed for {surface!r}")
        sequence = ids + (int(tokenizer.eos_token_id),)
        if sequence not in seen:
            seen.add(sequence)
            out.append({"surface": surface, "token_ids_with_eos": list(sequence)})
    return out


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if sha(SOURCE) != SOURCE_SHA256 or sha(SNAPSHOT / "config.json") != CONFIG_SHA256:
        raise ValueError("Frozen corpus or Voxtral config hash changed")
    tree = SNAPSHOT / ".cache/huggingface/trees" / f"{REVISION}.json"
    if not tree.is_file():
        raise FileNotFoundError("Pinned Voxtral snapshot revision metadata missing")
    manifest = json.loads(SOURCE.read_text())
    if len(manifest["pairs"]) != 192:
        raise ValueError("Expected 192 matched audio pairs")
    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), local_files_only=True)
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id != 2:
        raise ValueError("Unexpected Voxtral EOS ID")

    pilot_ids = sorted({p["pair_id"] for p in manifest["pairs"] if p["map_fit"]})
    categories = sorted({p.split("_", 1)[0] for p in pilot_ids})
    pilot_ids = [min(p for p in pilot_ids if p.startswith(category + "_")) for category in categories]
    if len(pilot_ids) != 4:
        raise ValueError("Expected one deterministic fit pilot pair per category")

    rows = []
    seen_clips = set()
    for pair in manifest["pairs"]:
        start, end = pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"]
        sample_bins = intervals(start, end)
        clips = {}
        pcm = {}
        for word in pair["words"]:
            info = pair["outputs"][word]
            path = (ROOT / info["path"]).resolve()
            if not path.is_relative_to((SOURCE.parent / "clips").resolve()):
                raise ValueError(f"Clip path outside corpus: {path}")
            if str(path) in seen_clips or sha(path) != info["file_sha256"]:
                raise ValueError(f"Duplicate or changed clip: {path}")
            seen_clips.add(str(path))
            samples, rate = sf.read(path, dtype="int16", always_2d=True)
            if rate != SAMPLE_RATE or samples.shape != (info["frames"], 1):
                raise ValueError(f"Invalid WAV shape/rate: {path}")
            pcm[word] = samples[:, 0]
            if hashlib.sha256(pcm[word].tobytes()).hexdigest() != info["pcm_int16_sha256"]:
                raise ValueError(f"Changed int16 PCM: {path}")
            clips[word] = {
                "path": info["path"], "wav_sha256": info["file_sha256"],
                "pcm_int16_sha256": info["pcm_int16_sha256"],
                "frames": info["frames"], "spoken_script": info["spoken_script"],
                "answer_candidate_sequences": candidate_sequences(tokenizer, word),
            }
        a, b = pair["words"]
        if len(pcm[a]) != len(pcm[b]) or not np.array_equal(pcm[a][:start], pcm[b][:start]) or not np.array_equal(pcm[a][end:], pcm[b][end:]):
            raise ValueError(f"Outside-word PCM mismatch: {pair['pair_id']}")
        if np.array_equal(pcm[a][start:end], pcm[b][start:end]):
            raise ValueError(f"Word slot identical: {pair['pair_id']}")
        if min(s for s, _ in sample_bins.values()) < 0 or max(e for _, e in sample_bins.values()) > len(pcm[a]):
            raise ValueError(f"Pre/post interval exceeds clip: {pair['pair_id']}")
        maps = {}
        for name, (s, e) in sample_bins.items():
            if s >= e:
                raise ValueError(f"Empty interval: {pair['pair_id']}/{name}")
            maps[name] = {
                "sample_start": s, "sample_end_exclusive": e,
                "mel_rows": intersecting_mel_rows(s, e),
                "encoder_rows_direct_receptive": intersecting_encoder_rows(s, e),
                "projector_rows_direct_receptive": intersecting_projector_rows(s, e),
            }
            if not maps[name]["projector_rows_direct_receptive"]:
                raise ValueError(f"No projector row for {pair['pair_id']}/{name}")
        rows.append({
            "pair_id": pair["pair_id"], "split": pair["split"],
            "map_fit": pair["map_fit"], "pilot_fit": pair["split"] == "fit_pool" and pair["pair_id"] in pilot_ids,
            "engine": pair["engine"], "voice": pair["voice"],
            "phrasing_id": pair["phrasing_id"], "words": pair["words"],
            "word_slot_start_sample": start, "word_slot_end_sample_exclusive": end,
            "time_bins": maps, "clips": clips,
        })
    counts = Counter(row["split"] for row in rows)
    if counts != {"fit_pool": 96, "development": 48, "locked_test": 48} or len(seen_clips) != 384:
        raise ValueError(f"Unexpected split/clip count: {counts}, {len(seen_clips)}")
    out = {
        "status": "CPU-only index; no candidate-model inference",
        "source_manifest_sha256": SOURCE_SHA256,
        "builder_sha256": sha(Path(__file__)),
        "model_id_for_processor_only": MODEL_ID, "model_revision": REVISION,
        "model_config_sha256": CONFIG_SHA256,
        "processor_config_sha256": sha(SNAPSHOT / "preprocessor_config.json"),
        "tokenizer_asset_sha256": sha(SNAPSHOT / "tekken.json"),
        "prompt": "Answer the question in the audio with exactly one word. No explanation or punctuation.",
        "max_new_tokens": 32,
        "pilot_selection_rule": "lexicographically first map_fit pair_id in each of the four categories; no ASR/model outcome used",
        "pilot_pair_ids": pilot_ids,
        "counts": {**dict(counts), "clips": len(seen_clips), "pilot_fit_pair_records": sum(r["pilot_fit"] for r in rows)},
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT}; SHA-256 {sha(OUTPUT)}; pilot IDs {pilot_ids}")


if __name__ == "__main__":
    main()
