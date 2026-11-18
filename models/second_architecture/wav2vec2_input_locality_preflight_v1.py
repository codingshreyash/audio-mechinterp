"""No-weight, no-inference Wav2Vec2 input-normalization locality check.

Only the already-open corpus-v1 FIT color_05 and object_02 pairs are read.
The arithmetic mirrors the pinned local Wav2Vec2 feature extractor's
unpadded zero-mean/unit-variance normalization; it runs no neural network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
PAIR_IDS = ("color_05", "object_02")
ENGINES = ("piper", "kokoro")
SLOTS = {
    ("color_05", "piper"): (30527, 49847),
    ("color_05", "kokoro"): (33920, 57120),
    ("object_02", "piper"): (30527, 42788),
    ("object_02", "kokoro"): (33920, 54320),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pcm(path: Path, expected_file_sha: str, expected_pcm_sha: str, frames: int) -> np.ndarray:
    if not path.is_file() or sha(path) != expected_file_sha:
        raise ValueError(f"WAV file bytes changed: {path}")
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                stream.getnframes()) != (1, 2, 16000, frames):
            raise ValueError(f"WAV format changed: {path}")
        raw = stream.readframes(frames)
    if hashlib.sha256(raw).hexdigest() != expected_pcm_sha:
        raise ValueError(f"PCM bytes changed: {path}")
    return np.frombuffer(raw, dtype="<i2").copy()


def normalize(values: np.ndarray, mean: np.float32 | None = None,
              variance: np.float32 | None = None) -> np.ndarray:
    """Mirror the installed no-padding processor formula on float32 input."""
    x = values.astype(np.float32) / np.float32(32768.0)
    mu = x.mean() if mean is None else mean
    var = x.var() if variance is None else variance
    return (x - mu) / np.sqrt(var + np.float32(1e-7))


def evaluate() -> dict:
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("FIT manifest bytes changed")
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for pair_id in PAIR_IDS:
        for engine in ENGINES:
            matches = [p for p in manifest["pairs"] if p["pair_id"] == pair_id
                       and p["phrasing_id"] == "fit_chosen" and p["engine"] == engine]
            if len(matches) != 1:
                raise ValueError(f"FIT pair coverage changed: {pair_id}/{engine}")
            pair = matches[0]
            if pair["split"] != "fit_pool" or pair["map_fit"] is not True:
                raise ValueError("non-FIT pair")
            slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
            if slot != SLOTS[(pair_id, engine)]:
                raise ValueError("frozen sample slot changed")
            words = pair["words"]
            if len(words) != 2:
                raise ValueError("expected two contrast words")
            tracks = []
            for word in words:
                output = pair["outputs"][word]
                tracks.append(pcm(ROOT / output["path"], output["file_sha256"],
                                  output["pcm_int16_sha256"], output["frames"]))
            a, b = tracks
            start, end = slot
            if len(a) != len(b) or not np.array_equal(a[:start], b[:start]) \
                    or not np.array_equal(a[end:], b[end:]) \
                    or np.array_equal(a[start:end], b[start:end]):
                raise ValueError("exact-PCM word-slot contrast changed")
            outside = np.r_[np.arange(start), np.arange(end, len(a))]
            stock_a, stock_b = normalize(a), normalize(b)
            shared = a[outside].astype(np.float32) / np.float32(32768.0)
            common_a = normalize(a, shared.mean(), shared.var())
            common_b = normalize(b, shared.mean(), shared.var())
            stock_off = float(np.max(np.abs(stock_a[outside] - stock_b[outside])))
            common_off = float(np.max(np.abs(common_a[outside] - common_b[outside])))
            if common_off != 0.0 or stock_off <= 0.0:
                raise ValueError("normalization locality invariant changed")
            # Seven configured convolutions have stride 320 and raw RF 400.
            ctc_window = (math.ceil((start - 399) / 320),
                          math.floor((end - 1) / 320) + 1)
            conv_frames = (len(a) - 400) // 320 + 1
            rows.append({
                "pair_id": pair_id, "engine": engine, "words": words,
                "frames": len(a), "slot_samples": list(slot),
                "raw_outside_max_abs": 0.0,
                "stock_normalized_outside_max_abs": stock_off,
                "common_normalized_outside_max_abs": common_off,
                "ctc_conv_frames": conv_frames,
                "nominal_raw_receptive_field_intersection": list(ctc_window),
            })
    return {"schema": "wav2vec2_input_locality_preflight_v1",
            "evidence": "no weights loaded; no model inference",
            "manifest_sha256": MANIFEST_SHA, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output already exists")
    args.output.write_text(json.dumps(evaluate(), indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
