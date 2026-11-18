"""Assemble equal-length red/blue questions from frozen TTS fragments.

For each TTS family the only differing PCM samples in the final pair must lie
inside the padded word slot. This gives the intervention a known acoustic span.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FRAGMENTS = ROOT / "eval" / "tts" / "outputs" / "controlled_fragments_v1" / "render_manifest.json"
OUTPUT = ROOT / "eval" / "tts" / "outputs" / "controlled_pair_v1"
RATE = 16000
GAP_SAMPLES = 1920  # 120 ms; fixed across versions


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(record: dict, engine: str, base: Path) -> tuple[np.ndarray, dict]:
    info = record["outputs"][engine]["processed"]
    path = (base / info["path"]).resolve()
    if sha(path) != info["file_sha256"]:
        raise ValueError(f"Hash mismatch: {path}")
    pcm, rate = sf.read(path, dtype="float32")
    if rate != RATE or pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError(f"Unexpected waveform: {path}")
    return pcm, {"path": str(path.relative_to(ROOT)), "file_sha256": sha(path)}


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    manifest = json.loads(FRAGMENTS.read_text())
    records = {r["item_id"]: r for r in manifest["records"]}
    if set(records) != {"common_prefix", "word_red", "word_blue", "common_suffix"}:
        raise ValueError("Unexpected fragment set")
    report = {
        "schema_version": 1,
        "purpose": "fit-only matched one-word acoustic change",
        "source_fragment_manifest_sha256": sha(FRAGMENTS),
        "source_fragment_input_sha256": manifest["input_manifest_sha256"],
        "sample_rate": RATE,
        "gap_samples": GAP_SAMPLES,
        "system_prompt": "Answer the spoken question with one color word.",
        "engines": {},
    }
    for engine in ("piper", "kokoro"):
        prefix, prefix_info = load(records["common_prefix"], engine, FRAGMENTS.parent)
        suffix, suffix_info = load(records["common_suffix"], engine, FRAGMENTS.parent)
        red, red_info = load(records["word_red"], engine, FRAGMENTS.parent)
        blue, blue_info = load(records["word_blue"], engine, FRAGMENTS.parent)
        slot_len = max(len(red), len(blue))
        gap = np.zeros(GAP_SAMPLES, dtype=np.float32)
        start = len(prefix) + GAP_SAMPLES
        end = start + slot_len
        outputs = {}
        waveforms = {}
        for word, pcm, info in (("red", red, red_info), ("blue", blue, blue_info)):
            slot = np.pad(pcm, (0, slot_len - len(pcm)))
            waveform = np.concatenate((prefix, gap, slot, gap, suffix))
            path = OUTPUT / f"{engine}_{word}.wav"
            sf.write(path, waveform, RATE, subtype="PCM_16", format="WAV")
            decoded, rate = sf.read(path, dtype="float32")
            if rate != RATE or len(decoded) != len(waveform):
                raise ValueError("WAV round trip changed length")
            waveforms[word] = decoded
            outputs[word] = {
                "path": str(path.relative_to(ROOT)),
                "file_sha256": sha(path),
                "frames": len(decoded),
                "duration_seconds": len(decoded) / RATE,
                "word_source": info,
                "word_source_frames": len(pcm),
                "peak": float(np.max(np.abs(decoded))),
                "rms": float(np.sqrt(np.mean(decoded.astype(np.float64) ** 2))),
            }
        outside_equal = bool(
            np.array_equal(waveforms["red"][:start], waveforms["blue"][:start])
            and np.array_equal(waveforms["red"][end:], waveforms["blue"][end:])
        )
        if not outside_equal or len(waveforms["red"]) != len(waveforms["blue"]):
            raise ValueError(f"Uncontrolled acoustic difference for {engine}")
        report["engines"][engine] = {
            "prefix_source": prefix_info,
            "suffix_source": suffix_info,
            "word_slot_start_sample": start,
            "word_slot_end_sample_exclusive": end,
            "word_slot_start_seconds": start / RATE,
            "word_slot_end_seconds": end / RATE,
            "prefix_and_suffix_pcm_identical": outside_equal,
            "outputs": outputs,
        }
    (OUTPUT / "controlled_pair_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({
        engine: {"word_slot_seconds": (
            row["word_slot_start_seconds"], row["word_slot_end_seconds"]
        ), "duration_seconds": row["outputs"]["red"]["duration_seconds"],
        "outside_identical": row["prefix_and_suffix_pcm_identical"]}
        for engine, row in report["engines"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
