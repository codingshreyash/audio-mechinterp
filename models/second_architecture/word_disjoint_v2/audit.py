"""Independently audit all 32 word-disjoint WAVs from saved files only.

No TTS, ASR, or assistant model is loaded. The audit reconstructs final PCM
from the saved converted fragments, not from the renderer's assembly function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "rendered_v2"
PAIRS = (
    ("banana", "tomato"), ("carrot", "cookie"), ("train", "plane"),
    ("ferry", "taxi"), ("guitar", "piano"), ("violin", "trumpet"),
    ("broom", "brush"), ("button", "ribbon"),
)
ENGINES = ("piper", "samantha")
GAP = bytes(3840)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(),
                stream.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"bad PCM16 mono WAV: {path}")
        data = stream.readframes(stream.getnframes())
    if not data or len(data) % 2:
        raise ValueError(f"bad PCM data: {path}")
    return data


def resolved_output_path(path: str, output: Path) -> Path:
    target = (ROOT / path).resolve()
    if not target.is_relative_to(output.resolve()) or not target.is_file():
        raise ValueError(f"manifest path missing or outside output: {path}")
    return target


def verify(manifest: dict, output: Path) -> dict:
    if (manifest.get("status") != "synthetic_stimuli_only_no_candidate_model_answers"
            or manifest.get("matched_pairs") != 16 or manifest.get("wavs") != 32
            or manifest.get("sample_rate") != 16000
            or manifest.get("subtype") != "PCM_16"
            or manifest.get("gap_samples") != 1920
            or manifest.get("prefix") != "I said"
            or manifest.get("suffix") != "and then stopped."):
        raise ValueError("manifest population or template changed")
    fragments = manifest.get("fragments", [])
    if len(fragments) != 36:
        raise ValueError("fragment population incomplete")
    index = {}
    seen_paths = set()
    for record in fragments:
        engine, fragment_id = record["engine"], record["fragment_id"]
        if engine not in ENGINES or (engine, fragment_id) in index:
            raise ValueError("fragment engine/identity duplicate")
        if record["text_sha256"] != hashlib.sha256(record["text"].encode()).hexdigest():
            raise ValueError("fragment text hash mismatch")
        for kind in ("raw", "processed"):
            file = record[kind]
            path = resolved_output_path(file["path"], output)
            if path in seen_paths or sha(path) != file["sha256"]:
                raise ValueError("fragment path duplicate or hash mismatch")
            seen_paths.add(path)
        processed = record["processed"]
        data = pcm(resolved_output_path(processed["path"], output))
        if processed["pcm_sha256"] != hashlib.sha256(data).hexdigest() or \
                processed["frames"] != len(data) // 2:
            raise ValueError("processed fragment PCM mismatch")
        index[(engine, fragment_id)] = data
    expected_fragments = {(engine, kind) for engine in ENGINES for kind in ("prefix", "suffix")}
    expected_fragments |= {(engine, f"word_{i:02d}_{word}")
                           for engine in ENGINES for i, pair in enumerate(PAIRS) for word in pair}
    if set(index) != expected_fragments:
        raise ValueError("fragment names differ from frozen pair list")
    for engine in ENGINES:
        if (next(r for r in fragments if r["engine"] == engine and r["fragment_id"] == "prefix")["text"] != "I said"
                or next(r for r in fragments if r["engine"] == engine and r["fragment_id"] == "suffix")["text"] != "and then stopped."):
            raise ValueError("prefix/suffix fragment text mismatch")
        for i, pair in enumerate(PAIRS):
            for word in pair:
                if next(r for r in fragments if r["engine"] == engine and r["fragment_id"] == f"word_{i:02d}_{word}")["text"] != word:
                    raise ValueError("word fragment text mismatch")
    rows = manifest.get("pairs", [])
    if len(rows) != 16:
        raise ValueError("pair population incomplete")
    seen_rows = set()
    clip_paths = set()
    pair_report = []
    for row in rows:
        pair_id, engine = row["pair_id"], row["engine"]
        if engine not in ENGINES or not pair_id.startswith("pair_"):
            raise ValueError("bad pair identity")
        i = int(pair_id[5:])
        if not 0 <= i < len(PAIRS) or pair_id != f"pair_{i:02d}" or \
                (i, engine) in seen_rows or row["words"] != list(PAIRS[i]):
            raise ValueError("pair identity/words mismatch")
        seen_rows.add((i, engine))
        prefix, suffix = index[(engine, "prefix")], index[(engine, "suffix")]
        words = {word: index[(engine, f"word_{i:02d}_{word}")] for word in PAIRS[i]}
        slot_frames = max(len(data) for data in words.values()) // 2
        start = len(prefix) // 2 + 1920
        end = start + slot_frames
        if (row["word_slot_start_sample"] != start
                or row["word_slot_end_sample_exclusive"] != end
                or row["prefix_fragment_id"] != "prefix"
                or row["suffix_fragment_id"] != "suffix"
                or row["word_fragment_ids"] != {word: f"word_{i:02d}_{word}" for word in PAIRS[i]}
                or set(row["outputs"]) != set(PAIRS[i])):
            raise ValueError("pair region/fragment mapping mismatch")
        final = {}
        for word in PAIRS[i]:
            record = row["outputs"][word]
            path = resolved_output_path(record["path"], output)
            if path in seen_paths or path in clip_paths:
                raise ValueError("duplicate clip path")
            clip_paths.add(path)
            actual = pcm(path)
            expected = (prefix + GAP + words[word] +
                        bytes(2 * slot_frames - len(words[word])) + GAP + suffix)
            if (actual != expected or sha(path) != record["file_sha256"]
                    or hashlib.sha256(actual).hexdigest() != record["pcm_int16_sha256"]
                    or record["frames"] != len(actual) // 2
                    or record["word_fragment_frames"] != len(words[word]) // 2
                    or record["spoken_script"] != f"I said {word} and then stopped."):
                raise ValueError("final clip differs from saved fragments or hashes")
            final[word] = actual
        a, b = (final[word] for word in PAIRS[i])
        if (len(a) != len(b) or a[:2*start] != b[:2*start]
                or a[2*end:] != b[2*end:]
                or a[2*start:2*end] == b[2*start:2*end]
                or row["outside_slot_pcm_identical"] is not True
                or row["inside_slot_pcm_differs"] is not True):
            raise ValueError("pair outside-word PCM identity failed")
        pair_report.append({"pair_id": pair_id, "engine": engine,
                            "frames": len(a) // 2, "word_slot": [start, end]})
    if seen_rows != {(i, engine) for i in range(8) for engine in ENGINES} or len(clip_paths) != 32:
        raise ValueError("missing pair/clip")
    return {"status": "cpu_wav_technical_audit_passed_listening_pending",
            "matched_pairs": 16, "wavs": 32,
            "all_outside_word_pcm_equal": True,
            "candidate_model_inference_authorized": False,
            "pairs": pair_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or not output.is_dir():
        raise ValueError("output directory missing or outside repository")
    manifest_path = output / "manifest.json"
    binding_path = output / "TECHNICAL_BINDING.json"
    if binding_path.exists():
        raise FileExistsError("refusing to replace technical binding")
    manifest = json.loads(manifest_path.read_text())
    from . import render
    render.freeze_receipt()
    metadata_hashes = render.verify_population()
    conf = render.piper_configuration()
    if (manifest.get("renderer_sha256") != sha(HERE / "render.py")
            or manifest.get("plan_sha256") != sha(render.PLAN)
            or manifest.get("freeze_receipt_sha256") != sha(render.FREEZE)
            or manifest.get("source_tts_plan_sha256") != sha(render.SOURCE_PLAN)
            or manifest.get("source_tts_helper_sha256") != sha(render.SOURCE_HELPER)
            or manifest.get("metadata_sha256") != metadata_hashes
            or manifest.get("piper_checkpoint_sha256") != conf["model_sha256"]
            or manifest.get("piper_config_sha256") != conf["config_sha256"]
            or manifest.get("piper_package") != conf["package"]
            or manifest.get("piper_synthesis") != conf["synthesis"]):
        raise ValueError("render provenance does not bind reviewed source and metadata")
    report = verify(manifest, output)
    report.update(manifest_sha256=sha(manifest_path), auditor_sha256=sha(Path(__file__)))
    with binding_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"],
                      "binding_sha256": sha(binding_path),
                      "matched_pairs": 16, "wavs": 32}, indent=2))


if __name__ == "__main__":
    main()
