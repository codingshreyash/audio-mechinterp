#!/usr/bin/env python3
"""Build a single-file, offline viewer from the independently reviewed V2 evidence.

This program reads saved JSON and WAV bytes only. It never loads model weights or
re-runs inference. Any changed evidence hash is a hard error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
EXPECTED = {
    "render_20261119_0740Z/manifest.json": "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90",
    "descriptive_whisper_multilingual_20261119_v2/records.jsonl": "588165986eadcfc7df04f6a8869de78ccd656f5b9be578fe41b3e1395abeec42",
    "descriptive_whisper_english_20261119_v2/records.jsonl": "4fd1709d85ebe13ed6a46a7996491fbb227463df6592537435e81c4af91ee923",
    "descriptive_wav2vec2_20261119_v2/records.jsonl": "96880742d7762b1ec4517fd5f869ec37b39f6d9195e6c590e93825219cf060ff",
}
MODELS = {
    "multilingual": "descriptive_whisper_multilingual_20261119_v2",
    "english": "descriptive_whisper_english_20261119_v2",
    "ctc": "descriptive_wav2vec2_20261119_v2",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_case(row: dict, model: str) -> dict:
    out = row["output"]
    geom = row["input_geometry"]
    common = {
        "transcript": out["transcript"],
        "lexical_nonempty": out["lexical_nonempty"],
        "termination": out["termination"],
        "unicode_scripts": out["unicode_scripts"],
        "pcm_frames": geom["pcm_frames"],
    }
    if model == "ctc":
        common.update(
            ctc_frames=geom["ctc_frames"],
            active_ctc_frames=geom["active_ctc_frames"],
            active_input_samples=geom["active_input_samples"],
            attention_mask_kind=geom["attention_mask_kind"],
            blank_id=out["blank_id"],
            collapsed_nonblank_ids=out["collapsed_nonblank_ids"],
            frame_count=len(out["frame_ids"]),
            logits_sha256=out["logits_sha256"],
        )
    else:
        first = out["score_rows"][0]
        common.update(
            active_mel_frames=geom["active_mel_frames"],
            mel_frames=geom["mel_frames"],
            encoder_frames=geom["encoder_frames"],
            nominal_active_encoder_rows=geom["nominal_active_encoder_rows"],
            mask_sha256=geom["mask_sha256"],
            language_prefix=("<|en|>" if model == "multilingual" else None),
            language_detection=(
                out["language_detection"]["chosen_language_ids"]
                if model == "multilingual" else None
            ),
            prefix_ids=out["prefix_ids"],
            generated_ids=out["generated_ids"],
            first_raw_top_id=first["raw_top_id"],
            first_processed_top_id=first["top_id"],
            first_selected_id=first["token_id"],
            first_raw_logits_sha256=first["raw_logits_sha256"],
            first_processed_logits_sha256=first["processed_logits_sha256"],
            eos_seen=out["eos_seen"],
            censored=out["censored"],
            actual_generation_steps=out["actual_generation_steps"],
        )
    return common


def build_data() -> dict:
    for rel, digest in EXPECTED.items():
        path = RUNS / rel
        actual = sha(path)
        if actual != digest:
            raise ValueError(f"Evidence hash mismatch: {rel}: {actual} != {digest}")

    manifest = json.loads((RUNS / "render_20261119_0740Z/manifest.json").read_text())
    manifest_rows = manifest["rows"]
    assert len(manifest_rows) == 20
    assets = {}
    for row in manifest_rows:
        case_id = row["case_id"]
        assert case_id not in assets
        path = RUNS / "render_20261119_0740Z" / row["wav"]
        assert sha(path) == row["wav_sha256"]
        with wave.open(str(path), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (
                1, 2, 16000, row["frames"]
            )
        assets[case_id] = {
            "wav_sha256": row["wav_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "duration_s": row["frames"] / 16000,
            "rms": row["rms"],
            "peak": row["peak"],
            "audio": "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii"),
        }

    cases = {key: {"asset": assets[key]} for key in sorted(assets)}
    for model, run in MODELS.items():
        seen = set()
        repeats = []
        difference_count = 0
        types = []
        with (RUNS / run / "records.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                types.append(row["type"])
                if row["type"] == "case":
                    case_id = row["case_id"]
                    assert case_id in cases and row["status"] == "complete"
                    assert row["wav_sha256"] == assets[case_id]["wav_sha256"]
                    assert row["pcm_sha256"] == assets[case_id]["pcm_sha256"]
                    if row["repeat"]:
                        repeats.append(row)
                    else:
                        assert case_id not in seen
                        seen.add(case_id)
                        cases[case_id][model] = compact_case(row, model)
                elif row["type"] == "silence_difference":
                    difference_count += 1
        assert len(types) == 42 and len(seen) == 20 and len(repeats) == 2
        assert difference_count == 18
        assert {x["case_id"] for x in repeats} == {"d02_silence", "d08_silence"}
        for row in repeats:
            assert compact_case(row, model) == cases[row["case_id"]][model]

    summary = {}
    for model in MODELS:
        nonempty = sum(bool(row[model]["lexical_nonempty"]) for row in cases.values())
        different = sum(
            row[model]["transcript"] != cases[row_id[:3] + "_silence"][model]["transcript"]
            for row_id, row in cases.items() if not row_id.endswith("_silence")
        )
        summary[model] = {"nonempty": nonempty, "different_from_silence": different}
    assert summary == {
        "multilingual": {"nonempty": 20, "different_from_silence": 4},
        "english": {"nonempty": 20, "different_from_silence": 0},
        "ctc": {"nonempty": 3, "different_from_silence": 3},
    }
    assert all(cases[x]["multilingual"]["language_prefix"] == "<|en|>" for x in cases)
    for case in cases.values():
        for model, eos_id in (("multilingual", 50257), ("english", 50256)):
            out = case[model]
            assert out["termination"] == "eos" and out["eos_seen"] and not out["censored"]
            assert out["generated_ids"][-1] == eos_id
            assert out["first_processed_top_id"] == out["first_selected_id"] == out["generated_ids"][0]
        assert case["ctc"]["termination"] == "complete_ctc_frames"
        assert case["ctc"]["ctc_frames"] == case["ctc"]["frame_count"]
    return {"schema": "nonspeech_descriptive_viewer_v1", "hashes": EXPECTED,
            "summary": summary, "cases": cases}


def render(data: dict) -> str:
    template = (HERE / "descriptive_v1_template.html").read_text()
    assert template.count("__EMBEDDED_EVIDENCE_JSON__") == 1
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return template.replace("__EMBEDDED_EVIDENCE_JSON__", payload.replace("<", "\\u003c"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="Fail if checked-in HTML is stale")
    args = p.parse_args()
    output = HERE / "descriptive_v1.html"
    built = render(build_data())
    if args.check:
        assert output.read_text() == built, "Checked-in HTML differs from deterministic build"
        print("PASS: pinned evidence and deterministic HTML")
    else:
        output.write_text(built)
        print(output)


if __name__ == "__main__":
    main()
