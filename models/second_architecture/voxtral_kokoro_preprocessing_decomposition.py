"""Kokoro af_sarah fit-voice replication of Voxtral preprocessing assay.

Default mode is CPU-only. --run explicitly enables the six-condition MPS
generation/teacher-forcing assay from the committed Piper implementation.
No Piper script, protocol, or result is modified by this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch

from voxtral_preprocessing_decomposition import (
    CONDITION_ORDER, feature_frame_l2, feature_sha256,
    independent_raw_log_mel, make_conditions, normalize_raw,
    run_model, source_hashes,
)
from voxtral_representation_spread import (
    CONNECTOR_ROWS, MEL_FRAMES, MEL_HOP, l2_summary,
    nominal_word_connector_rows, nominal_word_mel_frames,
    partition_indices,
)
from voxtral_smoke import (
    HERE, MANIFEST, REVISION, TRANSFORMERS_COMMIT,
    package_versions, prepare_processor, select_clip, sha256_file,
    validate_snapshot,
)


OUT_DIR = HERE / "runs" / "voxtral_kokoro_preprocessing_fit_v1"
ENGINE = "kokoro"
VOICE = "af_sarah"
EXPECTED_MANIFEST_SHA256 = "318807c92971babacad93b69b7d8478e29f397219da0cb9ed6e3134649aea98b"
EXPECTED_TTS_PINS_SHA256 = "a934a5c1a8391d463a0a305e0932a39630049963838832e30a3e5b18f5f73612"
EXPECTED_PIPER_SCRIPT_SHA256 = "026445da1ac2c4c2f4b8f45ab2f8d7a1f4679924774714951b16d5cd29de0788"
EXPECTED_WAV_SHA256 = {
    "red": "b3f09333295ede3cc909f3e1b6f1d84493f3f0d730d7bde8b3ed068ec267cfdb",
    "blue": "fd7864c3bc14c210fc22f49888ee6274e7a0c64e4283d12303d915db2aeeb962",
}
EXPECTED_FEATURE_SHA256 = {
    "word_red_outside_red": "0f7178b63810e1838845a7a38e60c7c7c9030a5a500114590a7b687eea9c5c47",
    "word_blue_outside_blue": "c766d366a5260558bc34ad872f14c781570263c96e34aa268ba4b55c904f10a0",
    "word_red_outside_blue": "fb528d66bd0346cc011a9a538e5309ddf98856d0ee5b65bf42f4f1c2f2be640a",
    "word_blue_outside_red": "bb8ae02fde06297e0b0b2367b2488a8df76e8b217bda6eb688478ed9fd85b524",
    "red_with_blue_padding": "9328a3f16d84b68a210e5bb96b3650ad99d30f5d27ff86bddc23c2aabba36f83",
    "blue_with_red_padding": "f6495044318f726b0b8ba2ab22faa5eebbdc2a0be1fdaa25dddb2d378187ab4b",
}


def prepare_kokoro() -> tuple[dict, object, dict, dict]:
    tts_pins_path = HERE.parent.parent / "eval" / "tts" / "pins.json"
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Controlled-pair manifest differs from frozen protocol")
    if sha256_file(tts_pins_path) != EXPECTED_TTS_PINS_SHA256:
        raise ValueError("TTS pins differ from frozen protocol")
    if sha256_file(HERE / "voxtral_preprocessing_decomposition.py") != EXPECTED_PIPER_SCRIPT_SHA256:
        raise ValueError("Committed Piper decomposition source changed")
    snapshot = validate_snapshot()
    versions = package_versions()
    sources = source_hashes()
    manifest = json.loads(MANIFEST.read_text())
    group = manifest["engines"][ENGINE]
    if not group["prefix_and_suffix_pcm_identical"]:
        raise ValueError("Kokoro manifest does not assert identical outside-word PCM")
    red_path, red_stimulus = select_clip(ENGINE, "red")
    blue_path, blue_stimulus = select_clip(ENGINE, "blue")
    if {"red": red_stimulus["wav_sha256"], "blue": blue_stimulus["wav_sha256"]} != EXPECTED_WAV_SHA256:
        raise ValueError("Kokoro WAV hashes differ from frozen protocol")
    if red_stimulus["frames"] != blue_stimulus["frames"]:
        raise ValueError("Kokoro paired WAVs differ in duration")
    red_pcm, red_rate = sf.read(red_path, dtype="float32")
    blue_pcm, blue_rate = sf.read(blue_path, dtype="float32")
    start = int(group["word_slot_start_sample"])
    end = int(group["word_slot_end_sample_exclusive"])
    if (red_rate, blue_rate) != (16000, 16000) or red_pcm.ndim != 1 or blue_pcm.ndim != 1:
        raise ValueError("Expected 16 kHz mono Kokoro PCM")
    if len(red_pcm) != len(blue_pcm) or len(red_pcm) != red_stimulus["frames"]:
        raise ValueError("Kokoro PCM length differs from manifest")
    if not (0 <= start < end <= len(red_pcm)):
        raise ValueError("Invalid Kokoro word interval")
    outside_equal = bool((red_pcm[:start] == blue_pcm[:start]).all()
                         and (red_pcm[end:] == blue_pcm[end:]).all())
    if not outside_equal:
        raise ValueError("Kokoro PCM differs outside the word interval")

    processor_red, red_batch, red_prep = prepare_processor(red_path)
    _, blue_batch, blue_prep = prepare_processor(blue_path)
    if red_prep["input_ids"] != blue_prep["input_ids"]:
        raise ValueError("Kokoro red/blue prompt token IDs differ")
    if red_batch["input_features"].shape != blue_batch["input_features"].shape:
        raise ValueError("Kokoro red/blue feature shapes differ")
    if red_prep["audio_placeholder_count"] != CONNECTOR_ROWS:
        raise ValueError("Unexpected Kokoro audio placeholder count")
    word_frames = nominal_word_mel_frames(start, end)
    word_rows = nominal_word_connector_rows(start, end)
    feature_padding_from = min(MEL_FRAMES, (len(red_pcm) + MEL_HOP - 1) // MEL_HOP)
    connector_padding_from = min(CONNECTOR_ROWS, (len(red_pcm) + 8 * MEL_HOP - 1) // (8 * MEL_HOP))
    feature_groups = partition_indices(MEL_FRAMES, word_frames, feature_padding_from)
    connector_groups = partition_indices(CONNECTOR_ROWS, word_rows, connector_padding_from)

    raw = {}
    maxima = {}
    errors = {}
    for word, wav, batch in (("red", red_path, red_batch), ("blue", blue_path, blue_batch)):
        raw[word], maxima[word] = independent_raw_log_mel(wav, processor_red.feature_extractor.mel_filters)
        reproduced = normalize_raw(raw[word], maxima[word])
        errors[word] = float((reproduced - batch["input_features"][0]).abs().max())
        if errors[word] > 1e-6:
            raise ValueError(f"Independent Kokoro mel mismatch for {word}: {errors[word]}")
    shared_max = max(maxima.values())
    shared_red = normalize_raw(raw["red"], shared_max).unsqueeze(0)
    shared_blue = normalize_raw(raw["blue"], shared_max).unsqueeze(0)
    raw_l2 = feature_frame_l2(raw["red"].unsqueeze(0), raw["blue"].unsqueeze(0))
    shared_l2 = feature_frame_l2(shared_red, shared_blue)
    word_set = set(word_frames)
    outside = [frame for frame in range(MEL_FRAMES) if frame not in word_set]
    raw_off_max = max(raw_l2[i] for i in outside)
    shared_off_max = max(shared_l2[i] for i in outside)
    if raw_off_max > 1e-6 or shared_off_max > 1e-6:
        raise ValueError(f"Kokoro off-word raw/shared feature mismatch: {raw_off_max}, {shared_off_max}")
    features = {"red": red_batch["input_features"].clone(), "blue": blue_batch["input_features"].clone()}
    conditions = make_conditions(features, word_frames, feature_padding_from)
    feature_hashes = {name: feature_sha256(condition["features"]) for name, condition in conditions.items()}
    if feature_hashes != EXPECTED_FEATURE_SHA256:
        raise ValueError(f"Kokoro condition feature hashes changed: {feature_hashes}")

    report = {
        "schema_version": 1,
        "status": "preflight",
        "phase": "fit-voice replication; no development or held-out audio",
        "assay": "causal_preprocessing_decomposition",
        "engine": ENGINE,
        "voice": VOICE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "piper_decomposition_script_sha256": sha256_file(HERE / "voxtral_preprocessing_decomposition.py"),
        "model_id": "mistralai/Voxtral-Mini-3B-2507",
        "model_revision": REVISION,
        "transformers_release_tag_commit": TRANSFORMERS_COMMIT,
        "installed_transformers_source_sha256": sources,
        "dependency_versions": versions,
        "requirements_freeze_sha256": sha256_file(HERE / "requirements.freeze.txt"),
        "snapshot": snapshot,
        "tts_pins_sha256": EXPECTED_TTS_PINS_SHA256,
        "stimuli": {"red": red_stimulus, "blue": blue_stimulus},
        "processor": {"red": red_prep, "blue": blue_prep},
        "pcm_outside_word_identical": outside_equal,
        "word_interval_samples": [start, end],
        "shape_math": {
            "sample_rate": 16000,
            "mel_hop_samples": MEL_HOP,
            "mel_window_samples": 400,
            "mel_centered": True,
            "mel_frames": MEL_FRAMES,
            "encoder_rows": 1500,
            "connector_rows": CONNECTOR_ROWS,
            "connector_group_encoder_rows": 4,
            "nominal_feature_padding_from": feature_padding_from,
            "nominal_connector_padding_from": connector_padding_from,
            "word_mel_frames": word_frames,
            "word_connector_rows": word_rows,
            "feature_partition": {k: {"count": len(v), "start": min(v) if v else None,
                                      "end_exclusive": max(v) + 1 if v else None}
                                  for k, v in feature_groups.items()},
            "connector_partition": {k: {"count": len(v), "start": min(v) if v else None,
                                        "end_exclusive": max(v) + 1 if v else None}
                                    for k, v in connector_groups.items()},
        },
        "preprocessing_decomposition": {
            "raw_log_mel_maximum": maxima,
            "processor_reproduction_max_abs_error": errors,
            "raw_off_word_max_frame_l2": raw_off_max,
            "raw_frame_l2": raw_l2,
            "shared_raw_log_mel_maximum": shared_max,
            "shared_max_off_word_max_frame_l2": shared_off_max,
            "shared_max_frame_l2": shared_l2,
            "native_frame_l2": feature_frame_l2(features["red"], features["blue"]),
            "shared_max_partition_summary": l2_summary(shared_l2, feature_groups),
            "condition_order": list(CONDITION_ORDER),
            "conditions": {name: {**{k: v for k, v in condition.items() if k != "features"},
                                  "feature_sha256": feature_hashes[name]}
                           for name, condition in conditions.items()},
            "off_manifold_note": "Crossed and padding-swap mel tensors may have discontinuities absent from natural audio.",
        },
    }
    return report, processor_red, red_batch, conditions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="load local weights and run six Kokoro MPS conditions")
    parser.add_argument("--output", type=Path, help="new JSON artifact path; never overwrite")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report, processor, native_batch, conditions = prepare_kokoro()
    if not args.run:
        concise = {
            "status": report["status"],
            "phase": report["phase"],
            "voice": report["voice"],
            "script_sha256": report["script_sha256"],
            "piper_decomposition_script_sha256": report["piper_decomposition_script_sha256"],
            "manifest_sha256": report["stimuli"]["red"]["manifest_sha256"],
            "word_interval_samples": report["word_interval_samples"],
            "shape_math": {k: v for k, v in report["shape_math"].items()
                           if k not in ("word_mel_frames", "word_connector_rows")},
            "word_mel_frame_span": [min(report["shape_math"]["word_mel_frames"]),
                                    max(report["shape_math"]["word_mel_frames"])],
            "word_connector_row_span": [min(report["shape_math"]["word_connector_rows"]),
                                        max(report["shape_math"]["word_connector_rows"])],
            "preprocessing": {k: v for k, v in report["preprocessing_decomposition"].items()
                              if k not in ("raw_frame_l2", "shared_max_frame_l2", "native_frame_l2")},
        }
        print(json.dumps(concise, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or OUT_DIR / f"kokoro_sarah_red_blue_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    try:
        run_model(report, processor, native_batch, conditions)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(f"Kokoro assay failed; evidence saved to {output}", file=sys.stderr)
        raise
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Kokoro assay saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
