"""No-weight, pinned-processor geometry and acoustic feature-spread receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from models.qwen_omni.adapter import MODEL_REVISION, prepare_audio_batch
from models.second_architecture.qwen_action_method_v1.audio_candidate_v1 import validate_audio_batch
from . import calibrate, render

HERE = Path(__file__).resolve().parent
PROMPT = "Listen to the audio. Reply with the single color word spoken, and nothing else."
EXPECTED_CALIBRATED_FRAME_SUPPORT = [118, 159]


def _support(a: np.ndarray, b: np.ndarray) -> list[int]:
    changed = np.flatnonzero(np.any(a != b, axis=(0, 1)))
    return [int(changed[0]), int(changed[-1] + 1)] if len(changed) else [0, 0]


def capture(snapshot: Path) -> dict:
    if snapshot.name != MODEL_REVISION:
        raise ValueError("Pinned processor snapshot revision changed")
    render.verify()
    calibrate.verify()
    from transformers import Qwen2_5OmniProcessor  # No model weights.
    processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True)
    records = {}
    arrays = {}
    for family, paths in (("original", render.WAV), ("calibrated", calibrate.WAV)):
        records[family], arrays[family] = {}, {}
        for word, path in paths.items():
            batch, digest = prepare_audio_batch(processor, path, system_prompt=PROMPT)
            record = validate_audio_batch(batch, processor, wav_sha256=digest,
                                          case_id=f"LEXICAL_{family.upper()}_{word.upper()}")
            records[family][word] = record
            arrays[family][word] = {key: batch[key].detach().cpu().numpy()
                                    for key in ("input_ids", "attention_mask", "input_features",
                                                "feature_attention_mask")}
        red, blue = arrays[family]["red"], arrays[family]["blue"]
        for key in ("input_ids", "attention_mask", "feature_attention_mask"):
            if not np.array_equal(red[key], blue[key]):
                raise ValueError(f"{family} pair {key} differs")
    comparisons = {}
    for family in ("original", "calibrated"):
        a = arrays[family]["red"]["input_features"]
        b = arrays[family]["blue"]["input_features"]
        diff = np.abs(a-b)
        support = _support(a, b)
        changed_columns = int(np.count_nonzero(np.any(a != b, axis=(0, 1))))
        comparisons[family] = {
            "feature_difference_support_exclusive": support,
            "changed_feature_columns": changed_columns,
            "max_abs_feature_difference": float(diff.max()),
            "max_abs_padding_difference": float(diff[:, :, 1600:].max()),
            "red_feature_max": float(a.max()),
            "blue_feature_max": float(b.max()),
            "red_padding_floor": float(a[0, 0, 1600]),
            "blue_padding_floor": float(b[0, 0, 1600]),
        }
    if (comparisons["calibrated"]["feature_difference_support_exclusive"]
            != EXPECTED_CALIBRATED_FRAME_SUPPORT
            or comparisons["calibrated"]["changed_feature_columns"] != 41
            or comparisons["calibrated"]["max_abs_padding_difference"] != 0.0):
        raise ValueError("Calibrated feature localization changed")
    for family in records:
        red, blue = records[family]["red"], records[family]["blue"]
        for key in ("prompt_ids", "batch_shapes", "mel_frames", "encoder_row_count",
                    "projected_row_count", "audio_placeholder_positions"):
            if red[key] != blue[key]:
                raise ValueError(f"{family} processor geometry differs: {key}")
    return {"schema": "qwen_lexical_processor_only_v1", "model_revision": MODEL_REVISION,
            "system_prompt": PROMPT, "records": records, "comparison": comparisons,
            "interpretation": "Original pair has global log-mel floor spread; calibrated pair has exact feature differences in frames [118,159). Encoder attention can still distribute information across its chunk."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = capture(args.snapshot)
    if args.output.exists():
        if json.loads(args.output.read_text()) != result:
            raise ValueError("Existing processor-only receipt differs")
    else:
        with args.output.open("x") as stream:
            json.dump(result, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")


if __name__ == "__main__":
    main()
