"""CPU Whisper Tiny ASR audit of the 12 frozen fit-only TTS renders.

The output is a candidate intelligibility screen, not a human transcription.
Only processed audio in the supplied fit-only render manifest is decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor


MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"


def normalized_words(text: str) -> list[str]:
    # Same transparent normalization as the pinned Audio Jacobian Lens pilot.
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())


def alignment(reference: list[str], hypothesis: list[str]) -> list[dict]:
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )
    steps = []
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            steps.append({"op": "match" if reference[i - 1] == hypothesis[j - 1] else "substitute", "reference": reference[i - 1], "hypothesis": hypothesis[j - 1]})
            i -= 1
            j -= 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            steps.append({"op": "delete", "reference": reference[i - 1], "hypothesis": None})
            i -= 1
        else:
            steps.append({"op": "insert", "reference": None, "hypothesis": hypothesis[j - 1]})
            j -= 1
    steps.reverse()
    return steps


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_quiet_seconds(samples: np.ndarray, rate: int, threshold: float = 0.001) -> float:
    quiet = np.abs(samples) <= threshold
    longest = current = 0
    for value in quiet:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest / rate


def waveform_checks(path: Path, expected_hash: str) -> tuple[np.ndarray, int, dict]:
    if hash_file(path) != expected_hash:
        raise ValueError(f"Audio hash mismatch: {path}")
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.shape[1] != 1 or rate != 16000:
        raise ValueError(f"Expected 16 kHz mono: {path}")
    samples = samples[:, 0]
    if not np.isfinite(samples).all():
        raise ValueError(f"Nonfinite samples: {path}")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    checks = {
        "sample_rate": rate,
        "frames": len(samples),
        "duration_seconds": len(samples) / rate,
        "peak": peak,
        "rms": rms,
        "dc_offset": float(np.mean(samples.astype(np.float64))),
        "fraction_abs_below_0p001": float(np.mean(np.abs(samples) <= 0.001)),
        "longest_abs_below_0p001_seconds": longest_quiet_seconds(samples, rate),
        "fraction_abs_ge_0p999": float(np.mean(np.abs(samples) >= 0.999)),
        "has_nonfinite": False,
    }
    checks["audio_quality_flags"] = [
        name for name, bad in (
            ("near_silent", rms < 0.003),
            ("hard_clipping", checks["fraction_abs_ge_0p999"] > 0.001),
            ("large_dc_offset", abs(checks["dc_offset"]) > 0.05),
            ("long_quiet_dropout", checks["longest_abs_below_0p001_seconds"] > 2.0),
        ) if bad
    ]
    return samples, rate, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.render_manifest.resolve()
    render = json.loads(manifest_path.read_text())
    if render["selection"]["split"] != "fit" or len(render["records"]) != 12:
        raise ValueError("This audit requires exactly the 12 frozen fit-only smoke records")
    if any(record["source_item"].get("split") != "fit" for record in render["records"]):
        raise ValueError("Non-fit item found in render manifest")

    torch.set_num_threads(4)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).to("cpu").eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Whisper must run entirely on CPU")

    result = {
        "evidence": "single pinned ASR-model screening of fit-only synthetic clips; ASR errors are not proven TTS omissions",
        "render_manifest_path": str(manifest_path),
        "render_manifest_sha256": hash_file(manifest_path),
        "source_input_manifest_sha256": render["input_manifest_sha256"],
        "asr_model_id": MODEL_ID,
        "asr_model_revision": MODEL_REVISION,
        "device": "cpu",
        "torch_threads": 4,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "normalization": "casefold; Unicode letter/number words with internal apostrophes; join normalized words with spaces for CER",
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "attention_mask": True},
        "records": [],
        "aggregate": {},
    }
    totals = defaultdict(lambda: defaultdict(int))
    for record in render["records"]:
        item_id = record["item_id"]
        reference = record["source_item"]["audio_input"]["script"]
        entry = {"item_id": item_id, "reference": reference, "engines": {}}
        for engine in ("piper", "kokoro"):
            output_info = record["outputs"][engine]
            path = manifest_path.parent / output_info["processed"]["path"]
            samples, rate, quality = waveform_checks(path, output_info["processed"]["file_sha256"])
            raw_path = manifest_path.parent / output_info["raw"]["path"]
            if hash_file(raw_path) != output_info["raw"]["file_sha256"]:
                raise ValueError(f"Raw audio hash mismatch: {raw_path}")
            raw_duration = output_info["raw"]["duration_seconds"]
            quality["raw_duration_seconds"] = raw_duration
            quality["resample_duration_delta_seconds"] = quality["duration_seconds"] - raw_duration
            if abs(quality["resample_duration_delta_seconds"]) > 0.01:
                quality["audio_quality_flags"].append("duration_shift_over_10ms")
            if quality["duration_seconds"] > 30:
                raise ValueError(f"Whisper Tiny single-window input exceeds 30 seconds: {path}")
            with torch.inference_mode():
                batch = processor(samples, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)
                ids = model.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=128)
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            reference_words = normalized_words(reference)
            hypothesis_words = normalized_words(transcript)
            operations = alignment(reference_words, hypothesis_words)
            counts = {name: sum(step["op"] == name for step in operations) for name in ("substitute", "delete", "insert")}
            word_errors = sum(counts.values())
            reference_characters = list(" ".join(reference_words))
            hypothesis_characters = list(" ".join(hypothesis_words))
            character_errors = sum(step["op"] != "match" for step in alignment(reference_characters, hypothesis_characters))
            longest_deletion = current = 0
            for step in operations:
                current = current + 1 if step["op"] == "delete" else 0
                longest_deletion = max(longest_deletion, current)
            candidate_omission = word_errors / len(reference_words) >= 0.30 or longest_deletion >= 3
            engine_result = {
                "processed_audio_path": output_info["processed"]["path"],
                "processed_audio_sha256": output_info["processed"]["file_sha256"],
                "raw_audio_sha256": output_info["raw"]["file_sha256"],
                "transcript": transcript,
                "reference_normalized_words": reference_words,
                "transcript_normalized_words": hypothesis_words,
                "word_errors": word_errors,
                "reference_word_count": len(reference_words),
                "wer": word_errors / len(reference_words),
                "character_errors": character_errors,
                "reference_character_count": len(reference_characters),
                "cer": character_errors / len(reference_characters),
                "edit_counts": counts,
                "word_edit_alignment": [step for step in operations if step["op"] != "match"],
                "max_consecutive_deleted_reference_words": longest_deletion,
                "candidate_severe_omission": candidate_omission,
                "quality": quality,
            }
            entry["engines"][engine] = engine_result
            for key, value in (("word_errors", word_errors), ("reference_words", len(reference_words)), ("character_errors", character_errors), ("reference_characters", len(reference_characters)), ("candidate_omission_count", int(candidate_omission)), ("quality_flag_count", len(quality["audio_quality_flags"]))):
                totals[engine][key] += value
            print(f"{item_id} {engine}: WER={engine_result['wer']:.3f}, CER={engine_result['cer']:.3f}, {transcript!r}", flush=True)
        result["records"].append(entry)
    for engine, counts in totals.items():
        result["aggregate"][engine] = {
            **dict(counts),
            "wer": counts["word_errors"] / counts["reference_words"],
            "cer": counts["character_errors"] / counts["reference_characters"],
        }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
