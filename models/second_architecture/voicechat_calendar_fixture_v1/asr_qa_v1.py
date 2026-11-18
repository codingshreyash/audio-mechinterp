"""Prospective, CPU-only ASR QA for 12 frozen VoiceChat calendar WAVs.

Source candidate. --validate-only performs no ASR inference. --run requires
an independent freeze receipt. --compare-only opens completed raw logs after
both recognizers have finished; raw ASR output contains no reference text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unicodedata
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RENDER = HERE / "rendered_v1/20261118T1145Z"
MANIFEST = RENDER / "render_manifest.json"
SCRIPTS = HERE / "scripts.json"
PROTOCOL = HERE / "ASR_QA_PROSPECTIVE_V1.md"
FREEZE = HERE / "ASR_QA_FREEZE_V1.json"
V2 = ROOT / "eval/audio_jlens/corpus_v2"
V2_ASR_SOURCE = V2 / "asr_screen_v2.py"
PINS = V2 / "ASR_MODEL_PINS_PROSPECTIVE_V1.json"
EXPECTED = {
    "manifest": "bfafef903eaff14a444c015d0c38886e3bedde928ef8c5ca9100aac00370ca7c",
    "scripts": "c3416890395c89c03bdbf2bad9e62e0215bf46d6d9a749f8cb043af1cbe2f99d",
    "protocol": "575d94a642ec061eb1c81766bd516ddbd0a1a54501a7d018d7bdb6665d77f2d5",
    "pins": "1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b",
    "v2_asr_source": "6cc4e6eaa3cf93c3cdf1617589fb1d2ad5eb23890f376aafc5124029474ddcb5",
}
CASES = ("A", "B", "C", "D", "A_literal", "B_literal")
VOICES = ("apple_samantha_en_us", "piper_amy_en_us")
MODELS = ("whisper", "wav2vec2")
N = 12
THREADS = 4


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, body: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_v2():
    """Import the exact previously reviewed ASR loader/decoder source."""
    import importlib.util
    require(V2_ASR_SOURCE.is_file() and sha(V2_ASR_SOURCE) == EXPECTED["v2_asr_source"],
            "v2 ASR source changed")
    spec = importlib.util.spec_from_file_location("voicechat_pinned_v2_asr", V2_ASR_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load pinned ASR source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs() -> tuple[list[dict], list[dict]]:
    """Return audio-only records and separate reference map, in fixed order."""
    for key, path in (("manifest", MANIFEST), ("scripts", SCRIPTS),
                      ("protocol", PROTOCOL), ("pins", PINS),
                      ("v2_asr_source", V2_ASR_SOURCE)):
        require(path.is_file() and sha(path) == EXPECTED[key],
                f"frozen {key} hash changed")
    manifest = json.loads(MANIFEST.read_text())
    scripts = json.loads(SCRIPTS.read_text())
    require(manifest.get("status") == "complete_synthetic_render_no_qa"
            and manifest.get("planned") == manifest.get("rendered") == N
            and len(manifest.get("rows", [])) == N,
            "render manifest is incomplete")
    case_rows = scripts.get("cases")
    require(isinstance(case_rows, list)
            and [row["id"] for row in case_rows] == list(CASES),
            "frozen script cases changed")
    script_by_case = {row["id"]: row["script"] for row in case_rows}
    expected_pairs = [(voice, case) for voice in VOICES for case in CASES]
    observed_pairs = [(row.get("voice_id"), row.get("case_id"))
                      for row in manifest["rows"]]
    require(observed_pairs == expected_pairs, "manifest case/voice matrix changed")
    inputs, references, ids = [], [], set()
    for row in manifest["rows"]:
        voice, case = row["voice_id"], row["case_id"]
        require(row.get("script") == script_by_case[case],
                "manifest script differs from frozen reference")
        filename = f"{case}__{voice}.final.wav"
        require(row.get("final_path") == filename, "unexpected final WAV path")
        path = (RENDER / filename).resolve()
        require(path.is_relative_to(RENDER.resolve()) and path.is_file()
                and sha(path) == row.get("final_sha256"),
                "final WAV missing or hash changed")
        with wave.open(str(path), "rb") as reader:
            require((reader.getnchannels(), reader.getsampwidth(),
                     reader.getframerate(), reader.getcomptype())
                    == (1, 2, 16000, "NONE"), "final WAV geometry changed")
            frames = reader.getnframes()
            data = reader.readframes(frames)
        require(80000 < frames < 30 * 16000 and len(data) == frames * 2
                and data[-160000:] == b"\0" * 160000
                and data[-160002:-160000] != b"\0\0",
                "final WAV duration or exact zero-tail boundary changed")
        pcm_sha = hashlib.sha256(data).hexdigest()
        opaque = hashlib.sha256(f"voicechat-asr-v1:{case}:{voice}:{row['final_sha256']}".encode()).hexdigest()[:20]
        require(opaque not in ids, "opaque ID collision")
        ids.add(opaque)
        inputs.append({"opaque_id": opaque, "path": path, "wav_sha256": row["final_sha256"],
                       "pcm_int16_sha256": pcm_sha, "frames": frames})
        references.append({"opaque_id": opaque, "case_id": case, "voice_id": voice,
                           "script": script_by_case[case]})
    require(len(inputs) == len(references) == len(ids) == N, "12-input coverage failed")
    return inputs, references


def check_freeze() -> dict:
    require(FREEZE.is_file(), "independent ASR QA freeze receipt absent")
    receipt = json.loads(FREEZE.read_text())
    require(receipt.get("status") == "independently_reviewed_voicechat_asr_qa_frozen"
            and receipt.get("runner_sha256") == sha(Path(__file__))
            and all(receipt.get(f"{key}_sha256") == value
                    for key, value in EXPECTED.items()),
            "ASR QA freeze does not bind exact reviewed sources")
    return receipt


def load_audio(item: dict):
    import numpy as np
    import soundfile as sf
    samples, rate = sf.read(item["path"], dtype="int16", always_2d=True)
    require(rate == 16000 and samples.shape == (item["frames"], 1),
            "ASR input geometry changed during run")
    pcm = samples[:, 0].astype("<i2", copy=False)
    require(hashlib.sha256(pcm.tobytes()).hexdigest() == item["pcm_int16_sha256"]
            and sha(item["path"]) == item["wav_sha256"],
            "ASR input bytes changed during run")
    return pcm.astype(np.float32) / 32768.0


def verify_raw(output: Path, model_key: str, inputs: list[dict]) -> str:
    raw = output / "raw.jsonl"
    require(raw.is_file(), "raw transcript log absent")
    lines = raw.read_text(encoding="utf-8").splitlines()
    require(len(lines) == N and len(inputs) == N, "raw transcript denominator incomplete")
    allowed_common = {"ordinal", "opaque_id", "wav_sha256", "pcm_int16_sha256",
                      "raw_transcript"}
    allowed_extra = ({"generated_token_ids", "decoder_cap", "ended_with_eos"}
                     if model_key == "whisper" else
                     {"frame_argmax_token_ids", "ctc_frame_count"})
    for ordinal, (line, item) in enumerate(zip(lines, inputs, strict=True), 1):
        row = json.loads(line)
        require(set(row) == allowed_common | allowed_extra
                and row["ordinal"] == ordinal
                and row["opaque_id"] == item["opaque_id"]
                and row["wav_sha256"] == item["wav_sha256"]
                and row["pcm_int16_sha256"] == item["pcm_int16_sha256"]
                and isinstance(row["raw_transcript"], str),
                "raw ASR row identity, transcript, or field set changed")
        key = "generated_token_ids" if model_key == "whisper" else "frame_argmax_token_ids"
        tokens = row[key]
        require(isinstance(tokens, list) and bool(tokens)
                and all(type(t) is int and t >= 0 for t in tokens),
                "raw token IDs malformed")
        require((row["decoder_cap"] == 128 and type(row["ended_with_eos"]) is bool)
                if model_key == "whisper" else row["ctc_frame_count"] == len(tokens),
                "decoding metadata changed")
    return sha(raw)


def run(model_key: str, output: Path, inputs: list[dict], pins: dict,
        snapshots: dict, v2) -> None:
    require(model_key in MODELS and not output.exists()
            and not output.resolve().is_relative_to(RENDER.resolve()),
            "model or output directory invalid")
    check_freeze()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raw = output / "raw.jsonl"
    meta = {"status": "started_raw_asr_only", "model_key": model_key,
            "model_id": pins["models"][model_key]["repo_id"],
            "model_revision": pins["models"][model_key]["revision"],
            "model_files_sha256": pins["models"][model_key]["files_sha256"],
            "runner_sha256": sha(Path(__file__)), "freeze_sha256": sha(FREEZE),
            "manifest_sha256": EXPECTED["manifest"], "pins_sha256": EXPECTED["pins"],
            "planned_clips": N, "device": "cpu", "threads": THREADS,
            "python": sys.version.split()[0], "platform": platform.platform(),
            "scope": "audio-only raw transcripts; no reference comparison or VoiceChat"}
    exclusive_json(output / "meta.json", meta)
    try:
        import torch
        from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                                  Wav2Vec2ForCTC, Wav2Vec2Processor)
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        model_dir = str(snapshots[model_key])
        processor = (WhisperProcessor if model_key == "whisper" else Wav2Vec2Processor).from_pretrained(
            model_dir, local_files_only=True)
        model = (WhisperForConditionalGeneration if model_key == "whisper" else Wav2Vec2ForCTC).from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True).to("cpu").eval()
        require(all(p.device.type == "cpu" for p in model.parameters()),
                "ASR weights not wholly on CPU")
        with raw.open("x", encoding="utf-8") as stream:
            for ordinal, item in enumerate(inputs, 1):
                result = v2.transcribe(model_key, processor, model, load_audio(item))
                row = {"ordinal": ordinal, "opaque_id": item["opaque_id"],
                       "wav_sha256": item["wav_sha256"],
                       "pcm_int16_sha256": item["pcm_int16_sha256"], **result}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        raw_sha = verify_raw(output, model_key, inputs)
        exclusive_json(output / "complete.json", {
            "status": "complete_raw_asr_no_reference_comparison",
            "model_key": model_key, "planned_clips": N, "raw_rows": N,
            "raw_sha256": raw_sha, "meta_sha256": sha(output / "meta.json"),
            "elapsed_seconds": time.monotonic() - started,
            "no_voicechat_inference": True})
    except BaseException as exc:
        exclusive_json(output / "error.json", {
            "status": "incomplete_raw_asr", "model_key": model_key,
            "type": type(exc).__name__, "error": str(exc),
            "raw_rows": raw.read_text().count("\n") if raw.is_file() else 0,
            "raw_sha256": sha(raw) if raw.is_file() else None})
        raise


def normalized_words(value: str) -> list[str]:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(c if c.isalnum() else " " for c in value)
    return value.split()


def word_errors(reference: list[str], observed: list[str]) -> int:
    previous = list(range(len(observed) + 1))
    for i, target in enumerate(reference, 1):
        current = [i]
        for j, token in enumerate(observed, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (target != token)))
        previous = current
    return previous[-1]


def compare_one(output: Path, model_key: str, inputs: list[dict],
                references: list[dict]) -> tuple[list[dict], dict]:
    require(not (output / "error.json").exists(), "ASR run has error receipt")
    complete = json.loads((output / "complete.json").read_text())
    meta = json.loads((output / "meta.json").read_text())
    pinned_model = json.loads(PINS.read_text())["models"][model_key]
    require(complete.get("status") == "complete_raw_asr_no_reference_comparison"
            and complete.get("model_key") == model_key
            and complete.get("planned_clips") == complete.get("raw_rows") == N
            and complete.get("no_voicechat_inference") is True
            and complete.get("raw_sha256") == verify_raw(output, model_key, inputs)
            and complete.get("meta_sha256") == sha(output / "meta.json")
            and meta.get("model_key") == model_key
            and meta.get("model_id") == pinned_model["repo_id"]
            and meta.get("model_revision") == pinned_model["revision"]
            and meta.get("model_files_sha256") == pinned_model["files_sha256"]
            and meta.get("device") == "cpu" and meta.get("threads") == THREADS
            and meta.get("planned_clips") == N
            and meta.get("runner_sha256") == sha(Path(__file__))
            and meta.get("freeze_sha256") == sha(FREEZE)
            and meta.get("manifest_sha256") == EXPECTED["manifest"]
            and meta.get("pins_sha256") == EXPECTED["pins"],
            "ASR completion or metadata provenance failed")
    raw_rows = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
    rows = []
    for ref, raw in zip(references, raw_rows, strict=True):
        require(ref["opaque_id"] == raw["opaque_id"], "ASR reference join changed")
        wanted, heard = normalized_words(ref["script"]), normalized_words(raw["raw_transcript"])
        errors = word_errors(wanted, heard)
        rows.append({"case_id": ref["case_id"], "voice_id": ref["voice_id"],
                     "opaque_id": ref["opaque_id"], "reference": ref["script"],
                     "raw_transcript": raw["raw_transcript"],
                     "normalized_reference": " ".join(wanted),
                     "normalized_transcript": " ".join(heard),
                     "word_errors": errors, "reference_words": len(wanted),
                     "full_script_match": errors == 0})
    return rows, {"raw_sha256": complete["raw_sha256"],
                  "complete_sha256": sha(output / "complete.json"),
                  "meta_sha256": sha(output / "meta.json")}


def compare(whisper: Path, wav2vec2: Path, output: Path, inputs: list[dict],
            references: list[dict]) -> None:
    check_freeze()
    require(whisper.resolve() != wav2vec2.resolve()
            and not output.exists()
            and not output.resolve().is_relative_to(RENDER.resolve()),
            "comparison paths invalid")
    by_model, sources = {}, {}
    for model, directory in (("whisper", whisper), ("wav2vec2", wav2vec2)):
        by_model[model], sources[model] = compare_one(directory, model, inputs, references)
    output.mkdir(parents=True, exist_ok=False)
    counts = {model: {"matched": sum(row["full_script_match"] for row in rows),
                      "denominator": N,
                      "total_word_errors": sum(row["word_errors"] for row in rows),
                      "reference_words": sum(row["reference_words"] for row in rows)}
              for model, rows in by_model.items()}
    exclusive_json(output / "comparison.json", {
        "status": "descriptive_asr_qa_comparison_no_model_inference",
        "runner_sha256": sha(Path(__file__)), "freeze_sha256": sha(FREEZE),
        "manifest_sha256": EXPECTED["manifest"], "scripts_sha256": EXPECTED["scripts"],
        "sources": sources, "counts": counts, "total_observations": 2 * N,
        "rows": by_model, "human_listening_pending": True,
        "voicechat_inference_authorized": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", choices=MODELS)
    mode.add_argument("--compare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--whisper-dir", type=Path)
    parser.add_argument("--wav2vec2-dir", type=Path)
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false"})
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(THREADS)
    inputs, references = validate_inputs()
    if args.compare_only:
        require(args.output_dir is not None and args.whisper_dir is not None
                and args.wav2vec2_dir is not None, "comparison needs all three directories")
        compare(args.whisper_dir, args.wav2vec2_dir, args.output_dir, inputs, references)
        return
    v2 = load_v2()
    pins = json.loads(PINS.read_text())
    snapshots = v2.check_model_pins(pins)
    if args.validate_only:
        print(json.dumps({"status": "validated_only_no_asr_output", "clips": len(inputs),
                          "manifest_sha256": EXPECTED["manifest"],
                          "pins_sha256": EXPECTED["pins"],
                          "models": {k: pins["models"][k]["revision"] for k in MODELS}}, indent=2))
        return
    require(args.output_dir is not None, "--run needs an exclusive output directory")
    run(args.run, args.output_dir, inputs, pins, snapshots, v2)


if __name__ == "__main__":
    main()
