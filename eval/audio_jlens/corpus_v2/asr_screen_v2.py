"""Prospective, two-model CPU ASR screen of all 480 frozen v2 WAVs.

``--validate-only`` hashes sources, weights, and WAVs without loading a model.
``--run`` additionally requires an independent freeze receipt. Model output is
written raw, one row per WAV, before any reference comparison or normalization.
No Qwen or Voxtral model is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "rendered_v2/manifest.json"
BINDING = HERE / "rendered_v2/TECHNICAL_BINDING.json"
INPUTS = HERE / "ASR_INPUTS_PROSPECTIVE_V1.json"
PINS = HERE / "ASR_MODEL_PINS_PROSPECTIVE_V1.json"
PROTOCOL = HERE / "ASR_LISTENING_QA_PROTOCOL_PROSPECTIVE_V1.md"
AMENDMENT = HERE / "ASR_MIDDLE_SLOT_FLAG_AMENDMENT_V1.md"
FREEZE = HERE / "ASR_SCREEN_FREEZE_V1.json"
EXPECTED = {
    "manifest": "15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d",
    "binding": "858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93",
    "cases": "d9a08a61b76e8ab56ab3a9b2d72a4aedc5871555487c7a33b712097fe3206ed8",
    "inputs": "74f38f6704d1995b8e7040b329f7c0c0ea68b3cae36e774b875b4c358ae76d33",
    "pins": "1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b",
    "protocol": "30285a6c897e742eff59f593b3d12a1bd92d62585cd72171837c3e18fabe662e",
    "amendment": "9a6c12199e591ac623ee74938d13a4a6b11dbbbadce21e79cb75bd888176246d",
}
CLIP_ROOT = (HERE / "rendered_v2/clips").resolve()
N_CLIPS = 480
N_THREADS = 4
MAX_NEW_TOKENS = 128


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def exclusive_json(path: Path, body: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_sources() -> tuple[dict, dict]:
    for key, path in (("manifest", MANIFEST), ("binding", BINDING),
                      ("inputs", INPUTS), ("pins", PINS), ("protocol", PROTOCOL),
                      ("amendment", AMENDMENT)):
        require(path.is_file() and sha(path) == EXPECTED[key], f"frozen {key} SHA changed")
    manifest = json.loads(MANIFEST.read_text())
    binding = json.loads(BINDING.read_text())
    cases = json.loads(INPUTS.read_text())
    require(set(cases) == {"status", "provenance_cases_sha256", "planned_clips", "cases"}
            and binding["status"] == "cpu_wav_technical_audit_passed_intelligibility_pending_no_model_output"
            and binding["manifest_sha256"] == EXPECTED["manifest"]
            and binding["matched_records"] == 240 and binding["wavs"] == N_CLIPS
            and cases["status"] == "prospective_reference_free_asr_inputs"
            and cases["provenance_cases_sha256"] == EXPECTED["cases"]
            and cases["planned_clips"] == N_CLIPS and len(cases["cases"]) == N_CLIPS,
            "corpus/QA binding counts changed")
    source = {}
    for pair in manifest["pairs"]:
        for word in pair["words"]:
            item = pair["outputs"][word]
            key = item["path"]
            require(key not in source, "manifest duplicate WAV path")
            source[key] = item
    require(len(source) == N_CLIPS, "manifest is not 480 unique WAVs")
    seen_ids = set()
    seen_paths = set()
    inputs = []
    for case in cases["cases"]:
        require(set(case) == {"opaque_id", "audio_path", "wav_sha256",
                              "pcm_int16_sha256", "frames"},
                "ASR input contains an unplanned field")
        opaque_id = case["opaque_id"]
        relative = case["audio_path"]
        path = (ROOT / relative).resolve()
        require(isinstance(opaque_id, str) and len(opaque_id) == 20
                and all(c in "0123456789abcdef" for c in opaque_id)
                and opaque_id not in seen_ids and relative not in seen_paths,
                "duplicate or malformed ASR input identity")
        require(path.is_relative_to(CLIP_ROOT) and path.is_file()
                and relative in source, "ASR input escaped frozen clips")
        item = source[relative]
        require(case["wav_sha256"] == item["file_sha256"] == sha(path)
                and case["pcm_int16_sha256"] == item["pcm_int16_sha256"]
                and case["frames"] == item["frames"]
                and 0 < item["duration_seconds"] < 30,
                "ASR input WAV/hash/geometry mismatch")
        seen_ids.add(opaque_id)
        seen_paths.add(relative)
        inputs.append({"opaque_id": opaque_id, "path": path,
                       "wav_sha256": case["wav_sha256"],
                       "pcm_int16_sha256": case["pcm_int16_sha256"],
                       "frames": case["frames"]})
    require(seen_paths == set(source) and len(seen_ids) == N_CLIPS,
            "ASR input does not cover exactly 480 WAVs")
    for item in inputs:
        load_audio(item)  # PCM readback, still no model load or transcript.
    return json.loads(PINS.read_text()), {"inputs": sorted(inputs, key=lambda x: x["opaque_id"]),
                                         "binding": binding}


def check_model_pins(pins: dict) -> dict[str, Path]:
    """Resolve only already cached checkpoints; no ASR weights are loaded."""
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    import huggingface_hub
    from huggingface_hub import snapshot_download

    require(sys.version_info[:2] == (3, 12)
            and pins["runtime"]["python"] == "3.12", "Python version drift")
    installed = {"torch": torch.__version__, "transformers": transformers.__version__,
                 "huggingface_hub": huggingface_hub.__version__, "numpy": np.__version__,
                 "soundfile": sf.__version__}
    require(all(installed[key] == version for key, version in pins["runtime"].items()
                if key != "python"), "ASR dependency version drift")
    source_root = Path(transformers.__file__).resolve().parent
    for name, expected in pins["transformers_sources_sha256"].items():
        path = source_root / name
        require(path.is_file() and sha(path) == expected,
                f"Transformers ASR implementation changed: {name}")
    snapshots = {}
    for key in ("whisper", "wav2vec2"):
        model = pins["models"][key]
        path = Path(snapshot_download(model["repo_id"], revision=model["revision"],
                                      allow_patterns=list(model["files_sha256"]),
                                      local_files_only=True)).resolve()
        for name, expected in model["files_sha256"].items():
            file = path / name
            require(file.is_file() and sha(file) == expected,
                    f"cached {key} checkpoint file changed: {name}")
        snapshots[key] = path
    require(pins["models"]["whisper"]["files_sha256"]["model.safetensors"]
            == "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad"
            and pins["models"]["wav2vec2"]["files_sha256"]["model.safetensors"]
            == "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a",
            "ASR weight pins changed")
    return snapshots


def check_freeze() -> dict:
    require(AMENDMENT.is_file() and sha(AMENDMENT) == EXPECTED["amendment"],
            "frozen ASR middle-slot amendment SHA changed")
    require(FREEZE.is_file(), "ASR screen freeze receipt absent; independent review required")
    receipt = json.loads(FREEZE.read_text())
    require(receipt.get("status") == "independently_reviewed_asr_screen_frozen"
            and receipt.get("script_sha256") == sha(Path(__file__))
            and all(receipt.get(f"{key}_sha256") == EXPECTED[key]
                    for key in ("manifest", "binding", "cases", "inputs", "pins",
                                "protocol", "amendment")),
            "ASR screen freeze receipt does not bind exact sources")
    return receipt


def load_audio(item: dict):
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(item["path"], dtype="int16", always_2d=True)
    require(rate == 16000 and samples.shape == (item["frames"], 1),
            "ASR input PCM geometry changed")
    pcm = samples[:, 0].astype("<i2", copy=False)
    require(hashlib.sha256(pcm.tobytes()).hexdigest() == item["pcm_int16_sha256"]
            and sha(item["path"]) == item["wav_sha256"],
            "ASR input WAV/PCM changed during run")
    return pcm.astype(np.float32) / 32768.0


def transcribe(model_key: str, processor, model, audio) -> dict:
    """Model receives the waveform only, never target text or candidate words."""
    import torch

    with torch.inference_mode():
        if model_key == "whisper":
            batch = processor(audio, sampling_rate=16000, return_tensors="pt",
                              return_attention_mask=True)
            ids = model.generate(**batch, do_sample=False, num_beams=1,
                                 max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
            raw = processor.batch_decode([ids], skip_special_tokens=True)[0]
            return {"raw_transcript": raw, "generated_token_ids": ids,
                    "decoder_cap": MAX_NEW_TOKENS,
                    "ended_with_eos": bool(ids and ids[-1] == model.generation_config.eos_token_id)}
        if model_key == "wav2vec2":
            batch = processor(audio, sampling_rate=16000, return_tensors="pt")
            logits = model(**batch).logits
            require(logits.ndim == 3 and logits.shape[0] == 1
                    and torch.isfinite(logits).all().item(), "nonfinite CTC logits")
            ids = logits.argmax(dim=-1)[0].tolist()
            raw = processor.batch_decode([ids])[0]
            return {"raw_transcript": raw, "frame_argmax_token_ids": ids,
                    "ctc_frame_count": len(ids)}
    raise ValueError(f"unknown ASR model: {model_key}")


def verify_raw_log(path: Path, inputs: list[dict], model_key: str) -> str:
    """Fail closed on any missing, duplicated, reordered, or labeled raw row."""
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(inputs), "raw ASR log omitted a planned WAV")
    forbidden = {"reference", "target_word", "paired_alternative", "pair_id",
                 "split", "category", "normalized_transcript", "wer"}
    for ordinal, (line, item) in enumerate(zip(lines, inputs, strict=True), 1):
        row = json.loads(line)
        require(not forbidden.intersection(row)
                and row.get("ordinal") == ordinal
                and row.get("opaque_id") == item["opaque_id"]
                and row.get("wav_sha256") == item["wav_sha256"]
                and row.get("pcm_int16_sha256") == item["pcm_int16_sha256"]
                and isinstance(row.get("raw_transcript"), str),
                "raw ASR row identity/transcript mismatch or reference leaked")
        tokens = row.get("generated_token_ids" if model_key == "whisper"
                         else "frame_argmax_token_ids")
        require(isinstance(tokens, list) and tokens
                and all(isinstance(t, int) and t >= 0 for t in tokens),
                "raw ASR token sequence missing or malformed")
        if model_key == "whisper":
            require(row.get("decoder_cap") == MAX_NEW_TOKENS
                    and isinstance(row.get("ended_with_eos"), bool),
                    "Whisper termination metadata missing")
        else:
            require(row.get("ctc_frame_count") == len(tokens),
                    "Wav2Vec2 CTC frame count changed")
    return sha(path)


def run(model_key: str, output: Path, pins: dict, inputs: list[dict],
        snapshots: dict[str, Path]) -> dict:
    check_freeze()
    import torch
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              Wav2Vec2ForCTC, Wav2Vec2Processor)

    require(model_key in ("whisper", "wav2vec2"), "unknown ASR model")
    require(not output.exists(), "ASR output directory already exists")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
    spec = pins["models"][model_key]
    model_dir = str(snapshots[model_key])
    metadata = {"status": "started_raw_asr_only", "model_key": model_key,
                "model_id": spec["repo_id"], "model_revision": spec["revision"],
                "model_files_sha256": spec["files_sha256"],
                "script_sha256": sha(Path(__file__)), "pins_sha256": sha(PINS),
                "freeze_sha256": sha(FREEZE), "technical_binding_sha256": sha(BINDING),
                "asr_inputs_sha256": sha(INPUTS),
                "provenance_cases_sha256": EXPECTED["cases"],
                "planned_clips": N_CLIPS,
                "device": "cpu", "threads": N_THREADS,
                "python": sys.version.split()[0], "platform": platform.platform(),
                "generation": {"whisper": {"do_sample": False, "num_beams": 1,
                                             "max_new_tokens": MAX_NEW_TOKENS,
                                             "processor_attention_mask": True},
                               "wav2vec2": {"decoder": "frame_argmax_ctc_greedy"}},
                "scope": "raw transcripts only, no reference comparison or Qwen output"}
    exclusive_json(output / "meta.json", metadata)
    raw_path = output / "raw.jsonl"
    try:
        processor = (WhisperProcessor if model_key == "whisper" else Wav2Vec2Processor).from_pretrained(
            model_dir, local_files_only=True)
        model = (WhisperForConditionalGeneration if model_key == "whisper" else Wav2Vec2ForCTC).from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True).to("cpu").eval()
        require(all(p.device.type == "cpu" for p in model.parameters()),
                "ASR weights are not entirely on CPU")
        count = 0
        with raw_path.open("x", encoding="utf-8") as stream:
            for item in inputs:
                audio = load_audio(item)
                result = transcribe(model_key, processor, model, audio)
                require(isinstance(result["raw_transcript"], str),
                        "ASR transcript is not text")
                row = {"ordinal": count + 1, "opaque_id": item["opaque_id"],
                       "wav_sha256": item["wav_sha256"],
                       "pcm_int16_sha256": item["pcm_int16_sha256"], **result}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                count += 1
        require(count == N_CLIPS, "ASR pass incomplete")
        raw_sha = verify_raw_log(raw_path, inputs, model_key)
        complete = {"status": "complete_raw_asr_no_reference_comparison",
                    "model_key": model_key, "planned_clips": N_CLIPS,
                    "raw_rows": count, "raw_sha256": raw_sha,
                    "meta_sha256": sha(output / "meta.json"),
                    "elapsed_seconds": time.monotonic() - started,
                    "no_model_answer_inference": True}
        exclusive_json(output / "complete.json", complete)
        return complete
    except BaseException as exc:
        exclusive_json(output / "error.json", {"status": "incomplete_raw_asr",
                    "model_key": model_key, "error_type": type(exc).__name__,
                    "error": str(exc), "elapsed_seconds": time.monotonic() - started,
                    "raw_sha256": sha(raw_path) if raw_path.is_file() else None,
                    "raw_rows": raw_path.read_text().count("\n") if raw_path.is_file() else 0})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", choices=("whisper", "wav2vec2"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(N_THREADS)
    pins, checked = check_sources()
    snapshots = check_model_pins(pins)
    if args.validate_only:
        print(json.dumps({"status": "validated_only_no_asr_output",
                          "clips": len(checked["inputs"]),
                          "models": {k: {"id": pins["models"][k]["repo_id"],
                                         "revision": pins["models"][k]["revision"],
                                         "safetensors_sha256": pins["models"][k]["files_sha256"]["model.safetensors"]}
                                     for k in snapshots},
                          "pins_sha256": sha(PINS)}, indent=2))
        return
    require(args.output_dir is not None, "--run needs an exclusive output directory")
    run(args.run, args.output_dir.resolve(), pins, checked["inputs"], snapshots)


if __name__ == "__main__":
    main()
