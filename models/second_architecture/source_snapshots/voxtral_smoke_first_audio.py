"""Bounded, offline Voxtral audio Q&A smoke with matched typed controls.

The explicit --run flag is required to load model weights or touch MPS.
See VOXTRAL_PLAN.md for the frozen prompt and interpretation limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAPSHOT = HERE / "hf_cache" / "voxtral"
MANIFEST = ROOT / "eval" / "tts" / "outputs" / "controlled_pair_v1" / "controlled_pair_manifest.json"
RUN_DIR = HERE / "runs" / "voxtral_smoke_v1"
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
REVISION = "3060fe34b35ba5d44202ce9ff3c097642914f8f3"
TRANSFORMERS_COMMIT = "18329f1065382e54db698e242cf98bd3f3e5a766"
PROMPT = "Answer the question in the audio with one color word."
MAX_NEW_TOKENS = 32
MEMORY_LIMIT = 30 * 1024**3
EXPECTED_CONFIG_SHA256 = "368124c9a9171b3a6c0cdb35cd7fedeff465eefecf315e2a621684bbed7dcd7e"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_snapshot() -> dict:
    """Inspect only local small files/weight sizes, never the network."""
    required = (
        "config.json", "preprocessor_config.json", "generation_config.json",
        "tekken.json", "model.safetensors.index.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    )
    missing = [name for name in required if not (SNAPSHOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Pinned local snapshot incomplete: {missing}")
    tree = SNAPSHOT / ".cache" / "huggingface" / "trees" / f"{REVISION}.json"
    if not tree.is_file():
        raise FileNotFoundError(f"Missing revision-specific local tree: {tree}")
    metadata_dir = SNAPSHOT / ".cache" / "huggingface" / "download"
    for name in required:
        metadata = metadata_dir / f"{name}.metadata"
        if not metadata.is_file() or metadata.read_text().splitlines()[0] != REVISION:
            raise ValueError(f"Local file lacks pinned revision metadata: {name}")
    config_hash = sha256_file(SNAPSHOT / "config.json")
    if config_hash != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"Local config changed: {config_hash}")
    config = json.loads((SNAPSHOT / "config.json").read_text())
    if config.get("model_type") != "voxtral" or config.get("audio_token_id") != 24:
        raise ValueError("Unexpected local Voxtral configuration")
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())
    indexed_shards = set(index["weight_map"].values())
    expected_shards = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    if indexed_shards != expected_shards:
        raise ValueError(f"Unexpected weight shards: {indexed_shards}")
    tree_data = json.loads(tree.read_text())["files"]
    shard_sizes = {}
    for name in sorted(expected_shards):
        actual = (SNAPSHOT / name).stat().st_size
        expected = int(tree_data[name]["size"])
        if actual != expected:
            raise ValueError(f"Incorrect local shard size for {name}: {actual} != {expected}")
        shard_sizes[name] = actual
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot": str(SNAPSHOT),
        "tree_sha256": sha256_file(tree),
        "config_sha256": config_hash,
        "index_sha256": sha256_file(SNAPSHOT / "model.safetensors.index.json"),
        "shard_sizes_bytes": shard_sizes,
    }


def select_clip(engine: str, word: str) -> tuple[Path, dict]:
    manifest = json.loads(MANIFEST.read_text())
    info = manifest["engines"][engine]["outputs"][word]
    wav = ROOT / info["path"]
    actual = sha256_file(wav)
    if actual != info["file_sha256"]:
        raise ValueError(f"Stimulus hash mismatch: {wav}")
    return wav, {
        "engine": engine,
        "word": word,
        "manifest_sha256": sha256_file(MANIFEST),
        "wav": str(wav),
        "wav_sha256": actual,
        "sample_rate": manifest["sample_rate"],
        "frames": info["frames"],
        "duration_seconds": info["duration_seconds"],
        "word_slot_start_sample": manifest["engines"][engine]["word_slot_start_sample"],
        "word_slot_end_sample_exclusive": manifest["engines"][engine]["word_slot_end_sample_exclusive"],
    }


def select_typed(word: str) -> tuple[str, dict]:
    text = f"I said the color {word}. Which color did I say? Answer with one color word."
    return text, {
        "input_modality": "typed",
        "engine": None,
        "word": word,
        "manifest_sha256": sha256_file(MANIFEST),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "wav": None,
        "wav_sha256": None,
        "sample_rate": None,
        "frames": None,
        "duration_seconds": None,
        "word_slot_start_sample": None,
        "word_slot_end_sample_exclusive": None,
    }


def prepare_processor(wav: Path | None, typed_text: str | None = None) -> tuple[object, object, dict]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), local_files_only=True)
    if typed_text is None:
        if wav is None:
            raise ValueError("Audio input requires a WAV path")
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "path": str(wav)},
                {"type": "text", "text": PROMPT},
            ],
        }]
    else:
        if wav is not None:
            raise ValueError("Typed input must not include a WAV path")
        conversation = [{"role": "user", "content": [{"type": "text", "text": typed_text}]}]
    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    token_ids = batch["input_ids"][0].tolist()
    audio_id = int(processor.audio_token_id)
    if audio_id != 24:
        raise ValueError("Unexpected Voxtral processor output")
    if batch["input_ids"].shape[0] != 1:
        raise ValueError(f"Unexpected processor shapes: {[(k, tuple(v.shape)) for k, v in batch.items()]}")
    if typed_text is None:
        if "input_features" not in batch or batch["input_features"].shape != (1, 128, 3000):
            raise ValueError(f"Unexpected audio features: {[(k, tuple(v.shape)) for k, v in batch.items()]}")
        input_features_shape = list(batch["input_features"].shape)
    else:
        if "input_features" in batch:
            raise ValueError("Typed control unexpectedly contains audio features")
        input_features_shape = None
    report = {
        "processor_class": type(processor).__name__,
        "user_message": conversation,
        "input_ids": token_ids,
        "input_ids_shape": list(batch["input_ids"].shape),
        "input_features_shape": input_features_shape,
        "audio_token_id": audio_id,
        "audio_placeholder_count": token_ids.count(audio_id),
    }
    expected_placeholders = 375 if typed_text is None else 0
    if report["audio_placeholder_count"] != expected_placeholders:
        raise ValueError(f"Unexpected audio placeholder count: {report['audio_placeholder_count']}")
    return processor, batch, report


def package_versions() -> dict:
    from importlib import metadata

    names = ("torch", "transformers", "mistral-common", "accelerate", "soundfile", "librosa", "huggingface-hub")
    versions = {name: metadata.version(name) for name in names}
    required = {"torch": "2.15.0", "transformers": "5.20.2", "mistral-common": "1.13.1",
                "accelerate": "1.16.0", "soundfile": "0.14.0", "librosa": "0.11.0"}
    mismatch = {name: (versions[name], version) for name, version in required.items() if versions[name] != version}
    if mismatch:
        raise RuntimeError(f"Dependency pins differ from plan: {mismatch}")
    return versions


class MemorySampler:
    """Collect approximate process MPS high-water marks during load and generate."""

    def __init__(self, torch_module):
        self.torch = torch_module
        self.stop_event = threading.Event()
        self.current_max = 0
        self.driver_max = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_event.is_set():
            self.current_max = max(self.current_max, int(self.torch.mps.current_allocated_memory()))
            self.driver_max = max(self.driver_max, int(self.torch.mps.driver_allocated_memory()))
            self.stop_event.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.current_max = max(self.current_max, int(self.torch.mps.current_allocated_memory()))
        self.driver_max = max(self.driver_max, int(self.torch.mps.driver_allocated_memory()))


def run_model(processor, batch, expected_word: str) -> dict:
    import torch
    from transformers import VoxtralForConditionalGeneration

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; model inference was not started")
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("Cannot establish MPS recommended maximum memory")
    fraction = min(1.0, MEMORY_LIMIT / recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    t0 = time.monotonic()
    with MemorySampler(torch) as memory:
        model = VoxtralForConditionalGeneration.from_pretrained(
            str(SNAPSHOT),
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="mps",
            attn_implementation="sdpa",
        ).eval()
        load_seconds = time.monotonic() - t0
        if torch.mps.driver_allocated_memory() > MEMORY_LIMIT:
            raise MemoryError("MPS allocation exceeded 30 GiB after model load")
        inputs = batch.to("mps")
        eos = int(model.generation_config.eos_token_id)
        if eos != 2:
            raise ValueError(f"Unexpected EOS token: {eos}")
        t1 = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=eos,
                pad_token_id=int(model.generation_config.pad_token_id),
            )
        generation_seconds = time.monotonic() - t1
        if torch.mps.driver_allocated_memory() > MEMORY_LIMIT:
            raise MemoryError("MPS allocation exceeded 30 GiB after generation")
        generated_ids = output[0, batch["input_ids"].shape[1]:].tolist()
    answer = processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()
    stopped_on_eos = bool(generated_ids and generated_ids[-1] == eos)
    return {
        "model_class": type(model).__name__,
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "eos_token_id": eos, "pad_token_id": int(model.generation_config.pad_token_id),
                       "use_cache": True, "attn_implementation": "sdpa"},
        "generated_token_ids": generated_ids,
        "generated_token_count": len(generated_ids),
        "full_answer": answer,
        "stopped_on_eos": stopped_on_eos,
        "stop_reason": "eos" if stopped_on_eos else "max_new_tokens_or_other",
        "exact_one_color_word": bool(re.fullmatch(rf"{re.escape(expected_word)}[.!]?", answer, flags=re.I)),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "mps_recommended_max_bytes": recommended,
        "mps_fraction_limit": fraction,
        "mps_current_observed_max_bytes": memory.current_max,
        "mps_driver_observed_max_bytes": memory.driver_max,
        "process_max_rss_bytes_darwin": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="load local weights and run one MPS generation")
    parser.add_argument("--engine", choices=("piper", "kokoro"), default="piper")
    parser.add_argument("--word", choices=("red", "blue"), default="red")
    parser.add_argument("--typed", action="store_true", help="use frozen typed control instead of WAV")
    parser.add_argument("--output", type=Path, help="new JSON output path; refuses overwrite")
    args = parser.parse_args()

    # Disable network access in both processor-only and inference modes.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    snapshot = validate_snapshot()
    if args.typed:
        typed_text, stimulus = select_typed(args.word)
        wav = None
    else:
        wav, stimulus = select_clip(args.engine, args.word)
        typed_text = None
    versions = package_versions()
    processor, batch, prepared = prepare_processor(wav, typed_text)
    report = {
        "schema_version": 1,
        "status": "preflight" if not args.run else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "transformers_source_commit": TRANSFORMERS_COMMIT,
        "dependency_versions": versions,
        "requirements_freeze_sha256": sha256_file(HERE / "requirements.freeze.txt"),
        "snapshot": snapshot,
        "stimulus": stimulus,
        "processor": prepared,
    }
    if not args.run:
        print(json.dumps(report, indent=2))
        return 0

    # A new output path is chosen before weights are loaded; prior results are never overwritten.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = "typed" if args.typed else args.engine
    output = args.output or (RUN_DIR / f"{label}_{args.word}_{stamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    try:
        report["result"] = run_model(processor, batch, args.word)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(f"Smoke failed; evidence saved to {output}", file=sys.stderr)
        raise
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Smoke saved to {output}")
    print(f"Full answer: {report['result']['full_answer']!r}; EOS: {report['result']['stopped_on_eos']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
