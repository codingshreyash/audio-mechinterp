"""Frozen Voxtral corpus_v1 FIT-pilot capability gate (32 audio clips only).

--validate-only performs CPU processor preflight without loading model weights.
--run explicitly enables one bounded BF16 MPS pass and immutable result files.
No development or locked-test WAV is ever opened by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import resource
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = HERE / "voxtral_corpus_v1_index.json"
SOURCE_MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
SNAPSHOT = HERE / "hf_cache/voxtral"
RUNS = HERE / "runs/voxtral_corpus_v1_fit_pilot"
EXPECTED_INDEX_SHA256 = "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971"
EXPECTED_SOURCE_MANIFEST_SHA256 = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
REVISION = "3060fe34b35ba5d44202ce9ff3c097642914f8f3"
MODEL_CONFIG_SHA256 = "368124c9a9171b3a6c0cdb35cd7fedeff465eefecf315e2a621684bbed7dcd7e"
GENERATION_CONFIG_SHA256 = "cebeed3d4a1680c9b311863385d32b91008480dfbf8cf7abc2142447b8f73b76"
PROMPT = "Answer the question in the audio with exactly one word. No explanation or punctuation."
PILOT_PAIR_IDS = ("animal_06", "color_03", "nature_00", "object_02")
MAX_NEW_TOKENS = 32
MEMORY_LIMIT_BYTES = 30 * 1024**3
EXPECTED_TRANSFORMERS_SOURCE_SHA256 = {
    "modeling_voxtral": "2d3ec9d78c469b77c42a8d5ba258185fb1c58f4c2c7bf6007ec724436dfc7056",
    "processing_voxtral": "239e84657d1a0287ab27a540c946a787a69b4b0bf45727d6b24bee21ccec0296",
    "feature_extraction_whisper": "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_pilot(index: dict) -> list[tuple[dict, str, dict]]:
    if index["source_manifest_sha256"] != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Frozen source manifest binding mismatch")
    if index["model_revision"] != REVISION or index["prompt"] != PROMPT or index["max_new_tokens"] != MAX_NEW_TOKENS:
        raise ValueError("Frozen Voxtral index settings mismatch")
    if tuple(index["pilot_pair_ids"]) != PILOT_PAIR_IDS:
        raise ValueError("Frozen pilot pair IDs changed")
    pilot_rows = [r for r in index["rows"] if r["pilot_fit"]]
    if len(pilot_rows) != 16:
        raise ValueError(f"Expected 16 pilot pair records, got {len(pilot_rows)}")
    if any(r["split"] != "fit_pool" or not r["map_fit"] or r["pair_id"] not in PILOT_PAIR_IDS for r in pilot_rows):
        raise ValueError("Non-fit or unselected row marked as pilot")
    counts = Counter((r["pair_id"], r["engine"]) for r in pilot_rows)
    if set(counts) != {(p, e) for p in PILOT_PAIR_IDS for e in ("piper", "kokoro")} or set(counts.values()) != {2}:
        raise ValueError("Pilot voice/template balance changed")
    jobs = [(r, w, r["clips"][w]) for r in pilot_rows for w in r["words"]]
    jobs.sort(key=lambda x: (x[0]["pair_id"], x[0]["engine"], x[0]["phrasing_id"], x[1]))
    if len(jobs) != 32:
        raise ValueError("Expected exactly 32 pilot clips")
    return jobs


def classify_answer(decoded: str, expected_word: str, eos: bool, disallowed_special_ids: list[int] | None = None) -> dict:
    stripped = decoded.strip()
    has_one_word = len(stripped.split()) == 1
    disallowed_special_ids = disallowed_special_ids or []
    exact = eos and not disallowed_special_ids and has_one_word and stripped.casefold() == expected_word.casefold()
    return {
        "decoded_stripped": stripped,
        "one_whitespace_word": has_one_word,
        "disallowed_special_ids": disallowed_special_ids,
        "exact_expected_word_and_eos": exact,
        "classification": "exact" if exact else ("missing_eos" if not eos else (
            "disallowed_special" if disallowed_special_ids else "other")),
    }


def canonical_ids(clip: dict, word: str, eos: int) -> list[int]:
    surface = word.capitalize()
    rows = [a for a in clip["answer_candidate_sequences"] if a["surface"] == surface]
    if len(rows) != 1:
        raise ValueError(f"Expected one frozen canonical surface for {word}")
    ids = rows[0]["token_ids_with_eos"]
    if len(ids) < 2 or ids[-1] != eos or any(not isinstance(x, int) for x in ids):
        raise ValueError(f"Invalid frozen canonical sequence for {word}")
    return ids


def verify_pinned_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> dict:
    """Hash actual bytes; a matching file size alone is never accepted."""
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Pinned file size changed: {path.name}: {size} != {expected_size}")
    actual_sha256 = sha(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Pinned file SHA-256 changed: {path.name}: {actual_sha256} != {expected_sha256}")
    return {"size_bytes": size, "sha256": actual_sha256}


def validate_snapshot(index: dict) -> dict:
    config = SNAPSHOT / "config.json"
    config_verified = verify_pinned_file(config, MODEL_CONFIG_SHA256)
    tree = SNAPSHOT / ".cache/huggingface/trees" / f"{REVISION}.json"
    if not tree.is_file():
        raise FileNotFoundError("Pinned snapshot revision tree missing")
    required = (
        "config.json", "preprocessor_config.json", "generation_config.json", "tekken.json",
        "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    )
    if any(not (SNAPSHOT / name).is_file() for name in required):
        raise FileNotFoundError("Pinned local snapshot incomplete")
    tree_files = json.loads(tree.read_text())["files"]
    weight_shards = {}
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        expected = tree_files[name]
        if expected["size"] != expected["lfs_size"]:
            raise ValueError(f"Pinned shard tree sizes disagree: {name}")
        weight_shards[name] = verify_pinned_file(
            SNAPSHOT / name, expected["lfs_sha256"], int(expected["lfs_size"]),
        )
    processor_config = verify_pinned_file(
        SNAPSHOT / "preprocessor_config.json", index["processor_config_sha256"],
        int(tree_files["preprocessor_config.json"]["size"]),
    )
    tokenizer_asset = verify_pinned_file(
        SNAPSHOT / "tekken.json", index["tokenizer_asset_sha256"],
        int(tree_files["tekken.json"]["size"]),
    )
    generation_config = verify_pinned_file(
        SNAPSHOT / "generation_config.json", GENERATION_CONFIG_SHA256,
        int(tree_files["generation_config.json"]["size"]),
    )
    return {
        "model_id": MODEL_ID, "revision": REVISION, "config": config_verified,
        "snapshot_tree_sha256": sha(tree), "weight_shards": weight_shards,
        "processor_config": processor_config, "tokenizer_asset": tokenizer_asset,
        "generation_config": generation_config,
    }


def validate_transformers_source() -> dict[str, str]:
    from transformers.models.voxtral import modeling_voxtral, processing_voxtral
    from transformers.models.whisper import feature_extraction_whisper

    modules = {
        "modeling_voxtral": modeling_voxtral,
        "processing_voxtral": processing_voxtral,
        "feature_extraction_whisper": feature_extraction_whisper,
    }
    actual = {name: sha(Path(inspect.getfile(module))) for name, module in modules.items()}
    if actual != EXPECTED_TRANSFORMERS_SOURCE_SHA256:
        raise ValueError(f"Installed Transformers source hash changed: {actual}")
    return actual


def preflight(index: dict, jobs: list[tuple[dict, str, dict]]) -> tuple[object, list[dict]]:
    import soundfile as sf
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), local_files_only=True)
    if processor.tokenizer.eos_token_id != 2 or int(processor.audio_token_id) != 24:
        raise ValueError("Pinned processor token IDs changed")
    prepared = []
    pair_input_ids = defaultdict(list)
    for row, word, clip in jobs:
        wav = (ROOT / clip["path"]).resolve()
        if not wav.is_relative_to((ROOT / "eval/audio_jlens/corpus_v1/rendered/clips").resolve()):
            raise ValueError(f"Pilot WAV path escaped corpus: {wav}")
        if sha(wav) != clip["wav_sha256"]:
            raise ValueError(f"Pilot WAV hash changed: {wav}")
        with sf.SoundFile(wav) as f:
            if f.samplerate != 16000 or f.channels != 1 or f.frames != clip["frames"]:
                raise ValueError(f"Pilot WAV format changed: {wav}")
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": str(wav)}, {"type": "text", "text": PROMPT},
        ]}]
        batch = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt",
        )
        input_ids = batch["input_ids"][0].tolist()
        if tuple(batch["input_features"].shape) != (1, 128, 3000) or input_ids.count(24) != 375:
            raise ValueError(f"Unexpected processor shape/placeholders for {wav}")
        if batch["input_ids"].shape[0] != 1 or batch["attention_mask"].shape != batch["input_ids"].shape:
            raise ValueError(f"Unexpected prompt shape for {wav}")
        key = (row["pair_id"], row["engine"], row["phrasing_id"])
        pair_input_ids[key].append(input_ids)
        for candidate_word in row["words"]:
            ids = canonical_ids(row["clips"][candidate_word], candidate_word, 2)
            if processor.tokenizer.encode(candidate_word.capitalize(), add_special_tokens=False) + [2] != ids:
                raise ValueError(f"Frozen canonical token IDs changed for {candidate_word}")
        prepared.append({
            "row": row, "word": word, "clip": clip, "batch": batch,
            "input_ids": input_ids, "input_features_shape": list(batch["input_features"].shape),
            "audio_placeholder_count": input_ids.count(24),
        })
    if len(pair_input_ids) != 16 or any(len(v) != 2 or v[0] != v[1] for v in pair_input_ids.values()):
        raise ValueError("Matched pilot pair prompt token IDs differ")
    return processor, prepared


class MemorySampler:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.stop = threading.Event()
        self.driver_max = 0
        self.current_max = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            self.driver_max = max(self.driver_max, int(self.torch.mps.driver_allocated_memory()))
            self.current_max = max(self.current_max, int(self.torch.mps.current_allocated_memory()))
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        self.driver_max = max(self.driver_max, int(self.torch.mps.driver_allocated_memory()))
        self.current_max = max(self.current_max, int(self.torch.mps.current_allocated_memory()))


def score_sequence(model, batch, sequence_with_eos: list[int]) -> dict:
    import torch

    if sequence_with_eos[-1] != 2:
        raise ValueError("Canonical sequence lacks EOS")
    prompt_len = int(batch["input_ids"].shape[1])
    continuation = torch.tensor([sequence_with_eos[:-1]], dtype=batch["input_ids"].dtype, device="mps")
    full_ids = torch.cat((batch["input_ids"].to("mps"), continuation), dim=1)
    attention = torch.cat((batch["attention_mask"].to("mps"), torch.ones_like(continuation)), dim=1)
    with torch.inference_mode():
        logits = model(
            input_ids=full_ids, attention_mask=attention,
            input_features=batch["input_features"].to("mps"),
            use_cache=False, logits_to_keep=0,
        ).logits[0]
        token_ll = []
        for offset, token_id in enumerate(sequence_with_eos):
            row = logits[prompt_len - 1 + offset].float()
            token_ll.append(float((row[token_id] - torch.logsumexp(row, dim=-1)).cpu()))
    return {
        "token_ids_with_eos": sequence_with_eos,
        "token_logprobs": token_ll,
        "complete_reply_logprob": sum(token_ll),
    }


def append_line(path: Path, value: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def exclusive_json(path: Path, value: dict) -> None:
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run(processor, prepared: list[dict], meta: dict, output_dir: Path) -> None:
    import torch
    from transformers import VoxtralForConditionalGeneration

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no weights loaded")
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("Unable to establish MPS memory recommendation")
    fraction = min(1.0, MEMORY_LIMIT_BYTES / recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    output_dir.mkdir(parents=True, exist_ok=False)
    meta = {**meta, "mps_recommended_max_bytes": recommended, "mps_fraction_limit": fraction}
    exclusive_json(output_dir / "run_meta.json", meta)
    progress = output_dir / "progress.jsonl"
    progress.touch(exist_ok=False)
    results = []
    try:
        with MemorySampler(torch) as memory:
            load_start = time.monotonic()
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(SNAPSHOT), local_files_only=True, dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa",
            ).eval()
            load_seconds = time.monotonic() - load_start
            if torch.mps.driver_allocated_memory() > MEMORY_LIMIT_BYTES:
                raise MemoryError("MPS driver allocation exceeded 30 GiB after load")
            eos = int(model.generation_config.eos_token_id)
            pad = int(model.generation_config.pad_token_id)
            if eos != 2:
                raise ValueError("Pinned EOS ID changed")
            for ordinal, job in enumerate(prepared, 1):
                row, word, clip, batch = (job[k] for k in ("row", "word", "clip", "batch"))
                start = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        **batch.to("mps"), do_sample=False, use_cache=True,
                        max_new_tokens=MAX_NEW_TOKENS,
                        eos_token_id=eos, pad_token_id=pad,
                    )
                torch.mps.synchronize()
                elapsed = time.monotonic() - start
                suffix_ids = generated[0, batch["input_ids"].shape[1]:].tolist()
                raw_decoded = processor.batch_decode([suffix_ids], skip_special_tokens=True)[0]
                decoded_with_special_tokens = processor.batch_decode([suffix_ids], skip_special_tokens=False)[0]
                ended = bool(suffix_ids and suffix_ids[-1] == eos)
                special_ids = set(processor.tokenizer.all_special_ids)
                disallowed_special_ids = [token_id for position, token_id in enumerate(suffix_ids)
                                          if token_id in special_ids and not (position == len(suffix_ids) - 1 and token_id == eos)]
                classification = classify_answer(raw_decoded, word, ended, disallowed_special_ids)
                candidates = {}
                for candidate_word in row["words"]:
                    ids = canonical_ids(row["clips"][candidate_word], candidate_word, eos)
                    candidates[candidate_word] = score_sequence(model, batch, ids)
                other_word = next(w for w in row["words"] if w != word)
                record = {
                    "ordinal": ordinal, "pair_id": row["pair_id"], "split": row["split"],
                    "engine": row["engine"], "voice": row["voice"],
                    "phrasing_id": row["phrasing_id"], "expected_word": word,
                    "other_word": other_word, "wav_path": clip["path"],
                    "wav_sha256": clip["wav_sha256"], "pcm_int16_sha256": clip["pcm_int16_sha256"],
                    "prompt_input_ids": job["input_ids"],
                    "audio_placeholder_count": job["audio_placeholder_count"],
                    "generated_token_ids": suffix_ids, "raw_decoded": raw_decoded,
                    "decoded_with_special_tokens": decoded_with_special_tokens,
                    "stopped_on_eos": ended, "classification": classification,
                    "canonical_scores": candidates,
                    "other_minus_expected_complete_reply_logprob": (
                        candidates[other_word]["complete_reply_logprob"]
                        - candidates[word]["complete_reply_logprob"]
                    ),
                    "generation_seconds": elapsed,
                    "mps_driver_allocated_bytes_after": int(torch.mps.driver_allocated_memory()),
                    "mps_current_allocated_bytes_after": int(torch.mps.current_allocated_memory()),
                }
                append_line(progress, record)
                results.append(record)
                if memory.driver_max > MEMORY_LIMIT_BYTES or torch.mps.driver_allocated_memory() > MEMORY_LIMIT_BYTES:
                    raise MemoryError("MPS driver allocation exceeded 30 GiB")
                print(f"{ordinal}/32 {row['engine']} {row['pair_id']} {row['phrasing_id']} {word}: {raw_decoded!r}, eos={ended}", flush=True)
        by_engine = defaultdict(Counter)
        for record in results:
            for key in ("all", record["engine"]):
                by_engine[key]["clips"] += 1
                by_engine[key]["complete_exact"] += int(record["classification"]["exact_expected_word_and_eos"])
        counts = {k: dict(v) for k, v in by_engine.items()}
        gate = counts["all"]["complete_exact"] >= 24 and all(counts[e]["complete_exact"] >= 12 for e in ("piper", "kokoro"))
        final = {
            "status": "complete", "run_meta_sha256": sha(output_dir / "run_meta.json"),
            "progress_sha256": sha(progress), "progress_records": len(results),
            "model_load_seconds": load_seconds,
            "mps_driver_observed_max_bytes": memory.driver_max,
            "mps_current_observed_max_bytes": memory.current_max,
            "process_max_rss_bytes_darwin": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "counts": counts, "fit_capability_gate_pass": gate,
            "records": results,
        }
        if len(results) != 32 or counts["all"]["clips"] != 32:
            raise ValueError("Incomplete pilot result")
        exclusive_json(output_dir / "result.json", final)
        print(f"saved {output_dir / 'result.json'}; sha256={sha(output_dir / 'result.json')}")
    except Exception as exc:
        exclusive_json(output_dir / "error.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "completed_records": len(results), "progress_sha256": sha(progress),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-only", action="store_true", help="CPU processor preflight only")
    actions.add_argument("--run", action="store_true", help="explicitly load BF16 model and run MPS pilot")
    parser.add_argument("--output-dir", type=Path, help="new immutable run directory; only valid with --run")
    args = parser.parse_args()
    if args.validate_only and args.output_dir:
        parser.error("--output-dir requires --run")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if sha(INDEX) != EXPECTED_INDEX_SHA256:
        raise ValueError("Frozen Voxtral CPU index SHA-256 changed")
    actual_source_manifest_sha256 = sha(SOURCE_MANIFEST)
    if actual_source_manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Frozen source corpus manifest SHA-256 changed")
    index = json.loads(INDEX.read_text())
    jobs = select_pilot(index)
    snapshot = validate_snapshot(index)
    source_hashes = validate_transformers_source()
    processor, prepared = preflight(index, jobs)
    versions = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": __import__("torch").__version__,
        "transformers": __import__("transformers").__version__,
    }
    if versions["torch"] != "2.15.0" or versions["transformers"] != "5.20.2":
        raise ValueError("Pinned dependency versions changed")
    meta = {
        "status": "frozen_fit_pilot_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha(Path(__file__)), "index_sha256": EXPECTED_INDEX_SHA256,
        "source_manifest_sha256": actual_source_manifest_sha256,
        "protocol_sha256": sha(HERE / "VOXTRAL_CORPUS_V1_PROTOCOL.md"),
        "requirements_freeze_sha256": sha(HERE / "requirements.freeze.txt"),
        "snapshot": snapshot, "versions": versions,
        "installed_transformers_source_sha256": source_hashes,
        "prompt": PROMPT, "pilot_pair_ids": list(PILOT_PAIR_IDS),
        "generation": {"dtype": "bfloat16", "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "eos_token_id": 2, "use_cache": True, "attn_implementation": "sdpa"},
        "secondary_score": "canonical initial-capital answer token sequence plus EOS; other minus expected",
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "planned_clips": 32,
    }
    if args.validate_only:
        print(json.dumps({
            "status": "validated_cpu_only", "index_sha256": EXPECTED_INDEX_SHA256,
            "runner_sha256": meta["runner_sha256"],
            "pilot_pair_ids": list(PILOT_PAIR_IDS), "clips": len(prepared),
            "pair_records": 16, "processor_shapes": sorted({tuple(x["input_features_shape"]) for x in prepared}),
            "audio_placeholder_counts": sorted({x["audio_placeholder_count"] for x in prepared}),
            "model_weights_loaded": False,
        }, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or RUNS / stamp
    run(processor, prepared, meta, output_dir)


if __name__ == "__main__":
    main()
