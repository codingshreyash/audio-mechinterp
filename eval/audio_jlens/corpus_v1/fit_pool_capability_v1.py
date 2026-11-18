"""Full-answer capability run on all frozen FIT-POOL audio, and nothing else.

Use --validate-only for CPU input validation. Default mode loads the pinned
Qwen Thinker on MPS, evaluates all 192 fit-pool clips without selection, and
saves progress after each answer. Development/test clips are never prepared.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_DIR = ROOT / "models" / "qwen_omni"
EVAL_DIR = ROOT / "eval" / "audio_jlens"
sys.path[:0] = [str(MODEL_DIR), str(EVAL_DIR)]
from adapter import MODEL_ID, MODEL_REVISION, TRANSFORMERS_REVISION, prepare_audio_batch  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import MAX_MPS_DRIVER_BYTES, load_pinned_thinker, record_mps_memory  # noqa: E402


SPEC = HERE / "SPEC_V1.json"
MANIFEST = HERE / "rendered" / "manifest.json"
CLIPS = (HERE / "rendered" / "clips").resolve()
SPEC_SHA = "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
SPLIT = "fit_pool"
N_CASES = 192
MAX_NEW_TOKENS = 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def classify_answer(raw_text: str, token_ids: list[int], eos_id: int,
                    expected_word: str, aliases: list[dict]) -> dict:
    """Freeze raw, strict, and one-terminal-punctuation semantic readings."""
    if not isinstance(raw_text, str):
        raise ValueError("Generated text must be a string")
    stopped = bool(token_ids and token_ids[-1] == eos_id)
    trimmed = raw_text.strip()
    strict = trimmed.casefold() == expected_word.casefold()
    semantic = trimmed[:-1].strip() if trimmed.endswith((".", "?", "!")) else trimmed
    semantic = semantic.casefold() == expected_word.casefold()
    allowed_ids = {int(alias["token_id"]) for alias in aliases}
    first_alias = bool(token_ids and token_ids[0] in allowed_ids)
    return {
        "decoded_trimmed": trimmed,
        "stopped_on_eos": stopped,
        "strict_exact_word": strict,
        "semantic_one_word": semantic,
        "strict_word_plus_eos": strict and stopped,
        "semantic_word_plus_eos": semantic and stopped,
        "first_token_in_expected_aliases": first_alias,
        "first_token_id": token_ids[0] if token_ids else None,
        "expected_alias_token_ids": sorted(allowed_ids),
    }


def load_fit_cases() -> tuple[dict, dict, list[dict]]:
    """Validate frozen metadata and only FIT-POOL WAVs; no model is loaded."""
    if sha(SPEC) != SPEC_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Frozen corpus selection or render manifest changed")
    spec, manifest = json.loads(SPEC.read_text()), json.loads(MANIFEST.read_text())
    if (spec["status"] != "selection_frozen_before_cpu_render_and_candidate_answers"
            or manifest["spec_sha256"] != SPEC_SHA
            or manifest["status"] != "cpu_stimuli_only_no_candidate_or_asr_model_answers"):
        raise ValueError("Corpus provenance status changed")
    if spec["model_id_for_tokenizer_only"] != MODEL_ID or spec["model_revision_for_tokenizer_only"] != MODEL_REVISION:
        raise ValueError("Model/tokenizer pin mismatch")
    fit_pairs = {p["pair_id"]: p for p in spec["word_pairs"][SPLIT]}
    template_ids = {p["id"] for p in spec["templates"]["fit"]}
    engines = set(spec["render"]["voices"])
    selected = [p for p in manifest["pairs"] if p["split"] == SPLIT]
    if len(selected) != 96:
        raise ValueError("FIT-POOL pair-record count is not 96")
    seen_groups = set()
    cases = []
    for pair in selected:
        pair_id, template, engine = pair["pair_id"], pair["phrasing_id"], pair["engine"]
        key = (pair_id, template, engine)
        if key in seen_groups or pair_id not in fit_pairs or template not in template_ids or engine not in engines:
            raise ValueError(f"Unexpected/duplicate FIT-POOL group: {key}")
        seen_groups.add(key)
        chosen = fit_pairs[pair_id]
        if pair["words"] != chosen["words"] or pair["map_fit"] != chosen["map_fit"]:
            raise ValueError(f"FIT-POOL group differs from frozen selection: {key}")
        pcm = {}
        for word in pair["words"]:
            info = pair["outputs"][word]
            path = (ROOT / info["path"]).resolve()
            if not path.is_relative_to(CLIPS) or sha(path) != info["file_sha256"]:
                raise ValueError(f"FIT-POOL audio path/hash invalid: {path}")
            samples, rate = sf.read(path, dtype="int16")
            if rate != 16000 or samples.ndim != 1 or len(samples) != info["frames"]:
                raise ValueError(f"FIT-POOL WAV invalid: {path}")
            pcm[word] = samples
            cases.append({
                "case_id": f"{pair_id}/{template}/{engine}/{word}",
                "pair_id": pair_id, "category": chosen["category"],
                "template": template, "engine": engine, "word": word,
                "map_fit": bool(chosen["map_fit"] and template == "fit_chosen"),
                "audio_path": str(path.relative_to(ROOT)),
                "audio_sha256": info["file_sha256"],
                "expected_aliases": spec["aliases"][word],
            })
        a, b = (pcm[w] for w in pair["words"])
        start, end = pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"]
        if (len(a) != len(b) or not np.array_equal(a[:start], b[:start])
                or not np.array_equal(a[end:], b[end:])
                or np.array_equal(a[start:end], b[start:end])):
            raise ValueError(f"FIT-POOL PCM pair identity failed: {key}")
    if len(cases) != N_CASES or len({c["case_id"] for c in cases}) != N_CASES:
        raise ValueError("FIT-POOL case count/IDs invalid")
    if len(seen_groups) != len(fit_pairs) * len(template_ids) * len(engines):
        raise ValueError("A FIT-POOL group is missing")
    return spec, manifest, sorted(cases, key=lambda case: case["case_id"])


def validate_tokenizer(spec: dict) -> None:
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                      local_files_only=True))
    if (sha(snapshot / "tokenizer.json") != spec["tokenizer_json_sha256"]
            or sha(snapshot / "tokenizer_config.json") != spec["tokenizer_config_sha256"]):
        raise ValueError("Pinned tokenizer files changed")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id != spec["tokenizer_eos_id"]:
        raise ValueError("EOS token changed")
    for word, aliases in spec["aliases"].items():
        for alias in aliases:
            if tokenizer.encode(alias["surface"], add_special_tokens=False) != [alias["token_id"]]:
                raise ValueError(f"Answer alias changed for {word}")


def summarize(rows: list[dict]) -> dict:
    """Keep every planned row; count subsets without selecting replacements."""
    result = {}
    for field in ("category", "engine", "template", "map_fit"):
        groups = {}
        for value in sorted({str(row[field]) for row in rows}):
            subset = [r for r in rows if str(r[field]) == value]
            groups[value] = {
                "planned": len(subset),
                "ended_on_eos": sum(r["classification"]["stopped_on_eos"] for r in subset),
                "strict_word_plus_eos": sum(r["classification"]["strict_word_plus_eos"] for r in subset),
                "semantic_word_plus_eos": sum(r["classification"]["semantic_word_plus_eos"] for r in subset),
                "first_token_expected_alias": sum(r["classification"]["first_token_in_expected_aliases"] for r in subset),
            }
        result[field] = groups
    result["overall"] = {
        "planned": N_CASES, "completed": len(rows),
        "ended_on_eos": sum(r["classification"]["stopped_on_eos"] for r in rows),
        "strict_word_plus_eos": sum(r["classification"]["strict_word_plus_eos"] for r in rows),
        "semantic_word_plus_eos": sum(r["classification"]["semantic_word_plus_eos"] for r in rows),
        "first_token_expected_alias": sum(r["classification"]["first_token_in_expected_aliases"] for r in rows),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=HERE / "fit_pool_capability_v1_results.json")
    args = parser.parse_args()
    spec, manifest, cases = load_fit_cases()
    validate_tokenizer(spec)
    if args.validate_only:
        print(json.dumps({
            "status": "fit_input_only_validated_no_qwen_weights_or_answers",
            "cases": len(cases), "map_fit_primary": sum(c["map_fit"] for c in cases),
            "spec_sha256": SPEC_SHA, "manifest_sha256": MANIFEST_SHA,
            "cases_sha256": digest_json(cases),
            "categories": dict(Counter(c["category"] for c in cases)),
            "engines": dict(Counter(c["engine"] for c in cases)),
            "templates": dict(Counter(c["template"] for c in cases)),
        }, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 capability runner requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision changed")
    output = args.output.resolve()
    progress = output.with_suffix(".progress.jsonl")
    memory = output.with_suffix(".memory.jsonl")
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists():
        raise FileExistsError("Final capability result already exists")
    metadata = {
        "status": "fit_pool_only",
        "script_sha256": sha(Path(__file__)),
        "spec_sha256": SPEC_SHA, "manifest_sha256": MANIFEST_SHA,
        "cases_sha256": digest_json(cases),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "system_prompt": spec["system_prompt"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "dtype": "float32", "device": "mps",
        "memory_limit_driver_bytes": MAX_MPS_DRIVER_BYTES,
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Resume metadata differs from frozen run")
        if not progress.exists() or not memory.exists():
            raise ValueError("Resume sidecars incomplete")
    else:
        if progress.exists() or memory.exists():
            raise ValueError("Orphaned progress or memory log")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        progress.write_text("")
        memory.write_text("")
    rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    if len(rows) > N_CASES or [r["case_id"] for r in rows] != [c["case_id"] for c in cases[:len(rows)]]:
        raise ValueError("Saved progress is not a prefix of frozen cases")
    recommended = int(torch.mps.recommended_max_memory())
    torch.mps.set_per_process_memory_fraction(mps_cap_fraction(recommended))
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32 or processor.tokenizer.eos_token_id != spec["tokenizer_eos_id"]:
        raise ValueError("Model/processor dtype or EOS changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record_mps_memory(memory, "after_model_load")
    for index, case in enumerate(cases[len(rows):], start=len(rows)):
        audio_path = ROOT / case["audio_path"]
        batch, digest = prepare_audio_batch(
            processor, audio_path, system_prompt=spec["system_prompt"]
        )
        if digest != case["audio_sha256"]:
            raise ValueError("Audio changed between validation and generation")
        batch = batch.to(model.device, dtype=model.dtype)
        tick = time.perf_counter()
        with torch.inference_mode():
            generation = model.generate(
                **batch, do_sample=False, use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=spec["tokenizer_eos_id"],
                pad_token_id=spec["tokenizer_eos_id"],
                return_dict_in_generate=True, output_scores=False,
            )
        torch.mps.synchronize()
        seconds = time.perf_counter() - tick
        n_prompt = int(batch["input_ids"].shape[1])
        token_ids = generation.sequences[0, n_prompt:].tolist()
        raw_text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        classification = classify_answer(
            raw_text, token_ids, spec["tokenizer_eos_id"],
            case["word"], case["expected_aliases"],
        )
        record = {**case,
                  "prompt_tokens": n_prompt,
                  "generated_token_ids": token_ids,
                  "decoded_raw": raw_text,
                  "decoded_with_special_tokens": processor.tokenizer.decode(
                      token_ids, skip_special_tokens=False
                  ),
                  "generation_seconds": seconds,
                  "classification": classification}
        rows.append(record)
        with progress.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        record_mps_memory(memory, f"after_case_{index+1}")
        if (index + 1) % 16 == 0:
            print(f"completed {index+1}/{N_CASES} fit-pool clips", flush=True)
    if len(rows) != N_CASES:
        raise ValueError("FIT-POOL capability run incomplete")
    report = {
        **metadata, "status": "fit_pool_complete_all_planned",
        "progress_sha256": sha(progress), "memory_sha256": sha(memory),
        "cases": rows, "counts": summarize(rows),
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "counts": report["counts"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
