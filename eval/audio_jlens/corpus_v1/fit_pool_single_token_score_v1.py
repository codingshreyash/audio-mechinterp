"""FIT-only prompt-last assay of the *original frozen single-token* aliases.

Runs exactly the 48 map-fitting inputs, all selected before any model answer.
It makes one FP32 unpatched Qwen forward per input, records first-step class
mass and margin, and never opens development/test audio or model answers.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "models" / "qwen_omni"),
                str(ROOT / "eval" / "audio_jlens")]
from adapter import MODEL_ID, MODEL_REVISION, TRANSFORMERS_REVISION, prepare_audio_batch  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import MAX_MPS_DRIVER_BYTES, load_pinned_thinker, record_mps_memory  # noqa: E402
from fit_pool_capability_v1 import (  # noqa: E402
    MANIFEST_SHA, SPEC_SHA, digest_json, load_fit_cases, sha, validate_tokenizer,
)


N_CONTEXTS = 48


def first_step_metrics(logits: torch.Tensor, expected_ids: list[int],
                       foil_ids: list[int]) -> dict:
    """Class log masses over the full vocabulary, using original alias IDs."""
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected finite one-position logits")
    expected = sorted(set(expected_ids))
    foil = sorted(set(foil_ids))
    if not expected or not foil or set(expected) & set(foil):
        raise ValueError("Alias classes empty or overlapping")
    if min(expected + foil) < 0 or max(expected + foil) >= logits.shape[0]:
        raise ValueError("Alias token ID outside vocabulary")
    values = logits.float()
    log_z = torch.logsumexp(values, dim=0)
    expected_logmass = torch.logsumexp(values[expected], dim=0) - log_z
    foil_logmass = torch.logsumexp(values[foil], dim=0) - log_z
    top = int(values.argmax())
    return {
        "expected_alias_token_ids": expected,
        "foil_alias_token_ids": foil,
        "expected_log_mass": float(expected_logmass),
        "foil_log_mass": float(foil_logmass),
        "expected_mass": float(expected_logmass.exp()),
        "foil_mass": float(foil_logmass.exp()),
        "correct_minus_foil_log_odds": float(expected_logmass - foil_logmass),
        "correct_mass_gt_foil": bool(expected_logmass > foil_logmass),
        "top_vocab_token_id": top,
        "top_vocab_log_probability": float(values[top] - log_z),
    }


def selected_cases(spec: dict, all_fit_cases: list[dict]) -> list[dict]:
    chosen = [case for case in all_fit_cases if case["map_fit"]]
    if len(chosen) != N_CONTEXTS or {case["template"] for case in chosen} != {"fit_chosen"}:
        raise ValueError("Map-fitting context selection drifted")
    pairs = {p["pair_id"]: p for p in spec["word_pairs"]["fit_pool"]}
    for case in chosen:
        a, b = pairs[case["pair_id"]]["words"]
        foil = b if case["word"] == a else a if case["word"] == b else None
        if foil is None:
            raise ValueError("Case word not in frozen pair")
        case["foil_word"] = foil
        case["foil_aliases"] = spec["aliases"][foil]
    return chosen


def summarize(rows: list[dict]) -> dict:
    result = {"planned": N_CONTEXTS, "completed": len(rows),
              "expected_mass_gt_foil": sum(r["metrics"]["correct_mass_gt_foil"] for r in rows)}
    groups = defaultdict(list)
    for row in rows:
        for key in (f"pair/{row['pair_id']}", f"engine/{row['engine']}",
                    f"category/{row['category']}"):
            groups[key].append(row)
    result["groups"] = {
        key: {"planned": len(items),
              "expected_mass_gt_foil": sum(x["metrics"]["correct_mass_gt_foil"] for x in items),
              "median_log_odds": float(torch.tensor([
                  x["metrics"]["correct_minus_foil_log_odds"] for x in items
              ]).median())}
        for key, items in sorted(groups.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=HERE / "fit_pool_single_token_score_v1_results.json")
    args = parser.parse_args()
    spec, _manifest, all_cases = load_fit_cases()
    validate_tokenizer(spec)
    cases = selected_cases(spec, all_cases)
    if args.validate_only:
        print(json.dumps({
            "status": "fit_only_input_validated_no_model_forward",
            "contexts": len(cases), "spec_sha256": SPEC_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "cases_sha256": digest_json(cases),
        }, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 score assay requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision drift")
    output = args.output.resolve()
    progress = output.with_suffix(".progress.jsonl")
    memory = output.with_suffix(".memory.jsonl")
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or progress.exists() or memory.exists() or metadata_path.exists():
        raise FileExistsError("Score assay refuses to overwrite an existing run")
    metadata = {
        "status": "fit_only_single_token_alias_score",
        "script_sha256": sha(Path(__file__)),
        "spec_sha256": SPEC_SHA, "manifest_sha256": MANIFEST_SHA,
        "cases_sha256": digest_json(cases),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "system_prompt": spec["system_prompt"],
        "site": "unpatched prompt-last Thinker logits",
        "dtype": "float32", "device": "mps",
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "policy": "original SPEC_V1.json one-token aliases only; no expanded tokenization",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    progress.write_text("")
    memory.write_text("")
    torch.mps.set_per_process_memory_fraction(
        mps_cap_fraction(int(torch.mps.recommended_max_memory()))
    )
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
    rows = []
    for index, case in enumerate(cases):
        batch, digest = prepare_audio_batch(
            processor, ROOT / case["audio_path"], system_prompt=spec["system_prompt"]
        )
        if digest != case["audio_sha256"]:
            raise ValueError("Audio changed after frozen input validation")
        batch = batch.to(model.device, dtype=model.dtype)
        tick = time.perf_counter()
        with torch.no_grad():
            outputs = model(**batch, use_cache=False, return_dict=True)
            scores = first_step_metrics(
                outputs.logits[0, -1],
                [a["token_id"] for a in case["expected_aliases"]],
                [a["token_id"] for a in case["foil_aliases"]],
            )
        torch.mps.synchronize()
        seconds = time.perf_counter() - tick
        scores["top_vocab_token_text"] = processor.tokenizer.decode(
            [scores["top_vocab_token_id"]], skip_special_tokens=False
        )
        row = {**case, "prompt_tokens": int(batch["input_ids"].shape[1]),
               "forward_seconds": seconds, "metrics": scores}
        rows.append(row)
        with progress.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        record_mps_memory(memory, f"after_case_{index+1}")
        if (index + 1) % 12 == 0:
            print(f"completed {index+1}/{N_CONTEXTS} fit contexts", flush=True)
    report = {**metadata, "status": "fit_only_complete_all_48",
              "progress_sha256": sha(progress), "memory_sha256": sha(memory),
              "cases": rows, "summary": summarize(rows)}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
