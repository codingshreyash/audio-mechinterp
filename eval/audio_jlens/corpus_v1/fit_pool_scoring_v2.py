"""FIT-only comparison of V1 aliases with post-fit V2 prefix/sequence scores.

Evaluates the same 48 map-fit inputs. V1 remains the frozen primary first-step
score; V2 is secondary, explicitly adapted after FIT capability outputs.
No development or locked-test candidate audio is prepared.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "models" / "qwen_omni"),
                str(ROOT / "eval" / "audio_jlens")]
from adapter import MODEL_ID, MODEL_REVISION, TRANSFORMERS_REVISION, prepare_audio_batch  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import (  # noqa: E402
    MAX_MPS_DRIVER_BYTES, load_pinned_thinker, record_mps_memory, scored_forward,
)
from fit_pool_capability_v1 import load_fit_cases, sha, validate_tokenizer  # noqa: E402
from fit_pool_single_token_score_v1 import first_step_metrics, selected_cases  # noqa: E402


SCORING_V2 = HERE / "SCORING_V2.json"
SCORING_V2_SHA = "e7ea18cd0d5e2638bec64b8733049a5a2754ecd674080d93766db8409b46dd9b"
V1_RESULT = HERE / "fit_pool_single_token_score_v1_results.json"
V1_RESULT_SHA = "34128f325b1e08818be0369fb91754cedda6331ac42c07439326cb1b09608c45"
N_CONTEXTS = 48


def load_scoring(spec: dict, cases: list[dict]) -> dict:
    if sha(SCORING_V2) != SCORING_V2_SHA:
        raise ValueError("V2 scoring artifact changed")
    policy = json.loads(SCORING_V2.read_text())
    if (policy["source_spec_sha256"] != "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265"
            or policy["model_revision_for_tokenizer_only"] != MODEL_REVISION
            or policy["eos_token_id"] != spec["tokenizer_eos_id"]):
        raise ValueError("V2 scoring provenance mismatch")
    if sha(V1_RESULT) != V1_RESULT_SHA:
        raise ValueError("Original-policy V1 result changed")
    prior = json.loads(V1_RESULT.read_text())
    if [c["case_id"] for c in cases] != [c["case_id"] for c in prior["cases"]]:
        raise ValueError("V2 FIT cases differ from V1 cases")
    for case in cases:
        expected = policy["words"][case["word"]]
        foil = policy["words"][case["foil_word"]]
        if set(expected["first_token_ids"]) & set(foil["first_token_ids"]):
            raise ValueError("Expanded first-token classes overlap")
    return policy


def complete_class_score(model, batch: dict, variants: list[list[int]]) -> tuple[float, list[float]]:
    if not variants or len({tuple(v) for v in variants}) != len(variants):
        raise ValueError("Candidate sequences missing or duplicated")
    with torch.no_grad():
        values = [float(scored_forward(model, batch, ids)) for ids in variants]
    return float(torch.logsumexp(torch.tensor(values, dtype=torch.float64), dim=0)), values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=HERE / "fit_pool_scoring_v2_results.json")
    args = parser.parse_args()
    spec, manifest, all_cases = load_fit_cases()
    validate_tokenizer(spec)
    cases = selected_cases(spec, all_cases)
    policy = load_scoring(spec, cases)
    if args.validate_only:
        print(json.dumps({
            "status": "V2_fit_inputs_validated_no_model_forward",
            "cases": len(cases), "v2_sha256": SCORING_V2_SHA,
            "v1_result_sha256": V1_RESULT_SHA,
            "manifest_sha256": sha(HERE / "rendered" / "manifest.json"),
        }, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 V2 score comparison requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source changed")
    output = args.output.resolve()
    progress, memory = output.with_suffix(".progress.jsonl"), output.with_suffix(".memory.jsonl")
    if output.exists() or progress.exists() or memory.exists():
        raise FileExistsError("V2 score comparison refuses overwrite")
    progress.write_text("")
    memory.write_text("")
    torch.mps.set_per_process_memory_fraction(
        mps_cap_fraction(int(torch.mps.recommended_max_memory()))
    )
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32 or processor.tokenizer.eos_token_id != policy["eos_token_id"]:
        raise ValueError("Model/processor FP32/EOS pin mismatch")
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
            raise ValueError("Audio changed after FIT validation")
        batch = batch.to(model.device, dtype=model.dtype)
        tick = time.perf_counter()
        with torch.no_grad():
            logits = model(**batch, use_cache=False, return_dict=True).logits[0, -1]
            v1 = first_step_metrics(
                logits,
                [a["token_id"] for a in case["expected_aliases"]],
                [a["token_id"] for a in case["foil_aliases"]],
            )
            v2_prefix = first_step_metrics(
                logits, policy["words"][case["word"]]["first_token_ids"],
                policy["words"][case["foil_word"]]["first_token_ids"],
            )
        expected_sequences = policy["words"][case["word"]]["complete_answer_sequences_plus_eos"]
        foil_sequences = policy["words"][case["foil_word"]]["complete_answer_sequences_plus_eos"]
        expected_class, expected_values = complete_class_score(model, batch, expected_sequences)
        foil_class, foil_values = complete_class_score(model, batch, foil_sequences)
        torch.mps.synchronize()
        row = {
            "case_id": case["case_id"], "pair_id": case["pair_id"],
            "category": case["category"], "engine": case["engine"],
            "word": case["word"], "foil_word": case["foil_word"],
            "audio_sha256": digest,
            "v1_first_step": v1,
            "v2_first_step": v2_prefix,
            "v2_complete_sequence": {
                "expected_sequences": expected_sequences,
                "foil_sequences": foil_sequences,
                "expected_sequence_log_probabilities": expected_values,
                "foil_sequence_log_probabilities": foil_values,
                "expected_class_log_probability": expected_class,
                "foil_class_log_probability": foil_class,
                "correct_minus_foil_log_odds": expected_class - foil_class,
                "correct_class_gt_foil": expected_class > foil_class,
            },
            "case_seconds": time.perf_counter() - tick,
        }
        rows.append(row)
        with progress.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        record_mps_memory(memory, f"after_case_{index+1}")
        if (index + 1) % 12 == 0:
            print(f"completed {index+1}/{N_CONTEXTS} FIT V2 comparisons", flush=True)
    old = json.loads(V1_RESULT.read_text())
    prior_by_id = {c["case_id"]: c for c in old["cases"]}
    v1_differences = [abs(r["v1_first_step"]["correct_minus_foil_log_odds"] -
                          prior_by_id[r["case_id"]]["metrics"]["correct_minus_foil_log_odds"])
                      for r in rows]
    summary = {
        "planned": N_CONTEXTS, "completed": len(rows),
        "v1_expected_gt_foil": sum(r["v1_first_step"]["correct_mass_gt_foil"] for r in rows),
        "v2_prefix_expected_gt_foil": sum(r["v2_first_step"]["correct_mass_gt_foil"] for r in rows),
        "v2_complete_expected_gt_foil": sum(r["v2_complete_sequence"]["correct_class_gt_foil"] for r in rows),
        "max_abs_v1_rerun_log_odds_difference": max(v1_differences),
    }
    report = {
        "status": "fit_only_v2_comparison_complete",
        "script_sha256": sha(Path(__file__)),
        "v2_scoring_sha256": SCORING_V2_SHA,
        "v1_score_result_sha256": V1_RESULT_SHA,
        "spec_sha256": policy["source_spec_sha256"],
        "manifest_sha256": sha(HERE / "rendered" / "manifest.json"),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "system_prompt": spec["system_prompt"],
        "progress_sha256": sha(progress), "memory_sha256": sha(memory),
        "rows": rows, "summary": summary,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
