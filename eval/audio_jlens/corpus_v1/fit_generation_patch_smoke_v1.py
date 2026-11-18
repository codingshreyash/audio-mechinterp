"""FIT-only integration check for Stage A's greedy generation patch helper.

Uses the first frozen map-fit FIT context, both source sites, and alpha 0/.025.
No development or locked-test audio/model outputs are opened.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "eval" / "audio_jlens"
sys.path[:0] = [str(ROOT / "models" / "qwen_omni"), str(EVAL)]
from adapter import TRANSFORMERS_REVISION, resolve_hook_points  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import MAX_MPS_DRIVER_BYTES, load_pinned_thinker, record_mps_memory  # noqa: E402
from fit_pool_capability_v1 import load_fit_cases, sha, validate_tokenizer  # noqa: E402
from fit_pool_single_token_score_v1 import selected_cases  # noqa: E402
from fit_rank16_transport_v1 import (SITES, SITE_NAMES, TARGET,
                                     atomic_json_save, case_batch,
                                     rows_for_case)  # noqa: E402
from development_transport_stage_a_v1 import generation_result  # noqa: E402


LENS = HERE / "fit_rank16_transport_v1.lens.pt"
STAGE1 = HERE / "fit_rank16_transport_v1.stage1.pt"
FIT_REPORT = HERE / "fit_rank16_transport_v1.report.json"
SCORING = HERE / "SCORING_V2.json"
SCORING_SHA = "e7ea18cd0d5e2638bec64b8733049a5a2754ecd674080d93766db8409b46dd9b"
ALPHAS = (0.0, 0.025)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=HERE / "fit_generation_patch_smoke_v1.json")
    args = parser.parse_args()
    spec, manifest, all_cases = load_fit_cases()
    validate_tokenizer(spec)
    cases = selected_cases(spec, all_cases)
    first = cases[0]
    donor = next(case for case in cases if
                 case["pair_id"] == first["pair_id"]
                 and case["template"] == first["template"]
                 and case["engine"] == first["engine"]
                 and case["word"] == first["foil_word"])
    if sha(SCORING) != SCORING_SHA:
        raise ValueError("V2 scoring artifact changed")
    fit_report = json.loads(FIT_REPORT.read_text())
    if (sha(LENS) != fit_report["lens_sha256"]
            or sha(STAGE1) != fit_report["stage1_sha256"]):
        raise ValueError("FIT lens/stage1 report binding changed")
    metadata = {
        "status": "fit_only_stage_a_generation_helper_smoke",
        "script_sha256": sha(Path(__file__)),
        "stage_a_script_sha256": sha(HERE / "development_transport_stage_a_v1.py"),
        "fit_report_sha256": sha(FIT_REPORT),
        "stage1_sha256": sha(STAGE1),
        "lens_sha256": sha(LENS),
        "scoring_v2_sha256": SCORING_SHA,
        "case_id": first["case_id"],
        "donor_case_id": donor["case_id"],
        "alphas": list(ALPHAS),
        "sites": list(SITES),
    }
    if args.validate_only:
        print(json.dumps({"status": "fit_only_inputs_valid_no_model_forward",
                          "metadata": metadata}, indent=2))
        return
    output = args.output.resolve()
    memory = output.with_suffix(".memory.jsonl")
    if output.exists() or memory.exists():
        raise FileExistsError("FIT generation smoke refuses overwrite")
    if not torch.backends.mps.is_available():
        raise RuntimeError("FIT generation smoke requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers revision changed")
    stage1 = torch.load(STAGE1, map_location="cpu", weights_only=True)
    if stage1["metadata"] != fit_report["metadata"]:
        raise ValueError("FIT stage1 metadata differs from report")
    by_id = {record["case_id"]: i for i, record in enumerate(stage1["records"])}
    i, donor_i = by_id[first["case_id"]], by_id[donor["case_id"]]
    policy = json.loads(SCORING.read_text())
    memory.write_text("")
    torch.mps.set_per_process_memory_fraction(
        mps_cap_fraction(int(torch.mps.recommended_max_memory())))
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32 or processor.tokenizer.eos_token_id != policy["eos_token_id"]:
        raise ValueError("FIT smoke dtype/EOS changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record_mps_memory(memory, "after_model_load")
    points = {p.name: p.module for p in resolve_hook_points(
        model, audio_layers=(31,), text_layers=(35,))}
    modules = {site: points[SITE_NAMES[site]] for site in SITES}
    modules[TARGET] = points[TARGET]
    groups = {(p["pair_id"], p["phrasing_id"], p["engine"]): p
              for p in manifest["pairs"] if p["split"] == "fit_pool"}
    group = {"system_prompt": spec["system_prompt"], "groups": groups}
    batch, row_map, intervals = case_batch(model, processor, first, group)
    # A clean capture verifies the source shapes used by the generation hook.
    from fit_uniform_calibration_v1 import clean_forward  # fit-only helper
    captured, _ = clean_forward(model, batch, modules)
    rows = rows_for_case(row_map, intervals, captured)
    if rows != stage1["records"][i]["row_indices"]:
        raise ValueError("FIT smoke source rows changed")
    clean = generation_result(model, processor, batch, first, policy)
    record_mps_memory(memory, "after_clean_generation")
    by_site = {}
    for site in SITES:
        if (rows[site] != stage1["records"][donor_i]["row_indices"][site]
                or list(captured[site].shape) != stage1["records"][i]["source_shapes"][site]
                or list(captured[site].shape) != stage1["records"][donor_i]["source_shapes"][site]):
            raise ValueError("FIT smoke donor/recipient rows or shapes changed")
        delta = (stage1["source_bin_means"][site][donor_i]
                 - stage1["source_bin_means"][site][i]).float()
        outcomes = []
        for alpha in ALPHAS:
            outcome = generation_result(
                model, processor, batch, first, policy,
                modules[site], rows[site], delta, alpha,
                tuple(captured[site].shape),
            )
            if alpha == 0 and outcome["token_ids"] != clean["token_ids"]:
                raise ValueError("FIT generation alpha-zero token invariant failed")
            if outcome["audio_source_hook_calls"] != 1:
                raise ValueError("FIT generation hook count changed")
            outcomes.append({"alpha": alpha, **outcome})
            record_mps_memory(memory, f"after_{site}_alpha_{alpha}")
        by_site[site] = {"source_shape": list(captured[site].shape),
                         "row_counts": [len(r) for r in rows[site]],
                         "outcomes": outcomes}
    atomic_json_save(output, {"metadata": metadata, "clean": clean,
                              "sites": by_site, "memory_sha256": sha(memory)})
    print(json.dumps({"status": "fit_generation_patch_smoke_completed",
                      "output": str(output), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
