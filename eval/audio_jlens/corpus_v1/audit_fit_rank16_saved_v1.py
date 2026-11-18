"""CPU-only independent audit of saved FIT rank-16 transport artifacts.

No model weights, inference, development WAVs, or locked-test WAVs are read.
Run ``--stage1-only`` on a smoke prefix; omit it for a completed full fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from fit_pool_capability_v1 import digest_json, load_fit_cases
from fit_pool_single_token_score_v1 import selected_cases


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RANK = 16
N_FIT = 48
SITES = {"encoder31": 1280, "connector": 2048}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def check(condition: bool, name: str, errors: list[str]) -> None:
    if not condition:
        errors.append(name)


def equal_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a, b)


def finite_tensor(value: torch.Tensor, shape: tuple[int, ...]) -> bool:
    return tuple(value.shape) == shape and bool(torch.isfinite(value).all())


def audit(prefix: Path, *, stage1_only: bool) -> dict:
    errors: list[str] = []
    paths = {suffix: prefix.with_suffix(f".{suffix}") for suffix in
             ("stage1.pt", "checkpoint.pt", "progress.jsonl", "memory.jsonl",
              "lens.pt", "report.json", "smoke.json")}
    required = ("stage1.pt", "memory.jsonl") if stage1_only else (
        "stage1.pt", "checkpoint.pt", "progress.jsonl", "memory.jsonl",
        "lens.pt", "report.json")
    for name in required:
        check(paths[name].is_file(), f"missing:{name}", errors)
    if errors:
        return {"passed": False, "errors": errors}

    spec, _manifest, all_fit = load_fit_cases()
    cases = selected_cases(spec, all_fit)
    stage1 = torch.load(paths["stage1.pt"], map_location="cpu", weights_only=True)
    memory = [json.loads(line) for line in paths["memory.jsonl"].read_text().splitlines()
              if line.strip()]
    metadata = stage1["metadata"]
    source_files = {
        "script_sha256": HERE / "fit_rank16_transport_v1.py",
        "spec_sha256": HERE / "SPEC_V1.json",
        "manifest_sha256": HERE / "rendered" / "manifest.json",
        "scoring_v2_sha256": HERE / "SCORING_V2.json",
        "proposal_sha256": ROOT / "eval" / "audio_jlens" / "NEXT_PHASE_CORPUS_LENS_PROPOSAL.md",
    }
    for key, path in source_files.items():
        check(metadata.get(key) == sha(path), f"source_hash:{key}", errors)
    check(metadata.get("cases_sha256") == digest_json(cases), "case_digest", errors)
    check(metadata.get("n_fit") == N_FIT and metadata.get("rank") == RANK,
          "fit_dimensions_metadata", errors)
    check(metadata.get("sites") == list(SITES), "site_metadata", errors)
    check(stage1.get("target_basis_sha256") == tensor_sha(stage1["target_basis"]),
          "basis_hash", errors)

    target = stage1["target_states"]
    basis = stage1["target_basis"]
    singular = stage1["singular_values"]
    check(finite_tensor(target, (N_FIT, 2048)), "target_states_shape_or_finite", errors)
    check(finite_tensor(basis, (2048, RANK)), "basis_shape_or_finite", errors)
    check(finite_tensor(singular, (24,)), "singular_shape_or_finite", errors)
    gram_error = float((basis.T @ basis - torch.eye(RANK)).abs().max())
    check(gram_error <= 1e-4, "basis_orthogonality", errors)
    check(bool(torch.all(singular[:-1] >= singular[1:])), "singular_order", errors)
    check(float(singular[RANK - 1]) > float(singular[0]) * 1e-6,
          "basis_rank_threshold", errors)

    records = stage1["records"]
    check(len(records) == N_FIT, "stage1_record_count", errors)
    row_count_ranges = {site: [10**9, 0] for site in SITES}
    for i, (record, case) in enumerate(zip(records, cases)):
        check((record["case_id"], record["audio_sha256"]) ==
              (case["case_id"], case["audio_sha256"]), f"case_record:{i}", errors)
        check(equal_tensor(record["target_state"], target[i]),
              f"target_row:{i}", errors)
        check(record["target_sha256"] == tensor_sha(record["target_state"]),
              f"target_hash:{i}", errors)
        for site, width in SITES.items():
            rows = record["row_indices"][site]
            shape = record["source_shapes"][site]
            flattened = [index for group in rows for index in group]
            valid = (len(shape) == 2 and shape[1] == width and len(rows) == 6
                     and all(rows) and len(flattened) == len(set(flattened))
                     and all(0 <= index < shape[0] for index in flattened))
            check(valid, f"row_map:{site}:{i}", errors)
            if valid:
                row_count_ranges[site][0] = min(row_count_ranges[site][0],
                                                 min(map(len, rows)))
                row_count_ranges[site][1] = max(row_count_ranges[site][1],
                                                 max(map(len, rows)))

    for site, width in SITES.items():
        check(finite_tensor(stage1["source_bin_means"][site], (N_FIT, 6, width)),
              f"source_means:{site}", errors)
    # Reconstruct the 24 undirected, fit-only differences without calling the fitter.
    index = {case["case_id"]: i for i, case in enumerate(cases)}
    differences = []
    for pair in spec["word_pairs"]["fit_pool"]:
        if not pair.get("map_fit"):
            continue
        a, b = pair["words"]
        for engine in spec["render"]["voices"]:
            stem = f"{pair['pair_id']}/fit_chosen/{engine}/"
            differences.append(target[index[stem + b]] - target[index[stem + a]])
    check(len(differences) == 24, "basis_input_count", errors)
    _, singular_recomputed, vh = torch.linalg.svd(
        torch.stack(differences).float(), full_matrices=False)
    check(torch.allclose(singular, singular_recomputed, atol=1e-4, rtol=1e-5),
          "singular_reconstruction", errors)
    reconstructed_basis = vh[:RANK].T
    projector_error = float((basis @ basis.T -
                             reconstructed_basis @ reconstructed_basis.T).abs().max())
    check(projector_error <= 1e-4, "basis_subspace_reconstruction", errors)

    for category in sorted({case["category"] for case in cases}):
        ids = [i for i, case in enumerate(cases) if case["category"] == category]
        check(len(ids) == 12, f"category_count:{category}", errors)
        saved = stage1["category_means"][category]
        expected = target[ids].double().mean(dim=0).float()
        check(equal_tensor(saved["target"], expected),
              f"category_target_mean:{category}", errors)
        for site in SITES:
            expected = stage1["source_bin_means"][site][ids].double().mean(dim=0).float()
            check(equal_tensor(saved["sources"][site], expected),
                  f"category_source_mean:{category}:{site}", errors)

    peak = max((int(row["driver_allocated_bytes"]) for row in memory), default=0)
    over = sum(bool(row["over_limit"]) for row in memory)
    check(bool(memory), "empty_memory_log", errors)
    check(over == 0 and peak <= int(metadata["max_mps_driver_bytes"]),
          "memory_limit", errors)
    result = {
        "passed": not errors, "errors": errors,
        "stage1_sha256": sha(paths["stage1.pt"]),
        "basis_sha256": tensor_sha(basis),
        "basis_orthogonality_max_abs": gram_error,
        "basis_subspace_reconstruction_max_abs": projector_error,
        "singular_value_1": float(singular[0]),
        "singular_value_16": float(singular[15]),
        "singular_value_17": float(singular[16]),
        "stage1_records": len(records),
        "source_bin_row_count_min_max": row_count_ranges,
        "memory_samples": len(memory), "memory_peak_driver_gib": peak / 2**30,
        "memory_over_limit_samples": over,
    }
    if stage1_only:
        if paths["smoke.json"].is_file():
            smoke = json.loads(paths["smoke.json"].read_text())
            check(smoke["stage1_sha256"] == result["stage1_sha256"],
                  "smoke_stage1_hash", errors)
            check(smoke["memory_sha256"] == sha(paths["memory.jsonl"]),
                  "smoke_memory_hash", errors)
        result["passed"] = not errors
        return result

    checkpoint = torch.load(paths["checkpoint.pt"], map_location="cpu", weights_only=True)
    lens = torch.load(paths["lens.pt"], map_location="cpu", weights_only=True)
    report = json.loads(paths["report.json"].read_text())
    progress = [json.loads(line) for line in paths["progress.jsonl"].read_text().splitlines()
                if line.strip()]
    phase_counts = Counter(row["phase"] for row in memory)
    expected_memory_phases = {"after_model_load", "after_stage1"}
    expected_memory_phases.update(f"stage1_after_case_{i}" for i in range(1, N_FIT + 1))
    expected_memory_phases.update(
        f"stage2_{case['case_id']}_coordinate_{k}"
        for case in cases for k in range(RANK)
    )
    check(all(phase_counts[phase] >= 1 for phase in expected_memory_phases),
          "memory_phase_coverage", errors)
    for name, saved in (("stage1", stage1), ("checkpoint", checkpoint),
                        ("lens", lens), ("report", report)):
        check(saved["metadata"] == metadata, f"metadata:{name}", errors)
    for key, path in (("stage1_sha256", paths["stage1.pt"]),
                      ("checkpoint_sha256", paths["checkpoint.pt"]),
                      ("lens_sha256", paths["lens.pt"]),
                      ("progress_sha256", paths["progress.jsonl"]),
                      ("memory_sha256", paths["memory.jsonl"])):
        check(report[key] == sha(path), f"report_hash:{key}", errors)
    check(lens["stage1_sha256"] == sha(paths["stage1.pt"]), "lens_stage1_hash", errors)
    check(lens["checkpoint_sha256"] == sha(paths["checkpoint.pt"]),
          "lens_checkpoint_hash", errors)
    check(lens["basis_sha256"] == tensor_sha(basis), "lens_basis_hash", errors)
    check(equal_tensor(lens["basis"], basis), "lens_basis", errors)
    check(equal_tensor(lens["singular_values"], singular), "lens_singular", errors)
    check(lens["n_fit"] == N_FIT and report["completed_fit_contexts"] == N_FIT,
          "completed_count", errors)
    check(lens["status"] == report["status"] ==
          "fit_only_corpus_transport_no_efficacy_evaluation", "status", errors)
    check(checkpoint["stage1_basis_sha256"] == tensor_sha(basis),
          "checkpoint_basis_hash", errors)
    check(checkpoint["completed_case_ids"] == [case["case_id"] for case in cases],
          "checkpoint_case_ids", errors)
    check(len(checkpoint["completed_case_rows"]) == len(progress) == N_FIT,
          "completed_rows_count", errors)
    check(checkpoint["completed_case_rows"] == progress,
          "checkpoint_progress_rows", errors)
    vjp_times: list[float] = []
    mode_gaps: list[float] = []
    for i, (row, record, case) in enumerate(zip(progress, records, cases)):
        check((row["case_id"], row["audio_sha256"]) ==
              (case["case_id"], case["audio_sha256"]), f"fit_row:{i}", errors)
        check(len(row["vjp_seconds"]) == RANK and
              all(math.isfinite(t) and t > 0 for t in row["vjp_seconds"]),
              f"vjp_timing:{i}", errors)
        vjp_times.extend(row["vjp_seconds"])
        mode_gaps.append(row["grad_vs_no_grad_terminal_max_abs"])
        check(math.isfinite(mode_gaps[-1]) and
              mode_gaps[-1] <= metadata["grad_vs_no_grad_terminal_max_abs_tolerance"],
              f"mode_gap:{i}", errors)
        for site in SITES:
            check(row["source_rows_per_bin"][site] ==
                  [len(group) for group in record["row_indices"][site]],
                  f"row_counts:{site}:{i}", errors)
    matrix_differences = {}
    matrix_nonzero = {}
    matrix_frobenius = {}
    for site, width in SITES.items():
        sums = checkpoint["jacobian_sums"][site]
        saved = lens["jacobians"][site]
        expected = (sums / N_FIT).float()
        check(finite_tensor(sums, (6, RANK, width)),
              f"jacobian_sums:{site}", errors)
        check(finite_tensor(saved, (6, RANK, width)),
              f"jacobian_map:{site}", errors)
        check(equal_tensor(saved, expected), f"jacobian_average:{site}", errors)
        check(report["matrix_shapes"][site] == [6, RANK, width],
              f"matrix_shape_report:{site}", errors)
        check(report["matrix_sha256"][site] == tensor_sha(saved),
              f"matrix_hash_report:{site}", errors)
        matrix_differences[site] = float((saved - expected).abs().max())
        matrix_nonzero[site] = int(torch.count_nonzero(saved))
        matrix_frobenius[site] = float(saved.norm())
        check(matrix_nonzero[site] > 0, f"zero_matrix:{site}", errors)
    for category in stage1["category_means"]:
        for kind in ("target",):
            check(equal_tensor(lens["category_means"][category][kind],
                               stage1["category_means"][category][kind]),
                  f"lens_category_target:{category}", errors)
        for site in SITES:
            check(equal_tensor(lens["category_means"][category]["sources"][site],
                               stage1["category_means"][category]["sources"][site]),
                  f"lens_category_source:{category}:{site}", errors)
    result.update({
        "passed": not errors, "errors": errors,
        "checkpoint_sha256": sha(paths["checkpoint.pt"]),
        "lens_sha256": sha(paths["lens.pt"]),
        "report_sha256": sha(paths["report.json"]),
        "completed_fit_contexts": len(progress),
        "vjp_time_samples": len(vjp_times),
        "vjp_time_sum_seconds": sum(vjp_times),
        "vjp_time_max_seconds": max(vjp_times, default=float("nan")),
        "memory_expected_phases_present": sum(phase_counts[phase] >= 1
                                              for phase in expected_memory_phases),
        "memory_expected_phases": len(expected_memory_phases),
        "mode_gap_max_abs": max(mode_gaps, default=float("nan")),
        "matrix_reconstruction_max_abs": matrix_differences,
        "matrix_nonzero_elements": matrix_nonzero,
        "matrix_frobenius_norm": matrix_frobenius,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "fit_rank16_transport_v1")
    parser.add_argument("--stage1-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.output_prefix.resolve(), stage1_only=args.stage1_only)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
