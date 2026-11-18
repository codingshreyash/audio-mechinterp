"""Read-only CPU audit of the frozen old-FIT numerical follow-up.

This never imports a model or opens the fresh v2 corpus. A successful audit
checks numerical and receipt consistency, not intervention efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "voxtral_v1_fit_backward_numerical_followup_v1.py"
PROTOCOL = HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_FOLLOWUP_PROTOCOL_V1.md"
OLD = HERE / "runs/voxtral_v1_fit_backward_feasibility/20261118T1122Z"
PINS = {
    "runner": "ae9d2620ad4e06d01ea01c920f1f8f4aee8c21764598bac343e383869c2940d2",
    "protocol": "5778a4b84e535f7182a3daa2b973637a1c9127118ce2434ce82ba80fee467a82",
    "old_meta": "f151ced4e4e830d041e1f8e109eb3b37a05e57ba8b2e38a4ff06914334603558",
    "old_result": "d4ccfd213cf3a5ebebd3a7c0ecebab691dc15632f61076669d7535bd58154ebe",
    "old_details": "1bc49aa650b96c7ac6cb2f545da98e198edfa18998ebae8dc263c70ad5ad7b3c",
}
SOURCE_HASHES = {
    "modeling_voxtral": "2d3ec9d78c469b77c42a8d5ba258185fb1c58f4c2c7bf6007ec724436dfc7056",
    "processing_voxtral": "239e84657d1a0287ab27a540c946a787a69b4b0bf45727d6b24bee21ccec0296",
    "feature_extraction_whisper": "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb",
    "modeling_llama": "affc6a4ab36b562a6b4c90b84c4e7939c5417a69ad7bd37e858be5583f189830",
}
SITES = ("encoder31", "projector")
ROWS = {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]}
WIDTH = {"encoder31": 1280, "projector": 3072}
ORDER = (0.00625, -0.00625, 0.025, -0.025, 0.1, -0.1)
CYCLES = (ORDER, tuple(reversed(ORDER)), ORDER)
CAP = {"bf16": 900, "fp32": 1200}
MEMORY_CAP = 30 * 1024**3


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def json_read(path: Path) -> dict:
    need(path.is_file(), f"missing {path.name}")
    obj = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    need(isinstance(obj, dict), f"{path.name} is not an object")
    finite_tree(obj, path.name)
    return obj


def finite_tree(obj, label: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            finite_tree(value, f"{label}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            finite_tree(value, f"{label}[{i}]")
    elif isinstance(obj, float):
        need(math.isfinite(obj), f"nonfinite {label}")


def tensor(obj, shape: tuple[int, ...], dtype: torch.dtype, label: str) -> torch.Tensor:
    need(isinstance(obj, torch.Tensor) and obj.device.type == "cpu" and
         tuple(obj.shape) == shape and obj.dtype == dtype, f"{label} shape/dtype")
    need(bool(torch.isfinite(obj).all()), f"{label} nonfinite")
    return obj


def number(value, label: str) -> float:
    need(isinstance(value, (int, float)) and not isinstance(value, bool) and
         math.isfinite(float(value)), f"{label} missing/nonfinite")
    return float(value)


def compare(actual, expected, label: str) -> None:
    """Require all fields; allow only FP32 reduction-scale rounding differences."""
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"{label} keys")
        for key in expected:
            compare(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f"{label} length")
        for i, pair in enumerate(zip(actual, expected)):
            compare(*pair, f"{label}[{i}]")
    elif isinstance(expected, float):
        a, b = number(actual, label), number(expected, f"expected {label}")
        need(math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-7), f"{label} differs: {a} vs {b}")
    else:
        need(actual == expected, f"{label} differs")


def source_context() -> dict:
    files = {"runner": RUNNER, "protocol": PROTOCOL, "old_meta": OLD / "meta.json",
             "old_result": OLD / "result.json", "old_details": OLD / "details.pt"}
    for key, path in files.items():
        need(path.is_file() and sha(path) == PINS[key], f"pinned {key} drift")
    old_meta, old_result = json_read(files["old_meta"]), json_read(files["old_result"])
    need(old_meta["record_key"] == "animal_08|piper|fit_chosen" and
         old_meta["source_rows"] == ROWS and old_meta["fresh_v2_corpus_opened"] is False and
         old_meta["model_snapshot"]["revision"] == "3060fe34b35ba5d44202ce9ff3c097642914f8f3" and
         set(old_meta["wav_hashes"]) == {"beaver", "monkey"}, "old FIT binding")
    need(old_result["status"] == "completed_numerical_feasibility_only" and
         old_result["details_sha256"] == PINS["old_details"], "old FIT result binding")
    raw = torch.load(files["old_details"], map_location="cpu", weights_only=True)
    directions = {}
    for site in SITES:
        shape = (1500, 1280) if site == "encoder31" else (375, 3072)
        rec = tensor(raw["source_states"][site], shape, torch.bfloat16, f"old {site} recipient")
        donor = tensor(raw["donor_states"][site], shape, torch.bfloat16, f"old {site} donor")
        need(tensor_sha(rec) == old_result["source_hashes"][site] and
             tensor_sha(donor) == old_result["donor_hashes"][site], f"old {site} hash")
        delta = donor[ROWS[site]].float().mean(0) - rec[ROWS[site]].float().mean(0)
        need(bool(torch.isfinite(delta).all()) and float(torch.linalg.vector_norm(delta)) > 0,
             f"old {site} direction invalid")
        directions[site] = delta.contiguous()
    return {"meta": old_meta, "result": old_result, "directions": directions}


def progress_expected() -> list[str]:
    out = ["attempt_started", "preflight_validated", "mps_ready", "before_model_load",
           "model_loaded", "natural_generation", "passive_generation", "donor_capture",
           "no_grad_clean_0", "no_grad_clean_1", "no_grad_clean_2", "clean_shared_graph"]
    for coordinate in range(4):
        out += [f"before_vjp_{coordinate}", f"after_vjp_{coordinate}"]
    for site in SITES:
        for cycle, order in enumerate(CYCLES):
            for alpha in order:
                for label in ("zero", "alpha"):
                    out += [f"before_{site}_{cycle}_{label}_{alpha}",
                            f"after_{site}_{cycle}_{label}_{alpha}"]
    return out + ["details_saved"]


def progress(run_dir: Path, arm: str) -> list[dict]:
    path = run_dir / "progress.jsonl"
    need(path.is_file(), "missing progress receipt")
    lines = path.read_text().splitlines()
    need(bool(lines), "empty progress receipt")
    entries = [json.loads(line) for line in lines]
    finite_tree(entries, "progress")
    need(all(isinstance(e, dict) for e in entries), "invalid progress entry")
    first = entries[0]
    need(first.get("phase") == "attempt_started" and first.get("arm") == arm and
         first.get("script_sha256") == PINS["runner"] and
         first.get("protocol_sha256") == PINS["protocol"] and
         isinstance(first.get("wall_utc"), str), "attempt start binding")
    previous = -1.0
    for e in entries:
        elapsed = number(e.get("elapsed_seconds"), "progress elapsed")
        need(previous <= elapsed < CAP[arm], "progress elapsed/cap")
        previous = elapsed
    return entries


def events(value, dtype: torch.dtype, label: str) -> None:
    expected = {"encoder31": [1, 1500, 1280], "projector": [375, 3072],
                "lm29": [1, 396, 3072]}
    need(isinstance(value, dict) and set(value) == set(expected), f"{label} sites")
    for site, shape in expected.items():
        need(value[site] == {"calls": 1, "shape": shape, "dtype": str(dtype)},
             f"{label}/{site} hook receipt")


def ulps(values: torch.Tensor) -> list[float]:
    above = torch.nextafter(values, torch.full_like(values, float("inf")))
    below = torch.nextafter(values, torch.full_like(values, float("-inf")))
    return torch.maximum((above.float() - values.float()).abs(),
                         (values.float() - below.float()).abs()).tolist()


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.double()))


def metrics_for_site(records: list[dict], gradients: torch.Tensor) -> dict:
    zeros = torch.stack([r["zero_terminal"] for r in records]).double()
    noise = max(norm(zeros[i, :4] - zeros[j, :4]) for i in range(18) for j in range(i + 1, 18))
    noise_full = max(norm(zeros[i] - zeros[j]) for i in range(18) for j in range(i + 1, 18))
    groups = {}
    for alpha in ORDER:
        selected = [r for r in records if r["alpha"] == alpha]
        need(len(selected) == 3 and {r["cycle"] for r in selected} == {0, 1, 2}, "repeat grid")
        measured = torch.stack([(r["alpha_terminal"] - r["zero_terminal"])[:4] for r in selected]).double()
        pred = {kind: torch.tensor([r["prediction"][kind] for r in selected], dtype=torch.float64)
                for kind in ("rowwise", "uniform", "ideal")}
        median = {kind: p.median(0).values for kind, p in pred.items()}
        pm, dm = median["rowwise"], measured.median(0).values
        pn, dn = norm(pm), norm(dm)
        repeat_rows = []
        for i, row in enumerate(selected):
            d, p = measured[i], pred["rowwise"][i]
            repeat_rows.append({"cycle": row["cycle"], "paired_delta_four": d.tolist(),
                                "rowwise_prediction_four": p.tolist(),
                                "uniform_prediction_four": row["prediction"]["uniform"],
                                "ideal_prediction_four": row["prediction"]["ideal"],
                                "absolute_l2_error": norm(p - d), "measured_l2": norm(d),
                                "uniform_absolute_l2_error": norm(pred["uniform"][i] - d),
                                "ideal_absolute_l2_error": norm(pred["ideal"][i] - d),
                                "edit_metrics": row["edit_metrics"],
                                "zero_terminal_sha256": row["zero_sha"],
                                "alpha_terminal_sha256": row["alpha_sha"],
                                "zero_source_sha256": row["zero_source_sha"],
                                "alpha_source_sha256": row["alpha_source_sha"],
                                "zero_terminal_ulp_four": row["zero_ulp_four"],
                                "measured_change_in_ulp_four":
                                [float(x / u) if u > 0 else None for x, u in
                                 zip(d.tolist(), row["zero_ulp_four"])]})
        err = norm(pm - dm)
        cosine = float(torch.dot(pm, dm) / (pn * dn)) if pn > max(3 * noise, 1e-6) and dn > max(3 * noise, 1e-6) else None
        groups[str(alpha)] = {"per_repeat": repeat_rows,
                              "median_prediction_four": pm.tolist(),
                              "median_uniform_prediction_four": median["uniform"].tolist(),
                              "median_ideal_prediction_four": median["ideal"].tolist(),
                              "median_measured_four": dm.tolist(),
                              "median_absolute_l2_error": err,
                              "median_relative_l2_error": err / max(dn, noise, 1e-6),
                              "median_uniform_absolute_l2_error": norm(median["uniform"] - dm),
                              "median_uniform_relative_l2_error": norm(median["uniform"] - dm) / max(dn, noise, 1e-6),
                              "median_ideal_absolute_l2_error": norm(median["ideal"] - dm),
                              "median_ideal_relative_l2_error": norm(median["ideal"] - dm) / max(dn, noise, 1e-6),
                              "median_measured_l2": dn,
                              "below_repeatability_resolution": dn < 3 * noise,
                              "zero_measured_signal": dn == 0,
                              "cosine_if_resolved": cosine}
    odd_even = {}
    for h in (0.00625, 0.025, 0.1):
        plus, minus = groups[str(h)], groups[str(-h)]
        dp = torch.tensor(plus["median_measured_four"], dtype=torch.float32)
        dn = torch.tensor(minus["median_measured_four"], dtype=torch.float32)
        pp = torch.tensor(plus["median_prediction_four"], dtype=torch.float32)
        pn = torch.tensor(minus["median_prediction_four"], dtype=torch.float32)
        odd_even[str(h)] = {"measured_odd_four": ((dp - dn) / 2).tolist(),
                            "measured_even_four": ((dp + dn) / 2).tolist(),
                            "predicted_rowwise_odd_four": ((pp - pn) / 2).tolist()}
    return {"zero_repeatability_l2_four": noise, "zero_repeatability_l2_full": noise_full,
            "alpha_groups": groups, "odd_even": odd_even}


def audit_completed(run_dir: Path, arm: str, ctx: dict, entries: list[dict]) -> dict:
    need({p.name for p in run_dir.iterdir()} == {"progress.jsonl", "meta.json", "details.pt", "result.json"},
         "completed file set")
    need([e["phase"] for e in entries] == progress_expected(), "completed phase sequence")
    patch_index = 0
    observed_peak = 0.0
    observed_current_peak = 0.0
    for i, e in enumerate(entries):
        is_patch_end = e["phase"].startswith("after_") and (
            "_zero_" in e["phase"] or "_alpha_" in e["phase"])
        need(("patch_count" in e) == is_patch_end, "patch count placement")
        if is_patch_end:
            patch_index += 1
            need(e["patch_count"] == patch_index, "patch count sequence")
        if i >= 3:
            need(0 <= number(e["driver_bytes"], "driver") < MEMORY_CAP and
                 0 <= number(e["current_bytes"], "current") and
                 0 <= number(e["peak_driver_bytes"], "peak driver") < MEMORY_CAP,
                 "progress memory cap")
            observed_peak = max(observed_peak, number(e["peak_driver_bytes"], "peak driver"))
            observed_current_peak = max(observed_current_peak, number(e["current_bytes"], "current"))
    need(patch_index == 72, "patch count total")
    meta, result = json_read(run_dir / "meta.json"), json_read(run_dir / "result.json")
    for key, name in (("meta_sha256", "meta.json"), ("progress_sha256", "progress.jsonl"),
                      ("details_sha256", "details.pt")):
        need(result[key] == sha(run_dir / name), f"{name} result hash")
    need(result["status"] == "completed_old_fit_numerical_diagnostic_only" and
         result["arm"] == arm and result["scope"] == "one old v1 FIT case; no v2 transfer or efficacy",
         "result status/scope")
    need(meta["arm"] == arm and meta["purpose"] == "old_fit_numerical_diagnostic_only" and
         meta["script_sha256"] == PINS["runner"] and meta["protocol_sha256"] == PINS["protocol"] and
         meta["old_result_sha256"] == PINS["old_result"] and
         meta["old_details_sha256"] == PINS["old_details"] and
         meta["source_hashes"] == SOURCE_HASHES and meta["rows"] == ROWS and
         meta["wav_hashes"] == ctx["meta"]["wav_hashes"] and
         meta["model_snapshot"] == ctx["meta"]["model_snapshot"] and
         meta["fresh_v2_corpus_opened"] is False, "meta provenance")
    need(meta["direction_hashes"] == {s: tensor_sha(ctx["directions"][s]) for s in SITES},
         "frozen direction hashes")
    need(meta["max_seconds"] == CAP[arm] and meta["max_mps_driver_bytes"] == MEMORY_CAP and
         meta["planned_patch_forwards"] == 72 and meta["planned_pairs"] == 36 and
         meta["planned_other_prompt_forwards"] == 5 and meta["planned_vjps"] == 4 and
         meta["planned_generations"] == 2 and
         0 < number(meta["mps_fraction_limit"], "mps fraction") <= 1, "meta workload/caps")
    need(result["patch_forward_count"] == 72 and result["zero_nonzero_pair_count"] == 36 and
         result["paired_observations_per_site"] == 18 and
         result["selected_gradient_storage_dtype"] == "torch.float32" and
         result["source_element_unchanged_label"] ==
         ("unchanged_by_bf16_cast" if arm == "bf16" else "unchanged_by_fp32_addition_rounding"),
         "result workload/dtype labels")
    need(result["natural_suffix_ids"] == result["passive_suffix_ids"] and
         isinstance(result["natural_suffix_ids"], list) and
         len(result["natural_suffix_ids"]) <= 32 and
         (arm != "bf16" or result["natural_suffix_ids"] == ctx["result"]["natural_suffix_ids"]),
         "natural/passive suffix parity")
    need(result["passive_hook_shapes"]["encoder31"] == [[1, 1500, 1280]] and
         result["passive_hook_shapes"]["projector"] == [[375, 3072]] and
         result["passive_hook_shapes"]["lm29"][0] == [1, 396, 3072], "passive hook shapes")
    dtype = torch.bfloat16 if arm == "bf16" else torch.float32
    events(result["clean_hook_events"], dtype, "clean")
    need(result["raw_vjp_dtypes"] == {s: [str(dtype)] * 4 for s in SITES}, "raw VJP dtype")
    need(0 <= number(result["elapsed_seconds"], "elapsed") < CAP[arm] and
         number(result["elapsed_seconds"], "elapsed") >= entries[-1]["elapsed_seconds"] and
         0 <= number(result["peak_mps_driver_bytes"], "peak") < MEMORY_CAP and
         number(result["peak_mps_driver_bytes"], "peak") >= observed_peak and
         number(result["peak_mps_current_bytes"], "current peak") >= observed_current_peak,
         "result resource bounds")
    raw = torch.load(run_dir / "details.pt", map_location="cpu", weights_only=True)
    need(isinstance(raw, dict) and set(raw) == {"directions", "selected_gradients_fp32",
         "clean_terminal_fp32", "donor_terminal_fp32", "no_grad_terminals_fp32", "observations"},
         "details fields")
    tensor(raw["clean_terminal_fp32"], (3072,), torch.float32, "clean terminal")
    tensor(raw["donor_terminal_fp32"], (3072,), torch.float32, "donor terminal")
    tensor(raw["no_grad_terminals_fp32"], (3, 3072), torch.float32, "no-grad terminals")
    need(tensor_sha(raw["clean_terminal_fp32"]) == result["clean_terminal_sha256"],
         "clean terminal hash")
    need(set(raw["directions"]) == set(SITES) and
         set(raw["selected_gradients_fp32"]) == set(SITES) and
         set(raw["observations"]) == set(SITES), "sidecar site set")
    recomputed = {}
    for site in SITES:
        direction = tensor(raw["directions"][site], (WIDTH[site],), torch.float32, site + " direction")
        need(torch.equal(direction, ctx["directions"][site]), site + " frozen direction")
        gradient = tensor(raw["selected_gradients_fp32"][site],
                          (4, len(ROWS[site]), WIDTH[site]), torch.float32, site + " gradients")
        need(tensor_sha(gradient) == result["gradient_hashes"][site] and
             bool(torch.linalg.vector_norm(gradient.reshape(4, -1), dim=1).gt(0).all()),
             site + " gradient hash/nonzero")
        records = raw["observations"][site]
        need(isinstance(records, list) and len(records) == 18, site + " observation count")
        for i, (cycle, alpha) in enumerate((c, a) for c, order in enumerate(CYCLES) for a in order):
            row = records[i]
            need(isinstance(row, dict) and set(row) == {
                "site", "cycle", "alpha", "zero_terminal", "alpha_terminal",
                "zero_source_rows", "alpha_source_rows", "effective_edit",
                "zero_sha", "alpha_sha", "zero_source_sha", "alpha_source_sha",
                "zero_ulp_four", "zero_hook_events", "alpha_hook_events",
                "prediction", "edit_metrics"}, f"{site} {i} observation fields")
            need(row["site"] == site and row["cycle"] == cycle and row["alpha"] == alpha,
                 site + " paired order")
            for phase in ("zero", "alpha"):
                terminal = tensor(row[phase + "_terminal"], (3072,), torch.float32,
                                  f"{site} {i} {phase} terminal")
                if arm == "bf16":
                    need(torch.equal(terminal, terminal.to(dtype).float()), "BF16 terminal roundtrip")
                need(tensor_sha(terminal.to(dtype)) == row[phase + "_sha"],
                     f"{site} {i} {phase} terminal hash")
                source = tensor(row[phase + "_source_rows"],
                                (len(ROWS[site]), WIDTH[site]), dtype,
                                f"{site} {i} {phase} source")
                need(tensor_sha(source) == row[phase + "_source_sha"],
                     f"{site} {i} {phase} source hash")
                events(row[phase + "_hook_events"], dtype, f"{site} {i} {phase}")
            source = row["alpha_source_rows"]
            edit = tensor(row["effective_edit"], (len(ROWS[site]), WIDTH[site]),
                          torch.float32, f"{site} {i} edit")
            expected_edit = (source.float() + alpha * direction).to(dtype).float() - source.float()
            need(torch.equal(edit, expected_edit), f"{site} {i} realized edit")
            need(torch.equal((row["zero_source_rows"].float() + 0 * direction).to(dtype),
                             row["zero_source_rows"]), "zero source edit")
            g = gradient
            mean, ideal = edit.mean(0), alpha * direction
            predicted = {"rowwise": (g * edit.unsqueeze(0)).sum((1, 2)).tolist(),
                         "uniform": (g.sum(1) * mean).sum(1).tolist(),
                         "ideal": (g.sum(1) * ideal).sum(1).tolist()}
            compare(row["prediction"], predicted, f"{site} {i} predictions")
            ideal_norm = norm(ideal)
            edit_metrics = {"effective_norm": float(torch.linalg.vector_norm(edit)),
                            "mean_fidelity_ratio": norm(mean - ideal) / max(ideal_norm, 1e-6),
                            "row_nonuniformity_ratio": norm(edit - mean) / math.sqrt(len(ROWS[site])) /
                            max(norm(mean), 1e-6),
                            "unchanged_fraction": float((edit == 0).float().mean())}
            compare(row["edit_metrics"], edit_metrics, f"{site} {i} edit metrics")
            compare(row["zero_ulp_four"], ulps(row["zero_terminal"][:4].to(dtype)),
                    f"{site} {i} ULP")
        recomputed[site] = metrics_for_site(records, gradient)
    compare(result["analysis"], recomputed, "analysis")
    for site in SITES:
        for name in ("clean_source_hashes", "donor_source_hashes"):
            need(isinstance(result[name][site], str) and len(result[name][site]) == 64,
                 f"{name}/{site} receipt")
    return {"status": "audited_completed_numerical_consistency_only", "arm": arm,
            "result_sha256": sha(run_dir / "result.json"),
            "details_sha256": sha(run_dir / "details.pt"),
            "auditor_sha256": sha(Path(__file__)),
            "runner_sha256": PINS["runner"], "protocol_sha256": PINS["protocol"],
            "pairs_verified": 36, "patch_forwards_verified": 72,
            "claim_limit": "one old-FIT case; no efficacy or v2 transfer"}


def audit(run_dir: Path, arm: str) -> dict:
    need(arm in CAP, "arm must be bf16 or fp32")
    ctx = source_context()
    need(run_dir.is_dir(), "missing run directory")
    entries = progress(run_dir, arm)
    completed = (run_dir / "result.json").exists()
    failed = (run_dir / "error.json").exists()
    need(completed != failed, "expected exactly one result or error receipt")
    if completed:
        return audit_completed(run_dir, arm, ctx, entries)
    error = json_read(run_dir / "error.json")
    need(error["status"] in ("resource_limited_arm", "invalid_old_fit_arm") and
         error["arm"] == arm and error["progress_sha256"] == sha(run_dir / "progress.jsonl") and
         isinstance(error["error_type"], str) and isinstance(error["error"], str) and
         number(error["elapsed_seconds"], "error elapsed") >= entries[-1]["elapsed_seconds"],
         "failed attempt receipt")
    return {"status": "audited_failed_attempt_receipt_only", "arm": arm,
            "error_status": error["status"], "progress_sha256": error["progress_sha256"],
            "auditor_sha256": sha(Path(__file__)), "runner_sha256": PINS["runner"],
            "protocol_sha256": PINS["protocol"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--arm", choices=tuple(CAP), required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.arm), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
