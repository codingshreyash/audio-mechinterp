"""Fit zero/global/six-bin comparators from frozen FIT uniform-edit data.

CPU only. No development/test audio, model activations, or answers are read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_pool_capability_v1 import sha  # noqa: E402
from fit_rank16_transport_v1 import atomic_json_save, atomic_torch_save  # noqa: E402


GAMMAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SITES = ("encoder31", "connector")
FEATURES = ("global", "six_bin")
PROTOCOL = HERE / "PRE_DEV_EVAL_PROTOCOL_V1.md"


def feature_matrix(records: list[dict], site: str, feature: str) -> torch.Tensor:
    bins = torch.stack([record["sites"][site]["delta_bins"] for record in records]).double()
    if bins.ndim != 3 or bins.shape[1] != 6 or not torch.isfinite(bins).all():
        raise ValueError("Invalid six-bin FIT inputs")
    if feature == "global":
        counts = torch.tensor([
            [len(group) for group in record["sites"][site]["source_rows"]]
            for record in records
        ], dtype=torch.float64)
        if counts.shape != (len(records), 6) or bool((counts <= 0).any()):
            raise ValueError("Invalid FIT recipient row counts")
        return (bins * counts[..., None]).sum(dim=1) / counts.sum(dim=1)[:, None]
    if feature == "six_bin":
        return bins.flatten(1)
    raise ValueError("Unknown FIT comparator feature")


def projected_target(records: list[dict], site: str, basis: torch.Tensor) -> torch.Tensor:
    terminal = torch.stack([
        record["sites"][site]["terminal_delta_per_alpha"] for record in records
    ]).double()
    if terminal.shape != (48, 2048) or basis.shape != (2048, 16):
        raise ValueError("FIT terminal/basis shape mismatch")
    if not torch.isfinite(terminal).all():
        raise ValueError("Nonfinite FIT terminal difference")
    return terminal @ basis.double()


def ridge_fit(train_x: torch.Tensor, train_y: torch.Tensor, gamma: float) -> dict:
    if (train_x.ndim != 2 or train_y.ndim != 2 or train_x.shape[0] != train_y.shape[0]
            or train_x.shape[0] < 2 or train_y.shape[1] != 16
            or not torch.isfinite(train_x).all() or not torch.isfinite(train_y).all()
            or gamma not in GAMMAS):
        raise ValueError("Invalid ridge FIT arrays or gamma")
    scale = float(train_x.square().mean().sqrt())
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Zero/nonfinite FIT feature RMS")
    z = train_x.double() / scale
    k = z @ z.T / z.shape[1]
    lambda_base = float(torch.trace(k) / len(k))
    if lambda_base <= 0 or not math.isfinite(lambda_base):
        raise ValueError("Zero/nonfinite FIT kernel trace")
    lam = gamma * lambda_base
    coeff = torch.linalg.solve(k + lam * torch.eye(len(k), dtype=torch.float64),
                               train_y.double())
    return {"train_z": z, "dual_coeff": coeff, "scale": scale,
            "lambda": lam, "gamma": gamma, "width": z.shape[1]}


def ridge_predict(fit: dict, test_x: torch.Tensor) -> torch.Tensor:
    if test_x.ndim != 2 or test_x.shape[1] != fit["width"]:
        raise ValueError("Ridge prediction feature width mismatch")
    z = test_x.double() / fit["scale"]
    return (z @ fit["train_z"].T / fit["width"]) @ fit["dual_coeff"]


def pair_cv(x: torch.Tensor, y: torch.Tensor, pair_ids: list[str]) -> tuple[dict, dict]:
    if len(pair_ids) != 48 or x.shape[0] != 48 or y.shape != (48, 16):
        raise ValueError("Expected 48 FIT comparator examples")
    pairs = list(dict.fromkeys(pair_ids))
    if len(pairs) != 12 or any(pair_ids.count(pair) != 4 for pair in pairs):
        raise ValueError("Expected 12 FIT pairs with four contexts each")
    table = {}
    for gamma in GAMMAS:
        errors = []
        for pair in pairs:
            held = torch.tensor([p == pair for p in pair_ids], dtype=torch.bool)
            fit = ridge_fit(x[~held], y[~held], gamma)
            prediction = ridge_predict(fit, x[held])
            errors.append(float((prediction - y[held]).square().mean()))
        table[str(gamma)] = {"pair_mse": errors,
                             "pair_mean_mse": sum(errors) / len(errors),
                             "pair_rmse": math.sqrt(sum(errors) / len(errors))}
    # GAMMAS is ascending, so min keeps the smaller gamma on an exact tie.
    winner = min(GAMMAS, key=lambda gamma: table[str(gamma)]["pair_mean_mse"])
    return ridge_fit(x, y, winner), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path,
                        default=HERE / "fit_uniform_calibration_v1.raw.pt")
    parser.add_argument("--lens", type=Path,
                        default=HERE / "fit_rank16_transport_v1.lens.pt")
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "fit_uniform_baselines_v1")
    args = parser.parse_args()
    calibration_path, lens_path = args.calibration.resolve(), args.lens.resolve()
    raw = torch.load(calibration_path, map_location="cpu", weights_only=True)
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    if (raw["metadata"]["protocol_sha256"] != sha(PROTOCOL)
            or raw["metadata"]["lens_sha256"] != sha(lens_path)
            or raw["summary"]["completed_fit_contexts"] != 48
            or lens["metadata"] != raw["metadata"]["fit_metadata"]):
        raise ValueError("FIT calibration/lens/protocol provenance mismatch")
    records = raw["records"]
    pair_ids = [record["pair_id"] for record in records]
    prefix = args.output_prefix.resolve()
    output = prefix.with_suffix(".pt")
    report = prefix.with_suffix(".json")
    if output.exists() or report.exists():
        raise FileExistsError("FIT comparator output exists")
    models, tables = {}, {}
    for site in SITES:
        y = projected_target(records, site, lens["basis"])
        models[site], tables[site] = {}, {}
        for feature in FEATURES:
            x = feature_matrix(records, site, feature)
            models[site][feature], tables[site][feature] = pair_cv(x, y, pair_ids)
        models[site]["zero"] = {"description": "zero terminal delta"}
    artifact = {
        "status": "fit_only_comparators_no_development_or_test",
        "metadata": {"script_sha256": sha(Path(__file__)),
                     "protocol_sha256": sha(PROTOCOL),
                     "calibration_sha256": sha(calibration_path),
                     "lens_sha256": sha(lens_path),
                     "gamma_grid": list(GAMMAS),
                     "alpha_trained": raw["metadata"]["alpha"],
                     "n_fit": 48, "n_fit_pairs": 12},
        "models": models, "cv_tables": tables,
        "numerical_floor_v2_by_site": raw["summary"]["numerical_floor_v2_by_site"],
    }
    atomic_torch_save(output, artifact)
    atomic_json_save(report, {
        "metadata": artifact["metadata"],
        "status": artifact["status"],
        "baseline_sha256": sha(output),
        "chosen_gamma": {s: {f: models[s][f]["gamma"] for f in FEATURES} for s in SITES},
        "fit_pair_cv_rmse": {s: {f: tables[s][f][str(models[s][f]["gamma"])]["pair_rmse"]
                                   for f in FEATURES} for s in SITES},
        "numerical_floor_v2_by_site": artifact["numerical_floor_v2_by_site"],
    })
    print(json.dumps({"status": artifact["status"],
                      "chosen_gamma": {s: {f: models[s][f]["gamma"]
                                          for f in FEATURES} for s in SITES}}, indent=2))


if __name__ == "__main__":
    main()
