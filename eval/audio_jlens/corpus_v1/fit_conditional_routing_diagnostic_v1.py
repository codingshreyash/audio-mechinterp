"""CPU-only, FIT-only pair-excluded routing of saved Qwen local Jacobians.

The analysis is fixed by FIT_CONDITIONAL_ROUTING_DIAGNOSTIC_V1.md. No model
weights, WAVs, development answers, or locked-test inputs are opened. This is
an exploratory diagnostic on the existing FIT basis, not a transfer result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "FIT_CONDITIONAL_ROUTING_DIAGNOSTIC_V1.md"
EXPECTED_PROTOCOL_SHA = "e69bb992a753df568c77e4c56d8444f543c24d9b93e2a0af0d6f0241d99e211d"
FILES = {
    "stage1": HERE / "fit_rank16_transport_v1.stage1.pt",
    "heterogeneity": HERE / "fit_jacobian_heterogeneity_v1.result.pt",
    "calibration": HERE / "fit_uniform_calibration_v1.raw.pt",
}
EXPECTED = {
    "stage1": "8276aca0e24fcaa1397693e0353f9c7ee59190036f3516c0c13ebcbc6208b641",
    "heterogeneity": "336a2589743c06a2e5be71f3fce694e7a01c78450dc81dc07c6368a01bc58903",
    "calibration": "76085f0dab8f37b6052024b582ff34533cac3d1c579b27b39a631c536d9cd110",
}
SITES = ("encoder31", "connector")
WIDTHS = {"encoder31": 1280, "connector": 2048}
ALPHA = 0.025
N_CASES = 48
N_PAIRS = 12
RANK = 16
PCA_DIM = 8
SHRINKAGE = 0.5
BOOTSTRAP_REPS = 10_000
MAX_SECONDS = 20 * 60
MAX_RSS_BYTES = 16 * 1024**3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def max_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def guard(start: float) -> None:
    if time.monotonic() - start >= MAX_SECONDS:
        raise TimeoutError("20-minute CPU diagnostic cap reached")
    if max_rss_bytes() >= MAX_RSS_BYTES:
        raise MemoryError("16-GiB CPU diagnostic RSS cap reached")


def case_parts(case_id: str) -> tuple[str, str, str, str]:
    parts = tuple(case_id.split("/"))
    check(len(parts) == 4 and all(parts), f"malformed FIT case ID: {case_id}")
    return parts  # type: ignore[return-value]


def pair_and_slot(case_ids: list[str]) -> tuple[list[str], dict[str, list[int]], dict[int, int]]:
    by_pair: dict[str, list[int]] = {}
    parsed = [case_parts(item) for item in case_ids]
    for i, (pair, _template, _engine, _word) in enumerate(parsed):
        by_pair.setdefault(pair, []).append(i)
    check(len(by_pair) == N_PAIRS and all(len(v) == 4 for v in by_pair.values()),
          "FIT population must contain 12 pairs x four contexts")
    templates = {template for _pair, template, _engine, _word in parsed}
    check(len(templates) == 1, "FIT template changed")
    pair_slots: dict[str, dict[tuple[str, int], int]] = {}
    for pair, indices in by_pair.items():
        words = sorted({parsed[i][3] for i in indices})
        engines = sorted({parsed[i][2] for i in indices})
        check(len(words) == 2 and len(engines) == 2, f"FIT pair imbalance: {pair}")
        slots = {(parsed[i][2], words.index(parsed[i][3])): i for i in indices}
        check(len(slots) == 4, f"FIT pair duplicate slot: {pair}")
        pair_slots[pair] = slots
    pairs = sorted(by_pair)
    wrong_query: dict[int, int] = {}
    for k, pair in enumerate(pairs):
        next_pair = pairs[(k + 1) % len(pairs)]
        for slot, i in pair_slots[pair].items():
            wrong_query[i] = pair_slots[next_pair][slot]
    check(len(wrong_query) == N_CASES and all(case_ids[i] != case_ids[j]
                                               for i, j in wrong_query.items()),
          "wrong-router assignment invalid")
    return pairs, by_pair, wrong_query


def load_pinned() -> dict:
    for key, path in FILES.items():
        check(path.is_file() and sha(path) == EXPECTED[key], f"pinned FIT artifact changed: {key}")
    stage1 = torch.load(FILES["stage1"], map_location="cpu", weights_only=True)
    hetero = torch.load(FILES["heterogeneity"], map_location="cpu", weights_only=True)
    cal = torch.load(FILES["calibration"], map_location="cpu", weights_only=True)
    check(sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA, "FIT routing protocol changed")
    check(hetero["metadata"]["stage1_sha256"] == EXPECTED["stage1"] and
          hetero["metadata"]["calibration_sha256"] == EXPECTED["calibration"] and
          hetero["metadata"]["alpha"] == ALPHA and
          hetero["metadata"]["rank"] == RANK and
          hetero["metadata"]["planned_fit_contexts"] == N_CASES and
          cal["metadata"]["stage1_sha256"] == EXPECTED["stage1"] and
          cal["metadata"]["alpha"] == ALPHA,
          "FIT artifact provenance mismatch")
    sr, hr, cr = stage1["records"], hetero["records"], cal["records"]
    check(len(sr) == len(hr) == len(cr) == N_CASES, "expected 48 FIT contexts")
    case_ids = [record["case_id"] for record in sr]
    check(case_ids == hetero["metadata"]["case_ids"] and
          case_ids == [r["case_id"] for r in hr] == [r["case_id"] for r in cr] and
          len(set(case_ids)) == N_CASES, "FIT case order or identity changed")
    for s, h, c in zip(sr, hr, cr, strict=True):
        check(s["audio_sha256"] == h["audio_sha256"] == c["audio_sha256"],
              "FIT audio hash or case order changed")
        check(c["pair_id"] == case_parts(s["case_id"])[0], "FIT pair ID changed")
    pairs, by_pair, wrong_query = pair_and_slot(case_ids)
    basis = stage1["target_basis"].double()
    check(tuple(basis.shape) == (2048, RANK) and torch.isfinite(basis).all().item(),
          "target basis invalid")
    gram_err = float((basis.T @ basis - torch.eye(RANK, dtype=torch.float64)).abs().max())
    check(gram_err < 1e-3, "target basis is not orthonormal")
    site_data = {}
    for site in SITES:
        width = WIDTHS[site]
        features = stage1["source_bin_means"][site]
        check(tuple(features.shape) == (N_CASES, 6, width) and
              torch.isfinite(features).all().item(), f"source features invalid: {site}")
        matrices = []
        deltas = []
        actuals = []
        for i, (h, c) in enumerate(zip(hr, cr, strict=True)):
            j = h["jacobians"][site]
            d = c["sites"][site]["delta_bins"]
            t = c["sites"][site]["terminal_delta_per_alpha"]
            check(tuple(j.shape) == (6, RANK, width) and
                  tuple(d.shape) == (6, width) and tuple(t.shape) == (2048,),
                  f"saved FIT tensor shape changed: {site}/{i}")
            check(torch.isfinite(j).all().item() and torch.isfinite(d).all().item() and
                  torch.isfinite(t).all().item(), f"nonfinite FIT tensor: {site}/{i}")
            matrices.append(j.float())
            deltas.append(d.float())
            actuals.append(t.float() * ALPHA)
        site_data[site] = {
            "features": features.float(),
            "jacobians": torch.stack(matrices),
            "deltas": torch.stack(deltas),
            "actuals": torch.stack(actuals),
        }
    return {"case_ids": case_ids, "pairs": pairs, "by_pair": by_pair,
            "wrong_query": wrong_query, "basis": basis, "sites": site_data,
            "basis_gram_max_abs": gram_err}


def fit_fold_features(features: np.ndarray, train: np.ndarray, pair_ids: list[str]
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit z-score/PCA8/bandwidth exclusively on the outer training pairs."""
    x = features.reshape(features.shape[0], -1).astype(np.float64, copy=False)
    x_train = x[train]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    z_train = (x_train - mean) / std
    _u, singular, vt = np.linalg.svd(z_train, full_matrices=False)
    check(singular[0] > 0 and np.isfinite(singular).all() and
          int(np.sum(singular / singular[0] > 1e-6)) >= PCA_DIM,
          "training-only PCA rank below eight")
    components = vt[:PCA_DIM].copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    phi_train = z_train @ components.T
    phi_all = ((x - mean) / std) @ components.T
    check(np.isfinite(phi_all).all(), "nonfinite projected source feature")
    distances = []
    for a in range(len(train)):
        for b in range(a + 1, len(train)):
            if pair_ids[int(train[a])] != pair_ids[int(train[b])]:
                distance = float(np.linalg.norm(phi_train[a] - phi_train[b]))
                if distance > 0:
                    distances.append(distance)
    check(bool(distances), "no positive between-pair training distance")
    bandwidth = float(np.median(np.array(distances)))
    check(math.isfinite(bandwidth) and bandwidth > 0, "invalid Gaussian bandwidth")
    return phi_all, phi_train, bandwidth


def gaussian_weights(query: np.ndarray, train: np.ndarray, bandwidth: float) -> np.ndarray:
    check(math.isfinite(bandwidth) and bandwidth > 0, "invalid bandwidth")
    distance2 = np.square(train - query).sum(axis=1)
    exponents = -distance2 / (2.0 * bandwidth**2)
    weights = np.exp(exponents - exponents.max())
    total = float(weights.sum())
    check(math.isfinite(total) and total > 0, "zero/nonfinite routing weight sum")
    weights /= total
    check(np.isfinite(weights).all() and abs(weights.sum() - 1) < 1e-10,
          "routing weights invalid")
    return weights


def case_errors(pred: torch.Tensor, target: torch.Tensor, actual: torch.Tensor,
                basis: torch.Tensor) -> dict:
    check(tuple(pred.shape) == (RANK,) and tuple(target.shape) == (RANK,),
          "projected prediction/target shape invalid")
    p = pred.double()
    t = target.double()
    a = actual.double()
    projected_l2 = float(torch.linalg.vector_norm(p - t))
    full_l2 = float(torch.linalg.vector_norm(basis @ p - a))
    check(math.isfinite(projected_l2) and math.isfinite(full_l2),
          "nonfinite projected/full error")
    return {"projected_terminal_l2": projected_l2, "full_terminal_l2": full_l2}


def response_coordinates(jac: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Return [query, derivative-source, rank] uniform-edit predictions."""
    n, bins, rank, width = jac.shape
    check(tuple(delta.shape) == (n, bins, width) and rank == RANK,
          "Jacobian/edit geometry mismatch")
    flattened = jac.permute(0, 2, 1, 3).contiguous().reshape(n * rank, bins * width)
    products = ALPHA * (flattened @ delta.reshape(n, bins * width).T)
    out = products.reshape(n, rank, n).permute(2, 0, 1).double().contiguous()
    check(torch.isfinite(out).all().item(), "nonfinite Jacobian/edit responses")
    return out


def evaluate(data: dict, start: float) -> dict:
    basis: torch.Tensor = data["basis"]
    case_ids: list[str] = data["case_ids"]
    pair_ids = [case_parts(item)[0] for item in case_ids]
    pairs: list[str] = data["pairs"]
    by_pair: dict[str, list[int]] = data["by_pair"]
    wrong_query: dict[int, int] = data["wrong_query"]
    results = {}
    for site in SITES:
        guard(start)
        item = data["sites"][site]
        features = item["features"].numpy()
        jac = item["jacobians"]
        delta = item["deltas"]
        actual = item["actuals"].double()
        target = actual @ basis
        # Query i, derivative source j, projected target coordinate k.
        response = response_coordinates(jac, delta)
        check(tuple(response.shape) == (N_CASES, N_CASES, RANK) and
              torch.isfinite(response).all().item(), f"nonfinite responses: {site}")
        rows: list[dict] = []
        for pair in pairs:
            guard(start)
            held = set(by_pair[pair])
            train = np.array([i for i in range(N_CASES) if i not in held], dtype=int)
            check(len(train) == 44 and all(pair_ids[i] != pair for i in train),
                  "pair leaked into derivative pool")
            phi_all, phi_train, bandwidth = fit_fold_features(features, train, pair_ids)
            for i in by_pair[pair]:
                guard(start)
                weights = gaussian_weights(phi_all[i], phi_train, bandwidth)
                wrong_weights = gaussian_weights(phi_all[wrong_query[i]], phi_train, bandwidth)
                candidate_weights = torch.from_numpy(weights).double()
                wrong_tweights = torch.from_numpy(wrong_weights).double()
                responses = response[i, train, :]
                mean_coords = responses.mean(0)
                routed_coords = candidate_weights @ responses
                wrong_coords = wrong_tweights @ responses
                predictions = {
                    "conditional": SHRINKAGE * mean_coords + SHRINKAGE * routed_coords,
                    "global_pair_excluded": mean_coords,
                    "zero": torch.zeros(RANK, dtype=torch.float64),
                    "local_in_sample_oracle": response[i, i],
                    "wrong_router": SHRINKAGE * mean_coords + SHRINKAGE * wrong_coords,
                }
                errors = {name: case_errors(pred, target[i], actual[i], basis)
                          for name, pred in predictions.items()}
                cap = float(torch.sum(target[i] ** 2) / torch.sum(actual[i] ** 2))
                check(math.isfinite(cap) and 0 <= cap <= 1.001,
                      "rank-16 target capture invalid")
                rows.append({"case_id": case_ids[i], "pair_id": pair,
                             "wrong_query_case_id": case_ids[wrong_query[i]],
                             "bandwidth": bandwidth,
                             "rank16_capture_fraction": cap, "errors": errors})
        check(len(rows) == N_CASES and len({r["case_id"] for r in rows}) == N_CASES,
              f"site {site} missing/duplicate case")
        rows.sort(key=lambda r: case_ids.index(r["case_id"]))
        summaries = {}
        for name in ("conditional", "global_pair_excluded", "zero",
                     "local_in_sample_oracle", "wrong_router"):
            projected = [r["errors"][name]["projected_terminal_l2"] for r in rows]
            full = [r["errors"][name]["full_terminal_l2"] for r in rows]
            summaries[name] = {"mean_projected_terminal_l2": float(np.mean(projected)),
                               "median_projected_terminal_l2": float(np.median(projected)),
                               "mean_full_terminal_l2": float(np.mean(full)),
                               "median_full_terminal_l2": float(np.median(full))}
        results[site] = {"rows": rows, "summary": summaries,
                         "mean_rank16_capture_fraction": float(np.mean(
                             [r["rank16_capture_fraction"] for r in rows]))}
    seed_material = "".join(EXPECTED[k] for k in ("stage1", "heterogeneity", "calibration"))
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    draws = np.random.default_rng(seed).integers(0, N_PAIRS, size=(BOOTSTRAP_REPS, N_PAIRS))
    for site in SITES:
        guard(start)
        rows = results[site]["rows"]
        ordered = np.array([[next(r for r in rows if r["case_id"] == case_ids[i])
                             for i in by_pair[pair]] for pair in pairs], dtype=object)
        cond = np.array([[r["errors"]["conditional"]["projected_terminal_l2"]
                          for r in group] for group in ordered], dtype=np.float64).mean(1)
        glob = np.array([[r["errors"]["global_pair_excluded"]["projected_terminal_l2"]
                          for r in group] for group in ordered], dtype=np.float64).mean(1)
        check(np.all(glob > 0), "undefined global comparator error")
        ratios = cond[draws].mean(axis=1) / glob[draws].mean(axis=1)
        check(np.isfinite(ratios).all(), "nonfinite bootstrap ratio")
        mean_ratio = float(cond.mean() / glob.mean())
        results[site]["conditional_over_global_projected_l2"] = mean_ratio
        results[site]["pair_bootstrap_95_percentile_ratio"] = [
            float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
    return results | {"bootstrap_seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, help="new JSON result path for --run")
    args = parser.parse_args()
    started = time.monotonic()
    data = load_pinned()
    guard(started)
    metadata = {"status": "fit_only_conditional_router_diagnostic_v1",
                "script_sha256": sha(Path(__file__)), "protocol_sha256": sha(PROTOCOL),
                "input_sha256": EXPECTED, "case_count": N_CASES, "pair_count": N_PAIRS,
                "site_shapes": {site: {"source_features": list(data["sites"][site]["features"].shape),
                                       "jacobians": list(data["sites"][site]["jacobians"].shape)}
                                for site in SITES},
                "rank": RANK, "alpha": ALPHA, "pca_components": PCA_DIM,
                "lambda": SHRINKAGE, "bootstrap_reps": BOOTSTRAP_REPS,
                "max_seconds": MAX_SECONDS, "max_rss_bytes": MAX_RSS_BYTES,
                "basis_gram_max_abs": data["basis_gram_max_abs"]}
    if args.validate_only:
        print(json.dumps({"status": "validated_inputs_only_no_outcomes", "metadata": metadata}, indent=2))
        return
    check(args.output is not None, "--run requires --output")
    check(not args.output.exists(), "output path already exists")
    results = evaluate(data, started)
    guard(started)
    out = {"metadata": metadata, "results": results,
           "elapsed_seconds": time.monotonic() - started,
           "peak_rss_bytes": max_rss_bytes(),
           "limits": "FIT-only projected-terminal diagnostic; basis includes all FIT pairs; no nonlinear score/answer/security claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete_fit_only_diagnostic", "output": str(args.output),
                      "elapsed_seconds": out["elapsed_seconds"],
                      "peak_rss_bytes": out["peak_rss_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
