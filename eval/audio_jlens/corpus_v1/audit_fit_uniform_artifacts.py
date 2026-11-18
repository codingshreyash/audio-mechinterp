"""CPU-only independent checks of saved FIT calibration and ridge artifacts."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

P = Path(__file__).resolve().parent
SITES = ("encoder31", "connector")
GAMMAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def sha(name):
    return hashlib.sha256((P / name).read_bytes()).hexdigest()


def close(a, b, tol=1e-6):
    assert abs(float(a) - float(b)) <= tol, (a, b)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def ridge(x, y, gamma):
    scale = x.square().mean().sqrt()
    z = x / scale
    kernel = z @ z.T / z.shape[1]
    lam = gamma * kernel.trace() / len(kernel)
    coeff = torch.linalg.solve(kernel + lam * torch.eye(len(kernel), dtype=torch.float64), y)
    return z, coeff, scale


def main():
    torch.set_num_threads(1)
    report = json.loads((P / "fit_uniform_calibration_v1.report.json").read_text())
    raw = torch.load(P / "fit_uniform_calibration_v1.raw.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(P / "fit_uniform_calibration_v1.checkpoint.pt", map_location="cpu", weights_only=True)
    base_report = json.loads((P / "fit_uniform_baselines_v1.json").read_text())
    base = torch.load(P / "fit_uniform_baselines_v1.pt", map_location="cpu", weights_only=True)
    stage1 = torch.load(P / "fit_rank16_transport_v1.stage1.pt", map_location="cpu", weights_only=True)
    lens = torch.load(P / "fit_rank16_transport_v1.lens.pt", map_location="cpu", weights_only=True)
    meta = raw["metadata"]
    for key, file in {
        "script_sha256": "fit_uniform_calibration_v1.py",
        "protocol_sha256": "PRE_DEV_EVAL_PROTOCOL_V1.md",
        "scoring_v2_sha256": "SCORING_V2.json",
        "lens_sha256": "fit_rank16_transport_v1.lens.pt",
        "stage1_sha256": "fit_rank16_transport_v1.stage1.pt",
        "fit_report_sha256": "fit_rank16_transport_v1.report.json",
    }.items():
        assert meta[key] == sha(file), (key, file)
    assert report["raw_sha256"] == sha("fit_uniform_calibration_v1.raw.pt")
    assert report["memory_sha256"] == sha("fit_uniform_calibration_v1.memory.jsonl")
    assert report["metadata"] == meta == checkpoint["metadata"]
    assert report["summary"] == raw["summary"]
    assert len(raw["records"]) == len(checkpoint["records"]) == 48
    assert same(raw["records"], checkpoint["records"])
    assert meta["fit_metadata"] == lens["metadata"] == stage1["metadata"]
    assert base_report["baseline_sha256"] == sha("fit_uniform_baselines_v1.pt")
    assert base_report["metadata"] == base["metadata"]
    assert base["metadata"]["calibration_sha256"] == report["raw_sha256"]
    assert base["metadata"]["calibration_report_sha256"] == sha("fit_uniform_calibration_v1.report.json")
    assert base["metadata"]["script_sha256"] == sha("fit_uniform_baselines_v1.py")
    cases = {r["case_id"]: (i, r) for i, r in enumerate(stage1["records"])}
    assert len(cases) == 48
    assert set(cases) == {r["case_id"] for r in raw["records"]}
    pairs = Counter(r["pair_id"] for r in raw["records"])
    assert len(pairs) == 12 and set(pairs.values()) == {4}
    assert raw["summary"]["planned_fit_contexts"] == raw["summary"]["completed_fit_contexts"] == 48
    memory = [json.loads(line) for line in (P / "fit_uniform_calibration_v1.memory.jsonl").read_text().splitlines()]
    assert len(memory) == 1 + 48 * (1 + 2 * (3 + 1)) == 433
    assert not any(x["over_limit"] or x["driver_allocated_bytes"] > x["limit_driver_bytes"] for x in memory)
    assert all(x["limit_driver_bytes"] == 32212254720 for x in memory)
    outcome = {"cases": 48, "pairs": 12, "memory_samples": len(memory),
               "peak_driver_gib": max(x["driver_allocated_bytes"] for x in memory) / 2**30,
               "sites": {}}
    basis = lens["basis"].double()
    policy = json.loads((P / "SCORING_V2.json").read_text())
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" / meta["fit_metadata"]["model_revision"]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    norm_key, head_key = "thinker.model.norm.weight", "thinker.lm_head.weight"
    assert index[norm_key] == index[head_key]
    with safe_open(snapshot / index[norm_key], framework="pt", device="cpu") as f:
        norm_weight = f.get_tensor(norm_key).float()
        head = f.get_tensor(head_key)

    def score_state(state, donor, recipient):
        donor_ids = policy["words"][donor]["first_token_ids"]
        recipient_ids = policy["words"][recipient]["first_token_ids"]
        ids = donor_ids + recipient_ids
        h = state.float()
        h = h * torch.rsqrt(h.square().mean() + 1e-6) * norm_weight
        logits = head[ids].float() @ h
        cut = len(donor_ids)
        return float(torch.logsumexp(logits[:cut].double(), 0) -
                     torch.logsumexp(logits[cut:].double(), 0))
    for site in SITES:
        measured, predicted, zeros = [], [], []
        score_diffs = []
        source_width = lens["jacobians"][site].shape[-1]
        for rec in raw["records"]:
            i, st = cases[rec["case_id"]]
            assert rec["audio_sha256"] == st["audio_sha256"]
            donor = [x for x in raw["records"] if x["pair_id"] == rec["pair_id"]
                     and x["case_id"].rsplit("/", 1)[0] == rec["case_id"].rsplit("/", 1)[0]
                     and x["case_id"] != rec["case_id"]]
            assert len(donor) == 1
            j, dst = cases[donor[0]["case_id"]]
            item = rec["sites"][site]
            delta = stage1["source_bin_means"][site][j] - stage1["source_bin_means"][site][i]
            assert torch.equal(delta, item["delta_bins"])
            assert item["source_rows"] == st["row_indices"][site] == dst["row_indices"][site]
            assert item["source_shape"] == st["source_shapes"][site] == dst["source_shapes"][site]
            assert delta.shape == (6, source_width)
            assert all(len(rows) > 0 for rows in item["source_rows"])
            assert item["hook_calls"] == {"source": 1, "target": 1}
            close(item["source_mean_vs_stage1_max_abs"], 0)
            assert len(item["zero_v2_scores"]) == 3
            assert max(abs(x - rec["baseline_v2_score"]) for x in item["zero_v2_scores"]) <= 1e-4
            counts = torch.tensor([len(rows) for rows in item["source_rows"]], dtype=torch.float64)
            edit_norm = 0.025 * torch.sqrt((counts * delta.double().square().sum(1)).sum())
            close(item["effective_source_edit_frobenius_norm"], edit_norm, 1e-5)
            coord = torch.einsum("bkd,bd->k", lens["jacobians"][site].double(), delta.double()) * 0.025
            terminal = item["terminal_delta_per_alpha"].double() * 0.025
            assert terminal.shape == (2048,) and torch.isfinite(terminal).all()
            project = basis @ coord
            measured_coord = basis.T @ terminal
            close(item["rank16_capture_fraction"], (basis @ measured_coord).square().sum() / terminal.square().sum(), 1e-5)
            close(item["projected_relative_error"], (project - basis @ measured_coord).norm() / (basis @ measured_coord).norm(), 1e-5)
            close(item["projected_cosine"], torch.nn.functional.cosine_similarity(project[None], (basis @ measured_coord)[None]).item(), 1e-5)
            close(item["map_absolute_v2_error"], abs(item["measured_v2_delta_score"] - item["map_predicted_v2_delta_score"]), 1e-8)
            recipient = rec["case_id"].rsplit("/", 1)[1]
            donor_word = donor[0]["case_id"].rsplit("/", 1)[1]
            clean_state = st["target_state"]
            clean_score = score_state(clean_state, donor_word, recipient)
            pred_score = score_state(clean_state + project.float(), donor_word, recipient) - clean_score
            score_diffs.append(abs(pred_score - item["map_predicted_v2_delta_score"]))
            assert score_diffs[-1] < 2e-4
            measured.append(item["measured_v2_delta_score"])
            predicted.append(item["map_predicted_v2_delta_score"])
            zeros.extend(item["zero_v2_scores"])
        floor = max(1e-4, 5 * max(abs(z - sum(zeros[k:k+3])/3) for k in range(0, len(zeros), 3) for z in zeros[k:k+3]))
        close(floor, raw["summary"]["numerical_floor_v2_by_site"][site])
        mae = sum(abs(a-b) for a,b in zip(measured,predicted))/48
        zero = sum(abs(a) for a in measured)/48
        determinate = [(a,b) for a,b in zip(measured,predicted) if abs(a) > floor]
        sign = sum((a > 0) == (b > 0) for a,b in determinate)
        outcome["sites"][site] = {"map_mae": mae, "zero_mae": zero,
                                  "determinate": len(determinate), "sign_matches": sign,
                                  "numerical_floor": floor,
                                  "max_cpu_vs_mps_map_score_delta": max(score_diffs)}
        y = torch.stack([r["sites"][site]["terminal_delta_per_alpha"] for r in raw["records"]]).double() @ basis
        for feature in ("global", "six_bin"):
            bins = torch.stack([r["sites"][site]["delta_bins"] for r in raw["records"]]).double()
            if feature == "global":
                weights = torch.tensor([[len(g) for g in r["sites"][site]["source_rows"]]
                                        for r in raw["records"]], dtype=torch.float64)
                x = (bins * weights[...,None]).sum(1) / weights.sum(1)[:,None]
            else:
                x = bins.flatten(1)
            saved = base["fit_training_arrays"][site][feature]
            assert torch.equal(x, saved["x"]) and torch.equal(y, saved["y"])
            ids = [r["pair_id"] for r in raw["records"]]
            table = base["cv_tables"][site][feature]
            for gamma in GAMMAS:
                errs = []
                for pair in dict.fromkeys(ids):
                    held = torch.tensor([pid == pair for pid in ids])
                    z, coeff, scale = ridge(x[~held], y[~held], gamma)
                    p = (x[held] / scale @ z.T / z.shape[1]) @ coeff
                    errs.append(float((p-y[held]).square().mean()))
                saved_errs = table[str(gamma)]["pair_mse"]
                assert max(abs(a-b) for a,b in zip(errs,saved_errs)) < 1e-7
            gamma = min(GAMMAS, key=lambda g: table[str(g)]["pair_mean_mse"])
            model = base["models"][site][feature]
            close(model["gamma"], gamma)
            z, coeff, scale = ridge(x, y, gamma)
            assert torch.allclose(model["train_z"], z, atol=1e-10, rtol=1e-10)
            assert torch.allclose(model["dual_coeff"], coeff, atol=1e-8, rtol=1e-8)
            close(model["scale"], scale, 1e-10)
            close(base_report["fit_pair_cv_rmse"][site][feature], table[str(gamma)]["pair_rmse"])
            outcome["sites"][site][feature+"_gamma"] = gamma
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
