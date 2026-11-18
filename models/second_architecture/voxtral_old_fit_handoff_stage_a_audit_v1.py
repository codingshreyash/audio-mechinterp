"""Independent CPU audit of one old-FIT Voxtral native Stage-A attempt.

This code reads saved receipts and tensors. It does not load or rerun a model.
It verifies arithmetic and provenance that can be checked from the sidecar;
model execution and the saved VJPs themselves remain source-level assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md"
NATIVE = HERE / "voxtral_old_fit_handoff_native_stage_a_v1.py"
RUNNER = HERE / "voxtral_old_fit_handoff_stage_a_runner_v1.py"
OLD_RESULT = HERE / "runs/voxtral_v1_fit_backward_feasibility/20261118T1122Z/result.json"
OLD_DETAILS = OLD_RESULT.with_name("details.pt")
OLD_RESULT_SHA = "d4ccfd213cf3a5ebebd3a7c0ecebab691dc15632f61076669d7535bd58154ebe"
OLD_DETAILS_SHA = "1bc49aa650b96c7ac6cb2f545da98e198edfa18998ebae8dc263c70ad5ad7b3c"
PROTOCOL_SHA = "61a67ac87a4e4e17cb0b1f7ad972f25fda4e0d90e0b70798e7144b3fd6f2484b"
STATUS = "audited_completed_old_fit_handoff_stage_a_native_only"
SITES = ("encoder31", "projector")
ROWS = {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]}
LMS = ("lm0", "lm7", "lm14", "lm22", "lm29")
STAGES = ("projector_word", "lm_input_word", *LMS, "lm29_raw")
ORDERS = ((.00625, -.00625, .025, -.025, .1, -.1),
          (-.1, .1, -.025, .025, -.00625, .00625),
          (.00625, -.00625, .025, -.025, .1, -.1))
EXPECTED_COUNTS = {"native_patch_forwards": 72, "native_pairs": 36,
                   "parity_prompt_forwards": 6, "donor_prompt_forwards": 1,
                   "clean_grad_prompt_forwards": 1, "vjps": 44,
                   "greedy_generations": 6, "lm0_control_zero_forwards": 24,
                   "lm0_control_edited_forwards": 24,
                   "lm0_control_self_forwards": 1}
CAP = int(1.5 * 1024**3)
MAX_SECONDS = {"bf16": 900, "fp32": 1200}
MAX_DRIVER = 30 * 1024**3


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    require(isinstance(value, dict), f"{path.name} is not an object")
    return value


def blob_path(attempt: Path, result: dict, name: str) -> Path:
    digest = result.get(f"{name}_sha256")
    relative = result.get(f"{name}_path")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(ch in "0123456789abcdef" for ch in digest) and
            relative == f"blobs/{digest}.pt", "content-addressed blob descriptor invalid")
    path = attempt / relative
    require(path.is_file() and path.resolve().parent == (attempt / "blobs").resolve() and
            sha(path) == digest, "content-addressed blob bytes differ")
    return path


def close(a, b, *, path: str = "root", atol: float = 2e-5, rtol: float = 2e-5):
    """Compare all recomputed finite numerical leaves, retaining exact topology."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), f"{path}: keys differ")
        for key in a:
            close(a[key], b[key], path=f"{path}.{key}", atol=atol, rtol=rtol)
    elif isinstance(a, (tuple, list)):
        require(isinstance(b, (tuple, list)) and len(a) == len(b), f"{path}: length differs")
        for i, (left, right) in enumerate(zip(a, b)):
            close(left, right, path=f"{path}[{i}]", atol=atol, rtol=rtol)
    elif isinstance(a, float) or isinstance(b, float):
        require(isinstance(a, (float, int)) and isinstance(b, (float, int)) and
                math.isfinite(a) and math.isfinite(b) and
                math.isclose(a, b, rel_tol=rtol, abs_tol=atol), f"{path}: number differs")
    else:
        require(a == b, f"{path}: value differs")


def _seed(label: str) -> int:
    digest = hashlib.sha256(bytes.fromhex(PROTOCOL_SHA) + label.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _rademacher(label: str, width: int, count: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_seed(label))
    signs = torch.randint(0, 2, (width, count), generator=generator, dtype=torch.int8)
    matrix = signs.to(torch.float64).mul_(2).sub_(1)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.contiguous()


def expected_probes() -> dict[str, torch.Tensor]:
    return {"lm": _rademacher("lm_residual_16", 3072, 16),
            "projector_word": _rademacher("projector_word_4", 9216, 4),
            "lm_input_word": _rademacher("lm_input_word_4", 9216, 4)}


def expected_unit() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_seed("lm0_non_audio_control"))
    value = torch.randint(0, 2, (3072,), generator=generator, dtype=torch.int8).to(torch.float64)
    value.mul_(2).sub_(1)
    return (value / torch.linalg.vector_norm(value)).to(torch.float32).contiguous()


def expected_hook_events(dtype: torch.dtype, *, lm_only: bool = False,
                         final_norm: bool = False) -> dict:
    name = str(dtype)
    events = {layer: {"calls": 1, "shape": [1, 396, 3072], "dtype": name}
              for layer in LMS}
    if not lm_only:
        events = {
            "encoder31": {"calls": 1, "shape": [1, 1500, 1280], "dtype": name},
            "projector": {"calls": 1, "shape": [375, 3072], "dtype": name},
            "lm_input": {"calls": 1, "shape": [1, 396, 3072], "dtype": name},
            **events,
        }
    if final_norm:
        events["final_norm"] = {"calls": 1, "shape": [1, 396, 3072], "dtype": name}
    return events


def projected(saved: dict, stage: str, probes: dict) -> torch.Tensor:
    if stage == "projector_word":
        vector = saved["projector_full"][ROWS["projector"]].reshape(-1).float()
        return vector @ probes[stage].float()
    if stage == "lm_input_word":
        positions = saved["audio_placeholder_positions"]
        vector = saved["lm_input_full"][0, [positions[i] for i in ROWS["projector"]]].reshape(-1).float()
        return vector @ probes[stage].float()
    vector = saved["lm_terminal"]["lm29" if stage == "lm29_raw" else stage].float()
    if stage == "lm29_raw":
        return vector[:4]
    count = 16 if stage == "lm29" else 4
    return vector @ probes["lm"].float()[:, :count]


def full_stage(saved: dict, stage: str) -> torch.Tensor:
    if stage == "projector_word":
        return saved["projector_full"].reshape(-1).float()
    if stage == "lm_input_word":
        return saved["lm_input_full"].reshape(-1).float()
    return saved["lm_terminal"]["lm29" if stage == "lm29_raw" else stage].float()


def recompute_odd_even(rows: list[dict]) -> dict:
    """Retain each signed pair before computing the optional cycle median."""
    output = {}
    for magnitude in (.00625, .025, .1):
        cycles = {}
        odds, evens = [], []
        for cycle in range(3):
            plus = next(r for r in rows if r["cycle"] == cycle and r["alpha"] == magnitude)
            minus = next(r for r in rows if r["cycle"] == cycle and r["alpha"] == -magnitude)
            plus_value = torch.tensor(plus["measured"], dtype=torch.float64)
            minus_value = torch.tensor(minus["measured"], dtype=torch.float64)
            odd, even = (plus_value - minus_value) / 2, (plus_value + minus_value) / 2
            odds.append(odd)
            evens.append(even)
            cycles[str(cycle)] = {
                "positive_alpha": magnitude, "negative_alpha": -magnitude,
                "measured_odd": odd.tolist(), "measured_even": even.tolist(),
            }
        output[str(magnitude)] = {
            "cycles": cycles,
            "median_measured_odd": torch.stack(odds).median(0).values.tolist(),
            "median_measured_even": torch.stack(evens).median(0).values.tolist(),
        }
    return output


def validate_trace(saved: dict, *, dtype: torch.dtype, clean: dict,
                   patch_site: str | None, cycle: int | None) -> None:
    require(isinstance(saved, dict) and set(saved["source_rows"]) == set(SITES), "trace source rows missing")
    require(saved.get("hook_events") == expected_hook_events(dtype),
            "native trace hook events differ")
    require(saved["audio_placeholder_positions"] == list(saved["audio_placeholder_positions"]) and
            len(saved["audio_placeholder_positions"]) == 375 and
            len(set(saved["audio_placeholder_positions"])) == 375 and
            all(0 <= i < 396 for i in saved["audio_placeholder_positions"]),
            "audio placeholder positions invalid")
    require(saved["projector_full"].shape == (375, 3072) and
            saved["lm_input_full"].shape == (1, 396, 3072) and
            saved["first_answer_logits"].shape == (131072,) and
            all(value.shape == (3072,) for value in saved["lm_terminal"].values()) and
            set(saved["lm_terminal"]) == set(LMS), "trace shape differs")
    tensors = [saved["projector_full"], saved["lm_input_full"], saved["first_answer_logits"],
               *saved["lm_terminal"].values(), *saved["source_rows"].values()]
    require(saved["source_rows"]["encoder31"].shape == (12, 1280) and
            saved["source_rows"]["projector"].shape == (3, 3072),
            "selected source row shape differs")
    require(all(isinstance(v, torch.Tensor) and v.dtype == dtype and torch.isfinite(v).all()
                for v in tensors), "trace dtype or finiteness differs")
    require(torch.equal(saved["lm_input_full"][0, saved["audio_placeholder_positions"]],
                        saved["projector_full"]), "projector-to-LM handoff differs")
    hashes = {"projector": tensor_sha(saved["projector_full"]),
              "lm_input": tensor_sha(saved["lm_input_full"]),
              "first_answer_logits": tensor_sha(saved["first_answer_logits"]),
              **{name: tensor_sha(value) for name, value in saved["lm_terminal"].items()}}
    require(saved["stage_hashes"] == hashes, "trace stage hashes differ")
    require(saved["audio_placeholder_positions"] == clean["audio_placeholder_positions"],
            "audio placeholder positions differ from clean trace")
    stage_values = {"projector_word": saved["projector_full"][ROWS["projector"]].reshape(-1),
                    "lm_input_word": saved["lm_input_full"][0, [saved["audio_placeholder_positions"][i]
                                                                 for i in ROWS["projector"]]].reshape(-1),
                    **saved["lm_terminal"]}
    require(set(saved["native_ulp_spacing"]) == set(stage_values), "native ULP stage set differs")
    for stage, value in stage_values.items():
        spacing = (torch.nextafter(value, torch.full_like(value, float("inf"))).float() - value.float()).contiguous()
        require(torch.equal(saved["native_ulp_spacing"][stage], spacing),
                "native ULP spacing differs")
    for hash_field in ("source_full_pre_hashes", "source_full_post_hashes"):
        require(set(saved[hash_field]) == set(SITES) and
                all(isinstance(h, str) and len(h) == 64 and
                    all(ch in "0123456789abcdef" for ch in h)
                    for h in saved[hash_field].values()), "source full hash schema differs")
    if patch_site is not None:
        require(torch.equal(saved["source_rows"][patch_site], clean["source_rows"][patch_site]),
                "prepatch source rows differ from clean gradient source")
        require(saved["source_full_pre_hashes"][patch_site] ==
                clean["source_full_pre_hashes"][patch_site],
                "prepatch full source hash differs from clean gradient source")
    if cycle == 0:
        require(set(saved["source_full_pre"]) == set(SITES) and
                set(saved["source_full_post"]) == set(SITES), "first-cycle full source missing")
        for site in SITES:
            pre = saved["source_full_pre"][site]
            post = saved["source_full_post"][site]
            expected = (1, 1500, 1280) if site == "encoder31" else (375, 3072)
            require(pre.shape == post.shape == expected and pre.dtype == post.dtype == dtype and
                    torch.isfinite(pre).all() and torch.isfinite(post).all(), "full source shape/dtype differs")
            require(tensor_sha(pre) == saved["source_full_pre_hashes"][site] and
                    tensor_sha(post) == saved["source_full_post_hashes"][site],
                    "archived full source hash differs")
            rows = pre[0, ROWS[site]] if site == "encoder31" else pre[ROWS[site]]
            require(torch.equal(rows, saved["source_rows"][site]), "full source/selected rows differ")
            if site == patch_site:
                require(torch.equal(pre, clean["source_full_pre"][site]),
                        "full prepatch source differs from clean gradient source")
                effective = (post[0, ROWS[site]] if site == "encoder31" else post[ROWS[site]]).float() - rows.float()
                require(torch.equal(effective, saved["source_effective_edit"]),
                        "full source edit differs from selected effective edit")
                other = post.clone()
                if site == "encoder31": other[0, ROWS[site]] = pre[0, ROWS[site]]
                else: other[ROWS[site]] = pre[ROWS[site]]
                require(torch.equal(other, pre), "full source changed outside selected rows")


def validate_layout(layout: dict, arm: str) -> None:
    require(layout.get("valid") is True and layout.get("arm") == arm and
            layout.get("named_parameter_count") == 762 and
            isinstance(layout.get("parameters"), list) and len(layout["parameters"]) == 762,
            "parameter layout invalid")
    names = [row["name"] for row in layout["parameters"]]
    require(len(set(names)) == 762 and "model.audio_tower.embed_positions.weight" in names,
            "position parameter missing or duplicate parameter name")
    counts = {"torch.bfloat16/mps": 761, "torch.float32/mps": 1} if arm == "bf16" else {"torch.float32/mps": 762}
    require(layout.get("dtype_device_counts") == counts, "model dtype/device counts differ")
    for row in layout["parameters"]:
        expected = "torch.float32" if arm == "fp32" or row["name"] == "model.audio_tower.embed_positions.weight" else "torch.bfloat16"
        require(row["dtype"] == expected and row["device"] == "mps" and
                isinstance(row["shape"], list) and math.prod(row["shape"]) == row["numel"],
                "model parameter dtype/device/shape differs")
        if row["name"] == "model.audio_tower.embed_positions.weight":
            require(row["shape"] == [1500, 1280], "audio position table shape changed")


def validate_pairs(details: dict, arm: str, probes: dict) -> dict:
    dtype = torch.bfloat16 if arm == "bf16" else torch.float32
    clean = details["clean_trace"]
    validate_trace(clean, dtype=dtype, clean=clean, patch_site=None, cycle=0)
    for site in SITES:
        require(torch.equal(clean["source_full_pre"][site], clean["source_full_post"][site]),
                "clean source was edited")
        require(details["directions"][site].shape == (1280 if site == "encoder31" else 3072,),
                "donor direction width differs")
    require(set(details["observations"]) == set(SITES) and
            set(details["source_vjps"]) == set(STAGES), "site/VJP stage matrix incomplete")
    vjps = details["source_vjps"]
    for stage in STAGES:
        expected_n = 16 if stage == "lm29" else 4
        require(set(vjps[stage]) == set(SITES), "source VJP sites incomplete")
        for site in SITES:
            shape = (expected_n, len(ROWS[site]), 1280 if site == "encoder31" else 3072)
            v = vjps[stage][site]
            require(isinstance(v, torch.Tensor) and v.dtype == torch.float32 and
                    tuple(v.shape) == shape and torch.isfinite(v).all(),
                    f"{stage}/{site} VJP shape/dtype/finiteness differs")
    analysis: dict[str, dict] = {}
    for site in SITES:
        records = details["observations"][site]
        expected_keys = [(cycle, alpha) for cycle, order in enumerate(ORDERS) for alpha in order]
        require(isinstance(records, list) and len(records) == 18 and
                [(r["cycle"], r["alpha"]) for r in records] == expected_keys,
                f"{site} pair order/count differs")
        for record in records:
            cycle = record["cycle"]
            zero, edited = record["zero"], record["edited"]
            validate_trace(zero, dtype=dtype, clean=clean, patch_site=site, cycle=cycle)
            validate_trace(edited, dtype=dtype, clean=clean, patch_site=site, cycle=cycle)
            require(zero["audio_placeholder_positions"] == clean["audio_placeholder_positions"] ==
                    edited["audio_placeholder_positions"], "audio-placeholder identity changed")
            require(torch.equal(zero["source_rows"][site], edited["source_rows"][site]),
                    "paired prepatch source rows differ")
            require(zero["source_edit_metrics"]["kind"] == "zero" and
                    edited["source_edit_metrics"]["kind"] == "alpha",
                    "pair labels differ")
            z_edit = zero["source_effective_edit"]
            a_edit = edited["source_effective_edit"]
            expected_shape = (len(ROWS[site]), 1280 if site == "encoder31" else 3072)
            require(z_edit.shape == a_edit.shape == expected_shape and
                    z_edit.dtype == a_edit.dtype == torch.float32 and
                    torch.count_nonzero(z_edit) == 0 and torch.isfinite(a_edit).all(),
                    "zero/edited source edit invalid")
            intended = record["alpha"] * details["directions"][site].float()
            source = edited["source_rows"][site]
            realized = ((source.float() + intended.to(source.device)).to(dtype).float() - source.float())
            require(torch.equal(a_edit, realized), "realized edit differs from saved source/direction")
            mean = a_edit.mean(0)
            denominator = max(float(torch.linalg.vector_norm(intended.double())), 1e-6)
            edit_metrics = {"kind": "alpha",
                            "effective_norm": float(torch.linalg.vector_norm(a_edit.double())),
                            "mean_fidelity_ratio": float(torch.linalg.vector_norm((mean - intended).double())) / denominator,
                            "row_nonuniformity_ratio": float(torch.sqrt(((a_edit - mean).double().square().sum(1)).mean())) /
                            max(float(torch.linalg.vector_norm(mean.double())), 1e-6),
                            "unchanged_fraction": float((a_edit == 0).float().mean())}
            close(edit_metrics, edited["source_edit_metrics"], path=f"{site}/edit_metrics")
        site_analysis: dict[str, dict] = {}
        for stage in STAGES:
            zeros = torch.stack([projected(r["zero"], stage, probes) for r in records]).double()
            noise = float(torch.cdist(zeros, zeros).max())
            resolution = max(3 * noise, 1e-6)
            rows = []
            for i, record in enumerate(records):
                measured = projected(record["edited"], stage, probes) - zeros[i].float()
                effective = record["edited"]["source_effective_edit"].float()
                gradient = vjps[stage][site]
                prediction = torch.einsum("nrd,rd->n", gradient, effective)
                uniform = (gradient.sum(1) * effective.mean(0)).sum(1)
                ideal = (gradient.sum(1) *
                         (float(record["alpha"]) * details["directions"][site].float())).sum(1)
                measured_norm = float(torch.linalg.vector_norm(measured.double()))
                prediction_norm = float(torch.linalg.vector_norm(prediction.double()))
                err = float(torch.linalg.vector_norm((prediction - measured).double()))
                resolved = measured_norm > resolution
                cosine = None
                if resolved and prediction_norm > resolution:
                    cosine = float(torch.dot(prediction.double(), measured.double()) /
                                   (prediction_norm * measured_norm))
                delta = full_stage(record["edited"], stage) - full_stage(record["zero"], stage)
                rows.append({"cycle": record["cycle"], "alpha": record["alpha"],
                             "measured": measured.tolist(), "predicted_rowwise": prediction.tolist(),
                             "predicted_uniform": uniform.tolist(), "predicted_ideal": ideal.tolist(),
                             "measured_l2": measured_norm, "absolute_l2_error": err,
                             "relative_l2_error_if_resolved": err / measured_norm if resolved else None,
                             "cosine_if_resolved": cosine, "resolved": resolved,
                             "changed_full_coordinates": int((delta != 0).sum()),
                             "full_coordinate_count": delta.numel(),
                             "full_delta_l2": float(torch.linalg.vector_norm(delta.double()))})
            decisive = [r for r in rows if abs(r["alpha"]) == .1]
            require(len(decisive) == 6, "decisive grid incomplete")
            if all(r["resolved"] and r["relative_l2_error_if_resolved"] <= .25 for r in decisive):
                verdict = "local_diagnostic_pass"
            elif all(r["resolved"] and r["relative_l2_error_if_resolved"] > .5 for r in decisive):
                verdict = "diagnostic_mismatch"
            else:
                verdict = "indeterminate"
            odd_even = recompute_odd_even(rows)
            site_analysis[stage] = {"zero_repeat_spread_l2": noise, "resolution_l2": resolution,
                                    "records": rows, "odd_even": odd_even, "verdict": verdict}
        analysis[site] = site_analysis
    return analysis


def choose_scale(source: torch.Tensor, unit: torch.Tensor, sign: int, target: float) -> dict:
    """Independent deterministic reimplementation of the protocol's CPU search."""
    if not math.isfinite(target) or target <= 0:
        return {"valid": False, "reason": "zero_or_nonfinite_target", "scalar": 0.0,
                "target_norm": target if math.isfinite(target) else None,
                "realized_norm": 0.0, "visited_candidates": 1}

    def norm(s):
        row = (source.float() + sign * s * unit).to(source.dtype)
        return float(torch.linalg.vector_norm((row.float() - source.float()).double()))

    seen = [(0.0, 0.0), (target, norm(target))]
    lo, hi = 0.0, target
    if seen[-1][1] < target:
        for _ in range(8):
            lo, hi = hi, hi * 2
            seen.append((hi, norm(hi)))
            if seen[-1][1] >= target:
                break
    if seen[-1][1] >= target:
        for _ in range(24):
            middle = (lo + hi) / 2
            seen.append((middle, norm(middle)))
            if seen[-1][1] >= target: hi = middle
            else: lo = middle
    finite = [(s, v) for s, v in seen if math.isfinite(v)]
    if not finite:
        return {"valid": False, "reason": "no_finite_candidate", "scalar": 0.0,
                "target_norm": target, "realized_norm": 0.0,
                "visited_candidates": len(seen)}
    scalar, actual = min(finite, key=lambda item: (abs(item[1] - target), item[0]))
    relative = abs(actual - target) / target
    return {"valid": relative <= .05, "reason": "matched" if relative <= .05 else "no_5pct_match",
            "scalar": scalar, "target_norm": target, "realized_norm": actual,
            "relative_match_error": relative, "visited_candidates": len(seen)}


def validate_lm0_controls(details: dict, arm: str) -> None:
    controls = details["lm0_controls"]
    unit = expected_unit()
    require(torch.equal(controls["unit"], unit), "LM0 control unit differs")
    expected = [(site, cycle, alpha) for site in SITES for cycle, order in enumerate(ORDERS)
                for alpha in order if alpha in (.025, -.025, .1, -.1)]
    records = controls["records"]
    require(len(records) == len(expected) == 24 and
            [(r["site"], r["cycle"], r["alpha"]) for r in records] == expected,
            "LM0 control cell order/count differs")
    dtype = torch.bfloat16 if arm == "bf16" else torch.float32
    for record in records:
        site, cycle, alpha = record["site"], record["cycle"], record["alpha"]
        native = next(r for r in details["observations"][site]
                      if r["cycle"] == cycle and r["alpha"] == alpha)
        baseline = native["zero"]["lm_terminal"]["lm0"]
        target = float(torch.linalg.vector_norm((native["edited"]["lm_terminal"]["lm0"].float() -
                                                 baseline.float()).double()))
        choice = choose_scale(baseline, unit, 1 if alpha > 0 else -1, target)
        close(choice, record["choice"], path="lm0_control.choice")
        zero, edited = record["zero"], record["edited"]
        for item in (zero, edited):
            require(item["prepatch_lm0"].shape == (3072,) and
                    item["prepatch_lm0"].dtype == dtype and
                    torch.equal(item["prepatch_lm0"], baseline) and
                    item.get("hook_events") == expected_hook_events(dtype, lm_only=True) and
                    set(item["lm_terminal"]) == set(LMS) and
                    all(v.shape == (3072,) and v.dtype == dtype and torch.isfinite(v).all()
                        for v in item["lm_terminal"].values()), "LM0 control baseline/state invalid")
        require(torch.count_nonzero(zero["effective_edit_lm0"]) == 0 and
                zero["kind"] == "zero" and edited["kind"] in ("edited", "zero"),
                "LM0 control zero/edited kind differs")
        expected_effect = (baseline.float() + (1 if alpha > 0 else -1) *
                           choice["scalar"] * unit).to(dtype).float() - baseline.float()
        require(torch.equal(edited["effective_edit_lm0"], expected_effect) and
                torch.equal(edited["lm_terminal"]["lm0"].float() - baseline.float(), expected_effect),
                "LM0 control edit differs")
        actual_norm = float(torch.linalg.vector_norm(expected_effect.double()))
        close(actual_norm, record["observed_realized_norm"], path="LM0 control realized norm")
    self_trace = controls["self"]
    require(self_trace["kind"] == "self" and torch.count_nonzero(self_trace["effective_edit_lm0"]) == 0 and
            self_trace.get("hook_events") == expected_hook_events(dtype, lm_only=True) and
            torch.equal(self_trace["prepatch_lm0"], self_trace["lm_terminal"]["lm0"]),
            "LM0 control self condition differs")


def committed_same(path: Path) -> None:
    repo = HERE.parents[1]
    relative = path.resolve().relative_to(repo.resolve())
    shown = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{relative.as_posix()}"],
                           capture_output=True, check=False)
    require(shown.returncode == 0 and shown.stdout == path.read_bytes(),
            f"source is not identical to committed HEAD: {path.name}")


def validate_progress(path: Path, arm: str) -> dict:
    lines = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
             for line in path.read_text().splitlines()]
    require(lines and lines[0].get("phase") == "attempt_started" and
            lines[0].get("arm") == arm and lines[-1].get("phase") == "details_saved",
            "progress start/end differs")
    elapsed = [float(row["elapsed_seconds"]) for row in lines if "elapsed_seconds" in row]
    require(elapsed and all(math.isfinite(v) and 0 <= v < MAX_SECONDS[arm] for v in elapsed) and
            elapsed == sorted(elapsed), "progress elapsed time invalid")
    phases = [row["phase"] for row in lines]
    require(phases.count("before_model_load") == 1 and phases.count("model_loaded") == 1 and
            phases.count("after_44_vjps") == 1 and phases.count("details_saved") == 1 and
            phases.count("after_natural_generation") == 1 and
            phases.count("after_passive_generation") == 1 and
            phases.count("lm0_control_self") == 1,
            "progress fixed phases missing")
    patch_before = [row for row in lines if row["phase"].startswith("before_encoder31_") or
                    row["phase"].startswith("before_projector_")]
    patch_after = [row for row in lines if row["phase"].startswith("after_encoder31_") or
                   row["phase"].startswith("after_projector_")]
    expected_before = []
    expected_after = []
    for site in SITES:
        for cycle, order in enumerate(ORDERS):
            for alpha in order:
                for label in ("zero", "edited"):
                    expected_before.append(f"before_{site}_{cycle}_{label}_{alpha}")
                    expected_after.append(f"after_{site}_{cycle}_{label}_{alpha}")
    require(len(patch_before) == len(patch_after) == 72 and
            [row["phase"] for row in patch_before] == expected_before and
            [row["phase"] for row in patch_after] == expected_after and
            [row.get("patch_count") for row in patch_after] == list(range(1, 73)),
            "progress 72 patch forwards not receipted")
    lm0_phases = [row["phase"] for row in lines if row["phase"].startswith("lm0_control_") and
                  row["phase"] != "lm0_control_self"]
    expected_lm0 = []
    for site in SITES:
        for cycle, order in enumerate(ORDERS):
            for alpha in order:
                if alpha in (.025, -.025, .1, -.1):
                    expected_lm0.extend((f"lm0_control_{site}_{cycle}_{alpha}_zero",
                                         f"lm0_control_{site}_{cycle}_{alpha}_edited"))
    require(lm0_phases == expected_lm0, "progress LM0 control order/count differs")
    require(all(int(row.get("peak_driver_bytes", 0)) < MAX_DRIVER for row in lines),
            "progress observed MPS driver cap reached")
    return {"progress_lines": len(lines), "patch_receipts": 72,
            "lm0_zero_receipts": 24, "lm0_edited_receipts": 24}


def validate_parity(details: dict, result: dict, arm: str) -> None:
    parity = details["parity"]
    expected = {"pure", "passive", "encoder31_self", "encoder31_zero",
                "projector_self", "projector_zero"}
    require(parity.get("passed") is True and result.get("parity_passed") is True and
            set(parity["prompt_states"]) == set(parity["suffix_ids"]) == expected,
            "six-condition parity population differs")
    pure = parity["prompt_states"]["pure"]
    require(pure["final_full"].shape == (1, 396, 3072) and
            pure["first_answer_logits"].shape == (131072,), "pure prompt parity shape differs")
    for condition in expected:
        state = parity["prompt_states"][condition]
        expected_events = {} if condition == "pure" else expected_hook_events(
            torch.bfloat16 if arm == "bf16" else torch.float32, final_norm=True)
        require(state.get("hook_events") == expected_events and
                (condition != "pure" or state.get("hook_mode") == "none"),
                "parity prompt hook events differ")
        require(torch.equal(state["final_full"], pure["final_full"]) and
                torch.equal(state["first_answer_logits"], pure["first_answer_logits"]) and
                parity["suffix_ids"][condition] == parity["suffix_ids"]["pure"],
                "six-condition prompt/suffix parity differs")
    require(parity["suffix_ids"]["pure"] == result["natural_suffix_ids"],
            "result natural suffix differs")
    if arm == "bf16":
        require(result["natural_suffix_ids"] == [5934, 14844, 2],
                "BF16 old-FIT suffix parity differs")


def validate_old_direction(details: dict, meta: dict) -> None:
    require(sha(OLD_RESULT) == OLD_RESULT_SHA and sha(OLD_DETAILS) == OLD_DETAILS_SHA,
            "pinned old-FIT source receipts changed")
    old_result = read_json(OLD_RESULT)
    require(old_result.get("details_sha256") == OLD_DETAILS_SHA and
            old_result.get("status") == "completed_numerical_feasibility_only",
            "old-FIT result provenance differs")
    sidecar = torch.load(OLD_DETAILS, map_location="cpu", weights_only=True)
    for site in SITES:
        recipient = sidecar["source_states"][site]
        donor = sidecar["donor_states"][site]
        require(recipient.dtype == donor.dtype == torch.bfloat16 and
                tensor_sha(recipient) == old_result["source_hashes"][site] and
                tensor_sha(donor) == old_result["donor_hashes"][site],
                "old-FIT BF16 source/donor tensor differs")
        direction = donor[ROWS[site]].float().mean(0) - recipient[ROWS[site]].float().mean(0)
        require(torch.equal(details["directions"][site], direction) and
                tensor_sha(direction) == meta["directions_sha256"][site],
                "new attempt direction differs from frozen old-FIT donor mean")


def audit(attempt: Path) -> dict:
    require(attempt.is_dir(), "attempt directory absent")
    for path in (PROTOCOL, NATIVE, RUNNER):
        committed_same(path)
    require(sha(PROTOCOL) == PROTOCOL_SHA, "protocol bytes changed")
    result_file = attempt / "result.json"
    error_file = attempt / "error.json"
    require(result_file.is_file() != error_file.is_file(),
            "attempt must have exactly one completed/error receipt")
    if error_file.is_file():
        error = read_json(error_file)
        require(error.get("status") in ("resource_limited_stage_a_arm", "invalid_stage_a_arm") and
                error.get("arm") in MAX_SECONDS and
                error.get("progress_sha256") == sha(attempt / "progress.jsonl"),
                "failed attempt receipt invalid")
        return {"status": "audited_failed_stage_a_attempt_receipt_only",
                "arm": error["arm"], "protocol_sha256": PROTOCOL_SHA,
                "error_sha256": sha(error_file), "progress_sha256": error["progress_sha256"],
                "auditor_sha256": sha(Path(__file__))}
    result = read_json(result_file)
    arm = result.get("arm")
    require(result.get("status") == "completed_old_fit_handoff_stage_a_native_only" and arm in MAX_SECONDS,
            "completed attempt status/arm invalid")
    files = {"meta_sha256": "meta.json", "model_layout_sha256": "model_layout.json",
             "progress_sha256": "progress.jsonl"}
    require(all(sha(attempt / filename) == result.get(field) for field, filename in files.items()),
            "raw receipt file hash differs")
    probe_file = blob_path(attempt, result, "probes")
    detail_file = blob_path(attempt, result, "details")
    require(not any(path.is_symlink() for path in attempt.rglob("*")),
            "attempt contains a symlink")
    require(result.get("script_sha256") == sha(RUNNER) and
            result.get("native_source_sha256") == sha(NATIVE) and
            result.get("protocol_sha256") == PROTOCOL_SHA and
            result.get("planned_counts") == result.get("actual_counts") == EXPECTED_COUNTS,
            "source pin or planned/actual counts differ")
    final_bytes = sum(p.stat().st_size for p in attempt.rglob("*") if p.is_file())
    require(result.get("probes_bytes") == probe_file.stat().st_size and
            result.get("details_bytes") == detail_file.stat().st_size and
            result.get("max_output_bytes") == CAP and
            result.get("bytes_before_result") == final_bytes - result_file.stat().st_size and
            final_bytes <= CAP,
            "output byte cap/size receipt differs")
    require(0 < result.get("elapsed_seconds", 0) < MAX_SECONDS[arm] and
            0 < result.get("peak_mps_driver_bytes", 0) < MAX_DRIVER,
            "time/MPS cap receipt invalid")
    meta = read_json(attempt / "meta.json")
    layout = read_json(attempt / "model_layout.json")
    require(meta.get("arm") == arm and meta.get("protocol_sha256") == PROTOCOL_SHA and
            meta.get("script_sha256") == sha(RUNNER) and
            meta.get("native_source_sha256") == sha(NATIVE) and
            meta.get("planned_counts") == EXPECTED_COUNTS and
            meta.get("fresh_v2_corpus_opened") is False and
            meta.get("max_output_bytes") == CAP and
            meta.get("probes_path") == result.get("probes_path"),
            "metadata source/scope differs")
    validate_layout(layout, arm)
    progress = validate_progress(attempt / "progress.jsonl", arm)
    probes = torch.load(probe_file, map_location="cpu", weights_only=True)
    expected = expected_probes()
    require(set(probes) == set(expected) and
            all(torch.equal(probes[name], expected[name]) for name in expected),
            "fixed probe vectors differ")
    details = torch.load(detail_file, map_location="cpu", weights_only=True)
    require(set(details) == {"probes", "clean_trace", "donor_trace", "source_vjps",
                             "observations", "lm0_controls", "parity", "directions", "analysis"} and
            all(torch.equal(details["probes"][name], probes[name]) for name in expected),
            "detail/probe sidecar mismatch")
    validate_old_direction(details, meta)
    for site in SITES:
        require(tensor_sha(details["clean_trace"]["source_rows"][site]) ==
                result["source_hashes"][site],
                "clean source/direction receipt differs")
    donor = details["donor_trace"]
    dtype = torch.bfloat16 if arm == "bf16" else torch.float32
    require(set(donor.get("source_hashes", {})) == set(SITES) and
            donor.get("hook_events") == expected_hook_events(dtype) and
            all(isinstance(value, str) and len(value) == 64 for value in donor["source_hashes"].values()) and
            set(donor.get("lm_terminal", {})) == set(LMS) and
            all(isinstance(value, torch.Tensor) and value.shape == (3072,) and
                value.dtype == dtype and torch.isfinite(value).all()
                for value in donor["lm_terminal"].values()) and
            isinstance(donor.get("first_answer_logits"), torch.Tensor) and
            donor["first_answer_logits"].shape == (131072,) and
            donor["first_answer_logits"].dtype == dtype and
            torch.isfinite(donor["first_answer_logits"]).all(),
            "donor trace report-only field invalid")
    recomputed = validate_pairs(details, arm, probes)
    close(recomputed, details["analysis"], path="details.analysis")
    close(recomputed, result["native_stage_analysis"], path="result.native_stage_analysis")
    validate_lm0_controls(details, arm)
    validate_parity(details, result, arm)
    return {"status": STATUS, "arm": arm, "protocol_sha256": PROTOCOL_SHA,
            "runner_sha256": sha(RUNNER), "native_source_sha256": sha(NATIVE),
            "result_sha256": sha(result_file), "details_sha256": sha(detail_file),
            "probes_sha256": sha(probe_file),
            "auditor_sha256": sha(Path(__file__)),
            "audited_native_pairs": 36, "audited_patch_forwards": 72,
            "audited_vjps": 44, "audited_lm0_control_cells": 24,
            "observed_progress": progress,
            "scope": "saved_tensor_and_receipt_arithmetic_only_no_model_execution_attestation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.attempt), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


