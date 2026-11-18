"""CPU-only, read-only audit of one old-FIT Voxtral backward feasibility attempt.

No model, processor, MPS, active development result, or v2 corpus is opened.
The completed branch checks numerical provenance; it does not assess efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PILOT_SOURCE = HERE / "voxtral_v1_fit_backward_feasibility.py"
PROTOCOL = HERE / "VOXTRAL_V1_FIT_BACKWARD_FEASIBILITY_PROTOCOL.md"
FIRST_RUNNER = HERE / "voxtral_fit_v2_causal_first_record.py"
FRONTEND = HERE / "voxtral_preprocessing_decomposition.py"
MASKS = HERE / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json"
INDEX = HERE / "voxtral_corpus_v1_index.json"
SAVED_PILOT = HERE / "runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/result.json"
SNAPSHOT_TREE = HERE / "hf_cache/voxtral/.cache/huggingface/trees/3060fe34b35ba5d44202ce9ff3c097642914f8f3.json"
DEFAULT_RUNS = HERE / "runs/voxtral_v1_fit_backward_feasibility"
PIN = {
    "pilot_source": "cc48437b96d47c766f5ba9f86eb5789481fecd7d1469c09a503d225a42ed3be1",
    "protocol": "8fa6aa224e150b2579e335d14f7dafddd542eacdf733bc92ab7f3ff281384148",
    "first_runner": "64d799bbb40b58b128b049877d18bde7774c2dc16717af97e692aa4531e907b5",
    "frontend": "026445da1ac2c4c2f4b8f45ab2f8d7a1f4679924774714951b16d5cd29de0788",
    "masks": "a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e",
    "index": "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971",
    "snapshot_tree": "e8e4220e9f8a1ed5fd4eb7c640c7061d10516530476438cd7f57b555b7e1e88a",
    "saved_pilot": "6cc84dc1a4ced9ce17380eab3836079c462c9974f31337f146416730b5536ccf",
    "llama": "affc6a4ab36b562a6b4c90b84c4e7939c5417a69ad7bd37e858be5583f189830",
}
SOURCE_HASHES = {
    "modeling_voxtral": "2d3ec9d78c469b77c42a8d5ba258185fb1c58f4c2c7bf6007ec724436dfc7056",
    "processing_voxtral": "239e84657d1a0287ab27a540c946a787a69b4b0bf45727d6b24bee21ccec0296",
    "feature_extraction_whisper": "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb",
    "modeling_llama": PIN["llama"],
}
SITES = ("encoder31", "projector")
COORDS = (0, 1, 2, 3)
ALPHA = 0.025
TIME_CAP = 900
MEMORY_CAP = 30 * 1024**3
SHAPES = {"encoder31": (1500, 1280), "projector": (375, 3072)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def finite_number(value, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)), f"missing/nonfinite number: {label}")
    return float(value)


def require_all_finite(value, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_all_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_all_finite(item, f"{label}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"nonfinite JSON value: {label}")


def close(actual, expected, label: str, *, atol=1e-5, rtol=1e-5) -> None:
    a = finite_number(actual, label)
    b = finite_number(expected, f"expected {label}")
    require(math.isclose(a, b, abs_tol=atol, rel_tol=rtol),
            f"numerical mismatch {label}: {a} vs {b}")


def close_vector(actual, expected, label: str, *, atol=1e-5, rtol=1e-5) -> None:
    require(isinstance(actual, list) and len(actual) == len(expected), f"vector length {label}")
    for i, (a, b) in enumerate(zip(actual, expected)):
        close(a, b, f"{label}[{i}]", atol=atol, rtol=rtol)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    require_all_finite(value, path.name)
    return value


def source_context() -> dict:
    files = {"pilot_source": PILOT_SOURCE, "protocol": PROTOCOL,
             "first_runner": FIRST_RUNNER, "frontend": FRONTEND,
             "masks": MASKS, "index": INDEX, "snapshot_tree": SNAPSHOT_TREE,
             "saved_pilot": SAVED_PILOT}
    for key, path in files.items():
        require(path.is_file() and sha(path) == PIN[key], f"pinned source drift: {key}")
    masks = read_json(MASKS)
    first = masks["records"][0]
    require((first["pair_id"], first["engine"], first["phrasing_id"]) ==
            ("animal_08", "piper", "fit_chosen"), "FIT mask record changed")
    rows = {site: first["site_maps"][site]["bins"]["word_0"] for site in SITES}
    require(rows == {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]},
            "FIT word_0 row mask changed")
    wav_hashes = {clip["word"]: clip["wav_sha256"] for clip in first["clips"]}
    require(set(wav_hashes) == {"beaver", "monkey"}, "FIT paired clips changed")
    pilot = read_json(SAVED_PILOT)
    matching = [r for r in pilot["records"] if r["pair_id"] == "animal_08" and
                r["engine"] == "piper" and r["phrasing_id"] == "fit_chosen" and
                r["expected_word"] == "beaver"]
    require(len(matching) == 1 and matching[0]["classification"]["exact_expected_word_and_eos"],
            "saved FIT recipient answer changed")
    return {"rows": rows, "wav_hashes": wav_hashes,
            "saved_suffix": matching[0]["generated_token_ids"],
            "prompt_len": len(matching[0]["prompt_input_ids"]),
            "index": read_json(INDEX), "snapshot_tree": read_json(SNAPSHOT_TREE)}


def progress_receipt(run_dir: Path) -> tuple[list[dict], str]:
    path = run_dir / "progress.jsonl"
    require(path.is_file(), "missing attempt progress receipt")
    lines = path.read_text().splitlines()
    require(bool(lines), "empty attempt progress receipt")
    entries = [json.loads(line) for line in lines]
    require(all(isinstance(item, dict) for item in entries), "invalid progress line")
    require_all_finite(entries, "progress")
    first = entries[0]
    require(first.get("phase") == "attempt_started" and
            first.get("script_sha256") == PIN["pilot_source"] and
            first.get("protocol_sha256") == PIN["protocol"] and
            isinstance(first.get("wall_utc"), str), "unbound attempt start")
    previous = -1.0
    for item in entries:
        elapsed = finite_number(item.get("elapsed_seconds"), f"progress {item.get('phase')} elapsed")
        require(elapsed >= previous, "progress elapsed time reversed")
        previous = elapsed
    return entries, sha(path)


def tensor(value, shape: tuple[int, ...], dtype: torch.dtype, label: str) -> torch.Tensor:
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == dtype,
            f"tensor shape/dtype mismatch: {label}")
    require(bool(torch.isfinite(value).all()), f"nonfinite tensor: {label}")
    return value


def vector_error(pred: list[float], observed: list[float]) -> dict:
    p = torch.tensor(pred, dtype=torch.float64)
    m = torch.tensor(observed, dtype=torch.float64)
    absolute = float(torch.linalg.vector_norm(p - m))
    measured_norm = float(torch.linalg.vector_norm(m))
    return {"absolute_l2": absolute, "relative_l2": absolute / max(measured_norm, 1e-6),
            "measured_l2": measured_norm}


def hook_events(events: dict, prompt_len: int, label: str) -> None:
    expected = {"encoder31": [1, 1500, 1280], "projector": [375, 3072],
                "lm29": [1, prompt_len, 3072]}
    require(set(events) == set(expected), f"hook site mismatch: {label}")
    for site, shape in expected.items():
        require(events[site]["shape"] == shape and events[site]["dtype"] == "torch.bfloat16",
                f"hook shape/dtype mismatch: {label}/{site}")


def snapshot_receipt(snapshot: dict, context: dict) -> None:
    require(snapshot["revision"] == "3060fe34b35ba5d44202ce9ff3c097642914f8f3" and
            snapshot["model_id"] == "mistralai/Voxtral-Mini-3B-2507", "snapshot identity changed")
    tree_files = context["snapshot_tree"]["files"]
    expected = {
        "config": ("368124c9a9171b3a6c0cdb35cd7fedeff465eefecf315e2a621684bbed7dcd7e", tree_files["config.json"]["size"]),
        "snapshot_tree": (PIN["snapshot_tree"], SNAPSHOT_TREE.stat().st_size),
        "weight_index": ("7506794dd65e4324685f9634b3d83f20d36a723543650c1e6b72d81965289930", tree_files["model.safetensors.index.json"]["size"]),
        "processor_config": (context["index"]["processor_config_sha256"], tree_files["preprocessor_config.json"]["size"]),
        "tokenizer_asset": (context["index"]["tokenizer_asset_sha256"], tree_files["tekken.json"]["size"]),
        "generation_config": ("cebeed3d4a1680c9b311863385d32b91008480dfbf8cf7abc2142447b8f73b76", tree_files["generation_config.json"]["size"]),
    }
    for key, (digest, size) in expected.items():
        require(snapshot[key] == {"sha256": digest, "size_bytes": size},
                f"snapshot {key} receipt changed")
    names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    require(set(snapshot["weight_shards"]) == set(names), "weight shard set changed")
    for name in names:
        require(snapshot["weight_shards"][name] ==
                {"size_bytes": tree_files[name]["lfs_size"],
                 "sha256": tree_files[name]["lfs_sha256"]},
                f"weight shard receipt changed: {name}")


def audit_completed(run_dir: Path, entries: list[dict], progress_hash: str, context: dict) -> dict:
    expected_phases = ["attempt_started", "preflight_validated", "mps_ready", "before_model_load",
                       "model_loaded", "natural_generation", "passive_generation", "no_grad_prompt",
                       "donor_no_grad_prompt", "recipient_shared_graph"]
    for coord in COORDS:
        expected_phases += [f"before_vjp_{coord}", f"after_vjp_{coord}"]
    for site in SITES:
        expected_phases += [f"zero_patch_{site}", f"alpha_patch_{site}"]
    expected_phases += ["details_saved"]
    require([e.get("phase") for e in entries] == expected_phases, "completed phase sequence changed")
    result = read_json(run_dir / "result.json")
    meta = read_json(run_dir / "meta.json")
    require(not (run_dir / "error.json").exists(), "success and error receipts both present")
    require(result.get("status") == "completed_numerical_feasibility_only" and
            result.get("scope") == "one old FIT pair/direction, one bin, four raw coordinates; no v2 or efficacy",
            "completed status/scope changed")
    for key, filename in (("meta_sha256", "meta.json"), ("progress_sha256", "progress.jsonl"),
                          ("details_sha256", "details.pt")):
        require(result[key] == sha(run_dir / filename), f"completed {filename} hash mismatch")
    require(result["progress_sha256"] == progress_hash, "progress receipt hash changed")
    require(meta["script_sha256"] == PIN["pilot_source"] and
            meta["protocol_sha256"] == PIN["protocol"] and
            meta["first_runner_sha256"] == PIN["first_runner"] and
            meta["frontend_helper_sha256"] == PIN["frontend"] and
            meta["first_v2_fit_pilot_sha256"] == PIN["saved_pilot"] and
            meta["masks_sha256"] == PIN["masks"] and
            meta["llama_modeling_sha256"] == PIN["llama"], "meta source hashes changed")
    require(meta["installed_sources"] == SOURCE_HASHES and
            entries[1]["installed_sources"] == SOURCE_HASHES, "installed source receipt changed")
    snapshot_receipt(meta["model_snapshot"], context)
    require(meta["versions"]["torch"] == "2.15.0" and
            meta["versions"]["transformers"] == "5.20.2", "runtime versions changed")
    require(meta["record_key"] == "animal_08|piper|fit_chosen" and
            (meta["recipient"], meta["donor"], meta["bin"], meta["alpha"]) ==
            ("beaver", "monkey", "word_0", ALPHA) and
            meta["target_coordinates"] == list(COORDS) and
            meta["wav_hashes"] == context["wav_hashes"] and
            meta["source_rows"] == context["rows"] and
            meta["saved_natural_suffix_ids"] == context["saved_suffix"] and
            meta["fresh_v2_corpus_opened"] is False, "FIT fixture or no-v2 binding changed")
    require(meta["cap_seconds"] == TIME_CAP and meta["cap_mps_driver_bytes"] == MEMORY_CAP,
            "run resource caps changed")
    require(finite_number(meta["mps_recommended_max_bytes"], "MPS recommended max") > 0 and
            0 < finite_number(meta["mps_fraction_limit"], "MPS memory fraction") <= 1,
            "invalid MPS setup receipt")
    close(meta["preflight_elapsed_seconds"], entries[1]["elapsed_seconds"], "preflight duration")
    for key in ("elapsed_seconds", "model_load_seconds"):
        require(0 <= finite_number(result[key], key) < TIME_CAP, f"invalid {key}")
    require(result["elapsed_seconds"] >= entries[-1]["elapsed_seconds"], "result precedes details receipt")
    require(0 <= finite_number(result["peak_mps_driver_bytes"], "driver peak") < MEMORY_CAP,
            "MPS driver cap exceeded")
    require(0 <= finite_number(result["peak_mps_current_bytes"], "current peak") <=
            result["peak_mps_driver_bytes"], "invalid MPS current peak")
    for item in entries[3:]:
        require(all(key in item for key in ("driver_bytes", "current_bytes", "peak_driver_bytes")),
                "missing progress MPS memory fields")
        require(0 <= finite_number(item["driver_bytes"], "progress driver") < MEMORY_CAP and
                item["driver_bytes"] <= result["peak_mps_driver_bytes"],
                "progress driver exceeds cap or reported peak")
        require(0 <= finite_number(item["current_bytes"], "progress current") <=
                result["peak_mps_current_bytes"], "progress current exceeds reported peak")
        require(0 <= finite_number(item["peak_driver_bytes"], "progress driver peak") < MEMORY_CAP and
                item["peak_driver_bytes"] <= result["peak_mps_driver_bytes"],
                "progress driver peak exceeds cap or reported peak")
    require(len(result["vjp_seconds"]) == 4 and
            all(0 <= finite_number(x, "VJP duration") < TIME_CAP for x in result["vjp_seconds"]),
            "VJP timings invalid")
    require(result["natural_suffix_ids"] == result["passive_suffix_ids"] == context["saved_suffix"],
            "natural/passive FIT suffix parity failed")
    shapes = result["passive_hook_shapes"]
    require(shapes["encoder31"] == [[1, 1500, 1280]] and
            shapes["projector"] == [[375, 3072]] and
            bool(shapes["lm29"]) and shapes["lm29"][0] == [1, context["prompt_len"], 3072],
            "passive hook shape/count changed")
    for field in ("no_grad_hook_events", "shared_graph_hook_events"):
        hook_events(result[field], context["prompt_len"], field)
    require(result["shared_graph_vjp_connected_both_sources"] is True and
            result["raw_source_vjp_dtypes"] == {s: ["torch.bfloat16"] for s in SITES} and
            result["selected_gradient_storage_dtype"] == "torch.float32" and
            result["contraction_dtype"] == "torch.float32", "VJP connectivity/dtype claim changed")
    require(result["source_shapes"] == {s: list(SHAPES[s]) for s in SITES} and
            result["terminal_shape"] == [3072], "source/target result shape changed")
    details_path = run_dir / "details.pt"
    require(details_path.is_file(), "missing gradient/edit sidecar")
    details = torch.load(details_path, map_location="cpu", weights_only=True)
    require(set(details) == {"source_states", "donor_states", "selected_gradients", "effective_edits",
                             "uniform_means", "intended_edits", "clean_terminal", "no_grad_terminal",
                             "measured_terminals"}, "sidecar field set changed")
    clean = tensor(details["clean_terminal"], (3072,), torch.float32, "clean terminal")
    no_grad = tensor(details["no_grad_terminal"], (3072,), torch.float32, "no-grad terminal")
    close_vector(result["no_grad_minus_grad_four"], (no_grad-clean)[list(COORDS)].tolist(),
                 "no-grad minus grad")
    diagnostics = {}
    for site in SITES:
        shape = SHAPES[site]
        rows = context["rows"][site]
        recipient = tensor(details["source_states"][site], shape, torch.bfloat16, f"{site} recipient")
        donor = tensor(details["donor_states"][site], shape, torch.bfloat16, f"{site} donor")
        grads = tensor(details["selected_gradients"][site], (4, len(rows), shape[1]),
                       torch.float32, f"{site} selected gradients")
        effective = tensor(details["effective_edits"][site], (len(rows), shape[1]),
                           torch.float32, f"{site} realized edit")
        mean = tensor(details["uniform_means"][site], (shape[1],), torch.float32,
                      f"{site} uniform mean")
        intended = tensor(details["intended_edits"][site], (shape[1],), torch.float32,
                          f"{site} intended edit")
        require(bool((grads.to(torch.bfloat16).float() == grads).all()),
                f"{site} gradients are not exact BF16 copies")
        require(all(float(torch.linalg.vector_norm(grads[k])) > 0 for k in COORDS),
                f"{site} zero selected gradient")
        require(result["source_hashes"][site] == tensor_sha(recipient) and
                result["donor_hashes"][site] == tensor_sha(donor) and
                result["edits"][site]["effective_sha256"] == tensor_sha(effective),
                f"{site} tensor hash mismatch")
        source_rows = recipient[rows].float()
        expected_intended = ALPHA * (donor[rows].float().mean(0) - source_rows.mean(0))
        expected_effective = (source_rows + expected_intended).to(torch.bfloat16).float() - source_rows
        expected_mean = expected_effective.mean(0)
        require(torch.equal(intended, expected_intended) and
                torch.equal(effective, expected_effective) and torch.equal(mean, expected_mean),
                f"{site} intended/realized BF16 edit mismatch")
        intended_norm = float(torch.linalg.vector_norm(intended))
        realized_norm = float(torch.linalg.vector_norm(effective))
        mean_norm = float(torch.linalg.vector_norm(mean))
        fidelity = float(torch.linalg.vector_norm(mean-intended)) / max(intended_norm, 1e-6)
        nonuniformity = float(torch.linalg.vector_norm(effective-mean)) / math.sqrt(len(rows)) / max(mean_norm, 1e-6)
        edit_info = result["edits"][site]
        for key, expected in (("intended_norm", intended_norm), ("realized_norm", realized_norm),
                              ("mean_fidelity_ratio", fidelity), ("row_nonuniformity_ratio", nonuniformity)):
            close(edit_info[key], expected, f"{site} {key}")
        require(edit_info["mean_fidelity_over_10pct"] is (fidelity > 0.1) and
                edit_info["row_nonuniformity_over_10pct"] is (nonuniformity > 0.1),
                f"{site} BF16 diagnostic flag mismatch")
        expected_pred = {"uniform_mean": [], "rowwise_oracle": [], "ideal_uniform": []}
        for k in COORDS:
            g = grads[k]
            expected_pred["uniform_mean"].append(float(torch.sum(g.sum(0) * mean)))
            expected_pred["rowwise_oracle"].append(float(torch.sum(g * effective)))
            expected_pred["ideal_uniform"].append(float(torch.sum(g.sum(0) * intended)))
        for key, vector in expected_pred.items():
            close_vector(result["predictions"][site][key], vector, f"{site} {key} prediction")
        terminals = details["measured_terminals"][site]
        require(set(terminals) == {"zero", "alpha"}, f"{site} measured terminal phases")
        zero = tensor(terminals["zero"], (3072,), torch.float32, f"{site} zero terminal")
        alpha = tensor(terminals["alpha"], (3072,), torch.float32, f"{site} alpha terminal")
        require(result["measured_terminal_hashes"][site] ==
                {"zero": tensor_sha(zero), "alpha": tensor_sha(alpha)},
                f"{site} measured terminal hash mismatch")
        measured = result["measured"][site]
        delta = (alpha-zero)[list(COORDS)].tolist()
        close_vector(measured["delta_four"], delta, f"{site} measured delta")
        close_vector(measured["zero_minus_clean_four"], (zero-clean)[list(COORDS)].tolist(),
                     f"{site} zero minus clean")
        for phase in ("zero", "alpha"):
            hook_events(measured[f"{phase}_hook_events"], context["prompt_len"], f"{site} {phase}")
        for name, pred_key in (("uniform_error", "uniform_mean"),
                               ("rowwise_error", "rowwise_oracle"),
                               ("ideal_error", "ideal_uniform")):
            expected_error = vector_error(expected_pred[pred_key], delta)
            for key, value in expected_error.items():
                close(measured[name][key], value, f"{site} {name}/{key}")
        diagnostics[site] = {"fidelity_ratio": fidelity, "row_nonuniformity_ratio": nonuniformity,
                             "uniform_relative_l2": measured["uniform_error"]["relative_l2"],
                             "rowwise_relative_l2": measured["rowwise_error"]["relative_l2"]}
    return {"status": "audited_completed_numerical_feasibility_only",
            "run_dir": str(run_dir), "result_sha256": sha(run_dir / "result.json"),
            "details_sha256": result["details_sha256"], "sites": diagnostics,
            "scope": "old FIT numerical provenance only; no efficacy or security claim"}


def audit(run_dir: Path, *, allow_external_test_dir: bool = False) -> dict:
    context = source_context()
    if not allow_external_test_dir:
        require(run_dir.resolve().is_relative_to(DEFAULT_RUNS.resolve()),
                "attempt path outside old-FIT backward run root")
    require(run_dir.is_dir(), "attempt directory missing")
    entries, progress_hash = progress_receipt(run_dir)
    error_path, result_path = run_dir / "error.json", run_dir / "result.json"
    require(error_path.exists() != result_path.exists(), "need exactly one error or result receipt")
    if error_path.exists():
        error = read_json(error_path)
        require(error.get("status") == "failed" and isinstance(error.get("error_type"), str) and
                isinstance(error.get("error"), str) and
                error.get("progress_sha256") == progress_hash, "invalid error receipt")
        require(finite_number(error.get("elapsed_seconds"), "error elapsed") >=
                entries[-1]["elapsed_seconds"], "error elapsed before progress")
        return {"status": "audited_failed_attempt_no_numerical_result",
                "run_dir": str(run_dir), "error_type": error["error_type"],
                "progress_sha256": progress_hash}
    return audit_completed(run_dir, entries, progress_hash, context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.validate_only:
            context = source_context()
            result = {"status": "auditor_sources_validated_cpu_only_no_run_opened",
                      "pilot_source_sha256": PIN["pilot_source"],
                      "protocol_sha256": PIN["protocol"],
                      "record_key": "animal_08|piper|fit_chosen", "rows": context["rows"]}
        else:
            result = audit(args.run_dir.resolve())
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "failed_closed_cpu_audit", "error_type": type(exc).__name__,
                          "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
