"""One-shot development Stage A for the frozen Qwen corpus transport.

This code is intentionally separate from FIT calibration and reads only the
development split of corpus_v1. It does not open locked-test audio or answers.
No development inference has been executed while implementing this file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "eval" / "audio_jlens"
sys.path[:0] = [str(ROOT / "models" / "qwen_omni"), str(EVAL)]
from adapter import MODEL_REVISION, TRANSFORMERS_REVISION, resolve_hook_points  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import (MAX_MPS_DRIVER_BYTES, _output_tensor, _replace_output,
                         load_pinned_thinker, record_mps_memory)  # noqa: E402
from fit_pool_capability_v1 import (MANIFEST, MANIFEST_SHA, SPEC, SPEC_SHA,
                                    classify_answer, sha, validate_tokenizer)  # noqa: E402
from fit_rank16_transport_v1 import (SITES, SITE_NAMES, TARGET,
                                     atomic_json_save, atomic_torch_save,
                                     case_batch, rows_for_case)  # noqa: E402
from fit_uniform_baselines_v1 import ridge_predict  # noqa: E402
from fit_uniform_calibration_v1 import (add_uniform_bins, clean_forward,
                                        logits_from_residual, patched_forward,
                                        prefix_score)  # noqa: E402
from development_transport_metrics_v1 import summarize_primary  # noqa: E402


PROTOCOL = HERE / "PRE_DEV_EVAL_PROTOCOL_V1.md"
SCORING = HERE / "SCORING_V2.json"
SCORING_SHA = "e7ea18cd0d5e2638bec64b8733049a5a2754ecd674080d93766db8409b46dd9b"
LENS = HERE / "fit_rank16_transport_v1.lens.pt"
BASELINES = HERE / "fit_uniform_baselines_v1.pt"
BASELINE_REPORT = HERE / "fit_uniform_baselines_v1.json"
CALIBRATION_RAW = HERE / "fit_uniform_calibration_v1.raw.pt"
CALIBRATION_REPORT = HERE / "fit_uniform_calibration_v1.report.json"
PROTOCOL_SHA = "f6bbc9b61fb846ee1ca586ab4863d79f43ecc43d1d5a66c9fb6c38653077564a"
LENS_SHA = "fef84a20f437c37ca665e4026bb3645ccd558e549c7ffe2adff20e6a867ba12c"
BASELINES_SHA = "d84c358d2756aa7531cd8319884f14085d68c8e7fe84d3d85737eefd606385cf"
BASELINE_REPORT_SHA = "8ca3859fb1ecbd902c29b36c1dc9396570d0dba5af63d5045aed1c482ad39a67"
CALIBRATION_RAW_SHA = "76085f0dab8f37b6052024b582ff34533cac3d1c579b27b39a631c536d9cd110"
CALIBRATION_REPORT_SHA = "b130aa4af85a807671f537aaabbaddf922bc8f2632d8721c9188c2b6aa8e7bfa"
ALPHAS = (0.0, 0.005, 0.01, 0.025, 0.05, 1.0)
MAX_NEW_TOKENS = 16
N_CASES = 96


def load_development_inputs() -> tuple[dict, dict, list[dict]]:
    if sha(SPEC) != SPEC_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Frozen corpus spec/manifest changed")
    spec, manifest = json.loads(SPEC.read_text()), json.loads(MANIFEST.read_text())
    selected = [p for p in manifest["pairs"] if p["split"] == "development"]
    if len(selected) != 48:
        raise ValueError("Development must contain 48 paired clip groups")
    pair_specs = {p["pair_id"]: p for p in spec["word_pairs"]["development"]}
    templates = {t["id"] for t in spec["templates"]["development"]}
    engines = set(spec["render"]["voices"])
    cases = []
    seen_groups = set()
    for pair in selected:
        key = (pair["pair_id"], pair["phrasing_id"], pair["engine"])
        if (key in seen_groups or pair["pair_id"] not in pair_specs
                or pair["phrasing_id"] not in templates or pair["engine"] not in engines):
            raise ValueError("Development group allocation drifted")
        seen_groups.add(key)
        chosen = pair_specs[pair["pair_id"]]
        if pair["words"] != chosen["words"] or not pair["outside_slot_pcm_identical"]:
            raise ValueError("Development word/pair PCM metadata drifted")
        pcm = {}
        for word in pair["words"]:
            info = pair["outputs"][word]
            path = (ROOT / info["path"]).resolve()
            if not path.is_relative_to((HERE / "rendered" / "clips").resolve()) or sha(path) != info["file_sha256"]:
                raise ValueError("Development clip path/hash drifted")
            samples, rate = sf.read(path, dtype="int16")
            if rate != 16000 or samples.ndim != 1 or len(samples) != info["frames"]:
                raise ValueError("Development clip WAV drifted")
            pcm[word] = samples
            foil = next(w for w in pair["words"] if w != word)
            cases.append({
                "case_id": f"{pair['pair_id']}/{pair['phrasing_id']}/{pair['engine']}/{word}",
                "pair_id": pair["pair_id"], "category": chosen["category"],
                "template": pair["phrasing_id"], "engine": pair["engine"],
                "word": word, "foil_word": foil,
                "audio_path": str(path.relative_to(ROOT)),
                "audio_sha256": info["file_sha256"],
                "expected_aliases": spec["aliases"][word],
            })
        a, b = [pcm[w] for w in pair["words"]]
        start, end = pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"]
        if (len(a) != len(b) or not np.array_equal(a[:start], b[:start])
                or not np.array_equal(a[end:], b[end:])
                or np.array_equal(a[start:end], b[start:end])):
            raise ValueError("Development exact outside-word PCM invariant failed")
    if (len(cases) != N_CASES or len({c["case_id"] for c in cases}) != N_CASES
            or len({c["pair_id"] for c in cases}) != 12):
        raise ValueError("Development planned denominator changed")
    return spec, manifest, sorted(cases, key=lambda case: case["case_id"])


def load_frozen_fit_artifacts(spec: dict) -> tuple[dict, dict, dict]:
    if sha(SCORING) != SCORING_SHA:
        raise ValueError("V2 scoring artifact changed")
    for path, expected in ((PROTOCOL, PROTOCOL_SHA), (LENS, LENS_SHA),
                           (BASELINES, BASELINES_SHA),
                           (BASELINE_REPORT, BASELINE_REPORT_SHA),
                           (CALIBRATION_RAW, CALIBRATION_RAW_SHA),
                           (CALIBRATION_REPORT, CALIBRATION_REPORT_SHA)):
        if sha(path) != expected:
            raise ValueError(f"Frozen FIT artifact changed: {path}")
    policy = json.loads(SCORING.read_text())
    lens = torch.load(LENS, map_location="cpu", weights_only=True)
    baseline = torch.load(BASELINES, map_location="cpu", weights_only=True)
    baseline_report = json.loads(BASELINE_REPORT.read_text())
    if (lens["status"] != "fit_only_corpus_transport_no_efficacy_evaluation"
            or baseline["status"] != "fit_only_comparators_no_development_or_test"
            or baseline["metadata"]["protocol_sha256"] != PROTOCOL_SHA
            or baseline["metadata"]["lens_sha256"] != LENS_SHA
            or baseline["metadata"]["calibration_sha256"] != CALIBRATION_RAW_SHA
            or baseline["metadata"]["calibration_report_sha256"] != CALIBRATION_REPORT_SHA
            or baseline_report["baseline_sha256"] != BASELINES_SHA
            or baseline["metadata"]["n_fit"] != 48
            or baseline["metadata"]["alpha_trained"] != 0.025):
        raise ValueError("FIT lens/comparator/protocol provenance mismatch")
    if policy["eos_token_id"] != spec["tokenizer_eos_id"]:
        raise ValueError("EOS policy drifted")
    return policy, lens, baseline


def score_ids(policy: dict, spec: dict, word: str, foil: str) -> dict:
    v2_expected = policy["words"][word]["first_token_ids"]
    v2_foil = policy["words"][foil]["first_token_ids"]
    v1_expected = [a["token_id"] for a in spec["aliases"][word]]
    v1_foil = [a["token_id"] for a in spec["aliases"][foil]]
    if set(v2_expected) & set(v2_foil) or set(v1_expected) & set(v1_foil):
        raise ValueError("Candidate first-token classes overlap")
    return {"v2_expected": v2_expected, "v2_foil": v2_foil,
            "v1_expected": v1_expected, "v1_foil": v1_foil}


def classify_generated(raw: str, token_ids: list[int], tokenizer,
                       expected_word: str, aliases: list[dict], eos_id: int) -> dict:
    """Treat every nonterminal special token or early EOS as incomplete."""
    result = classify_answer(raw, token_ids, eos_id, expected_word, aliases)
    body = token_ids[:-1] if token_ids and token_ids[-1] == eos_id else token_ids
    special = set(int(token) for token in tokenizer.all_special_ids)
    nonterminal_special = sorted(set(body) & special)
    early_eos = eos_id in body
    valid_structure = bool(token_ids and token_ids[-1] == eos_id
                           and not nonterminal_special and not early_eos)
    result.update({
        "valid_token_structure": valid_structure,
        "early_eos": early_eos,
        "nonterminal_special_token_ids": nonterminal_special,
        "strict_word_plus_eos": bool(result["strict_exact_word"] and valid_structure),
        "semantic_word_plus_eos": bool(result["semantic_one_word"] and valid_structure),
    })
    return result


def generation_result(model, processor, batch, case: dict, policy: dict,
                      source_module=None, rows=None, delta=None,
                      alpha: float = 0.0, expected_source_shape=None) -> dict:
    calls = {"source": 0}
    handle = None
    if source_module is not None:
        def hook(_module, _inputs, output):
            calls["source"] += 1
            state = _output_tensor(output)
            if tuple(state.shape) != tuple(expected_source_shape):
                raise ValueError("Generation patch source shape changed")
            return _replace_output(output, add_uniform_bins(
                state, rows, delta, alpha
            ))
        handle = source_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result = model.generate(
                **batch, do_sample=False, use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=policy["eos_token_id"],
                pad_token_id=policy["eos_token_id"],
                return_dict_in_generate=True, output_scores=False,
            )
    finally:
        if handle is not None:
            handle.remove()
    if source_module is not None and calls["source"] != 1:
        raise ValueError("Generation audio source hook did not fire exactly once")
    token_ids = result.sequences[0, batch["input_ids"].shape[1]:].tolist()
    raw = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    classification = classify_generated(
        raw, token_ids, processor.tokenizer, case["word"],
        case["expected_aliases"], policy["eos_token_id"],
    )
    first = token_ids[0] if token_ids else None
    # V2 classes alone decide the descriptive prefix label.
    expected = set(policy["words"][case["word"]]["first_token_ids"])
    foil = set(policy["words"][case["foil_word"]]["first_token_ids"])
    prefix_class = "expected" if first in expected else "foil" if first in foil else "unclassified"
    indicated = case["word"] if prefix_class == "expected" else case["foil_word"]
    pure = bool(prefix_class != "unclassified"
                and classification["valid_token_structure"]
                and raw.strip().casefold() == indicated.casefold())
    return {
        "token_ids": token_ids, "decoded_raw": raw,
        "decoded_with_special_tokens": processor.tokenizer.decode(
            token_ids, skip_special_tokens=False),
        "classification": classification,
        "semantic_word_plus_eos": classification["semantic_word_plus_eos"],
        "prefix_class": prefix_class,
        "prefix_class_complete_answer_match": pure,
        "audio_source_hook_calls": calls["source"],
    }


def expected_vs_foil(model, residual: torch.Tensor, ids: dict) -> float:
    logits = logits_from_residual(model, residual.to(model.device)).detach().float().cpu()
    return prefix_score(logits, ids["v2_expected"], ids["v2_foil"])


def predictions(model, clean_h: torch.Tensor, basis: torch.Tensor,
                q_by_method: dict[str, torch.Tensor], alpha: float,
                ids: dict) -> dict[str, float]:
    baseline = expected_vs_foil(model, clean_h, ids)
    output = {"zero": 0.0}
    for name, coordinates in q_by_method.items():
        h = clean_h + alpha * (basis @ coordinates.float())
        # Expected-minus-foil is the negative of donor-minus-recipient.
        output[name] = -(expected_vs_foil(model, h, ids) - baseline)
    return output


def comparator_coordinates(lens: dict, baseline: dict, site: str,
                           delta: torch.Tensor, rows: list[list[int]]) -> dict:
    j = lens["jacobians"][site]
    if j.shape != (6, 16, delta.shape[-1]):
        raise ValueError("Saved FIT map width changed")
    weighted = torch.tensor([len(group) for group in rows], dtype=torch.float64)
    if bool((weighted <= 0).any()):
        raise ValueError("Empty development source bin")
    global_x = (delta.double() * weighted[:, None]).sum(0) / weighted.sum()
    return {
        "map": torch.einsum("bkd,bd->k", j.double(), delta.double()),
        "global": ridge_predict(baseline["models"][site]["global"], global_x[None])[0],
        "six_bin": ridge_predict(baseline["models"][site]["six_bin"],
                                 delta.double().flatten()[None])[0],
    }


def absolute_anchor_improvement(model, lens: dict, site: str,
                                category: str, source: torch.Tensor,
                                rows: list[list[int]], ids: dict) -> dict:
    anchor = lens["category_means"][category]
    z = torch.stack([source[group].float().mean(dim=0).cpu() for group in rows])
    delta = z - anchor["sources"][site]
    q = torch.einsum("bkd,bd->k", lens["jacobians"][site], delta)
    h_const = anchor["target"]
    h_abs = h_const + lens["basis"] @ q
    score_const = expected_vs_foil(model, h_const, ids)
    score_abs = expected_vs_foil(model, h_abs, ids)
    return {"absolute_anchor_improved": bool(score_abs > score_const),
            "absolute_affine_expected_vs_foil_v2": score_abs,
            "constant_anchor_expected_vs_foil_v2": score_const}


def metadata(spec: dict, cases: list[dict]) -> dict:
    return {"status": "development_stage_a_once_no_locked_test",
            "script_sha256": sha(Path(__file__)),
            "metrics_script_sha256": sha(HERE / "development_transport_metrics_v1.py"),
            "protocol_sha256": sha(PROTOCOL),
            "spec_sha256": SPEC_SHA, "manifest_sha256": MANIFEST_SHA,
            "scoring_v2_sha256": SCORING_SHA,
            "lens_sha256": LENS_SHA, "baselines_sha256": BASELINES_SHA,
            "baseline_report_sha256": BASELINE_REPORT_SHA,
            "calibration_raw_sha256": CALIBRATION_RAW_SHA,
            "calibration_report_sha256": CALIBRATION_REPORT_SHA,
            "model_revision": MODEL_REVISION,
            "alphas": list(ALPHAS),
            "case_ids": [c["case_id"] for c in cases],
            "planned": N_CASES,
            "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-input-only", action="store_true")
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "development_transport_stage_a_v1")
    args = parser.parse_args()
    if sha(SPEC) != SPEC_SHA:
        raise ValueError("Frozen corpus spec changed before development input read")
    policy, lens, baseline = load_frozen_fit_artifacts(json.loads(SPEC.read_text()))
    spec, manifest, cases = load_development_inputs()
    validate_tokenizer(spec)
    meta = metadata(spec, cases)
    if args.validate_input_only:
        print(json.dumps({"status": "development_inputs_validated_no_model_forward",
                          "planned": len(cases), "metadata": meta}, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 development requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers revision changed")
    prefix = args.output_prefix.resolve()
    clean_path = prefix.with_suffix(".clean.pt")
    progress_path = prefix.with_suffix(".progress.pt")
    final_path = prefix.with_suffix(".result.pt")
    summary_path = prefix.with_suffix(".summary.json")
    memory_path = prefix.with_suffix(".memory.jsonl")
    if final_path.exists() or summary_path.exists():
        raise FileExistsError("Development Stage A refuses overwrite")
    if not memory_path.exists():
        memory_path.write_text("")
    torch.mps.set_per_process_memory_fraction(
        mps_cap_fraction(int(torch.mps.recommended_max_memory())))
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32 or processor.tokenizer.eos_token_id != policy["eos_token_id"]:
        raise ValueError("Development model dtype/EOS changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record_mps_memory(memory_path, "after_model_load")
    points = {p.name: p.module for p in resolve_hook_points(
        model, audio_layers=(31,), text_layers=(35,))}
    modules = {site: points[SITE_NAMES[site]] for site in SITES}
    modules[TARGET] = points[TARGET]
    groups = {(p["pair_id"], p["phrasing_id"], p["engine"]): p
              for p in manifest["pairs"] if p["split"] == "development"}
    group = {"system_prompt": spec["system_prompt"], "groups": groups}
    if clean_path.exists():
        saved = torch.load(clean_path, map_location="cpu", weights_only=True)
        if saved["metadata"] != meta:
            raise ValueError("Clean checkpoint provenance changed")
        clean_rows = saved["records"]
    else:
        clean_rows = []
    if [r["case_id"] for r in clean_rows] != [c["case_id"] for c in cases[:len(clean_rows)]]:
        raise ValueError("Clean checkpoint is not frozen prefix")
    for i in range(len(clean_rows), N_CASES):
        case = cases[i]
        batch, row_map, intervals = case_batch(model, processor, case, group)
        captured, raw_logits = clean_forward(model, batch, modules)
        rows = rows_for_case(row_map, intervals, captured)
        h = captured[TARGET][0, -1].float().cpu().contiguous()
        reconstructed = logits_from_residual(model, h.to(model.device)).detach().float().cpu()
        if (reconstructed - raw_logits).abs().max() > 1e-3:
            raise ValueError("Development clean residual/logits parity failed")
        generation = generation_result(model, processor, batch, case, policy)
        record_mps_memory(memory_path, f"clean_{i}")
        clean_rows.append({"case_id": case["case_id"],
                           "h": h, "source": {site: captured[site].float().cpu().contiguous()
                                            for site in SITES},
                           "rows": rows, "generation": generation,
                           "raw_top_token_id": int(raw_logits.argmax())})
        atomic_torch_save(clean_path, {"metadata": meta, "records": clean_rows})
        if (i+1) % 8 == 0:
            print(f"captured {i+1}/{N_CASES} development clean contexts", flush=True)
    by_key = {(c["pair_id"], c["template"], c["engine"], c["word"]): i
              for i, c in enumerate(cases)}
    if progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        if saved["metadata"] != meta or saved["clean_sha256"] != sha(clean_path):
            raise ValueError("Development progress provenance changed")
        output_rows = saved["records"]
    else:
        output_rows = []
    if [r["case_id"] for r in output_rows] != [c["case_id"] for c in cases[:len(output_rows)]]:
        raise ValueError("Development progress is not frozen prefix")
    for i in range(len(output_rows), N_CASES):
        case = cases[i]
        donor_i = by_key[(case["pair_id"], case["template"], case["engine"], case["foil_word"])]
        own, donor = clean_rows[i], clean_rows[donor_i]
        ids = score_ids(policy, spec, case["word"], case["foil_word"])
        batch, row_map, intervals = case_batch(model, processor, case, group)
        if rows_for_case(row_map, intervals, own["source"]) != own["rows"]:
            raise ValueError("Development row map changed")
        current, _ = clean_forward(model, batch, modules)
        if float((current[TARGET][0, -1].float().cpu() - own["h"]).abs().max()) > 1e-3:
            raise ValueError("Development clean terminal state changed after checkpoint")
        for site in SITES:
            if (tuple(current[site].shape) != tuple(own["source"][site].shape)
                    or float((current[site].float().cpu() - own["source"][site]).abs().max()) > 1e-3):
                raise ValueError("Development clean source changed after checkpoint")
        record_mps_memory(memory_path, f"case_{i}_clean_recheck")
        clean_expected_score = expected_vs_foil(model, own["h"], ids)
        site_results = {}
        for site in SITES:
            if own["source"][site].shape != donor["source"][site].shape or own["rows"][site] != donor["rows"][site]:
                raise ValueError("Development paired source shape/rows differ")
            delta = torch.stack([
                donor["source"][site][group_rows].mean(0)
                - own["source"][site][group_rows].mean(0)
                for group_rows in own["rows"][site]
            ])
            q = comparator_coordinates(lens, baseline, site, delta, own["rows"][site])
            source_edit_norm_per_alpha = sum(
                len(group_rows) * float(delta[b].square().sum())
                for b, group_rows in enumerate(own["rows"][site])
            ) ** .5
            absolute = absolute_anchor_improvement(
                model, lens, site, case["category"], own["source"][site],
                own["rows"][site], ids)
            alpha_rows = []
            clean_v1 = prefix_score(logits_from_residual(
                model, own["h"].to(model.device)).detach().float().cpu(),
                ids["v1_foil"], ids["v1_expected"])
            for alpha in ALPHAS:
                h_patch, raw_logits, calls = patched_forward(
                    model, batch, modules[site], modules[TARGET],
                    own["rows"][site], delta, alpha)
                measured = -prefix_score(raw_logits, ids["v2_expected"], ids["v2_foil"]) + clean_expected_score
                # measured is donor-minus-recipient change, matching predictions.
                measured_v1 = (prefix_score(raw_logits, ids["v1_foil"], ids["v1_expected"])
                               - clean_v1)
                pred = predictions(model, own["h"], lens["basis"], q, alpha, ids)
                generated = generation_result(
                    model, processor, batch, case, policy,
                    modules[site], own["rows"][site], delta, alpha,
                    tuple(own["source"][site].shape))
                terminal_delta = h_patch - own["h"]
                projected_coordinates = lens["basis"].T @ terminal_delta
                projected_norm = float(projected_coordinates.norm())
                actual_norm = float(terminal_delta.norm())
                if alpha == 0 and (abs(measured) > 1e-4 or actual_norm > 1e-3
                                   or generated["token_ids"] != own["generation"]["token_ids"]):
                    raise ValueError("Development alpha-zero score/answer invariant failed")
                alpha_rows.append({"alpha": alpha,
                                   "measured_v2_delta_score": measured,
                                   "measured_v1_delta_score": measured_v1,
                                   "predicted_v2_delta_score": pred,
                                   "terminal_delta_norm": actual_norm,
                                   "rank16_capture_fraction": (
                                       projected_norm**2 / actual_norm**2 if actual_norm > 0 else None),
                                   "effective_source_edit_frobenius_norm": abs(alpha) * source_edit_norm_per_alpha,
                                   "generation": generated, "hook_calls": calls})
                record_mps_memory(memory_path, f"case_{i}_{site}_alpha_{alpha}")
            site_results[site] = {**absolute, "alphas": alpha_rows,
                                  "source_rows": own["rows"][site],
                                  "source_shape": list(own["source"][site].shape),
                                  "delta_bin_norms": [float(row.norm()) for row in delta]}
        output_rows.append({"case_id": case["case_id"], "pair_id": case["pair_id"],
                            "category": case["category"], "engine": case["engine"],
                            "template": case["template"], "word": case["word"],
                            "foil_word": case["foil_word"],
                            "audio_sha256": case["audio_sha256"],
                            "clean": own["generation"], "sites": site_results})
        atomic_torch_save(progress_path, {"metadata": meta,
                                          "clean_sha256": sha(clean_path),
                                          "records": output_rows})
        if (i+1) % 8 == 0:
            print(f"measured {i+1}/{N_CASES} development directional contexts", flush=True)
    summary = summarize_primary(output_rows, baseline["numerical_floor_v2_by_site"])
    result = {"metadata": meta, "clean_sha256": sha(clean_path),
              "baseline_sha256": sha(BASELINES), "records": output_rows,
              "summary": summary}
    atomic_torch_save(final_path, result)
    atomic_json_save(summary_path, {"metadata": meta, "summary": summary,
                                    "result_sha256": sha(final_path),
                                    "progress_sha256": sha(progress_path),
                                    "memory_sha256": sha(memory_path)})
    print(json.dumps(summary, indent=2))


def record_terminal_failure(error: Exception) -> None:
    """Keep a reviewable all-planned failure when execution stops early."""
    if "--validate-input-only" in sys.argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "development_transport_stage_a_v1")
    args, _ = parser.parse_known_args()
    prefix = args.output_prefix.resolve()
    index = 0
    while prefix.with_suffix(f".failure.{index:04d}.json").exists():
        index += 1
    def checkpoint_info(suffix: str) -> dict:
        path = prefix.with_suffix(suffix)
        if not path.exists():
            return {"exists": False, "completed": 0, "sha256": None}
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            completed = len(saved.get("records", []))
        except Exception:
            completed = None
        return {"exists": True, "completed": completed, "sha256": sha(path)}
    atomic_json_save(prefix.with_suffix(f".failure.{index:04d}.json"), {
        "status": "development_stage_a_aborted_not_a_pass",
        "planned_directions": N_CASES,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "clean_checkpoint": checkpoint_info(".clean.pt"),
        "progress_checkpoint": checkpoint_info(".progress.pt"),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record_terminal_failure(error)
        raise
