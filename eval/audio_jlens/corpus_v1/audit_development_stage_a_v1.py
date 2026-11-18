"""Independent read-only audit of a completed Qwen corpus Stage A result.

This module does not import the Stage A runner or metrics implementation. It
recomputes denominators, numerical gates, and pair bootstrap from saved data.
It opens development outputs only when invoked after Stage A completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE / "development_transport_stage_a_v1"
SITES = ("encoder31", "connector")
METHODS = ("map", "zero", "global", "six_bin")
ALPHAS = (0.0, .005, .01, .025, .05, 1.0)
SMALL = (.005, .01, .025)
N_CASES = 96
N_CONDITIONS = 288
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261118
EOS = 151645
CAP_BYTES = 30 * 1024**3
FROZEN = {
    "script_sha256": ("development_transport_stage_a_v1.py", "6f314b7a2322ff33ab0fdccb190a805702f26e59197f30460fcd36b8faac8745"),
    "metrics_script_sha256": ("development_transport_metrics_v1.py", "448f619332a9f96ea58358dcb07cab37f332b818817033b6ce449621d599b67c"),
    "protocol_sha256": ("PRE_DEV_EVAL_PROTOCOL_V1.md", "f6bbc9b61fb846ee1ca586ab4863d79f43ecc43d1d5a66c9fb6c38653077564a"),
    "spec_sha256": ("SPEC_V1.json", "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265"),
    "manifest_sha256": ("rendered/manifest.json", "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"),
    "scoring_v2_sha256": ("SCORING_V2.json", "e7ea18cd0d5e2638bec64b8733049a5a2754ecd674080d93766db8409b46dd9b"),
    "lens_sha256": ("fit_rank16_transport_v1.lens.pt", "fef84a20f437c37ca665e4026bb3645ccd558e549c7ffe2adff20e6a867ba12c"),
    "baselines_sha256": ("fit_uniform_baselines_v1.pt", "d84c358d2756aa7531cd8319884f14085d68c8e7fe84d3d85737eefd606385cf"),
    "baseline_report_sha256": ("fit_uniform_baselines_v1.json", "8ca3859fb1ecbd902c29b36c1dc9396570d0dba5af63d5045aed1c482ad39a67"),
    "calibration_raw_sha256": ("fit_uniform_calibration_v1.raw.pt", "76085f0dab8f37b6052024b582ff34533cac3d1c579b27b39a631c536d9cd110"),
    "calibration_report_sha256": ("fit_uniform_calibration_v1.report.json", "b130aa4af85a807671f537aaabbaddf922bc8f2632d8721c9188c2b6aa8e7bfa"),
}


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual, expected, label: str, tolerance: float = 1e-7) -> None:
    if actual is None or expected is None:
        require(actual is expected, f"{label}: missing/extra value")
    else:
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
                f"{label}: nonfinite")
        require(math.isclose(float(actual), float(expected),
                             rel_tol=tolerance, abs_tol=tolerance),
                f"{label}: {actual} != {expected}")


def expected_development_cases(spec: dict, manifest: dict) -> tuple[list[str], dict]:
    pairs = {p["pair_id"]: p for p in spec["word_pairs"]["development"]}
    templates = {p["id"] for p in spec["templates"]["development"]}
    engines = set(spec["render"]["voices"])
    selected = [p for p in manifest["pairs"] if p["split"] == "development"]
    require(len(selected) == 48 and len(pairs) == 12, "Development paired allocation changed")
    expected = {}
    seen = set()
    for group in selected:
        pair_id, template, engine = group["pair_id"], group["phrasing_id"], group["engine"]
        key = (pair_id, template, engine)
        require(key not in seen and pair_id in pairs and template in templates
                and engine in engines, "Unexpected/duplicate development group")
        seen.add(key)
        pair = pairs[pair_id]
        require(group["words"] == pair["words"]
                and group["outside_slot_pcm_identical"]
                and group["inside_slot_pcm_differs"], "Development word/PCM metadata drifted")
        pcm = {}
        for word in group["words"]:
            info = group["outputs"][word]
            path = (ROOT / info["path"]).resolve()
            require(path.is_relative_to((HERE / "rendered" / "clips").resolve())
                    and sha(path) == info["file_sha256"], "Development WAV hash/path changed")
            samples, rate = sf.read(path, dtype="int16")
            require(rate == 16000 and samples.ndim == 1 and len(samples) == info["frames"],
                    "Development WAV shape/rate changed")
            pcm[word] = samples
            case_id = f"{pair_id}/{template}/{engine}/{word}"
            expected[case_id] = {"pair_id": pair_id,
                                 "category": pair["category"],
                                 "template": template, "engine": engine,
                                 "word": word,
                                 "foil_word": next(w for w in pair["words"] if w != word),
                                 "audio_sha256": info["file_sha256"],
                                 "expected_aliases": spec["aliases"][word]}
        a, b = [pcm[word] for word in group["words"]]
        start, end = group["word_slot_start_sample"], group["word_slot_end_sample_exclusive"]
        require(len(a) == len(b) and np.array_equal(a[:start], b[:start])
                and np.array_equal(a[end:], b[end:])
                and not np.array_equal(a[start:end], b[start:end]),
                "Development outside-word PCM changed")
    ids = sorted(expected)
    require(len(ids) == N_CASES and len(set(ids)) == N_CASES,
            "Development planned IDs not 96 unique")
    return ids, expected


def verify_generation(generation: dict, word: str, foil: str,
                      policy: dict, tokenizer, aliases: list[dict],
                      *, clean: bool) -> None:
    tokens = generation["token_ids"]
    require(isinstance(tokens, list) and all(isinstance(x, int) for x in tokens),
            "Generation token IDs malformed")
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    with_special = tokenizer.decode(tokens, skip_special_tokens=False)
    require(raw == generation["decoded_raw"]
            and with_special == generation["decoded_with_special_tokens"],
            "Generated text does not decode from saved IDs")
    body = tokens[:-1] if tokens and tokens[-1] == EOS else tokens
    invalid_specials = sorted(set(body) & set(tokenizer.all_special_ids))
    structure = bool(tokens and tokens[-1] == EOS and not invalid_specials and EOS not in body)
    strict = raw.strip().casefold() == word.casefold()
    semantic_text = raw.strip()
    if semantic_text.endswith((".", "?", "!")):
        semantic_text = semantic_text[:-1].strip()
    semantic = semantic_text.casefold() == word.casefold()
    classification = generation["classification"]
    alias_ids = sorted({int(alias["token_id"]) for alias in aliases})
    require(classification["decoded_trimmed"] == raw.strip()
            and classification["valid_token_structure"] == structure
            and classification["nonterminal_special_token_ids"] == invalid_specials
            and classification["early_eos"] == (EOS in body)
            and classification["stopped_on_eos"] == bool(tokens and tokens[-1] == EOS)
            and classification["strict_exact_word"] == strict
            and classification["semantic_one_word"] == semantic
            and classification["first_token_id"] == (tokens[0] if tokens else None)
            and classification["expected_alias_token_ids"] == alias_ids
            and classification["first_token_in_expected_aliases"] == bool(tokens and tokens[0] in alias_ids)
            and classification["strict_word_plus_eos"] == (strict and structure)
            and classification["semantic_word_plus_eos"] == (semantic and structure)
            and generation["semantic_word_plus_eos"] == (semantic and structure),
            "Generated complete-answer classification disagrees with tokens")
    first = tokens[0] if tokens else None
    expected_ids = set(policy["words"][word]["first_token_ids"])
    foil_ids = set(policy["words"][foil]["first_token_ids"])
    require(not expected_ids & foil_ids, "V2 first-token pair classes overlap")
    prefix_class = "expected" if first in expected_ids else "foil" if first in foil_ids else "unclassified"
    indicated = word if prefix_class == "expected" else foil
    pure = bool(prefix_class != "unclassified" and structure
                and raw.strip().casefold() == indicated.casefold())
    require(generation["prefix_class"] == prefix_class
            and generation["prefix_class_complete_answer_match"] == pure,
            "Prefix class/purity disagree with complete tokens")
    require(generation["audio_source_hook_calls"] == (0 if clean else 1),
            "Generation hook count changed")


def audit_case_rows(records: list[dict], clean: list[dict], expected: dict,
                    policy: dict, tokenizer) -> dict:
    require(len(records) == len(clean) == N_CASES, "Completed result lacks 96 directions")
    ids = sorted(expected)
    require([r["case_id"] for r in records] == ids
            and [r["case_id"] for r in clean] == ids,
            "Result/clean IDs differ from frozen planned order")
    by_id = {row["case_id"]: i for i, row in enumerate(clean)}
    pair_counts = {}
    for i, record in enumerate(records):
        case_id = ids[i]
        case = expected[case_id]
        for key in ("pair_id", "category", "engine", "template", "word", "foil_word", "audio_sha256"):
            require(record[key] == case[key], f"{case_id}: {key} differs from manifest")
        pair_counts[case["pair_id"]] = pair_counts.get(case["pair_id"], 0) + 1
        verify_generation(record["clean"], case["word"], case["foil_word"],
                          policy, tokenizer, case["expected_aliases"], clean=True)
        require(record["clean"] == clean[i]["generation"],
                f"{case_id}: clean generation differs from clean checkpoint")
        donor_id = "/".join(case_id.split("/")[:-1] + [case["foil_word"]])
        donor = clean[by_id[donor_id]]
        recipient = clean[i]
        require(set(record["sites"]) == set(SITES), f"{case_id}: sites changed")
        for site in SITES:
            site_row = record["sites"][site]
            source = recipient["source"][site]
            donor_source = donor["source"][site]
            rows = recipient["rows"][site]
            require(tuple(source.shape) == tuple(donor_source.shape)
                    and list(source.shape) == site_row["source_shape"]
                    and rows == donor["rows"][site]
                    and rows == site_row["source_rows"],
                    f"{case_id}/{site}: paired source shape/rows changed")
            require(len(rows) == 6 and all(group for group in rows),
                    f"{case_id}/{site}: source bin missing")
            flat = [index for group in rows for index in group]
            require(len(set(flat)) == len(flat) and min(flat) >= 0
                    and max(flat) < source.shape[0],
                    f"{case_id}/{site}: selected source rows invalid")
            delta = torch.stack([
                donor_source[group].mean(0) - source[group].mean(0)
                for group in rows
            ])
            require(len(site_row["delta_bin_norms"]) == 6,
                    f"{case_id}/{site}: delta-bin norm count changed")
            for b, value in enumerate(site_row["delta_bin_norms"]):
                near(value, float(delta[b].norm()), f"{case_id}/{site}/delta_norm_{b}", 1e-5)
            edit_unit_norm = sum(len(group) * float(delta[b].square().sum())
                                 for b, group in enumerate(rows)) ** .5
            alpha_rows = site_row["alphas"]
            require([float(row["alpha"]) for row in alpha_rows] == list(ALPHAS),
                    f"{case_id}/{site}: alpha grid/order changed")
            for row in alpha_rows:
                alpha = float(row["alpha"])
                near(row["effective_source_edit_frobenius_norm"],
                     alpha * edit_unit_norm, f"{case_id}/{site}/edit_norm_{alpha}", 1e-5)
                require(row["hook_calls"] == {"source": 1, "target": 1},
                        f"{case_id}/{site}: score hook count changed")
                verify_generation(row["generation"], case["word"], case["foil_word"],
                                  policy, tokenizer, case["expected_aliases"], clean=False)
                require(set(row["predicted_v2_delta_score"]) == set(METHODS)
                        and row["predicted_v2_delta_score"]["zero"] == 0,
                        f"{case_id}/{site}: comparator names/zero changed")
                require(all(math.isfinite(float(row["predicted_v2_delta_score"][name]))
                            for name in METHODS)
                        and math.isfinite(float(row["measured_v2_delta_score"]))
                        and math.isfinite(float(row["measured_v1_delta_score"]))
                        and math.isfinite(float(row["terminal_delta_norm"])),
                        f"{case_id}/{site}: nonfinite score or terminal norm")
                if alpha == 0:
                    require(row["generation"]["token_ids"] == record["clean"]["token_ids"]
                            and abs(row["measured_v2_delta_score"]) <= 1e-4
                            and row["terminal_delta_norm"] <= 1e-3,
                            f"{case_id}/{site}: alpha-zero invariant failed")
                    for name in METHODS:
                        require(abs(row["predicted_v2_delta_score"][name]) <= 1e-6,
                                f"{case_id}/{site}: alpha-zero prediction drift")
                if alpha > 0:
                    capture = row["rank16_capture_fraction"]
                    require(capture is None or (-1e-6 <= capture <= 1 + 1e-4),
                            f"{case_id}/{site}: projection capture outside [0,1]")
    require(len(pair_counts) == 12 and all(n == 8 for n in pair_counts.values()),
            "Pair-level denominator is not 12×8")
    return {"planned": N_CASES, "pairs": len(pair_counts),
            "conditions_per_site": N_CASES * len(ALPHAS),
            "small_conditions_per_site": N_CONDITIONS}


def recompute_task(records: list[dict]) -> dict:
    good = sum(bool(r["clean"]["semantic_word_plus_eos"]) for r in records)
    classified = [r for r in records if r["clean"]["prefix_class"] in ("expected", "foil")]
    pure = sum(bool(r["clean"]["prefix_class_complete_answer_match"]) for r in classified)
    purity = pure / len(classified) if classified else None
    return {"planned_directions": N_CASES, "available_directions": len(records),
            "semantic_word_plus_eos": good, "prefix_classified": len(classified),
            "prefix_class_pure": pure, "prefix_purity": purity,
            "task_fidelity_gate_passed": bool(good >= 72 and len(classified) >= 72
                                              and purity is not None and purity >= .75)}


def recompute_site(records: list[dict], site: str, floor: float) -> dict:
    errors = {name: [] for name in METHODS}
    signs = []
    improved = 0
    for record in records:
        site_row = record["sites"][site]
        require(isinstance(site_row["absolute_anchor_improved"], bool),
                "Absolute anchor result missing")
        improved += int(site_row["absolute_anchor_improved"])
        by_alpha = {float(row["alpha"]): row for row in site_row["alphas"]}
        for alpha in SMALL:
            row = by_alpha[alpha]
            observed = float(row["measured_v2_delta_score"])
            require(math.isfinite(observed), "Nonfinite measured V2 change")
            for name in METHODS:
                predicted = float(row["predicted_v2_delta_score"][name])
                require(math.isfinite(predicted), "Nonfinite predicted V2 change")
                errors[name].append(abs(predicted - observed))
            if abs(observed) > floor:
                signs.append(int(float(row["predicted_v2_delta_score"]["map"]) * observed > 0))
    require(all(len(values) == N_CONDITIONS for values in errors.values()),
            "Missing small-alpha condition")
    mae = {name: sum(values) / N_CONDITIONS for name, values in errors.items()}
    sign_rate = sum(signs) / len(signs) if signs else None
    ratios = {name: (mae["map"] / mae[name] if mae[name] > 0 else None)
              for name in ("zero", "global", "six_bin")}
    gate = bool(len(signs) >= 48 and sign_rate is not None and sign_rate >= .75
                and improved >= 68
                and all(mae[name] > 0 and mae["map"] <= .75 * mae[name]
                        for name in ("zero", "global", "six_bin")))
    return {"mae": mae, "ratios": ratios,
            "determinate": len(signs), "sign_rate": sign_rate,
            "absolute_improved": improved, "pass": gate,
            "pair_component": {
                pair: {
                    "errors": {name: sum(abs(float(row["predicted_v2_delta_score"][name])
                                             - float(row["measured_v2_delta_score"]))
                                         for record in records if record["pair_id"] == pair
                                         for row in record["sites"][site]["alphas"]
                                         if float(row["alpha"]) in SMALL)
                               for name in METHODS},
                    "determinate": sum(abs(float(row["measured_v2_delta_score"])) > floor
                                        for record in records if record["pair_id"] == pair
                                        for row in record["sites"][site]["alphas"]
                                        if float(row["alpha"]) in SMALL),
                    "signs": sum(float(row["predicted_v2_delta_score"]["map"])
                                 * float(row["measured_v2_delta_score"]) > 0
                                 for record in records if record["pair_id"] == pair
                                 for row in record["sites"][site]["alphas"]
                                 if float(row["alpha"]) in SMALL
                                 and abs(float(row["measured_v2_delta_score"])) > floor),
                    "clean_good": sum(bool(r["clean"]["semantic_word_plus_eos"])
                                      for r in records if r["pair_id"] == pair),
                    "absolute": sum(bool(r["sites"][site]["absolute_anchor_improved"])
                                    for r in records if r["pair_id"] == pair),
                }
                for pair in dict.fromkeys(r["pair_id"] for r in records)
            }}


def recompute_bootstrap(component: dict, pair_order: list[str]) -> dict:
    require(len(pair_order) == 12 and len(set(pair_order)) == 12,
            "Bootstrap pair order is not 12 unique pairs")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {key: [] for key in ("map_zero_ratio", "map_global_ratio",
                                  "map_six_bin_ratio", "sign", "clean", "absolute")}
    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(pair_order, size=12, replace=True)
        totals = {name: sum(component[p]["errors"][name] for p in chosen)
                  for name in METHODS}
        for metric, comparator in (("map_zero_ratio", "zero"),
                                   ("map_global_ratio", "global"),
                                   ("map_six_bin_ratio", "six_bin")):
            if totals[comparator] > 0:
                values[metric].append(totals["map"] / totals[comparator])
        n_sign = sum(component[p]["determinate"] for p in chosen)
        if n_sign:
            values["sign"].append(sum(component[p]["signs"] for p in chosen) / n_sign)
        values["clean"].append(sum(component[p]["clean_good"] for p in chosen) / N_CASES)
        values["absolute"].append(sum(component[p]["absolute"] for p in chosen) / N_CASES)
    return {name: {"n_valid_replicates": len(rows),
                   "interval_95": np.quantile(rows, [.025, .975]).tolist() if rows else None}
            for name, rows in values.items()}


def verify_descriptive_groups(records: list[dict], site: str, floor: float,
                              saved_groups: dict) -> None:
    for field in ("category", "engine", "template", "pair_id"):
        expected_values = sorted({str(record[field]) for record in records})
        require(set(saved_groups[field]) == set(expected_values),
                f"{site}/{field}: descriptive group keys changed")
        for value in expected_values:
            subset = [record for record in records if str(record[field]) == value]
            row = saved_groups[field][value]
            require(row["available_directions"] == len(subset)
                    and row["small_alpha_valid_conditions"] == len(subset) * 3
                    and row["invalid_conditions"] == 0,
                    f"{site}/{field}/{value}: group denominator changed")
            errors = {name: [] for name in METHODS}
            signs = []
            improved = 0
            for record in subset:
                improved += int(record["sites"][site]["absolute_anchor_improved"])
                for condition in record["sites"][site]["alphas"]:
                    if float(condition["alpha"]) not in SMALL:
                        continue
                    observed = float(condition["measured_v2_delta_score"])
                    for name in METHODS:
                        errors[name].append(abs(float(condition["predicted_v2_delta_score"][name])
                                                - observed))
                    if abs(observed) > floor:
                        signs.append(int(float(condition["predicted_v2_delta_score"]["map"])
                                         * observed > 0))
            for name in METHODS:
                near(row["mae"][name], sum(errors[name]) / len(errors[name]),
                     f"{site}/{field}/{value}/MAE_{name}")
            require(row["determinate_conditions"] == len(signs)
                    and row["absolute_anchor_improved_directions"] == improved,
                    f"{site}/{field}/{value}: sign/anchor counts changed")
            near(row["determinate_sign_agreement"],
                 sum(signs) / len(signs) if signs else None,
                 f"{site}/{field}/{value}/sign")


def verify_memory(memory_path: Path, reported_sha: str) -> dict:
    require(sha(memory_path) == reported_sha, "Memory sidecar SHA differs")
    rows = [json.loads(line) for line in memory_path.read_text().splitlines() if line.strip()]
    phases = {row["phase"] for row in rows}
    expected = {"after_model_load"}
    expected.update(f"clean_{i}" for i in range(N_CASES))
    expected.update(f"case_{i}_clean_recheck" for i in range(N_CASES))
    expected.update(f"case_{i}_{site}_alpha_{alpha}"
                    for i in range(N_CASES) for site in SITES for alpha in ALPHAS)
    require(expected.issubset(phases), "Memory log misses at least one planned forward phase")
    require(all(row["limit_driver_bytes"] == CAP_BYTES
                and not row["over_limit"]
                and row["driver_allocated_bytes"] <= CAP_BYTES
                for row in rows), "Memory cap exceeded or drifted")
    return {"samples": len(rows), "expected_unique_phases": len(expected),
            "peak_driver_gib": max(row["driver_allocated_bytes"] for row in rows) / 1024**3}


def run(prefix: Path) -> dict:
    prefix = prefix.resolve()
    result_path = prefix.with_suffix(".result.pt")
    clean_path = prefix.with_suffix(".clean.pt")
    progress_path = prefix.with_suffix(".progress.pt")
    summary_path = prefix.with_suffix(".summary.json")
    memory_path = prefix.with_suffix(".memory.jsonl")
    for path in (result_path, clean_path, progress_path, summary_path, memory_path):
        require(path.is_file(), f"Missing completed Stage A artifact: {path}")
    result = torch.load(result_path, map_location="cpu", weights_only=True)
    clean = torch.load(clean_path, map_location="cpu", weights_only=True)
    progress = torch.load(progress_path, map_location="cpu", weights_only=True)
    summary_sidecar = json.loads(summary_path.read_text())
    metadata = result["metadata"]
    for key, (relative, expected) in FROZEN.items():
        require(sha(HERE / relative) == expected and metadata[key] == expected,
                f"Frozen source/artifact hash mismatch: {key}")
    require(metadata["planned"] == N_CASES and metadata["alphas"] == list(ALPHAS)
            and metadata["max_mps_driver_bytes"] == CAP_BYTES,
            "Stage A planned constants changed")
    require(result["metadata"] == clean["metadata"] == progress["metadata"]
            == summary_sidecar["metadata"], "Stage A artifact metadata differ")
    require(sha(clean_path) == result["clean_sha256"] == progress["clean_sha256"]
            and sha(progress_path) == summary_sidecar["progress_sha256"]
            and sha(result_path) == summary_sidecar["result_sha256"],
            "Stage A checkpoint/result hashes disagree")
    require(result["baseline_sha256"] == FROZEN["baselines_sha256"][1]
            and result["records"] == progress["records"],
            "Final progress/results disagree")
    spec = json.loads((HERE / "SPEC_V1.json").read_text())
    manifest = json.loads((HERE / "rendered" / "manifest.json").read_text())
    policy = json.loads((HERE / "SCORING_V2.json").read_text())
    ids, expected = expected_development_cases(spec, manifest)
    require(metadata["case_ids"] == ids, "Stage A metadata case IDs differ from manifest")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        local_files_only=True, trust_remote_code=True,
    )
    require(tokenizer.eos_token_id == EOS and policy["eos_token_id"] == EOS,
            "Pinned tokenizer EOS changed")
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        "Qwen/Qwen2.5-Omni-3B",
        revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        local_files_only=True,
    ))
    require(sha(snapshot / "tokenizer.json") == spec["tokenizer_json_sha256"]
            and sha(snapshot / "tokenizer_config.json") == spec["tokenizer_config_sha256"],
            "Pinned tokenizer bytes changed")
    cases = audit_case_rows(result["records"], clean["records"], expected,
                            policy, tokenizer)
    saved_summary = result["summary"]
    require(saved_summary == summary_sidecar["summary"], "Summary sidecar differs from result")
    task = recompute_task(result["records"])
    for key, value in task.items():
        if isinstance(value, float) or value is None:
            near(value, saved_summary["clean_task"][key], f"task/{key}")
        else:
            require(saved_summary["clean_task"][key] == value, f"task/{key} differs")
    baseline = torch.load(HERE / "fit_uniform_baselines_v1.pt", map_location="cpu", weights_only=True)
    floors = baseline["numerical_floor_v2_by_site"]
    sites = {}
    pair_order = list(dict.fromkeys(r["pair_id"] for r in result["records"]))
    for site in SITES:
        floor = float(floors[site])
        actual = recompute_site(result["records"], site, floor)
        saved = saved_summary["sites"][site]
        for name in METHODS:
            near(actual["mae"][name], saved["mae"][name], f"{site}/MAE/{name}")
        for name in ("zero", "global", "six_bin"):
            near(actual["ratios"][name], saved["map_to_comparator_mae_ratio"][name],
                 f"{site}/ratio/{name}")
        require(saved["determinate_conditions"] == actual["determinate"]
                and saved["absolute_anchor_improved_directions"] == actual["absolute_improved"]
                and saved["small_alpha_planned_conditions"] == N_CONDITIONS
                and saved["small_alpha_valid_conditions"] == N_CONDITIONS
                and saved["invalid_conditions"] == 0
                and saved["site_primary_gate_passed"] == actual["pass"],
                f"{site}: all-planned counts/gate differ")
        near(saved["determinate_sign_agreement"], actual["sign_rate"], f"{site}/sign")
        boot = recompute_bootstrap(actual["pair_component"], pair_order)
        for metric, row in boot.items():
            saved_row = saved["pair_bootstrap"][metric]
            require(saved_row["n_valid_replicates"] == row["n_valid_replicates"],
                    f"{site}/{metric}: bootstrap replicate count differs")
            if row["interval_95"] is None:
                require(saved_row["interval_95"] is None, f"{site}/{metric}: missing CI differs")
            else:
                for i in (0, 1):
                    near(saved_row["interval_95"][i], row["interval_95"][i],
                         f"{site}/{metric}/CI_{i}", 1e-6)
        verify_descriptive_groups(result["records"], site, floor,
                                  saved["descriptive_groups"])
        sites[site] = {"mae": actual["mae"], "ratios": actual["ratios"],
                       "determinate": actual["determinate"],
                       "sign_rate": actual["sign_rate"],
                       "absolute_improved": actual["absolute_improved"],
                       "site_gate": actual["pass"]}
    final_gate = bool(task["task_fidelity_gate_passed"]
                      and all(sites[s]["site_gate"] for s in SITES))
    require(final_gate == saved_summary["development_primary_gate_passed"],
            "Overall development gate differs")
    memory = verify_memory(memory_path, summary_sidecar["memory_sha256"])
    return {"status": "independent_stage_a_audit_passed",
            "result_sha256": sha(result_path), "summary_sha256": sha(summary_path),
            "cases": cases, "task": task, "sites": sites,
            "development_primary_gate_passed": final_gate,
            "memory": memory,
            "scope": "read-only Stage A artifact audit; no new model inference"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=PREFIX)
    parser.add_argument("--report", type=Path,
                        help="Optional new audit JSON; refuses overwrite")
    args = parser.parse_args()
    result = run(args.prefix)
    if args.report:
        output = args.report.resolve()
        require(not output.exists(), "Audit report already exists")
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
