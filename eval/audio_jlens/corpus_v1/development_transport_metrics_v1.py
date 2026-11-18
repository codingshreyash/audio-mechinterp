"""Pure CPU primary gate for the prospective corpus transport development run."""

from __future__ import annotations

import math

import numpy as np

SITES = ("encoder31", "connector")
METHODS = ("map", "zero", "global", "six_bin")
SMALL_ALPHAS = (0.005, 0.01, 0.025)
PLANNED = 96
MIN_CLEAN = 72
MIN_PREFIX_COVERAGE = 72
MIN_DETERMINATE = 48
MIN_ABS_IMPROVED = 68
BOOTSTRAP_SEED = 20261118
BOOTSTRAP_REPLICATES = 10000


def primary_site_stats(records: list[dict], site: str, floor: float,
                       *, require_complete: bool = True) -> dict:
    errors = {name: [] for name in METHODS}
    signs = []
    improved = 0
    invalid_absolute = 0
    for record in records:
        site_row = record.get("sites", {}).get(site)
        if site_row is None or not isinstance(site_row.get("absolute_anchor_improved"), bool):
            invalid_absolute += 1
            continue
        improved += int(site_row["absolute_anchor_improved"])
        alpha_rows = site_row.get("alphas", [])
        by_alpha = {float(row["alpha"]): row for row in alpha_rows
                    if isinstance(row, dict) and "alpha" in row}
        if len(by_alpha) != len(alpha_rows):
            continue
        for alpha in SMALL_ALPHAS:
            row = by_alpha.get(alpha)
            if row is None:
                continue
            observed = row.get("measured_v2_delta_score")
            predicted = row.get("predicted_v2_delta_score", {})
            if (not isinstance(observed, (int, float)) or not math.isfinite(observed)
                    or not isinstance(predicted, dict)
                    or set(predicted) != set(METHODS)
                    or any(not isinstance(predicted[name], (int, float))
                           or not math.isfinite(predicted[name]) for name in METHODS)):
                continue
            for name in METHODS:
                errors[name].append(abs(float(predicted[name]) - float(observed)))
            if abs(observed) > floor:
                signs.append(int(predicted["map"] * observed > 0))
    expected = len(records) * len(SMALL_ALPHAS)
    invalid = expected - len(errors["map"])
    mae = {name: sum(values) / len(values) if values else None
           for name, values in errors.items()}
    comparator_valid = all(mae[name] is not None and mae[name] > 0
                           for name in ("zero", "global", "six_bin"))
    accuracy = sum(signs) / len(signs) if signs else None
    gate = bool((not require_complete or (len(records) == PLANNED
                 and invalid == 0 and invalid_absolute == 0))
                and comparator_valid and mae["map"] is not None
                and len(signs) >= MIN_DETERMINATE
                and accuracy is not None and accuracy >= .75
                and all(mae["map"] <= .75 * mae[name]
                        for name in ("zero", "global", "six_bin"))
                and improved >= (MIN_ABS_IMPROVED if require_complete else 0))
    return {
        "site": site, "planned_directions": PLANNED if require_complete else len(records),
        "available_directions": len(records), "invalid_conditions": invalid,
        "invalid_absolute_anchor_directions": invalid_absolute,
        "small_alpha_planned_conditions": PLANNED * 3 if require_complete else expected,
        "small_alpha_valid_conditions": len(errors["map"]),
        "mae": mae,
        "map_to_comparator_mae_ratio": {
            name: (mae["map"] / mae[name] if mae["map"] is not None
                   and mae[name] is not None and mae[name] > 0 else None)
            for name in ("zero", "global", "six_bin")
        },
        "determinate_conditions": len(signs),
        "determinate_sign_agreement": accuracy,
        "absolute_anchor_improved_directions": improved,
        "absolute_anchor_minimum": MIN_ABS_IMPROVED,
        "site_primary_gate_passed": gate,
    }


def clean_task_stats(records: list[dict]) -> dict:
    good = sum(bool(r.get("clean", {}).get("semantic_word_plus_eos")) for r in records)
    covered = [r for r in records if r.get("clean", {}).get("prefix_class") in
               ("expected", "foil")]
    pure = sum(bool(r["clean"].get("prefix_class_complete_answer_match"))
               for r in covered)
    purity = pure / len(covered) if covered else None
    return {"planned_directions": PLANNED, "available_directions": len(records),
            "semantic_word_plus_eos": good, "prefix_classified": len(covered),
            "prefix_class_pure": pure, "prefix_purity": purity,
            "task_fidelity_gate_passed": bool(len(records) == PLANNED
                and good >= MIN_CLEAN and len(covered) >= MIN_PREFIX_COVERAGE
                and purity is not None and purity >= .75)}


def bootstrap_pairs(records: list[dict], site: str, floor: float,
                    repetitions: int = BOOTSTRAP_REPLICATES) -> dict:
    pairs = list(dict.fromkeys(r["pair_id"] for r in records))
    if len(pairs) != 12 or any(sum(r["pair_id"] == p for r in records) != 8 for p in pairs):
        raise ValueError("Expected 12 development pairs with eight directions each")
    groups = {p: [r for r in records if r["pair_id"] == p] for p in pairs}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {name: [] for name in ("map_zero_ratio", "map_global_ratio",
                                    "map_six_bin_ratio", "sign", "clean", "absolute")}
    for _ in range(repetitions):
        chosen = rng.choice(pairs, size=12, replace=True)
        sample = [r for p in chosen for r in groups[p]]
        s = primary_site_stats(sample, site, floor, require_complete=False)
        c = clean_task_stats(sample)
        ratios = s["map_to_comparator_mae_ratio"]
        for name, key in (("map_zero_ratio", "zero"),
                          ("map_global_ratio", "global"),
                          ("map_six_bin_ratio", "six_bin")):
            if ratios[key] is not None:
                values[name].append(ratios[key])
        if s["determinate_sign_agreement"] is not None:
            values["sign"].append(s["determinate_sign_agreement"])
        values["clean"].append(c["semantic_word_plus_eos"] / PLANNED)
        values["absolute"].append(s["absolute_anchor_improved_directions"] / PLANNED)
    return {name: {"n_valid_replicates": len(rows),
                   "interval_95": (np.quantile(rows, [.025, .975]).tolist()
                                   if rows else None)}
            for name, rows in values.items()}


def summarize_primary(records: list[dict], floor_by_site: dict,
                      *, bootstrap: bool = True) -> dict:
    if len(records) > PLANNED or len({r["case_id"] for r in records}) != len(records):
        raise ValueError("Development records duplicated or overfull")
    clean = clean_task_stats(records)
    sites = {site: primary_site_stats(records, site, float(floor_by_site[site]))
             for site in SITES}
    for site in SITES:
        groups = {}
        for field in ("category", "engine", "template", "pair_id"):
            groups[field] = {}
            for value in sorted({str(r[field]) for r in records}):
                subset = [r for r in records if str(r[field]) == value]
                row = primary_site_stats(subset, site, float(floor_by_site[site]),
                                         require_complete=False)
                row.pop("site_primary_gate_passed")
                groups[field][value] = row
        sites[site]["descriptive_groups"] = groups
    if bootstrap and len(records) == PLANNED:
        for site in SITES:
            sites[site]["pair_bootstrap"] = bootstrap_pairs(
                records, site, float(floor_by_site[site]))
    return {"status": "development_primary_stage_a",
            "clean_task": clean, "sites": sites,
            "development_primary_gate_passed": bool(
                clean["task_fidelity_gate_passed"]
                and all(sites[s]["site_primary_gate_passed"] for s in SITES)),
            "secondary_stage_b_required_for_positive_claim": True}
