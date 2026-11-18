"""Prospective 48-record Voxtral development transfer; no model run by default.

Each record writes an append-only attempt log and an immutable completion
checkpoint. A resumed run replays an interrupted record in a new attempt,
leaving the partial attempt untouched. No locked-test WAV is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import voxtral_fit_pilot_capability_v1 as base
import voxtral_fit_v2_causal_first_record as first
import voxtral_dev_transfer_masks as dev_masks


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs/voxtral_dev_transfer_v1"
MASKS = dev_masks.OUTPUT
MASKS_SHA = "eeed0dc96aeda651be1034381a770fd0ed2022706d3f33c08f967391638c1181"
FIT_RUN = HERE / "runs/voxtral_fit_v2_causal_full/20261118T0928Z/result.json"
FIT_RUN_SHA = "e29a686e30505a69a3e4b26bddde606a02db753a3a5410c38432cdd14ac3baaf"
FIT_RUNNER = HERE / "voxtral_fit_v2_causal_full.py"
FIT_RUNNER_SHA = "4e0a17f80d70ca1498f2487d8cca81b20e36381bfeb35945807b2587b89373cd"
FIRST_HELPER_SHA = "64d799bbb40b58b128b049877d18bde7774c2dc16717af97e692aa4531e907b5"
PREPROCESS_HELPER = HERE / "voxtral_preprocessing_decomposition.py"
PREPROCESS_HELPER_SHA = "026445da1ac2c4c2f4b8f45ab2f8d7a1f4679924774714951b16d5cd29de0788"
MAX_SECONDS = 120 * 60
CONTROL_NAMES = ("pre", "post", "padding", "random")


def validate_population() -> tuple[list[dict], dict]:
    """Validate only indexed development inputs and prior FIT provenance."""
    dev_masks.verify_sources()
    first.verify_frozen()
    if (base.sha(MASKS) != MASKS_SHA or base.sha(FIT_RUN) != FIT_RUN_SHA or
            base.sha(FIT_RUNNER) != FIT_RUNNER_SHA or
            base.sha(Path(first.__file__)) != FIRST_HELPER_SHA or
            base.sha(PREPROCESS_HELPER) != PREPROCESS_HELPER_SHA):
        raise ValueError("Frozen development masks or behavior-critical source/result changed")
    masks = json.loads(MASKS.read_text())
    index = json.loads(dev_masks.INDEX.read_text())
    rows = {(r["pair_id"], r["engine"], r["phrasing_id"]): r for r in index["rows"]}
    expected = dev_masks.make_manifest(index)
    if masks != expected or len(masks["records"]) != 48:
        raise ValueError("Development mask artifact does not reconstruct")
    if any(rows[(r["pair_id"], r["engine"], r["phrasing_id"])]["split"] != "development"
           for r in masks["records"]):
        raise ValueError("Nondevelopment input selected")
    return masks["records"], rows


def prepare_inputs(processor, records: list[dict], index_rows: dict) -> list[dict]:
    prepared = []
    for record in records:
        key = (record["pair_id"], record["engine"], record["phrasing_id"])
        row = index_rows[key]
        batches = first.make_batches(processor, record)
        frontend = first.frontend_check(processor, batches, record)
        blank = {}
        for w in record["words"]:
            blank[w] = {k: v.clone() for k, v in batches[w].items()}
            blank[w]["input_features"] = first.blank_features(batches[w]["input_features"], record)
        canonical = {w: base.canonical_ids(row["clips"][w], w, 2) for w in record["words"]}
        for w in record["words"]:
            if processor.tokenizer.encode(w.capitalize(), add_special_tokens=False) + [2] != canonical[w]:
                raise ValueError("Canonical answer IDs changed")
        features = first.feature_conditions(batches, record)
        prepared.append({"record": record, "row": row, "batches": batches, "blank": blank,
                         "canonical": canonical, "features": features, "frontend": frontend})
    return prepared


def prepare_dev_edit(base_state, donor_state, rows: list[int], whole_rows: list[int],
                     mode: str, case_id: str, site: str, arm: str):
    """FIT edit rule with a new, development-specific random seed namespace."""
    if mode != "random":
        return first.prepare_edit(base_state, donor_state, rows, whole_rows,
                                  mode, case_id, site, arm)
    import torch
    if (base_state.shape != donor_state.shape or not rows or not whole_rows or
            base_state.dtype != torch.bfloat16 or donor_state.dtype != torch.bfloat16):
        raise ValueError("Development random patch source/mask invalid")
    b = base_state[rows].float()
    word_b = base_state[whole_rows].float()
    word_d = donor_state[whole_rows].float()
    target = float(torch.linalg.vector_norm(word_d.to(torch.bfloat16).float() - word_b))
    label = f"voxtral-dev-transfer-v1|{case_id}|{site}|{arm}|random_0"
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest(), "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(b.shape), generator=generator, dtype=torch.float32)
    noise -= noise.mean()
    norm = float(torch.linalg.vector_norm(noise))
    info = {"mode": mode, "rows": rows, "target_bf16_norm": target,
            "unscaled_edit_norm": norm, "random_seed_label": label,
            "random_seed": seed, "unscaled_random_sha256": first.tensor_sha(noise),
            "alpha": None, "valid": True}
    if target <= 0 or norm <= 0:
        info.update(valid=False, invalid_reason="zero_target_or_random_norm")
        return None, info
    info["alpha"] = target / norm
    edited = (b.cpu() + info["alpha"] * noise).to(torch.bfloat16).to(base_state.device)
    effective = float(torch.linalg.vector_norm(edited.float() - b))
    info.update(effective_bf16_norm=effective, edited_bf16_sha256=first.tensor_sha(edited))
    if abs(effective - target) > max(first.NORM_REL_TOL * target, first.NORM_ABS_TOL):
        info.update(valid=False, invalid_reason="bf16_effective_norm_out_of_tolerance")
        return None, info
    return edited, info


def record_key(record: dict) -> str:
    return f"{record['pair_id']}|{record['engine']}|{record['phrasing_id']}"


def attempt_path(output: Path, ordinal: int) -> Path:
    attempts = sorted(output.glob(f"record_{ordinal:02d}_attempt_*.jsonl"))
    taken = [int(path.stem.rsplit("_", 1)[-1]) for path in attempts]
    return output / f"record_{ordinal:02d}_attempt_{max(taken, default=0)+1:02d}.jsonl"


def checkpoint_path(output: Path, ordinal: int) -> Path:
    return output / f"record_{ordinal:02d}.checkpoint.json"


def resume_budget_seconds(output: Path, completed: int, active_stamp: str | None = None) -> float:
    """Account for checkpointed work and failed attempts; fail closed after a kill."""
    checkpoint_budget = (json.loads(checkpoint_path(output, completed).read_text())["cumulative_budget_seconds"]
                         if completed else 0.0)
    referenced = {json.loads(checkpoint_path(output, i).read_text())["attempt_file"]
                  for i in range(1, completed + 1)}
    incomplete = {path.name for path in output.glob("record_*_attempt_*.jsonl")} - referenced
    error_paths = sorted(output.glob("invocation_*.error.json"))
    errors = [json.loads(path.read_text()) for path in error_paths]
    error_stamps = {path.name.removeprefix("invocation_").removesuffix(".error.json")
                    for path in error_paths}
    complete_stamps = {p.name.removeprefix("invocation_").removesuffix(".complete.json")
                       for p in output.glob("invocation_*.complete.json")}
    start_stamps = {p.name.removeprefix("invocation_").removesuffix(".start.json")
                    for p in output.glob("invocation_*.start.json")}
    if active_stamp is not None and active_stamp not in start_stamps:
        raise ValueError("Active invocation has no start receipt")
    if not (start_stamps - {active_stamp}).issubset(error_stamps | complete_stamps):
        raise ValueError("Unreceipted prior invocation; elapsed time unknown")
    if (active_stamp in error_stamps or active_stamp in complete_stamps or
            not error_stamps.issubset(start_stamps) or not complete_stamps.issubset(start_stamps)):
        raise ValueError("Terminal receipt lacks invocation start")
    memory_stamps = {p.name.removeprefix("invocation_").removesuffix(".memory.jsonl")
                     for p in output.glob("invocation_*.memory.jsonl")}
    if not memory_stamps.issubset(start_stamps):
        raise ValueError("Memory log lacks invocation start")
    debts = [checkpoint_budget]
    documented = set()
    previous_error_debt = 0.0
    for error in errors:
        debt = error.get("cumulative_budget_seconds")
        if (not isinstance(debt, (int, float)) or not math.isfinite(debt)
                or debt < previous_error_debt):
            raise ValueError("Failed invocation has no valid elapsed budget record")
        debts.append(float(debt))
        previous_error_debt = float(debt)
        if error.get("attempt_file"):
            documented.add(error["attempt_file"])
    for path in output.glob("invocation_*.complete.json"):
        receipt = json.loads(path.read_text())
        debt = receipt.get("cumulative_budget_seconds")
        if not isinstance(debt, (int, float)) or not math.isfinite(debt):
            raise ValueError("Completion receipt lacks elapsed budget")
        debts.append(float(debt))
    if not incomplete.issubset(documented):
        raise ValueError("Interrupted attempt lacks elapsed-time evidence; restart in a fresh bounded run")
    # Every prior start has an error or completion receipt. A hard kill fails closed.
    return max(debts)


def completed_prefix(output: Path, records: list[dict], meta: dict) -> int:
    complete = 0
    previous_budget = 0.0
    for ordinal, record in enumerate(records, 1):
        path = checkpoint_path(output, ordinal)
        if not path.exists():
            if any(checkpoint_path(output, j).exists() for j in range(ordinal + 1, len(records) + 1)):
                raise ValueError("Noncontiguous completed checkpoints")
            break
        item = json.loads(path.read_text())
        log = output / item["attempt_file"]
        if (item["ordinal"] != ordinal or item["record_key"] != record_key(record)
                or item["run_meta_sha256"] != base.sha(output / "meta.json")
                or not isinstance(item.get("invocation_stamp"), str)
                or not log.is_file() or base.sha(log) != item["attempt_sha256"]
                or item["progress_records"] != 108 or item["capture_records"] != 8
                or item["generation_records"] + item["invalid_conditions"] != 100
                or item["teacher_forced_forwards"] != 2 * item["generation_records"]
                or not previous_budget < item["cumulative_budget_seconds"] <= MAX_SECONDS):
            raise ValueError(f"Completed checkpoint invalid: {path}")
        lines = [json.loads(line) for line in log.read_text().splitlines()]
        if (len(lines) != 108 or sum(r["type"] == "capture" for r in lines) != 8
                or sum(r["type"] == "generation" for r in lines) != item["generation_records"]
                or sum(r["type"] == "invalid_condition" for r in lines) != item["invalid_conditions"]):
            raise ValueError(f"Checkpoint attempt content/count invalid: {log}")
        previous_budget = item["cumulative_budget_seconds"]
        complete += 1
    return complete


def run_record(model, processor, job: dict, log: Path,
               run_start: float, memory) -> dict:
    """FIT operation grammar on one frozen development record."""
    import torch
    record, batches, blank_batches = job["record"], job["batches"], job["blank"]
    words, canonical, features = record["words"], job["canonical"], job["features"]
    source = {}
    progress = {"captures": 0, "generations": 0, "invalid": 0}

    def guard() -> None:
        if time.monotonic() - run_start >= MAX_SECONDS:
            raise TimeoutError("Two-hour development wall limit reached")
        if max(memory.driver_max, int(torch.mps.driver_allocated_memory())) >= base.MEMORY_LIMIT_BYTES:
            raise MemoryError("30 GiB MPS driver cap reached")

    for site in first.SITES:
        for w in words:
            for kind, batch in (("clean", batches[w]), ("blank", blank_batches[w])):
                guard()
                state, capture_meta = first.capture_state(model, batch, site)
                source[(site, w, kind)] = state
                base.append_line(log, {"type": "capture", "site": site, "word": w,
                                       "kind": kind, "sha256": first.tensor_sha(state), **capture_meta})
                progress["captures"] += 1
                guard()
    baseline = {}

    def answer(kind: str, recipient: str, opposite: str, batch, site=None,
               rows=None, edited=None, info=None):
        guard()
        started = time.monotonic()
        result = first.answer_and_scores(model, processor, batch, recipient, opposite,
                                         canonical, site, rows, edited)
        suffix_ids = result["generated_token_ids"]
        nonterminal_ids = suffix_ids[:-1] if result["terminal_eos"] else suffix_ids
        special_ids = set(processor.tokenizer.all_special_ids)
        result["nonterminal_special_token_ids"] = [token for token in nonterminal_ids
                                                   if token in special_ids]
        result.update({"type": "generation", "kind": kind, "site": site,
                       "pair_id": record["pair_id"], "engine": record["engine"],
                       "phrasing_id": record["phrasing_id"],
                       "recipient": recipient, "opposite": opposite,
                       "edit": info, "seconds_including_scores": time.monotonic()-started,
                       "mps_driver_bytes_after": int(torch.mps.driver_allocated_memory())})
        base.append_line(log, result)
        progress["generations"] += 1
        guard()
        return result

    for name, expected, tensor in features:
        other = next(x for x in words if x != expected)
        batch = {k: v.clone() for k, v in batches[expected].items()}
        batch["input_features"] = tensor
        result = answer(f"feature:{name}", expected, other, batch)
        if name == f"native_{expected}":
            baseline[(expected, "clean")] = result["generated_token_ids"]
    if len(baseline) != 2:
        raise ValueError("Both clean native development baselines missing")
    for w in words:
        other = next(x for x in words if x != w)
        b = answer("blank_baseline", w, other, blank_batches[w])
        baseline[(w, "blank")] = b["generated_token_ids"]
    for site in first.SITES:
        smap = record["site_maps"][site]
        for w in words:
            other = next(x for x in words if x != w)
            for kind, batch in (("clean", batches[w]), ("blank", blank_batches[w])):
                result = answer(f"passive:{kind}", w, other, batch, site)
                if result["generated_token_ids"] != baseline[(w, kind)]:
                    raise ValueError(f"Passive {site}/{w}/{kind} parity failed")
            for arm in first.ARMS:
                source_kind = "blank" if arm == "restore" else "clean"
                donor_kind = "clean" if arm == "restore" else "blank"
                bstate = source[(site, w, source_kind)]
                dstate = source[(site, w, donor_kind)]
                batch = blank_batches[w] if arm == "restore" else batches[w]
                case_id = f"{record['pair_id']}|{record['engine']}|{record['phrasing_id']}|{w}"
                for name in first.PATCHES:
                    rows = first.row_indices(smap, name)
                    donor = bstate if name == "self" else dstate
                    edited, info = prepare_dev_edit(bstate, donor, rows, smap["whole_word_rows"],
                                                    name, case_id, site, arm)
                    if not info["valid"]:
                        if name == "self":
                            raise ValueError(f"Required self-parity patch invalid: {case_id}/{site}/{arm}")
                        base.append_line(log, {"type": "invalid_condition", "kind": f"patch:{arm}:{name}",
                                               "site": site, "pair_id": record["pair_id"],
                                               "engine": record["engine"],
                                               "phrasing_id": record["phrasing_id"], "recipient": w,
                                               "opposite": other, "classification": "invalid",
                                               "recipient_minus_opposite": None, "edit": info})
                        progress["invalid"] += 1
                        continue
                    result = answer(f"patch:{arm}:{name}", w, other, batch, site, rows, edited, info)
                    if name == "self" and result["generated_token_ids"] != baseline[(w, source_kind)]:
                        raise ValueError(f"Self {site}/{w}/{arm} parity failed")
                if arm == "restore":
                    rows = smap["whole_word_rows"]
                    donor = source[(site, other, "clean")]
                    answer("opposite_word_control", w, other, batch, site, rows, donor[rows],
                           {"mode": "opposite_word", "rows": rows, "donor_sha256": first.tensor_sha(donor)})
    if progress["captures"] != 8 or progress["generations"] + progress["invalid"] != 100:
        raise ValueError(f"Incomplete development record workload: {progress}")
    return progress


def summarize(records: list[dict], case_ids: list[str]) -> dict:
    """Apply all 96 development directions, including invalid/malformed cases."""
    if len(case_ids) != 96 or len(set(case_ids)) != 96:
        raise ValueError("Expected 96 distinct development directions")
    directions = defaultdict(dict)
    for record in records:
        if record["type"] not in ("generation", "invalid_condition"):
            continue
        if record["type"] == "generation":
            scores = record.get("canonical_scores", {})
            a, b = record["recipient"], record["opposite"]
            ca = scores.get(a)
            cb = scores.get(b)
            contrast = record.get("recipient_minus_opposite")
            if (not all(isinstance(x, (int, float)) and math.isfinite(x)
                        for x in (ca, cb, contrast)) or
                    not math.isclose(contrast, ca - cb, rel_tol=1e-8, abs_tol=1e-8)):
                raise ValueError("Generated condition missing finite candidate score or contrast")
        key = (f"{record['pair_id']}|{record['engine']}|"
               f"{record['phrasing_id']}|{record['recipient']}")
        if key not in case_ids:
            raise ValueError("Generation outside frozen development directions")
        slot = (record["site"] or "none", record["kind"])
        if slot in directions[key]:
            raise ValueError(f"Duplicate generation slot: {key}/{slot}")
        directions[key][slot] = record
    if set(directions) != set(case_ids):
        raise ValueError("Missing development directions")
    clean_correct = {}
    restore_eligible = {}
    blank_classes = defaultdict(int)
    for key in case_ids:
        d = directions[key]
        if len(d) != 50:
            raise ValueError(f"Missing or extra planned development condition: {key}/{len(d)}")
        w = key.rsplit("|", 1)[-1]
        blank = d[("none", "blank_baseline")]
        clean = d[("none", f"feature:native_{w}")]
        clean_correct[key] = clean["classification"] == "recipient"
        restore_eligible[key] = clean_correct[key] and blank["classification"] != "recipient"
        blank_classes[blank["classification"]] += 1
    n_clean = sum(clean_correct.values())
    n_eligible = sum(restore_eligible.values())
    for key in case_ids:
        d = directions[key]
        w = key.rsplit("|", 1)[-1]
        clean_ids = d[("none", f"feature:native_{w}")].get("generated_token_ids")
        blank_ids = d[("none", "blank_baseline")].get("generated_token_ids")
        if not isinstance(clean_ids, list) or not isinstance(blank_ids, list):
            raise ValueError(f"Natural generation suffix IDs missing: {key}")
        for site in first.SITES:
            for kind, expected in (("clean", clean_ids), ("blank", blank_ids)):
                row = d[(site, f"passive:{kind}")]
                if row["type"] != "generation" or row.get("generated_token_ids") != expected:
                    raise ValueError(f"Passive exact-suffix parity failed: {key}/{site}/{kind}")
            for arm, expected in (("restore", blank_ids), ("loss", clean_ids)):
                row = d[(site, f"patch:{arm}:self")]
                if row["type"] != "generation" or row.get("generated_token_ids") != expected:
                    raise ValueError(f"Self exact-suffix parity failed: {key}/{site}/{arm}")
    results = {"planned_directions": 96, "clean_correct": n_clean,
               "restoration_eligible": n_eligible, "blank_classes": dict(blank_classes),
               "task_validity_gate": n_clean >= 80 and n_eligible >= 72, "sites": {}}
    valid_answers = {"recipient", "opposite", "other"}
    for site in first.SITES:
        by_arm = {}
        for arm in first.ARMS:
            def success(key: str, name: str) -> bool:
                label = directions[key][(site, f"patch:{arm}:{name}")]["classification"]
                return ((restore_eligible[key] and label == "recipient") if arm == "restore"
                        else (clean_correct[key] and label in ("opposite", "other")))
            whole_count = sum(success(k, "whole") for k in case_ids)
            quarter_counts = {f"word_{q}": sum(success(k, f"word_{q}") for k in case_ids)
                              for q in range(4)}
            control_counts = {name: sum(success(k, name) for k in case_ids) for name in CONTROL_NAMES}
            subgroup = {}
            for label, position, expected in (("category", 0, 24), ("voice", 1, 48),
                                               ("template", 2, 48)):
                buckets = defaultdict(list)
                for key in case_ids:
                    fields = key.split("|")
                    group = fields[0].split("_", 1)[0] if position == 0 else fields[position]
                    buckets[group].append(key)
                subgroup[label] = {g: {"planned": expected,
                                       "whole_success": sum(success(k, "whole") for k in subset)}
                                   for g, subset in sorted(buckets.items())}
                if any(len(subset) != expected for subset in buckets.values()):
                    raise ValueError(f"Development {label} balance changed")
            invalid_count = sum(directions[k][(site, f"patch:{arm}:{name}")]["classification"] == "invalid"
                                for k in case_ids for name in ("whole", *CONTROL_NAMES))
            score_comparisons = {}
            for control in CONTROL_NAMES:
                valid = wins = 0
                finite_only = finite_only_wins = 0
                for key in case_ids:
                    d = directions[key]
                    a = d[(site, f"patch:{arm}:whole")]
                    b = d[(site, f"patch:{arm}:{control}")]
                    ca, cb = a["recipient_minus_opposite"], b["recipient_minus_opposite"]
                    if (ca is None or cb is None or not isinstance(ca, (int, float)) or
                            not isinstance(cb, (int, float)) or
                            not math.isfinite(ca) or not math.isfinite(cb)):
                        continue
                    finite_only += 1
                    finite_only_wins += ca > cb if arm == "restore" else ca < cb
                    if a["classification"] not in valid_answers or b["classification"] not in valid_answers:
                        continue
                    valid += 1
                    wins += ca > cb if arm == "restore" else ca < cb
                score_comparisons[control] = {"valid_of_96": valid, "strict_wins_of_96": wins,
                                              "finite_score_only_pairs_of_96": finite_only,
                                              "finite_score_only_wins_of_96": finite_only_wins,
                                              "passes": valid >= 84 and wins >= 72}
            denominator = n_eligible if arm == "restore" else n_clean
            eligible_rate = whole_count / denominator if denominator else None
            subgroup_gate = (
                all(v["whole_success"] >= 12 for v in subgroup["category"].values()) and
                all(v["whole_success"] >= 30 for v in subgroup["voice"].values()) and
                all(v["whole_success"] >= 30 for v in subgroup["template"].values()))
            effect_gate = (results["task_validity_gate"] and invalid_count == 0 and
                           whole_count >= 72 and eligible_rate is not None and
                           eligible_rate >= .75 and subgroup_gate and
                           all(whole_count > v and v <= 18 for v in control_counts.values()))
            time_gate = (invalid_count == 0 and whole_count > max(control_counts.values()) and
                         all(v["passes"] for v in score_comparisons.values()))
            by_arm[arm] = {"whole_word_success_of_96": whole_count,
                           "eligible_success_rate": eligible_rate,
                           "whole_word_effect_gate": effect_gate,
                           "word_quarter_success_of_96": quarter_counts,
                           "subgroup_breakdown": subgroup,
                           "control_success_of_96": control_counts,
                           "invalid_main_or_control_conditions": invalid_count,
                           "score_comparisons": score_comparisons,
                           "time_preference_gate": time_gate}
        results["sites"][site] = by_arm
        opposite_classes = defaultdict(int)
        for key in case_ids:
            opposite_classes[directions[key][(site, "opposite_word_control")]["classification"]] += 1
        results["sites"][site]["opposite_word_control_classes_of_96"] = dict(opposite_classes)
    results["development_advance_gate"] = (
        results["task_validity_gate"] and
        all(results["sites"][site][arm]["whole_word_effect_gate"]
            for site in first.SITES for arm in first.ARMS) and
        all(results["sites"][site]["loss"]["time_preference_gate"]
            for site in first.SITES))
    return results


def finalize_output(output: Path, records: list[dict], metadata: dict,
                    case_ids: list[str]) -> dict:
    """Finalize immutable checkpoints without reloading model weights."""
    if completed_prefix(output, records, metadata) != 48:
        raise ValueError("Not all 48 development records checkpointed")
    all_rows = []
    for i in range(1, 49):
        cp = json.loads(checkpoint_path(output, i).read_text())
        for line in (output / cp["attempt_file"]).read_text().splitlines():
            all_rows.append(json.loads(line))
    if len(all_rows) != 5184:
        raise ValueError("Expected 5184 development capture/generation/invalid rows")
    summary = summarize(all_rows, case_ids)
    final = {"status": "development_transfer_complete", "meta_sha256": base.sha(output / "meta.json"),
             "checkpoint_sha256": {str(i): base.sha(checkpoint_path(output, i)) for i in range(1, 49)},
             "directions": 96, "progress_rows": len(all_rows),
             "generations": sum(r["type"] == "generation" for r in all_rows),
             "captures": sum(r["type"] == "capture" for r in all_rows),
             "invalid_conditions": sum(r["type"] == "invalid_condition" for r in all_rows),
             "summary": summary,
             "scope": "One-shot development transfer with synthetic speech and constructed hidden-state edits; no locked test or security claim"}
    base.exclusive_json(output / "result.json", final)
    return final


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="CPU-only frozen input check")
    mode.add_argument("--run", action="store_true", help="explicit MPS development run")
    parser.add_argument("--output-dir", type=Path, help="new run directory or existing incomplete run for resume")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Start receipt precedes WAV decoding, processor load, and feature preparation.
    # A hard kill leaves an unreceipted start and makes this directory unresumable.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or RUNS / stamp
    prior_budget_seconds = 0.0
    if args.run:
        if output.exists():
            if not args.output_dir or not output.is_dir() or (output / "result.json").exists():
                raise FileExistsError("Existing or completed output is not an incomplete explicit resume")
            # Count completed attempt logs as referenced before preprocessing.
            # Full ordered checkpoint/hash verification follows preparation.
            prior_budget_seconds = resume_budget_seconds(
                output, len(list(output.glob("record_*.checkpoint.json"))))
        else:
            output.mkdir(parents=True, exist_ok=False)
        base.exclusive_json(output / f"invocation_{stamp}.start.json", {
            "status": "started", "stamp": stamp, "prior_budget_seconds": prior_budget_seconds,
            "wall_start_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": base.sha(Path(__file__)), "protocol_sha256": dev_masks.PIN["protocol"],
            "masks_sha256": MASKS_SHA, "wall_cap_seconds": MAX_SECONDS})
    budget_start = started - prior_budget_seconds
    try:
        _execute(args, output, stamp, started, budget_start)
    except Exception as exc:
        if args.run:
            meta_path = output / "meta.json"
            receipts = {
                "status": "incomplete_invocation", "stamp": stamp,
                "error_type": type(exc).__name__, "error": str(exc),
                "attempt_file": _CURRENT_ATTEMPT.get(stamp),
                "cumulative_budget_seconds": time.monotonic() - budget_start,
                "meta_sha256": base.sha(meta_path) if meta_path.exists() else None,
            }
            memory_path = output / f"invocation_{stamp}.memory.jsonl"
            receipts["memory_log_sha256"] = base.sha(memory_path) if memory_path.exists() else None
            base.exclusive_json(output / f"invocation_{stamp}.error.json", receipts)
        raise


_CURRENT_ATTEMPT: dict[str, str | None] = {}


def _execute(args, output: Path, stamp: str, started: float, budget_start: float) -> None:
    records, index_rows = validate_population()
    index = json.loads(dev_masks.INDEX.read_text())
    snapshot = base.validate_snapshot(index)
    sources = base.validate_transformers_source()
    import torch
    import transformers
    from transformers import AutoProcessor
    if torch.__version__ != "2.15.0" or transformers.__version__ != "5.20.2":
        raise ValueError("Pinned runtime versions changed")
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    prepared = prepare_inputs(processor, records, index_rows)
    case_ids = [f"{r['pair_id']}|{r['engine']}|{r['phrasing_id']}|{w}"
                for r in records for w in r["words"]]
    fit_checkpoints = [json.loads(p.read_text()) for p in sorted(FIT_RUN.parent.glob("record_*.checkpoint.json"))]
    if len(fit_checkpoints) != 8:
        raise ValueError("Audited FIT checkpoint count changed")
    fit_result = json.loads(FIT_RUN.read_text())
    for i in range(1, 9):
        path = FIT_RUN.parent / f"record_{i:02d}.checkpoint.json"
        if base.sha(path) != fit_result["checkpoint_sha256"][str(i)]:
            raise ValueError("Audited FIT checkpoint hash changed")
    fit_max_record_seconds = max(cp["seconds"] for cp in fit_checkpoints)
    fit_peak_bytes = max(cp["mps_driver_observed_max_bytes"] for cp in fit_checkpoints)
    projection = {"fit_max_record_seconds": fit_max_record_seconds,
                  "fit_peak_mps_driver_bytes": fit_peak_bytes,
                  "observed_cpu_preparation_seconds": time.monotonic() - started,
                  "projected_total_seconds": (time.monotonic() - started) +
                  1.25 * fit_max_record_seconds * len(records) + 30,
                  "factor_on_fit_max_record_seconds": 1.25,
                  "fixed_load_allowance_seconds": 30}
    projection["passes"] = (projection["projected_total_seconds"] < MAX_SECONDS and
                            fit_peak_bytes < base.MEMORY_LIMIT_BYTES)
    metadata = {"status": "voxtral_dev_transfer_v1", "script_sha256": base.sha(Path(__file__)),
                "protocol_sha256": dev_masks.PIN["protocol"], "mask_builder_sha256": base.sha(Path(dev_masks.__file__)),
                "masks_sha256": MASKS_SHA, "index_sha256": dev_masks.PIN["index"],
                "fit_runner_sha256": FIT_RUNNER_SHA, "fit_run_sha256": FIT_RUN_SHA,
                "first_record_helper_sha256": FIRST_HELPER_SHA,
                "preprocessing_helper_sha256": PREPROCESS_HELPER_SHA,
                "fit_design_sha256": first.EXPECTED["design"],
                "fit_masks_sha256": first.EXPECTED["masks"],
                "fit_selection_sha256": first.EXPECTED["selection"],
                "snapshot": snapshot, "installed_sources": sources,
                "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                             "torch": torch.__version__, "transformers": transformers.__version__},
                "prompt": base.PROMPT, "case_ids": case_ids,
                "record_keys": [record_key(r) for r in records],
                "frontend_checks": {record_key(j["record"]): j["frontend"] for j in prepared},
                "feature_hashes": {record_key(j["record"]): {
                    "native": {w: first.tensor_sha(j["batches"][w]["input_features"])
                               for w in j["record"]["words"]},
                    "blank": {w: first.tensor_sha(j["blank"][w]["input_features"])
                              for w in j["record"]["words"]},
                    "six_conditions": {name: first.tensor_sha(t) for name, _, t in j["features"]},
                } for j in prepared},
                "planned": {"records": 48, "directions": 96,
                            "generations_or_invalid": 4800, "captures": 384,
                            "teacher_forced_forwards_max": 9600,
                            "progress_rows": 5184},
                "resource_projection": projection,
                "memory_cap_bytes": base.MEMORY_LIMIT_BYTES, "wall_cap_seconds": MAX_SECONDS}
    if args.validate_only:
        print(json.dumps({"status": "validated_cpu_only", "metadata": metadata}, indent=2))
        return
    if not projection["passes"]:
        raise ValueError("Conservative development resource projection failed")
    if time.monotonic() - budget_start >= MAX_SECONDS:
        raise TimeoutError("Cumulative development wall budget exhausted before model load")
    if (output / "meta.json").exists():
        prior = json.loads((output / "meta.json").read_text())
        # Projection is invocation specific; all frozen source and feature facts remain invariant.
        prior_projection = prior.pop("resource_projection", None)
        current_projection = metadata.pop("resource_projection")
        if prior != metadata or not prior_projection or not prior_projection.get("passes"):
            raise ValueError("Resume metadata/source/feature hashes changed")
        metadata["resource_projection"] = current_projection
        prior["resource_projection"] = prior_projection
    else:
        if list(output.glob("record_*.checkpoint.json")) or list(output.glob("record_*_attempt_*.jsonl")):
            raise ValueError("Resume output lacks frozen metadata")
        base.exclusive_json(output / "meta.json", metadata)
    prefix = completed_prefix(output, records, metadata)
    # Revalidate cumulative debt now that the checkpoint prefix is known.
    prior_debt = resume_budget_seconds(output, prefix, active_stamp=stamp)
    if abs(prior_debt - (started - budget_start)) > 1e-6:
        raise ValueError("Prior elapsed debt changed during preparation")
    forecast_remaining = (time.monotonic() - budget_start +
                          1.25 * fit_max_record_seconds * (len(records) - prefix) + 30)
    if forecast_remaining >= MAX_SECONDS:
        raise TimeoutError("Projected remaining development work exceeds cumulative wall cap")
    if time.monotonic() - budget_start >= MAX_SECONDS:
        raise TimeoutError("Cumulative development wall budget exhausted before model load")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    from transformers import VoxtralForConditionalGeneration
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("MPS recommended limit unavailable")
    torch.mps.set_per_process_memory_fraction(min(1.0, base.MEMORY_LIMIT_BYTES / recommended))
    if prefix == len(records):
        final = finalize_output(output, records, metadata, case_ids)
        base.exclusive_json(output / f"invocation_{stamp}.complete.json", {
            "status": "complete", "stamp": stamp,
            "cumulative_budget_seconds": time.monotonic() - budget_start,
            "result_sha256": base.sha(output / "result.json")})
        print(json.dumps({"status": final["status"], "summary": final["summary"]}, indent=2))
        return
    memory_log = output / f"invocation_{stamp}.memory.jsonl"
    memory_log.touch(exist_ok=False)
    with base.MemorySampler(torch) as memory:
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(base.SNAPSHOT), local_files_only=True, dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
            if torch.mps.driver_allocated_memory() >= base.MEMORY_LIMIT_BYTES:
                raise MemoryError("30 GiB MPS driver cap reached after load")
            for i in range(prefix, len(records)):
                if time.monotonic() - budget_start >= MAX_SECONDS:
                    raise TimeoutError("Cumulative two-hour development limit reached")
                log = attempt_path(output, i + 1)
                log.touch(exist_ok=False)
                _CURRENT_ATTEMPT[stamp] = log.name
                t0 = time.monotonic()
                counts = run_record(model, processor, prepared[i], log, budget_start, memory)
                checkpoint = {"status": "complete", "ordinal": i + 1,
                              "invocation_stamp": stamp,
                              "record_key": record_key(records[i]), "attempt_file": log.name,
                              "attempt_sha256": base.sha(log), "run_meta_sha256": base.sha(output / "meta.json"),
                              "progress_records": 108, "capture_records": counts["captures"],
                              "generation_records": counts["generations"],
                              "teacher_forced_forwards": 2 * counts["generations"],
                              "invalid_conditions": counts["invalid"],
                              "seconds": time.monotonic() - t0,
                              "cumulative_budget_seconds": time.monotonic() - budget_start,
                              "mps_driver_observed_max_bytes": memory.driver_max}
                base.exclusive_json(checkpoint_path(output, i + 1), checkpoint)
                _CURRENT_ATTEMPT[stamp] = None
                base.append_line(memory_log, {"checkpoint": i + 1,
                                              "driver_max_bytes": memory.driver_max,
                                              "elapsed_seconds": time.monotonic() - started})
    final = finalize_output(output, records, metadata, case_ids)
    base.exclusive_json(output / f"invocation_{stamp}.complete.json", {
            "status": "complete", "stamp": stamp,
            "cumulative_budget_seconds": time.monotonic() - budget_start,
            "result_sha256": base.sha(output / "result.json"),
            "memory_log_sha256": base.sha(memory_log)})
    print(json.dumps({"status": final["status"], "summary": final["summary"]}, indent=2))


if __name__ == "__main__":
    main()
