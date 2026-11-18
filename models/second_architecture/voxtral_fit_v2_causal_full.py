"""Eight-record FIT-only Voxtral v2 causal assay; no model run by default.

Each record writes an append-only attempt log and an immutable completion
checkpoint. A resumed run replays an interrupted record in a new attempt,
leaving the partial attempt untouched. No development/test WAV is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import voxtral_fit_pilot_capability_v1 as base
import voxtral_fit_v2_causal_first_record as first


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs/voxtral_fit_v2_causal_full"
FIRST_RUN = HERE / "runs/voxtral_fit_v2_causal_first_record/20261118T091057373532Z/result.json"
FIRST_RUN_SHA = "7803ff057c309b64b603735ddf229a29ef9fc076594314c32758d1f5e2d176f4"
FIRST_RUNNER_SHA = "64d799bbb40b58b128b049877d18bde7774c2dc16717af97e692aa4531e907b5"
MAX_SECONDS = 45 * 60
CONTROL_NAMES = ("pre", "post", "padding", "random")


def validate_population() -> tuple[list[dict], dict, dict, dict]:
    """Validate all 16 selected FIT WAVs and their frozen causal row masks."""
    first.verify_frozen()
    if base.sha(Path(first.__file__)) != FIRST_RUNNER_SHA or base.sha(FIRST_RUN) != FIRST_RUN_SHA:
        raise ValueError("Audited first-record source/result changed")
    first_result = json.loads(FIRST_RUN.read_text())
    if not first_result["resource_projection"]["passes"] or first_result["progress_records"] != 108:
        raise ValueError("Audited first-record resource gate failed")
    masks = json.loads(first.MASKS.read_text())
    index = json.loads(first.INDEX.read_text())
    selection = json.loads(first.SELECTION.read_text())
    pilot = json.loads(first.PILOT.read_text())
    selected = {(c["pair_id"], c["engine"], c["phrasing_id"], c["word"]): c
                for c in selection["clips"]}
    rows = {(r["pair_id"], r["engine"], r["phrasing_id"]): r for r in index["rows"]}
    if len(masks["records"]) != 8 or len(selected) != 32:
        raise ValueError("Frozen population count changed")
    seen = set()
    for record in masks["records"]:
        key = (record["pair_id"], record["engine"], record["phrasing_id"])
        row = rows[key]
        if row["split"] != "fit_pool" or not row["map_fit"] or row["pilot_fit"] or record["phrasing_id"] != "fit_chosen":
            raise ValueError(f"Non-v2-FIT record: {key}")
        if row["words"] != record["words"] or len(record["clips"]) != 2:
            raise ValueError(f"Record words changed: {key}")
        for clip in record["clips"]:
            w = clip["word"]
            s = selected[(key[0], key[1], key[2], w)]
            if s["wav_sha256"] != clip["wav_sha256"] or s["wav_path"] != clip["wav_path"]:
                raise ValueError("Selection/mask WAV mismatch")
            wav = (ROOT / clip["wav_path"]).resolve()
            if not wav.is_relative_to((ROOT / "eval/audio_jlens/corpus_v1/rendered/clips").resolve()):
                raise ValueError("WAV outside FIT corpus")
            if base.sha(wav) != clip["wav_sha256"]:
                raise ValueError("Selected FIT WAV bytes changed")
            matches = [p for p in pilot["records"] if p["pair_id"] == key[0] and
                       p["engine"] == key[1] and p["phrasing_id"] == key[2] and p["expected_word"] == w]
            if len(matches) != 1 or not matches[0]["classification"]["exact_expected_word_and_eos"]:
                raise ValueError("V2 natural endpoint failed or missing")
            seen.add((key, w))
        for site in first.SITES:
            smap = record["site_maps"][site]
            if any(not smap["bins"][b] for b in ("pre", "post", "word_0", "word_1", "word_2", "word_3")):
                raise ValueError("Empty frozen source time bin")
            if len(smap["whole_word_rows"]) != len(smap["padding_control_rows"]):
                raise ValueError("Padding control size mismatch")
    if len(seen) != 16:
        raise ValueError("Expected 16 first-template directional cases")
    return masks["records"], rows, pilot, first_result


def prepare_inputs(processor, records: list[dict], index_rows: dict, pilot: dict) -> list[dict]:
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
            saved = next(p for p in pilot["records"] if p["pair_id"] == key[0] and
                         p["engine"] == key[1] and p["phrasing_id"] == key[2] and p["expected_word"] == w)
            if saved["prompt_input_ids"] != batches[w]["input_ids"][0].tolist():
                raise ValueError("V2 pilot prompt token IDs changed")
        canonical = {w: base.canonical_ids(row["clips"][w], w, 2) for w in record["words"]}
        for w in record["words"]:
            if processor.tokenizer.encode(w.capitalize(), add_special_tokens=False) + [2] != canonical[w]:
                raise ValueError("Canonical answer IDs changed")
        features = first.feature_conditions(batches, record)
        prepared.append({"record": record, "row": row, "batches": batches, "blank": blank,
                         "canonical": canonical, "features": features, "frontend": frontend})
    return prepared


def record_key(record: dict) -> str:
    return f"{record['pair_id']}|{record['engine']}|{record['phrasing_id']}"


def attempt_path(output: Path, ordinal: int) -> Path:
    attempts = sorted(output.glob(f"record_{ordinal:02d}_attempt_*.jsonl"))
    taken = [int(path.stem.rsplit("_", 1)[-1]) for path in attempts]
    return output / f"record_{ordinal:02d}_attempt_{max(taken, default=0)+1:02d}.jsonl"


def checkpoint_path(output: Path, ordinal: int) -> Path:
    return output / f"record_{ordinal:02d}.checkpoint.json"


def resume_budget_seconds(output: Path, completed: int) -> float:
    """Account for checkpointed work and failed attempts; fail closed after a kill."""
    checkpoint_budget = (json.loads(checkpoint_path(output, completed).read_text())["cumulative_budget_seconds"]
                         if completed else 0.0)
    referenced = {json.loads(checkpoint_path(output, i).read_text())["attempt_file"]
                  for i in range(1, completed + 1)}
    checkpoint_stamps = {json.loads(checkpoint_path(output, i).read_text()).get("invocation_stamp")
                         for i in range(1, completed + 1)}
    incomplete = {path.name for path in output.glob("record_*_attempt_*.jsonl")} - referenced
    error_paths = sorted(output.glob("invocation_*.error.json"))
    errors = [json.loads(path.read_text()) for path in error_paths]
    error_stamps = {path.name.removeprefix("invocation_").removesuffix(".error.json")
                    for path in error_paths}
    memory_stamps = {path.name.removeprefix("invocation_").removesuffix(".memory.jsonl")
                     for path in output.glob("invocation_*.memory.jsonl")}
    if not memory_stamps.issubset(checkpoint_stamps | error_stamps):
        raise ValueError("Unaccounted invocation memory log; interrupted elapsed time is unknown")
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
    if not incomplete.issubset(documented):
        raise ValueError("Interrupted attempt lacks elapsed-time evidence; restart in a fresh bounded run")
    # Both checkpoint and error artifacts record elapsed from the same run origin.
    # Their maximum is cumulative, so a failed attempt cannot receive new time.
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


def run_record(model, processor, job: dict, pilot: dict, log: Path,
               run_start: float, memory) -> dict:
    """Audited first-record operation sequence, generalized to one frozen FIT record."""
    import torch
    record, batches, blank_batches = job["record"], job["batches"], job["blank"]
    words, canonical, features = record["words"], job["canonical"], job["features"]
    source = {}
    progress = {"captures": 0, "generations": 0, "invalid": 0}

    def guard() -> None:
        if time.monotonic() - run_start >= MAX_SECONDS:
            raise TimeoutError("45-minute FIT invocation limit reached; resume from checkpoint")
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

    for w in words:
        other = next(x for x in words if x != w)
        saved = next(p for p in pilot["records"] if p["pair_id"] == record["pair_id"] and
                     p["engine"] == record["engine"] and p["phrasing_id"] == record["phrasing_id"] and
                     p["expected_word"] == w)
        baseline[(w, "clean")] = saved["generated_token_ids"]
        b = answer("blank_baseline", w, other, blank_batches[w])
        baseline[(w, "blank")] = b["generated_token_ids"]
    for name, expected, tensor in features:
        other = next(x for x in words if x != expected)
        batch = {k: v.clone() for k, v in batches[expected].items()}
        batch["input_features"] = tensor
        result = answer(f"feature:{name}", expected, other, batch)
        if name == f"native_{expected}" and result["generated_token_ids"] != baseline[(expected, "clean")]:
            raise ValueError("Native replay differs from v2 natural pilot")
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
                case_id = f"{record['pair_id']}|{record['engine']}|fit_chosen|{w}"
                for name in first.PATCHES:
                    rows = first.row_indices(smap, name)
                    donor = bstate if name == "self" else dstate
                    edited, info = first.prepare_edit(bstate, donor, rows, smap["whole_word_rows"],
                                                      name, case_id, site, arm)
                    if not info["valid"]:
                        base.append_line(log, {"type": "invalid_condition", "kind": f"patch:{arm}:{name}",
                                               "site": site, "pair_id": record["pair_id"],
                                               "engine": record["engine"], "recipient": w,
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
        raise ValueError(f"Incomplete first-record workload: {progress}")
    return progress


def summarize(records: list[dict], case_ids: list[str]) -> dict:
    """Apply frozen all-planned denominators to complete eight-record logs."""
    if len(case_ids) != 16 or len(set(case_ids)) != 16:
        raise ValueError("Expected 16 distinct FIT directional cases")
    directions = defaultdict(dict)
    for record in records:
        if record["type"] not in ("generation", "invalid_condition"):
            continue
        key = f"{record['pair_id']}|{record['engine']}|fit_chosen|{record['recipient']}"
        if key not in case_ids:
            raise ValueError("Generation outside frozen FIT directions")
        slot = (record["site"] or "none", record["kind"])
        if slot in directions[key]:
            raise ValueError(f"Duplicate generation slot: {key}/{slot}")
        directions[key][slot] = record
    if set(directions) != set(case_ids):
        raise ValueError("Missing FIT directions")
    eligible = {}
    for key in case_ids:
        d = directions[key]
        w = key.rsplit("|", 1)[-1]
        blank = d[("none", "blank_baseline")]
        eligible[key] = blank["classification"] != "recipient"
        if ("none", f"feature:native_{w}") not in d:
            raise ValueError("Missing clean native baseline")
    total_eligible = sum(eligible.values())
    results = {"planned_directions": 16, "restoration_eligible": total_eligible,
               "restoration_eligibility_gate": total_eligible >= 12, "sites": {}}
    valid_answers = {"recipient", "opposite", "other"}
    for site in first.SITES:
        by_arm = {}
        for arm in first.ARMS:
            def success(key: str, name: str) -> bool:
                label = directions[key][(site, f"patch:{arm}:{name}")]["classification"]
                return (eligible[key] and label == "recipient") if arm == "restore" else label in ("opposite", "other")
            whole_count = sum(success(k, "whole") for k in case_ids)
            quarter_counts = {f"word_{q}": sum(success(k, f"word_{q}") for k in case_ids)
                              for q in range(4)}
            control_counts = {name: sum(success(k, name) for k in case_ids) for name in CONTROL_NAMES}
            pair_voice = {}
            for pair_engine in dict.fromkeys("|".join(k.split("|")[:2]) for k in case_ids):
                subset = [k for k in case_ids if k.startswith(pair_engine + "|")]
                if len(subset) != 2:
                    raise ValueError("Pair/voice directional balance changed")
                pair_voice[pair_engine] = {"planned": 2, "whole_success": sum(success(k, "whole") for k in subset)}
            invalid_count = sum(directions[k][(site, f"patch:{arm}:{name}")]["classification"] == "invalid"
                                for k in case_ids for name in ("whole", *CONTROL_NAMES))
            score_comparisons = {}
            for control in CONTROL_NAMES:
                valid = wins = 0
                for key in case_ids:
                    d = directions[key]
                    a = d[(site, f"patch:{arm}:whole")]
                    b = d[(site, f"patch:{arm}:{control}")]
                    if a["classification"] not in valid_answers or b["classification"] not in valid_answers:
                        continue
                    ca, cb = a["recipient_minus_opposite"], b["recipient_minus_opposite"]
                    if not math.isfinite(ca) or not math.isfinite(cb):
                        continue
                    valid += 1
                    wins += ca > cb if arm == "restore" else ca < cb
                score_comparisons[control] = {"valid_of_16": valid, "strict_wins_of_16": wins,
                                              "passes": valid >= 14 and wins >= 12}
            eligible_rate = whole_count / total_eligible if total_eligible else None
            if arm == "restore":
                effect_gate = whole_count >= 12 and total_eligible >= 12 and eligible_rate >= .75
            else:
                effect_gate = whole_count >= 12
            time_gate = (invalid_count == 0 and whole_count > max(control_counts.values()) and
                         all(v["passes"] for v in score_comparisons.values()))
            by_arm[arm] = {"whole_word_success_of_16": whole_count,
                           "eligible_success_rate": eligible_rate if arm == "restore" else None,
                           "whole_word_effect_gate": effect_gate,
                           "word_quarter_success_of_16": quarter_counts,
                           "pair_voice_breakdown": pair_voice,
                           "control_success_of_16": control_counts,
                           "invalid_main_or_control_conditions": invalid_count,
                           "score_comparisons": score_comparisons,
                           "time_preference_gate": time_gate}
        results["sites"][site] = by_arm
        opposite_classes = defaultdict(int)
        for key in case_ids:
            opposite_classes[directions[key][(site, "opposite_word_control")]["classification"]] += 1
        results["sites"][site]["opposite_word_control_classes_of_16"] = dict(opposite_classes)
    return results


def finalize_output(output: Path, records: list[dict], metadata: dict,
                    case_ids: list[str]) -> dict:
    """Finalize immutable checkpoints without reloading model weights."""
    if completed_prefix(output, records, metadata) != 8:
        raise ValueError("Not all eight FIT records checkpointed")
    all_rows = []
    for i in range(1, 9):
        cp = json.loads(checkpoint_path(output, i).read_text())
        for line in (output / cp["attempt_file"]).read_text().splitlines():
            all_rows.append(json.loads(line))
    if len(all_rows) != 864:
        raise ValueError("Expected 864 planned capture/generation/invalid progress rows")
    summary = summarize(all_rows, case_ids)
    final = {"status": "eight_record_fit_complete", "meta_sha256": base.sha(output / "meta.json"),
             "checkpoint_sha256": {str(i): base.sha(checkpoint_path(output, i)) for i in range(1, 9)},
             "directions": 16, "progress_rows": len(all_rows),
             "generations": sum(r["type"] == "generation" for r in all_rows),
             "captures": sum(r["type"] == "capture" for r in all_rows),
             "invalid_conditions": sum(r["type"] == "invalid_condition" for r in all_rows),
             "summary": summary,
             "scope": "FIT-only constructed feature and hidden-state edits; no dev/test, security claim, or Stage A rescue"}
    base.exclusive_json(output / "result.json", final)
    return final


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="CPU-only frozen input check")
    mode.add_argument("--run", action="store_true", help="explicit MPS FIT-only run")
    parser.add_argument("--output-dir", type=Path, help="new run directory or existing incomplete run for resume")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    records, index_rows, pilot, first_result = validate_population()
    index = json.loads(first.INDEX.read_text())
    snapshot = base.validate_snapshot(index)
    sources = base.validate_transformers_source()
    import torch
    import transformers
    from transformers import AutoProcessor
    if torch.__version__ != "2.15.0" or transformers.__version__ != "5.20.2":
        raise ValueError("Pinned runtime versions changed")
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    prepared = prepare_inputs(processor, records, index_rows, pilot)
    case_ids = [f"{r['pair_id']}|{r['engine']}|fit_chosen|{w}" for r in records for w in r["words"]]
    metadata = {"status": "voxtral_v2_fit_only_causal_full", "script_sha256": base.sha(Path(__file__)),
                "first_runner_sha256": FIRST_RUNNER_SHA, "first_run_sha256": FIRST_RUN_SHA,
                "design_sha256": first.EXPECTED["design"], "masks_sha256": first.EXPECTED["masks"],
                "selection_sha256": first.EXPECTED["selection"], "index_sha256": first.EXPECTED["index"],
                "pilot_sha256": first.EXPECTED["pilot"], "gate_sha256": first.EXPECTED["gate"],
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
                "planned": {"records": 8, "directions": 16, "generations": 800,
                            "captures": 64, "teacher_forced_forwards": 1600},
                "memory_cap_bytes": base.MEMORY_LIMIT_BYTES, "wall_cap_seconds": MAX_SECONDS}
    if args.validate_only:
        print(json.dumps({"status": "validated_cpu_only", "metadata": metadata}, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if not first_result["resource_projection"]["passes"]:
        raise ValueError("First-record projection gate failed")
    from transformers import VoxtralForConditionalGeneration
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("MPS recommended limit unavailable")
    torch.mps.set_per_process_memory_fraction(min(1.0, base.MEMORY_LIMIT_BYTES / recommended))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or RUNS / stamp
    if output.exists():
        if not args.output_dir or not (output / "meta.json").is_file():
            raise FileExistsError("Existing output lacks explicit resume metadata")
        if json.loads((output / "meta.json").read_text()) != metadata:
            raise ValueError("Resume metadata/source hashes changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        base.exclusive_json(output / "meta.json", metadata)
    if (output / "result.json").exists():
        raise FileExistsError("Completed full FIT result exists")
    prefix = completed_prefix(output, records, metadata)
    prior_budget_seconds = resume_budget_seconds(output, prefix)
    budget_start = started - prior_budget_seconds
    if time.monotonic() - budget_start >= MAX_SECONDS:
        raise TimeoutError("Cumulative FIT wall budget exhausted before resume/finalization")
    if prefix == len(records):
        final = finalize_output(output, records, metadata, case_ids)
        print(json.dumps({"status": final["status"], "summary": final["summary"]}, indent=2))
        return
    memory_log = output / f"invocation_{stamp}.memory.jsonl"
    memory_log.touch(exist_ok=False)
    current_attempt = None
    try:
        with base.MemorySampler(torch) as memory:
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(base.SNAPSHOT), local_files_only=True, dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
            if torch.mps.driver_allocated_memory() >= base.MEMORY_LIMIT_BYTES:
                raise MemoryError("30 GiB MPS driver cap reached after load")
            for i in range(prefix, len(records)):
                if time.monotonic() - budget_start >= MAX_SECONDS:
                    raise TimeoutError("45-minute invocation limit; resume completed checkpoints")
                log = attempt_path(output, i + 1)
                log.touch(exist_ok=False)
                current_attempt = log.name
                t0 = time.monotonic()
                counts = run_record(model, processor, prepared[i], pilot, log, budget_start, memory)
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
                current_attempt = None
                base.append_line(memory_log, {"checkpoint": i + 1,
                                              "driver_max_bytes": memory.driver_max,
                                              "elapsed_seconds": time.monotonic() - started})
        final = finalize_output(output, records, metadata, case_ids)
        print(json.dumps({"status": final["status"], "summary": final["summary"]}, indent=2))
    except Exception as exc:
        base.exclusive_json(output / f"invocation_{stamp}.error.json", {
            "status": "incomplete_invocation", "error_type": type(exc).__name__,
            "error": str(exc), "completed_records": completed_prefix(output, records, metadata),
            "attempt_file": current_attempt,
            "cumulative_budget_seconds": time.monotonic() - budget_start,
            "memory_log_sha256": base.sha(memory_log)})
        raise


if __name__ == "__main__":
    main()
