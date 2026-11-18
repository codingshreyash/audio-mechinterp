"""Fit-informed Voxtral v2 capability pilot on 32 new FIT clips only.

Uses reviewed v1 processor, checkpoint, score, classifier, and MPS run logic.
--validate-only is CPU-only; --run explicitly enables model inference.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import voxtral_fit_pilot_capability_v1 as base


HERE = Path(__file__).resolve().parent
INDEX = HERE / "voxtral_corpus_v1_index.json"
SELECTION = HERE / "VOXTRAL_FIT_V2_SELECTION.json"
RUNS = HERE / "runs/voxtral_corpus_v1_fit_pilot_v2"
INDEX_SHA256 = "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971"
SELECTION_SHA256 = "f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b"
BASE_RUNNER_SHA256 = "1d4d65a528d558e5bdda3b733eaab6e7e849ab9456d6842d5ee178d3efdfb8d8"
PAIR_IDS = ("animal_08", "color_05", "nature_09", "object_05")


def select_v2(index: dict, selection: dict) -> list[tuple[dict, str, dict]]:
    if selection["source_index_sha256"] != INDEX_SHA256 or tuple(selection["selected_pair_ids"]) != PAIR_IDS:
        raise ValueError("Frozen v2 selection binding or pair IDs changed")
    if selection["planned_clips"] != 32 or selection["candidate_model_inference_run"]:
        raise ValueError("Selection is not the frozen CPU-only 32-clip v2 plan")
    if index["source_manifest_sha256"] != base.EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Source corpus manifest binding changed")
    if index["model_revision"] != base.REVISION or index["prompt"] != base.PROMPT:
        raise ValueError("Frozen model revision or prompt changed")
    by_record = {}
    for row in index["rows"]:
        key = (row["pair_id"], row["engine"], row["phrasing_id"])
        if key in by_record:
            raise ValueError("Duplicate corpus pair record")
        by_record[key] = row
    jobs = []
    seen = set()
    for item in selection["clips"]:
        key = (item["pair_id"], item["engine"], item["phrasing_id"])
        row = by_record.get(key)
        word = item["word"]
        if row is None or row["split"] != "fit_pool" or not row["map_fit"] or row["pilot_fit"]:
            raise ValueError(f"v2 selection points to old pilot or non-fit row: {key}")
        if row["pair_id"] not in PAIR_IDS or item["split"] != "fit_pool" or item["voice"] != row["voice"]:
            raise ValueError(f"v2 selection metadata changed: {key}")
        if word not in row["words"]:
            raise ValueError(f"v2 selected a word outside its pair: {key}/{word}")
        clip = row["clips"][word]
        if item["wav_path"] != clip["path"] or item["wav_sha256"] != clip["wav_sha256"]:
            raise ValueError(f"v2 selected WAV hash/path changed: {key}/{word}")
        unique = (key, word)
        if unique in seen:
            raise ValueError(f"Duplicate selected clip: {unique}")
        seen.add(unique)
        jobs.append((row, word, clip))
    if len(jobs) != 32:
        raise ValueError(f"Expected 32 v2 FIT clips, got {len(jobs)}")
    counts = Counter((row["pair_id"], row["engine"]) for row, _, _ in jobs)
    if set(counts) != {(p, e) for p in PAIR_IDS for e in ("piper", "kokoro")} or set(counts.values()) != {4}:
        raise ValueError("v2 pair/voice/template/word balance changed")
    jobs.sort(key=lambda x: (x[0]["pair_id"], x[0]["engine"], x[0]["phrasing_id"], x[1]))
    return jobs


def stage2_gate(records: list[dict], capability_gate_pass: bool) -> dict:
    if len(records) != 32:
        raise ValueError("v2 result requires exactly 32 records")
    by_group = defaultdict(list)
    for record in records:
        if record["split"] != "fit_pool" or record["pair_id"] not in PAIR_IDS:
            raise ValueError("v2 result contains nonselected or non-fit record")
        by_group[(record["pair_id"], record["engine"], record["phrasing_id"])].append(record)
    if len(by_group) != 16 or any(len(v) != 2 for v in by_group.values()):
        raise ValueError("v2 result has incomplete pair/template cells")
    selected = []
    for pair_id in PAIR_IDS:
        for engine in ("piper", "kokoro"):
            templates = sorted(k[2] for k in by_group if k[:2] == (pair_id, engine))
            if len(templates) != 2:
                raise ValueError(f"Missing two templates for {pair_id}/{engine}")
            first = templates[0]
            endpoints = sorted(by_group[(pair_id, engine, first)], key=lambda r: r["expected_word"])
            if len({r["expected_word"] for r in endpoints}) != 2:
                raise ValueError(f"Duplicate word endpoint for {pair_id}/{engine}")
            selected.extend(endpoints)
    passing = sum(bool(r["classification"]["exact_expected_word_and_eos"]) for r in selected)
    failures = [{"pair_id": r["pair_id"], "engine": r["engine"],
                 "phrasing_id": r["phrasing_id"], "word": r["expected_word"],
                 "raw_decoded": r["raw_decoded"], "generated_token_ids": r["generated_token_ids"]}
                for r in selected if not r["classification"]["exact_expected_word_and_eos"]]
    return {
        "capability_gate_pass": bool(capability_gate_pass),
        "first_template_natural_endpoints_passing": passing,
        "first_template_natural_endpoints_planned": 16,
        "stage2_all_16_exact_pass": passing == 16,
        "intervention_gate_pass": bool(capability_gate_pass and passing == 16),
        "first_template_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="CPU processor preflight; no weights")
    mode.add_argument("--run", action="store_true", help="explicitly run bounded MPS FIT pilot")
    parser.add_argument("--output-dir", type=Path, help="new output directory for --run only")
    args = parser.parse_args()
    if args.validate_only and args.output_dir:
        parser.error("--output-dir requires --run")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if base.sha(INDEX) != INDEX_SHA256 or base.sha(SELECTION) != SELECTION_SHA256:
        raise ValueError("Frozen v2 CPU index or selection SHA-256 changed")
    if base.sha(Path(base.__file__)) != BASE_RUNNER_SHA256:
        raise ValueError("Reviewed v1 runner source changed")
    source_manifest_sha256 = base.sha(base.SOURCE_MANIFEST)
    if source_manifest_sha256 != base.EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Actual corpus manifest SHA-256 changed")
    index, selection = json.loads(INDEX.read_text()), json.loads(SELECTION.read_text())
    jobs = select_v2(index, selection)
    snapshot = base.validate_snapshot(index)
    installed_source = base.validate_transformers_source()
    processor, prepared = base.preflight(index, jobs)  # Opens only the 32 selected FIT WAVs.
    import torch
    import transformers
    if torch.__version__ != "2.15.0" or transformers.__version__ != "5.20.2":
        raise ValueError("Pinned dependency versions changed")
    if args.validate_only:
        print(json.dumps({
            "status": "validated_cpu_only", "runner_sha256": base.sha(Path(__file__)),
            "selection_sha256": SELECTION_SHA256, "selected_pair_ids": list(PAIR_IDS),
            "clips": len(prepared), "pair_records": len(prepared) // 2,
            "processor_shapes": sorted({tuple(p["input_features_shape"]) for p in prepared}),
            "audio_placeholder_counts": sorted({p["audio_placeholder_count"] for p in prepared}),
            "model_weights_loaded": False,
        }, indent=2))
        return
    meta = {
        "status": "frozen_fit_informed_pilot_v2",
        "adaptation_disclosure": selection["adaptation_disclosure"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": base.sha(Path(__file__)),
        "reused_v1_runner_sha256": BASE_RUNNER_SHA256,
        "selection_sha256": SELECTION_SHA256, "index_sha256": INDEX_SHA256,
        "actual_source_manifest_sha256": source_manifest_sha256,
        "v2_draft_sha256": base.sha(HERE / "VOXTRAL_FIT_V2_DRAFT.md"),
        "snapshot": snapshot, "installed_transformers_source_sha256": installed_source,
        "requirements_freeze_sha256": base.sha(HERE / "requirements.freeze.txt"),
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "torch": torch.__version__, "transformers": transformers.__version__},
        "prompt": base.PROMPT, "pilot_pair_ids": list(PAIR_IDS),
        "generation": {"dtype": "bfloat16", "do_sample": False,
                       "max_new_tokens": base.MAX_NEW_TOKENS, "eos_token_id": 2,
                       "use_cache": True, "attn_implementation": "sdpa"},
        "secondary_score": "v1 canonical initial-capital answer+EOS; other minus expected",
        "capability_thresholds": {"overall_min": 24, "per_voice_min": 12},
        "stage2_threshold": "all 16 first-template natural endpoints exact",
        "memory_limit_bytes": base.MEMORY_LIMIT_BYTES, "planned_clips": 32,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or RUNS / stamp
    base.run(processor, prepared, meta, output_dir)
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text())
    gate = {
        "status": "complete", "runner_sha256": base.sha(Path(__file__)),
        "selection_sha256": SELECTION_SHA256, "result_sha256": base.sha(result_path),
        **stage2_gate(result["records"], result["fit_capability_gate_pass"]),
    }
    base.exclusive_json(output_dir / "v2_gate.json", gate)
    print(f"v2 gate: capability={gate['capability_gate_pass']}, stage2={gate['stage2_all_16_exact_pass']}; {output_dir / 'v2_gate.json'}")


if __name__ == "__main__":
    main()
