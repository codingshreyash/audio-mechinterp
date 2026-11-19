"""No-weight evidence contract for a future Qwen action-method runner.

This module does not import torch, Transformers, or model weights. It validates
saved candidate records supplied by a future separately reviewed backend.
Candidate execution is deliberately unavailable in this source-only version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import audit_sources_v2, endpoint_v2, gate_v3, listening_bundle_v2
from . import readout_v2, render_sources_v2
from .endpoint import LABELS, SPEC_PATH, load_spec, validate_local_source

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "READOUT_FREEZE_V2.json"
SOURCE = render_sources_v2.ATTEMPT
BUNDLE = listening_bundle_v2.BUNDLE
PILOT_MPS_LIMIT = 24 * 1024**3
PILOT_RSS_LIMIT = 30 * 1024**3
PILOT_SECONDS = 300
FIT_SECONDS = 3600
SITES = ("text.residual.17", "text.residual.34")
WIDTH = 2048


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_preflight() -> dict[str, Any]:
    """Re-audit source bytes and deliberately return no inference clearance."""
    spec = load_spec()
    validate_local_source(spec)
    _require(digest(SPEC_PATH) == gate_v3.V1_SPEC_SHA256
             and digest(FREEZE) == gate_v3.V2_FREEZE_SHA256,
             "action method freeze changed")
    source_audit = audit_sources_v2.audit(SOURCE)
    bundle_audit = listening_bundle_v2.audit_bundle(SOURCE, BUNDLE)
    saved = json.loads((SOURCE.parent / "attempt_001_audit.json").read_text())
    _require(saved == source_audit
             and source_audit["cases"] == 18
             and source_audit["raw_and_converted_and_final_files"] == 54
             and bundle_audit["copied_wavs"] == 36
             and bundle_audit["human_reviews"] == 0,
             "speech source or blinded bundle evidence changed")
    return {"schema": "qwen_action_runner_source_preflight_v1",
            "status": "source_bytes_pass_candidate_inference_hold",
            "model_revision": spec["model_revision"],
            "spec_sha256": digest(SPEC_PATH),
            "readout_freeze_sha256": digest(FREEZE),
            "source_manifest_sha256": source_audit["source_manifest_sha256"],
            "bundle_mapping_sha256": bundle_audit["bundle_mapping_sha256"],
            "human_reviews": 0,
            "candidate_inference_authorized": False}


def planned_case_ids() -> dict[str, list[str]]:
    """Explicit stage order; locked WAVs never appear in development work."""
    spec = load_spec()
    fit = [f"{r['id']}__typed" for r in spec["scripts"]
           if r["split"] == "fit_text"]
    dev = [r for r in spec["scripts"] if r["split"] == "development"]
    locked = [r for r in spec["scripts"] if r["split"] == "locked"]
    return {"typed_pilot": [fit[0]], "fit_text": fit,
            "development_typed": [f"{r['id']}__typed" for r in dev],
            "development_audio": [f"{r['id']}__piper_en_US_amy_medium"
                                  for r in dev],
            "locked_typed": [f"{r['id']}__typed" for r in locked],
            "locked_audio": [f"{r['id']}__{voice}" for voice in
                             spec["speech_voices"]["locked"] for r in locked]}


def validate_teacher_forced_triplet(rows: list[dict[str, Any]],
                                    *, width: int = WIDTH) -> dict[str, Any]:
    """Recompute full label+EOS scores from two full-vocabulary logit vectors.

    The future runner must save the exact logits, three identical prompt states,
    VJPs, prompt-token hash, and hook counts. This validator cannot establish
    model provenance without a separately reviewed executable and raw trace.
    """
    _require(len(rows) == 3 and [r.get("label") for r in rows] == list(LABELS),
             "teacher-forced target order changed")
    scores: list[float] = []
    states: dict[str, list[np.ndarray]] = {site: [] for site in SITES}
    hashes: list[dict[str, str]] = []
    prompt_ids = rows[0].get("prompt_ids")
    _require(isinstance(prompt_ids, list) and prompt_ids
             and all(type(x) is int and x >= 0 for x in prompt_ids),
             "prompt token IDs missing or invalid")
    prompt_hash = hashlib.sha256(json.dumps(prompt_ids,
                              separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        label = row["label"]
        _require(row.get("target_ids") == [readout_v2.LABEL_TOKEN_IDS[
            LABELS.index(label)], readout_v2.EOS_TOKEN_ID]
            and row.get("prompt_ids") == prompt_ids
            and row.get("prompt_ids_sha256") == prompt_hash
            and row.get("teacher_forced_input_ids") == prompt_ids + [
                row["target_ids"][0]]
            and row.get("logit_positions") == [len(prompt_ids) - 1,
                                                len(prompt_ids)]
            and row.get("hook_calls") == {site: 1 for site in SITES},
            "teacher-forced target, prompt, logits alignment or hook identity changed")
        first = np.asarray(row.get("first_logits"))
        second = np.asarray(row.get("after_label_logits"))
        _require(first.dtype == second.dtype == np.float32
                 and first.ndim == second.ndim == 1
                 and first.shape == second.shape
                 and first.size > readout_v2.EOS_TOKEN_ID
                 and np.isfinite(first).all() and np.isfinite(second).all(),
                 "full-vocabulary logits missing or invalid")
        score = readout_v2.complete_label_log_score(
            first, second, row["target_ids"][0])
        _require(abs(score - float(row.get("claimed_score", float("nan")))) <= 1e-5,
                 "teacher-forced score differs from saved full logits")
        scores.append(score)
        hashes.append({"first_logits_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                       "after_label_logits_sha256": hashlib.sha256(second.tobytes()).hexdigest()})
        for site in SITES:
            state = np.asarray(row.get("states", {}).get(site))
            gradient = np.asarray(row.get("gradients", {}).get(site))
            _require(state.dtype == gradient.dtype == np.float32
                     and state.shape == gradient.shape == (width,)
                     and np.isfinite(state).all() and np.isfinite(gradient).all(),
                     "prompt-last state/VJP invalid")
            states[site].append(state)
    maxabs = {site: max(float(np.max(np.abs(value - states[site][0])))
                        for value in states[site][1:]) for site in SITES}
    _require(all(value <= 1e-5 for value in maxabs.values()),
             "cross-suffix prompt-last state changed")
    return {"scores": scores, "cross_suffix_state_maxabs": maxabs,
            "logit_hashes": hashes, "prompt_ids_sha256": prompt_hash}


def validate_complete_controls(no_hook: dict[str, Any],
                               read_only: dict[str, Any],
                               self_edit: dict[str, Any],
                               full_donor: dict[str, Any],
                               donor_no_hook: dict[str, Any],
                               tokenizer: Any) -> dict[str, bool]:
    """Require exact full-token parity before a narrower causal claim."""
    verified = []
    for row in (no_hook, read_only, self_edit, full_donor, donor_no_hook):
        observed = endpoint_v2.score_complete_answer(
            row.get("continuation_ids"), tokenizer)
        _require(observed["status"] == "complete"
                 and row.get("status") == observed["status"]
                 and row.get("label") == observed["label"],
                 "control output not verified EOS-complete")
        verified.append(observed)
    no_hook, read_only, self_edit, full_donor, donor_no_hook = verified
    result = {
        "read_only_matches_no_hook": endpoint_v2.exact_complete_parity(
            no_hook, read_only),
        "self_matches_no_hook": endpoint_v2.exact_complete_parity(
            no_hook, self_edit),
        "full_donor_matches_donor": endpoint_v2.exact_complete_parity(
            full_donor, donor_no_hook),
    }
    _require(all(result.values()), "complete-token control parity failed")
    return result


def validate_pilot_resources(record: dict[str, Any]) -> dict[str, Any]:
    """Numerical pilot limits; separate host-log audit remains mandatory."""
    fields = ("wall_seconds", "peak_mps_driver_bytes", "peak_tree_rss_bytes")
    _require(all(key in record and type(record[key]) in (int, float)
                 for key in fields), "pilot resource record incomplete")
    wall, mps, rss = (record[key] for key in fields)
    _require(0 < wall <= PILOT_SECONDS and 0 < mps < PILOT_MPS_LIMIT
             and 0 < rss < PILOT_RSS_LIMIT,
             "pilot wall or memory limit exceeded")
    return {"wall_seconds": wall, "peak_mps_driver_bytes": mps,
            "peak_tree_rss_bytes": rss, "host_audit_still_required": True}


def refuse_candidate_inference() -> None:
    raise RuntimeError("candidate inference HOLD: human QA and reviewed real backend/GO absent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run-development", action="store_true")
    parser.add_argument("--run-locked", action="store_true")
    args = parser.parse_args()
    if any((args.pilot, args.run_development, args.run_locked)):
        refuse_candidate_inference()
    if args.source_preflight:
        print(json.dumps(source_preflight(), sort_keys=True))
    elif args.show_plan:
        print(json.dumps(planned_case_ids(), sort_keys=True))
    else:
        parser.error("choose --source-preflight or --show-plan")


if __name__ == "__main__":
    main()
