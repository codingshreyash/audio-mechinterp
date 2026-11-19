"""Read-only saved-evidence audit for the frozen Whisper causal attempt.

This module never runs the candidate model. It validates the parent, full
scheduled denominator, source/donor tensors, edits, complete output paths and
binary float32 raw decoder evidence. Its JSON report is written outside the
one-shot attempt directory.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
from . import causal_parent_v1 as parent
from . import causal_plan_v1 as plan
from . import causal_worker_v1 as worker
from . import descriptive_worker_v1 as descriptive

AUDIT_PATH = parent.ATTEMPT.parent / "attempt_001_audit.json"
SITES = ("pre_block0", "final_encoder")
PHASES = ("detect", "transcribe")


def utc(value: str) -> datetime:
    when = datetime.fromisoformat(value)
    worker.require(when.tzinfo is not None, "naive evidence timestamp")
    return when


def saved_array(output: Path, ref: dict) -> np.ndarray:
    """Load only an authenticated .npy sidecar under worker output."""
    worker.require(isinstance(ref, dict) and isinstance(ref.get("path"), str),
                   "tensor sidecar reference missing")
    relative = PurePosixPath(ref["path"])
    worker.require(not relative.is_absolute() and ".." not in relative.parts
                   and relative.suffix == ".npy", "tensor sidecar path invalid")
    path = output / relative
    worker.require(path.is_file() and not path.is_symlink()
                   and path.resolve().is_relative_to(output.resolve()),
                   "tensor sidecar unavailable or escaped worker output")
    worker.require(path.stat().st_size == ref["bytes"]
                   and worker.sha_file(path) == ref["file_sha256"],
                   "tensor sidecar byte hash changed")
    value = np.load(path, allow_pickle=False)
    worker.require(value.dtype == np.float32 and str(value.dtype) == ref["dtype"]
                   and list(value.shape) == ref["shape"]
                   and np.isfinite(value).all()
                   and descriptive.sha_bytes(np.ascontiguousarray(value).tobytes())
                   == ref["tensor_sha256"],
                   "tensor sidecar content invalid")
    return value


def saved_host(attempt: Path, label: str) -> tuple[dict, str]:
    path = attempt / f"{label}_host.json"
    worker.require(path.is_file() and not path.is_symlink(),
                   "saved host receipt missing")
    evidence = json.loads(path.read_text())
    log = host_gate.load_archive(attempt, evidence, label)
    host_gate.validate_power(evidence)
    return evidence, log


def audit_parent(attempt: Path, admission: Path, go: Path) -> tuple[dict, dict]:
    receipt = worker.validate_go(admission, go)
    start_path = attempt / "launch_start.json"
    result_path = attempt / "launch_result.json"
    worker.require(start_path.is_file() and result_path.is_file()
                   and not start_path.is_symlink() and not result_path.is_symlink(),
                   "one-shot parent ledger missing")
    start, result = json.loads(start_path.read_text()), json.loads(result_path.read_text())
    names = parent.expected_names()
    worker.require(len(names) == len(plan.schedule())
                   and start.get("schema") == "nonspeech_whisper_causal_launch_v1"
                   and start.get("scheduled_call_names") == names
                   and start.get("source_identity") == receipt["sources"]
                   and start.get("admission_sha256") == worker.sha_file(admission)
                   and start.get("go_sha256") == worker.sha_file(go)
                   and result.get("admission_sha256") == worker.sha_file(admission)
                   and result.get("go_sha256") == worker.sha_file(go),
                   "parent/source/call binding invalid")
    progress = attempt / "worker_output/call_progress.jsonl"
    census = parent.progress_census(progress, names)
    worker.require(result.get("worker_call_census") == census
                   and result.get("worker_call_progress_sha256") == (
                       worker.sha_file(progress) if progress.is_file() else None),
                   "parent call census differs from durable worker progress")
    if result.get("status") != "completed_pending_saved_audit":
        return start, {"status": "invalid_or_partial", "result": result,
                       "call_census": census}
    worker.require(len(census["completed_call_names"]) == len(names)
                   and census["failed_attempted_call_name"] is None
                   and not census["censored_call_names"], "complete parent census invalid")
    pre, pre_log = saved_host(attempt, "pre")
    post, post_log = saved_host(attempt, "post")
    started = utc(start["start_utc"])
    exited = utc(result["worker_exit_observed_utc"])
    collected = utc(post["collected_utc"])
    worker.require(utc(pre["collected_utc"]) <= started <= exited <= collected
                   and 0 <= (collected-exited).total_seconds()
                   <= parent.MAX_POST_PROBE_LAG_SECONDS,
                   "host timing invalid")
    host_gate.validate_log(pre_log, utc(pre["collected_utc"])-timedelta(minutes=10),
                           utc(pre["collected_utc"]), utc(pre["collected_utc"]))
    host_gate.validate_log(post_log, started, collected, collected)
    worker.require(result.get("pre_host_sha256") == worker.sha_file(attempt / "pre_host.json")
                   and result.get("post_host_sha256") == worker.sha_file(attempt / "post_host.json")
                   and result.get("host_collected_utc") == [pre["collected_utc"],
                                                             post["collected_utc"]],
                   "parent host hashes differ")
    rss_path = attempt / "worker_rss_samples.jsonl"
    worker.require(rss_path.is_file() and not rss_path.is_symlink()
                   and worker.sha_file(rss_path) == result["worker_rss_sample_log_sha256"],
                   "parent RSS log changed")
    rss_rows = [json.loads(line) for line in rss_path.read_text().splitlines()]
    worker.require(rss_rows and len(rss_rows) == result["worker_rss_samples"]
                   and all(row["sample_index"] == index for index, row in enumerate(rss_rows, 1))
                   and max(row["tree_rss_bytes"] for row in rss_rows)
                   == result["worker_peak_tree_rss_bytes"]
                   and result["worker_peak_tree_rss_bytes"] < worker.MAX_RSS_BYTES
                   and result["worker_rss_limit_bytes"] == worker.MAX_RSS_BYTES
                   and result["worker_stop_reason"] == "normal_exit"
                   and result["worker_returncode"] == 0
                   and not result["worker_timed_out"]
                   and result["worker_elapsed_seconds"] <= worker.MAX_SECONDS
                   and parent.output_bytes(attempt) <= worker.MAX_EVIDENCE_BYTES,
                   "parent resource or completion evidence invalid")
    return start, {"status": "complete", "result": result, "call_census": census,
                   "rss_samples": len(rss_rows),
                   "sampled_peak_tree_rss_bytes": result["worker_peak_tree_rss_bytes"]}


def _processor_and_geometry():
    from transformers import WhisperProcessor
    assets = descriptive.verify_local_assets("whisper_multilingual")
    processor = WhisperProcessor.from_pretrained(assets["snapshot"], local_files_only=True)
    geometry = {}
    for case in descriptive.trusted_cases():
        audio = case["pcm"].astype(np.float32) / np.float32(32768.0)
        processed = processor(audio, sampling_rate=16000, return_tensors="pt",
                              return_attention_mask=True)
        geometry[case["case_id"]] = descriptive.whisper_geometry(
            processed.attention_mask, len(case["pcm"]), 3000)
    return processor, geometry


def _validate_output(output: dict, sidecars: dict, plan_row: dict,
                     saved_dir: Path, tokenizer: Any, lang_to_id: dict[str, int],
                     seen_refs: set[str]) -> tuple[list[int], list[np.ndarray]]:
    ids = output["sequence_ids"]
    prefix = output["prefix_ids"]
    generated = output["generated_ids"]
    worker.require(isinstance(ids, list) and all(isinstance(x, int) for x in ids)
                   and ids == prefix + generated
                   and len(prefix) == 4 and 1 <= len(generated) <= 128
                   and output["actual_generation_steps"] == len(generated)
                   and prefix[0] == 50258 and prefix[1] in set(lang_to_id.values())
                   and prefix[2:] == [50359, 50363],
                   "complete Whisper sequence or prefix invalid")
    detection = output.get("language_detection")
    if plan_row["arm"] != "no_hook":
        worker.require(isinstance(detection, dict)
                       and detection.get("chosen_language_ids") == [prefix[1]]
                       and detection.get("candidate_count") == len(lang_to_id)
                       and set(detection.get("language_logits", {})) == set(lang_to_id),
                       "phase-specific language evidence incomplete")
        selected = min(lang_to_id.values(), key=lambda token: (
            -float(next(value for label, value in detection["language_logits"].items()
                        if lang_to_id[label] == token)), token))
        worker.require(selected == prefix[1],
                       "language token differs from recorded detection logits")
    eos = int(tokenizer.eos_token_id)
    worker.require(output["eos_seen"] == (eos in generated)
                   and (eos not in generated or generated[-1] == eos)
                   and output["termination"] == (
                       "eos" if generated[-1] == eos else
                       "length_censored" if len(generated) == 128 else "incomplete_unknown")
                   and output["censored"] == (output["termination"] == "length_censored")
                   and output["transcript"] == tokenizer.decode(
                       generated, skip_special_tokens=True)
                   and output["text_with_special_tokens"] == tokenizer.decode(
                       ids, skip_special_tokens=False),
                   "Whisper EOS/censor or independent decoding invalid")
    first = saved_array(saved_dir, sidecars["first_raw"])
    seen_refs.add(sidecars["first_raw"]["path"])
    worker.require(list(first.shape) == [51865], "first raw logit shape changed")
    hidden = saved_array(saved_dir, sidecars["first_decoder_state"])
    seen_refs.add(sidecars["first_decoder_state"]["path"])
    worker.require(list(hidden.shape) == [1, len(prefix), 384],
                   "first decoder state shape changed")
    if "first_decoder_step" in output:
        worker.require(output["first_decoder_step"]["sha256"]
                       == descriptive.sha_bytes(hidden.tobytes())
                       and output["first_decoder_step"]["shape"] == list(hidden.shape),
                       "first decoder state summary differs")
    if "first_raw_logits" in output:
        worker.require(output["first_raw_logits"]["sha256"]
                       == descriptive.sha_bytes(first.tobytes())
                       and output["first_raw_logits"]["shape"] == list(first.shape),
                       "first raw logit summary differs")
    score_rows = output["score_rows"]
    worker.require(len(score_rows) == len(generated)
                   and all(score["token_id"] == token and score["top_id"] == token
                           for score, token in zip(score_rows, generated, strict=True))
                   and score_rows[0]["raw_logits_sha256"]
                   == descriptive.sha_bytes(first.tobytes())
                   and score_rows[0]["raw_top_id"] == int(first.argmax())
                   and abs(score_rows[0]["raw_selected_logit"]
                           - float(first[generated[0]])) <= 1e-6,
                   "raw first-step logit row differs")
    all_raw = []
    full_refs = sidecars.get("full_raw", [])
    if plan_row["save_full_raw_logits_each_step"]:
        worker.require(len(full_refs) == len(generated),
                       "full raw logit steps missing")
    else:
        worker.require(not full_refs, "unexpected full raw logit sidecars")
    for step, ref in enumerate(full_refs):
        value = saved_array(saved_dir, ref)
        seen_refs.add(ref["path"])
        worker.require(list(value.shape) == [51865]
                       and score_rows[step]["raw_logits_sha256"]
                       == descriptive.sha_bytes(value.tobytes())
                       and score_rows[step]["raw_top_id"] == int(value.argmax())
                       and abs(score_rows[step]["raw_selected_logit"]
                               - float(value[generated[step]])) <= 1e-6,
                       "full raw logit step differs")
        all_raw.append(value)
    return ids, all_raw


def expected_edit(native: np.ndarray, donor: np.ndarray, plan_row: dict,
                  phase: str) -> tuple[np.ndarray, list[int], int | None]:
    """Rebuild the edited tensor from frozen source/donor and arm identity."""
    arm = plan_row["arm"]
    active = plan_row["active_rows"]
    if arm in ("no_hook", "read_only"):
        return native.copy(), [0, 0], None
    if arm == "self_detect" and phase != "detect" or \
            arm == "self_transcribe" and phase != "transcribe":
        return native.copy(), [0, 0], None
    if arm in ("self_detect", "self_transcribe", "self_both"):
        return native.copy(), [0, 1500], None
    if arm == "donor_detect" and phase != "detect" or \
            arm == "donor_transcribe" and phase != "transcribe":
        return native.copy(), [0, 0], None
    if arm in ("donor_detect", "donor_transcribe", "donor_both"):
        return donor.copy(), [0, 1500], None
    if arm == "active_both":
        result = native.copy()
        result[:, :active, :] = donor[:, :active, :]
        return result, [0, active], None
    if arm == "displaced_both":
        result = native.copy()
        result[:, active:2*active, :] = donor[:, active:2*active, :]
        return result, [active, 2*active], None
    if arm in ("random_1_both", "random_2_both"):
        seed = plan_row["seeds"][phase]
        noise = np.random.default_rng(seed).standard_normal(native.shape).astype(np.float32)
        target = float(np.linalg.norm((donor-native).astype(np.float64).ravel()))
        if target == 0.0:
            edit = np.zeros_like(native)
        else:
            unit = noise.astype(np.float64) / np.linalg.norm(
                noise.astype(np.float64).ravel())
            edit = (unit*target).astype(np.float32)
        return (native + edit).astype(np.float32), [0, 1500], seed
    raise ValueError(f"unknown frozen arm: {arm}")


def audit_hooks(row: dict, plan_row: dict,
                native: dict[tuple[str, str, str], np.ndarray]) -> None:
    stage, case, arm = plan_row["stage"], plan_row["case_id"], plan_row["arm"]
    hooks = row["hook"]
    if arm == "no_hook":
        worker.require(hooks == {} and row.get("native_state_refs") == {},
                       "true no-hook call contains hook evidence")
        return
    worker.require(set(hooks) == set(PHASES), "hook phase inventory invalid")
    selected = {"A": "final_encoder", "B": "pre_block0"}.get(stage)
    for phase in PHASES:
        worker.require(set(hooks[phase]) == set(SITES),
                       "hook site inventory invalid")
        for site in SITES:
            item = hooks[phase][site]
            worker.require(item["hook_calls"] == 1
                           and isinstance(item["native_sha256"], str)
                           and isinstance(item["edited_sha256"], str),
                           "hook count or hash missing")
            if stage == "donor" or arm == "read_only":
                reference = native[(case, site, phase)]
                digest = descriptive.sha_bytes(reference.tobytes())
                worker.require(item["native_sha256"] == digest
                               and item["edited_sha256"] == digest,
                               "baseline native hook tensor differs")
            if site != selected:
                worker.require(item["edited_sha256"] == item["native_sha256"]
                               and item["support_rows"] == [0, 0]
                               and item.get("seed") is None,
                               "unexpected edit at unselected hook")
                if stage == "A":
                    worker.require(item["native_sha256"] == descriptive.sha_bytes(
                        native[(case, site, phase)].tobytes()),
                        "pre-block native changed during final-site intervention")
                continue
            if stage == "donor":
                continue
            source = native[(case, site, phase)]
            donor = native[(plan_row["donor_id"], site, phase)]
            expected, support, seed = expected_edit(source, donor, plan_row, phase)
            actual_edit_l2 = float(np.linalg.norm(
                (expected-source).astype(np.float64).ravel()))
            donor_l2 = float(np.linalg.norm(
                (donor-source).astype(np.float64).ravel()))
            worker.require(item["native_sha256"] == descriptive.sha_bytes(source.tobytes())
                           and item["edited_sha256"] == descriptive.sha_bytes(expected.tobytes())
                           and item["support_rows"] == support
                           and item.get("seed") == seed
                           and math.isclose(item["edit_l2"], actual_edit_l2,
                                            rel_tol=1e-5, abs_tol=1e-6)
                           and math.isclose(item["donor_delta_l2"], donor_l2,
                                            rel_tol=1e-5, abs_tol=1e-6),
                           "selected-site edited tensor or norm differs")


def audit_attempt(attempt: Path, admission: Path, go: Path) -> dict:
    """Recompute saved evidence; no model.forward or generate call occurs."""
    _start, parent_result = audit_parent(attempt, admission, go)
    if parent_result["status"] != "complete":
        return {"schema": "nonspeech_whisper_causal_saved_audit_v1",
                "status": "invalid_or_partial", "call_census": parent_result["call_census"]}
    worker_dir = attempt / "worker_output"
    records_path = worker_dir / "records.jsonl"
    worker.require(records_path.is_file() and not records_path.is_symlink(),
                   "worker records missing")
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    schedule = plan.schedule()
    worker.require(len(records) == len(schedule)+2
                   and records[0].get("type") == "start"
                   and records[0].get("planned_calls") == len(schedule)
                   and records[0].get("schedule_sha256") == worker.sha_bytes(
                       worker.canonical(schedule).encode())
                   and records[0].get("admission_sha256") == worker.sha_file(admission)
                   and records[0].get("go_sha256") == worker.sha_file(go)
                   and records[-1].get("type") == "complete"
                   and records[-1].get("planned_calls") == len(schedule)
                   and records[-1].get("attempted_calls") == len(schedule),
                   "worker records inventory invalid")
    processor, geometry = _processor_and_geometry()
    tokenizer = processor.tokenizer
    snapshot = descriptive.verify_local_assets("whisper_multilingual")["snapshot"]
    generation_config = json.loads((Path(snapshot) / "generation_config.json").read_text())
    lang_to_id = {name: int(token) for name, token in generation_config["lang_to_id"].items()}
    native: dict[tuple[str, str, str], np.ndarray] = {}
    outputs: dict[tuple[str, str, str], list[int]] = {}
    full_logits: dict[tuple[str, str, str], list[np.ndarray]] = {}
    worker.require(worker.sha_file(worker.DESCRIPTIVE_V2_RECORDS)
                   == worker.DESCRIPTIVE_V2_RECORDS_SHA,
                   "valid descriptive V2 comparator bytes changed")
    v2_outputs = {item["case_id"]: item["output"]["sequence_ids"]
                  for item in (json.loads(line) for line in
                               worker.DESCRIPTIVE_V2_RECORDS.read_text().splitlines())
                  if item.get("type") == "case"}
    worker.require(len(v2_outputs) == 20 and set(v2_outputs) == set(plan.frozen_cases()),
                   "valid descriptive V2 case census changed")
    sidecar_paths: set[str] = set()
    parity_failures: list[str] = []
    censored_calls: list[str] = []
    donor_output_changes = 0
    for index, (plan_row, row) in enumerate(zip(schedule, records[1:-1], strict=True), 1):
        name = f'{plan_row["stage"]}_{plan_row["case_id"]}_{plan_row["arm"]}'
        worker.require(row.get("type") == "forward"
                       and row.get("call_index") == index
                       and row.get("call_name") == name
                       and row.get("plan_row") == plan_row
                       and row.get("input_geometry") == geometry[plan_row["case_id"]]
                       and row.get("hook_counts") == (
                           {} if plan_row["arm"] == "no_hook" else {
                               "detect": {"pre_block0": 1, "final_encoder": 1},
                               "transcribe": {"pre_block0": 1, "final_encoder": 1}}),
                       "forward row diverges from frozen schedule")
        ids, raw_vectors = _validate_output(row["output"], row["sidecars"],
                                             plan_row, worker_dir, tokenizer,
                                             lang_to_id,
                                             sidecar_paths)
        if row["output"]["termination"] != "eos":
            censored_calls.append(name)
        worker.require(row["first_raw_meta"]["sha256"]
                       == row["sidecars"]["first_raw"]["tensor_sha256"]
                       and row["first_raw_meta"]["shape"] == [plan.VOCAB_SIZE]
                       and row["first_decoder_meta"]["sha256"]
                       == row["sidecars"]["first_decoder_state"]["tensor_sha256"]
                       and row["first_decoder_meta"]["shape"] == [1, 4, 384],
                       "first-step metadata differs from binary sidecars")
        key = (plan_row["stage"], plan_row["case_id"], plan_row["arm"])
        outputs[key] = ids
        if raw_vectors:
            full_logits[key] = raw_vectors
        refs = row.get("native_state_refs", {})
        if refs:
            worker.require(plan_row["stage"] == "donor" and plan_row["arm"] == "read_only"
                           or plan_row["stage"] == "A" and plan_row["arm"] == "read_only",
                           "unexpected native tensor capture row")
            for site in SITES:
                for phase in PHASES:
                    ref = refs[site][phase]
                    tensor = saved_array(worker_dir, ref)
                    sidecar_paths.add(ref["path"])
                    worker.require(list(tensor.shape) == [1, 1500, 384],
                                   "native state geometry changed")
                    native_key = (plan_row["case_id"], site, phase)
                    worker.require(native_key not in native,
                                   "duplicate native state evidence")
                    native[native_key] = tensor
        # Hook algebra and parity are checked after all baseline source states exist.
    worker.require(len(native) == 80, "native source/donor state inventory incomplete")
    worker.require(len(full_logits) == 38, "full donor raw logit inventory incomplete")
    actual_sidecars = {str(path.relative_to(worker_dir)) for path in worker_dir.rglob("*.npy")}
    worker.require(actual_sidecars == sidecar_paths,
                   "unexpected or missing binary sidecar")
    # Complete control, donor parity and selected-site tensor algebra are
    # re-evaluated without trusting worker parity flags or edit summaries.
    for plan_row, row in zip(schedule, records[1:-1], strict=True):
        audit_hooks(row, plan_row, native)
        stage, case, arm = plan_row["stage"], plan_row["case_id"], plan_row["arm"]
        ids = outputs[(stage, case, arm)]
        flags = row["parity"]
        worker.require(flags.get("descriptive_v2_tokens_match")
                       == (ids == v2_outputs[case]),
                       "V2 comparator parity flag differs from saved tokens")
        if arm == "no_hook" and not flags["descriptive_v2_tokens_match"]:
            parity_failures.append(f"fresh_no_hook_vs_descriptive_v2:{stage}:{case}")
        if stage == "donor":
            natural = outputs[("donor", case, "no_hook")]
            worker.require(flags.get("recipient_tokens_match") == (ids == natural)
                           and flags.get("donor_tokens_match") == (ids == natural),
                           "donor baseline parity flags differ")
            if arm == "read_only" and ids != outputs[("donor", case, "no_hook")]:
                parity_failures.append(f"donor_read_only:{case}")
            continue
        recipient = outputs[("A", case, "no_hook")]
        donor = outputs[("donor", plan_row["donor_id"], "no_hook")]
        worker.require(flags.get("recipient_tokens_match") == (ids == recipient)
            and flags.get("donor_tokens_match") == (ids == donor),
            "worker token parity flags differ from saved full output")
        if stage == "B" and arm == "no_hook":
            worker.require(flags.get("repeated_no_hook_match") == (ids == recipient),
                           "Stage B repeat parity flag invalid")
        if arm in ("no_hook", "read_only", "self_detect", "self_transcribe", "self_both"):
            if ids != recipient:
                parity_failures.append(f"self_or_read_only:{stage}:{case}:{arm}")
        if arm == "donor_both":
            if ids != donor:
                parity_failures.append(f"donor_output:{stage}:{case}")
            else:
                donor_output_changes += int(recipient != donor)
            observed_raw = full_logits[(stage, case, arm)]
            donor_raw = full_logits[("donor", plan_row["donor_id"], "no_hook")]
            if len(observed_raw) != len(donor_raw) or any(
                    float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))
                    > 1e-5 for left, right in zip(observed_raw, donor_raw)):
                parity_failures.append(f"donor_raw_logits:{stage}:{case}")
            if len(observed_raw) == len(donor_raw):
                measured = max(float(np.max(np.abs(
                    left.astype(np.float64)-right.astype(np.float64))))
                    for left, right in zip(observed_raw, donor_raw))
                worker.require(math.isclose(flags["donor_raw_maxabs"], measured,
                                            rel_tol=1e-6, abs_tol=1e-6)
                               and flags["donor_raw_within_1e-5"] == (measured <= 1e-5),
                               "worker raw-logit parity flag differs")
            else:
                worker.require(flags["donor_raw_maxabs"] is None
                               and flags["donor_raw_within_1e-5"] is False,
                               "worker raw-logit censor flag differs")
    return {"schema": "nonspeech_whisper_causal_saved_audit_v1",
            "status": "saved_evidence_complete" if not parity_failures and not censored_calls
                      else "censored_or_parity_failed",
            "forwards": len(schedule), "native_tensors": len(native),
            "binary_sidecars": len(sidecar_paths),
            "donor_both_complete_output_changes": donor_output_changes,
            "parity_failures": parity_failures, "censored_calls": censored_calls,
            "sampled_peak_tree_rss_bytes": parent_result["sampled_peak_tree_rss_bytes"],
            "records_sha256": worker.sha_file(records_path),
            "limitations": ["No attacker-suppliable audio attack or security property tested.",
                            "The saved auditor validates source/donor state edits, not a new model forward."]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, default=parent.ATTEMPT)
    parser.add_argument("--admission", type=Path, default=worker.ADMISSION)
    parser.add_argument("--go", type=Path, default=worker.GO)
    parser.add_argument("--output", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    worker.require(args.output.is_absolute() or args.output.parent == AUDIT_PATH.parent,
                   "saved audit output path invalid")
    worker.require(not args.output.exists() and args.output.parent.is_dir(),
                   "saved audit output must be new")
    result = audit_attempt(args.attempt, args.admission, args.go)
    worker.write_once(args.output, result)
    print(worker.canonical({"status": result["status"],
                            "forwards": result.get("forwards"),
                            "parity_failures": len(result.get("parity_failures", []))}))


if __name__ == "__main__":
    main()
