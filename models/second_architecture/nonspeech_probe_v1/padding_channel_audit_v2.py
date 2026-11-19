"""Read-only, no-model audit of the 436-call padding-channel V2 attempt."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

import numpy as np

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
from . import causal_parent_v2 as bounded
from . import descriptive_worker_v1 as descriptive
from . import padding_channel_core_v1 as core
from . import padding_channel_plan_v5 as plan
from . import padding_channel_worker_v2 as worker
from . import padding_channel_parent_v2 as parent

AUDIT_PATH = worker.ATTEMPT.parent / "attempt_001_audit.json"


def saved_array(output: Path, ref: dict) -> np.ndarray:
    worker.require(isinstance(ref, dict) and isinstance(ref.get("path"), str),
                   "sidecar reference missing")
    relative = PurePosixPath(ref["path"])
    worker.require(not relative.is_absolute() and ".." not in relative.parts
                   and relative.suffix == ".npy", "unsafe sidecar path")
    path = output / relative
    worker.require(path.is_file() and not path.is_symlink()
                   and path.resolve().is_relative_to(output.resolve())
                   and path.stat().st_size == ref["bytes"]
                   and worker.sha_file(path) == ref["file_sha256"],
                   "sidecar missing or byte hash changed")
    value = np.load(path, allow_pickle=False)
    worker.require(value.dtype == np.float32 and list(value.shape) == ref["shape"]
                   and str(value.dtype) == ref["dtype"]
                   and core.sha_array(value) == ref["tensor_sha256"],
                   "sidecar tensor hash/shape/dtype changed")
    processed = relative.parts[0] == "processed"
    if processed:
        worker.require(not np.isnan(value).any() and not np.isposinf(value).any()
                       and bool(np.isfinite(value).any()), "processed score invalid")
    else:
        worker.require(bool(np.isfinite(value).all()), "nonfinite raw/state sidecar")
    return value


def _utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    worker.require(timestamp.tzinfo is not None, "naive timestamp")
    return timestamp


def audit_parent(attempt: Path, admission: Path, go: Path) -> dict:
    receipt = worker.validate_go(admission, go)
    start = json.loads((attempt / "launch_start.json").read_text())
    result = json.loads((attempt / "launch_result.json").read_text())
    names = parent.expected_names()
    worker.require(start.get("schema") == "whisper_padding_launch_v2"
                   and start.get("scheduled_call_names") == names
                   and start.get("source_identity") == receipt["sources"]
                   and start.get("admission_sha256") == worker.sha_file(admission)
                   and start.get("go_sha256") == worker.sha_file(go)
                   and result.get("admission_sha256") == worker.sha_file(admission)
                   and result.get("go_sha256") == worker.sha_file(go),
                   "parent source/schedule identity invalid")
    progress = attempt / "worker_output/call_progress.jsonl"
    census = bounded.progress_census(progress, names)
    worker.require(result.get("worker_call_census") == census
                   and result.get("worker_call_progress_sha256") == (
                       worker.sha_file(progress) if progress.is_file() else None),
                   "parent progress census invalid")
    completion_path = attempt / "launch_completion.json"
    if result.get("status") != "candidate_complete" or not completion_path.is_file():
        return {"status": "invalid_or_partial", "call_census": census}
    completion = json.loads(completion_path.read_text())
    worker.require(completion.get("schema") == "whisper_padding_completion_v2"
                   and completion.get("status") == "completed_pending_saved_audit"
                   and completion.get("result_sha256")
                   == worker.sha_file(attempt / "launch_result.json")
                   and 0 < completion.get("parent_elapsed_seconds", float("inf"))
                   <= plan.WALL_SECONDS
                   and 0 < result.get("worker_seconds_granted", 0)
                   < plan.WALL_SECONDS
                   and result.get("worker_elapsed_seconds", float("inf"))
                   <= result["worker_seconds_granted"]
                   and _utc(completion["completed_utc"])-_utc(start["start_utc"])
                   <= timedelta(seconds=plan.WALL_SECONDS),
                   "whole-parent completion/deadline evidence invalid")
    worker.require(len(census["completed_call_names"]) == 436
                   and census["failed_attempted_call_name"] is None
                   and not census["censored_call_names"],
                   "parent complete census invalid")
    pre = json.loads((attempt / "pre_host.json").read_text())
    post = json.loads((attempt / "post_host.json").read_text())
    pre_log = host_gate.load_archive(attempt, pre, "pre")
    post_log = host_gate.load_archive(attempt, post, "post")
    host_gate.validate_power(pre)
    host_gate.validate_power(post)
    started, pre_time, exited, collected = (_utc(start["start_utc"]),
        _utc(pre["collected_utc"]), _utc(result["worker_exit_observed_utc"]),
        _utc(post["collected_utc"]))
    worker.require(started <= pre_time <= exited <= collected
                   <= _utc(result["end_utc"])
                   <= _utc(completion["completed_utc"])
                   and 0 <= (collected-exited).total_seconds()
                   <= parent.MAX_POST_PROBE_LAG_SECONDS,
                   "host timing invalid")
    host_gate.validate_log(pre_log, pre_time-timedelta(minutes=10),
                           pre_time, pre_time)
    host_gate.validate_log(post_log, started, collected, collected)
    worker.require(result["pre_host_sha256"] == worker.sha_file(attempt / "pre_host.json")
                   and result["post_host_sha256"] == worker.sha_file(attempt / "post_host.json"),
                   "host receipt hashes differ")
    rss_path = attempt / "worker_rss_samples.jsonl"
    samples = [json.loads(line) for line in rss_path.read_text().splitlines()]
    worker.require(samples and worker.sha_file(rss_path)
                   == result["worker_rss_sample_log_sha256"]
                   and len(samples) == result["worker_rss_samples"]
                   and max(row["tree_rss_bytes"] for row in samples)
                   == result["worker_peak_tree_rss_bytes"]
                   and result["worker_peak_tree_rss_bytes"] < plan.RSS_LIMIT_BYTES
                   and result["worker_returncode"] == 0
                   and result["worker_stop_reason"] == "normal_exit"
                   and result["worker_elapsed_seconds"] <= plan.WALL_SECONDS
                   and bounded.output_bytes(attempt) <= plan.EVIDENCE_LIMIT_BYTES,
                   "parent resource evidence invalid")
    return {"status": "complete", "call_census": census,
            "sampled_peak_tree_rss_bytes": result["worker_peak_tree_rss_bytes"]}


def _sequence(output: dict, tokenizer: object) -> list[int]:
    ids = output["sequence_ids"]
    prefix, generated = output["prefix_ids"], output["generated_ids"]
    worker.require(ids == prefix + generated and len(prefix) == 4
                   and 1 <= len(generated) <= plan.MAX_NEW_TOKENS
                   and output["actual_generation_steps"] == len(generated)
                   and prefix[0] == 50258 and prefix[2:] == [50359, 50363],
                   "complete prefix/generated path invalid")
    eos = int(tokenizer.eos_token_id)
    worker.require(output["eos_seen"] == (eos in generated)
                   and (eos not in generated or generated[-1] == eos)
                   and output["termination"] == (
                       "eos" if generated[-1] == eos else
                       "length_censored" if len(generated) == plan.MAX_NEW_TOKENS
                       else "incomplete_unknown")
                   and output["transcript"] == tokenizer.decode(
                       generated, skip_special_tokens=True)
                   and output["text_with_special_tokens"] == tokenizer.decode(
                       ids, skip_special_tokens=False),
                   "independent text/EOS decode invalid")
    detection = output.get("language_detection")
    if detection is not None:
        worker.require(detection.get("chosen_language_ids") == [prefix[1]]
                       and detection.get("candidate_count")
                       == len(detection.get("language_logits", {})),
                       "phase language evidence differs from prefix")
    return ids


def _check_sidecars(output_dir: Path, item: dict,
                    row: dict, seen: set[str]) -> None:
    refs, output = item["sidecars"], item["output"]
    first_raw = saved_array(output_dir, refs["first_raw"])
    first_processed = saved_array(output_dir, refs["first_processed"])
    first_decoder = saved_array(output_dir, refs["first_decoder"])
    worker.validate_first_step_arrays(first_raw, first_processed, first_decoder)
    selected = saved_array(output_dir, refs["selected_scores"])
    for ref in (refs["first_raw"], refs["first_processed"],
                refs["first_decoder"], refs["selected_scores"]):
        seen.add(ref["path"])
    worker.require(first_raw.shape == first_processed.shape == (plan.VOCAB_SIZE,)
                   and first_decoder.shape == (1, 4, 384)
                   and selected.shape == (len(output["generated_ids"]),)
                   and bool(np.isfinite(selected).all())
                   and core.sha_array(first_raw) == item["first_raw_meta"]["sha256"]
                   and core.sha_array(first_decoder) == item["first_decoder_meta"]["sha256"]
                   and output["score_rows"][0]["raw_logits_sha256"]
                   == core.sha_array(first_raw)
                   and output["score_rows"][0]["processed_logits_sha256"]
                   == core.sha_array(first_processed),
                   "first-step array evidence invalid")
    worker.require(len(output["score_rows"]) == len(selected)
                   and all(score["token_id"] == output["generated_ids"][i]
                           and score["top_id"] == output["generated_ids"][i]
                           and math.isclose(float(selected[i]),
                                        float(score["selected_logit"]),
                                        rel_tol=1e-6, abs_tol=1e-6)
                           for i, score in enumerate(output["score_rows"])),
                   "selected-token scores differ")
    full = refs["full_raw"]
    worker.require(len(full) == (len(output["generated_ids"])
                                if row["arm"] in ("no_hook", "DD") else 0),
                   "full raw-logit inventory invalid")
    for step, ref in enumerate(full):
        tensor = saved_array(output_dir, ref)
        seen.add(ref["path"])
        worker.require(tensor.shape == (plan.VOCAB_SIZE,)
                       and core.sha_array(tensor)
                       == output["score_rows"][step]["raw_logits_sha256"],
                       "full raw-logit step invalid")


def _check_hooks(item: dict, row: dict,
                 native: dict[tuple[str, str, str], np.ndarray]) -> None:
    if row["hook_policy"] == "none":
        worker.require(item["hook"] == {} and item["hook_counts"] == {},
                       "true no-hook contains hook evidence")
        return
    worker.require(item["hook_counts"] == {
        phase: {site: 1 for site in plan.SITES} for phase in plan.PHASES},
        "hook phase/site count changed")
    for phase in plan.PHASES:
        for site in plan.SITES:
            hook = item["hook"][phase][site]
            worker.require(hook["calls"] == 1, "hook count invalid")
            if row["hook_policy"] == "capture_both_sites_both_phases":
                value = native[(row["case_id"], site, phase)]
                worker.require(hook["native_sha256"] == core.sha_array(value)
                               and hook["edited_sha256"] == core.sha_array(value)
                               and hook["support"] == [0, 0],
                               "read-only capture changed tensor")
            elif core.selected(row, site, phase):
                recipient = native[(row["case_id"], site, phase)]
                donor = native[(row["donor_id"], site, phase)]
                expected, meta = core.edit(recipient, recipient, donor,
                                           row, site, phase)
                worker.require(hook["native_sha256"] == core.sha_array(recipient)
                               and hook["edited_sha256"] == core.sha_array(expected)
                               and hook["support"] == meta["support"]
                               and hook["seed"] == meta["seed"]
                               and math.isclose(hook["target_l2"], meta["target_l2"],
                                                rel_tol=1e-6, abs_tol=1e-6)
                               and math.isclose(hook["realized_l2"], meta["realized_l2"],
                                                rel_tol=1e-6, abs_tol=1e-6)
                               and (hook["cosine"] == meta["cosine"]
                                    or hook["cosine"] is not None
                                    and meta["cosine"] is not None
                                    and math.isclose(hook["cosine"], meta["cosine"],
                                                     rel_tol=1e-6, abs_tol=1e-6)),
                               "selected-site edit reconstruction failed")
            else:
                worker.require(hook["native_sha256"] == hook["edited_sha256"]
                               and hook["support"] == [0, 0],
                               "nonselected site/phase was edited")


def verify_input_binding(output_dir: Path, item: dict, case_id: str,
                         expected: np.ndarray, expected_mask: list[int],
                         known_refs: dict[str, dict], seen: set[str]) -> None:
    """Bind one call's actual pre/post feature hash to trusted PCM processing."""
    feature_ref = item["input_feature_ref"]
    if case_id not in known_refs:
        features = saved_array(output_dir, feature_ref)
        core.assert_array(features, "log_mel")
        worker.require(np.array_equal(features, expected),
                       "saved input features differ from pinned PCM/processor")
        known_refs[case_id] = feature_ref
        seen.add(feature_ref["path"])
    worker.require(feature_ref == known_refs[case_id]
                   and feature_ref["path"] == f"input/{case_id}_features.npy"
                   and item["input_feature_sha256_before"]
                   == item["input_feature_sha256_after"]
                   == feature_ref["tensor_sha256"] == core.sha_array(expected)
                   and item["input_geometry"]["attention_mask"] == expected_mask,
                   "per-call input feature or mask binding invalid")


def audit_attempt(attempt: Path, admission: Path, go: Path) -> dict:
    parent_result = audit_parent(attempt, admission, go)
    if parent_result["status"] != "complete":
        return {"schema": "whisper_padding_saved_audit_v1",
                "status": "invalid_or_partial", **parent_result}
    output_dir = attempt / "worker_output"
    records = [json.loads(line) for line in (output_dir / "records.jsonl")
               .read_text().splitlines()]
    rows = plan.schedule()
    worker.require(len(records) == len(rows)+2
                   and records[0]["type"] == "start"
                   and records[0]["schedule_sha256"] == plan.schedule_sha256()
                   and records[-1]["type"] == "complete"
                   and records[-1]["attempted_calls"] == 436,
                   "worker record census invalid")
    snapshot = descriptive.verify_local_assets("whisper_multilingual")["snapshot"]
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    tokenizer = processor.tokenizer
    trusted = {case["case_id"]: case for case in descriptive.trusted_cases()}
    worker.require(set(trusted) == set(plan.population()),
                   "trusted PCM population changed")
    expected_inputs = {}
    expected_masks = {}
    for case_id, case in trusted.items():
        audio = case["pcm"].astype(np.float32) / np.float32(32768.0)
        processed = processor(audio, sampling_rate=16000, return_tensors="pt",
                              return_attention_mask=True)
        expected = np.ascontiguousarray(processed.input_features.detach().numpy())
        core.assert_array(expected, "log_mel")
        expected_inputs[case_id] = expected
        expected_masks[case_id] = [int(x) for x in processed.attention_mask[0].tolist()]
    native: dict[tuple[str, str, str], np.ndarray] = {}
    feature_refs: dict[str, dict] = {}
    seen: set[str] = set()
    outputs: dict[tuple[str, str, str], dict] = {}
    masks: dict[str, list[int]] = {}
    censored: list[str] = []
    for index, (plan_row, item) in enumerate(zip(rows, records[1:-1], strict=True), 1):
        name = worker.call_name(plan_row)
        worker.require(item["type"] == "forward" and item["call_index"] == index
                       and item["call_name"] == name and item["plan_row"] == plan_row,
                       "worker forward differs from frozen schedule")
        geometry = item["input_geometry"]
        active = plan.geometry(plan_row["case_id"], "log_mel")["active"]
        mask = geometry["attention_mask"]
        worker.require(geometry["pcm_frames"] == (32000 if active == 200 else 128000)
                       and geometry["mel_frames"] == 3000
                       and geometry["active_mel_frames"] == active
                       and geometry["nominal_active_encoder_rows"] == active // 2
                       and mask == [1] * active + [0] * (3000-active)
                       and geometry["mask_sha256"] == worker.sha_bytes(bytes(mask)),
                       "processor attention-mask geometry differs")
        if plan_row["arm"] == "no_hook":
            masks[plan_row["case_id"]] = mask
        else:
            worker.require(mask == masks[plan_row["donor_id"]],
                           "recipient/donor mask differs")
        case_id = plan_row["case_id"]
        verify_input_binding(output_dir, item, case_id,
                             expected_inputs[case_id], expected_masks[case_id],
                             feature_refs, seen)
        _sequence(item["output"], tokenizer)
        worker.require((item["output"].get("language_detection") is None)
                       == (plan_row["hook_policy"] == "none"),
                       "language-detection evidence/hook policy mismatch")
        _check_sidecars(output_dir, item, plan_row, seen)
        if item["output"]["termination"] != "eos":
            censored.append(name)
        outputs[(plan_row["site"], plan_row["case_id"], plan_row["arm"])] = item
        refs = item["native_state_refs"]
        if refs:
            worker.require(plan_row["arm"] in ("read_only_donor_capture", "read_only"),
                           "unexpected native state sidecars")
            for site in plan.SITES:
                for phase in plan.PHASES:
                    tensor = saved_array(output_dir, refs[site][phase])
                    seen.add(refs[site][phase]["path"])
                    core.assert_array(tensor, site)
                    key = (plan_row["case_id"], site, phase)
                    worker.require(key not in native, "duplicate native tensor sidecar")
                    native[key] = tensor
    worker.require(len(native) == 80 and len(feature_refs) == 20,
                   "native/input tensor inventory incomplete")
    worker.require({str(path.relative_to(output_dir))
                    for path in output_dir.rglob("*.npy")} == seen,
                   "unreferenced or missing binary sidecar")
    failures: list[str] = []
    for plan_row, item in zip(rows, records[1:-1], strict=True):
        name = worker.call_name(plan_row)
        _check_hooks(item, plan_row, native)
        arm, case, site = (plan_row["arm"], plan_row["case_id"],
                           plan_row["site"])
        reference = None
        if arm == "read_only_donor_capture":
            reference = outputs[("none", case, "no_hook")]
        elif arm in ("read_only", "self"):
            reference = outputs[(site, case, "RR")]
        elif arm == "RR" and site == "pre_block0":
            reference = outputs[("log_mel", case, "RR")]
        elif arm == "DD":
            reference = outputs[("none", plan_row["donor_id"], "no_hook")]
        if reference is not None:
            actual = worker.first_parity(item, reference, output_dir)
            recorded = next(iter(item["parity"].values()))
            worker.require(actual == recorded, "worker first-parity flags changed")
            if not (actual["tokens_match"] and actual["language_match"]
                    and actual["within_1e-5"]):
                failures.append(f"first_or_token_parity:{name}")
        if arm == "DD":
            target = outputs[("none", plan_row["donor_id"], "no_hook")]
            actual = worker.full_raw_parity(item, target, output_dir)
            worker.require(actual == item["parity"]["donor_full_raw"],
                           "worker full raw parity flags changed")
            if not actual["within_1e-5"]:
                failures.append(f"full_raw_parity:{name}")
    return {"schema": "whisper_padding_saved_audit_v1",
            "status": "saved_evidence_complete" if not failures and not censored
                      else "censored_or_parity_failed",
            "forwards": len(rows), "native_tensors": len(native),
            "unique_input_feature_tensors": len(feature_refs),
            "binary_sidecars": len(seen),
            "parity_failures": failures, "length_censored_calls": censored,
            "sampled_peak_tree_rss_bytes": parent_result["sampled_peak_tree_rss_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker.require(args.attempt.resolve() == worker.ATTEMPT.resolve()
                   and args.output.resolve() == AUDIT_PATH.resolve()
                   and not args.output.exists(), "fixed exclusive audit path")
    result = audit_attempt(args.attempt, args.admission, args.go)
    worker.write_once(args.output, result)
    print(worker.canonical(result))


if __name__ == "__main__":
    main()
