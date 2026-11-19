"""Read-only, no-weight audit of a gated Whisper time-translation attempt."""
from __future__ import annotations

import argparse
import json
import math
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as power_host
from . import natural_tail_saved_audit_v4 as v4_audit
from . import whisper_tail_causal_host_v1 as host
from . import whisper_tail_causal_saved_audit_v3 as prior_audit
from . import whisper_tail_end_isolation_gate_v2 as gate
from . import whisper_tail_end_isolation_parent_v2 as parent
from . import whisper_tail_end_isolation_plan_v2 as plan_source
from . import whisper_tail_end_isolation_worker_v2 as worker


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _jsonl(path: Path, cap: int) -> list[dict]:
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap,
         "missing or oversized evidence JSONL")
    data = path.read_bytes()
    need(data.endswith(b"\n"), "incomplete evidence JSONL")
    return [json.loads(line) for line in data.splitlines()]


def audit_parent(stage: str, receipt: dict, attempt: Path,
                 admission: Path, go: Path) -> dict:
    start = json.loads((attempt / "launch_start.json").read_text())
    result = json.loads((attempt / "parent_result.json").read_text())
    shared = ("stage", "attempt", "parent_pid", "boot_time_unix_us",
              "start_utc", "start_monotonic",
              "deadline_monotonic", "nonce_sha256", "admission_sha256",
              "go_sha256", "source_identity")
    need(start.get("schema") == result.get("schema") == "whisper_tail_end_isolation_parent_v2"
         and start.get("status") == "started"
         and result.get("status") == "completed_pending_saved_audit"
         and all(start.get(key) == result.get(key) for key in shared)
         and result["stage"] == stage and result["attempt"] == str(attempt)
         and result["source_identity"] == receipt["source_identity"]
         and result["admission_sha256"] == gate.sha(admission)
         and result["go_sha256"] == gate.sha(go)
         and type(result["boot_time_unix_us"]) is int
         and result["boot_time_unix_us"] > 0, "parent/GO binding invalid")
    elapsed = result["whole_elapsed_seconds"]
    need(0 < elapsed < gate.WALL_SECONDS
         and result["deadline_monotonic"] - result["start_monotonic"] == gate.WALL_SECONDS
         and result["evidence_bytes_before_result"] < gate.EVIDENCE_BYTES
         and parent.evidence_bytes(attempt) < gate.EVIDENCE_BYTES,
         "whole-run deadline or evidence cap invalid")
    monitor = result["monitor"]
    need(monitor["worker_returncode"] == 0
         and monitor["worker_stop_reason"] == "normal_exit"
         and monitor["worker_monitor_error_type"] is None
         and monitor["parent_heartbeat_count"] > 0
         and 0 < monitor["worker_peak_tree_rss_bytes"] < gate.RSS_BYTES
         and monitor["worker_rss_samples"] > 0, "worker monitor invalid")
    rss_file = attempt / "worker_rss_samples.jsonl"
    need(gate.sha(rss_file) == monitor["worker_rss_sample_log_sha256"],
         "RSS evidence hash invalid")
    rss = _jsonl(rss_file, 1_000_000)
    need(len(rss) == monitor["worker_rss_samples"]
         and [row["sample_index"] for row in rss] == list(range(1, len(rss) + 1))
         and max(row["tree_rss_bytes"] for row in rss)
         == monitor["worker_peak_tree_rss_bytes"]
         and all(0 < row["tree_rss_bytes"] < gate.RSS_BYTES
                 and row["evidence_bytes"] < gate.EVIDENCE_BYTES
                 and result["start_monotonic"] <= row["monotonic_seconds"]
                 <= result["deadline_monotonic"] for row in rss),
         "RSS samples invalid")
    host_file = attempt / "host_samples.jsonl"
    need(gate.sha(host_file) == result["host_samples_sha256"],
         "host sample hash invalid")
    samples = _jsonl(host_file, 1_000_000)
    need(len(samples) == result["host_samples_count"] and len(samples) >= 2,
         "host sample count invalid")
    host.validate_cadence(samples, start=result["start_monotonic"],
                          end=result["host_monitor_end_monotonic"])
    for row in samples:
        host.validate_saved_text(row)
    first = datetime.fromisoformat(result["start_utc"])
    worker_exit = datetime.fromisoformat(result["worker_exit_utc"])
    last = datetime.fromisoformat(result["end_utc"])
    need(first <= worker_exit <= last
         and abs((last - first).total_seconds() - elapsed) <= 0.25
         and datetime.fromisoformat(samples[-1]["collected_utc"]) >= worker_exit,
         "whole-run clock invalid")
    power_file = attempt / "post_power_log.txt"
    need(gate.sha(power_file) == result["post_power_log_sha256"],
         "power log hash invalid")
    power_host.validate_log(power_file.read_text(), first,
                            datetime.fromisoformat(samples[-1]["collected_utc"]),
                            datetime.fromisoformat(result["post_power_log_collected_utc"]))
    need(parent.complete_worker_output(attempt, stage)
         and result["progress"] == parent.progress_census(attempt, stage),
         "parent worker census invalid")
    return result


def _outcome(attempt: Path, row: dict, active: int, config: dict) -> dict:
    outcome = row["output"]
    refs = outcome["sidecars"]
    need(set(refs) == {"mel", "mask", "first_raw_logits",
                       "first_processed_scores", "first_decoder_state"},
         "Whisper sidecar inventory invalid")
    mel = v4_audit.load_sidecar(attempt, refs["mel"], (80, 3000), "float32")
    mask = v4_audit.load_sidecar(attempt, refs["mask"], (3000,), "int64")
    raw = v4_audit.load_sidecar(attempt, refs["first_raw_logits"], (51865,), "float32")
    scores = v4_audit.load_sidecar(attempt, refs["first_processed_scores"], (51865,), "float32")
    v4_audit.load_sidecar(attempt, refs["first_decoder_state"], (1, 4, 384), "float32")
    need(np.isfinite(mel).all() and np.isfinite(raw).all()
         and np.array_equal(mask[:active], np.ones(active, dtype=np.int64))
         and np.array_equal(mask[active:], np.zeros(3000-active, dtype=np.int64))
         and outcome["geometry"]["active_mel_frames"] == active
         and outcome["geometry"]["mask_sha256"]
         == worker.hashlib.sha256(bytes(mask.astype(np.uint8))).hexdigest(),
         "Whisper input geometry invalid")
    output = outcome["output"]
    ids, prefix, generated = (output[key] for key in
                              ("sequence_ids", "prefix_ids", "generated_ids"))
    score_rows = output["score_rows"]
    need(ids == prefix + generated and 1 <= len(generated) <= 128
         and len(generated) == output["actual_generation_steps"] == len(score_rows)
         and (generated[-1] == 50257) == output["eos_seen"]
         and (output["termination"] == "eos") == output["eos_seen"]
         and (output["termination"] == "length_censored") == output["censored"]
         and (output["eos_seen"] or len(generated) == 128)
         and 50257 not in generated[:-1], "complete EOS path invalid")
    need(all(item["token_id"] == generated[i]
                 and math.isfinite(item["raw_selected_logit"])
                 and math.isfinite(item["selected_logit"])
                 and isinstance(item["raw_logits_sha256"], str)
                 and len(item["raw_logits_sha256"]) == 64
                 and isinstance(item["processed_logits_sha256"], str)
                 and len(item["processed_logits_sha256"]) == 64
             for i, item in enumerate(score_rows)), "score rows invalid")
    first, token = score_rows[0], generated[0]
    need(first["raw_logits_sha256"] == worker.hashlib.sha256(raw.tobytes()).hexdigest()
         and first["processed_logits_sha256"]
         == worker.hashlib.sha256(scores.tobytes()).hexdigest()
         and first["raw_top_id"] == int(raw.argmax())
         and first["top_id"] == int(scores.argmax())
         and math.isclose(first["raw_selected_logit"], float(raw[token]), abs_tol=1e-5)
         and math.isclose(first["selected_logit"], float(scores[token]), abs_tol=1e-5),
         "first-step evidence invalid")
    need(outcome["capture_counts"].get("generate")
         == {key: 1 for key in ("mel", "post_conv_pos", "post_block0", "final_encoder")}
         and outcome["capture_counts"].get("detect")
         == {key: 1 for key in ("post_conv_pos", "post_block0", "final_encoder")}
         and outcome["generation_kwargs"].get("task") == "transcribe"
         and outcome["generation_kwargs"].get("language") is None
         and outcome["generation_kwargs"].get("past_key_values") is None
         and outcome["generation_kwargs"].get("encoder_outputs") is None,
         "Whisper generation/detection census invalid")
    detection = output["language_detection"]
    candidates = detection.get("language_logits", {}) if isinstance(detection, dict) else {}
    lang_ids = config["lang_to_id"]
    chosen = min(candidates, key=lambda key: (-candidates[key], lang_ids[key])) if candidates else None
    need(set(candidates) == set(lang_ids)
         and detection["candidate_count"] == len(candidates)
         and detection["chosen_language_ids"] == [lang_ids[chosen]]
         and prefix == [config["decoder_start_token_id"], lang_ids[chosen],
                        config["task_to_id"]["transcribe"],
                        config["no_timestamps_token_id"]],
         "language prefix invalid")
    return {"output": output, "raw": raw, "mel": mel, "mask": mask}


def _same_complete(left: dict, right: dict) -> bool:
    return worker._complete(left["output"]) and left["output"]["sequence_ids"] == right["output"]["sequence_ids"]


def verify_baseline_source(item: dict, checked: dict, processor: object,
                           source_dir: Path) -> None:
    """Independently bind a full saved baseline tensor to the frozen WAV."""
    descriptor = item["input"]
    name = descriptor["wav"]
    need(type(name) is str and name == Path(name).name, "frozen WAV name invalid")
    path = source_dir / name
    need(path.is_file() and not path.is_symlink()
         and gate.sha(path) == descriptor["wav_sha256"],
         "frozen baseline WAV hash changed")
    with wave.open(str(path), "rb") as audio:
        frames = descriptor["frames"]
        need((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
              audio.getnframes()) == (1, 2, 16_000, frames),
             "frozen baseline WAV geometry changed")
        pcm = audio.readframes(frames)
    need(len(pcm) == frames * 2, "frozen baseline PCM truncated")
    if "pcm_sha256" in descriptor:
        need(worker.hashlib.sha256(pcm).hexdigest() == descriptor["pcm_sha256"],
             "frozen baseline PCM hash changed")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
    processed = processor(samples, sampling_rate=16_000, return_tensors="np",
                          return_attention_mask=True)
    mel = np.asarray(processed.input_features)[0]
    mask = np.asarray(processed.attention_mask)[0]
    need(mel.shape == (80, 3000) and mask.shape == (3000,)
         and np.array_equal(mel, checked["mel"])
         and np.array_equal(mask, checked["mask"]),
         "saved baseline full mel/mask differs from frozen WAV")


def summarize(stage: str, evidence: list[dict], baselines: dict, states: dict,
              eligibility: dict, mel_shift: dict) -> dict:
    """Score complete paths; report native deltas even when a layout is skipped."""
    keyed = {(row["plan"]["layout"], row["plan"]["location"],
              row["plan"]["kind"]): row for row in evidence}

    def native_norm(layout: str, location: str) -> float:
        offset = 0 if stage == "original" else int(stage.removeprefix("offset_"))
        rows = plan_source.rows(offset, location)
        donor = states[(layout, "zero")][plan_source.SITE]
        recipient = states[(layout, "noise")][plan_source.SITE]
        return float(np.linalg.norm((donor[rows] - recipient[rows]).astype(np.float64)))

    def qualified(layout: str, location: str) -> dict:
        try:
            direct, sham, unrelated, displaced, full = (
                keyed[(layout, location, "target_transfer")],
                keyed[(layout, location, "identity_sham")],
                keyed[(layout, location, "source_random")],
                keyed[(layout, location, "norm_matched_shifted_delta")],
                keyed[(layout, "fixed", "full_donor")])
        except KeyError:
            return {"qualified": False, "reason": "ineligible_or_missing_arm"}
        donor, recipient = baselines[(layout, "zero")], baselines[(layout, "noise")]
        controls = {
            "direct_full_donor_path": _same_complete(direct["checked"], donor),
            "identity_full_recipient_path": _same_complete(sham["checked"], recipient),
            "identity_raw_recipient": np.array_equal(sham["checked"]["raw"], recipient["raw"]),
            "unrelated_full_recipient_path": _same_complete(unrelated["checked"], recipient),
            "displaced_full_recipient_path": _same_complete(displaced["checked"], recipient),
            "full_donor_path": _same_complete(full["checked"], donor),
            "full_donor_raw_vector": np.array_equal(full["checked"]["raw"], donor["raw"]),
            "native_delta_nonzero": native_norm(layout, location)
                                     > plan_source.NONZERO_DELTA_L2,
        }
        return {"qualified": all(controls.values()), "controls": controls,
                "native_delta_l2": native_norm(layout, location)}

    if stage == "original":
        original = qualified("original", "fixed")
        original["qualified"] = (original["qualified"]
                                 and eligibility["original"]["eligible"])
        original["clean_contrast_eligible"] = eligibility["original"]["eligible"]
        return {"original_qualified": original,
                "clean_eligible": eligibility["original"]["eligible"],
                "claim_status": "original_viability_only"}

    layouts = []
    for layout, aligned, competing in (("prepend", "translated", "fixed"),
                                       ("append", "fixed", "translated")):
        donor, recipient = baselines[(layout, "zero")], baselines[(layout, "noise")]
        opposite = keyed.get((layout, competing, "target_transfer"))
        relocated = keyed.get((layout, competing, "relocated_correct_delta"))
        opposite_failed_donor = bool(opposite) and not _same_complete(opposite["checked"], donor)
        opposite_preserved_recipient = bool(opposite) and _same_complete(opposite["checked"], recipient)
        relocation_preserved = bool(relocated) and _same_complete(relocated["checked"], recipient)
        relocation_source_norm = relocated["patch"].get("target_delta_l2") if relocated else None
        relocation_applied_norm = relocated["patch"].get("inside_delta_l2") if relocated else None
        location_norms = {name: native_norm(layout, name)
                          for name in ("fixed", "translated")}
        aligned_qualification = qualified(layout, aligned)
        layouts.append({"layout": layout, "eligible": eligibility[layout]["eligible"],
                        "aligned_location": aligned, "competing_location": competing,
                        "native_delta_l2": location_norms,
                        "aligned": aligned_qualification,
                        "competing_direct_failed_donor": opposite_failed_donor,
                        "competing_direct_preserved_recipient": opposite_preserved_recipient,
                        "competing_direct_third_or_censored": bool(opposite) and
                            opposite_failed_donor and not opposite_preserved_recipient,
                        "relocated_aligned_delta_preserved_recipient": relocation_preserved,
                        "relocation_source_l2": relocation_source_norm,
                        "relocation_applied_l2": relocation_applied_norm,
                        "append_new_end_native_below_threshold": layout == "append" and
                            location_norms["translated"] <= plan_source.NONZERO_DELTA_L2})
    strict = (all(item["eligible"] and item["aligned"]["qualified"]
                  and item["competing_direct_failed_donor"]
                  and item["relocated_aligned_delta_preserved_recipient"]
                  for item in layouts)
              and mel_shift["pass_le_1e_5"])
    return {"strict_translated_crossover": strict,
            "secondary_opposite_direct_preserved_recipient": strict and
                all(item["competing_direct_preserved_recipient"] for item in layouts),
            "layouts": layouts, "mel_shift_qualified": mel_shift["pass_le_1e_5"],
            "claim_status": "descriptive_only_pending_original_stage_join",
            "note": "The append translated rows are a new waveform end with no native donor contrast when their measured norm is below threshold."}


def _join_verified_reports(original: dict, offset_8000: dict, offset_16000: dict) -> dict:
    """Apply the V2 decision only to outputs freshly made by audit()."""
    reports = (original, offset_8000, offset_16000)
    need(all(item.get("schema") == "whisper_tail_end_isolation_saved_audit_v2"
             and item.get("status") == "PASS" for item in reports)
         and [item["stage"] for item in reports]
         == ["original", "offset_8000", "offset_16000"]
         and len({item["plan_sha256"] for item in reports}) == 1
         and all(item["source_identity"] == original["source_identity"]
                 for item in reports), "three matched V2 audited stages required")
    viability = original["summary"]["original_qualified"]["qualified"]
    for earlier, later in ((original, offset_8000), (offset_8000, offset_16000)):
        need(datetime.fromisoformat(earlier["parent_end_utc"])
             <= datetime.fromisoformat(later["parent_start_utc"]),
             "V2 stage UTC order invalid")
        if earlier["boot_time_unix_us"] == later["boot_time_unix_us"]:
            need(earlier["parent_start_monotonic"]
                 + earlier["whole_elapsed_seconds"]
                 <= later["parent_start_monotonic"],
                 "V2 same-boot monotonic stage order invalid")
    cells = []
    for report in (offset_8000, offset_16000):
        candidate = report["summary"]["strict_translated_crossover"]
        cells.append({"stage": report["stage"], "original_qualified": viability,
                      "translated_strict": candidate,
                      "strict_isolated_old_tail_end_crossover": viability and candidate,
                      "secondary_opposite_direct_preserved_recipient":
                          viability and report["summary"]["secondary_opposite_direct_preserved_recipient"],
                      "append_new_end_native_below_threshold":
                          next(item["append_new_end_native_below_threshold"]
                               for item in report["summary"]["layouts"]
                               if item["layout"] == "append")})
    count = sum(cell["strict_isolated_old_tail_end_crossover"] for cell in cells)
    return {"schema": "whisper_tail_end_isolation_joint_decision_v2",
            "status": "THREE_RAW_AUDITS_JOINED",
            "strict_isolated_old_tail_end_offsets": cells,
            "strict_count_out_of_2": count,
            "confirmatory_within_pair_boundary_timing": count == 2,
            "interpretation_limit": "One engineered beep/noise pair at a pre-attention coordinate; this cannot isolate local acoustics from positional context or establish a security problem."}


def audit_joint() -> dict:
    """Reaudit all three fixed attempts before computing any joint claim."""
    verified = [audit(stage, gate.ADMISSIONS[stage], gate.GOS[stage])
                for stage in gate.STAGES]
    return _join_verified_reports(*verified)


def audit(stage: str, admission: Path, go: Path) -> dict:
    receipt = gate.validate_go(stage, admission, go)
    attempt = gate.ATTEMPTS[stage]
    need(attempt.is_dir() and not attempt.is_symlink(), "fixed attempt missing")
    result = audit_parent(stage, receipt, attempt, admission, go)
    frozen = json.loads(plan_source.PLAN.read_text())
    need(frozen == plan_source.build(), "frozen plan changed")
    rows = _jsonl(attempt / "records.jsonl", parent.MAX_RECORD_BYTES)
    need(rows[-1]["type"] == "complete" and rows[-1]["stage"] == stage,
         "missing worker terminal")
    baselines: dict = {}
    states: dict = {}
    evidence = []
    config = json.loads((Path(receipt["v4_admission"]["assets"]["snapshot"])
                         / "generation_config.json").read_text())
    from transformers import WhisperProcessor
    assets = receipt["v4_admission"]["assets"]
    snapshot = Path(assets["snapshot"])
    need(snapshot.is_dir() and not snapshot.is_symlink()
         and snapshot.name == assets["revision"]
         and all(gate.sha(snapshot / name) == digest for name, digest in
                 assets["files_sha256"].items() if name != "model.safetensors"),
         "pinned processor assets changed")
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    baseline_plan = worker.baselines_for(stage, frozen)
    for index, row in enumerate(rows[:-1], 1):
        need(row["type"] == "call" and row["index"] == index, "call order invalid")
        call = row["plan"]
        if index <= len(baseline_plan):
            need(call == baseline_plan[index - 1] and row["execution_arm"] is None
                 and row["patch"] == {"hook_calls": 0}, "baseline order/hook invalid")
        else:
            need(call["type"] == "arm" and row["execution_arm"] == worker.execution_arm(call),
                 "arm execution mapping invalid")
        active = 1000 + (int(stage.removeprefix("offset_")) // 160 if stage != "original" else 0)
        checked = _outcome(attempt, row, active, config)
        layout = call["layout"]
        if call["type"] == "baseline":
            kind = call["variant"]
            verify_baseline_source(call, checked, processor,
                                   worker.v4.RENDER if stage == "original"
                                   else worker.source.OUTPUT)
            if stage == "original":
                worker._original_parity(call, row["output"], attempt / "sidecars")
            refs = row["state_refs"]
            need(set(refs) == set(plan_source.prior.original.SITES),
                 "baseline state inventory invalid")
            states[(layout, kind)] = {site: prior_audit.state(attempt, ref)
                                      for site, ref in refs.items()}
            baselines[(layout, kind)] = checked
        else:
            need(row["state_refs"] is None, "arm unexpectedly saved baseline states")
            recipient = baselines[(layout, "noise")]
            donor = baselines[(layout, "zero")]
            need(np.array_equal(checked["mel"], recipient["mel"])
                 and np.array_equal(checked["mask"], recipient["mask"])
                 and checked["output"]["prefix_ids"] == recipient["output"]["prefix_ids"]
                 and checked["output"]["language_detection"]
                 == recipient["output"]["language_detection"],
                 "patched call input/detection changed")
            site = call["site"]
            native = prior_audit.state(attempt, row["patch"]["native_ref"])
            edited = prior_audit.state(attempt, row["patch"]["edited_ref"])
            prior_audit.verify_patch(native, edited, states[(layout, "noise")][site],
                                     states[(layout, "zero")][site], row["execution_arm"],
                                     row["patch"])
            row["checked"] = checked
            evidence.append(row)
    eligibility = {layout: worker.pair_eligibility(
        {"output": baselines[(layout, "zero")]["output"],
         "sidecars": rows[next(i for i, row in enumerate(rows[:len(baseline_plan)])
                                if row["plan"]["layout"] == layout and row["plan"]["variant"] == "zero")]["output"]["sidecars"]},
        {"output": baselines[(layout, "noise")]["output"],
         "sidecars": rows[next(i for i, row in enumerate(rows[:len(baseline_plan)])
                                if row["plan"]["layout"] == layout and row["plan"]["variant"] == "noise")]["output"]["sidecars"]},
        attempt / "sidecars") for layout in (("original",) if stage == "original" else ("prepend", "append"))}
    mel_shift = worker.mel_shift_receipt(stage,
        {(row["plan"]["layout"], row["plan"]["variant"]): row["output"]
         for row in rows[:len(baseline_plan)]}, attempt / "sidecars", frozen)
    calls, skipped = worker.planned_calls(stage, frozen,
        {layout: item["eligible"] for layout, item in eligibility.items()})
    terminal = rows[-1]
    need(terminal == {"type": "complete", "stage": stage,
                     "executed_calls": len(calls),
                     "planned_max_calls": frozen["stages"][stage]["max_calls"],
                     "skipped_arms": skipped, "eligibility": eligibility,
                     "mel_shift": mel_shift}
         and [row["plan"] for row in rows[:-1]] == calls,
         "frozen conditional schedule or terminal invalid")
    summary = summarize(stage, evidence, baselines, states, eligibility, mel_shift)
    return {"schema": "whisper_tail_end_isolation_saved_audit_v2", "status": "PASS",
            "stage": stage, "attempt": str(attempt), "calls": len(calls),
            "plan_sha256": gate.sha(plan_source.PLAN),
            "source_identity": receipt["source_identity"],
            "parent_start_utc": result["start_utc"],
            "parent_end_utc": result["end_utc"],
            "parent_start_monotonic": result["start_monotonic"],
            "boot_time_unix_us": result["boot_time_unix_us"],
            "planned_max_calls": frozen["stages"][stage]["max_calls"],
            "skipped_arms": skipped, "eligibility": eligibility,
            "mel_shift": mel_shift, "summary": summary,
            "whole_elapsed_seconds": result["whole_elapsed_seconds"],
            "peak_tree_rss_bytes": result["monitor"]["worker_peak_tree_rss_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=gate.STAGES)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--go", type=Path)
    parser.add_argument("--join", action="store_true")
    args = parser.parse_args()
    if args.join:
        need(args.stage is None and args.admission is None and args.go is None,
             "join mode takes no stage or report paths")
        result = audit_joint()
    else:
        need(args.stage is not None and args.admission is not None and args.go is not None,
             "stage, admission and GO required for one-attempt audit")
        result = audit(args.stage, args.admission, args.go)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
