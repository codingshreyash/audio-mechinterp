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
from . import whisper_time_translation_gate_v1 as gate
from . import whisper_time_translation_parent_v1 as parent
from . import whisper_time_translation_plan_v1 as plan_source
from . import whisper_time_translation_worker_v1 as worker


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
    shared = ("stage", "attempt", "parent_pid", "start_utc", "start_monotonic",
              "deadline_monotonic", "nonce_sha256", "admission_sha256",
              "go_sha256", "source_identity")
    need(start.get("schema") == result.get("schema") == "whisper_time_translation_parent_v1"
         and start.get("status") == "started"
         and result.get("status") == "completed_pending_saved_audit"
         and all(start.get(key) == result.get(key) for key in shared)
         and result["stage"] == stage and result["attempt"] == str(attempt)
         and result["source_identity"] == receipt["source_identity"]
         and result["admission_sha256"] == gate.sha(admission)
         and result["go_sha256"] == gate.sha(go), "parent/GO binding invalid")
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


def summarize(stage: str, calls: list[dict], evidence: list[dict],
              baselines: dict, eligibility: dict, mel_shift: dict) -> dict:
    keyed = {(row["plan"]["layout"], row["plan"]["site"],
              row["plan"]["region"], row["plan"]["location"],
              row["plan"]["kind"]): row for row in evidence
             if row["plan"]["type"] == "arm"}
    def qualified(layout: str, site: str, region: str, location: str) -> bool:
        key = (layout, site, region, location)
        try:
            direct = keyed[(*key, "target_transfer")]
            sham = keyed[(*key, "identity_sham")]
            random = keyed[(*key, "source_random")]
            shifted = keyed[(*key, "norm_matched_shifted_delta")]
            full = keyed[(layout, site, "all", "fixed", "full_donor")]
        except KeyError:
            return False
        donor = baselines[(layout, "zero")]
        recipient = baselines[(layout, "noise")]
        return (all((_same_complete(direct["checked"], donor),
                     _same_complete(sham["checked"], recipient),
                     np.array_equal(sham["checked"]["raw"], recipient["raw"]),
                     _same_complete(random["checked"], recipient),
                     _same_complete(shifted["checked"], recipient),
                     _same_complete(full["checked"], donor),
                     np.array_equal(full["checked"]["raw"], donor["raw"])))
                and direct["patch"]["inside_delta_l2"] > 0)
    original = {f"{site}:{region}": qualified("original", site, region, "fixed")
                for site in plan_source.SITES for region in plan_source.ORIGINAL_WINDOWS} if stage == "original" else {}
    cells = []
    if stage != "original":
        for region, site in plan_source.TRANSLATED_WINDOWS:
            align = [("prepend", "translated", "fixed"),
                     ("append", "fixed", "translated")]
            both = all(eligibility[layout]["eligible"] for layout, _, _ in align)
            details = []
            for layout, correct, opposite in align:
                direct_wrong = keyed.get((layout, site, region, opposite, "target_transfer"))
                relocation = keyed.get((layout, site, region, opposite, "relocated_correct_delta"))
                recipient = baselines[(layout, "noise")]
                details.append({"layout": layout,
                    "correct_qualified": qualified(layout, site, region, correct),
                    "opposite_failed_transfer": bool(direct_wrong) and
                        not _same_complete(direct_wrong["checked"], baselines[(layout, "zero")]),
                    "relocation_preserved_recipient": bool(relocation) and
                        _same_complete(relocation["checked"], recipient)})
            strict = (both and mel_shift["pass_le_1e_5"]
                      and all(all(item[key] for key in
                                  ("correct_qualified", "opposite_failed_transfer",
                                   "relocation_preserved_recipient")) for item in details))
            cells.append({"site": site, "region": region, "strict_translated_crossover": strict,
                          "layouts": details,
                          "requires_original_viability": f"{site}:{region}"})
    return {"original_qualified": original, "translated_cells": cells,
            "claim_status": "descriptive_only_pending_original_stage_join"
            if stage != "original" else "original_viability_only",
            "note": "A later cross-attempt audit must join original viability; no isolated tracking or security claim follows from one attempt."}


def _join_verified_reports(original: dict, offset_8000: dict, offset_16000: dict) -> dict:
    """Apply the decision rule only to outputs freshly made by audit()."""
    reports = (original, offset_8000, offset_16000)
    need(all(item.get("schema") == "whisper_time_translation_saved_audit_v1"
             and item.get("status") == "PASS" for item in reports)
         and [item["stage"] for item in reports]
         == ["original", "offset_8000", "offset_16000"]
         and len({item["plan_sha256"] for item in reports}) == 1
         and all(item["source_identity"] == original["source_identity"]
                 for item in reports), "three matched audited stages required")
    viability = original["summary"]["original_qualified"]
    union = []
    isolated = []
    for offset in (offset_8000, offset_16000):
        cells = offset["summary"]["translated_cells"]
        need(len(cells) == 3 and {(c["site"], c["region"]) for c in cells}
             == {(site, region) for region, site in plan_source.TRANSLATED_WINDOWS},
             "translated cell census invalid")
        for cell in cells:
            item = {"stage": offset["stage"], "site": cell["site"],
                    "region": cell["region"],
                    "original_viable": viability.get(f'{cell["site"]}:{cell["region"]}') is True,
                    "translated_strict": cell["strict_translated_crossover"]}
            item["joint_qualified"] = item["original_viable"] and item["translated_strict"]
            (union if cell["region"] == "union" else isolated).append(item)
    need(len(union) == 4 and len(isolated) == 2,
         "cross-attempt primary cell census invalid")
    return {"schema": "whisper_time_translation_joint_decision_v1",
            "status": "THREE_AUDITS_JOINED",
            "union_combined_boundary_cells": union,
            "union_qualified_count_out_of_4": sum(item["joint_qualified"] for item in union),
            "isolated_source_tail_cells": isolated,
            "isolated_source_tail_qualified_count_out_of_2":
                sum(item["joint_qualified"] for item in isolated),
            "original_isolated_tail_end_viability":
                {site: viability[f"{site}:tail_end"] for site in plan_source.SITES},
            "interpretation_limit": "Engineered non-speech pair only; qualified timing-linked behavior does not uniquely identify a circuit or establish a security problem."}


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
            need(set(refs) == set(plan_source.original.SITES),
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
    need(stage != "original" or eligibility["original"]["eligible"],
         "original replay lost contrast")
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
    summary = summarize(stage, calls, evidence, baselines, eligibility, mel_shift)
    return {"schema": "whisper_time_translation_saved_audit_v1", "status": "PASS",
            "stage": stage, "attempt": str(attempt), "calls": len(calls),
            "plan_sha256": gate.sha(plan_source.PLAN),
            "source_identity": receipt["source_identity"],
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
