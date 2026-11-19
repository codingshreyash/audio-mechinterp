"""Checkpoint-aware V3 saved audit for immutable English Whisper V2 records."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from . import descriptive_worker_v2 as worker
from . import descriptive_worker_v1 as base
from . import descriptive_parent_v2 as timing
from . import descriptive_audit_v2 as timing_audit

HERE = Path(__file__).resolve().parent
MODEL = "whisper_english"
ATTEMPT = worker.MODEL_PATHS[MODEL]["attempt"]
V2_RECEIPT = worker.MODEL_PATHS[MODEL]["receipt"]
V2_GO = worker.MODEL_PATHS[MODEL]["review"]
V3_RECEIPT = HERE / "runs/descriptive_english_saved_audit_preflight_v3/preflight.json"
V3_GO = HERE / "INDEPENDENT_DESCRIPTIVE_ENGLISH_SAVED_AUDIT_GO_V3.json"
V3_OUTPUT = HERE / "runs/descriptive_english_saved_audit_v3"
ATTEMPT_SHA256 = {
    "parent_result.json": "89dc84e4d8a3fdb8733c0c8f17a137beabd5414cd680cd2004039740cd61339a",
    "records.jsonl": "4fd1709d85ebe13ed6a46a7996491fbb227463df6592537435e81c4af91ee923",
}
V3_SOURCES = ("descriptive_english_saved_audit_v3.py",
              "test_descriptive_english_saved_audit_v3.py",
              "DESCRIPTIVE_ENGLISH_SAVED_AUDIT_REPAIR_PROSPECTIVE_V3.md")


@lru_cache(maxsize=2)
def whisper_contract(model: str) -> dict:
    """Read revision-pinned metadata/tokenizer; no model weights are deserialized."""
    worker.require(model in ("whisper_multilingual", "whisper_english"),
                   "unknown Whisper checkpoint")
    if model == "whisper_multilingual":
        assets = base.verify_local_assets(model)
    else:
        assets = worker.verify_local_assets(model)
    snapshot = Path(assets["snapshot"])
    config = json.loads((snapshot / "config.json").read_text())
    generation = json.loads((snapshot / "generation_config.json").read_text())
    processor_config = json.loads((snapshot / "preprocessor_config.json").read_text())
    from transformers import WhisperProcessor
    tokenizer = WhisperProcessor.from_pretrained(snapshot, local_files_only=True).tokenizer
    eos = generation["eos_token_id"]
    pad = generation["pad_token_id"]
    worker.require(config["eos_token_id"] == eos == pad == config["pad_token_id"]
                   == tokenizer.eos_token_id == tokenizer.pad_token_id,
                   "pinned Whisper tokenizer/config EOS or pad mismatch")
    worker.require(config["decoder_start_token_id"] == generation["decoder_start_token_id"]
                   and config["vocab_size"] == len(tokenizer)
                   and processor_config["hop_length"] == 160
                   and processor_config["feature_size"] == config["num_mel_bins"]
                   and processor_config["n_samples"] == 2 * config["max_source_positions"]
                   * processor_config["hop_length"],
                   "pinned Whisper token or feature geometry mismatch")
    return {"eos_token_id": eos, "pad_token_id": pad,
            "decoder_start_token_id": generation["decoder_start_token_id"],
            "no_timestamps_token_id": generation["no_timestamps_token_id"],
            "vocab_size": config["vocab_size"], "d_model": config["d_model"],
            "num_mel_bins": config["num_mel_bins"],
            "encoder_frames": config["max_source_positions"],
            "mel_frames": 2 * config["max_source_positions"],
            "hop_length": processor_config["hop_length"],
            "lang_to_id": generation.get("lang_to_id"),
            "task_to_id": generation.get("task_to_id"),
            "forced_decoder_ids": generation.get("forced_decoder_ids"),
            "revision": assets["revision"],
            "config_sha256": assets["files_sha256"]["config.json"],
            "generation_config_sha256": assets["files_sha256"]["generation_config.json"]}


def source_identity() -> dict[str, str]:
    names = (*V3_SOURCES, "descriptive_remaining_audit_v2.py",
             "descriptive_worker_v2.py", "descriptive_remaining_parent_v2.py",
             "descriptive_parent_v2.py", "descriptive_audit_v2.py")
    # The V2 GO binds these upstream files too. Repeat their identities in the
    # V3 receipt so the saved-audit review is self-contained.
    return {**worker.source_identity(),
            **{name: worker.sha_file(HERE / name) for name in names}}


def preflight_receipt() -> dict:
    worker.require_independent_go(MODEL, V2_RECEIPT, V2_GO)
    for name, digest in ATTEMPT_SHA256.items():
        path = ATTEMPT / name
        worker.require(path.is_file() and not path.is_symlink()
                       and worker.sha_file(path) == digest,
                       f"immutable English V2 attempt bytes changed: {name}")
    return {"schema": "nonspeech_english_saved_audit_preflight_v3",
            "status": "source_only_no_model_load_no_output_inspection",
            "model": MODEL, "attempt_path": str(ATTEMPT.resolve()),
            "attempt_files_sha256": ATTEMPT_SHA256,
            "source_identity": source_identity(),
            "v2_receipt_sha256": worker.sha_file(V2_RECEIPT),
            "v2_go_sha256": worker.sha_file(V2_GO),
            "whisper_contract": whisper_contract(MODEL),
            "multilingual_control_contract": whisper_contract("whisper_multilingual"),
            "review_path": str(V3_GO.resolve()),
            "output_path": str(V3_OUTPUT.resolve())}


def preflight_only() -> dict:
    worker.require(V3_RECEIPT.parent.is_dir() and not V3_RECEIPT.exists(),
                   "exclusive V3 audit preflight receipt path required")
    receipt = preflight_receipt()
    with V3_RECEIPT.open("x", encoding="utf-8") as stream:
        stream.write(worker.canonical(receipt) + "\n")
    return receipt


def require_v3_go() -> dict:
    worker.require(V3_RECEIPT.is_file() and V3_GO.is_file()
                   and not V3_RECEIPT.is_symlink() and not V3_GO.is_symlink(),
                   "V3 independent saved-audit GO or receipt missing")
    receipt = json.loads(V3_RECEIPT.read_text())
    worker.require(receipt == preflight_receipt(), "V3 saved-audit preflight differs")
    review = json.loads(V3_GO.read_text())
    worker.require(review.get("schema") == "nonspeech_english_saved_audit_review_v3"
                   and review.get("status") == "GO"
                   and review.get("preflight_sha256") == worker.sha_file(V3_RECEIPT)
                   and review.get("source_identity") == receipt["source_identity"]
                   and review.get("attempt_files_sha256") == ATTEMPT_SHA256,
                   "V3 independent GO does not bind immutable saved evidence")
    review_commit = worker._file_commit(V3_GO)
    for name in V3_SOURCES:
        source_commit = worker._file_commit(HERE / name)
        worker.require(source_commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", source_commit, review_commit),
            cwd=HERE.parents[2], check=False).returncode == 0,
            f"V3 review must follow frozen source: {name}")
    return review


def rows(path: Path) -> list[dict]:
    worker.require(path.is_file() and not path.is_symlink(), "records missing")
    raw = path.read_bytes()
    worker.require(raw.endswith(b"\n") and len(raw) <= worker.MAX_EVIDENCE_BYTES,
                   "records truncated or oversized")
    result = []
    for line in raw.decode("utf-8").splitlines():
        value = json.loads(line)
        worker.require(isinstance(value, dict), "record is not an object")
        result.append(value)
    return result


def finite_nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def ctc_length(frames: int) -> int:
    length = frames
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2),
                              (5, 2, 2, 2, 2, 2, 2), strict=True):
        length = (length - kernel) // stride + 1
    return length


def check_geometry(geometry: dict, model: str, frames: int) -> None:
    worker.require(isinstance(geometry, dict) and geometry.get("pcm_frames") == frames,
                   "saved input geometry missing or changed")
    if model.startswith("whisper"):
        mask = geometry.get("attention_mask")
        contract = whisper_contract(model)
        active = (frames + contract["hop_length"] - 1) // contract["hop_length"]
        mel_frames = contract["mel_frames"]
        worker.require(geometry.get("mel_frames") == mel_frames
                       and isinstance(mask, list) and len(mask) == mel_frames
                       and mask == [1] * active + [0] * (mel_frames - active)
                       and geometry.get("active_mel_frames") == active
                       and geometry.get("encoder_frames") == contract["encoder_frames"]
                       and geometry.get("nominal_active_encoder_rows") == (active + 1) // 2
                       and geometry.get("encoder_time_map") ==
                       "conv1(k3,s1,p1),conv2(k3,s2,p1); nominal row r uses mel centers 2r"
                       and geometry.get("mask_sha256") == worker.sha_bytes(bytes(mask)),
                       "Whisper active/padded mask or encoder map invalid")
    else:
        length = ctc_length(frames)
        worker.require(geometry == worker.wav_geometry(frames, length),
                       "Wav2Vec2 all-active/CTC geometry invalid")


def expected_stage_shape(stage: str, model: str, frames: int) -> tuple[list[int], int]:
    if model.startswith("whisper"):
        contract = whisper_contract(model)
        return ([1, contract["num_mel_bins"], contract["mel_frames"]], 2) if stage == "mel" else (
            [1, contract["encoder_frames"], contract["d_model"]], 1)
    first = (frames - 10) // 5 + 1
    final = ctc_length(frames)
    if stage == "input_values":
        return [1, frames], 1
    if stage in ("first_conv_raw", "first_conv_norm"):
        return [1, 512, first], 2
    if stage == "logits":
        return [1, final, 32], 1
    return [1, final, 768], 1


def check_trace(traces: dict, counts: dict, model: str, frames: int) -> None:
    phases = ("detect", "generate") if model == "whisper_multilingual" else (
        "generate",) if model == "whisper_english" else ("ctc",)
    worker.require(isinstance(traces, dict) and isinstance(counts, dict)
                   and set(traces) == set(phases) and set(counts) == set(phases),
                   "capture phase coverage changed")
    for phase, stages in traces.items():
        expected = set(worker.WHISPER_STAGES)
        if phase == "detect":
            expected.remove("mel")
        elif phase == "ctc":
            expected = set(worker.WAV_STAGES)
        worker.require(isinstance(stages, dict) and set(stages) == expected
                       and counts[phase] == {name: 1 for name in expected},
                       "capture stage/count coverage changed")
        for name, metric in stages.items():
            shape, curve = metric.get("shape"), metric.get("row_l2")
            expected_shape, axis = expected_stage_shape(name, model, frames)
            worker.require(metric.get("dtype") == "float32" and shape == expected_shape
                           and metric.get("time_axis") == axis
                           and isinstance(curve, list) and len(curve) == shape[axis]
                           and all(finite_nonnegative(x) for x in curve)
                           and isinstance(metric.get("sha256"), str)
                           and HEX64.fullmatch(metric["sha256"]) is not None,
                           f"trace metric invalid: {phase}/{name}")


def check_whisper(output: dict, model: str) -> None:
    contract = whisper_contract(model)
    multilingual = model == "whisper_multilingual"
    ids = output.get("sequence_ids")
    prefix = output.get("prefix_ids")
    generated = output.get("generated_ids")
    steps = output.get("actual_generation_steps")
    scores = output.get("score_rows")
    worker.require(type(steps) is int and 1 <= steps <= worker.MAX_NEW_TOKENS
                   and isinstance(ids, list) and isinstance(prefix, list)
                   and isinstance(generated, list) and len(generated) == steps
                   and ids == prefix + generated and isinstance(scores, list)
                   and len(scores) == steps and len(prefix) >= 1,
                   "Whisper complete sequence/step alignment invalid")
    worker.require([row.get("token_id") for row in scores] == generated,
                   "Whisper selected score IDs differ")
    worker.require(all(type(x) is int and x >= 0 for x in ids)
                   and all(type(row.get("selected_logit")) in (int, float)
                           and math.isfinite(row["selected_logit"])
                           and type(row.get("selected_logprob")) in (int, float)
                           and math.isfinite(row["selected_logprob"])
                           and isinstance(row.get("processed_logits_sha256"), str)
                           and HEX64.fullmatch(row["processed_logits_sha256"]) is not None
                           and isinstance(row.get("raw_logits_sha256"), str)
                           and HEX64.fullmatch(row["raw_logits_sha256"]) is not None
                           and type(row.get("raw_selected_logit")) in (int, float)
                           and math.isfinite(row["raw_selected_logit"])
                           and type(row.get("raw_top_id")) is int
                           and type(row.get("top_id")) is int for row in scores),
                   "Whisper IDs or scores invalid")
    first_state = output.get("first_decoder_step")
    raw = output.get("first_raw_logits")
    vocab_size = contract["vocab_size"]
    worker.require(isinstance(first_state, dict) and isinstance(raw, dict)
                   and first_state.get("site") ==
                   "decoder_hidden_states[0][-1] (final decoder layer, first generation step)"
                   and first_state.get("shape") == [1, len(prefix), contract["d_model"]]
                   and raw.get("shape") == [vocab_size],
                   "first decoder state or raw logit geometry invalid")
    hidden = np.asarray(first_state.get("values"), dtype=np.float32)
    raw_values = np.asarray(raw.get("values"), dtype=np.float32)
    worker.require(list(hidden.shape) == first_state["shape"]
                   and list(raw_values.shape) == raw["shape"]
                   and bool(np.isfinite(hidden).all()) and bool(np.isfinite(raw_values).all())
                   and worker.sha_bytes(hidden.tobytes()) == first_state.get("sha256")
                   and worker.sha_bytes(raw_values.tobytes()) == raw.get("sha256")
                   and raw["sha256"] == scores[0]["raw_logits_sha256"]
                   and float(raw_values[generated[0]]) == scores[0]["raw_selected_logit"]
                   and int(raw_values.argmax()) == scores[0]["raw_top_id"],
                   "first decoder state/raw logits hash or value invalid")
    eos_id = contract["eos_token_id"]
    eos_seen = eos_id in generated
    worker.require(output.get("eos_seen") is eos_seen
                   and output.get("censored") is (not eos_seen and steps == worker.MAX_NEW_TOKENS)
                   and output.get("termination") == (
                       "eos" if eos_seen else "length_censored" if steps == worker.MAX_NEW_TOKENS
                       else "incomplete_unknown"), "Whisper EOS/censor accounting invalid")
    if eos_seen:
        worker.require(generated[-1] == eos_id, "tokens after first generated EOS")
    worker.require(isinstance(output.get("text_with_special_tokens"), str),
                   "Whisper full text missing")
    if multilingual:
        detection = output.get("language_detection")
        language_ids = contract["lang_to_id"]
        expected_prefix_tail = [contract["task_to_id"]["transcribe"],
                                contract["no_timestamps_token_id"]]
        worker.require(isinstance(detection, dict)
                       and detection.get("candidate_count") == len(language_ids)
                       and isinstance(detection.get("language_logits"), dict)
                       and set(detection["language_logits"]) == set(language_ids)
                       and HEX64.fullmatch(detection.get("raw_logits_sha256", "")) is not None
                       and isinstance(detection.get("chosen_language_ids"), list)
                       and len(detection["chosen_language_ids"]) == 1
                       and len(prefix) == 4
                       and prefix[0] == contract["decoder_start_token_id"]
                       and prefix[1] == detection["chosen_language_ids"][0]
                       and prefix[2:] == expected_prefix_tail,
                       "multilingual detection/prefix evidence invalid")
        language_scores = detection["language_logits"]
        worker.require(all(type(x) in (int, float) and math.isfinite(x)
                           for x in language_scores.values()),
                       "language detection scores nonfinite")
        maximum = max(language_scores.values())
        expected_id = min(language_ids[name] for name, value in language_scores.items()
                          if value == maximum)
        worker.require(detection["chosen_language_ids"] == [expected_id],
                       "detected language is not the argmax of saved logits")
    else:
        worker.require(output.get("language_detection") is None,
                       "English Whisper unexpectedly has language detection")
        worker.require(contract["forced_decoder_ids"] == [[1,
                       contract["no_timestamps_token_id"]]]
                       and prefix == [contract["decoder_start_token_id"],
                                      contract["no_timestamps_token_id"]],
                       "English-only Whisper decoder prefix invalid")


def check_ctc(output: dict, frames: int) -> None:
    ids = output.get("frame_ids")
    # Wav2Vec2's 7 fixed feature extractor kernels/strides map 16 kHz samples.
    length = frames
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2), strict=True):
        length = (length - kernel) // stride + 1
    worker.require(isinstance(ids, list) and len(ids) == length
                   and all(type(x) is int and 0 <= x < 32 for x in ids),
                   "CTC frame path invalid")
    blank = output.get("blank_id")
    collapsed, prior = [], None
    for token in ids:
        if token != prior and token != blank:
            collapsed.append(token)
        prior = token
    worker.require(output.get("collapsed_nonblank_ids") == collapsed
                   and output.get("termination") == "complete_ctc_frames",
                   "CTC collapse or completion invalid")


def audit_attempt() -> dict:
    require_v3_go()
    attempt, receipt_path, review_path, model = ATTEMPT, V2_RECEIPT, V2_GO, MODEL
    worker.require(attempt.is_dir() and not attempt.is_symlink(),
                   "immutable English V2 attempt missing")
    worker.require_independent_go(model, receipt_path, review_path)
    parent = json.loads((attempt / "parent_result.json").read_text())
    start = json.loads((attempt / "parent_start.json").read_text())
    worker.require(parent.get("schema") == "nonspeech_descriptive_parent_result_v2"
                   and parent.get("status") == "worker_exited_pending_audit"
                   and parent.get("exit_code") == 0 and parent.get("limit_reason") is None
                   and parent.get("host_after_ok") is True, "parent did not finish cleanly")
    worker.require(parent.get("model") == model and start.get("model") == model
                   and start.get("schema") == "nonspeech_descriptive_parent_start_v2"
                   and start.get("review_sha256") == worker.sha_file(review_path)
                   and start.get("preflight_sha256") == worker.sha_file(receipt_path)
                   and parent.get("review_sha256") == worker.sha_file(review_path)
                   and parent.get("preflight_sha256") == worker.sha_file(receipt_path)
                   and parent.get("prior_failed_sha256") == worker.MODEL_PATHS[model][
                       "prior_sha256"],
                   "parent source binding changed")
    from . import descriptive_remaining_parent_v2 as model_parent
    worker.require(parent.get("parent_source_sha256") == worker.sha_file(
        Path(model_parent.__file__)), "V2 parent source hash changed")
    before, _ = timing_audit.saved_host(attempt, "before")
    after, log = timing_audit.saved_host(attempt, "after")
    started = timing.utc(start["utc"])
    exited = timing.utc(parent["worker_exit_observed_utc"])
    collected = timing.utc(after["collected_utc"])
    worker.require(timing.utc(before["collected_utc"]) <= started,
                   "before-host probe was collected after launch")
    expected_timing = timing.validate_post_host(log, started, exited, collected)
    worker.require(all(parent.get(key) == value for key, value in expected_timing.items()),
                   "V2 host timing receipt changed")
    data = rows(attempt / "records.jsonl")
    types = [row.get("type") for row in data]
    worker.require(len(data) == 42 and types[0] == "start" and types[-1] == "complete"
                   and types.count("start") == 1 and types.count("case") == 22
                   and types.count("silence_difference") == 18
                   and types.count("complete") == 1
                   and set(types) == {"start", "case", "silence_difference", "complete"},
                   "worker record inventory incomplete or contains extra rows")
    header, end = data[0], data[-1]
    worker.require(header.get("schema") == "nonspeech_descriptive_remaining_v2"
                   and header.get("model") == model
                   and header.get("source_identity") == worker.source_identity()
                   and header.get("manifest_sha256") == worker.MANIFEST_SHA256
                   and end.get("status") == "complete"
                   and end.get("planned_cases") == 20 and end.get("planned_repeats") == 2,
                   "worker start/end evidence invalid")
    cases = worker.trusted_cases()
    expected = [x["case_id"] for x in cases]
    planned = expected + [name for name in expected if name.endswith("_silence")]
    case_rows = [row for row in data if row.get("type") == "case"]
    worker.require([row.get("case_id") for row in case_rows] == planned
                   and len(case_rows) == 22 and all(row.get("status") == "complete"
                                                    for row in case_rows),
                   "planned case and repeat coverage invalid")
    by_name = {x["case_id"]: x for x in cases}
    for index, row in enumerate(case_rows):
        name = row["case_id"]
        repeated = index >= 20
        worker.require(row.get("run_id") == name + ("__repeat" if repeated else "")
                       and row.get("repeat") is repeated
                       and row.get("wav_sha256") == by_name[name]["wav_sha256"]
                       and row.get("pcm_sha256") == by_name[name]["pcm_sha256"]
                       and finite_nonnegative(row.get("elapsed_seconds"))
                       and row["elapsed_seconds"] <= worker.MAX_SECONDS,
                       "case identity/timing invalid")
        check_geometry(row.get("input_geometry"), model, by_name[name]["frames"])
        out = row.get("output")
        worker.require(isinstance(out, dict) and isinstance(out.get("transcript"), str)
                       and out.get("lexical_nonempty") is worker.lexical_nonempty(out["transcript"])
                       and out.get("unicode_scripts") == worker.scripts(out["transcript"]),
                       "case text scoring invalid")
        if model.startswith("whisper"):
            check_whisper(out, model)
        else:
            check_ctc(out, by_name[name]["frames"])
        check_trace(row.get("traces"), row.get("capture_counts"), model,
                    by_name[name]["frames"])
    for repeat in case_rows[20:]:
        original = next(row for row in case_rows[:20] if row["case_id"] == repeat["case_id"])
        worker.require(repeat["output"] == original["output"]
                       and repeat["traces"] == original["traces"]
                       and repeat["capture_counts"] == original["capture_counts"]
                       and repeat["input_geometry"] == original["input_geometry"],
                       "deterministic silence repeat changed")
    diffs = [row for row in data if row.get("type") == "silence_difference"]
    expected_diffs = [name for name in expected if not name.endswith("_silence")]
    worker.require(len(diffs) == 18 and {row.get("case_id") for row in diffs} == set(expected_diffs),
                   "silence contrast coverage invalid")
    for row in diffs:
        name = row["case_id"]
        worker.require(row.get("donor_case_id") == name[:3] + "_silence"
                       and isinstance(row.get("differences"), dict),
                       "silence donor identity invalid")
        trace = next(x for x in case_rows if x["case_id"] == name)["traces"]
        worker.require(set(row["differences"]) == set(trace), "difference phases changed")
        for phase, stages in trace.items():
            worker.require(set(row["differences"][phase]) == set(stages),
                           "difference stage coverage changed")
            for stage, metric in stages.items():
                difference = row["differences"][phase][stage]
                worker.require(finite_nonnegative(difference.get("whole_l2"))
                               and isinstance(difference.get("row_l2"), list)
                               and len(difference["row_l2"]) == len(metric["row_l2"])
                               and all(finite_nonnegative(x) for x in difference["row_l2"])
                               and difference.get("nonzero_rows") == sum(
                                   x != 0 for x in difference["row_l2"])
                               and math.isclose(difference["whole_l2"],
                                                math.sqrt(sum(x * x for x in difference["row_l2"])),
                                                rel_tol=1e-6, abs_tol=1e-5),
                               "difference curve invalid")
    # Difference rows must follow both the recipient and the same-duration silence
    # row. Repeats are last; this forbids fabricated early contrasts.
    positions = {row["case_id"]: i for i, row in enumerate(data)
                 if row.get("type") == "case" and not row["repeat"]}
    first_repeat = min(i for i, row in enumerate(data)
                       if row.get("type") == "case" and row["repeat"])
    for index, row in enumerate(data):
        if row.get("type") == "silence_difference":
            worker.require(index > positions[row["case_id"]]
                           and index > positions[row["donor_case_id"]]
                           and index < first_repeat,
                           "silence difference record order invalid")
    worker.require(worker._disk_bytes(attempt) <= worker.MAX_EVIDENCE_BYTES,
                   "evidence cap exceeded")
    return {"schema": "nonspeech_english_saved_audit_v3", "status": "PASS_STRUCTURAL",
            "model": model, "cases": 20, "repeats": 2, "silence_differences": 18,
            "records_sha256": worker.sha_file(attempt / "records.jsonl"),
            "v3_preflight_sha256": worker.sha_file(V3_RECEIPT),
            "v3_review_sha256": worker.sha_file(V3_GO),
            "limit": "Saved-record audit cannot reconstruct model forwards or activation hashes without raw tensors."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(worker.canonical(preflight_only()))
    else:
        result = audit_attempt()
        worker.require(V3_OUTPUT.parent.is_dir() and not V3_OUTPUT.exists(),
                       "exclusive V3 saved-audit output required")
        V3_OUTPUT.mkdir()
        with (V3_OUTPUT / "audit.json").open("x", encoding="utf-8") as stream:
            stream.write(worker.canonical(result) + "\n")
        print(worker.canonical(result))


if __name__ == "__main__":
    main()
