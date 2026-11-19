"""Saved-record structural audit; never loads candidate model weights."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from . import descriptive_worker_v1 as worker


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


def check_trace(traces: dict, model: str) -> None:
    phases = ("detect", "generate") if model == "whisper_multilingual" else (
        "generate",) if model == "whisper_english" else ("ctc",)
    worker.require(set(traces) == set(phases), "capture phase coverage changed")
    for phase, stages in traces.items():
        expected = set(worker.WHISPER_STAGES)
        if phase == "detect":
            expected.remove("mel")
        elif phase == "ctc":
            expected = set(worker.WAV_STAGES)
        worker.require(set(stages) == expected, "capture stage coverage changed")
        for name, metric in stages.items():
            shape, curve = metric.get("shape"), metric.get("row_l2")
            worker.require(metric.get("dtype") == "float32" and isinstance(shape, list)
                           and all(type(x) is int and x > 0 for x in shape)
                           and isinstance(curve, list) and len(curve) == shape[metric["time_axis"]]
                           and all(finite_nonnegative(x) for x in curve)
                           and isinstance(metric.get("sha256"), str)
                           and len(metric["sha256"]) == 64,
                           f"trace metric invalid: {phase}/{name}")


def check_whisper(output: dict, multilingual: bool) -> None:
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
                           and isinstance(row.get("logits_sha256"), str)
                           and len(row["logits_sha256"]) == 64 for row in scores),
                   "Whisper IDs or scores invalid")
    eos_seen = 50257 in generated
    worker.require(output.get("eos_seen") is eos_seen
                   and output.get("censored") is (not eos_seen and steps == worker.MAX_NEW_TOKENS)
                   and output.get("termination") == (
                       "eos" if eos_seen else "length_censored" if steps == worker.MAX_NEW_TOKENS
                       else "incomplete_unknown"), "Whisper EOS/censor accounting invalid")
    if eos_seen:
        worker.require(generated[-1] == 50257, "tokens after first generated EOS")
    worker.require(isinstance(output.get("text_with_special_tokens"), str),
                   "Whisper full text missing")
    if multilingual:
        detection = output.get("language_detection")
        worker.require(isinstance(detection, dict)
                       and detection.get("candidate_count") == 99
                       and isinstance(detection.get("language_logits"), dict)
                       and len(detection["language_logits"]) == 99
                       and isinstance(detection.get("chosen_language_ids"), list)
                       and len(detection["chosen_language_ids"]) == 1
                       and len(prefix) == 4 and prefix[1] == detection["chosen_language_ids"][0]
                       and prefix[2:] == [50359, 50363],
                       "multilingual detection/prefix evidence invalid")
    else:
        worker.require(output.get("language_detection") is None,
                       "English Whisper unexpectedly has language detection")


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


def audit_attempt(attempt: Path, receipt_path: Path, review_path: Path) -> dict:
    worker.require(attempt.is_dir() and attempt.is_absolute(), "attempt directory missing")
    worker.require_independent_go(receipt_path, review_path)
    parent = json.loads((attempt / "parent_result.json").read_text())
    start = json.loads((attempt / "parent_start.json").read_text())
    worker.require(parent.get("schema") == "nonspeech_descriptive_parent_result_v1"
                   and parent.get("status") == "worker_exited_pending_audit"
                   and parent.get("exit_code") == 0 and parent.get("limit_reason") is None
                   and parent.get("host_after_ok") is True, "parent did not finish cleanly")
    model = parent["model"]
    worker.require(model in worker.MODEL_NAMES and start.get("model") == model
                   and start.get("review_sha256") == worker.sha_file(review_path)
                   and start.get("preflight_sha256") == worker.sha_file(receipt_path),
                   "parent source binding changed")
    for label in ("before", "after"):
        host = json.loads((attempt / f"host_{label}.json").read_text())
        archive = host["power_log_archive"]
        raw_path = attempt / archive["file"]
        worker.require(raw_path.is_file() and raw_path.name == f"host_{label}_power_log.txt.gz"
                       and worker.sha_file(raw_path) == archive["compressed_sha256"]
                       and worker.sha_bytes(gzip.decompress(raw_path.read_bytes()))
                       == archive["raw_sha256"], "host log archive changed")
    data = rows(attempt / "records.jsonl")
    worker.require(len(data) >= 24 and data[0].get("type") == "start"
                   and data[-1].get("type") == "complete", "worker record book incomplete")
    header, end = data[0], data[-1]
    worker.require(header.get("schema") == "nonspeech_descriptive_v1"
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
                       and finite_nonnegative(row.get("elapsed_seconds")),
                       "case identity/timing invalid")
        out = row.get("output")
        worker.require(isinstance(out, dict) and isinstance(out.get("transcript"), str)
                       and out.get("lexical_nonempty") is worker.lexical_nonempty(out["transcript"])
                       and out.get("unicode_scripts") == worker.scripts(out["transcript"]),
                       "case text scoring invalid")
        if model.startswith("whisper"):
            check_whisper(out, model == "whisper_multilingual")
        else:
            check_ctc(out, by_name[name]["frames"])
        check_trace(row.get("traces"), model)
    for repeat in case_rows[20:]:
        original = next(row for row in case_rows[:20] if row["case_id"] == repeat["case_id"])
        worker.require(repeat["output"] == original["output"]
                       and repeat["traces"] == original["traces"],
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
                               and all(finite_nonnegative(x) for x in difference["row_l2"]),
                               "difference curve invalid")
    worker.require(worker._disk_bytes(attempt) <= worker.MAX_EVIDENCE_BYTES,
                   "evidence cap exceeded")
    return {"schema": "nonspeech_descriptive_saved_audit_v1", "status": "PASS_STRUCTURAL",
            "model": model, "cases": 20, "repeats": 2, "silence_differences": 18,
            "records_sha256": worker.sha_file(attempt / "records.jsonl"),
            "limit": "Saved-record audit cannot reconstruct model forwards or activation hashes without raw tensors."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    result = audit_attempt(args.attempt, args.preflight, args.review)
    path = args.attempt / "saved_audit.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(worker.canonical(result) + "\n")
    print(worker.canonical(result))


if __name__ == "__main__":
    main()
