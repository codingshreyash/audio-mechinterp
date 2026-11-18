"""Independent, CPU-only audit of frozen v2 ASR and human listening outcomes.

No model is imported or run. The private output contains references and original
human answers; never serve it to a listener. Missing evidence always means stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import wave
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "rendered_v2/manifest.json"
BINDING = HERE / "rendered_v2/TECHNICAL_BINDING.json"
CASES = HERE / "QA_CASES_PROSPECTIVE_V1.json"
QUEUE = HERE / "LISTENING_QUEUE_PROSPECTIVE_V1.json"
INPUTS = HERE / "ASR_INPUTS_PROSPECTIVE_V1.json"
PINS = HERE / "ASR_MODEL_PINS_PROSPECTIVE_V1.json"
PROTOCOL = HERE / "ASR_LISTENING_QA_PROTOCOL_PROSPECTIVE_V1.md"
AMENDMENT = HERE / "ASR_MIDDLE_SLOT_FLAG_AMENDMENT_V1.md"
RUNNER = HERE / "asr_screen_v2.py"
SERVER = HERE / "listening_server_v1.py"
HTML = HERE / "listening_v1.html"
APP = HERE / "listening_v1.js"
FREEZE = HERE / "ASR_SCREEN_FREEZE_V1.json"
EXPECTED = {
    "manifest": "15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d",
    "binding": "858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93",
    "cases": "d9a08a61b76e8ab56ab3a9b2d72a4aedc5871555487c7a33b712097fe3206ed8",
    "queue": "19319ac394a24f8d742285d4bb3ecfd78020b1d131fa57def5b3362fd0b73873",
    "inputs": "74f38f6704d1995b8e7040b329f7c0c0ea68b3cae36e774b875b4c358ae76d33",
    "pins": "1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b",
    "protocol": "30285a6c897e742eff59f593b3d12a1bd92d62585cd72171837c3e18fabe662e",
    "amendment": "9a6c12199e591ac623ee74938d13a4a6b11dbbbadce21e79cb75bd888176246d",
    "runner": "7df490c768d929e63845e00397ae1f0f77ba5b7f03722479fa8f5826527b4b78",
}
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
OPAQUE = re.compile(r"[0-9a-f]{20}\Z")
REVIEWER = re.compile(r"[A-Za-z0-9_.-]{3,64}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
STAGES = ("primary", "secondary", "tertiary")
KINDS = ("initial", "correction", "adjudication")
RESPONSE_KEYS = {"kind", "stage", "reviewer_id", "opaque_id", "middle_word",
                 "clear", "confidence", "play_count", "note", "reason",
                 "prior_sequence", "sequence", "timestamp_utc", "browser_user_agent",
                 "previous_sha256",
                 "record_sha256"}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def tokens(text: str) -> list[str]:
    require(isinstance(text, str), "transcript is not text")
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, 1):
        current = [index]
        for column, candidate in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (word != candidate)))
        previous = current
    return previous[-1]


def frozen_cases() -> tuple[list[dict], list[str], dict]:
    for name, path in (("manifest", MANIFEST), ("binding", BINDING),
                       ("cases", CASES), ("queue", QUEUE), ("inputs", INPUTS),
                       ("pins", PINS), ("protocol", PROTOCOL),
                       ("amendment", AMENDMENT), ("runner", RUNNER)):
        require(path.is_file() and sha(path) == EXPECTED[name],
                f"frozen {name} SHA-256 changed")
    manifest, binding = json.loads(MANIFEST.read_text()), json.loads(BINDING.read_text())
    private, public = json.loads(CASES.read_text()), json.loads(QUEUE.read_text())
    asr_inputs = json.loads(INPUTS.read_text())
    require(binding["status"] == "cpu_wav_technical_audit_passed_intelligibility_pending_no_model_output"
            and binding["manifest_sha256"] == EXPECTED["manifest"]
            and binding["matched_records"] == 240 and binding["wavs"] == 480
            and manifest["matched_records"] == 240 and manifest["wavs"] == 480
            and manifest["primary_pairs"] == 60 and manifest["reserve_pairs_used"] == 0,
            "technical binding or manifest population changed")
    cases, order = private["cases"], public["opaque_ids_in_order"]
    require(private["planned_clips"] == public["planned_clips"] == 480
            and len(cases) == len(order) == 480,
            "private case map or listening queue incomplete")
    by_id = {case["opaque_id"]: case for case in cases}
    require(len(by_id) == 480 and len(set(order)) == 480 and set(order) == set(by_id)
            and all(OPAQUE.fullmatch(ident) for ident in order),
            "case IDs or queue identities changed")
    manifest_outputs = {}
    for pair in manifest["pairs"]:
        for word in pair["words"]:
            item = pair["outputs"][word]
            require(item["path"] not in manifest_outputs, "manifest duplicate WAV")
            manifest_outputs[item["path"]] = (pair, word, item)
    require(len(manifest_outputs) == 480, "manifest missing WAV")
    projection = ("opaque_id", "audio_path", "wav_sha256", "pcm_int16_sha256", "frames")
    require(asr_inputs["planned_clips"] == 480
            and asr_inputs["cases"] == [{key: case[key] for key in projection}
                                        for case in cases],
            "ASR input projection does not match case map")
    seen_paths = set()
    for case in cases:
        path_text = case["audio_path"]
        require(path_text in manifest_outputs and path_text not in seen_paths,
                "case path missing or repeated")
        seen_paths.add(path_text)
        pair, word, item = manifest_outputs[path_text]
        expected_id = hashlib.sha256(("qwen-v2-qa-v1|" + EXPECTED["manifest"] +
                                      "|" + item["file_sha256"]).encode()).hexdigest()[:20]
        require(case["opaque_id"] == expected_id and case["pair_id"] == pair["pair_id"]
                and case["split"] == pair["split"] and case["category"] == pair["category"]
                and case["engine"] == pair["engine"]
                and case["template_id"] == pair["phrasing_id"]
                and case["target_word"] == word
                and case["paired_alternative"] in pair["words"]
                and case["paired_alternative"] != word
                and case["reference_script"] == item["spoken_script"]
                and case["wav_sha256"] == item["file_sha256"]
                and case["pcm_int16_sha256"] == item["pcm_int16_sha256"]
                and case["frames"] == item["frames"],
                "private case identity/reference differs from manifest")
        wav = (ROOT / path_text).resolve()
        require(wav.is_relative_to((HERE / "rendered_v2/clips").resolve())
                and wav.is_file() and sha(wav) == case["wav_sha256"],
                "case WAV path/hash changed")
        with wave.open(str(wav), "rb") as source:
            require(source.getnchannels() == 1 and source.getsampwidth() == 2
                    and source.getframerate() == 16000
                    and source.getnframes() == case["frames"],
                    "case PCM geometry changed")
            pcm = source.readframes(source.getnframes())
        require(hashlib.sha256(pcm).hexdigest() == case["pcm_int16_sha256"],
                "case PCM hash changed")
    remaining = sorted(by_id, key=lambda ident: hashlib.sha256(
        ("qwen-v2-listening-order-v1|" + EXPECTED["manifest"] + "|" + ident).encode()
    ).hexdigest())
    rebuilt = []
    while remaining:
        previous_pair = by_id[rebuilt[-1]]["pair_id"] if rebuilt else None
        chosen = next((i for i, ident in enumerate(remaining)
                       if by_id[ident]["pair_id"] != previous_pair), None)
        require(chosen is not None, "queue pair-spacing reconstruction failed")
        rebuilt.append(remaining.pop(chosen))
    require(order == rebuilt, "blind listening order changed")
    return cases, order, json.loads(PINS.read_text())


def verify_freeze() -> str:
    require(FREEZE.is_file(), "ASR freeze receipt absent")
    receipt = json.loads(FREEZE.read_text())
    require(receipt.get("status") == "independently_reviewed_asr_screen_frozen"
            and receipt.get("script_sha256") == EXPECTED["runner"]
            and all(receipt.get(f"{key}_sha256") == EXPECTED[key]
                    for key in ("manifest", "binding", "cases", "inputs", "pins",
                                "protocol", "amendment"))
            and receipt.get("auditor_sha256") == sha(Path(__file__))
            and receipt.get("queue_sha256") == EXPECTED["queue"]
            and receipt.get("server_sha256") == sha(SERVER)
            and receipt.get("html_sha256") == sha(HTML)
            and receipt.get("app_sha256") == sha(APP),
            "freeze does not bind auditor, runner, queue and listener sources")
    return sha(FREEZE)


def verify_asr_run(run_dir: Path, model_key: str, cases: list[dict],
                   pins: dict, freeze_sha: str, complete_sha: str) -> dict[str, dict]:
    require(HEX64.fullmatch(complete_sha) is not None, "expected ASR receipt SHA invalid")
    meta_path, raw_path, done_path = (run_dir / name for name in
                                      ("meta.json", "raw.jsonl", "complete.json"))
    require(not (run_dir / "error.json").exists()
            and all(path.is_file() for path in (meta_path, raw_path, done_path))
            and sha(done_path) == complete_sha, f"{model_key} complete receipt missing/changed")
    meta, done = json.loads(meta_path.read_text()), json.loads(done_path.read_text())
    model = pins["models"][model_key]
    require(meta.get("status") == "started_raw_asr_only"
            and meta.get("model_key") == model_key
            and meta.get("model_id") == model["repo_id"]
            and meta.get("model_revision") == model["revision"]
            and meta.get("model_files_sha256") == model["files_sha256"]
            and meta.get("script_sha256") == EXPECTED["runner"]
            and meta.get("pins_sha256") == EXPECTED["pins"]
            and meta.get("freeze_sha256") == freeze_sha
            and meta.get("technical_binding_sha256") == EXPECTED["binding"]
            and meta.get("asr_inputs_sha256") == EXPECTED["inputs"]
            and meta.get("provenance_cases_sha256") == EXPECTED["cases"]
            and meta.get("planned_clips") == len(cases)
            and meta.get("device") == "cpu" and meta.get("threads") == 4,
            f"{model_key} ASR metadata/provenance changed")
    expected_generation = {
        "whisper": {"do_sample": False, "num_beams": 1,
                    "max_new_tokens": 128, "processor_attention_mask": True},
        "wav2vec2": {"decoder": "frame_argmax_ctc_greedy"},
    }
    require(meta.get("generation") == expected_generation,
            "ASR decoding settings changed")
    require(done.get("status") == "complete_raw_asr_no_reference_comparison"
            and done.get("model_key") == model_key
            and done.get("planned_clips") == done.get("raw_rows") == len(cases)
            and done.get("meta_sha256") == sha(meta_path)
            and done.get("raw_sha256") == sha(raw_path)
            and done.get("no_model_answer_inference") is True,
            f"{model_key} ASR completion/hash mismatch")
    raw_bytes = raw_path.read_bytes()
    raw_lines = raw_bytes.splitlines(keepends=True)
    require(len(raw_lines) == len(cases), f"{model_key} raw row count is not {len(cases)}")
    shared_keys = {"ordinal", "opaque_id", "wav_sha256", "pcm_int16_sha256",
                   "raw_transcript"}
    decoder_keys = ({"generated_token_ids", "decoder_cap", "ended_with_eos"}
                    if model_key == "whisper" else
                    {"frame_argmax_token_ids", "ctc_frame_count"})
    result = {}
    for ordinal, (line_bytes, case) in enumerate(zip(raw_lines, cases, strict=True), 1):
        require(line_bytes.endswith(b"\n"), f"{model_key} raw row truncated")
        line = line_bytes.decode("utf-8")
        row = json.loads(line)
        token_key = "generated_token_ids" if model_key == "whisper" else "frame_argmax_token_ids"
        token_ids = row.get(token_key)
        require(set(row) == shared_keys | decoder_keys
                and row.get("ordinal") == ordinal
                and row.get("opaque_id") == case["opaque_id"]
                and row.get("wav_sha256") == case["wav_sha256"]
                and row.get("pcm_int16_sha256") == case["pcm_int16_sha256"]
                and isinstance(row.get("raw_transcript"), str)
                and isinstance(token_ids, list) and bool(token_ids)
                and all(type(token) is int and token >= 0 for token in token_ids),
                f"{model_key} raw row {ordinal} changed or incomplete")
        if model_key == "whisper":
            require(row.get("decoder_cap") == 128
                    and type(row.get("ended_with_eos")) is bool,
                    "Whisper token cap/EOS metadata missing")
        else:
            require(row.get("ctc_frame_count") == len(token_ids),
                    "Wav2Vec2 CTC frame count mismatch")
        result[case["opaque_id"]] = row
    return result


def verify_responses(path: Path, expected_sha: str, order: list[str],
                     meta_path: Path) -> list[dict]:
    require(HEX64.fullmatch(expected_sha) is not None and path.is_file()
            and sha(path) == expected_sha, "listening response SHA-256 missing/changed")
    require(meta_path.is_file(), "listening run metadata missing")
    meta = json.loads(meta_path.read_text())
    provenance = meta.get("provenance", {})
    require(meta.get("status") == "blind_listening_collection_no_quality_verdict"
            and meta.get("server_version") == "BlindListening/1"
            and all(provenance.get(f"{name}_sha256") == EXPECTED[name]
                    for name in ("manifest", "binding", "queue"))
            and provenance.get("private_sha256") == EXPECTED["cases"]
            and provenance.get("server_sha256") == sha(SERVER)
            and provenance.get("html_sha256") == sha(HTML)
            and provenance.get("app_sha256") == sha(APP),
            "listening run provenance/source changed")
    records = []
    previous = "0" * 64
    initials = {}
    stage_owners = {}
    for raw in path.open("rb"):
        require(raw.endswith(b"\n"), "truncated response row")
        record = json.loads(raw)
        require(set(record) == RESPONSE_KEYS
                and type(record["sequence"]) is int
                and record["sequence"] == len(records) + 1
                and record["previous_sha256"] == previous,
                "listening row schema/order/hash chain changed")
        saved = record["record_sha256"]
        require(isinstance(saved, str) and HEX64.fullmatch(saved) is not None
                and hashlib.sha256(canonical({k: v for k, v in record.items()
                                              if k != "record_sha256"})).hexdigest() == saved,
                "listening response hash chain changed")
        previous = saved
        kind, stage, reviewer, ident = (record[key] for key in
                                       ("kind", "stage", "reviewer_id", "opaque_id"))
        require(kind in KINDS and stage in STAGES and ident in order
                and isinstance(reviewer, str) and REVIEWER.fullmatch(reviewer)
                and isinstance(record["middle_word"], str)
                and 0 < len(record["middle_word"].strip()) <= 120
                and "\n" not in record["middle_word"]
                and "\r" not in record["middle_word"]
                and type(record["clear"]) is bool
                and type(record["confidence"]) is int and 1 <= record["confidence"] <= 5
                and type(record["play_count"]) is int and 1 <= record["play_count"] <= 100
                and isinstance(record["note"], str) and len(record["note"]) <= 500
                and isinstance(record["reason"], str) and len(record["reason"]) <= 500
                and isinstance(record["browser_user_agent"], str)
                and 1 <= len(record["browser_user_agent"].strip()) <= 500
                and record["browser_user_agent"] != "direct_test_client_no_browser"
                and "\n" not in record["browser_user_agent"]
                and "\r" not in record["browser_user_agent"],
                "listening response values invalid")
        stamp = datetime.fromisoformat(record["timestamp_utc"])
        require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0,
                "listening timestamp lacks UTC")
        key = stage, ident
        if kind == "initial":
            require(record["prior_sequence"] is None and not record["reason"]
                    and key not in initials
                    and (stage not in stage_owners or stage_owners[stage] == reviewer)
                    and all(other_reviewer != reviewer for (other_stage, other_id), other_reviewer
                            in ((key2, row["reviewer_id"]) for key2, row in initials.items())
                            if other_id == ident),
                    "duplicate or non-independent initial response")
            if stage == "secondary":
                require(("primary", ident) in initials,
                        "secondary response precedes primary")
            if stage == "tertiary":
                require(("secondary", ident) in initials,
                        "tertiary response precedes secondary")
            initials[key] = record
            stage_owners[stage] = reviewer
        else:
            prior = record["prior_sequence"]
            require(type(prior) is int and 1 <= prior <= len(records)
                    and record["reason"].strip(), "revision lacks prior row/reason")
            original = records[prior - 1]
            require(original["opaque_id"] == ident and original["stage"] == stage,
                    "revision points to a different clip/stage")
            if kind == "correction":
                require(original["reviewer_id"] == reviewer,
                        "correction not from original reviewer")
            else:
                require(stage == "tertiary" and original["kind"] == "initial"
                        and ("primary", ident) in initials
                        and ("secondary", ident) in initials
                        and reviewer not in {row["reviewer_id"] for (s, i), row
                                             in initials.items() if i == ident},
                        "adjudication lacks independent reviewer")
        records.append(record)
    primary_order = [record["opaque_id"] for record in records
                     if record["kind"] == "initial" and record["stage"] == "primary"]
    require(primary_order == order[:len(primary_order)],
            "primary listening responses do not follow frozen queue order")
    return records


def compare_asr(case: dict, row: dict, key: str) -> dict:
    hypothesis = tokens(row["raw_transcript"])
    reference = tokens(case["reference_script"])
    target = tokens(case["target_word"])
    alternative = tokens(case["paired_alternative"])
    require(len(target) == len(alternative) == 1 and bool(reference),
            "case target/reference tokenization invalid")
    errors = word_errors(reference, hypothesis)
    return {"raw_transcript": row["raw_transcript"], "normalized_tokens": hypothesis,
            "literal_script_exact": row["raw_transcript"] == case["reference_script"],
            "normalized_script_exact": hypothesis == reference,
            "word_errors": errors, "reference_words": len(reference),
            "wer": errors / len(reference),
            "target_exactly_once": hypothesis.count(target[0]) == 1,
            "paired_alternative_present": alternative[0] in hypothesis,
            "empty": not hypothesis,
            "truncated": (key == "whisper" and not row["ended_with_eos"]
                          and len(row["generated_token_ids"]) >= row["decoder_cap"])}


def summarize(rows: list[dict]) -> dict:
    word_errors_total = sum(row["word_errors"] for row in rows)
    reference_words_total = sum(row["reference_words"] for row in rows)
    return {"clips": len(rows),
            "literal_script_exact": sum(row["literal_script_exact"] for row in rows),
            "normalized_script_exact": sum(row["normalized_script_exact"] for row in rows),
            "target_exactly_once": sum(row["target_exactly_once"] for row in rows),
            "paired_alternative_present": sum(row["paired_alternative_present"] for row in rows),
            "empty": sum(row["empty"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "word_errors": word_errors_total, "reference_words": reference_words_total,
            "wer": word_errors_total / reference_words_total if reference_words_total else None}


def audit(cases: list[dict], order: list[str], asr: dict[str, dict[str, dict]],
          responses: list[dict]) -> dict:
    by_id = {case["opaque_id"]: case for case in cases}
    require(set(by_id) == set(order) and all(set(rows) == set(by_id) for rows in asr.values())
            and set(asr) == {"whisper", "wav2vec2"}, "ASR/case coverage mismatch")
    initial = {(row["stage"], row["opaque_id"]): row for row in responses
               if row["kind"] == "initial"}
    revisions = defaultdict(list)
    for row in responses:
        if row["kind"] != "initial":
            revisions[row["opaque_id"]].append(row)
    details = []
    aggregate = {model: {group: defaultdict(list) for group in
                         ("split", "category", "engine", "template_id", "pair_id")}
                 for model in asr}
    flags = Counter()
    failed = []
    for ident in order:
        case = by_id[ident]
        measured = {model: compare_asr(case, asr[model][ident], model)
                    for model in ("whisper", "wav2vec2")}
        disagreement = measured["whisper"]["normalized_tokens"] != measured["wav2vec2"]["normalized_tokens"]
        first = initial.get(("primary", ident))
        second = initial.get(("secondary", ident))
        third = initial.get(("tertiary", ident))
        target = tokens(case["target_word"])
        def answer(row: dict | None) -> list[str] | None:
            return tokens(row["middle_word"]) if row else None
        reasons = []
        if any(not value["normalized_script_exact"] for value in measured.values()):
            reasons.append("asr_script_mismatch")
        if disagreement:
            reasons.append("asr_disagreement")
        if first is None:
            reasons.append("missing_primary")
        else:
            if not first["clear"]:
                reasons.append("primary_unclear")
            if first["confidence"] <= 2:
                reasons.append("primary_low_confidence")
            if answer(first) != target:
                reasons.append("primary_target_miss")
        flagged = bool(reasons)
        needs_third = (second is not None and first is not None
                       and (answer(first) != answer(second)
                            or first["clear"] != second["clear"]))
        votes = [row for row in (first, second, third)
                 if row and answer(row) == target and row["clear"]]
        unique_voters = {row["reviewer_id"] for row in votes}
        confirmed = (not flagged and first is not None
                     and answer(first) == target and first["clear"])
        if flagged:
            confirmed = (second is not None and (not needs_third or third is not None)
                         and len(unique_voters) >= 2)
        problems = []
        if first is None:
            problems.append("missing_primary")
        if flagged and second is None:
            problems.append("missing_required_secondary")
        if not flagged and second is not None:
            problems.append("unflagged_extra_secondary")
        if needs_third and third is None:
            problems.append("missing_required_tertiary")
        if not needs_third and third is not None:
            problems.append("unrequired_tertiary")
        if revisions[ident]:
            problems.append("revision_requires_independent_review")
        if not confirmed:
            problems.append("not_quality_confirmed")
        if problems:
            failed.append({"opaque_id": ident, "problems": problems})
        for reason in reasons:
            flags[reason] += 1
        detail = {"opaque_id": ident, "pair_id": case["pair_id"],
                  "split": case["split"], "category": case["category"],
                  "engine": case["engine"], "template_id": case["template_id"],
                  "target_word": case["target_word"],
                  "paired_alternative": case["paired_alternative"],
                  "reference_script": case["reference_script"],
                  "asr": measured, "asr_disagreement": disagreement,
                  "flag_reasons": reasons, "requires_secondary": flagged,
                  "requires_tertiary": needs_third,
                  "original_responses": [row for row in (first, second, third) if row],
                  "revisions": revisions[ident],
                  "quality_confirmed_from_originals": confirmed,
                  "problems": problems}
        details.append(detail)
        for model, metric in measured.items():
            for group in aggregate[model]:
                aggregate[model][group][str(case[group])].append(metric)
    results = {model: {"all": summarize([detail["asr"][model] for detail in details]),
                       **{group: {name: summarize(rows) for name, rows in values.items()}
                          for group, values in aggregate[model].items()}}
               for model in asr}
    return {"status": "go" if not failed and len(initial) >= len(cases) else "stop",
            "planned_clips": len(cases), "asr_raw_outcomes": {key: len(rows) for key, rows in asr.items()},
            "listener_initial_counts": dict(Counter(stage for stage, _ in initial)),
            "flag_counts": dict(flags),
            "asr_disagreement_count": sum(detail["asr_disagreement"] for detail in details),
            "quality_confirmed_count": sum(detail["quality_confirmed_from_originals"]
                                           and not detail["problems"] for detail in details),
            "aggregate": results, "failed_cases": failed, "cases": details,
            "limits": "Input-quality gate only; no Qwen mechanism or security claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--whisper-dir", type=Path)
    parser.add_argument("--whisper-complete-sha256")
    parser.add_argument("--wav2vec2-dir", type=Path)
    parser.add_argument("--wav2vec2-complete-sha256")
    parser.add_argument("--listening-dir", type=Path)
    parser.add_argument("--responses-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        cases, order, _pins = frozen_cases()
        print(json.dumps({"status": "validated_sources_only_no_outcome_opened",
                          "clips": len(cases), "queue": len(order)}, indent=2))
        return
    require(args.output is not None, "exclusive --output path required")
    require(not args.output.exists(), "audit output already exists")
    try:
        require(all((args.whisper_dir, args.whisper_complete_sha256,
                     args.wav2vec2_dir, args.wav2vec2_complete_sha256,
                     args.listening_dir, args.responses_sha256)),
                "all ASR/listening paths and external receipt hashes required")
        cases, order, pins = frozen_cases()
        freeze_sha = verify_freeze()
        asr = {
            "whisper": verify_asr_run(args.whisper_dir, "whisper", cases, pins, freeze_sha,
                                      args.whisper_complete_sha256),
            "wav2vec2": verify_asr_run(args.wav2vec2_dir, "wav2vec2", cases, pins, freeze_sha,
                                       args.wav2vec2_complete_sha256),
        }
        responses = verify_responses(args.listening_dir / "responses.jsonl",
                                     args.responses_sha256, order,
                                     args.listening_dir / "run_meta.json")
        report = audit(cases, order, asr, responses)
        report["provenance"] = {"auditor_sha256": sha(Path(__file__)),
                                "freeze_sha256": freeze_sha,
                                "manifest_sha256": EXPECTED["manifest"],
                                "binding_sha256": EXPECTED["binding"],
                                "cases_sha256": EXPECTED["cases"],
                                "queue_sha256": EXPECTED["queue"],
                                "amendment_sha256": EXPECTED["amendment"],
                                "whisper_complete_sha256": args.whisper_complete_sha256,
                                "wav2vec2_complete_sha256": args.wav2vec2_complete_sha256,
                                "responses_sha256": args.responses_sha256,
                                "listening_meta_sha256": sha(args.listening_dir / "run_meta.json")}
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
        report = {"status": "stop", "gate_errors": [f"{type(exc).__name__}: {exc}"],
                  "quality_confirmed_count": 0, "failed_cases": [],
                  "provenance": {"auditor_sha256": sha(Path(__file__)),
                                 "manifest_sha256": EXPECTED["manifest"],
                                 "binding_sha256": EXPECTED["binding"],
                                 "amendment_sha256": EXPECTED["amendment"],
                                 "whisper_complete_sha256": args.whisper_complete_sha256,
                                 "wav2vec2_complete_sha256": args.wav2vec2_complete_sha256,
                                 "responses_sha256": args.responses_sha256},
                  "limits": "Invalid or incomplete evidence; no Qwen inference authorized"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"],
                      "quality_confirmed_count": report["quality_confirmed_count"],
                      "failed_cases": len(report["failed_cases"]),
                      "output_sha256": sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
