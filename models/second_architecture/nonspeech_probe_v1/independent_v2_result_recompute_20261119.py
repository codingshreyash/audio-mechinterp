"""Read-only independent recomputation for the first multilingual V2 attempt.

This deliberately loads only tokenizer metadata and saved evidence, never model
weights. It does not import the candidate runner or its saved-result auditor.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import plistlib
import re
import unicodedata
import wave
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from transformers import WhisperTokenizerFast


HERE = Path(__file__).resolve().parent
RENDER = HERE / "runs/render_20261119_0740Z"
ATTEMPT = HERE / "runs/descriptive_whisper_multilingual_20261119_v2"
PRIOR = HERE / "runs/descriptive_whisper_multilingual_20261119_v1"
PRIOR_PARENT_SHA = "806bfd0c84899e88cec47b24b0966274561cdd155908c52369d0699496c2404a"
PRIOR_RECORDS_SHA = "984625850ac33cd4c5d039304162a9b4396bd2a2ad3d31513a231eb452e708a3"
RECORDS_SHA = "588165986eadcfc7df04f6a8869de78ccd656f5b9be578fe41b3e1395abeec42"
EVENT = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+([^\t]+)\t", re.M)
TERMINAL = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+: Showing all currently held IOKit power assertions\s*$", re.M)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw(path: Path) -> bytes:
    assert path.is_file() and not path.is_symlink(), path
    return path.read_bytes()


def timestamp(match: re.Match) -> datetime:
    return datetime.strptime(" ".join(match.group(1, 2, 3)), "%Y-%m-%d %H:%M:%S %z")


def script_names(text: str) -> list[str]:
    return sorted({unicodedata.name(c, "UNKNOWN").split()[0]
                   for c in text if unicodedata.category(c).startswith("L")})


def host(label: str) -> tuple[dict, str]:
    saved = json.loads(raw(ATTEMPT / f"host_{label}.json"))
    archive = saved["power_log_archive"]
    compressed = raw(ATTEMPT / f"host_{label}_power_log.txt.gz")
    log_bytes = gzip.decompress(compressed)
    assert digest(compressed) == archive["compressed_sha256"]
    assert digest(log_bytes) == archive["raw_sha256"]
    assert "Now drawing from 'AC Power'" in saved["outputs"]["power"]
    assert plistlib.loads(saved["outputs"]["battery"].encode())[0]["ExternalConnected"] is True
    assert plistlib.loads(saved["outputs"]["clamshell"].encode())[0]["AppleClamshellState"] is False
    return saved, log_bytes.decode()


def main() -> dict:
    # Prior invalid attempt is a separately hashed failure, never pooled here.
    assert digest(raw(PRIOR / "parent_result.json")) == PRIOR_PARENT_SHA
    assert digest(raw(PRIOR / "records.jsonl")) == PRIOR_RECORDS_SHA
    assert json.loads(raw(PRIOR / "parent_result.json"))["status"] == "invalid_or_partial"

    manifest_bytes = raw(RENDER / "manifest.json")
    assert digest(manifest_bytes) == "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90"
    manifest = json.loads(manifest_bytes)
    expected = {r["case_id"]: r for r in manifest["rows"]}
    assert len(expected) == 20
    for case in expected.values():
        blob = raw(RENDER / case["wav"])
        assert digest(blob) == case["wav_sha256"]
        with wave.open(str(RENDER / case["wav"]), "rb") as sound:
            assert (sound.getnchannels(), sound.getsampwidth(), sound.getframerate(),
                    sound.getnframes(), sound.getcomptype()) == (
                        1, 2, 16000, case["frames"], "NONE")
            pcm = sound.readframes(sound.getnframes())
        assert digest(pcm) == case["pcm_sha256"]

    pins = json.loads(raw(HERE / "descriptive_pins_v1.json"))
    cfg = pins["models"]["whisper_multilingual"]
    snapshot = Path.home() / ".cache/huggingface/hub/models--openai--whisper-tiny/snapshots" / cfg["revision"]
    metadata_pin = json.loads(raw(HERE / cfg["metadata_file"]))
    assert metadata_pin["revision"] == cfg["revision"]
    for name, expected_sha in metadata_pin["files_sha256"].items():
        # Hugging Face snapshots use symlinks to the content-addressed blob store.
        path = snapshot / name
        assert path.is_file() and path.resolve().is_relative_to(snapshot.parents[1])
        assert digest(path.read_bytes()) == expected_sha, name
    tokenizer = WhisperTokenizerFast.from_pretrained(snapshot, local_files_only=True)
    assert tokenizer.eos_token_id == tokenizer.pad_token_id == 50257
    lang_to_id = json.loads((snapshot / "generation_config.json").read_bytes())["lang_to_id"]
    id_to_lang = {v: k for k, v in lang_to_id.items()}

    records_bytes = raw(ATTEMPT / "records.jsonl")
    assert digest(records_bytes) == RECORDS_SHA and records_bytes.endswith(b"\n")
    records = [json.loads(line) for line in records_bytes.splitlines()]
    types = Counter(r["type"] for r in records)
    assert len(records) == 42 and types == {"start": 1, "case": 22,
                                            "silence_difference": 18, "complete": 1}
    case_rows = [r for r in records if r["type"] == "case"]
    first_rows = case_rows[:20]
    assert [r["case_id"] for r in first_rows] == sorted(expected)
    assert [r["case_id"] for r in case_rows[20:]] == ["d02_silence", "d08_silence"]
    assert all(r["status"] == "complete" for r in case_rows)
    assert records[-1]["status"] == "complete"
    v1_receipt = json.loads(raw(HERE / "runs/descriptive_source_preflight_v2/preflight.json"))
    assert records[0]["source_identity"] == v1_receipt["source_identity"]
    assert records[0]["installed"] == v1_receipt["installed"]
    assert records[0]["assets"] == v1_receipt["models"]["whisper_multilingual"]
    assert records[0]["manifest_sha256"] == digest(manifest_bytes)

    report_rows = []
    for row in case_rows:
        name = row["case_id"]
        ref = expected[name]
        assert row["frames"] == ref["frames"] and row["wav_sha256"] == ref["wav_sha256"]
        assert row["pcm_sha256"] == ref["pcm_sha256"]
        out = row["output"]
        ids, prefix, gen = out["sequence_ids"], out["prefix_ids"], out["generated_ids"]
        assert ids == prefix + gen and len(gen) == out["actual_generation_steps"]
        assert prefix[0] == 50258 and prefix[2:] == [50359, 50363]
        assert len(out["score_rows"]) == len(gen)
        assert tokenizer.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) == out["text_with_special_tokens"]
        text = tokenizer.decode(gen, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        assert text == out["transcript"]
        lexical = any(unicodedata.category(c)[0] in "LN" for c in text)
        assert lexical == out["lexical_nonempty"] and script_names(text) == out["unicode_scripts"]
        eos = 50257 in gen
        assert eos == out["eos_seen"] and (not eos or gen[-1] == 50257)
        assert out["censored"] == (not eos and len(gen) == 128)
        assert out["termination"] == ("eos" if eos else "length_censored" if len(gen) == 128
                                      else "incomplete_unknown")
        # This is a pretranscription detection pass, not a generated text token.
        detection = out["language_detection"]
        scores = detection["language_logits"]
        assert len(scores) == 99 and set(scores) == set(lang_to_id)
        assert all(math.isfinite(v) for v in scores.values())
        winner = min((lang_to_id[k] for k, v in scores.items() if v == max(scores.values())))
        assert prefix[1] == winner == detection["chosen_language_ids"][0]

        geometry = row["input_geometry"]
        active = math.ceil(ref["frames"] / 160)
        assert geometry["pcm_frames"] == ref["frames"]
        assert geometry["attention_mask"] == [1] * active + [0] * (3000 - active)
        assert geometry["mask_sha256"] == digest(bytes(geometry["attention_mask"]))
        assert geometry["active_mel_frames"] == active
        assert geometry["nominal_active_encoder_rows"] == math.ceil(active / 2)
        assert geometry["encoder_frames"] == 1500
        assert row["capture_counts"] == {"detect": {"post_conv_pos": 1,
                                                       "post_block0": 1, "final_encoder": 1},
                                         "generate": {"mel": 1, "post_conv_pos": 1,
                                                      "post_block0": 1, "final_encoder": 1}}
        for phase, stages in row["traces"].items():
            for stage, trace in stages.items():
                shape = [1, 80, 3000] if stage == "mel" else [1, 1500, 384]
                axis = 2 if stage == "mel" else 1
                assert trace["shape"] == shape and trace["time_axis"] == axis
                assert len(trace["row_l2"]) == shape[axis]
                assert all(math.isfinite(v) and v >= 0 for v in trace["row_l2"])
        # The only full saved activations/logits are the first decoder state and raw step.
        state = out["first_decoder_step"]
        state_values = np.asarray(state["values"], dtype=np.float32)
        assert list(state_values.shape) == state["shape"] == [1, len(prefix), 384]
        assert digest(state_values.tobytes()) == state["sha256"]
        raw_logits = out["first_raw_logits"]
        logits = np.asarray(raw_logits["values"], dtype=np.float32)
        assert list(logits.shape) == raw_logits["shape"] == [51865]
        assert np.isfinite(logits).all() and digest(logits.tobytes()) == raw_logits["sha256"]
        first_score = out["score_rows"][0]
        assert first_score["token_id"] == gen[0]
        assert first_score["raw_logits_sha256"] == raw_logits["sha256"]
        assert first_score["raw_selected_logit"] == float(logits[gen[0]])
        assert first_score["raw_top_id"] == int(logits.argmax())
        assert all(s["token_id"] == gen[i] for i, s in enumerate(out["score_rows"]))
        report_rows.append({"case_id": name, "repeat": row["repeat"],
                            "language": id_to_lang[winner], "text": text,
                            "generated_ids": gen, "lexical": lexical,
                            "eos": eos, "steps": len(gen),
                            "first_raw_top_id": int(logits.argmax())})
    for repeat in report_rows[20:]:
        original = next(r for r in report_rows[:20] if r["case_id"] == repeat["case_id"])
        assert {k: v for k, v in repeat.items() if k != "repeat"} == {
            k: v for k, v in original.items() if k != "repeat"}
    for diff in (r for r in records if r["type"] == "silence_difference"):
        assert diff["donor_case_id"] == diff["case_id"][:3] + "_silence"
        assert set(diff["differences"]) == {"detect", "generate"}
        for phase, stages in diff["differences"].items():
            assert set(stages) == ({"post_conv_pos", "post_block0", "final_encoder"}
                                   if phase == "detect" else {"mel", "post_conv_pos",
                                                              "post_block0", "final_encoder"})
            for metric in stages.values():
                values = metric["row_l2"]
                assert metric["nonzero_rows"] == sum(v != 0 for v in values)
                assert math.isclose(metric["whole_l2"], math.sqrt(sum(v*v for v in values)),
                                    rel_tol=1e-6, abs_tol=1e-5)

    parent = json.loads(raw(ATTEMPT / "parent_result.json"))
    before, before_log = host("before")
    after, log = host("after")
    parent_start = json.loads(raw(ATTEMPT / "parent_start.json"))
    start = datetime.fromisoformat(parent_start["utc"])
    assert parent_start["preflight_sha256"] == digest(raw(
        HERE / "runs/descriptive_source_preflight_v2/preflight.json"))
    assert parent_start["review_sha256"] == digest(raw(
        HERE / "INDEPENDENT_DESCRIPTIVE_SOURCE_GO_V2.json"))
    assert parent_start["v2_preflight_sha256"] == digest(raw(
        HERE / "runs/descriptive_host_timing_preflight_v2/preflight.json"))
    assert parent_start["v2_review_sha256"] == digest(raw(
        HERE / "INDEPENDENT_DESCRIPTIVE_HOST_TIMING_GO_V2.json"))
    exit_observed = datetime.fromisoformat(parent["worker_exit_observed_utc"])
    collected = datetime.fromisoformat(after["collected_utc"])
    events, terminal = list(EVENT.finditer(log)), list(TERMINAL.finditer(log))
    assert events and terminal and min(map(timestamp, events)) <= start
    assert timestamp(terminal[-1]) >= exit_observed - timedelta(seconds=1)
    assert timestamp(terminal[-1]) >= collected - timedelta(seconds=5)
    assert 0 <= (collected - exit_observed).total_seconds() <= 120
    assert all(not (start <= timestamp(e) <= collected and
                    (e.group(4).strip() in ("Sleep", "DarkWake", "Wake") or
                     "Thermal" in e.group(4))) for e in events)
    assert datetime.fromisoformat(before["collected_utc"]) <= start <= exit_observed <= collected
    before_events = list(EVENT.finditer(before_log))
    before_term = list(TERMINAL.finditer(before_log))
    before_time = datetime.fromisoformat(before["collected_utc"])
    before_start = before_time - timedelta(minutes=10)
    assert before_events and before_term and min(map(timestamp, before_events)) <= before_start
    assert before_time - timedelta(seconds=5) <= timestamp(before_term[-1]) <= before_time + timedelta(seconds=1)
    assert all(not (before_start <= timestamp(e) <= before_time and
                    (e.group(4).strip() in ("Sleep", "DarkWake", "Wake") or
                     "Thermal" in e.group(4))) for e in before_events)
    assert parent["status"] == "worker_exited_pending_audit" and parent["exit_code"] == 0
    assert parent["host_after_ok"] is True and parent["limit_reason"] is None
    saved_audit = json.loads(raw(ATTEMPT / "saved_audit_v2.json"))
    assert saved_audit["status"] == "PASS_STRUCTURAL"
    assert saved_audit["model_audit"]["records_sha256"] == RECORDS_SHA

    first = report_rows[:20]
    silence = {r["case_id"]: r for r in first if r["case_id"].endswith("_silence")}
    changed_from_silence = [r["case_id"] for r in first
                            if not r["case_id"].endswith("_silence") and
                            r["generated_ids"] != silence[r["case_id"][:3]+"_silence"]["generated_ids"]]
    return {"schema": "independent_nonspeech_multilingual_v2_recompute_20261119",
            "status": "GO_SAVED_EVIDENCE", "records_sha256": RECORDS_SHA,
            "cases": 20, "repeats": 2, "difference_rows": 18,
            "lexical_count": sum(r["lexical"] for r in first),
            "eos_count": sum(r["eos"] for r in first),
            "language_counts": dict(Counter(r["language"] for r in first)),
            "script_counts": dict(Counter(s for r in first for s in script_names(r["text"]))),
            "changed_from_same_duration_silence": changed_from_silence,
            "rows": first,
            "limitations": ["No raw encoder activations or later decoder logits were saved; their reported hashes cannot be independently recomputed.",
                            "All outputs are descriptive generated text from engineered sound; no causal intervention or security consequence is tested."]}


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))
