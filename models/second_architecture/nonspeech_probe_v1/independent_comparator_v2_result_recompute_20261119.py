"""Read-only independent saved-result check for the two non-speech controls.

This loads tokenizer metadata only. It never constructs a candidate model or
imports the experiment worker or its auditors.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import hashlib
import json
import math
import plistlib
import re
import unicodedata
import wave

import numpy as np
from transformers import WhisperTokenizerFast, Wav2Vec2Processor

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RENDER = RUNS / "render_20261119_0740Z"
MANIFEST_SHA = "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90"
MODEL = {
    "whisper_english": {
        "records_sha": "4fd1709d85ebe13ed6a46a7996491fbb227463df6592537435e81c4af91ee923",
        "prior": RUNS / "descriptive_whisper_english_20261119_v1",
        "receipt": RUNS / "descriptive_remaining_repair_preflight_v2/whisper_english.json",
        "review": HERE / "INDEPENDENT_DESCRIPTIVE_WHISPER_ENGLISH_REPAIR_GO_V2.json",
    },
    "wav2vec2": {
        "records_sha": "96880742d7762b1ec4517fd5f869ec37b39f6d9195e6c590e93825219cf060ff",
        "prior": RUNS / "descriptive_wav2vec2_20261119_v1",
        "receipt": RUNS / "descriptive_remaining_repair_preflight_v2/wav2vec2.json",
        "review": HERE / "INDEPENDENT_DESCRIPTIVE_WAV2VEC2_REPAIR_GO_V2.json",
    },
}
EVENT = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+([^\t]+)\t", re.M)
TERMINAL = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+: Showing all currently held IOKit power assertions\s*$", re.M)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> bytes:
    assert path.is_file() and not path.is_symlink(), path
    return path.read_bytes()


def iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def event_time(match: re.Match) -> datetime:
    return datetime.strptime(" ".join(match.group(1, 2, 3)), "%Y-%m-%d %H:%M:%S %z")


def scripts(value: str) -> list[str]:
    return sorted({unicodedata.name(c, "UNKNOWN").split()[0]
                   for c in value if unicodedata.category(c).startswith("L")})


def lexical(value: str) -> bool:
    return any(unicodedata.category(c)[0] in ("L", "N") for c in value)


def family(case_id: str) -> str:
    parts = case_id.split("_")
    return parts[1] if len(parts) == 2 else "_".join(parts[1:-1])


def host(attempt: Path, side: str) -> tuple[dict, str]:
    saved = json.loads(read(attempt / f"host_{side}.json"))
    info = saved["power_log_archive"]
    packed = read(attempt / f"host_{side}_power_log.txt.gz")
    raw = gzip.decompress(packed)
    assert sha(packed) == info["compressed_sha256"]
    assert sha(raw) == info["raw_sha256"]
    assert "Now drawing from 'AC Power'" in saved["outputs"]["power"]
    assert plistlib.loads(saved["outputs"]["battery"].encode())[0]["ExternalConnected"] is True
    assert plistlib.loads(saved["outputs"]["clamshell"].encode())[0]["AppleClamshellState"] is False
    return saved, raw.decode()


def host_interval(attempt: Path, result: dict) -> dict:
    pre, pre_log = host(attempt, "before")
    post, post_log = host(attempt, "after")
    start = iso(json.loads(read(attempt / "parent_start.json"))["utc"])
    exit_time = iso(result["worker_exit_observed_utc"])
    collected = iso(post["collected_utc"])
    assert start <= exit_time <= collected
    assert result["host_fence_end_utc"] == result["worker_exit_observed_utc"]
    assert result["host_probe_collected_utc"] == post["collected_utc"]
    assert abs((collected-exit_time).total_seconds()-result["host_probe_lag_seconds"]) < 1e-5
    assert 0 <= (collected-exit_time).total_seconds() <= 120
    for log, window_start, end, sampled in (
        (pre_log, iso(pre["collected_utc"])-timedelta(minutes=10),
         iso(pre["collected_utc"]), iso(pre["collected_utc"])),
        (post_log, start, collected, collected),
    ):
        events, terminals = list(EVENT.finditer(log)), list(TERMINAL.finditer(log))
        assert events and terminals
        assert min(event_time(m) for m in events) <= window_start
        # pmset emits whole-second stamps; collection uses fractional seconds.
        terminal = event_time(terminals[-1])
        assert end-timedelta(seconds=1) <= terminal <= sampled+timedelta(seconds=1)
        assert terminal >= sampled-timedelta(seconds=5)
        assert not any(window_start <= event_time(m) <= end and
                       (m.group(4).strip() in ("Sleep", "DarkWake", "Wake")
                        or "Thermal" in m.group(4)) for m in events)
    return {"start": start.isoformat(), "worker_exit": exit_time.isoformat(),
            "post_collected": collected.isoformat(), "lag_seconds": (collected-exit_time).total_seconds()}


def rendered() -> tuple[dict, bytes]:
    data = read(RENDER / "manifest.json")
    assert sha(data) == MANIFEST_SHA
    manifest = json.loads(data)
    expected = {row["case_id"]: row for row in manifest["rows"]}
    assert len(expected) == 20
    for row in expected.values():
        wav = RENDER / row["wav"]
        assert sha(read(wav)) == row["wav_sha256"]
        with wave.open(str(wav), "rb") as sound:
            assert (sound.getnchannels(), sound.getsampwidth(), sound.getframerate(),
                    sound.getnframes(), sound.getcomptype()) == (
                        1, 2, 16000, row["frames"], "NONE")
            assert sha(sound.readframes(sound.getnframes())) == row["pcm_sha256"]
    return expected, data


def snapshot(model: str) -> Path:
    pin = json.loads(read(HERE / "descriptive_pins_v1.json"))["models"][model]
    repo = pin["repo"].replace("/", "--")
    path = Path.home() / ".cache/huggingface/hub" / f"models--{repo}" / "snapshots" / pin["revision"]
    assert path.is_dir()
    for name, expected in pin["files_sha256"].items():
        item = path / name
        assert item.is_file() and item.resolve().is_relative_to(path.parents[1]), name
        assert sha(item.read_bytes()) == expected, name
    return path


def ctc_frames(samples: int) -> int:
    result = samples
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2), strict=True):
        result = (result-kernel)//stride+1
    return result


def audit(model: str, expected: dict, manifest_data: bytes) -> dict:
    info = MODEL[model]
    attempt = RUNS / f"descriptive_{model}_20261119_v2"
    result = json.loads(read(attempt / "parent_result.json"))
    assert result["status"] == "worker_exited_pending_audit" and result["exit_code"] == 0
    assert result["host_after_ok"] is True and result["limit_reason"] is None
    assert result["model"] == model and result["peak_monitored_rss_bytes"] < 4*1024**3
    assert result["elapsed_seconds"] <= 1800 and result["evidence_bytes"] <= 128*1024**2
    prior = info["prior"]
    prior_result = json.loads(read(prior / "parent_result.json"))
    assert prior_result["status"] == "invalid_or_partial"
    assert result["prior_failed_sha256"] == {
        "parent_result.json": sha(read(prior / "parent_result.json")),
        "records.jsonl": sha(read(prior / "records.jsonl"))}
    receipt, review = read(info["receipt"]), read(info["review"])
    receipt_obj = json.loads(receipt)
    assert result["preflight_sha256"] == sha(receipt)
    assert result["review_sha256"] == sha(review)
    host_times = host_interval(attempt, result)
    records_bytes = read(attempt / "records.jsonl")
    assert sha(records_bytes) == info["records_sha"] and records_bytes.endswith(b"\n")
    records = [json.loads(line) for line in records_bytes.splitlines()]
    assert len(records) == 42
    assert Counter(r["type"] for r in records) == {
        "start": 1, "case": 22, "silence_difference": 18, "complete": 1}
    assert records[0]["model"] == model and records[0]["manifest_sha256"] == sha(manifest_data)
    assert records[0]["source_identity"] == receipt_obj["source_identity"]
    assert records[0]["installed"] == receipt_obj["installed"]
    assert records[0]["assets"] == receipt_obj["model_assets"]
    assert records[0]["cases"] == sorted(expected)
    assert records[-1]["status"] == "complete"
    cases = [r for r in records if r["type"] == "case"]
    assert [r["case_id"] for r in cases] == sorted(expected) + ["d02_silence", "d08_silence"]
    assert [r["case_id"] for r in cases if r["repeat"]] == ["d02_silence", "d08_silence"]
    assert all(r["status"] == "complete" for r in cases)
    differences = [r for r in records if r["type"] == "silence_difference"]
    assert [r["case_id"] for r in differences] == [
        name for name in sorted(expected) if not name.endswith("silence")]
    assert all(r["donor_case_id"] == r["case_id"].split("_")[0]+"_silence" for r in differences)
    case_by_id = {r["case_id"]: r for r in cases[:20]}
    for difference in differences:
        same_case = case_by_id[difference["case_id"]]
        assert set(difference["differences"]) == set(same_case["traces"])
        for phase, stages in difference["differences"].items():
            assert set(stages) == set(same_case["traces"][phase])
            for stage, metric in stages.items():
                curve = metric["row_l2"]
                assert len(curve) == len(same_case["traces"][phase][stage]["row_l2"])
                assert all(math.isfinite(v) and v >= 0 for v in curve)
                assert metric["nonzero_rows"] == sum(v > 0 for v in curve)
                assert math.isclose(metric["whole_l2"], math.sqrt(sum(v*v for v in curve)),
                                    rel_tol=1e-8, abs_tol=1e-8)
    for row in cases:
        source = expected[row["case_id"]]
        assert row["frames"] == source["frames"]
        assert row["wav_sha256"] == source["wav_sha256"]
        assert row["pcm_sha256"] == source["pcm_sha256"]
        assert row["output"]["unicode_scripts"] == scripts(row["output"]["transcript"])
        assert row["output"]["lexical_nonempty"] is lexical(row["output"]["transcript"])
        phases = ("generate",) if model == "whisper_english" else ("ctc",)
        assert set(row["traces"]) == set(phases)
        for phase, stages in row["traces"].items():
            expected_stages = (
                "mel", "post_conv_pos", "post_block0", "final_encoder"
            ) if model == "whisper_english" else (
                "input_values", "first_conv_raw", "first_conv_norm",
                "feature_projection", "post_block0", "final_encoder", "logits")
            assert set(stages) == set(expected_stages)
            for stage, trace in stages.items():
                if model == "whisper_english":
                    shape, axis = ([1, 80, 3000], 2) if stage == "mel" else ([1, 1500, 384], 1)
                elif stage == "input_values":
                    shape, axis = [1, row["frames"]], 1
                elif stage in ("first_conv_raw", "first_conv_norm"):
                    shape, axis = [1, 512, (row["frames"]-10)//5+1], 2
                elif stage == "logits":
                    shape, axis = [1, ctc_frames(row["frames"]), 32], 1
                else:
                    shape, axis = [1, ctc_frames(row["frames"]), 768], 1
                assert trace["shape"] == shape and trace["dtype"] == "float32"
                assert trace["time_axis"] == axis and len(trace["row_l2"]) == shape[axis]
                assert all(math.isfinite(v) and v >= 0 for v in trace["row_l2"])
    for original, repeat in zip((next(r for r in cases if r["case_id"] == s and not r["repeat"])
                                  for s in ("d02_silence", "d08_silence")), cases[-2:], strict=True):
        assert original["output"] == repeat["output"]
        assert original["traces"] == repeat["traces"]
        assert original["capture_counts"] == repeat["capture_counts"]

    if model == "whisper_english":
        tokenizer = WhisperTokenizerFast.from_pretrained(snapshot(model), local_files_only=True)
        generation = json.loads((snapshot(model) / "generation_config.json").read_bytes())
        eos = generation["eos_token_id"]
        start = generation["decoder_start_token_id"]
        no_time = generation["no_timestamps_token_id"]
        assert eos == tokenizer.eos_token_id == 50256
        for row in cases:
            out = row["output"]
            ids, prefix, generated = out["sequence_ids"], out["prefix_ids"], out["generated_ids"]
            assert prefix == [start, no_time] and ids == prefix+generated
            assert 1 <= len(generated) == out["actual_generation_steps"] == len(out["score_rows"]) <= 128
            assert out["language_detection"] is None
            assert tokenizer.decode(ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False) == out["text_with_special_tokens"]
            assert tokenizer.decode(generated, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False) == out["transcript"]
            eos_seen = eos in generated
            assert out["eos_seen"] is eos_seen and (not eos_seen or generated[-1] == eos)
            assert out["censored"] is (not eos_seen and len(generated) == 128)
            assert out["termination"] == ("eos" if eos_seen else
                                          "length_censored" if len(generated) == 128 else "incomplete_unknown")
            assert [r["token_id"] for r in out["score_rows"]] == generated
            raw = out["first_raw_logits"]
            logits = np.asarray(raw["values"], dtype=np.float32)
            assert logits.shape == (len(tokenizer),) and np.isfinite(logits).all()
            assert sha(logits.tobytes()) == raw["sha256"] == out["score_rows"][0]["raw_logits_sha256"]
            assert int(logits.argmax()) == out["score_rows"][0]["raw_top_id"]
            assert float(logits[generated[0]]) == out["score_rows"][0]["raw_selected_logit"]
            state = out["first_decoder_step"]
            values = np.asarray(state["values"], dtype=np.float32)
            assert values.shape == (1, len(prefix), 384) and np.isfinite(values).all()
            assert sha(values.tobytes()) == state["sha256"]
            geometry = row["input_geometry"]
            active = (row["frames"]+159)//160
            mask = [1]*active+[0]*(3000-active)
            assert geometry["attention_mask"] == mask
            assert geometry["mask_sha256"] == sha(bytes(mask))
            assert geometry["nominal_active_encoder_rows"] == (active+1)//2
            assert row["capture_counts"] == {"generate": {
                "mel": 1, "post_conv_pos": 1, "post_block0": 1, "final_encoder": 1}}
    else:
        processor = Wav2Vec2Processor.from_pretrained(snapshot(model), local_files_only=True)
        blank = processor.tokenizer.pad_token_id
        vocabulary = json.loads((snapshot(model) / "vocab.json").read_bytes())
        token_by_id = {value: key for key, value in vocabulary.items()}
        for row in cases:
            out = row["output"]
            ids = out["frame_ids"]
            assert len(ids) == ctc_frames(row["frames"])
            assert all(type(v) is int and 0 <= v < 32 for v in ids)
            collapsed = []
            previous = None
            for value in ids:
                if value != previous and value != blank:
                    collapsed.append(value)
                previous = value
            assert out["blank_id"] == blank and out["collapsed_nonblank_ids"] == collapsed
            assert processor.batch_decode(np.asarray([ids]))[0] == out["transcript"]
            assert all(value not in (1, 2, 3) for value in collapsed)
            assert "".join(token_by_id[value] for value in collapsed).replace("|", " ").strip() == out["transcript"]
            assert out["termination"] == "complete_ctc_frames"
            geometry = row["input_geometry"]
            assert geometry == {"pcm_frames": row["frames"],
                                "active_input_samples": row["frames"],
                                "attention_mask_kind": "implicit_all_active_unpadded",
                                "ctc_frames": len(ids), "active_ctc_frames": len(ids)}
            assert row["capture_counts"] == {"ctc": {name: 1 for name in (
                "input_values", "first_conv_raw", "first_conv_norm",
                "feature_projection", "post_block0", "final_encoder", "logits")}}
    first = cases[:20]
    by_id = {r["case_id"]: r for r in first}
    changes = [r["case_id"] for r in first if not r["case_id"].endswith("silence")
               and r["output"]["transcript"] !=
               by_id[r["case_id"].split("_")[0]+"_silence"]["output"]["transcript"]]
    by_family = Counter(family(r["case_id"])
                        for r in first if r["output"]["lexical_nonempty"])
    return {"model": model, "records_sha256": sha(records_bytes), "cases": len(first),
            "repeats": 2, "silence_differences": len(differences),
            "lexical_nonempty": sum(r["output"]["lexical_nonempty"] for r in first),
            "silence": {s: by_id[s]["output"]["transcript"] for s in
                        ("d02_silence", "d08_silence")},
            "transcripts": {r["case_id"]: r["output"]["transcript"] for r in first},
            "different_from_silence": changes,
            "nonempty_by_family": dict(sorted(by_family.items())),
            "host": host_times,
            "limits": (["No raw encoder tensors or all-step processed-logit sidecars were saved."]
                       if model == "whisper_english" else
                       ["No raw encoder tensors or full CTC-logit sidecars were saved.",
                        "Saved CTC frame IDs can be decoded, but cannot be recomputed from logits."])}


def main() -> None:
    expected, manifest_data = rendered()
    result = {name: audit(name, expected, manifest_data) for name in MODEL}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
