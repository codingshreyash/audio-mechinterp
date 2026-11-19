"""Independent read-only check of the saved 328-call Whisper causal attempt.

This script does not construct or run the candidate model. It deliberately
reimplements the output, sidecar, intervention and call-census checks rather
than calling the worker's or primary auditor's result-check functions.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import plistlib
import re
import wave
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from transformers import WhisperProcessor, WhisperTokenizer
from transformers.models.whisper import feature_extraction_whisper, modeling_whisper

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "runs/causal_whisper_v1/attempt_002"
OUTPUT = ATTEMPT / "worker_output"
DESCRIPTIVE = HERE / "runs/descriptive_whisper_multilingual_20261119_v2/records.jsonl"
MANIFEST = HERE / "runs/render_20261119_0740Z/manifest.json"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--openai--whisper-tiny/snapshots"
            / "169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
A_ARMS = ("no_hook", "read_only", "self_detect", "self_transcribe", "self_both",
          "donor_detect", "donor_transcribe", "donor_both", "random_1_both",
          "random_2_both")
B_ARMS = ("no_hook", "read_only", "self_both", "donor_both", "active_both",
          "displaced_both", "random_1_both", "random_2_both")
PHASES = ("detect", "transcribe")
SITES = ("pre_block0", "final_encoder")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def must(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def wav_manifest() -> list[str]:
    must(file_sha(MANIFEST) ==
         "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90",
         "engineered manifest changed")
    rows = json.loads(MANIFEST.read_text())["rows"]
    must(len(rows) == 20 and len({row["case_id"] for row in rows}) == 20,
         "manifest case census")
    for row in rows:
        path = MANIFEST.parent / row["wav"]
        must(file_sha(path) == row["wav_sha256"], f"WAV changed: {row['case_id']}")
        with wave.open(str(path), "rb") as wav:
            must((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                  wav.getnframes()) == (1, 2, 16000, row["frames"]),
                 f"WAV geometry: {row['case_id']}")
            pcm = wav.readframes(row["frames"])
        must(sha(pcm) == row["pcm_sha256"], f"PCM changed: {row['case_id']}")
    return sorted(row["case_id"] for row in rows)


def expected_calls(cases: list[str]) -> list[tuple[str, str, str]]:
    calls = [("donor", f"d{duration:02d}_silence", arm)
             for duration in (2, 8) for arm in ("no_hook", "read_only")]
    for stage, arms in (("A", A_ARMS), ("B", B_ARMS)):
        calls.extend((stage, case, arm) for case in cases if "silence" not in case
                     for arm in arms)
    must(len(calls) == 328, "planned call count")
    return calls


def processor_padding(cases: list[str]) -> dict[str, dict]:
    """No-model check of nominally padded mel rows under the pinned processor."""
    must(file_sha(Path(feature_extraction_whisper.__file__)) ==
         "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb"
         and file_sha(Path(modeling_whisper.__file__)) ==
         "bf4f022d37ed523d09a472dae267b86007aaae0cd39d2ce37927de44bfc5b9e6",
         "Whisper processor/modeling source pin changed")
    processor = WhisperProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
    features = {}
    for case in cases:
        path = MANIFEST.parent / f"{case}.wav"
        with wave.open(str(path), "rb") as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        result = processor(pcm.astype(np.float32)/32768.0, sampling_rate=16000,
                           return_tensors="pt", return_attention_mask=True)
        active = int(result.attention_mask.sum())
        must(active == (200 if case.startswith("d02") else 800)
             and result.attention_mask[0].tolist() == [1]*active+[0]*(3000-active)
             and tuple(result.input_features.shape) == (1, 80, 3000),
             f"processor geometry differs: {case}")
        features[case] = (result.input_features.numpy(), active)
    result = {}
    for case in cases:
        if "silence" in case:
            continue
        feature, active = features[case]
        silence = features[f"{case[:3]}_silence"][0]
        difference = np.abs(feature-silence)[0]
        padded = difference[:, active:]
        result[case] = {
            "active_mel_frames": active,
            "padded_mel_frames": int(padded.shape[1]),
            "padded_frames_with_difference_gt_1e_6": int(
                (padded > 1e-6).any(axis=0).sum()),
            "padded_maxabs": float(padded.max()),
        }
    return result


def load_sidecar(ref: dict, seen: set[str]) -> np.ndarray:
    relative = Path(ref["path"])
    must(not relative.is_absolute() and ".." not in relative.parts
         and relative.suffix == ".npy" and ref["path"] not in seen,
         "duplicate/unsafe sidecar path")
    path = OUTPUT / relative
    must(path.is_file() and not path.is_symlink()
         and path.resolve().is_relative_to(OUTPUT.resolve())
         and path.stat().st_size == ref["bytes"]
         and file_sha(path) == ref["file_sha256"],
         f"sidecar file differs: {ref['path']}")
    array = np.load(path, allow_pickle=False)
    must(array.dtype == np.float32 and list(array.shape) == ref["shape"]
         and np.isfinite(array).all() and sha(np.ascontiguousarray(array).tobytes())
         == ref["tensor_sha256"], f"sidecar array differs: {ref['path']}")
    seen.add(ref["path"])
    return array


def host_and_progress(expected: list[tuple[str, str, str]]) -> dict:
    start = json.loads((ATTEMPT / "launch_start.json").read_text())
    result = json.loads((ATTEMPT / "launch_result.json").read_text())
    admission_path = HERE / "CAUSAL_WHISPER_ADMISSION_V2.json"
    go_path = HERE / "CAUSAL_WHISPER_EXECUTABLE_GO_V2.json"
    admission = json.loads(admission_path.read_text())
    go = json.loads(go_path.read_text())
    must(admission["status"] == "source_only_no_model_load"
         and admission["planned_calls"] == len(expected)
         and admission["schedule_sha256"] ==
         "a4a77079c9381758720712f7b5420186e0b08a18c8f3fbbf1a99beb4926d087b"
         and all(file_sha((HERE / name).resolve()) == digest
                 for name, digest in admission["sources"].items())
         and go["status"] == "GO"
         and go["admission_sha256"] == file_sha(admission_path)
         and go["sources"] == admission["sources"]
         and start["admission_sha256"] == result["admission_sha256"] == file_sha(admission_path)
         and start["go_sha256"] == result["go_sha256"] == file_sha(go_path),
         "admission/GO/source/run binding differs")
    names = [f"{stage}_{case}_{arm}" for stage, case, arm in expected]
    must(start["scheduled_call_names"] == names
         and result["status"] == "completed_pending_saved_audit"
         and result["worker_stop_reason"] == "normal_exit"
         and result["worker_returncode"] == 0
         and result["worker_call_census"]["completed_call_names"] == names
         and result["worker_call_census"]["failed_attempted_call_name"] is None
         and not result["worker_call_census"]["censored_call_names"],
         "parent run/census incomplete")
    progress = jsonl(OUTPUT / "call_progress.jsonl")
    must(len(progress) == 656, "progress row count")
    for index, name in enumerate(names, 1):
        must(progress[2*index-2] == {"type": "attempt", "call_index": index,
                                     "call_name": name}
             and progress[2*index-1] == {"type": "complete", "call_index": index,
                                        "call_name": name},
             f"progress differs: {name}")
    must(file_sha(OUTPUT / "call_progress.jsonl") == result["worker_call_progress_sha256"],
         "progress hash differs")
    rss_path = ATTEMPT / "worker_rss_samples.jsonl"
    samples = jsonl(rss_path)
    peak = max(row["tree_rss_bytes"] for row in samples)
    must(len(samples) == result["worker_rss_samples"]
         and all(row["sample_index"] == i for i, row in enumerate(samples, 1))
         and peak == result["worker_peak_tree_rss_bytes"] < 4*1024**3
         and file_sha(rss_path) == result["worker_rss_sample_log_sha256"]
         and result["worker_elapsed_seconds"] <= 1800
         and sum(path.stat().st_size for path in ATTEMPT.rglob("*") if path.is_file())
         <= 2*1024**3, "resource evidence differs")
    probes = {}
    terminals = []
    event_re = re.compile(r"^(\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d\s+[+-]\d{4})\s+([^\t]+)\t", re.M)
    terminal_re = re.compile(r"^(\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d\s+[+-]\d{4})\s+: Showing all currently held IOKit power assertions", re.M)
    fmt = "%Y-%m-%d %H:%M:%S %z"
    for label in ("pre", "post"):
        receipt_path = ATTEMPT / f"{label}_host.json"
        probe = json.loads(receipt_path.read_text())
        ref = probe["power_log_archive"]
        compressed = (ATTEMPT / ref["file"]).read_bytes()
        raw = gzip.decompress(compressed)
        must(sha(compressed) == ref["compressed_sha256"]
             and sha(raw) == ref["raw_sha256"]
             and len(raw) == ref["raw_bytes"]
             and "Now drawing from 'AC Power'" in probe["outputs"]["power"]
             and plistlib.loads(probe["outputs"]["battery"].encode())[0]
             .get("ExternalConnected") is True
             and plistlib.loads(probe["outputs"]["clamshell"].encode())[0]
             .get("AppleClamshellState") is False,
             f"host {label} hash/power")
        text = raw.decode()
        events = [(datetime.strptime(m.group(1), fmt), m.group(2).strip())
                  for m in event_re.finditer(text)]
        terminal = [datetime.strptime(m.group(1), fmt) for m in terminal_re.finditer(text)]
        must(events and terminal, f"host {label} log coverage absent")
        probes[label] = (probe, events)
        terminals.append(terminal[-1])
    started = datetime.fromisoformat(start["start_utc"])
    pre_time = datetime.fromisoformat(probes["pre"][0]["collected_utc"])
    exited = datetime.fromisoformat(result["worker_exit_observed_utc"])
    post_time = datetime.fromisoformat(probes["post"][0]["collected_utc"])
    must(started <= pre_time <= exited <= post_time
         and 0 <= (post_time-exited).total_seconds() <= 120
         and min(t for t, _ in probes["pre"][1]) <= pre_time-timedelta(minutes=10)
         and min(t for t, _ in probes["post"][1]) <= started
         and terminals[0] >= pre_time-timedelta(seconds=1)
         and terminals[1] >= post_time-timedelta(seconds=1)
         and not any(pre_time-timedelta(minutes=10) <= time <= pre_time and
                     (event in ("Sleep", "DarkWake", "Wake") or "Thermal" in event)
                     for time, event in probes["pre"][1])
         and not any(started <= time <= post_time and
                     (event in ("Sleep", "DarkWake", "Wake") or "Thermal" in event)
                     for time, event in probes["post"][1]),
         "host timing/events differ")
    must(result["pre_host_sha256"] == file_sha(ATTEMPT / "pre_host.json")
         and result["post_host_sha256"] == file_sha(ATTEMPT / "post_host.json"),
         "host receipt hashes differ")
    return {"rss_samples": len(samples), "sampled_peak_rss_bytes": peak,
            "worker_seconds": result["worker_elapsed_seconds"]}


def patched(native: np.ndarray, donor: np.ndarray, arm: str, phase: str,
            active_rows: int, seed: int | None) -> tuple[np.ndarray, list[int]]:
    out = native.copy()
    if arm.startswith("self_") or arm == "read_only" or (
        arm == "donor_detect" and phase != "detect") or (
        arm == "donor_transcribe" and phase != "transcribe"):
        if arm in ("self_detect", "self_transcribe") and phase not in arm:
            return out, [0, 0]
        return out, [0, 1500] if arm.startswith("self_") else [0, 0]
    if arm in ("donor_both", "donor_detect", "donor_transcribe"):
        return donor.copy(), [0, 1500]
    if arm == "active_both":
        out[:, :active_rows, :] = donor[:, :active_rows, :]
        return out, [0, active_rows]
    if arm == "displaced_both":
        out[:, active_rows:2*active_rows, :] = donor[:, active_rows:2*active_rows, :]
        return out, [active_rows, 2*active_rows]
    if arm in ("random_1_both", "random_2_both"):
        must(isinstance(seed, int), "random seed missing")
        noise = np.random.default_rng(seed).standard_normal(native.shape).astype(np.float32)
        norm = float(np.linalg.norm((donor-native).astype(np.float64).ravel()))
        unit = noise.astype(np.float64)/np.linalg.norm(noise.astype(np.float64).ravel())
        return (native+(unit*norm).astype(np.float32)).astype(np.float32), [0, 1500]
    must(False, f"unhandled arm: {arm}")


def main() -> dict:
    cases = wav_manifest()
    mel_padding = processor_padding(cases)
    expected = expected_calls(cases)
    host = host_and_progress(expected)
    tokenizer = WhisperTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    config = json.loads((SNAPSHOT / "generation_config.json").read_text())
    language = config["lang_to_id"]
    must(file_sha(DESCRIPTIVE) ==
         "588165986eadcfc7df04f6a8869de78ccd656f5b9be578fe41b3e1395abeec42",
         "audited descriptive source changed")
    descriptive_rows = jsonl(DESCRIPTIVE)
    descriptive_tokens = {row["case_id"]: row["output"]["sequence_ids"]
                          for row in descriptive_rows
                          if row.get("type") == "case" and not row.get("repeat")}
    must(set(descriptive_tokens) == set(cases), "descriptive baseline census")
    records_path = OUTPUT / "records.jsonl"
    records = jsonl(records_path)
    must(len(records) == 330 and records[0]["type"] == "start"
         and records[-1]["type"] == "complete"
         and records[-1]["planned_calls"] == 328,
         "worker record inventory")
    by = {}
    seen: set[str] = set()
    native: dict[tuple[str, str, str], np.ndarray] = {}
    full_raw: dict[tuple[str, str, str], list[np.ndarray]] = {}
    censored = []
    for index, ((stage, case, arm), row) in enumerate(zip(expected, records[1:-1], strict=True), 1):
        plan = row["plan_row"]
        key = (stage, case, arm)
        must(row["type"] == "forward" and row["call_index"] == index
             and row["call_name"] == f"{stage}_{case}_{arm}"
             and (plan["stage"], plan["case_id"], plan["arm"]) == key
             and plan["donor_id"] == f"{case[:3]}_silence"
             and plan["active_rows"] == (100 if case.startswith("d02") else 400),
             f"schedule row differs: {key}")
        output = row["output"]
        ids, prefix, generated = (output[field] for field in
                                  ("sequence_ids", "prefix_ids", "generated_ids"))
        must(ids == prefix+generated and prefix[0] == 50258
             and prefix[1] in language.values() and prefix[2:] == [50359, 50363]
             and 1 <= len(generated) <= 128
             and output["actual_generation_steps"] == len(generated)
             and output["eos_seen"] == (50257 in generated)
             and (50257 not in generated or generated[-1] == 50257)
             and output["censored"] == (len(generated) == 128 and 50257 not in generated)
             and output["termination"] == (
                 "eos" if generated[-1] == 50257 else
                 "length_censored" if len(generated) == 128 else "incomplete_unknown")
             and output["transcript"] == tokenizer.decode(generated, skip_special_tokens=True)
             and output["text_with_special_tokens"] == tokenizer.decode(ids, skip_special_tokens=False),
             f"complete token path differs: {key}")
        if output["censored"]:
            censored.append(row["call_name"])
        detection = output["language_detection"]
        if arm == "no_hook":
            must(detection is None and row["hook"] == {} and row["hook_counts"] == {},
                 f"no-hook evidence invalid: {key}")
        else:
            must(set(detection["language_logits"]) == set(language)
                 and detection["chosen_language_ids"] == [prefix[1]],
                 f"language logit map differs: {key}")
            top = min(language, key=lambda name: (-detection["language_logits"][name],
                                                  language[name]))
            must(language[top] == prefix[1], f"language argmax differs: {key}")
        refs = row["sidecars"]
        raw = load_sidecar(refs["first_raw"], seen)
        hidden = load_sidecar(refs["first_decoder_state"], seen)
        must(raw.shape == (51865,) and hidden.shape == (1, 4, 384)
             and row["first_raw_meta"]["sha256"] == sha(raw.tobytes())
             and row["first_decoder_meta"]["sha256"] == sha(hidden.tobytes())
             and row["first_raw_meta"]["shape"] == [51865]
             and row["first_decoder_meta"]["shape"] == [1, 4, 384],
             f"first decoder evidence differs: {key}")
        scores = output["score_rows"]
        must(len(scores) == len(generated)
             and all(score["token_id"] == token and score["top_id"] == token
                     for score, token in zip(scores, generated, strict=True))
             and scores[0]["raw_logits_sha256"] == sha(raw.tobytes())
             and scores[0]["raw_top_id"] == int(raw.argmax())
             and math.isclose(scores[0]["raw_selected_logit"], float(raw[generated[0]]),
                              rel_tol=0, abs_tol=1e-6),
             f"first raw/score evidence differs: {key}")
        all_raw = [load_sidecar(ref, seen) for ref in refs["full_raw"]]
        must(len(all_raw) == (len(generated) if plan["save_full_raw_logits_each_step"] else 0),
             f"full raw inventory differs: {key}")
        for step, vector in enumerate(all_raw):
            must(vector.shape == (51865,)
                 and scores[step]["raw_logits_sha256"] == sha(vector.tobytes())
                 and scores[step]["raw_top_id"] == int(vector.argmax())
                 and math.isclose(scores[step]["raw_selected_logit"],
                                  float(vector[generated[step]]), rel_tol=0, abs_tol=1e-6),
                 f"raw logit step differs: {key} step {step}")
        if all_raw:
            full_raw[key] = all_raw
        state_refs = row["native_state_refs"]
        if stage == "donor" and arm == "read_only" or stage == "A" and arm == "read_only":
            must(set(state_refs) == set(SITES), f"native state refs missing: {key}")
            for site in SITES:
                for phase in PHASES:
                    tensor = load_sidecar(state_refs[site][phase], seen)
                    must(tensor.shape == (1, 1500, 384), f"native shape: {key}")
                    native[(case, site, phase)] = tensor
        else:
            must(not state_refs, f"unexpected native capture: {key}")
        by[key] = row
    must(len(native) == 80 and len(full_raw) == 38
         and len(seen) == 812
         and {str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.npy")} == seen,
         "tensor evidence inventory differs")
    preblock_padding = {}
    for case in cases:
        if "beep_train" not in case:
            continue
        donor_id = f"{case[:3]}_silence"
        active = 100 if case.startswith("d02") else 400
        difference = np.max(np.abs(native[(case, "pre_block0", "transcribe")]
                                   - native[(donor_id, "pre_block0", "transcribe")]),
                            axis=(0, 2))
        preblock_padding[case] = {
            "padded_encoder_rows_with_difference_gt_1e_6": int(
                (difference[active:] > 1e-6).sum()),
            "padded_encoder_rows": 1500-active,
            "padded_maxabs": float(difference[active:].max()),
        }
    must(file_sha(records_path) ==
         "dbacb49bcb1e158b312ee2e76235c2955b28a75dc05b8af2566826b8dd65753f",
         "frozen records hash differs")
    arm_counts: dict[str, dict[str, int]] = {stage: {} for stage in ("A", "B")}
    informative = [case for case in cases if "beep_train" in case]
    donor_transfer = {"A": Counter(), "B": Counter()}
    parity_errors = []
    detection_shift = []
    first_raw_detect_vs_recipient = []
    first_raw_transcribe_vs_donor = []
    for case in cases:
        if "silence" in case:
            must(by[("donor", case, "no_hook")]["output"]["sequence_ids"]
                 == descriptive_tokens[case]
                 and by[("donor", case, "read_only")]["output"]["sequence_ids"]
                 == descriptive_tokens[case], f"silence descriptive parity: {case}")
        else:
            for stage in ("A", "B"):
                must(by[(stage, case, "no_hook")]["output"]["sequence_ids"]
                     == descriptive_tokens[case],
                     f"fresh no-hook versus descriptive: {stage}:{case}")
    for stage, case, arm in expected:
        row = by[(stage, case, arm)]
        if arm == "no_hook":
            continue
        hooks = row["hook"]
        must(set(hooks) == set(PHASES), f"hook phases differ: {stage,case,arm}")
        selected_site = "final_encoder" if stage == "A" else "pre_block0"
        for phase in PHASES:
            must(set(hooks[phase]) == set(SITES), "hook site set differs")
            for site in SITES:
                item = hooks[phase][site]
                must(item["hook_calls"] == 1 and
                     (site == selected_site and stage != "donor" or
                      item["edited_sha256"] == item["native_sha256"]),
                     f"hook count/nonselected edit differs: {stage,case,arm,phase,site}")
                if stage == "donor" or arm == "read_only":
                    ref = native[(case, site, phase)]
                    must(item["native_sha256"] == sha(ref.tobytes())
                         and item["edited_sha256"] == sha(ref.tobytes()),
                         f"native hook differs: {stage,case,arm,phase,site}")
                if stage in ("A", "B") and site == selected_site:
                    ref = native[(case, site, phase)]
                    donor = native[(row["plan_row"]["donor_id"], site, phase)]
                    seed = row["plan_row"].get("seeds", {}).get(phase)
                    edited, support = patched(ref, donor, arm, phase,
                                              row["plan_row"]["active_rows"], seed)
                    edit_l2 = float(np.linalg.norm((edited-ref).astype(np.float64).ravel()))
                    donor_l2 = float(np.linalg.norm((donor-ref).astype(np.float64).ravel()))
                    must(item["native_sha256"] == sha(ref.tobytes())
                         and item["edited_sha256"] == sha(edited.tobytes())
                         and item["support_rows"] == support
                         and item["seed"] == (seed if arm.startswith("random") else None)
                         and math.isclose(item["edit_l2"], edit_l2,
                                          rel_tol=1e-5, abs_tol=1e-6)
                         and math.isclose(item["donor_delta_l2"], donor_l2,
                                          rel_tol=1e-5, abs_tol=1e-6),
                         f"edited tensor differs: {stage,case,arm,phase,site}")
        if stage == "donor":
            continue
        recipient = by[(stage, case, "no_hook")]["output"]["sequence_ids"]
        donor_ids = by[("donor", row["plan_row"]["donor_id"], "no_hook")]["output"]["sequence_ids"]
        ids = row["output"]["sequence_ids"]
        if arm in ("read_only", "self_detect", "self_transcribe", "self_both") and ids != recipient:
            parity_errors.append(f"self/read_only:{stage}:{case}:{arm}")
        if arm == "donor_both":
            if ids != donor_ids:
                parity_errors.append(f"donor:{stage}:{case}")
            left = full_raw[(stage, case, arm)]
            right = full_raw[("donor", row["plan_row"]["donor_id"], "no_hook")]
            if len(left) != len(right) or any(np.max(np.abs(a-b)) > 1e-5
                                              for a, b in zip(left, right, strict=True)):
                parity_errors.append(f"donor_raw:{stage}:{case}")
        counts = arm_counts[stage].setdefault(arm, {"changed_full_sequence": 0,
                                                    "matched_donor": 0,
                                                    "censored": 0})
        counts["changed_full_sequence"] += int(ids != recipient)
        counts["matched_donor"] += int(ids == donor_ids)
        counts["censored"] += int(row["output"]["censored"])
        if case in informative:
            donor_transfer[stage][arm] += int(ids == donor_ids and not row["output"]["censored"])
        if stage == "A" and arm == "donor_detect":
            a = row["output"]["language_detection"]["language_logits"]
            b = by[(stage, case, "read_only")]["output"]["language_detection"]["language_logits"]
            donor_scores = by[("donor", row["plan_row"]["donor_id"], "read_only")]["output"]["language_detection"]["language_logits"]
            must(all(a[k] == donor_scores[k] for k in a),
                 f"detection scores fail full-donor equality: {case}")
            detection_shift.append({"case": case,
                                    "maxabs": max(abs(a[k]-b[k]) for k in a),
                                    "english_delta": a["<|en|>"]-b["<|en|>"]})
            own_raw = np.load(OUTPUT / row["sidecars"]["first_raw"]["path"],
                              allow_pickle=False)
            recipient_raw = np.load(OUTPUT / by[(stage, case, "no_hook")]["sidecars"]["first_raw"]["path"],
                                    allow_pickle=False)
            first_raw_detect_vs_recipient.append(float(np.max(np.abs(own_raw-recipient_raw))))
        if stage == "A" and arm == "donor_transcribe":
            own_raw = np.load(OUTPUT / row["sidecars"]["first_raw"]["path"],
                              allow_pickle=False)
            donor_raw = np.load(OUTPUT / by[("donor", row["plan_row"]["donor_id"], "no_hook")]["sidecars"]["first_raw"]["path"],
                                allow_pickle=False)
            first_raw_transcribe_vs_donor.append(float(np.max(np.abs(own_raw-donor_raw))))
    must(not parity_errors, f"parity failed: {parity_errors}")
    must(len(detection_shift) == len(first_raw_detect_vs_recipient)
         == len(first_raw_transcribe_vs_donor) == 18
         and all(row["maxabs"] > 0 for row in detection_shift)
         and max(first_raw_detect_vs_recipient) == 0
         and max(first_raw_transcribe_vs_donor) == 0,
         "phase-specific detection/transcription evidence differs")
    old = json.loads((HERE / "runs/causal_whisper_v1/attempt_001/launch_result.json").read_text())
    must(old["status"] == "invalid_or_partial" and old["worker_call_census"]["completed_call_names"] == [],
         "first invalid attempt reinterpreted")
    return {"schema": "independent_nonspeech_causal_v2_result_check_20261119",
            "status": "GO_SAVED_EVIDENCE_WITH_CENSORING",
            "records_sha256": file_sha(records_path),
            "sidecars": len(seen), "native_tensors": len(native),
            "call_count": len(expected), "censored_calls": censored,
            "parity_errors": parity_errors, "host": host,
            "arm_counts": arm_counts,
            "informative_cases": informative,
            "informative_exact_donor_matches": {stage: dict(counts)
                                                for stage, counts in donor_transfer.items()},
            "donor_detect_language_logit_shift": detection_shift,
            "donor_detect_first_raw_vs_recipient_maxabs": max(first_raw_detect_vs_recipient),
            "donor_transcribe_first_raw_vs_donor_maxabs": max(first_raw_transcribe_vs_donor),
            "mel_padding": mel_padding,
            "beep_preblock0_padding": preblock_padding}


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":"), allow_nan=False))
