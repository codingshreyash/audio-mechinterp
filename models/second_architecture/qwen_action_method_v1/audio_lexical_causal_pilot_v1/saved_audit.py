"""Weight-free, fail-closed audit of the one-shot Qwen lexical causal pilot.

The complete audit runs in the admitted Dallas environment. It reconstructs
processor tensors there; the Mac processor receipt is only a source/control
record because CPU feature bytes differ between platforms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import gate, parent, plan, processor_check, worker


def load_sidecar(folder: Path, meta: dict, name: str) -> np.ndarray:
    gate.require(isinstance(meta, dict) and meta.get("path") == name
                 and Path(name).name == name and name.endswith(".npy"),
                 "sidecar name invalid")
    path = folder / name
    gate.require(path.is_file() and not path.is_symlink()
                 and plan.sha(path) == meta.get("sha256"),
                 "sidecar absent or hash invalid")
    value = np.load(path, allow_pickle=False)
    gate.require(value.size > 0 and list(value.shape) == meta.get("shape")
                 and str(value.dtype) == meta.get("dtype")
                 and np.issubdtype(value.dtype, np.number)
                 and np.isfinite(value).all(), "sidecar data invalid")
    return value


def read_jsonl(path: Path) -> list[dict]:
    gate.require(path.is_file() and not path.is_symlink()
                 and path.stat().st_size < plan.EVIDENCE_BYTES,
                 "JSONL evidence absent or oversized")
    rows = []
    for line in path.read_text().splitlines():
        gate.require(bool(line.strip()), "empty JSONL line")
        def unique(pairs: list[tuple[str, Any]]) -> dict:
            result = {}
            for key, value in pairs:
                gate.require(key not in result, "duplicate JSONL key")
                result[key] = value
            return result
        row = json.loads(line, object_pairs_hook=unique)
        gate.require(isinstance(row, dict), "JSONL row not object")
        rows.append(row)
    return rows


def valid_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def fp32_roundoff_bound(*values: np.ndarray) -> float:
    """Predeclared conservative add/subtract cancellation bound in float32."""
    scale = max(1.0, *(float(np.max(np.abs(x))) for x in values))
    return max(1e-5, 16 * np.finfo(np.float32).eps * scale)


def verify_heartbeat_boundary(result: dict, last_sample_elapsed: float) -> None:
    marker = result.get("heartbeat_clean_exit_boundary")
    observed = result.get("heartbeat_boundary_elapsed_seconds")
    gate.require(type(marker) is bool, "closed-heartbeat marker missing")
    if marker:
        gate.require(valid_number(observed)
                     and 0 <= observed - last_sample_elapsed <= 2.0
                     and observed <= result["worker_elapsed_seconds"]
                     and result["worker_returncode"] == 0
                     and result["worker_stop_reason"] == "normal_exit",
                     "closed-heartbeat timing invalid")
    else:
        gate.require(observed is None, "unexpected closed-heartbeat time")


def verify_parent(attempt: Path, go: dict) -> tuple[dict, dict, dict]:
    gate.require(attempt.is_dir() and not attempt.is_symlink()
                 and parent.output_bytes(attempt) < plan.EVIDENCE_BYTES,
                 "fixed attempt missing or exceeds evidence cap")
    expected = {"pre_host.json", "start.json", "post_host.json", "result.json",
                "terminal.json", "worker_stdout.txt", "worker_stderr.txt",
                "worker_resource_samples.jsonl", "worker_output"}
    gate.require({p.name for p in attempt.iterdir()} == expected,
                 "parent evidence inventory changed")
    start, result, terminal = (gate.read(attempt / name) for name in
                               ("start.json", "result.json", "terminal.json"))
    pre, post = (gate.read(attempt / name) for name in
                 ("pre_host.json", "post_host.json"))
    worker_start = gate.read(attempt / "worker_output/start.json")
    gate.require(start.get("schema") == result.get("schema") == parent.SCHEMA
                 and start.get("status") == "worker_authorized"
                 and result.get("status") == terminal.get("status")
                    == "candidate_saved_pending_audit"
                 and terminal.get("schema") == "qwen_audio_lexical_terminal_v1"
                 and worker_start.get("schema") == worker.SCHEMA
                 and worker_start.get("status") == "started"
                 and start.get("attempt") == str(attempt.resolve())
                 and start.get("go_sha256") == result.get("go_sha256")
                    == worker_start.get("go_sha256") == plan.sha(plan.GO)
                 and start.get("source_identity") == go["source_identity"]
                 and start.get("runtime_identity") == pre.get("runtime_identity")
                    == post.get("runtime_identity") == go["runtime_identity"]
                 and start.get("pre_host_sha256") == result.get("pre_host_sha256")
                    == plan.sha(attempt / "pre_host.json")
                 and result.get("post_host_sha256") == plan.sha(attempt / "post_host.json")
                 and worker_start.get("parent_start_sha256")
                    == terminal.get("start_sha256") == plan.sha(attempt / "start.json")
                 and terminal.get("result_sha256") == plan.sha(attempt / "result.json")
                 and worker_start.get("parent_pid") == start.get("parent_pid"),
                 "parent, worker, and GO receipt chain invalid")
    baseline = go["gpu_baseline"]
    for host in (pre, post):
        gpu = host.get("gpu_inventory", {})
        gate.require(all(gpu.get(k) == baseline.get(k) for k in
                         ("gpu_uuid", "gpu_name", "gpu_memory_mib", "mig_mode",
                          "compute_processes", "idle_gpu_limit_bytes"))
                     and type(gpu.get("gpu_used_bytes")) is int
                     and 0 <= gpu["gpu_used_bytes"] < gate.IDLE_GPU_BYTES,
                     "parent GPU isolation changed")
    began = datetime.fromisoformat(result["start_utc"])
    exited = datetime.fromisoformat(result["worker_exit_utc"])
    ended = datetime.fromisoformat(result["end_utc"])
    gate.require(began <= exited <= ended
                 and result["start_utc"] == start["start_utc"]
                 and valid_number(result.get("elapsed_seconds_before_result"))
                 and 0 < result["elapsed_seconds_before_result"] < plan.MAX_SECONDS
                 and result.get("worker_returncode") == 0
                 and result.get("worker_stop_reason") == "normal_exit"
                 and result.get("worker_monitor_error") is None
                 and valid_number(result.get("worker_elapsed_seconds"))
                 and 0 < result["worker_elapsed_seconds"]
                    <= result["worker_seconds_granted"]
                 and type(result.get("worker_resource_samples")) is int
                 and result["worker_resource_samples"] > 0
                 and type(result.get("worker_parent_heartbeats")) is int
                 and result["worker_parent_heartbeats"] > 0,
                 "parent completion invalid")
    path = attempt / "worker_resource_samples.jsonl"
    rows = read_jsonl(path)
    gate.require(len(rows) == result["worker_resource_samples"]
                 and plan.sha(path) == result["worker_resource_log_sha256"]
                 and [r.get("sample_index") for r in rows]
                    == list(range(1, len(rows) + 1))
                 and all(valid_number(r.get("elapsed_seconds"))
                         and r["elapsed_seconds"] >= 0 for r in rows)
                 and all(a["elapsed_seconds"] < b["elapsed_seconds"]
                         for a, b in zip(rows, rows[1:]))
                 and all(type(r.get("tree_rss_bytes")) is int
                         and type(r.get("gpu_used_bytes")) is int
                         and type(r.get("evidence_bytes")) is int
                         and 0 <= r["tree_rss_bytes"] < plan.RSS_BYTES
                         and 0 <= r["gpu_used_bytes"] < plan.GPU_BYTES
                         and 0 <= r["evidence_bytes"] < plan.EVIDENCE_BYTES
                         for r in rows)
                 and max(r["tree_rss_bytes"] for r in rows)
                    == result["worker_peak_tree_rss_bytes"]
                 and max(r["gpu_used_bytes"] for r in rows)
                    == result["worker_peak_gpu_used_bytes"],
                 "resource sampling invalid")
    verify_heartbeat_boundary(result, rows[-1]["elapsed_seconds"])
    return start, result, worker_start


def verify_cuda(folder: Path, payload: dict, baseline_pass: bool) -> tuple[list[dict], int]:
    path = folder / "cuda_samples.jsonl"
    rows = read_jsonl(path)
    expected = ["after_model_load"]
    expected += [f"calibrated_baseline_{w}" for w in plan.WORDS]
    expected += [f"original_observation_{w}" for w in plan.WORDS]
    if baseline_pass:
        expected += [f"read_only_{w}" for w in plan.WORDS]
        expected += [f"intervention_{si}_{di}_{arm}"
                     for si, _ in enumerate(plan.SITES)
                     for di, _ in enumerate(plan.WORDS) for arm in plan.ARMS]
    gate.require(plan.sha(path) == payload.get("cuda_log_sha256")
                 and [r.get("phase") for r in rows] == expected
                 and all(type(r.get("allocated_bytes")) is int
                         and type(r.get("reserved_bytes")) is int
                         and type(r.get("peak_reserved_bytes")) is int
                         and 0 <= r["allocated_bytes"] <= r["reserved_bytes"]
                         <= r["peak_reserved_bytes"] < plan.GPU_BYTES
                         for r in rows), "framework GPU evidence invalid")
    return rows, max(r["peak_reserved_bytes"] for r in rows)


def verify_processor(folder: Path, payload: dict, admission: dict) -> tuple[Any, set[str]]:
    from transformers import Qwen2_5OmniProcessor  # Processor only; no weights.
    from models.qwen_omni.adapter import prepare_audio_batch
    from ..audio_candidate_v1 import validate_audio_batch

    snapshot = Path(admission["processor_preflight"]["snapshot"]["snapshot_path"])
    runtime_receipt = admission.get("processor_runtime_receipt")
    gate.require(isinstance(runtime_receipt, dict)
                 and processor_check.capture(snapshot) == runtime_receipt,
                 "Dallas runtime processor receipt changed")
    gate.require(payload.get("processor_records") == runtime_receipt.get("records")
                 and set(payload.get("batch_sidecars", {})) == {"calibrated", "original"},
                 "saved processor records differ from Dallas admission")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True)
    names: set[str] = set()
    for family, paths in (("calibrated", plan.CALIBRATED),
                          ("original", plan.ORIGINAL)):
        gate.require(set(payload["batch_sidecars"][family]) == set(plan.WORDS),
                     "processor sidecar family invalid")
        for word, wav in paths.items():
            batch, digest = prepare_audio_batch(processor, wav,
                                                 system_prompt=plan.SYSTEM_PROMPT)
            gate.require(digest == plan.EXPECTED_WAV_SHA256[family][word],
                         "WAV bytes differ")
            record = validate_audio_batch(batch, processor,
                wav_sha256=digest, case_id=f"LEXICAL_{family.upper()}_{word.upper()}")
            gate.require(record == runtime_receipt["records"][family][word]
                         and set(payload["batch_sidecars"][family][word]) == set(batch),
                         "runtime processor batch record differs")
            for key, tensor in batch.items():
                name = f"batch_{family}_{word}_{key}.npy"
                sidecar = load_sidecar(folder,
                    payload["batch_sidecars"][family][word][key], name)
                gate.require(np.array_equal(sidecar, tensor.cpu().numpy()),
                             "saved processor tensor differs")
                names.add(name)
    return processor.tokenizer, names


def verify_output(row: dict, tokenizer: Any) -> None:
    gate.require(isinstance(row, dict) and isinstance(row.get("ids"), list),
                 "generated output absent")
    expected = worker.classify(row["ids"], tokenizer)
    gate.require(row == expected, "raw complete generated path/decoding differs")


def verify_reference(folder: Path, payload: dict, tokenizer: Any) -> tuple[dict, set[str]]:
    reference = payload.get("reference")
    gate.require(isinstance(reference, dict)
                 and set(reference) == {"references", "outside_source_difference_maxabs"}
                 and set(reference["references"]) == set(plan.WORDS),
                 "read-only reference absent")
    expected_names: set[str] = set()
    states: dict[str, dict[str, np.ndarray]] = {}
    for word in plan.WORDS:
        row = reference["references"][word]
        gate.require(set(row) == {"first_logits_maxabs", "sidecars",
                                  "read_only_output", "read_only_calls",
                                  "source_drift_maxabs"},
                     "read-only row malformed")
        sc = row["sidecars"]
        gate.require(set(sc) == {"stock_logits", "read_only_logits", "states"}
                     and set(sc["states"]) == set(plan.SITES),
                     "reference sidecar census invalid")
        stock_name, read_name = f"{word}_stock_logits.npy", f"{word}_read_logits.npy"
        stock = load_sidecar(folder, sc["stock_logits"], stock_name)
        read = load_sidecar(folder, sc["read_only_logits"], read_name)
        expected_names.update((stock_name, read_name))
        gate.require(stock.shape == read.shape and stock.ndim == 1
                     and stock.size > plan.EOS_ID
                     and stock.dtype == read.dtype == np.float32,
                     "full first-step vocabulary logits invalid")
        delta = float(np.max(np.abs(stock-read)))
        gate.require(valid_number(row["first_logits_maxabs"])
                     and abs(delta-row["first_logits_maxabs"]) <= 1e-7
                     and delta <= plan.FIRST_LOGIT_TOLERANCE,
                     "stock/read-only first-logit parity invalid")
        gate.require(row["read_only_calls"] == {site: 1 for site in plan.SITES}
                     and set(row["source_drift_maxabs"]) == set(plan.SITES)
                     and all(valid_number(x) and 0 <= x <= plan.SOURCE_DRIFT_TOLERANCE
                             for x in row["source_drift_maxabs"].values()),
                     "read-only audio hook call/drift invalid")
        verify_output(row["read_only_output"], tokenizer)
        gate.require(row["read_only_output"]["ids"]
                     == payload["baselines"]["calibrated"][word]["ids"],
                     "read-only complete path changed")
        states[word] = {}
        for site in plan.SITES:
            name = f"{word}_{site.replace('.', '_')}_state.npy"
            state = load_sidecar(folder, sc["states"][site], name)
            gate.require(state.shape == (plan.WINDOWS[site]["total"],
                                         plan.WINDOWS[site]["width"])
                         and state.dtype == np.float32,
                         "reference audio state shape invalid")
            states[word][site] = state
            expected_names.add(name)
    outside = reference["outside_source_difference_maxabs"]
    gate.require(set(outside) == set(plan.SITES), "outside-source census invalid")
    pre = states["red"]["audio.pre_attention.0"] - states["blue"]["audio.pre_attention.0"]
    projected = states["red"]["audio.to_language"] - states["blue"]["audio.to_language"]
    lo, hi = plan.WINDOWS["audio.pre_attention.0"]["local"]
    actual = {"audio.pre_attention.0": float(max(np.abs(pre[:lo]).max(),
                                                 np.abs(pre[hi:]).max())),
              "audio.to_language": float(np.abs(projected[50:]).max())}
    gate.require(all(valid_number(outside[site])
                     and abs(outside[site]-actual[site]) <= 1e-6
                     and outside[site] <= plan.OUTSIDE_SUPPORT_TOLERANCE
                     for site in plan.SITES),
                 "reference source spread differs from saved states")
    return states, expected_names


def verify_interventions(folder: Path, payload: dict, states: dict,
                         tokenizer: Any) -> tuple[list[dict], set[str]]:
    rows = payload.get("interventions")
    expected = [(site, recipient, "blue" if recipient == "red" else "red", arm,
                 plan.NOISE_SEED + si*100 + di*10)
                for si, site in enumerate(plan.SITES)
                for di, recipient in enumerate(plan.WORDS) for arm in plan.ARMS]
    gate.require(isinstance(rows, list) and len(rows) == len(expected),
                 "intervention schedule incomplete")
    names: set[str] = set()
    nonces: set[str] = set()
    for index, (row, (site, recipient, donor, arm, seed)) in enumerate(zip(rows, expected)):
        site_index = index // (len(plan.WORDS) * len(plan.ARMS))
        direction_index = (index // len(plan.ARMS)) % len(plan.WORDS)
        arm_id = f"{site_index}_{direction_index}_{arm}"
        nonce = row.get("call_nonce") if isinstance(row, dict) else None
        gate.require(isinstance(row, dict)
                     and (row.get("site"), row.get("recipient"), row.get("donor"),
                          row.get("arm")) == (site, recipient, donor, arm)
                     and row.get("arm_id") == arm_id
                     and isinstance(nonce, str)
                     and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None
                     and nonce not in nonces
                     and isinstance(row.get("prepatch_state_sha256"), str)
                     and re.fullmatch(r"[0-9a-f]{64}",
                                      row["prepatch_state_sha256"]) is not None
                     and row.get("hook_calls") == 1
                     and valid_number(row.get("source_drift_maxabs"))
                     and 0 <= row["source_drift_maxabs"] <= plan.SOURCE_DRIFT_TOLERANCE,
                     "intervention hook schedule/census/drift invalid")
        nonces.add(nonce)
        if row["source_drift_maxabs"] == row["source_drift_l2"] == 0.0:
            gate.require(row["prepatch_state_sha256"] == hashlib.sha256(
                states[recipient][site].tobytes()).hexdigest(),
                "zero-drift prepatch state hash differs from saved source")
        verify_output(row.get("output"), tokenizer)
        w = plan.WINDOWS[site]
        local = w["local"]
        delta = states[donor][site][local[0]:local[1]] - states[recipient][site][local[0]:local[1]]
        norm = float(np.linalg.norm(delta.astype(np.float64)))
        gate.require(norm >= plan.MIN_LOCAL_DELTA_L2 and valid_number(norm)
                     and valid_number(row.get("local_delta_l2"))
                     and abs(norm-row["local_delta_l2"]) <= max(1e-3, norm*1e-4),
                     "local donor delta norm invalid")
        wanted_window = (w["relocated"] if arm == "relocated_delta" else
                         w["chunk"] if arm == "chunk_donor" else
                         [0, w["total"]] if arm in ("full_donor", "self") else w["local"])
        gate.require(row.get("window") == wanted_window
                     and row.get("seed") == (seed if arm == "equal_l2_noise" else None)
                     and valid_number(row.get("source_drift_l2"))
                     and 0 <= row["source_drift_l2"]
                        <= row["local_delta_l2"] * plan.MAX_PREPATCH_DRIFT_FRACTION
                     and valid_number(row.get("realized_edit_l2"))
                     and row["realized_edit_l2"] >= 0,
                     "intervention window/seed/norm receipt invalid")
        name = f"effective_delta_{arm_id}_{nonce}.npy"
        effective = load_sidecar(folder, row.get("effective_delta"), name)
        gate.require(effective.shape == states[recipient][site].shape
                     and effective.dtype == np.float32,
                     "effective intervention sidecar shape invalid")
        names.add(name)
        a, b = wanted_window
        outside = max(float(np.max(np.abs(effective[:a]))) if a else 0.0,
                      float(np.max(np.abs(effective[b:]))) if b < len(effective) else 0.0)
        gate.require(row.get("outside_window_maxabs") == outside == 0.0,
                     "effective edit changed rows outside frozen window")
        raw_effective_norm = float(np.linalg.norm(effective.astype(np.float64)))
        gate.require(abs(raw_effective_norm-row["realized_edit_l2"])
                     <= max(2e-3, raw_effective_norm*2e-4),
                     "saved effective edit norm differs from worker receipt")
        if arm in ("local_donor", "chunk_donor", "full_donor"):
            expected_edit = states[donor][site][a:b] - states[recipient][site][a:b]
            numerical = row["source_drift_maxabs"] + fp32_roundoff_bound(
                states[donor][site][a:b], states[recipient][site][a:b], effective[a:b])
        elif arm == "self":
            expected_edit = np.zeros_like(effective[a:b])
            numerical = row["source_drift_maxabs"] + fp32_roundoff_bound(
                states[recipient][site][a:b], effective[a:b])
        elif arm == "relocated_delta":
            expected_edit = delta
            numerical = fp32_roundoff_bound(
                states[recipient][site][a:b], delta, effective[a:b])
        else:
            generator = np.random.default_rng(seed)
            noise = generator.standard_normal(tuple(delta.shape)).astype(np.float32)
            expected_edit = noise * (norm / float(np.linalg.norm(noise.astype(np.float64))))
            numerical = max(1e-4, norm*1e-5, fp32_roundoff_bound(
                states[recipient][site][a:b], expected_edit, effective[a:b]))
        gate.require(effective[a:b].shape == expected_edit.shape
                     and float(np.max(np.abs(effective[a:b]-expected_edit))) <= numerical,
                     "effective edit does not match donor/seeded control")
        realized = row["realized_edit_l2"]
        if arm in ("local_donor", "relocated_delta", "equal_l2_noise"):
            gate.require(abs(realized-norm)
                         <= row["source_drift_l2"] + max(2e-3, norm*2e-4),
                         "equal-local-norm intervention invalid")
        if arm in ("chunk_donor", "full_donor"):
            a, b = wanted_window
            expected_norm = float(np.linalg.norm(
                (states[donor][site][a:b]-states[recipient][site][a:b]).astype(np.float64)))
            gate.require(abs(realized-expected_norm)
                         <= row["source_drift_l2"]
                            + max(2e-3, expected_norm*2e-4),
                         "donor-state edit norm invalid")
        if arm == "self":
            gate.require(abs(realized-row["source_drift_l2"]) <= 1e-4,
                         "self edit norm exceeds source drift")
            gate.require(row["output"]["ids"]
                         == payload["baselines"]["calibrated"][recipient]["ids"],
                         "self complete path differs from recipient")
        if arm == "full_donor":
            gate.require(row["output"]["ids"]
                         == payload["baselines"]["calibrated"][donor]["ids"],
                         "full-donor complete path differs from donor")
    return rows, names


def verify_progress(folder: Path, payload: dict, baseline_pass: bool) -> None:
    rows = read_jsonl(folder / "progress.jsonl")
    expected = ([{"kind": "calibrated_baseline", "word": w,
                  "output": payload["baselines"]["calibrated"][w]}
                 for w in plan.WORDS]
                + [{"kind": "original_observation", "word": w,
                    "output": payload["baselines"]["original"][w]}
                   for w in plan.WORDS])
    if baseline_pass:
        expected += [{"kind": "read_only_reference", "word": w,
                      "output": payload["reference"]["references"][w]["read_only_output"],
                      "first_logits_maxabs": payload["reference"]["references"][w]["first_logits_maxabs"],
                      "source_drift_maxabs": payload["reference"]["references"][w]["source_drift_maxabs"]}
                     for w in plan.WORDS]
        expected += [{"kind": "intervention", **row}
                     for row in payload["interventions"]]
    gate.require(rows == expected and plan.sha(folder / "progress.jsonl")
                 == payload.get("progress_sha256"),
                 "durable progress differs from saved schedule")


def audit() -> dict:
    go = gate.validate_go(live=False)
    admission = gate.read(plan.ADMISSION)
    from .. import typed_pilot_cuda_admission_v2
    gate.require_torch_binary_preflight(admission,
        typed_pilot_cuda_admission_v2.verify_binaries())
    plan.verify_inputs()
    attempt = plan.ATTEMPT
    _, parent_result, worker_start = verify_parent(attempt, go)
    folder = attempt / "worker_output"
    payload = gate.read(folder / "result.json")
    gate.require(payload.get("schema") == worker.SCHEMA
                 and payload.get("status") in
                    ("baseline_failed_no_interventions",
                     "candidate_saved_pending_parent_and_audit")
                 and payload.get("model_revision")
                    == "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
                 and worker_start.get("go_sha256") == plan.sha(plan.GO),
                 "worker result identity invalid")
    tokenizer, expected_names = verify_processor(folder, payload, admission)
    baselines = payload.get("baselines")
    gate.require(isinstance(baselines, dict)
                 and set(baselines) == {"calibrated", "original"}
                 and all(set(baselines[family]) == set(plan.WORDS)
                         for family in baselines), "baseline census invalid")
    for family in ("calibrated", "original"):
        for word in plan.WORDS:
            verify_output(baselines[family][word], tokenizer)
    baseline_pass = all(baselines["calibrated"][word]["label"] == word
                        and baselines["calibrated"][word]["terminal_eos"]
                        for word in plan.WORDS)
    status = "negative_baseline_only"
    if baseline_pass:
        gate.require(payload["status"] == "candidate_saved_pending_parent_and_audit",
                     "completed baseline lacks causal schedule")
        states, extra_names = verify_reference(folder, payload, tokenizer)
        expected_names.update(extra_names)
        _, edit_names = verify_interventions(folder, payload, states, tokenizer)
        expected_names.update(edit_names)
        status = "technical_fit_only_causal_schedule_audited"
    else:
        gate.require(payload["status"] == "baseline_failed_no_interventions"
                     and payload.get("reference") is None
                     and payload.get("interventions") == [],
                     "negative baseline improperly continued to intervention")
    verify_progress(folder, payload, baseline_pass)
    cuda, peak_reserved = verify_cuda(folder, payload, baseline_pass)
    expected_names.update({"start.json", "result.json", "progress.jsonl",
                           "cuda_samples.jsonl"})
    gate.require({p.name for p in folder.iterdir()} == expected_names,
                 "worker evidence inventory changed")
    return {"schema": "qwen_audio_lexical_saved_audit_v1",
            "status": status, "worker_status": payload["status"],
            "worker_result_sha256": plan.sha(folder / "result.json"),
            "parent_result_sha256": plan.sha(attempt / "result.json"),
            "calibrated_baseline_labels": {w: baselines["calibrated"][w]["label"]
                                           for w in plan.WORDS},
            "interventions_audited": len(payload["interventions"]),
            "terminal_eos_interventions": sum(row["output"]["terminal_eos"]
                                               for row in payload["interventions"]),
            "peak_gpu_used_bytes": parent_result["worker_peak_gpu_used_bytes"],
            "peak_tree_rss_bytes": parent_result["worker_peak_tree_rss_bytes"],
            "peak_framework_reserved_bytes": peak_reserved,
            "heartbeat_clean_exit_boundary": parent_result["heartbeat_clean_exit_boundary"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args()
    result = audit()
    output = plan.ATTEMPT.parent / "attempt_001_saved_audit_v1.json"
    gate.require(not output.exists(), "saved audit already exists")
    gate.write_once(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
