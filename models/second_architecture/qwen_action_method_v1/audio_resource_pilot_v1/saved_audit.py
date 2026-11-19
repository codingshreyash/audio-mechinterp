"""No-weight audit of raw non-study pilot evidence; no semantic grading."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import gate, parent, plan, worker


def load_sidecar(folder: Path, metadata: dict, *, expected: str | None = None) -> np.ndarray:
    name = metadata.get("path")
    gate.require(isinstance(name, str) and name.endswith(".npy")
                 and Path(name).name == name and name not in (".", ".."),
                 "sidecar path invalid")
    path = folder / name
    gate.require(path.is_file() and not path.is_symlink()
                 and plan.sha(path) == metadata.get("sha256"),
                 "sidecar hash invalid")
    array = np.load(path, allow_pickle=False)
    gate.require(list(array.shape) == metadata.get("shape")
                 and str(array.dtype) == metadata.get("dtype")
                 and array.size > 0 and np.isfinite(array).all(),
                 "sidecar array invalid")
    if expected is not None:
        gate.require(name == expected, "sidecar name changed")
    return array


def verify_generation_shapes(shapes: dict, prompt_length: int) -> None:
    gate.require(set(shapes) == set(plan.SITES), "generation hook sites invalid")
    for site in plan.SITES:
        rows = shapes[site]
        gate.require(isinstance(rows, list) and len(rows) >= 1
                     and rows[0] == [1, prompt_length, 2048]
                     and all(row == [1, 1, 2048] for row in rows[1:]),
                     "read-only generation hook shape invalid")


def audit() -> dict:
    go = gate.validate_go(live=False)
    attempt = plan.ATTEMPT
    gate.require(attempt.is_dir() and not attempt.is_symlink()
                 and parent.output_bytes(attempt) < plan.EVIDENCE_BYTES,
                 "fixed bounded attempt missing or oversized")
    start, result, terminal = (gate.read(attempt / name) for name in
                               ("start.json", "result.json", "terminal.json"))
    folder = attempt / "worker_output"
    worker_start, payload = (gate.read(folder / name) for name in
                             ("start.json", "result.json"))
    pre, post = (gate.read(attempt / name) for name in ("pre_host.json", "post_host.json"))
    expected_parent = {"pre_host.json", "start.json", "post_host.json",
                       "result.json", "terminal.json", "worker_stdout.txt",
                       "worker_stderr.txt", "worker_resource_samples.jsonl",
                       "worker_output"}
    gate.require({path.name for path in attempt.iterdir()} == expected_parent,
                 "parent attempt file inventory differs")
    gate.require(start.get("schema") == result.get("schema") == "qwen_audio_resource_parent_v1"
                 and start.get("status") == "worker_authorized"
                 and result.get("status") == terminal.get("status") == "candidate_saved_pending_audit"
                 and terminal.get("schema") == "qwen_audio_resource_terminal_v1"
                 and worker_start.get("schema") == payload.get("schema") == worker.SCHEMA
                 and worker_start.get("status") == "started"
                 and payload.get("status") == "candidate_saved_pending_parent_and_audit"
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
                 and worker_start.get("parent_pid") == start.get("parent_pid")
                 and worker_start.get("case_id") == payload.get("case_id") == plan.CASE_ID,
                 "parent/worker/GO chain invalid")
    baseline = go["gpu_baseline"]
    for host in (pre, post):
        observed_gpu = host.get("gpu_inventory", {})
        gate.require(all(observed_gpu.get(key) == baseline.get(key) for key in
                         ("gpu_uuid", "gpu_name", "gpu_memory_mib", "mig_mode",
                          "compute_processes", "idle_gpu_limit_bytes"))
                     and type(observed_gpu.get("gpu_used_bytes")) is int
                     and 0 <= observed_gpu["gpu_used_bytes"] < gate.IDLE_GPU_BYTES,
                     "parent GPU baseline/isolation invalid")
    for key in ("start_utc", "worker_exit_utc", "end_utc"):
        gate.require(isinstance(result.get(key), str), "missing parent time")
    began = datetime.fromisoformat(result["start_utc"])
    exited = datetime.fromisoformat(result["worker_exit_utc"])
    ended = datetime.fromisoformat(result["end_utc"])
    gate.require(began <= exited <= ended and result["start_utc"] == start["start_utc"]
                 and result["elapsed_seconds_before_result"] < plan.MAX_SECONDS
                 and result["worker_returncode"] == 0
                 and result["worker_stop_reason"] == "normal_exit"
                 and result["worker_monitor_error"] is None
                 and 0 < result["worker_elapsed_seconds"] <= result["worker_seconds_granted"]
                 and result["worker_resource_samples"] > 0
                 and result["worker_parent_heartbeats"] > 0,
                 "parent time/completion invalid")
    resource = attempt / "worker_resource_samples.jsonl"
    rows = [json.loads(line) for line in resource.read_text().splitlines()]
    gate.require(rows and len(rows) == result["worker_resource_samples"]
                 and plan.sha(resource) == result["worker_resource_log_sha256"]
                 and [x["sample_index"] for x in rows] == list(range(1, len(rows)+1))
                 and all(x["elapsed_seconds"] >= 0 for x in rows)
                 and all(a["elapsed_seconds"] < b["elapsed_seconds"]
                         for a, b in zip(rows, rows[1:]))
                 and all(0 <= x["tree_rss_bytes"] < plan.RSS_BYTES
                         and 0 <= x["gpu_used_bytes"] < plan.GPU_BYTES
                         and x["evidence_bytes"] < plan.EVIDENCE_BYTES for x in rows)
                 and max(x["tree_rss_bytes"] for x in rows)
                    == result["worker_peak_tree_rss_bytes"]
                 and max(x["gpu_used_bytes"] for x in rows)
                    == result["worker_peak_gpu_used_bytes"],
                 "resource sampling invalid")
    cuda_path = folder / "cuda_samples.jsonl"
    cuda = [json.loads(line) for line in cuda_path.read_text().splitlines()]
    gate.require(plan.sha(cuda_path) == payload.get("cuda_log_sha256")
                 and [x.get("phase") for x in cuda]
                    == ["after_model_load", "after_forward", "after_generation"]
                 and all(0 <= x["allocated_bytes"] <= x["reserved_bytes"]
                         <= x["peak_reserved_bytes"] < plan.GPU_BYTES for x in cuda),
                 "framework CUDA evidence invalid")
    plan.verify_wav()
    from transformers import Qwen2_5OmniProcessor  # Processor only, no model weights.
    from models.qwen_omni.adapter import prepare_audio_batch
    from ..audio_candidate_v1 import validate_audio_batch
    snapshot = Path(gate.read(plan.ADMISSION)["processor_preflight"]["snapshot"]["snapshot_path"])
    processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True)
    fresh, digest = prepare_audio_batch(processor, plan.WAV,
                                        system_prompt=plan.SYSTEM_PROMPT)
    gate.require(digest == plan.WAV_SHA256
                 and payload.get("processor_tensors_unchanged_all_calls") is True
                 and payload.get("processor_record")
                    == validate_audio_batch(fresh, processor,
                        wav_sha256=digest, case_id=plan.CASE_ID),
                 "saved processor record differs from frozen WAV")
    gate.require(set(payload.get("batch_sidecars", {})) == set(fresh),
                 "processor tensor census invalid")
    for key, tensor in fresh.items():
        saved = load_sidecar(folder, payload["batch_sidecars"][key],
                             expected=f"batch_{key}.npy")
        gate.require(np.array_equal(saved, tensor.cpu().numpy()),
                     f"processor tensor differs: {key}")
    reference = payload["reference"]
    stock = load_sidecar(folder, reference["stock_first_logits"],
                         expected="stock_first_logits.npy")
    observed = load_sidecar(folder, reference["read_only_first_logits"],
                            expected="read_only_first_logits.npy")
    gate.require(stock.shape == observed.shape and stock.ndim == 1
                 and stock.dtype == observed.dtype == np.float32
                 and stock.size > 151645,
                 "full first-step logits absent")
    delta = float(np.max(np.abs(stock-observed)))
    gate.require(delta == reference["first_logits_maxabs"]
                 and delta <= plan.FIRST_LOGIT_TOLERANCE,
                 "stock/read-only logits parity invalid")
    gate.require(set(reference["read_only_states"]) == set(plan.SITES)
                 and set(reference["forward_hook_shapes"]) == set(plan.SITES),
                 "read-only hook census invalid")
    prompt_length = int(fresh["input_ids"].shape[1])
    for site in plan.SITES:
        state = load_sidecar(folder, reference["read_only_states"][site],
                             expected=f"state_{site.replace('.', '_')}.npy")
        gate.require(state.shape == (2048,) and state.dtype == np.float32
                     and reference["forward_hook_shapes"][site]
                        == [[1, prompt_length, 2048]],
                     "read-only forward hook shape invalid")
    generation = payload["generation"]
    expected = worker.compare_paths(generation["stock_ids"],
                                     generation["read_only_ids"], 151645)
    gate.require(all(generation.get(k) == v for k, v in expected.items()),
                 "saved generated paths/stop reasons differ")
    verify_generation_shapes(generation.get("read_only_hook_shapes", {}), prompt_length)
    expected_names = {"start.json", "result.json", "cuda_samples.jsonl",
                      *(f"batch_{key}.npy" for key in fresh),
                      "stock_first_logits.npy", "read_only_first_logits.npy",
                      *(f"state_{site.replace('.', '_')}.npy" for site in plan.SITES)}
    gate.require({p.name for p in folder.iterdir()} == expected_names,
                 "worker evidence inventory differs")
    return {"schema": "qwen_audio_resource_saved_audit_v1",
            "status": "technical_nonstudy_resource_parity_only",
            "case_id": plan.CASE_ID, "wav_sha256": plan.WAV_SHA256,
            "worker_result_sha256": plan.sha(folder / "result.json"),
            "stock_read_only_first_logits_maxabs": delta,
            "generation_stop_reason": generation["stock_stop_reason"],
            "generation_terminal_eos": generation["terminal_eos"],
            "peak_gpu_used_bytes": result["worker_peak_gpu_used_bytes"],
            "peak_tree_rss_bytes": result["worker_peak_tree_rss_bytes"],
            "peak_framework_reserved_bytes": max(x["peak_reserved_bytes"] for x in cuda)}


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
