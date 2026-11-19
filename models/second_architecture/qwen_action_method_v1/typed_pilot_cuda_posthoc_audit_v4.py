"""Read-only V4 pilot_003 posthoc audit for one trailing exit-boundary zero RSS.

The pinned V4 saved auditor, GO, and pilot attempt are not modified.
"""
from __future__ import annotations

import argparse
import json

from models.second_architecture.nonspeech_probe_v1 import causal_parent_v2 as bounded
from . import endpoint_v2, runner_contract_v1
from . import typed_pilot_audit_v3 as method_audit
from . import typed_pilot_cuda_backend_v4 as backend
from . import typed_pilot_cuda_parent_v4 as parent
from . import typed_pilot_cuda_admission_v3 as admission
from . import typed_pilot_cuda_posthoc_audit_v3 as posthoc


def verify_resource_samples(result: dict, samples: list[dict],
                            resource_file, limit: int) -> bool:
    """Accept only pilot_003's final zero-RSS sample at a clean exit boundary."""
    backend.require(len(samples) == result["worker_resource_samples"]
                    and backend.sha(resource_file)
                    == result["worker_resource_sample_log_sha256"]
                    and len(samples) > 1
                    and all(sample["sample_index"] == index
                            for index, sample in enumerate(samples, 1))
                    and all(sample["elapsed_seconds"] >= 0 for sample in samples)
                    and all(a["elapsed_seconds"] < b["elapsed_seconds"]
                            for a, b in zip(samples, samples[1:]))
                    and 0 <= (result["worker_elapsed_seconds"]
                              - samples[-1]["elapsed_seconds"]) <= 1.0
                    and result["worker_returncode"] == 0
                    and result["worker_stop_reason"] == "normal_exit"
                    and 0 < result["worker_peak_tree_rss_bytes"]
                        == max(x["tree_rss_bytes"] for x in samples)
                        < parent.RSS_BYTES
                    and max(x["gpu_used_bytes"] for x in samples)
                        == result["worker_peak_gpu_used_bytes"]
                    and all(0 < x["tree_rss_bytes"] < parent.RSS_BYTES
                            for x in samples[:-1])
                    and samples[-1]["tree_rss_bytes"] == 0
                    and len(samples) == 214
                    and all(0 <= x["gpu_used_bytes"] < limit for x in samples),
                    "CUDA sampled resource log invalid")
    return True


def audit(stage: str) -> dict:
    attempt, go = backend.stage_paths(stage)
    receipt = backend.validate_go(stage, verify_runtime=False)
    backend.require(attempt.is_dir() and not (attempt / "failure.json").exists(),
                    "fixed complete CUDA attempt required")
    start = json.loads((attempt / "start.json").read_text())
    result = json.loads((attempt / "result.json").read_text())
    terminal = json.loads((attempt / "terminal.json").read_text())
    worker_folder = attempt / "worker_output"
    worker_start = json.loads((worker_folder / "start.json").read_text())
    worker = json.loads((worker_folder / "result.json").read_text())
    pre = json.loads((attempt / "pre_host.json").read_text())
    post = json.loads((attempt / "post_host.json").read_text())
    stage_start = json.loads((attempt / "stage_start_gpu.json").read_text())
    backend.require(start["schema"] == result["schema"]
                    == "qwen_action_typed_pilot_cuda_parent_v4"
                    and start["stage"] == result["stage"] == terminal["stage"]
                    == worker_start["stage"] == worker["stage"] == stage
                    and start["status"] == "worker_authorized"
                    and result["status"] == terminal["status"]
                    == "candidate_saved_pending_audit"
                    and terminal["schema"] == "qwen_action_typed_pilot_cuda_terminal_v4"
                    and worker_start["schema"] == worker["schema"]
                    == "qwen_action_typed_pilot_cuda_worker_v4"
                    and worker_start["status"] == "started"
                    and worker["status"] == "candidate_saved_pending_parent_and_audit"
                    and start["go_sha256"] == result["go_sha256"]
                    == worker_start["go_sha256"] == backend.sha(go)
                    and start["source_identity"] == receipt["source_identity"]
                    and start["runtime_identity"] == pre["runtime_identity"]
                    == post["runtime_identity"] == receipt["runtime_identity"]
                    and start["approved_attempt"] == str(attempt.resolve())
                    and start["pre_host_sha256"] == result["pre_host_sha256"]
                    == worker_start["pre_host_sha256"] == backend.sha(attempt / "pre_host.json")
                    and start["stage_start_gpu_sha256"]
                    == worker_start["stage_start_gpu_sha256"]
                    == backend.sha(attempt / "stage_start_gpu.json")
                    and result["post_host_sha256"] == backend.sha(attempt / "post_host.json")
                    and worker_start["parent_start_sha256"]
                    == terminal["start_sha256"] == backend.sha(attempt / "start.json")
                    and terminal["result_sha256"] == backend.sha(attempt / "result.json")
                    and worker_start["parent_pid"] == start["parent_pid"]
                    and worker_start["nonce_sha256"] == start["nonce_sha256"]
                    and worker_start["case_id"] == worker["case_id"] == "F_L1__typed"
                    and worker["model_revision"] == backend.load_spec()["model_revision"],
                    "CUDA saved identity/terminal chain invalid")
    baseline = receipt["gpu_baseline"]
    for observed in (pre["gpu_inventory"], stage_start, post["gpu_inventory"]):
        backend.require(all(observed[key] == baseline[key] for key in
                            ("gpu_uuid", "gpu_name", "gpu_memory_mib", "mig_mode",
                             "compute_processes", "idle_gpu_limit_bytes"))
                        and type(observed["gpu_used_bytes"]) is int
                        and 0 <= observed["gpu_used_bytes"]
                            < admission.IDLE_GPU_LIMIT_BYTES,
                        "saved GPU isolation/baseline evidence invalid")
    timing = posthoc.verify_parent_timing(stage, start, result, terminal, pre, post)
    limit = backend.PILOT_GPU_BYTES
    backend.require(result["worker_returncode"] == 0
                    and result["worker_stop_reason"] == "normal_exit"
                    and 0 < result["worker_elapsed_seconds"] <= result["worker_seconds_granted"]
                    and result["worker_resource_samples"] > 0
                    and result["worker_parent_heartbeats"] > 0
                    and 0 < result["worker_peak_tree_rss_bytes"] < parent.RSS_BYTES
                    and 0 <= result["worker_peak_gpu_used_bytes"] < limit
                    and bounded.output_bytes(attempt) < parent.EVIDENCE_BYTES,
                    "CUDA resource/completion evidence invalid")
    resource_file = attempt / "worker_resource_samples.jsonl"
    samples = [json.loads(line) for line in resource_file.read_text().splitlines()]
    verify_resource_samples(result, samples, resource_file, limit)
    cuda_file = worker_folder / "cuda_samples.jsonl"
    cuda = [json.loads(line) for line in cuda_file.read_text().splitlines()]
    phases = ["after_model_load", "after_LIST_vjp", "after_ASK_vjp",
              "after_CANCEL_vjp", "after_generation"]
    backend.require(backend.sha(cuda_file) == worker["cuda_log_sha256"]
                    and [x["phase"] for x in cuda] == phases
                    and all(0 <= x["allocated_bytes"] <= x["reserved_bytes"]
                            <= x["peak_reserved_bytes"] < limit
                            and x["math_mode"] == receipt["runtime_identity"]["math_mode"]
                            for x in cuda),
                    "CUDA framework allocation evidence invalid")
    rows, sidecars = method_audit.reconstruct_triplet(worker_folder, worker["triplet"])
    check = runner_contract_v1.validate_teacher_forced_triplet(rows)
    for row in rows:
        method_audit.check_reference_parity(row)
        backend.require(abs(row["claimed_score"]-row["autograd_score"]) <= 1e-6,
                        "saved score/autograd mismatch")
    from transformers import AutoTokenizer
    spec = backend.load_spec()
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"],
        revision=spec["model_revision"], local_files_only=True)
    generation = worker["generation"]
    for key, raw in (("no_hook", "raw_no_hook_ids"),
                     ("read_only", "raw_read_only_ids")):
        backend.require(generation[key]
                        == endpoint_v2.score_complete_answer(generation[raw], tokenizer),
                        "saved generation differs from full IDs")
    backend.require(generation["raw_no_hook_ids"] == generation["raw_read_only_ids"]
                    and generation["no_hook"]["continuation_ids"]
                    == generation["read_only"]["continuation_ids"]
                    and generation["read_only_hook_calls"] > 0
                    and worker["prompt_text"] == backend.fixed_case()["text"],
                    "read-only generation parity or prompt invalid")
    return {"schema": "qwen_action_typed_pilot_cuda_posthoc_saved_audit_v4",
            "stage": "pilot", "status": "typed_fit_one_case_technical_pilot_only",
            "timing": timing,
            "teacher_forced_targets": 3, "binary_sidecars": len(sidecars),
            "cross_suffix_state_maxabs": check["cross_suffix_state_maxabs"],
            "generation_status": generation["no_hook"]["status"],
            "peak_sampled_gpu_used_bytes": result["worker_peak_gpu_used_bytes"],
            "peak_framework_reserved_bytes": max(x["peak_reserved_bytes"] for x in cuda),
            "peak_sampled_tree_rss_bytes": result["worker_peak_tree_rss_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot",), required=True)
    args = parser.parse_args()
    result = audit(args.stage)
    attempt, _ = backend.stage_paths(args.stage)
    output = attempt.parent / f"{attempt.name}_posthoc_audit_v4.json"
    backend.require(not output.exists(), "CUDA audit output already exists")
    backend.write_once(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
