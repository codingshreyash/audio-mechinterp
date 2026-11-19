"""Read-only saved audit for the two CUDA stages; never loads model weights."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from models.second_architecture.nonspeech_probe_v1 import causal_parent_v2 as bounded
from . import endpoint_v2, runner_contract_v1
from . import typed_pilot_audit_v3 as method_audit
from . import typed_pilot_cuda_backend_v1 as backend
from . import typed_pilot_cuda_parent_v1 as parent


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    backend.require(parsed.tzinfo is not None, "CUDA timestamp lacks timezone")
    return parsed


def audit(stage: str) -> dict:
    attempt, go = backend.stage_paths(stage)
    receipt = backend.validate_go(stage, verify_runtime=False)
    backend.require(attempt.is_dir() and not (attempt / "failure.json").exists(),
                    "fixed complete CUDA attempt required")
    start = json.loads((attempt / "start.json").read_text())
    result = json.loads((attempt / "result.json").read_text())
    terminal_path = attempt / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    worker_folder = attempt / "worker_output"
    worker_start = json.loads((worker_folder / "start.json").read_text())
    worker = json.loads((worker_folder / "result.json").read_text())
    pre = json.loads((attempt / "pre_host.json").read_text())
    post = json.loads((attempt / "post_host.json").read_text())
    backend.require(start["schema"] == result["schema"]
                    == "qwen_action_typed_pilot_cuda_parent_v1"
                    and start["stage"] == result["stage"] == terminal["stage"]
                    == worker_start["stage"] == worker["stage"] == stage
                    and start["status"] == "worker_authorized"
                    and result["status"] == terminal["status"]
                    == "candidate_saved_pending_audit"
                    and terminal["schema"] == "qwen_action_typed_pilot_cuda_terminal_v1"
                    and worker_start["schema"] == worker["schema"]
                    == "qwen_action_typed_pilot_cuda_worker_v1"
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
                    and result["post_host_sha256"] == backend.sha(attempt / "post_host.json")
                    and worker_start["parent_start_sha256"]
                    == terminal["start_sha256"] == backend.sha(attempt / "start.json")
                    and terminal["result_sha256"] == backend.sha(attempt / "result.json")
                    and worker_start["parent_pid"] == start["parent_pid"]
                    and worker_start["nonce_sha256"] == start["nonce_sha256"]
                    and worker_start["case_id"] == worker["case_id"] == "F_L1__typed"
                    and worker["model_revision"] == backend.load_spec()["model_revision"],
                    "CUDA saved identity/terminal chain invalid")
    began, ended, term = _time(start["start_utc"]), _time(result["end_utc"]), _time(terminal["terminal_utc"])
    backend.require(began <= _time(pre["collected_utc"])
                    <= _time(result["worker_exit_utc"])
                    <= _time(post["collected_utc"]) <= ended <= term
                    and term <= datetime.fromtimestamp(terminal_path.stat().st_mtime,
                                                       tz=term.tzinfo)
                    and term-began <= timedelta(seconds=parent.WALL_SECONDS[stage])
                    and 0 < result["elapsed_seconds_before_result"]
                    <= terminal["elapsed_seconds_after_result"]
                    <= parent.WALL_SECONDS[stage],
                    "CUDA whole-stage deadline proof invalid")
    limit = backend.LOAD_GPU_BYTES if stage == "load" else backend.PILOT_GPU_BYTES
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
    backend.require(len(samples) == result["worker_resource_samples"]
                    and backend.sha(resource_file) == result["worker_resource_sample_log_sha256"]
                    and max(x["tree_rss_bytes"] for x in samples)
                    == result["worker_peak_tree_rss_bytes"]
                    and max(x["gpu_used_bytes"] for x in samples)
                    == result["worker_peak_gpu_used_bytes"]
                    and all(0 < x["tree_rss_bytes"] < parent.RSS_BYTES
                            and 0 <= x["gpu_used_bytes"] < limit for x in samples),
                    "CUDA sampled resource log invalid")
    cuda_file = worker_folder / "cuda_samples.jsonl"
    cuda = [json.loads(line) for line in cuda_file.read_text().splitlines()]
    phases = (["after_model_load"] if stage == "load" else
              ["after_model_load", "after_LIST_vjp", "after_ASK_vjp",
               "after_CANCEL_vjp", "after_generation"])
    backend.require(backend.sha(cuda_file) == worker["cuda_log_sha256"]
                    and [x["phase"] for x in cuda] == phases
                    and all(0 <= x["allocated_bytes"] <= x["reserved_bytes"]
                            <= x["peak_reserved_bytes"] < limit for x in cuda),
                    "CUDA framework allocation evidence invalid")
    if stage == "load":
        backend.require(set(worker_folder.iterdir())
                        == {worker_folder / "start.json", worker_folder / "result.json",
                            worker_folder / "cuda_samples.jsonl"}
                        and "triplet" not in worker and "generation" not in worker,
                        "load-only stage produced scientific output")
        return {"schema": "qwen_action_typed_pilot_cuda_saved_audit_v1",
                "stage": "load", "status": "load_only_pass_no_forward_or_vjp",
                "peak_sampled_gpu_used_bytes": result["worker_peak_gpu_used_bytes"],
                "peak_framework_reserved_bytes": max(x["peak_reserved_bytes"] for x in cuda),
                "peak_sampled_tree_rss_bytes": result["worker_peak_tree_rss_bytes"]}
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
    return {"schema": "qwen_action_typed_pilot_cuda_saved_audit_v1",
            "stage": "pilot", "status": "typed_fit_one_case_technical_pilot_only",
            "teacher_forced_targets": 3, "binary_sidecars": len(sidecars),
            "cross_suffix_state_maxabs": check["cross_suffix_state_maxabs"],
            "generation_status": generation["no_hook"]["status"],
            "peak_sampled_gpu_used_bytes": result["worker_peak_gpu_used_bytes"],
            "peak_framework_reserved_bytes": max(x["peak_reserved_bytes"] for x in cuda),
            "peak_sampled_tree_rss_bytes": result["worker_peak_tree_rss_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("load", "pilot"), required=True)
    args = parser.parse_args()
    result = audit(args.stage)
    attempt, _ = backend.stage_paths(args.stage)
    output = attempt.parent / f"{attempt.name}_audit.json"
    backend.require(not output.exists(), "CUDA audit output already exists")
    backend.write_once(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
