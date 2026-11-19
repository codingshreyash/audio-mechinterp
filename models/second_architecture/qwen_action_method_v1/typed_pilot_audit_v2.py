"""Read-only, no-weight audit for one typed Qwen resource pilot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host
from models.second_architecture.nonspeech_probe_v1 import causal_parent_v2 as bounded
from . import endpoint_v2, runner_contract_v1, typed_pilot_backend_v2 as backend
from . import typed_pilot_parent_v2 as parent


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    backend.require(result.tzinfo is not None, "pilot timestamp lacks timezone")
    return result


def saved_array(folder: Path, ref: dict) -> np.ndarray:
    backend.require(set(ref) == {"path", "sha256", "shape", "dtype", "bytes"},
                    "pilot sidecar ref malformed")
    name = ref["path"]
    backend.require(isinstance(name, str) and name == Path(name).name
                    and name.endswith(".npy"), "pilot sidecar path unsafe")
    path = folder / name
    backend.require(path.is_file() and not path.is_symlink()
                    and path.stat().st_size == ref["bytes"]
                    and backend.sha(path) == ref["sha256"],
                    "pilot sidecar missing or altered")
    value = np.load(path, allow_pickle=False)
    backend.require(value.dtype == np.float32 and str(value.dtype) == ref["dtype"]
                    and list(value.shape) == ref["shape"]
                    and np.isfinite(value).all(),
                    "pilot sidecar tensor invalid")
    return value


def reconstruct_triplet(folder: Path, rows: list[dict]) -> tuple[list[dict], set[str]]:
    seen: set[str] = set()
    reconstructed = []
    for row in rows:
        refs = row["sidecars"]
        values = {}
        for key in ("first_logits", "after_label_logits"):
            values[key] = saved_array(folder, refs[key])
            backend.require(refs[key]["path"] not in seen, "duplicate sidecar")
            seen.add(refs[key]["path"])
        for kind in ("states", "gradients"):
            values[kind] = {}
            for site in backend.SITES:
                ref = refs[kind][site]
                values[kind][site] = saved_array(folder, ref)
                backend.require(ref["path"] not in seen, "duplicate sidecar")
                seen.add(ref["path"])
        reference = refs["reference"]
        values["reference"] = {"no_hook": {}, "read_only": {"states": {}}}
        for mode in ("no_hook", "read_only"):
            for kind in ("first_logits", "after_label_logits"):
                ref = reference[mode][kind]
                values["reference"][mode][kind] = saved_array(folder, ref)
                backend.require(ref["path"] not in seen, "duplicate reference sidecar")
                seen.add(ref["path"])
        for site in backend.SITES:
            ref = reference["read_only"]["states"][site]
            values["reference"]["read_only"]["states"][site] = saved_array(folder, ref)
            backend.require(ref["path"] not in seen, "duplicate reference state")
            seen.add(ref["path"])
        reconstructed.append({**{key: value for key, value in row.items()
                                if key != "sidecars"}, **values})
    backend.require({path.name for path in folder.glob("*.npy")} == seen
                    and len(seen) == 36, "pilot sidecar inventory incomplete")
    return reconstructed, seen


def check_reference_parity(row: dict) -> dict[str, float]:
    backend.require(row["reference_hook_calls"] == {site: 1 for site in backend.SITES},
                    "read-only teacher-forced hook calls invalid")
    reference = row["reference"]
    measured: dict[str, float] = {}
    for kind in ("first_logits", "after_label_logits"):
        native = reference["no_hook"][kind]
        read_only = reference["read_only"][kind]
        leaf = row[kind]
        backend.require(native.shape == read_only.shape == leaf.shape
                        and native.ndim == 1 and native.size > 151645,
                        "reference full logits shape invalid")
        measured[f"native_read_only_{kind}"] = float(np.max(np.abs(native-read_only)))
        measured[f"read_only_leaf_{kind}"] = float(np.max(np.abs(read_only-leaf)))
    for site in backend.SITES:
        native_state = reference["read_only"]["states"][site]
        leaf_state = row["states"][site]
        backend.require(native_state.shape == leaf_state.shape == (2048,),
                        "reference prompt-last state shape invalid")
        measured[f"read_only_leaf_{site}"] = float(np.max(np.abs(native_state-leaf_state)))
    backend.require(measured == row["reference_parity_maxabs"]
                    and all(value <= 1e-5 for value in measured.values()),
                    "saved unmodified/read-only/leaf parity failed")
    return measured


def audit(attempt: Path = backend.ATTEMPT) -> dict:
    receipt = backend.validate_go()
    backend.require(attempt.resolve() == backend.ATTEMPT.resolve()
                    and attempt.is_dir() and not (attempt / "failure.json").exists(),
                    "fixed complete pilot attempt required")
    start = json.loads((attempt / "start.json").read_text())
    result = json.loads((attempt / "result.json").read_text())
    terminal_path = attempt / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    backend.require(start["schema"] == result["schema"]
                    == "qwen_action_typed_pilot_parent_v2"
                    and result["status"] == terminal["status"]
                    == "candidate_saved_pending_audit"
                    and terminal["schema"] == "qwen_action_typed_pilot_terminal_v2"
                    and start["go_sha256"] == result["go_sha256"]
                    == backend.sha(backend.GO)
                    and start["source_identity"] == receipt["source_identity"]
                    and terminal["start_sha256"] == backend.sha(attempt / "start.json")
                    and terminal["result_sha256"] == backend.sha(attempt / "result.json"),
                    "pilot parent/terminal identity invalid")
    started, ended, terminal_time = (_time(start["start_utc"]),
                                    _time(result["end_utc"]),
                                    _time(terminal["terminal_utc"]))
    terminal_mtime = datetime.fromtimestamp(terminal_path.stat().st_mtime,
                                             tz=started.tzinfo)
    backend.require(started <= ended <= terminal_time <= terminal_mtime
                    and terminal_mtime-started <= timedelta(seconds=parent.WALL_SECONDS)
                    and 0 < result["elapsed_seconds_before_result"]
                    <= terminal["elapsed_seconds_after_result"] <= parent.WALL_SECONDS,
                    "pilot whole-attempt deadline proof invalid")
    backend.require(result.get("worker_returncode") == 0
                    and result.get("worker_stop_reason") == "normal_exit"
                    and result.get("worker_timed_out") is False
                    and result.get("worker_rss_samples", 0) > 0
                    and 0 < result["worker_peak_tree_rss_bytes"] < parent.RSS_BYTES
                    and 0 < result["worker_elapsed_seconds"]
                    <= result["worker_seconds_granted"]
                    and bounded.output_bytes(attempt) < parent.EVIDENCE_BYTES,
                    "pilot worker resource/completion evidence invalid")
    rss_file = attempt / "worker_rss_samples.jsonl"
    samples = [json.loads(line) for line in rss_file.read_text().splitlines()]
    backend.require(len(samples) == result["worker_rss_samples"]
                    and backend.sha(rss_file) == result["worker_rss_sample_log_sha256"]
                    and max(x["tree_rss_bytes"] for x in samples)
                    == result["worker_peak_tree_rss_bytes"]
                    and all(0 < x["tree_rss_bytes"] < parent.RSS_BYTES
                            for x in samples), "pilot RSS samples invalid")
    pre = json.loads((attempt / "pre_host.json").read_text())
    post = json.loads((attempt / "post_host.json").read_text())
    backend.require(backend.sha(attempt / "pre_host.json") == result["pre_host_sha256"]
                    and backend.sha(attempt / "post_host.json") == result["post_host_sha256"],
                    "pilot host hashes changed")
    for name, evidence in (("pre", pre), ("post", post)):
        host.validate_power(evidence)
        raw = host.load_archive(attempt, evidence, name)
        collected = _time(evidence["collected_utc"])
        window_start = started if name == "post" else collected-timedelta(minutes=10)
        host.validate_log(raw, window_start, collected, collected)
    backend.require(started <= _time(pre["collected_utc"])
                    <= _time(result["worker_exit_utc"])
                    <= _time(post["collected_utc"]) <= ended,
                    "pilot host ordering invalid")
    folder = attempt / "worker_output"
    worker_start = json.loads((folder / "start.json").read_text())
    worker = json.loads((folder / "result.json").read_text())
    backend.require(worker_start["schema"] == worker["schema"]
                    == "qwen_action_typed_pilot_worker_v2"
                    and worker_start["status"] == "started"
                    and worker["status"] == "candidate_saved_pending_parent_and_audit"
                    and worker["case_id"] == worker_start["case_id"]
                    == "F_L1__typed"
                    and worker["model_revision"] == backend.load_spec()["model_revision"]
                    and worker["prompt_text"] == backend.fixed_case()["text"]
                    and worker_start["pre_host_sha256"] == backend.sha(attempt / "pre_host.json")
                    and worker_start["go_sha256"] == backend.sha(backend.GO),
                    "pilot worker identity invalid")
    mps_file = folder / "mps_samples.jsonl"
    mps = [json.loads(line) for line in mps_file.read_text().splitlines()]
    backend.require(backend.sha(mps_file) == worker["mps_log_sha256"]
                    and [x["phase"] for x in mps]
                    == ["after_model_load", "after_LIST_vjp", "after_ASK_vjp",
                        "after_CANCEL_vjp", "after_generation"]
                    and all(0 < x["driver_allocated_bytes"] < backend.MAX_MPS_BYTES
                            for x in mps), "pilot sampled MPS evidence invalid")
    reconstructed, sidecars = reconstruct_triplet(folder, worker["triplet"])
    check = runner_contract_v1.validate_teacher_forced_triplet(reconstructed)
    for row in reconstructed:
        check_reference_parity(row)
        backend.require(abs(row["claimed_score"]-row["autograd_score"]) <= 1e-6,
                        "pilot saved-score/autograd scalar mismatch")
    generation = worker["generation"]
    from transformers import AutoTokenizer
    spec = backend.load_spec()
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"],
        revision=spec["model_revision"], local_files_only=True)
    for key, ids_key in (("no_hook", "raw_no_hook_ids"),
                         ("read_only", "raw_read_only_ids")):
        backend.require(generation[key]
            == endpoint_v2.score_complete_answer(generation[ids_key], tokenizer),
            "pilot generated endpoint record differs from raw tokens")
    backend.require(generation["raw_no_hook_ids"]
                    == generation["raw_read_only_ids"]
                    and generation["no_hook"]["continuation_ids"]
                    == generation["read_only"]["continuation_ids"]
                    and generation["read_only_hook_calls"] > 0,
                    "pilot generation read-only parity failed")
    return {"schema": "qwen_action_typed_pilot_saved_audit_v2",
            "status": "typed_fit_one_case_technical_pilot_only",
            "case_id": "F_L1__typed", "teacher_forced_targets": 3,
            "binary_sidecars": len(sidecars),
            "cross_suffix_state_maxabs": check["cross_suffix_state_maxabs"],
            "generation_status": generation["no_hook"]["status"],
            "peak_sampled_tree_rss_bytes": result["worker_peak_tree_rss_bytes"],
            "peak_sampled_mps_driver_bytes": max(x["driver_allocated_bytes"]
                                                   for x in mps)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, default=backend.ATTEMPT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend.require(args.output.resolve()
                    == (backend.ATTEMPT.parent / "attempt_001_audit.json").resolve()
                    and not args.output.exists(),
                    "fixed exclusive pilot audit destination required")
    result = audit(args.attempt)
    backend.write_once(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
