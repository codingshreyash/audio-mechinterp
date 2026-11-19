"""One-shot whole-attempt wall/evidence parent for padding assay V2."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
from . import causal_parent_v2 as bounded
from . import padding_channel_plan_v5 as plan
from . import padding_channel_worker_v2 as worker

ATTEMPT = worker.ATTEMPT
MAX_POST_PROBE_LAG_SECONDS = 120
POST_AND_FINAL_RESERVE_SECONDS = 480
FINAL_WRITE_RESERVE_SECONDS = 5


def guard_deadline(deadline: float, *, reserve_seconds: float = 0.0) -> None:
    worker.require(time.monotonic() + reserve_seconds < deadline,
                   "parent 1800-second wall deadline exhausted")


def remaining_worker_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic() - POST_AND_FINAL_RESERVE_SECONDS
    worker.require(remaining > 0, "no worker time remains after host/result reserve")
    return remaining


def guard_whole_attempt_write(attempt: Path, proposed_bytes: int,
                              *, terminal: bool = False) -> None:
    worker.require(proposed_bytes >= 0, "negative proposed evidence bytes")
    reserve = 0 if terminal else plan.EVIDENCE_RESERVE_BYTES
    worker.require(bounded.output_bytes(attempt) + proposed_bytes + reserve
                   <= plan.EVIDENCE_LIMIT_BYTES,
                   "whole-attempt evidence reserve/cap exceeded before write")


def write_parent_json(attempt: Path, filename: str, payload: dict,
                      *, terminal: bool = False) -> None:
    raw_size = len(worker.canonical(payload).encode()) + 1
    guard_whole_attempt_write(attempt, raw_size, terminal=terminal)
    worker.write_once(attempt / filename, payload)
    worker.require(bounded.output_bytes(attempt) <= plan.EVIDENCE_LIMIT_BYTES,
                   "whole-attempt evidence cap exceeded after JSON write")


def expected_names() -> list[str]:
    return [worker.call_name(row) for row in plan.schedule()]


def host_probe(attempt: Path, label: str, *, run_start: datetime | None = None,
               deadline: float) -> dict:
    guard_deadline(deadline)
    evidence = host_gate.probe(run_start=run_start)
    guard_deadline(deadline)
    packed = gzip.compress(evidence["outputs"]["power_log"].encode(), mtime=0)
    guard_whole_attempt_write(attempt, len(packed))
    host_gate.archive(attempt, evidence, label)
    worker.require((attempt / f"{label}_power_log.txt.gz").stat().st_size
                   == len(packed), "host archive size changed during write")
    guard_deadline(deadline)
    write_parent_json(attempt, f"{label}_host.json", evidence)
    return evidence


def launch(admission: Path, go: Path) -> dict:
    receipt = worker.validate_go(admission, go)
    names = expected_names()
    worker.require(len(names) == 436 and not ATTEMPT.exists(),
                   "attempt path or call census invalid")
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPT.mkdir()
    start_mono = time.monotonic()
    deadline = start_mono + plan.WALL_SECONDS
    start = datetime.now(timezone.utc)
    write_parent_json(ATTEMPT, "launch_start.json", {
        "schema": "whisper_padding_launch_v2", "status": "started",
        "start_utc": start.isoformat(), "attempt": str(ATTEMPT),
        "admission_sha256": worker.sha_file(admission),
        "go_sha256": worker.sha_file(go),
        "source_identity": receipt["sources"], "scheduled_call_names": names})
    result: dict = {"schema": "whisper_padding_launch_v2",
                    "status": "invalid_or_partial", "attempt": str(ATTEMPT),
                    "start_utc": start.isoformat(),
                    "admission_sha256": worker.sha_file(admission),
                    "go_sha256": worker.sha_file(go)}
    worker_exit: datetime | None = None
    candidate_complete = False
    try:
        pre = host_probe(ATTEMPT, "pre", deadline=deadline)
        worker_seconds = remaining_worker_seconds(deadline)
        result["worker_seconds_granted"] = worker_seconds
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.padding_channel_worker_v2",
                   "run", "--output", str(ATTEMPT / "worker_output"),
                   "--admission", str(admission.resolve()), "--go", str(go.resolve())]
        observation = bounded.run_bounded(
            command, ATTEMPT, seconds=worker_seconds,
            rss_bytes=plan.RSS_LIMIT_BYTES,
            evidence_bytes=plan.EVIDENCE_LIMIT_BYTES-plan.EVIDENCE_RESERVE_BYTES)
        worker_exit = datetime.now(timezone.utc)
        guard_deadline(deadline)
        result.update(observation)
        result["worker_exit_observed_utc"] = worker_exit.isoformat()
        progress = ATTEMPT / "worker_output/call_progress.jsonl"
        result["worker_call_progress_sha256"] = (
            worker.sha_file(progress) if progress.is_file() else None)
        result["worker_call_census"] = bounded.progress_census(progress, names)
        post = host_probe(ATTEMPT, "post", run_start=start, deadline=deadline)
        guard_deadline(deadline)
        collected = datetime.fromisoformat(post["collected_utc"])
        lag = (collected-worker_exit).total_seconds()
        worker.require(0 <= lag <= MAX_POST_PROBE_LAG_SECONDS,
                       "post-host collection lag exceeded")
        result["host_probe_lag_seconds"] = lag
        result["host_collected_utc"] = [pre["collected_utc"], post["collected_utc"]]
        result["pre_host_sha256"] = worker.sha_file(ATTEMPT / "pre_host.json")
        result["post_host_sha256"] = worker.sha_file(ATTEMPT / "post_host.json")
        census = result["worker_call_census"]
        records = ATTEMPT / "worker_output/records.jsonl"
        if (observation["worker_returncode"] == 0
                and observation["worker_stop_reason"] == "normal_exit"
                and not observation["worker_timed_out"]
                and 0 < observation["worker_peak_tree_rss_bytes"] < plan.RSS_LIMIT_BYTES
                and observation["worker_rss_samples"] > 0
                and len(census["completed_call_names"]) == len(names)
                and census["failed_attempted_call_name"] is None
                and not census["censored_call_names"] and records.is_file()
                and json.loads(records.read_text().splitlines()[-1])["type"] == "complete"):
            candidate_complete = True
        worker.require(bounded.output_bytes(ATTEMPT) <= plan.EVIDENCE_LIMIT_BYTES,
                       "parent evidence ceiling exceeded")
    except BaseException as error:
        result["launch_error"] = {"type": type(error).__name__, "message": str(error)}
    if candidate_complete:
        try:
            guard_deadline(deadline, reserve_seconds=FINAL_WRITE_RESERVE_SECONDS)
        except BaseException as error:
            candidate_complete = False
            result["launch_error"] = {"type": type(error).__name__, "message": str(error)}
    result["status"] = "candidate_complete" if candidate_complete else "invalid_or_partial"
    result["end_utc"] = datetime.now(timezone.utc).isoformat()
    result["parent_elapsed_seconds_before_result_write"] = time.monotonic()-start_mono
    result["evidence_bytes_before_result"] = bounded.output_bytes(ATTEMPT)
    write_parent_json(ATTEMPT, "launch_result.json", result, terminal=True)
    if candidate_complete:
        guard_deadline(deadline, reserve_seconds=FINAL_WRITE_RESERVE_SECONDS)
        completion = {"schema": "whisper_padding_completion_v2",
                      "status": "completed_pending_saved_audit",
                      "result_sha256": worker.sha_file(ATTEMPT / "launch_result.json"),
                      "parent_elapsed_seconds": time.monotonic()-start_mono,
                      "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_parent_json(ATTEMPT, "launch_completion.json", completion,
                          terminal=True)
        guard_deadline(deadline)
        result["status"] = "completed_pending_saved_audit"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.admission, args.go)
    print(worker.canonical({"status": result["status"],
                            "attempt": result["attempt"],
                            "completed_calls": len(result.get(
                                "worker_call_census", {}).get(
                                    "completed_call_names", []))}))
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
