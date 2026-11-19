"""One-checkpoint, one-shot bounded parent for the natural-tail V2 assay."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
from . import causal_parent_v2 as bounded
from . import natural_tail_runner_v2 as worker

ROOT = worker.ROOT
LOCK = worker.HERE/"runs/natural_tail_waveform_v1/active_model_lock"
POST_RESERVE_SECONDS = 480


def evidence_bytes(path: Path) -> int:
    return bounded.output_bytes(path)


def save_host(attempt: Path, label: str, *, start: datetime | None = None) -> dict:
    probe = host_gate.probe(run_start=start)
    host_gate.archive(attempt, probe, label)
    worker.write_once(attempt/f"{label}_host.json", probe)
    return probe


def progress_census(path: Path, plan: list[dict]) -> dict:
    if not path.is_file():
        return {"complete": 0, "attempted_incomplete": None,
                "not_attempted": len(plan)}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require = worker.require
    require(len(rows) <= 2*len(plan), "progress exceeds schedule")
    complete = 0
    for position, row in enumerate(plan):
        if 2*position >= len(rows):
            break
        require(rows[2*position] == {"type":"attempt", "index":row["index"]},
                "progress attempt order changed")
        if 2*position+1 >= len(rows):
            break
        require(rows[2*position+1] == {"type":"complete", "index":row["index"]},
                "progress completion order changed")
        complete += 1
    require(len(rows) in (2*complete, 2*complete+1), "progress extras")
    return {"complete": complete,
            "attempted_incomplete": plan[complete]["index"] if len(rows)%2 else None,
            "not_attempted": len(plan)-complete-(len(rows)%2)}


def launch(model: str, attempt: Path, admission: Path, go: Path) -> dict:
    receipt = worker.validate_go(admission, go, model)
    worker.require(attempt.is_absolute() and attempt.parent.is_dir()
                   and not attempt.exists(), "new absolute attempt path required")
    worker.require(shutil.disk_usage(attempt.parent).free >= 2*worker.EVIDENCE_BYTES,
                   "free disk below 2 GiB")
    worker.require(psutil.virtual_memory().available >= 5*1024**3,
                   "memory headroom below 5 GiB")
    worker.require(not LOCK.exists(), "another natural-tail checkpoint is active")
    # The host gate runs before the exclusive attempt is created.
    before = host_gate.probe()
    started_mono = time.monotonic()
    deadline = started_mono+worker.WALL_SECONDS
    started_utc = datetime.now(timezone.utc)
    LOCK.mkdir()
    try:
        attempt.mkdir()
    except BaseException:
        LOCK.rmdir()
        raise
    result = {"schema":"whisper_natural_tail_parent_v2", "model":model,
              "status":"invalid_or_partial", "attempt":str(attempt),
              "start_utc":started_utc.isoformat(),
              "admission_sha256":worker.old.sha_file(admission),
              "go_sha256":worker.old.sha_file(go),
              "source_identity":receipt["source_identity"]}
    try:
        host_gate.archive(attempt, before, "pre")
        worker.write_once(attempt/"pre_host.json", before)
        worker.write_once(attempt/"launch_start.json", {**result, "status":"started"})
        remaining = deadline-time.monotonic()-POST_RESERVE_SECONDS
        worker.require(remaining > 0, "no worker time remains")
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.natural_tail_runner_v2",
                   "run", "--model", model, "--attempt", str(attempt),
                   "--admission", str(admission), "--review", str(go)]
        observation = bounded.run_bounded(
            command, attempt, seconds=remaining, rss_bytes=worker.RSS_BYTES,
            evidence_bytes=worker.EVIDENCE_BYTES-worker.EVIDENCE_RESERVE)
        result["monitor"] = observation
        worker_exit = datetime.now(timezone.utc)
        result["worker_exit_utc"] = worker_exit.isoformat()
        result["progress"] = progress_census(attempt/"progress.jsonl", receipt["schedule"])
        worker.require(time.monotonic() < deadline, "whole-attempt wall deadline exhausted")
        post = save_host(attempt, "post", start=started_utc)
        lag = (datetime.fromisoformat(post["collected_utc"])-worker_exit).total_seconds()
        worker.require(0 <= lag <= 120, "post-host probe lag exceeded")
        result["post_probe_lag_seconds"] = lag
        records = attempt/"records.jsonl"
        completed = records.is_file() and bool(records.read_text().splitlines()) and (
            json.loads(records.read_text().splitlines()[-1])
            == {"type":"complete", "calls":160})
        if (observation["worker_returncode"] == 0
                and observation["worker_stop_reason"] == "normal_exit"
                and not observation["worker_timed_out"]
                and 0 < observation["worker_peak_tree_rss_bytes"] < worker.RSS_BYTES
                and observation["worker_rss_samples"] > 0
                and result["progress"]["complete"] == 160
                and result["progress"]["attempted_incomplete"] is None
                and completed and time.monotonic() < deadline
                and evidence_bytes(attempt) < worker.EVIDENCE_BYTES):
            result["status"] = "completed_pending_saved_audit"
    except BaseException as error:
        result["error"] = {"type":type(error).__name__, "message":str(error)}
    finally:
        result["end_utc"] = datetime.now(timezone.utc).isoformat()
        result["whole_elapsed_seconds"] = time.monotonic()-started_mono
        result["evidence_bytes_before_result"] = evidence_bytes(attempt)
        if result["whole_elapsed_seconds"] >= worker.WALL_SECONDS:
            result["status"] = "invalid_or_partial"
            result["error"] = {"type":"Deadline", "message":"whole wall deadline exceeded"}
        if evidence_bytes(attempt)+len(worker.canonical(result).encode())+1 > worker.EVIDENCE_BYTES:
            result["status"] = "invalid_or_partial"
            result["error"] = {"type":"EvidenceCap", "message":"whole evidence cap exceeded"}
        try:
            worker.write_once(attempt/"parent_result.json", result)
        finally:
            LOCK.rmdir()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=worker.MODELS, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.model, args.attempt, args.admission, args.go)
    print(worker.canonical({"status":result["status"], "attempt":result["attempt"]}))
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
