"""One-checkpoint, one-shot bounded parent for the natural-tail V4 assay."""
from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
from . import causal_parent_v2 as bounded
from . import natural_tail_runner_v4 as worker

ROOT = worker.ROOT
LOCK = worker.HERE/"runs/natural_tail_waveform_v1/active_model_lock"
POST_RESERVE_SECONDS = 480
RSS_POLL_SECONDS = 0.1
EXIT_RACE_GRACE_SECONDS = 1.0


def grace_sleep_seconds(grace_end: float, *, now: float | None = None) -> float:
    return max(0.0,min(RSS_POLL_SECONDS,
                       grace_end-(time.monotonic() if now is None else now)))


def complete_worker_output(attempt: Path, plan: list[dict]) -> bool:
    """Require complete ordered progress, terminal records, and 5 sidecars/call."""
    try:
        progress = progress_census(attempt/"progress.jsonl", plan)
        if progress != {"complete":len(plan),"attempted_incomplete":None,"not_attempted":0}:
            return False
        records_path = attempt/"records.jsonl"
        if not records_path.is_file() or records_path.is_symlink():
            return False
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        if len(records) != len(plan)+1 or records[-1] != {"type":"complete","calls":len(plan)}:
            return False
        sidecars = attempt/"sidecars"
        if not sidecars.is_dir() or sidecars.is_symlink():
            return False
        names = set()
        for row,record in zip(plan,records[:-1],strict=True):
            if record.get("type") != "call" or record.get("plan") != row:
                return False
            refs = record.get("sidecars",{})
            if set(refs) != {"mel","mask","first_raw_logits",
                             "first_processed_scores","first_decoder_state"}:
                return False
            for ref in refs.values():
                name = ref.get("file")
                if not isinstance(name,str) or name != Path(name).name:
                    return False
                path = sidecars/name
                if not path.is_file() or path.is_symlink():
                    return False
                names.add(name)
        return len(names) == 5*len(plan) and names == {p.name for p in sidecars.iterdir()}
    except (OSError,ValueError,KeyError,TypeError):
        return False


def remaining_worker_seconds(deadline: float, *, now: float | None = None) -> float:
    remaining = deadline-(time.monotonic() if now is None else now)-POST_RESERVE_SECONDS
    worker.require(remaining > 0, "pre-host probe exhausted worker budget")
    return remaining


def begin_host_window(*, probe=host_gate.probe, clock=time.monotonic) -> tuple:
    started_mono = clock()
    deadline = started_mono+worker.WALL_SECONDS
    started_utc = datetime.now(timezone.utc)
    before = probe()
    remaining_worker_seconds(deadline, now=clock())
    return started_mono,deadline,started_utc,before


def run_bounded(command: list[str], attempt: Path, *, nonce: str,
                deadline: float, seconds: float, plan: list[dict]) -> dict:
    """Pass a one-use pipe to the worker and monitor the whole process tree."""
    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    peak = samples = beats = 0
    reason = "normal_exit"
    monitor_error = None
    process = None
    log = attempt/"worker_rss_samples.jsonl"
    try:
        with (attempt/"worker_stdout.txt").open("xb") as stdout, \
                (attempt/"worker_stderr.txt").open("xb") as stderr, \
                log.open("x",encoding="utf-8") as sample_file:
            process = subprocess.Popen(command+["--heartbeat-fd",str(read_fd)],
                cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,
                pass_fds=(read_fd,),env={**os.environ,"HF_HUB_OFFLINE":"1",
                                      "TRANSFORMERS_OFFLINE":"1","OMP_NUM_THREADS":"4"})
            os.close(read_fd); read_fd = -1
            os.write(write_fd,b"HELLO "+nonce.encode("ascii")+b"\n")
            next_beat = time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline-POST_RESERVE_SECONDS or now-started >= seconds:
                    reason = "wall_limit"
                    bounded.kill_group(process)
                    break
                try:
                    rss = bounded.process_tree_rss_bytes(process.pid)
                    size = evidence_bytes(attempt)
                except (OSError,ValueError,subprocess.SubprocessError) as error:
                    reason,monitor_error = "monitor_error",repr(error)
                    bounded.kill_group(process)
                    break
                if rss is not None:
                    samples += 1
                    peak = max(peak,rss)
                    sample_file.write(worker.canonical({
                        "sample_index":samples,"elapsed_seconds":now-started,
                        "worker_pid":process.pid,"tree_rss_bytes":rss,
                        "evidence_bytes":size})+"\n")
                    sample_file.flush()
                    if rss >= worker.RSS_BYTES:
                        reason = "rss_limit"
                        bounded.kill_group(process)
                        break
                if size >= worker.EVIDENCE_BYTES-worker.EVIDENCE_RESERVE:
                    reason = "evidence_limit"
                    bounded.kill_group(process)
                    break
                if now >= next_beat:
                    try:
                        os.write(write_fd,b"H\n")
                        beats += 1
                    except BrokenPipeError:
                        grace_end = min(time.monotonic()+EXIT_RACE_GRACE_SECONDS,
                                        deadline-POST_RESERVE_SECONDS,started+seconds)
                        code = process.poll()
                        while code is None and time.monotonic() < grace_end:
                            grace_now = time.monotonic()
                            try:
                                rss = bounded.process_tree_rss_bytes(process.pid)
                                size = evidence_bytes(attempt)
                            except (OSError,ValueError,subprocess.SubprocessError) as error:
                                reason,monitor_error = "monitor_error",repr(error)
                                bounded.kill_group(process)
                                break
                            if rss is not None:
                                samples += 1
                                peak = max(peak,rss)
                                sample_file.write(worker.canonical({
                                    "sample_index":samples,"elapsed_seconds":grace_now-started,
                                    "worker_pid":process.pid,"tree_rss_bytes":rss,
                                    "evidence_bytes":size})+"\n")
                                sample_file.flush()
                                if rss >= worker.RSS_BYTES:
                                    reason = "rss_limit"
                                    bounded.kill_group(process)
                                    break
                            if size >= worker.EVIDENCE_BYTES-worker.EVIDENCE_RESERVE:
                                reason = "evidence_limit"
                                bounded.kill_group(process)
                                break
                            sleep_for = grace_sleep_seconds(grace_end)
                            if sleep_for > 0:
                                time.sleep(sleep_for)
                            code = process.poll()
                        if reason == "normal_exit":
                            if code == 0 and complete_worker_output(attempt,plan):
                                reason = "normal_exit"
                            else:
                                reason = "heartbeat_pipe_lost"
                                if process.poll() is None:
                                    bounded.kill_group(process)
                        break
                    next_beat=now+0.25
                time.sleep(RSS_POLL_SECONDS)
            process.wait(timeout=10)
            sample_file.flush();os.fsync(sample_file.fileno())
    except BaseException:
        if process is not None and process.poll() is None:
            bounded.kill_group(process)
        raise
    finally:
        if read_fd >= 0:os.close(read_fd)
        os.close(write_fd)
    elapsed = time.monotonic()-started
    return {"worker_returncode":process.returncode,
            "worker_stop_reason":reason,
            "worker_timed_out":reason == "wall_limit" or not math.isfinite(elapsed)
            or elapsed > seconds,
            "worker_elapsed_seconds":elapsed,
            "worker_peak_tree_rss_bytes":peak,
            "worker_rss_limit_bytes":worker.RSS_BYTES,
            "worker_rss_samples":samples,
            "worker_rss_sample_log_sha256":worker.old.sha_file(log),
            "worker_evidence_limit_hit":reason == "evidence_limit",
            "worker_rss_limit_hit":reason == "rss_limit",
            "worker_monitor_error":monitor_error,
            "parent_heartbeat_count":beats}


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
    worker.require(model in worker.MODELS and attempt == worker.ATTEMPTS[model]
                   and attempt.is_absolute() and attempt.parent.is_dir()
                   and not attempt.exists(), "new absolute attempt path required")
    worker.require(shutil.disk_usage(attempt.parent).free >= 2*worker.EVIDENCE_BYTES,
                   "free disk below 2 GiB")
    worker.require(psutil.virtual_memory().available >= 5*1024**3,
                   "memory headroom below 5 GiB")
    worker.require(not LOCK.exists(), "another natural-tail checkpoint is active")
    # This clock includes the pre-run host probe, unlike V2.
    started_mono,deadline,started_utc,before = begin_host_window()
    nonce = secrets.token_hex(16)
    LOCK.mkdir()
    try:
        attempt.mkdir()
    except BaseException:
        LOCK.rmdir()
        raise
    result = {"schema":"whisper_natural_tail_parent_v4", "model":model,
              "status":"invalid_or_partial", "attempt":str(attempt),
              "start_utc":started_utc.isoformat(),
              "parent_pid":os.getpid(),"deadline_monotonic":deadline,
              "nonce_sha256":worker.old.sha_bytes(nonce.encode("ascii")),
              "admission_sha256":worker.old.sha_file(admission),
              "go_sha256":worker.old.sha_file(go),
              "source_identity":receipt["source_identity"]}
    try:
        host_gate.archive(attempt, before, "pre")
        worker.write_once(attempt/"pre_host.json", before)
        worker.write_once(attempt/"launch_start.json", {**result, "status":"started"})
        remaining = remaining_worker_seconds(deadline)
        result["worker_seconds_granted"] = remaining
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.natural_tail_runner_v4",
                   "run", "--model", model, "--attempt", str(attempt),
                   "--admission", str(admission), "--review", str(go)]
        observation = run_bounded(command,attempt,nonce=nonce,
                                  deadline=deadline,seconds=remaining,
                                  plan=receipt["schedule"])
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
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.model, worker.ATTEMPTS[args.model], args.admission, args.go)
    print(worker.canonical({"status":result["status"], "attempt":result["attempt"]}))
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
