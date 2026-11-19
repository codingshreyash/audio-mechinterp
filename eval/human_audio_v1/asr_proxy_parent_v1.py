"""One-shot bounded parent for the independent Wav2Vec2 ASR screen."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from models.second_architecture.nonspeech_probe_v1 import whisper_tail_causal_host_v1 as host
from . import asr_proxy_worker_v1 as worker

ROOT = worker.HERE.parents[1]
WALL_SECONDS = 600
RSS_BYTES = 4 * 1024**3
EVIDENCE_BYTES = 256 * 1024**2


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def write_once(path: Path, value: dict) -> None:
    with path.open("x") as out:
        out.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        out.flush()
        os.fsync(out.fileno())


def tracked_clean(path: Path) -> None:
    rel = str(path.relative_to(ROOT))
    need(subprocess.run(("git", "ls-files", "--error-unmatch", "--", rel),
                        cwd=ROOT, capture_output=True).returncode == 0,
         f"untracked source: {rel}")
    need(subprocess.run(("git", "diff", "--quiet", "--", rel),
                        cwd=ROOT, capture_output=True).returncode == 0
         and subprocess.run(("git", "diff", "--cached", "--quiet", "--", rel),
                            cwd=ROOT, capture_output=True).returncode == 0,
         f"modified source: {rel}")


def validate_admission() -> dict:
    for name in worker.SOURCES:
        tracked_clean(worker.HERE / name)
    tracked_clean(worker.ADMISSION)
    need(json.loads(worker.ADMISSION.read_text()) == worker.admission(),
         "source-only admission changed")
    return json.loads(worker.ADMISSION.read_text())


def attempt_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def tree_rss(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        family = [process, *process.children(recursive=True)]
        return sum(p.memory_info().rss for p in family if p.is_running())
    except psutil.NoSuchProcess:
        return 0


def kill_worker(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def launch() -> dict:
    admission = validate_admission()
    before = host.quick_probe(initial=True)
    need(before["external_connected"] is True, "AC power required for ASR proxy")
    attempt = worker.ATTEMPT
    need(not attempt.exists(), "one-shot ASR attempt already exists")
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.mkdir(exist_ok=False)
    write_once(attempt / "host_before.json", before)
    started = time.monotonic()
    launch_record = {"schema": "human_asr_proxy_parent_v1",
                     "worker_parent_pid": os.getpid(),
                     "admission_sha256": worker.sha(worker.ADMISSION),
                     "started_utc": datetime.now(timezone.utc).isoformat(),
                     "started_monotonic": started,
                     "deadline_monotonic": started + WALL_SECONDS}
    write_once(attempt / "launch.json", launch_record)
    reason = "unknown"
    peak_rss = 0
    samples = 0
    exit_code = None
    process = None
    try:
        with (attempt / "stdout.log").open("xb") as stdout, \
             (attempt / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(
                (sys.executable, "-m", "eval.human_audio_v1.asr_proxy_worker_v1",
                 "run", "--attempt", str(attempt), "--admission", str(worker.ADMISSION)),
                cwd=ROOT, stdout=stdout, stderr=stderr)
            with (attempt / "resource_samples.jsonl").open("x") as log:
                while True:
                    now = time.monotonic()
                    rss = tree_rss(process.pid)
                    size = attempt_size(attempt)
                    peak_rss = max(peak_rss, rss)
                    log.write(json.dumps({"monotonic": now, "tree_rss_bytes": rss,
                                          "evidence_bytes": size}, sort_keys=True) + "\n")
                    log.flush()
                    samples += 1
                    if now - started > WALL_SECONDS:
                        reason = "wall_limit"
                        break
                    if rss > RSS_BYTES:
                        reason = "rss_limit"
                        break
                    if size > EVIDENCE_BYTES:
                        reason = "evidence_limit"
                        break
                    exit_code = process.poll()
                    if exit_code is not None:
                        reason = "normal_exit" if exit_code == 0 else "worker_nonzero"
                        break
                    time.sleep(0.2)
            if reason != "normal_exit":
                kill_worker(process)
            exit_code = process.wait(timeout=5)
        after = host.quick_probe(initial=False)
        need(after["external_connected"] is True, "AC lost during ASR proxy")
        write_once(attempt / "host_after.json", after)
        rows = [json.loads(line) for line in (attempt / "records.jsonl").read_text().splitlines()]
        need(len(rows) == 17 and rows[-1] == {"type": "complete", "calls": 16}
             and [r.get("index") for r in rows[:-1]] == list(range(1, 17)),
             "ASR worker output incomplete")
        need(len(list((attempt / "sidecars").glob("*.npy"))) == 32,
             "ASR sidecar census incomplete")
        need(reason == "normal_exit" and peak_rss <= RSS_BYTES
             and attempt_size(attempt) <= EVIDENCE_BYTES,
             "bounded ASR worker failed")
        status = "completed_pending_saved_audit"
    except Exception as error:
        if process is not None:
            kill_worker(process)
            exit_code = process.returncode
        status = "invalid_or_partial"
        reason = f"{reason}:{type(error).__name__}:{error}"
    result = {"schema": "human_asr_proxy_parent_result_v1", "status": status,
              "reason": reason, "attempt": str(attempt), "exit_code": exit_code,
              "whole_elapsed_seconds": time.monotonic() - started,
              "peak_tree_rss_bytes": peak_rss, "resource_samples": samples,
              "admission_sha256": worker.sha(worker.ADMISSION),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_once(attempt / "parent_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = launch()
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
