"""One-shot whole-attempt host, time and RSS gate for the typed Qwen pilot.

Importing or source-preflighting this file does not load model weights.
The fixed independently bound GO is absent until executable review.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host
from models.second_architecture.nonspeech_probe_v1 import causal_parent_v2 as bounded
from . import typed_pilot_backend_v1 as backend

ATTEMPT = backend.ATTEMPT
WALL_SECONDS = 300
RSS_BYTES = 30 * 1024**3
EVIDENCE_BYTES = backend.MAX_EVIDENCE_BYTES
RESERVE_SECONDS = 20


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bytes(path: Path) -> int:
    return bounded.output_bytes(path)


def _write(attempt: Path, name: str, value: dict) -> None:
    raw = backend.canonical(value)
    backend.require(_bytes(attempt) + len(raw) < EVIDENCE_BYTES,
                    "pilot whole-attempt evidence cap before write")
    backend.write_once(attempt / name, value)


def _probe(attempt: Path, label: str, *, start: datetime,
           deadline: float) -> dict:
    backend.require(time.monotonic() < deadline - RESERVE_SECONDS,
                    "pilot host probe reserve exhausted")
    evidence = host.probe(run_start=start if label == "post" else None)
    backend.require(time.monotonic() < deadline - RESERVE_SECONDS,
                    "pilot host probe deadline exhausted")
    host.validate_power(evidence)
    packed = gzip.compress(evidence["outputs"]["power_log"].encode(), mtime=0)
    backend.require(_bytes(attempt) + len(packed) < EVIDENCE_BYTES,
                    "pilot host archive exceeds evidence cap")
    host.archive(attempt, evidence, label)
    _write(attempt, f"{label}_host.json", evidence)
    return evidence


def launch() -> dict:
    receipt = backend.validate_go()  # Before attempt creation or candidate import.
    backend.require(not ATTEMPT.exists(), "fixed typed pilot attempt already used")
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPT.mkdir()
    began = datetime.now(timezone.utc)
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    _write(ATTEMPT, "start.json", {"schema": "qwen_action_typed_pilot_parent_v1",
        "status": "started", "start_utc": began.isoformat(),
        "approved_attempt": str(ATTEMPT), "go_sha256": backend.sha(backend.GO),
        "source_identity": receipt["source_identity"]})
    result: dict = {"schema": "qwen_action_typed_pilot_parent_v1",
                    "status": "invalid_or_partial", "start_utc": began.isoformat(),
                    "go_sha256": backend.sha(backend.GO)}
    candidate_complete = False
    try:
        pre = _probe(ATTEMPT, "pre", start=began, deadline=deadline)
        granted = deadline-time.monotonic()-RESERVE_SECONDS
        backend.require(granted > 0, "pilot worker has no time allowance")
        result["worker_seconds_granted"] = granted
        command = [sys.executable, "-m",
            "models.second_architecture.qwen_action_method_v1.typed_pilot_backend_v1",
            "--worker", "--attempt", str(ATTEMPT)]
        observation = bounded.run_bounded(command, ATTEMPT, seconds=granted,
            rss_bytes=RSS_BYTES, evidence_bytes=EVIDENCE_BYTES-2*1024**2)
        result.update(observation)
        result["worker_exit_utc"] = _utc()
        backend.require(time.monotonic() < deadline-RESERVE_SECONDS,
                        "pilot worker exhausted parent reserve")
        post = _probe(ATTEMPT, "post", start=began, deadline=deadline)
        result["pre_host_sha256"] = backend.sha(ATTEMPT / "pre_host.json")
        result["post_host_sha256"] = backend.sha(ATTEMPT / "post_host.json")
        result["host_collected_utc"] = [pre["collected_utc"], post["collected_utc"]]
        candidate_complete = (observation["worker_returncode"] == 0
            and observation["worker_stop_reason"] == "normal_exit"
            and not observation["worker_timed_out"]
            and observation["worker_rss_samples"] > 0
            and 0 < observation["worker_peak_tree_rss_bytes"] < RSS_BYTES
            and (ATTEMPT / "worker_output/result.json").is_file()
            and _bytes(ATTEMPT) < EVIDENCE_BYTES)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result["status"] = "candidate_saved_pending_audit" if candidate_complete else "invalid_or_partial"
    result["end_utc"] = _utc()
    result["elapsed_seconds_before_result"] = time.monotonic()-started
    _write(ATTEMPT, "result.json", result)
    if candidate_complete:
        terminal_path = ATTEMPT / "terminal.json"
        try:
            backend.require(time.monotonic() < deadline,
                            "pilot terminal deadline exhausted")
            terminal = {"schema": "qwen_action_typed_pilot_terminal_v1",
                "status": "candidate_saved_pending_audit",
                "result_sha256": backend.sha(ATTEMPT / "result.json"),
                "start_sha256": backend.sha(ATTEMPT / "start.json"),
                "elapsed_seconds_after_result": time.monotonic()-started,
                "terminal_utc": _utc()}
            _write(ATTEMPT, "terminal.json", terminal)
            backend.require(time.monotonic() < deadline,
                            "pilot terminal write exceeded deadline")
        except BaseException:
            if terminal_path.exists():
                terminal_path.unlink()
                fd = os.open(ATTEMPT, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.source_preflight:
        print(json.dumps({"status": "independent_go_required",
                          "attempt_exists": ATTEMPT.exists(),
                          "go_exists": backend.GO.is_file()}, sort_keys=True))
    elif args.run:
        outcome = launch()
        print(json.dumps({"status": outcome["status"],
                          "attempt": str(ATTEMPT)}, sort_keys=True))
        if outcome["status"] != "candidate_saved_pending_audit":
            raise SystemExit(2)
    else:
        parser.error("choose --source-preflight or --run")


if __name__ == "__main__":
    main()
