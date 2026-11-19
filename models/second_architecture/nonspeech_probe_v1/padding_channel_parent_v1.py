"""One-shot host, process-tree, and evidence parent for padding assay."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate
from . import causal_parent_v2 as bounded
from . import padding_channel_plan_v4 as plan
from . import padding_channel_worker_v1 as worker

ATTEMPT = worker.ATTEMPT
MAX_POST_PROBE_LAG_SECONDS = 120


def expected_names() -> list[str]:
    return [worker.call_name(row) for row in plan.schedule()]


def host_probe(attempt: Path, label: str, *, run_start: datetime | None = None) -> dict:
    evidence = host_gate.probe(run_start=run_start)
    host_gate.archive(attempt, evidence, label)
    worker.write_once(attempt / f"{label}_host.json", evidence)
    return evidence


def launch(admission: Path, go: Path) -> dict:
    receipt = worker.validate_go(admission, go)
    names = expected_names()
    worker.require(len(names) == 436 and not ATTEMPT.exists(),
                   "attempt path or call census invalid")
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPT.mkdir()
    start = datetime.now(timezone.utc)
    worker.write_once(ATTEMPT / "launch_start.json", {
        "schema": "whisper_padding_launch_v1", "status": "started",
        "start_utc": start.isoformat(), "attempt": str(ATTEMPT),
        "admission_sha256": worker.sha_file(admission),
        "go_sha256": worker.sha_file(go),
        "source_identity": receipt["sources"], "scheduled_call_names": names})
    result: dict = {"schema": "whisper_padding_launch_v1",
                    "status": "invalid_or_partial", "attempt": str(ATTEMPT),
                    "start_utc": start.isoformat(),
                    "admission_sha256": worker.sha_file(admission),
                    "go_sha256": worker.sha_file(go)}
    worker_exit: datetime | None = None
    try:
        pre = host_probe(ATTEMPT, "pre")
        worker.require(bounded.output_bytes(ATTEMPT) < plan.EVIDENCE_LIMIT_BYTES,
                       "pre-run evidence ceiling exceeded")
        command = [sys.executable, "-m",
                   "models.second_architecture.nonspeech_probe_v1.padding_channel_worker_v1",
                   "run", "--output", str(ATTEMPT / "worker_output"),
                   "--admission", str(admission.resolve()), "--go", str(go.resolve())]
        observation = bounded.run_bounded(
            command, ATTEMPT, seconds=plan.WALL_SECONDS,
            rss_bytes=plan.RSS_LIMIT_BYTES,
            evidence_bytes=plan.EVIDENCE_LIMIT_BYTES)
        worker_exit = datetime.now(timezone.utc)
        result.update(observation)
        result["worker_exit_observed_utc"] = worker_exit.isoformat()
        progress = ATTEMPT / "worker_output/call_progress.jsonl"
        result["worker_call_progress_sha256"] = (
            worker.sha_file(progress) if progress.is_file() else None)
        result["worker_call_census"] = bounded.progress_census(progress, names)
        post = host_probe(ATTEMPT, "post", run_start=start)
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
            result["status"] = "completed_pending_saved_audit"
        worker.require(bounded.output_bytes(ATTEMPT) <= plan.EVIDENCE_LIMIT_BYTES,
                       "parent evidence ceiling exceeded")
    except BaseException as error:
        result["launch_error"] = {"type": type(error).__name__, "message": str(error)}
    result["end_utc"] = datetime.now(timezone.utc).isoformat()
    result["evidence_bytes_before_result"] = bounded.output_bytes(ATTEMPT)
    worker.write_once(ATTEMPT / "launch_result.json", result)
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
