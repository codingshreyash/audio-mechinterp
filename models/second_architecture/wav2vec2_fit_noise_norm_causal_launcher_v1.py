"""One-shot Mac host and wall-clock gate for the FIT normalization worker."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_worker_v1 as worker
from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "runs/wav2vec2_fit_noise_norm_causal_v1/attempt_001"


def run_bounded(argv: list[str], stdout: Path, stderr: Path,
                *, seconds: int = worker.MAX_SECONDS) -> tuple[int, bool, float]:
    start = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(argv, cwd=worker.ROOT, stdout=out, stderr=err,
                                   start_new_session=True,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1",
                                        "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"})
        try:
            process.wait(timeout=seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            raise
    elapsed = time.monotonic() - start
    return process.returncode, timed_out or not math.isfinite(elapsed) or elapsed > seconds, elapsed


def probe_host(attempt: Path, name: str, *, run_start: datetime | None = None) -> dict:
    probe = host_gate.probe(run_start=run_start)
    host_gate.archive(attempt, probe, name)
    worker.write_once(attempt / f"{name}_host.json", probe)
    return probe


def launch(admission_path: Path, go_path: Path) -> dict:
    admission = worker.validate_go(admission_path, go_path)
    worker.require(not ATTEMPT.exists(), "frozen one-shot attempt already exists")
    ATTEMPT.mkdir(parents=True)
    start_utc = datetime.now(timezone.utc)
    worker.write_once(ATTEMPT / "launch_start.json",
                      {"schema": "fit_norm_causal_launch_v1",
                       "status": "started", "start_utc": start_utc.isoformat(),
                       "admission_sha256": worker.source.sha_file(admission_path),
                       "go_sha256": worker.source.sha_file(go_path),
                       "source_binding": admission["source_binding"],
                       "attempt": str(ATTEMPT)})
    result = {"schema": "fit_norm_causal_launch_v1", "status": "invalid_or_partial",
              "start_utc": start_utc.isoformat(), "attempt": str(ATTEMPT),
              "admission_sha256": worker.source.sha_file(admission_path),
              "go_sha256": worker.source.sha_file(go_path)}
    try:
        pre = probe_host(ATTEMPT, "pre")
        worker.require(worker.output_bytes(ATTEMPT) < worker.MAX_OUTPUT,
                       "pre-run output ceiling exceeded")
        cmd = [sys.executable, "-m",
               "models.second_architecture.wav2vec2_fit_noise_norm_causal_worker_v1",
               "run", "--output", str(ATTEMPT / "worker_output"),
               "--admission", str(admission_path.resolve()),
               "--go", str(go_path.resolve())]
        rc, timeout, elapsed = run_bounded(
            cmd, ATTEMPT / "worker_stdout.txt", ATTEMPT / "worker_stderr.txt")
        result.update(worker_returncode=rc, worker_timed_out=timeout,
                      worker_elapsed_seconds=elapsed)
        post = probe_host(ATTEMPT, "post", run_start=start_utc)
        result["pre_host_sha256"] = worker.source.sha_file(ATTEMPT / "pre_host.json")
        result["post_host_sha256"] = worker.source.sha_file(ATTEMPT / "post_host.json")
        result["host_collected_utc"] = [pre["collected_utc"], post["collected_utc"]]
        records = ATTEMPT / "worker_output/records.jsonl"
        if rc == 0 and not timeout and records.is_file():
            lines = records.read_text().splitlines()
            if lines and json.loads(lines[-1]).get("type") == "complete":
                result["status"] = "completed_pending_saved_audit"
        worker.require(worker.output_bytes(ATTEMPT) <= worker.MAX_OUTPUT,
                       "parent evidence ceiling exceeded")
    except BaseException as error:
        result["launch_error"] = {"type": type(error).__name__, "message": str(error)}
    result["end_utc"] = datetime.now(timezone.utc).isoformat()
    result["evidence_bytes_before_result"] = worker.output_bytes(ATTEMPT)
    worker.write_once(ATTEMPT / "launch_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.admission, args.go)
    print(worker.canonical(result))
    if result["status"] != "completed_pending_saved_audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
