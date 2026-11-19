"""One-shot external host and timeout gate for the FIT acoustic pilot.

This parent loads no candidate weights. The pilot worker is a subprocess only
after independently committed worker and launcher GO receipts pass. A timed-out
worker process group is killed; an attempt directory is never reused.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import acoustic_fit_noise_pilot_v1 as pilot
import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as host_gate

HERE = Path(__file__).resolve().parent
WORKER = HERE / "acoustic_fit_noise_pilot_v1.py"
PROTOCOL = HERE / "ACOUSTIC_FIT_NOISE_PILOT_PROSPECTIVE_V1.md"
TESTS = HERE / "test_acoustic_fit_noise_parent_launch_v1.py"
HANDOFF = HERE / "ACOUSTIC_FIT_NOISE_PARENT_LAUNCH_SOURCE_HANDOFF_V1.md"
MAX_WORKER_SECONDS = 1800
MAX_CAPTURE_BYTES = 16 * 1024**2


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(pilot.canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_binding() -> dict:
    return {
        "schema": "acoustic_fit_noise_parent_source_v1",
        "launcher_sha256": pilot.sha_file(Path(__file__)),
        "tests_sha256": pilot.sha_file(TESTS),
        "handoff_sha256": pilot.sha_file(HANDOFF),
        "worker_sha256": pilot.sha_file(WORKER),
        "protocol_sha256": pilot.sha_file(PROTOCOL),
        "launcher_commit": pilot.source_commit(Path(__file__)),
        "tests_commit": pilot.source_commit(TESTS),
        "handoff_commit": pilot.source_commit(HANDOFF),
        "worker_commit": pilot.source_commit(WORKER),
        "protocol_commit": pilot.source_commit(PROTOCOL),
        "host_probe_source_sha256": pilot.sha_file(Path(host_gate.__file__)),
        "host_probe_source_commit": pilot.source_commit(Path(host_gate.__file__)),
        "max_worker_seconds": MAX_WORKER_SECONDS,
        "max_capture_bytes": MAX_CAPTURE_BYTES,
    }


def require_launcher_go(path: Path, binding: dict, preflight_sha: str,
                        worker_review_sha: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing independent launcher GO")
    review = json.loads(path.read_text())
    if (review.get("schema") != "acoustic_fit_noise_parent_source_go_v1"
            or review.get("status") != "GO"
            or review.get("source_binding") != binding
            or review.get("preflight_sha256") != preflight_sha
            or review.get("worker_review_sha256") != worker_review_sha):
        raise ValueError("independent launcher GO does not bind current source/receipts")
    review_commit = pilot.source_commit(path)
    for source in (binding["launcher_commit"], binding["tests_commit"],
                   binding["handoff_commit"]):
        if review_commit == source or subprocess.run(
                ("git", "merge-base", "--is-ancestor", source, review_commit),
                cwd=pilot.ROOT, check=False).returncode != 0:
            raise ValueError("launcher review is not later than frozen source")
    return review


def probe_host(attempt: Path, label: str, *, run_start: datetime | None = None) -> dict:
    evidence = host_gate._host_probe(require_stable=True)
    outputs = evidence["outputs"]
    if "Now drawing from 'AC Power'" not in outputs["power"]:
        raise ValueError("host is not on AC power")
    if plistlib.loads(outputs["clamshell"].encode())[0].get("AppleClamshellState") is not False:
        raise ValueError("lid is closed or unknown")
    if plistlib.loads(outputs["battery"].encode())[0].get("ExternalConnected") is not True:
        raise ValueError("external power state differs")
    collected = datetime.fromisoformat(evidence["collected_utc"])
    if run_start is not None:
        host_gate._power_log_coverage(outputs["power_log"], run_start, collected, collected)
        if host_gate._power_events(outputs["power_log"], run_start, collected):
            raise ValueError("sleep, wake or thermal transition during worker slot")
    host_gate._store_host_probe(attempt, evidence, label)
    write_once(attempt / f"{label}_host.json", evidence)
    return evidence


def run_bounded(argv: list[str], stdout: Path, stderr: Path,
                *, timeout_seconds: int = MAX_WORKER_SECONDS) -> tuple[int | None, bool, float]:
    """Launch one process group and kill it on timeout, including descendants."""
    start = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        try:
            process.wait(timeout=max(0.01, timeout_seconds - (time.monotonic() - start)))
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
    return process.returncode, timed_out, time.monotonic() - start


def valid_worker_rows(rows: list[dict], model: str, cases_expected: dict) -> bool:
    cases = [r for r in rows if r.get("type") == "case"]
    observed = {(r.get("case"), r.get("arm")) for r in cases}
    planned = {(case, arm) for case, arms in cases_expected.items() for arm in arms}
    if not (len(rows) == 22 and len(cases) == len(observed) == len(planned) == 20
            and observed == planned and rows[0].get("type") == "start"
            and rows[0].get("model") == model
            and rows[-1].get("type") == "complete"
            and rows[-1].get("planned_cases") == 20):
        return False
    return all(r.get("model") == model
               and r.get("pcm_sha256") == cases_expected[r["case"]][r["arm"]]
               for r in cases)


def execute(model: str, receipt: Path, worker_review: Path,
            launcher_review: Path, attempt: Path) -> dict:
    if model not in ("whisper", "wav2vec2"):
        raise ValueError("only the two frozen model names are admitted")
    if attempt.exists():
        raise FileExistsError("attempt directory already exists; no retry or overwrite")
    current = pilot.preflight()  # no candidate model load
    pilot.require_review(worker_review, receipt, current)
    binding = source_binding()
    require_launcher_go(launcher_review, binding, pilot.sha_file(receipt),
                        pilot.sha_file(worker_review))

    attempt.mkdir(parents=True, exist_ok=False)
    started_utc, started_mono = utc(), time.monotonic()
    start_dt = datetime.fromisoformat(started_utc)
    worker_started = False
    worker_rc: int | None = None
    timed_out = False
    worker_seconds: float | None = None
    errors: list[str] = []
    try:
        probe_host(attempt, "pre")
        write_once(attempt / "launch_start.json", {
            "schema": "acoustic_fit_noise_launch_start_v1", "model": model,
            "start_utc": started_utc, "start_monotonic": started_mono,
            "receipt_sha256": pilot.sha_file(receipt),
            "worker_review_sha256": pilot.sha_file(worker_review),
            "launcher_review_sha256": pilot.sha_file(launcher_review),
            "source_binding": binding,
        })
        argv = [sys.executable, str(WORKER), "run", "--model", model,
                "--receipt", str(receipt), "--review", str(worker_review),
                "--output", str(attempt / "worker_output")]
        worker_started = True
        worker_rc, timed_out, worker_seconds = run_bounded(
            argv, attempt / "worker_stdout.txt", attempt / "worker_stderr.txt")
        if timed_out:
            errors.append("worker_timeout")
        if worker_rc != 0:
            errors.append(f"worker_exit_{worker_rc}")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
    finally:
        if worker_started:
            try:
                probe_host(attempt, "post", run_start=start_dt)
            except BaseException as exc:
                errors.append(f"post_host_{type(exc).__name__}:{exc}")
        ended_utc, ended_mono = utc(), time.monotonic()
        wall_seconds = (datetime.fromisoformat(ended_utc) - start_dt).total_seconds()
        monotonic_seconds = ended_mono - started_mono
        if abs(wall_seconds - monotonic_seconds) > 5:
            errors.append("utc_monotonic_elapsed_gap_gt_5s")
        capture_bytes = sum(p.stat().st_size for p in attempt.rglob("*")
                            if p.is_file() and "worker_output" not in p.parts)
        if capture_bytes > MAX_CAPTURE_BYTES:
            errors.append("parent_capture_over_16MiB")
        worker_output = attempt / "worker_output" / "records.jsonl"
        worker_complete = False
        if worker_output.is_file():
            try:
                rows = [json.loads(line) for line in worker_output.read_text().splitlines()]
                worker_complete = valid_worker_rows(rows, model, current["cases"])
            except (ValueError, OSError) as exc:
                errors.append(f"worker_record_parse_{type(exc).__name__}")
        if worker_started and not worker_complete:
            errors.append("worker_matrix_incomplete")
        result = {
            "schema": "acoustic_fit_noise_parent_result_v1", "model": model,
            "status": "complete" if not errors and worker_complete else "invalid_or_partial",
            "errors": errors, "worker_started": worker_started,
            "worker_returncode": worker_rc, "worker_timed_out": timed_out,
            "worker_elapsed_seconds": worker_seconds, "worker_complete": worker_complete,
            "start_utc": started_utc, "end_utc": ended_utc,
            "start_monotonic": started_mono, "end_monotonic": ended_mono,
            "wall_seconds": wall_seconds, "monotonic_seconds": monotonic_seconds,
            "capture_bytes_before_result": capture_bytes,
        }
        write_once(attempt / "launch_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-only", "run"))
    parser.add_argument("--model", choices=("whisper", "wav2vec2"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--worker-review", type=Path)
    parser.add_argument("--launcher-review", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.command == "validate-only":
        print(pilot.canonical(source_binding()))
        return
    if not all((args.model, args.receipt, args.worker_review,
                args.launcher_review, args.attempt)):
        parser.error("run requires model, receipt, both reviews and exclusive attempt")
    print(pilot.canonical(execute(args.model, args.receipt,
                                  args.worker_review, args.launcher_review,
                                  args.attempt)))


if __name__ == "__main__":
    main()
