"""Fail-closed one-shot gate for the distinct old-FIT graph/no-grad diagnostic.

Source-only until independent source reviews, a later external freeze, and a
still-later freeze review are committed. This never reopens Stage-B FP32.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import voxtral_old_fit_handoff_stage_b_launch_gate_v1 as old_gate
import voxtral_old_fit_grad_mode_parity_runner_v1 as runner


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs/voxtral_old_fit_grad_mode_parity_launch_v1"
FREEZE = HERE / "VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_EXTERNAL_FREEZE_V1.json"
FREEZE_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_EXTERNAL_FREEZE_INDEPENDENT_REVIEW_V1.md"
SOURCE_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_INDEPENDENT_SOURCE_REVIEW_V2.md"
AUDITOR_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_AUDITOR_INDEPENDENT_SOURCE_REVIEW_V2.md"
AUDITOR = HERE / "voxtral_old_fit_grad_mode_parity_audit_v1.py"
AUDITOR_TESTS = HERE / "test_voxtral_old_fit_grad_mode_parity_audit_v1.py"
TESTS = HERE / "test_voxtral_old_fit_grad_mode_parity_launch_gate_v1.py"
RUNNER_TESTS = HERE / "test_voxtral_old_fit_grad_mode_parity_runner_v1.py"
FAILED_MANIFEST_SHA = "24514b86315355c6f353f0bf5e7fc8518d4c3ccf16e06f195a90aca703d2aa97"
FAILED_REVIEW_SHA = "67a556f3d05698f577e149d27fd48072c29099d732175695dae26d7ff3447926"
FAILED_ID = "20261118T143734566440Z"
FAILED_SECONDS = 41.18471045792103
SOURCE_GO = "Decision: GO for graph/no-grad diagnostic source only; live model HOLD until external freeze."
AUDITOR_GO = "Decision: GO for graph/no-grad CPU auditor source only; live model HOLD until external freeze."
FREEZE_GO = "Decision: GO for graph/no-grad external freeze provenance; one diagnostic attempt may run under the pinned gate."
STATUS = "frozen_old_fit_graph_no_grad_diagnostic_v1"
MAX_CAPTURE_BYTES = 8 * 1024**2
MAX_OUTPUT_BYTES = 1024**3
AUDIT_SECONDS = 120
FINISH_RESERVE_BYTES = 64 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ATTEMPT = re.compile(r"[0-9]{8}T[0-9]{12}Z\Z")
BOUND = {
    "protocol_sha256": runner.PROTOCOL,
    "runner_sha256": Path(runner.__file__).resolve(),
    "runner_tests_sha256": RUNNER_TESTS,
    "gate_sha256": Path(__file__).resolve(),
    "gate_tests_sha256": TESTS,
    "auditor_sha256": AUDITOR,
    "auditor_tests_sha256": AUDITOR_TESTS,
    "source_review_sha256": SOURCE_REVIEW,
    "auditor_review_sha256": AUDITOR_REVIEW,
    "failed_manifest_sha256": runner.FAILED_MANIFEST,
    "failed_review_sha256": runner.FAILED_REVIEW,
    "old_fp32_freeze_sha256": old_gate.FP32_FREEZE,
    "old_fp32_freeze_review_sha256": old_gate.FP32_FREEZE_REVIEW,
}
REVIEWED_KEYS = ("protocol_sha256", "runner_sha256", "runner_tests_sha256",
                 "gate_sha256", "gate_tests_sha256", "auditor_sha256",
                 "auditor_tests_sha256")


def _json(path: Path) -> dict:
    value = json.loads(path.read_bytes(),
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _tree_bytes(path: Path) -> int:
    if path.is_symlink():
        raise ValueError("Diagnostic output root is symlinked")
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError("Diagnostic output contains symlink")
        if child.is_file():
            total += child.stat().st_size
    return total


def verify_failed_provenance() -> tuple[dict, dict, float]:
    """Bind the failed attempt as failed, including its exact ten raw files."""
    for path, expected in ((runner.FAILED_MANIFEST, FAILED_MANIFEST_SHA),
                           (runner.FAILED_REVIEW, FAILED_REVIEW_SHA)):
        old_gate.committed_same(REPO, path)
        old_gate.first_add(REPO, path)
        if runner.sha(path) != expected:
            raise ValueError("Failed Stage-B provenance hash differs")
    old_gate.ancestor(REPO, old_gate.first_add(REPO, runner.FAILED_MANIFEST),
                      old_gate.first_add(REPO, runner.FAILED_REVIEW))
    if ("Decision: GO for failed-attempt provenance; no Stage-B retry authorized."
            not in runner.FAILED_REVIEW.read_text() or
            FAILED_MANIFEST_SHA not in runner.FAILED_REVIEW.read_text()):
        raise ValueError("Independent failed-attempt review differs")
    manifest = _json(runner.FAILED_MANIFEST)
    expected_files = {
        f"runs/voxtral_old_fit_handoff_stage_b_launch_v1/fp32_replay/{FAILED_ID}/{name}"
        for name in ("launch_start.json", "worker_claim.json", "launch_finish.json",
                     "worker_stdout.txt", "worker_stderr.txt")
    } | {
        f"runs/voxtral_old_fit_handoff_stage_b_v1/fp32_replay/{FAILED_ID}/{name}"
        for name in ("meta.json", "progress.jsonl", "error.json", "model_layout.json")
    } | {"runs/voxtral_old_fit_handoff_stage_b_launch_v1/fp32_replay_reservation.json"}
    rows = manifest.get("files")
    if (manifest.get("schema") !=
            "voxtral_old_fit_handoff_stage_b_fp32_failed_attempt_manifest_v1" or
            manifest.get("status") != "frozen_failed_before_saved_result_or_audit_no_retry" or
            manifest.get("attempt_id") != FAILED_ID or
            manifest.get("launcher_status") != "launch_failed" or
            manifest.get("worker_error") != "Direct LM replay mismatch at lm0" or
            manifest.get("parent_model_work_elapsed_seconds") != FAILED_SECONDS or
            manifest.get("runner_returncode") != 1 or
            manifest.get("auditor_returncode") is not None or
            manifest.get("result_present") is not False or
            manifest.get("audit_present") is not False or
            manifest.get("tensor_sidecar_present") is not False or
            not isinstance(rows, list) or len(rows) != 10 or
            {row.get("path") for row in rows} != expected_files):
        raise ValueError("Failed Stage-B attempt descriptor differs")
    total = 0
    for row in rows:
        path = HERE / row["path"]
        old_gate.committed_same(REPO, path)
        old_gate.first_add(REPO, path)
        if (row.get("git_committed") is not True or
                row.get("sha256") != runner.sha(path) or
                row.get("bytes") != path.stat().st_size):
            raise ValueError("Failed Stage-B raw file differs")
        total += path.stat().st_size
    if total != manifest.get("total_raw_bytes") or total != 164304:
        raise ValueError("Failed Stage-B raw byte total differs")
    for directory in (old_gate.RUNS / "fp32_replay" / FAILED_ID,
                      runner.driver.RUNS / "fp32_replay" / FAILED_ID):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Failed Stage-B attempt directory absent")
    if ({path.relative_to(HERE).as_posix()
         for directory in (old_gate.RUNS / "fp32_replay" / FAILED_ID,
                           runner.driver.RUNS / "fp32_replay" / FAILED_ID)
         for path in directory.iterdir()} | {
             "runs/voxtral_old_fit_handoff_stage_b_launch_v1/fp32_replay_reservation.json"
         }) != expected_files:
        raise ValueError("Failed Stage-B raw attempt file set differs")
    if (runner.driver.RUNS / "fp32_replay" / FAILED_ID / "result.json").exists():
        raise ValueError("Failed Stage-B result appeared after review")
    return manifest, old_gate.verify_phase_freeze("fp32_replay")[1], \
        float(manifest["prior_model_work_seconds"]) + FAILED_SECONDS


def verify_freeze() -> tuple[dict, dict]:
    """All admission checks are read-only; missing new review/freeze is HOLD."""
    failed, stage_a, prior_seconds = verify_failed_provenance()
    for key, path in BOUND.items():
        old_gate.committed_same(REPO, path)
    if runner.sha(runner.FAILED_MANIFEST) != FAILED_MANIFEST_SHA or \
            runner.sha(runner.FAILED_REVIEW) != FAILED_REVIEW_SHA:
        raise ValueError("Failed-attempt fixed hash differs")
    if (SOURCE_GO not in SOURCE_REVIEW.read_text() or
            AUDITOR_GO not in AUDITOR_REVIEW.read_text()):
        raise ValueError("Independent diagnostic source review absent")
    source_commit = old_gate.first_add(REPO, SOURCE_REVIEW)
    auditor_commit = old_gate.first_add(REPO, AUDITOR_REVIEW)
    if source_commit == auditor_commit:
        raise ValueError("Diagnostic source and auditor reviews share a commit")
    for key in ("protocol_sha256", "runner_sha256", "runner_tests_sha256",
                "gate_sha256", "gate_tests_sha256"):
        path = BOUND[key]
        rel = old_gate.regular(REPO, path)
        if old_gate._git(REPO, "show", f"{source_commit}:{rel}") != path.read_bytes():
            raise ValueError(f"Source review does not contain reviewed {key}")
        source_change = old_gate._git(REPO, "log", "-1", "--format=%H", "HEAD", "--", rel).decode().strip()
        if source_change == source_commit:
            raise ValueError(f"Source review self-authored {key}")
        old_gate.ancestor(REPO, source_change, source_commit)
    for key in ("auditor_sha256", "auditor_tests_sha256"):
        path = BOUND[key]
        rel = old_gate.regular(REPO, path)
        if old_gate._git(REPO, "show", f"{auditor_commit}:{rel}") != path.read_bytes():
            raise ValueError(f"Auditor review does not contain reviewed {key}")
        source_change = old_gate._git(REPO, "log", "-1", "--format=%H", "HEAD", "--", rel).decode().strip()
        if source_change == auditor_commit:
            raise ValueError(f"Auditor review self-authored {key}")
        old_gate.ancestor(REPO, source_change, auditor_commit)
    old_gate.committed_same(REPO, FREEZE)
    freeze = _json(FREEZE)
    expected = {"status", "attempt_scope", "source_review_commit",
                "auditor_review_commit", "reviewed_utc", "process_cap_seconds",
                "max_mps_driver_bytes", "max_output_bytes", "max_capture_bytes",
                "max_worker_bytes",
                "prior_model_work_seconds", "failed_attempt_seconds",
                "full_forwards", "direct_forwards", "vjps", "generations",
                *BOUND}
    if (set(freeze) != expected or freeze["status"] != STATUS or
            freeze["attempt_scope"] != runner.SCOPE or
            freeze["source_review_commit"] != source_commit or
            freeze["auditor_review_commit"] != auditor_commit or
            freeze["process_cap_seconds"] != runner.MODEL_SECONDS or
            freeze["max_mps_driver_bytes"] != runner.MAX_DRIVER_BYTES or
            freeze["max_output_bytes"] != MAX_OUTPUT_BYTES or
            freeze["max_worker_bytes"] != runner.MAX_WORKER_BYTES or
            freeze["max_capture_bytes"] != MAX_CAPTURE_BYTES or
            freeze["prior_model_work_seconds"] != prior_seconds or
            freeze["failed_attempt_seconds"] != FAILED_SECONDS or
            (freeze["full_forwards"], freeze["direct_forwards"],
             freeze["vjps"], freeze["generations"]) != (8, 16, 0, 0)):
        raise ValueError("External diagnostic freeze schema or caps differ")
    old_gate.utc(freeze["reviewed_utc"])
    for key, path in BOUND.items():
        if freeze[key] != runner.sha(path) or HEX64.fullmatch(freeze[key]) is None:
            raise ValueError(f"Diagnostic freeze file pin differs: {key}")
    freeze_commit = old_gate.first_add(REPO, FREEZE)
    for earlier in (source_commit, auditor_commit,
                    old_gate.first_add(REPO, runner.FAILED_REVIEW)):
        if earlier == freeze_commit:
            raise ValueError("External freeze shares review commit")
        old_gate.ancestor(REPO, earlier, freeze_commit)
    old_gate.committed_same(REPO, FREEZE_REVIEW)
    review_commit = old_gate.first_add(REPO, FREEZE_REVIEW)
    if (review_commit == freeze_commit or FREEZE_GO not in FREEZE_REVIEW.read_text() or
            runner.sha(FREEZE) not in FREEZE_REVIEW.read_text()):
        raise ValueError("Independent external freeze review absent")
    old_gate.ancestor(REPO, freeze_commit, review_commit)
    if (stage_a["original_bf16_launch_status"] != "launch_failed" or
            prior_seconds + runner.MODEL_SECONDS >= runner.driver.MAX_TOTAL_SECONDS):
        raise ValueError("Old-FIT provenance or full time reservation differs")
    return freeze, {"stage_a": stage_a, "prior_model_work_seconds": prior_seconds,
                    "failed_manifest_sha256": FAILED_MANIFEST_SHA}


def verify_worker_start(attempt_id: str, freeze_sha256: str) -> dict:
    """Exclusive worker claim precedes model construction or output creation."""
    if ATTEMPT.fullmatch(attempt_id) is None:
        raise ValueError("Diagnostic attempt ID invalid")
    frozen, upstream = verify_freeze()
    if freeze_sha256 != runner.sha(FREEZE):
        raise ValueError("Diagnostic worker freeze differs")
    reserved = _json(RUNS / "reservation.json")
    start = _json(RUNS / attempt_id / "launch_start.json")
    if (reserved != {"status": "diagnostic_reserved_no_retry", "attempt_id": attempt_id,
                     "freeze_sha256": freeze_sha256} or
            start.get("status") != "diagnostic_launch_started" or
            start.get("attempt_id") != attempt_id or
            start.get("freeze_sha256") != freeze_sha256 or
            old_gate.utc(start.get("wall_utc")) <= old_gate.utc(frozen["reviewed_utc"]) or
            (runner.RUNS / attempt_id).exists()):
        raise ValueError("Diagnostic one-shot launch chronology differs")
    _put(RUNS / attempt_id / "worker_claim.json", {
        "status": "diagnostic_worker_claimed_no_retry", "attempt_id": attempt_id,
        "freeze_sha256": freeze_sha256,
        "wall_utc": datetime.now(timezone.utc).isoformat(),
    })
    return upstream


def _bounded(command: list[str], capture: Path, names: tuple[str, str],
             output: Path, seconds: int) -> tuple[int | None, float, str | None]:
    began = time.monotonic()
    reason: str | None = None
    with (capture / names[0]).open("xb") as stdout, \
            (capture / names[1]).open("xb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                env={**os.environ, "HF_HUB_OFFLINE": "1",
                                     "TRANSFORMERS_OFFLINE": "1"},
                                start_new_session=True)
        while True:
            code = proc.poll()
            elapsed = time.monotonic() - began
            if elapsed >= seconds:
                reason = "diagnostic_process_timeout"
            elif max((capture / names[0]).stat().st_size,
                     (capture / names[1]).stat().st_size) >= MAX_CAPTURE_BYTES:
                reason = "diagnostic_capture_byte_cap"
            elif _tree_bytes(capture) + _tree_bytes(output) >= \
                    MAX_OUTPUT_BYTES - FINISH_RESERVE_BYTES:
                reason = "diagnostic_combined_output_byte_cap"
            if reason and code is None:
                os.killpg(proc.pid, signal.SIGKILL if elapsed >= seconds else signal.SIGTERM)
                try:
                    code = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait(timeout=5)
                break
            if reason or code is not None:
                break
            time.sleep(0.1)
    return code, time.monotonic() - began, reason


def _write_finish_preserving_cap(capture: Path, output: Path, finish: dict) -> None:
    streams = ("worker_stdout.txt", "worker_stderr.txt",
               "audit_stdout.txt", "audit_stderr.txt")

    def trim(name: str, prefix_bytes: int = 512) -> None:
        stream = capture / name
        if not stream.is_file():
            return
        original_bytes = stream.stat().st_size
        original_sha = runner.sha(stream)
        with stream.open("r+b") as handle:
            prefix = handle.read(prefix_bytes)
            handle.truncate(0)
            handle.seek(0)
            handle.write(prefix)
        finish.setdefault("trimmed_streams", {})[name] = {
            "original_bytes": original_bytes,
            "original_sha256": original_sha,
            "retained_prefix_bytes": len(prefix),
        }

    for name in streams:
        stream = capture / name
        if stream.is_file() and stream.stat().st_size >= MAX_CAPTURE_BYTES:
            finish["status"] = "diagnostic_launch_failed_no_retry"
            finish["failure_reason"] = "diagnostic_capture_byte_cap"
            finish["exception_type"] = "OutputLimitError"
            trim(name)
    finish["worker_output_bytes"] = _tree_bytes(output)
    finish["launcher_output_bytes_before_finish"] = _tree_bytes(capture)
    if (finish["worker_output_bytes"] + finish["launcher_output_bytes_before_finish"] +
            FINISH_RESERVE_BYTES >= MAX_OUTPUT_BYTES):
        # Preserve the failure receipt if a subprocess overshot a polled
        # output limit. The worker cap leaves room for both captures, so
        # trimming only untrusted stream bytes is sufficient here.
        finish["status"] = "diagnostic_launch_failed_no_retry"
        finish["failure_reason"] = "diagnostic_combined_output_byte_cap"
        finish["exception_type"] = "OutputLimitError"
        for name in streams:
            if name not in finish.get("trimmed_streams", {}):
                trim(name)
        finish["launcher_output_bytes_before_finish"] = _tree_bytes(capture)
        if (finish["worker_output_bytes"] + finish["launcher_output_bytes_before_finish"] +
                FINISH_RESERVE_BYTES >= MAX_OUTPUT_BYTES):
            # A corrupted/external worker tree can exceed its own cap.
            # Retain an explicit failure receipt even then.
            finish["output_cap_unenforceable_external_mutation"] = True
    _put(capture / "launch_finish.json", finish)


def launch(attempt_id: str) -> dict:
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Diagnostic launch requires pinned second_architecture environment")
    if ATTEMPT.fullmatch(attempt_id) is None:
        raise ValueError("Diagnostic attempt ID invalid")
    frozen, upstream = verify_freeze()
    freeze_sha = runner.sha(FREEZE)
    capture = RUNS / attempt_id
    output = runner.RUNS / attempt_id
    if RUNS.is_symlink() or runner.RUNS.is_symlink():
        raise ValueError("Diagnostic run root is symlinked")
    RUNS.mkdir(parents=True, exist_ok=True)
    runner.RUNS.mkdir(parents=True, exist_ok=True)
    _put(RUNS / "reservation.json", {"status": "diagnostic_reserved_no_retry",
                                    "attempt_id": attempt_id, "freeze_sha256": freeze_sha})
    capture.mkdir(exist_ok=False)
    command = [sys.executable, str(Path(runner.__file__).resolve()),
               "--attempt-id", attempt_id, "--freeze-sha256", freeze_sha]
    _put(capture / "launch_start.json", {
        "status": "diagnostic_launch_started", "attempt_id": attempt_id,
        "freeze_sha256": freeze_sha, "wall_utc": datetime.now(timezone.utc).isoformat(),
        "worker_command": command, "worker_output": str(output),
        "prior_model_work_seconds": upstream["prior_model_work_seconds"],
        "failed_attempt_seconds": FAILED_SECONDS,
        "process_cap_seconds": runner.MODEL_SECONDS,
        "max_mps_driver_bytes": runner.MAX_DRIVER_BYTES,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "max_worker_bytes": runner.MAX_WORKER_BYTES,
        "max_capture_bytes": MAX_CAPTURE_BYTES,
        "planned_counts": {"full_forwards": 8, "direct_forwards": 16,
                           "vjps": 0, "generations": 0},
    })
    finish = {"status": "diagnostic_launch_failed_no_retry", "attempt_id": attempt_id,
              "freeze_sha256": freeze_sha, "worker_returncode": None,
              "auditor_returncode": None, "result_sha256": None,
              "error_sha256": None, "audit_sha256": None,
              "post_run_source_verified": False, "model_work_elapsed_seconds": 0.0,
              "prior_model_work_seconds": upstream["prior_model_work_seconds"],
              "failed_attempt_seconds": FAILED_SECONDS}
    began = time.monotonic()
    try:
        code, elapsed, reason = _bounded(command, capture,
                                         ("worker_stdout.txt", "worker_stderr.txt"),
                                         output, runner.MODEL_SECONDS)
        finish["worker_returncode"] = code
        finish["model_work_elapsed_seconds"] = elapsed
        if reason or code != 0 or elapsed >= runner.MODEL_SECONDS:
            raise ValueError(reason or "Diagnostic worker failed or exceeded cap")
        claim = _json(capture / "worker_claim.json")
        if (claim.get("status") != "diagnostic_worker_claimed_no_retry" or
                claim.get("attempt_id") != attempt_id or
                claim.get("freeze_sha256") != freeze_sha or
                not old_gate.utc(_json(capture / "launch_start.json")["wall_utc"]) <
                    old_gate.utc(claim.get("wall_utc"))):
            raise ValueError("Diagnostic worker claim absent or invalid")
        result = output / "result.json"
        if not result.is_file() or (output / "error.json").exists():
            raise ValueError("Diagnostic worker result absent or error present")
        finish["result_sha256"] = runner.sha(result)
        if runner.sha(FREEZE) != freeze_sha or verify_freeze()[0] != frozen:
            raise ValueError("Diagnostic source or freeze changed after model")
        finish["post_run_source_verified"] = True
        audit_command = [sys.executable, str(AUDITOR), "--run-dir", str(output),
                         "--output", str(output / "audit.json")]
        code, _, reason = _bounded(audit_command, capture,
                                    ("audit_stdout.txt", "audit_stderr.txt"),
                                    output, AUDIT_SECONDS)
        finish["auditor_returncode"] = code
        if reason or code != 0 or not (output / "audit.json").is_file():
            raise ValueError(reason or "Diagnostic CPU auditor rejected result")
        audit = _json(output / "audit.json")
        if (audit.get("status") !=
                "audited_old_fit_fp32_graph_inference_parity_saved_data_only" or
                audit.get("result_sha256") != finish["result_sha256"] or
                audit.get("failed_stage_b_fp32_manifest_sha256") != FAILED_MANIFEST_SHA):
            raise ValueError("Diagnostic CPU audit receipt differs")
        finish["audit_sha256"] = runner.sha(output / "audit.json")
        if runner.sha(FREEZE) != freeze_sha or verify_freeze()[0] != frozen:
            raise ValueError("Diagnostic source or freeze changed after audit")
        finish["status"] = "diagnostic_launch_completed_cpu_audited_only"
    except Exception as exc:
        finish["failure_reason"] = str(exc)
        finish["exception_type"] = type(exc).__name__
        error = output / "error.json"
        finish["error_sha256"] = runner.sha(error) if error.is_file() else None
    finally:
        finish["wall_utc"] = datetime.now(timezone.utc).isoformat()
        finish["elapsed_seconds"] = time.monotonic() - began
        _write_finish_preserving_cap(capture, output, finish)
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--attempt-id")
    args = parser.parse_args()
    if args.check_freeze:
        if args.attempt_id:
            parser.error("--check-freeze takes no attempt ID")
        verify_freeze()
        print(json.dumps({"status": "diagnostic_freeze_verified_only",
                          "freeze_sha256": runner.sha(FREEZE)}))
        return
    attempt_id = args.attempt_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = launch(attempt_id)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "diagnostic_launch_completed_cpu_audited_only":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
