"""Fail-closed, four-entry, one-shot launcher for the non-beaver FP32 pilot.

This source is inert until the V3 protocol, CPU preflight, both independent
source reviews, an external freeze, and its later independent review are
committed. A campaign reserves all four fixed contexts before any model load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import voxtral_old_fit_handoff_stage_b_launch_gate_v1 as old_gate
import voxtral_old_fit_grad_leaf_factorial_launch_gate_v1 as previous_gate
import voxtral_fp32_nonbeaver_linearity_runner_v1 as runner


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs/voxtral_fp32_nonbeaver_linearity_launch_v1"
AUDITOR = HERE / "voxtral_fp32_nonbeaver_linearity_audit_v1.py"
AUDITOR_TESTS = HERE / "test_voxtral_fp32_nonbeaver_linearity_audit_v1.py"
RUNNER_TESTS = HERE / "test_voxtral_fp32_nonbeaver_linearity_runner_v1.py"
GATE_TESTS = HERE / "test_voxtral_fp32_nonbeaver_linearity_launch_gate_v1.py"
PROTOCOL = HERE / "VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V3.md"
CONCEPTUAL_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_CONCEPTUAL_REVIEW_V3.md"
CPU_PREFLIGHT = HERE / "VOXTRAL_FP32_NONBEAVER_CPU_PREFLIGHT_V1.json"
SOURCE_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_INDEPENDENT_SOURCE_REVIEW_V1.md"
AUDITOR_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_AUDITOR_INDEPENDENT_SOURCE_REVIEW_V1.md"
FREEZE = HERE / "VOXTRAL_FP32_NONBEAVER_EXTERNAL_FREEZE_V1.json"
FREEZE_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_EXTERNAL_FREEZE_INDEPENDENT_REVIEW_V1.md"
PREVIOUS_MANIFEST = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_FACTORIAL_ATTEMPT_MANIFEST_V1.json"
PREVIOUS_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_FACTORIAL_RESULT_INDEPENDENT_REVIEW_V1.md"
PROTOCOL_SHA = "a222ce0f06a0e82b2c22bf6231e605983d726610b2f51dfbe7299de3ea8d79bd"
CONCEPTUAL_REVIEW_GO = ("Decision: GO for V3 non-beaver FP32 local-linearity "
                        "protocol design only; model launch HOLD until source "
                        "and external freeze reviews.")
SOURCE_GO = "Decision: GO for non-beaver FP32 runner and launch gate source only; model HOLD until external freeze."
AUDITOR_GO = "Decision: GO for non-beaver FP32 CPU auditor source only; model HOLD until external freeze."
FREEZE_GO = "Decision: GO for non-beaver FP32 external freeze provenance; four reserved one-shot contexts may run."
PREVIOUS_MANIFEST_SHA = "db171d28680dd7fedc588e49c926eb39d35663dcadaad9225956fd0c8731867f"
PREVIOUS_REVIEW_SHA = "7363631a2923dd72ec2671f451f9b843982552b4adcf48d05951528c38cdac39"
PREVIOUS_MODEL_SECONDS = 52.005679375026375
PREVIOUS_ATTEMPT = "20261118T154253722766Z"
STATUS = "frozen_voxtral_fp32_nonbeaver_linearity_four_contexts_v1"
CAMPAIGN_STATUS = "four_contexts_reserved_no_retry"
FINISH_OK = "context_launch_completed_cpu_audited_only"
FINISH_FAILED = "context_launch_failed_no_retry"
MAX_OUTPUT_BYTES = 2 * 1024**3
MAX_CAMPAIGN_BYTES = 8 * 1024**3
MIN_FREE_BYTES_BEFORE_RESERVATION = 12 * 1024**3
MAX_CAPTURE_BYTES = 8 * 1024**2
FINISH_RESERVE_BYTES = 64 * 1024
AUDIT_SECONDS = 120
TOTAL_MODEL_SECONDS = 4500
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CAMPAIGN_ID = re.compile(r"[0-9]{8}T[0-9]{12}Z\Z")
ATTEMPT_ID = re.compile(r"([0-9]{8}T[0-9]{12}Z)_0[0-3]\Z")
WAVS = {
    "piper_silver_wav_sha256": REPO / "eval/audio_jlens/corpus_v1/rendered/clips/piper_color_05_fit_chosen_silver.wav",
    "piper_teal_wav_sha256": REPO / "eval/audio_jlens/corpus_v1/rendered/clips/piper_color_05_fit_chosen_teal.wav",
    "kokoro_silver_wav_sha256": REPO / "eval/audio_jlens/corpus_v1/rendered/clips/kokoro_color_05_fit_chosen_silver.wav",
    "kokoro_teal_wav_sha256": REPO / "eval/audio_jlens/corpus_v1/rendered/clips/kokoro_color_05_fit_chosen_teal.wav",
}
EXPECTED_WAV_SHA = {
    "piper_silver_wav_sha256": "fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9",
    "piper_teal_wav_sha256": "112ed36abf5fb2be1ce77dcfffa9d1bb6deba10b4ecb2d14fc5960b8106626e9",
    "kokoro_silver_wav_sha256": "bcad702fa8b55c453286cb0e83ee3336a4bb6ade5e5ac9d4a42dc82c1a97c26e",
    "kokoro_teal_wav_sha256": "5038dbe167ae2ba6c9317f74c17bd9bb8ef7d07bbdee9da49f6dc0731e119365",
}
BOUND = {
    "protocol_sha256": PROTOCOL,
    "conceptual_review_sha256": CONCEPTUAL_REVIEW,
    "runner_sha256": Path(runner.__file__).resolve(),
    "runner_tests_sha256": RUNNER_TESTS,
    "gate_sha256": Path(__file__).resolve(),
    "gate_tests_sha256": GATE_TESTS,
    "auditor_sha256": AUDITOR,
    "auditor_tests_sha256": AUDITOR_TESTS,
    "cpu_preflight_sha256": CPU_PREFLIGHT,
    "source_review_sha256": SOURCE_REVIEW,
    "auditor_review_sha256": AUDITOR_REVIEW,
    "selection_sha256": HERE / "VOXTRAL_FIT_V2_SELECTION.json",
    "masks_sha256": HERE / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json",
    "previous_manifest_sha256": PREVIOUS_MANIFEST,
    "previous_review_sha256": PREVIOUS_REVIEW,
    "previous_freeze_sha256": previous_gate.FREEZE,
    "previous_freeze_review_sha256": previous_gate.FREEZE_REVIEW,
    **WAVS,
}
REVIEWED_RUNNER_KEYS = ("protocol_sha256", "runner_sha256", "runner_tests_sha256",
                        "gate_sha256", "gate_tests_sha256", "cpu_preflight_sha256")
REVIEWED_AUDITOR_KEYS = ("auditor_sha256", "auditor_tests_sha256")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("Run root is symlinked")
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError("Run output contains symlink")
        if child.is_file():
            total += child.stat().st_size
    return total


def _check_previous() -> float:
    """Retain all consumed old-FIT work and the audited factorial result."""
    _, upstream = previous_gate.verify_freeze()
    for path, expected in ((PREVIOUS_MANIFEST, PREVIOUS_MANIFEST_SHA),
                           (PREVIOUS_REVIEW, PREVIOUS_REVIEW_SHA)):
        old_gate.committed_same(REPO, path)
        if sha(path) != expected:
            raise ValueError("Prior factorial result provenance changed")
    manifest = _json(PREVIOUS_MANIFEST)
    if (manifest.get("schema") != "voxtral_old_fit_grad_leaf_factorial_attempt_manifest_v1" or
            manifest.get("status") != "launch_completed_cpu_audited_saved_data_only" or
            manifest.get("attempt_id") != PREVIOUS_ATTEMPT or
            manifest.get("parent_model_work_elapsed_seconds") != PREVIOUS_MODEL_SECONDS or
            manifest.get("launcher_status") != "diagnostic_launch_completed_cpu_audited_only" or
            manifest.get("raw_file_count") != 13 or
            manifest.get("total_raw_bytes") != 854641 or
            "Decision: GO for the one-case saved-result and provenance claims below only."
            not in PREVIOUS_REVIEW.read_text() or
            PREVIOUS_MANIFEST_SHA not in PREVIOUS_REVIEW.read_text()):
        raise ValueError("Prior factorial result or independent review differs")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != 13:
        raise ValueError("Prior raw receipt set differs")
    total = 0
    for row in rows:
        path = HERE / row["path"]
        old_gate.committed_same(REPO, path)
        if (row.get("git_committed") is not True or
                row.get("sha256") != sha(path) or row.get("bytes") != path.stat().st_size):
            raise ValueError("Prior raw receipt changed")
        total += path.stat().st_size
    if total != 854641:
        raise ValueError("Prior raw receipt byte total changed")
    sidecar = manifest["local_tensor_sidecar"]
    blob = HERE / sidecar["path"]
    if (blob.is_symlink() or not blob.is_file() or
            sidecar.get("git_committed") is not False or
            sidecar.get("sha256") != sha(blob) or
            sidecar.get("bytes") != blob.stat().st_size):
        raise ValueError("Prior local tensor sidecar differs")
    first_manifest = old_gate.first_add(REPO, PREVIOUS_MANIFEST)
    first_review = old_gate.first_add(REPO, PREVIOUS_REVIEW)
    if first_manifest == first_review:
        raise ValueError("Prior manifest self-reviewed")
    old_gate.ancestor(REPO, first_manifest, first_review)
    return float(upstream["prior_model_work_seconds"]) + PREVIOUS_MODEL_SECONDS


def _reviewed_at(commit: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        path = BOUND[key]
        rel = old_gate.regular(REPO, path)
        if old_gate._git(REPO, "show", f"{commit}:{rel}") != path.read_bytes():
            raise ValueError(f"Independent review does not contain {key}")
        source_commit = old_gate._git(REPO, "log", "-1", "--format=%H", "HEAD",
                                      "--", rel).decode().strip()
        if source_commit == commit:
            raise ValueError(f"Review self-authored {key}")
        old_gate.ancestor(REPO, source_commit, commit)


def verify_freeze() -> tuple[dict, dict]:
    """Read-only admission; every missing review or freeze is a hard HOLD."""
    if not PROTOCOL_SHA or sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("V3 protocol has no reviewed immutable source pin")
    prior_seconds = _check_previous()
    if prior_seconds + 4 * runner.MODEL_SECONDS > TOTAL_MODEL_SECONDS:
        raise ValueError("Cumulative four-entry model-work cap exceeded")
    if (runner.MODEL_SECONDS != 300 or runner.MAX_DRIVER_BYTES != 30 * 1024**3 or
            runner.MAX_OUTPUT_BYTES != MAX_OUTPUT_BYTES or
            runner.MAX_WORKER_BYTES >= MAX_OUTPUT_BYTES - 16 * 1024**2 or
            tuple(runner.CONTEXT_LABELS) !=
            ("piper_silver", "piper_teal", "kokoro_silver", "kokoro_teal")):
        raise ValueError("Runner schedule or resource cap differs from V3")
    for key, path in BOUND.items():
        old_gate.committed_same(REPO, path)
        digest = sha(path)
        if HEX64.fullmatch(digest) is None:
            raise ValueError(f"Bound file hash malformed: {key}")
    if (CONCEPTUAL_REVIEW_GO not in CONCEPTUAL_REVIEW.read_text() or
            PROTOCOL_SHA not in CONCEPTUAL_REVIEW.read_text()):
        raise ValueError("Independent V3 conceptual review absent")
    for key, expected in EXPECTED_WAV_SHA.items():
        if sha(BOUND[key]) != expected:
            raise ValueError(f"Selected FIT audio changed: {key}")
    if (SOURCE_GO not in SOURCE_REVIEW.read_text() or
            AUDITOR_GO not in AUDITOR_REVIEW.read_text()):
        raise ValueError("Independent source review absent")
    source_commit = old_gate.first_add(REPO, SOURCE_REVIEW)
    auditor_commit = old_gate.first_add(REPO, AUDITOR_REVIEW)
    if source_commit == auditor_commit:
        raise ValueError("Runner and auditor reviews share commit")
    _reviewed_at(source_commit, REVIEWED_RUNNER_KEYS)
    _reviewed_at(auditor_commit, REVIEWED_AUDITOR_KEYS)
    conceptual_commit = old_gate.first_add(REPO, CONCEPTUAL_REVIEW)
    for later in (source_commit, auditor_commit):
        if later == conceptual_commit:
            raise ValueError("Source review self-authored conceptual review")
        old_gate.ancestor(REPO, conceptual_commit, later)
    old_gate.committed_same(REPO, FREEZE)
    freeze = _json(FREEZE)
    expected = {"status", "reviewed_utc", "source_review_commit",
                "auditor_review_commit", "context_labels", "planned_counts_per_context",
                "process_cap_seconds", "audit_cap_seconds", "max_mps_driver_bytes",
                "max_output_bytes_per_context", "max_capture_bytes",
                "max_worker_bytes", "prior_model_work_seconds",
                "max_total_model_work_seconds", *BOUND}
    if (set(freeze) != expected or freeze["status"] != STATUS or
            freeze["source_review_commit"] != source_commit or
            freeze["auditor_review_commit"] != auditor_commit or
            freeze["context_labels"] != list(runner.CONTEXT_LABELS) or
            freeze["planned_counts_per_context"] !=
            {"patch_forwards": 96, "vjps": 8, "g1_clean_graphs": 2,
             "n0_clean_references": 2, "donor_captures": 1,
             "natural_generations": 2} or
            freeze["process_cap_seconds"] != 300 or
            freeze["audit_cap_seconds"] != AUDIT_SECONDS or
            freeze["max_mps_driver_bytes"] != 30 * 1024**3 or
            freeze["max_output_bytes_per_context"] != MAX_OUTPUT_BYTES or
            freeze["max_capture_bytes"] != MAX_CAPTURE_BYTES or
            freeze["max_worker_bytes"] != runner.MAX_WORKER_BYTES or
            freeze["prior_model_work_seconds"] != prior_seconds or
            freeze["max_total_model_work_seconds"] != TOTAL_MODEL_SECONDS):
        raise ValueError("External four-entry freeze schema or caps differ")
    old_gate.utc(freeze["reviewed_utc"])
    for key, path in BOUND.items():
        if freeze[key] != sha(path):
            raise ValueError(f"External freeze file pin differs: {key}")
    freeze_commit = old_gate.first_add(REPO, FREEZE)
    for earlier in (source_commit, auditor_commit,
                    old_gate.first_add(REPO, PREVIOUS_REVIEW)):
        if freeze_commit == earlier:
            raise ValueError("Freeze self-authored prerequisite")
        old_gate.ancestor(REPO, earlier, freeze_commit)
    old_gate.committed_same(REPO, FREEZE_REVIEW)
    review_commit = old_gate.first_add(REPO, FREEZE_REVIEW)
    if (review_commit == freeze_commit or
            FREEZE_GO not in FREEZE_REVIEW.read_text() or
            sha(FREEZE) not in FREEZE_REVIEW.read_text()):
        raise ValueError("Independent external freeze review absent")
    old_gate.ancestor(REPO, freeze_commit, review_commit)
    return freeze, {"prior_model_work_seconds": prior_seconds,
                    "cpu_preflight_sha256": sha(CPU_PREFLIGHT)}


def _reservation() -> dict:
    reservation = _json(RUNS / "campaign_reservation.json")
    campaign = reservation.get("campaign_id")
    if (reservation.get("status") != CAMPAIGN_STATUS or
            not isinstance(campaign, str) or CAMPAIGN_ID.fullmatch(campaign) is None or
            reservation.get("context_labels") != list(runner.CONTEXT_LABELS) or
            reservation.get("attempt_ids") !=
            [f"{campaign}_{i:02d}" for i in range(4)] or
            reservation.get("freeze_sha256") != sha(FREEZE)):
        raise ValueError("Four-entry campaign reservation differs")
    return reservation


def reserve_all(campaign_id: str) -> dict:
    if CAMPAIGN_ID.fullmatch(campaign_id) is None:
        raise ValueError("Campaign ID malformed")
    freeze, upstream = verify_freeze()
    if RUNS.is_symlink() or runner.RUNS.is_symlink():
        raise ValueError("Campaign run root is symlinked")
    if (RUNS.exists() and any(RUNS.iterdir())) or (runner.RUNS.exists() and
            any((runner.RUNS / name).exists() and
                any((runner.RUNS / name).iterdir())
                for name in runner.CONTEXT_LABELS)):
        raise ValueError("Four-entry campaign already exists; no retry")
    if shutil.disk_usage(HERE).free < MIN_FREE_BYTES_BEFORE_RESERVATION:
        raise ValueError("Less than 12 GiB free before campaign reservation")
    RUNS.mkdir(parents=True, exist_ok=True)
    runner.RUNS.mkdir(parents=True, exist_ok=True)
    reservation = {"status": CAMPAIGN_STATUS, "campaign_id": campaign_id,
                   "attempt_ids": [f"{campaign_id}_{i:02d}" for i in range(4)],
                   "context_labels": list(runner.CONTEXT_LABELS),
                   "freeze_sha256": sha(FREEZE),
                   "wall_utc": datetime.now(timezone.utc).isoformat(),
                   "prior_model_work_seconds": upstream["prior_model_work_seconds"]}
    if old_gate.utc(reservation["wall_utc"]) <= old_gate.utc(freeze["reviewed_utc"]):
        raise ValueError("Reservation predates external freeze")
    _put(RUNS / "campaign_reservation.json", reservation)
    return reservation


def _check_prior_entries(index: int, reservation: dict) -> None:
    for earlier in range(index):
        label = runner.CONTEXT_LABELS[earlier]
        attempt = reservation["attempt_ids"][earlier]
        finish = _json(RUNS / label / attempt / "launch_finish.json")
        if (finish.get("attempt_id") != attempt or
                finish.get("context_index") != earlier or
                finish.get("status") not in (FINISH_OK, FINISH_FAILED)):
            raise ValueError("Earlier fixed campaign entry lacks final receipt")
    for later in range(index + 1, 4):
        label = runner.CONTEXT_LABELS[later]
        attempt = reservation["attempt_ids"][later]
        if (RUNS / label / attempt / "launch_start.json").exists():
            raise ValueError("Later fixed campaign entry started out of order")


def verify_worker_start(index: int, attempt_id: str, freeze_sha256: str) -> dict:
    if not isinstance(index, int) or index not in range(4):
        raise ValueError("Context index invalid")
    if ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("Attempt ID malformed")
    freeze, upstream = verify_freeze()
    reservation = _reservation()
    _check_prior_entries(index, reservation)
    label = runner.CONTEXT_LABELS[index]
    capture = RUNS / label / attempt_id
    output = runner.RUNS / label / attempt_id
    start = _json(capture / "launch_start.json")
    if (reservation["attempt_ids"][index] != attempt_id or
            reservation["freeze_sha256"] != freeze_sha256 or
            start.get("status") != "context_launch_started" or
            start.get("attempt_id") != attempt_id or
            start.get("context_index") != index or
            start.get("context_label") != label or
            start.get("freeze_sha256") != freeze_sha256 or
            old_gate.utc(start.get("wall_utc")) <= old_gate.utc(freeze["reviewed_utc"]) or
            output.exists() or (capture / "worker_claim.json").exists()):
        raise ValueError("Worker claim failed four-entry one-shot admission")
    _put(capture / "worker_claim.json", {
        "status": "context_worker_claimed_no_retry", "context_index": index,
        "context_label": label, "attempt_id": attempt_id,
        "freeze_sha256": freeze_sha256,
        "wall_utc": datetime.now(timezone.utc).isoformat(),
    })
    return upstream


def _bounded(command: list[str], capture: Path, names: tuple[str, str],
             output: Path, seconds: int) -> tuple[int | None, float, str | None]:
    began = time.monotonic()
    reason = None
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
                reason = "context_process_timeout"
            elif max((capture / names[0]).stat().st_size,
                     (capture / names[1]).stat().st_size) >= MAX_CAPTURE_BYTES:
                reason = "context_capture_byte_cap"
            elif _tree_bytes(capture) + _tree_bytes(output) >= \
                    MAX_OUTPUT_BYTES - FINISH_RESERVE_BYTES:
                reason = "context_combined_output_byte_cap"
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


def _write_finish(capture: Path, output: Path, finish: dict) -> None:
    for name in ("worker_stdout.txt", "worker_stderr.txt",
                 "audit_stdout.txt", "audit_stderr.txt"):
        stream = capture / name
        if stream.is_file() and stream.stat().st_size >= MAX_CAPTURE_BYTES:
            original_bytes = stream.stat().st_size
            original_sha = sha(stream)
            with stream.open("r+b") as handle:
                prefix = handle.read(512)
                handle.truncate(0)
                handle.seek(0)
                handle.write(prefix)
            finish.setdefault("trimmed_streams", {})[name] = {
                "original_bytes": original_bytes,
                "original_sha256": original_sha,
                "retained_prefix_bytes": len(prefix),
            }
            finish["status"] = FINISH_FAILED
            finish["failure_reason"] = "context_capture_byte_cap"
    finish["worker_output_bytes"] = _tree_bytes(output)
    finish["launcher_output_bytes_before_finish"] = _tree_bytes(capture)
    if finish["worker_output_bytes"] + finish["launcher_output_bytes_before_finish"] + \
            FINISH_RESERVE_BYTES >= MAX_OUTPUT_BYTES:
        finish["status"] = FINISH_FAILED
        finish["failure_reason"] = "context_combined_output_byte_cap"
        for name in ("worker_stdout.txt", "worker_stderr.txt",
                     "audit_stdout.txt", "audit_stderr.txt"):
            stream = capture / name
            if stream.is_file() and name not in finish.get("trimmed_streams", {}):
                original_bytes = stream.stat().st_size
                original_sha = sha(stream)
                with stream.open("r+b") as handle:
                    prefix = handle.read(512)
                    handle.truncate(0)
                    handle.seek(0)
                    handle.write(prefix)
                finish.setdefault("trimmed_streams", {})[name] = {
                    "original_bytes": original_bytes,
                    "original_sha256": original_sha,
                    "retained_prefix_bytes": len(prefix),
                }
    _put(capture / "launch_finish.json", finish)


def launch_context(index: int) -> dict:
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Launch requires pinned second_architecture environment")
    if not isinstance(index, int) or index not in range(4):
        raise ValueError("Context index invalid")
    frozen, upstream = verify_freeze()
    reservation = _reservation()
    _check_prior_entries(index, reservation)
    label = runner.CONTEXT_LABELS[index]
    attempt_id = reservation["attempt_ids"][index]
    capture = RUNS / label / attempt_id
    output = runner.RUNS / label / attempt_id
    if capture.exists() or output.exists():
        raise ValueError("This context already started; no retry")
    if _tree_bytes(RUNS) + _tree_bytes(runner.RUNS) >= MAX_CAMPAIGN_BYTES:
        raise ValueError("Eight-GiB campaign output cap reached")
    capture.mkdir(parents=True, exist_ok=False)
    freeze_sha = sha(FREEZE)
    command = [sys.executable, str(Path(runner.__file__).resolve()),
               "--run-context", str(index), "--attempt-id", attempt_id,
               "--freeze-sha256", freeze_sha]
    _put(capture / "launch_start.json", {
        "status": "context_launch_started", "context_index": index,
        "context_label": label, "attempt_id": attempt_id,
        "freeze_sha256": freeze_sha,
        "wall_utc": datetime.now(timezone.utc).isoformat(),
        "worker_command": command, "worker_output": str(output),
        "prior_model_work_seconds": upstream["prior_model_work_seconds"],
        "process_cap_seconds": runner.MODEL_SECONDS,
        "max_mps_driver_bytes": runner.MAX_DRIVER_BYTES,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "max_worker_bytes": runner.MAX_WORKER_BYTES,
        "max_capture_bytes": MAX_CAPTURE_BYTES,
        "planned_counts": {"patch_forwards": 96, "vjps": 8,
                           "g1_clean_graphs": 2, "n0_clean_references": 2,
                           "donor_captures": 1, "natural_generations": 2},
    })
    finish = {"status": FINISH_FAILED, "context_index": index,
              "context_label": label, "attempt_id": attempt_id,
              "freeze_sha256": freeze_sha, "worker_returncode": None,
              "auditor_returncode": None, "result_sha256": None,
              "audit_sha256": None, "error_sha256": None,
              "post_run_source_verified": False, "model_work_elapsed_seconds": 0.0,
              "prior_model_work_seconds": upstream["prior_model_work_seconds"]}
    began = time.monotonic()
    try:
        code, elapsed, reason = _bounded(command, capture,
                                         ("worker_stdout.txt", "worker_stderr.txt"),
                                         output, runner.MODEL_SECONDS)
        finish["worker_returncode"] = code
        finish["model_work_elapsed_seconds"] = elapsed
        if reason or code != 0 or elapsed >= runner.MODEL_SECONDS:
            raise ValueError(reason or "Context worker failed or exceeded cap")
        claim = _json(capture / "worker_claim.json")
        if (claim.get("status") != "context_worker_claimed_no_retry" or
                claim.get("context_index") != index or
                claim.get("context_label") != label or
                claim.get("attempt_id") != attempt_id or
                claim.get("freeze_sha256") != freeze_sha or
                not old_gate.utc(_json(capture / "launch_start.json")["wall_utc"]) <
                    old_gate.utc(claim.get("wall_utc"))):
            raise ValueError("Worker claim missing or out of order")
        result_path = output / "result.json"
        if not result_path.is_file() or (output / "error.json").exists():
            raise ValueError("Worker result absent or error present")
        finish["result_sha256"] = sha(result_path)
        if sha(FREEZE) != freeze_sha or verify_freeze()[0] != frozen:
            raise ValueError("Source/freeze changed after model process")
        finish["post_run_source_verified"] = True
        audit_cmd = [sys.executable, str(AUDITOR), "--run-dir", str(output),
                     "--output", str(output / "audit.json")]
        code, _, reason = _bounded(audit_cmd, capture,
                                    ("audit_stdout.txt", "audit_stderr.txt"),
                                    output, AUDIT_SECONDS)
        finish["auditor_returncode"] = code
        audit_path = output / "audit.json"
        if reason or code != 0 or not audit_path.is_file():
            raise ValueError(reason or "Independent CPU auditor rejected context")
        audit = _json(audit_path)
        if (audit.get("status") != "audited_voxtral_fp32_nonbeaver_saved_data_only" or
                audit.get("context_index") != index or
                audit.get("context_label") != label or
                audit.get("attempt_id") != attempt_id or
                audit.get("result_sha256") != finish["result_sha256"] or
                audit.get("freeze_sha256") != freeze_sha):
            raise ValueError("CPU audit result binding differs")
        finish["audit_sha256"] = sha(audit_path)
        if sha(FREEZE) != freeze_sha or verify_freeze()[0] != frozen:
            raise ValueError("Source/freeze changed after audit")
        finish["status"] = FINISH_OK
    except Exception as exc:
        finish["failure_reason"] = str(exc)
        finish["exception_type"] = type(exc).__name__
        error = output / "error.json"
        finish["error_sha256"] = sha(error) if error.is_file() else None
    finally:
        finish["wall_utc"] = datetime.now(timezone.utc).isoformat()
        finish["elapsed_seconds"] = time.monotonic() - began
        _write_finish(capture, output, finish)
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--reserve-all", action="store_true")
    mode.add_argument("--run-context", type=int)
    parser.add_argument("--campaign-id")
    args = parser.parse_args()
    if args.check_freeze:
        if args.campaign_id is not None:
            parser.error("--check-freeze takes no campaign ID")
        verify_freeze()
        print(json.dumps({"status": "four_context_freeze_verified_only",
                          "freeze_sha256": sha(FREEZE)}))
        return
    if args.reserve_all:
        campaign = args.campaign_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        print(json.dumps(reserve_all(campaign), indent=2, allow_nan=False))
        return
    if args.campaign_id is not None:
        parser.error("--run-context uses the existing four-entry reservation")
    result = launch_context(args.run_context)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != FINISH_OK:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
