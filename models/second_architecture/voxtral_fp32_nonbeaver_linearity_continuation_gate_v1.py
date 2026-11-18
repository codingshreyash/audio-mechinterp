"""One-shot V2 gate for the two unstarted slots of the frozen V1 campaign.

The original V1 gate, runner, auditor, reservation, and completed artifacts are
read only. Missing reviews or the external continuation freeze are a hard HOLD.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import plistlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import voxtral_fp32_nonbeaver_linearity_launch_gate_v1 as v1

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs/voxtral_fp32_nonbeaver_linearity_continuation_launch_v1"
WORKER = HERE / "voxtral_fp32_nonbeaver_linearity_continuation_worker_v1.py"
TESTS = HERE / "test_voxtral_fp32_nonbeaver_linearity_continuation_gate_v1.py"
PROTOCOL = HERE / "VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_CONTINUATION_PROSPECTIVE_V1.md"
CONCEPTUAL_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_CONTINUATION_CONCEPTUAL_REVIEW_V1.md"
PIPER_MANIFEST = HERE / "VOXTRAL_FP32_NONBEAVER_PIPER_EVIDENCE_MANIFEST_V1.json"
PIPER_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_PIPER_EVIDENCE_INDEPENDENT_REVIEW_V1.md"
FAILED_ADMISSION = HERE / "VOXTRAL_FP32_NONBEAVER_CONTEXT2_FAILED_ADMISSION_V1.json"
POWER_EVIDENCE = HERE / "VOXTRAL_FP32_NONBEAVER_CONTEXT1_POWER_LOG_EVIDENCE_V1.txt"
SOURCE_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_CONTINUATION_SOURCE_INDEPENDENT_REVIEW_V1.md"
FREEZE = HERE / "VOXTRAL_FP32_NONBEAVER_CONTINUATION_EXTERNAL_FREEZE_V1.json"
FREEZE_REVIEW = HERE / "VOXTRAL_FP32_NONBEAVER_CONTINUATION_EXTERNAL_FREEZE_INDEPENDENT_REVIEW_V1.md"
CAMPAIGN = "20261118T182308460007Z"
PROTOCOL_SHA = "1e2fa220ad7fab8aad2758aa715fb0d73d11ac9f99d680af0f0ab91c9f9c6fdb"
MANIFEST_SHA = "4f52336d46faef0909206c9df2f4f9d630242183b54912c0db51b5103ceec83a"
FAILED_SHA = "1064196125fa258b384bca6a4bc1f703a8dad8d7cae765ce0400d99d3416863a"
POWER_SHA = "f9cd47f5035e05399b4ff139c9d2f6706f91e99698ea8f96f2326866b706be72"
RAW_MANIFEST_SHA = "bdd3f1d3161c60201b361a1014a4fddbaf4e28e939d936bd170d2f486173c645"
V1_FREEZE_SHA = "c28ffecbd7487b92ff98bc47bd2872af30f0a8400cbe65a4caac30b179cafb4f"
V1_RUNNER_SHA = "91e622f7fd8f4d9dac00ca7a0baaf02d74faf9326109d10674ebf5673310bc04"
V1_AUDITOR_SHA = "70a1ac3c4e4da92bb1fa1034bd866195024dcd423ea35a2386e233bce5dca8e2"
V1_GATE_SHA = "14cd58e9f5706be0da2453664a14aef11d7f5cb5f55f354b1028aca717df0a9d"
EXCEPTION_LAUNCH_GAP = 25.1243381670796
EXCEPTION_WORKER_GAP = 13.807611458983871
PRIOR_WORK = (138.8538012499921, 151.86878254101612)
SOURCE_GO = "Decision: GO for non-beaver FP32 continuation gate and worker entrypoint source only; model HOLD until continuation freeze."
FREEZE_GO = "Decision: GO for non-beaver FP32 continuation external freeze; only reserved contexts 02 and 03 may run."
STATUS = "frozen_voxtral_fp32_nonbeaver_continuation_v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
POWER_EVENT = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+([^\t]+)\t", re.MULTILINE)
POWER_TERMINAL = re.compile(
    r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+: Showing all currently held IOKit power assertions\s*$",
    re.MULTILINE)
HOST_COMMANDS = {
    "power": ["pmset", "-g", "batt"],
    "clamshell": ["ioreg", "-a", "-r", "-c", "IOPMrootDomain", "-d", "1"],
    "battery": ["ioreg", "-a", "-r", "-c", "AppleSmartBattery", "-d", "1"],
    "power_log": ["pmset", "-g", "log"],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hashes() -> dict[str, str]:
    return {
        "protocol_sha256": v1.sha(PROTOCOL),
        "conceptual_review_sha256": v1.sha(CONCEPTUAL_REVIEW),
        "piper_evidence_manifest_sha256": v1.sha(PIPER_MANIFEST),
        "piper_evidence_review_sha256": v1.sha(PIPER_REVIEW),
        "failed_admission_sha256": v1.sha(FAILED_ADMISSION),
        "power_log_evidence_sha256": v1.sha(POWER_EVIDENCE),
        "v1_freeze_sha256": v1.sha(v1.FREEZE),
        "v1_runner_sha256": v1.sha(Path(v1.runner.__file__)),
        "v1_auditor_sha256": v1.sha(v1.AUDITOR),
        "v1_gate_sha256": v1.sha(Path(v1.__file__)),
        "v2_gate_sha256": v1.sha(Path(__file__)),
        "v2_worker_sha256": v1.sha(WORKER),
        "v2_tests_sha256": v1.sha(TESTS),
        "v2_source_review_sha256": v1.sha(SOURCE_REVIEW),
    }


def _verify_manifest() -> dict:
    manifest = v1._json(PIPER_MANIFEST)
    if (manifest.get("schema") != "voxtral_fp32_nonbeaver_piper_evidence_manifest_v1" or
            manifest.get("raw_file_count") != 27 or
            manifest.get("raw_manifest_sha256") != RAW_MANIFEST_SHA or
            manifest.get("freeze", {}).get("sha256") != V1_FREEZE_SHA or
            manifest.get("failed_admission", {}).get("receipt_sha256") != FAILED_SHA or
            manifest.get("failed_admission", {}).get("remaining_attempt_directories_absent") is not True or
            manifest.get("reserved_context_labels") != list(v1.runner.CONTEXT_LABELS)):
        raise ValueError("Piper evidence manifest fields differ")
    lines = []
    files = manifest.get("raw_files")
    if not isinstance(files, list) or len(files) != 27:
        raise ValueError("Piper raw manifest count differs")
    seen = set()
    for row in files:
        rel = row.get("path")
        if not isinstance(rel, str):
            raise ValueError("Piper raw path malformed")
        unresolved = REPO / rel
        path = unresolved.resolve()
        if (not path.is_relative_to(REPO.resolve()) or
                any(parent.is_symlink() for parent in (unresolved, *unresolved.parents)
                    if parent.is_relative_to(REPO)) or
                not path.is_file() or path.stat().st_nlink != 1 or
                rel in seen or path.stat().st_size != row.get("bytes") or
                v1.sha(path) != row.get("sha256")):
            raise ValueError("Piper raw file changed or absent")
        seen.add(rel)
        lines.append(f"{row['sha256']}  {rel}\n")
    if hashlib.sha256("".join(lines).encode()).hexdigest() != RAW_MANIFEST_SHA:
        raise ValueError("Piper raw manifest digest differs")
    expected = {str((v1.RUNS / "campaign_reservation.json").relative_to(REPO))}
    for i in (0, 1):
        label, attempt = v1.runner.CONTEXT_LABELS[i], f"{CAMPAIGN}_{i:02d}"
        expected.update(str(p.relative_to(REPO)) for p in (v1.RUNS / label / attempt).iterdir() if p.is_file())
        output = v1.runner.RUNS / label / attempt
        expected.update(str(p.relative_to(REPO)) for p in output.rglob("*") if p.is_file())
    if seen != expected:
        raise ValueError("Piper raw file set differs")
    return manifest


def verify_freeze() -> tuple[dict, dict, dict]:
    """Read-only admission; require all committed reviews and both freezes."""
    original, upstream = v1.verify_freeze()
    for path, expected in ((PROTOCOL, PROTOCOL_SHA), (PIPER_MANIFEST, MANIFEST_SHA),
                           (FAILED_ADMISSION, FAILED_SHA), (POWER_EVIDENCE, POWER_SHA),
                           (v1.FREEZE, V1_FREEZE_SHA), (Path(v1.runner.__file__), V1_RUNNER_SHA),
                           (v1.AUDITOR, V1_AUDITOR_SHA), (Path(v1.__file__), V1_GATE_SHA)):
        if v1.sha(path) != expected:
            raise ValueError(f"Pinned continuation evidence changed: {path.name}")
        v1.old_gate.committed_same(REPO, path)
    if (PROTOCOL_SHA not in CONCEPTUAL_REVIEW.read_text() or
            "GO for implementing a new V2 continuation gate" not in CONCEPTUAL_REVIEW.read_text()):
        raise ValueError("Independent continuation conceptual review absent")
    v1.old_gate.committed_same(REPO, CONCEPTUAL_REVIEW)
    manifest = _verify_manifest()
    review = PIPER_REVIEW.read_text()
    if (MANIFEST_SHA not in review or RAW_MANIFEST_SHA not in review or
            "reran the pinned gate's `--verify-saved-prior`" not in review):
        raise ValueError("Independent Piper evidence review absent")
    v1.old_gate.committed_same(REPO, PIPER_REVIEW)
    if "9576089fe580f9d7a99db8de3d43dc98b8ff2a64a05ac031c5a3193b4eb828df" not in POWER_EVIDENCE.read_text():
        raise ValueError("Power-log evidence extraction differs")
    failed = v1._json(FAILED_ADMISSION)
    if (failed.get("attempt_id") != f"{CAMPAIGN}_02" or failed.get("exit_code") != 1 or
            failed.get("error_message") != "Earlier context model-work seconds invalid" or
            any(row.get("launch_exists_before") or row.get("launch_exists_after") or
                row.get("worker_exists_before") or row.get("worker_exists_after")
                for row in failed.get("attempt_directories", []))):
        raise ValueError("V1 failed admission record differs")
    for path in (SOURCE_REVIEW, FREEZE, FREEZE_REVIEW):
        v1.old_gate.committed_same(REPO, path)
    pins = _hashes()
    source = SOURCE_REVIEW.read_text()
    if (SOURCE_GO not in source or any(pins[key] not in source for key in
            ("v2_gate_sha256", "v2_worker_sha256", "v2_tests_sha256"))):
        raise ValueError("Independent V2 source review absent")
    conceptual_commit = v1.old_gate.first_add(REPO, CONCEPTUAL_REVIEW)
    evidence_commit = v1.old_gate.first_add(REPO, PIPER_REVIEW)
    source_commit = v1.old_gate.first_add(REPO, SOURCE_REVIEW)
    for prior in (conceptual_commit, evidence_commit):
        if prior == source_commit:
            raise ValueError("Source review shares prerequisite commit")
        v1.old_gate.ancestor(REPO, prior, source_commit)
    freeze = v1._json(FREEZE)
    reservation = v1._reservation(original, upstream)
    expected = {
        "status": STATUS, "reviewed_utc": freeze.get("reviewed_utc"),
        "source_review_commit": source_commit,
        "campaign_id": CAMPAIGN,
        "attempt_ids": reservation["attempt_ids"],
        "context_labels": list(v1.runner.CONTEXT_LABELS),
        "raw_manifest_sha256": RAW_MANIFEST_SHA,
        "planned_counts_per_context": v1.runner.COUNTS,
        "process_cap_seconds": 300, "audit_cap_seconds": 120,
        "max_mps_driver_bytes": 30 * 1024**3,
        "max_output_bytes_per_context": 2 * 1024**3,
        "max_worker_bytes": v1.runner.MAX_WORKER_BYTES,
        "max_capture_bytes": 8 * 1024**2,
        "max_campaign_bytes": 8 * 1024**3,
        "min_free_bytes_before_launch": 12 * 1024**3,
        "max_total_model_work_seconds": 4500,
        "prior_model_work_seconds": 368.4824181678705,
        "completed_model_work_seconds": list(PRIOR_WORK),
        **pins,
    }
    if freeze != expected:
        raise ValueError("Continuation freeze schema, source pins, or caps differ")
    if v1.old_gate.utc(freeze["reviewed_utc"]) <= v1.old_gate.utc(
            v1._json(v1.RUNS / "piper_teal" / f"{CAMPAIGN}_01" /
                     "launch_finish.json")["wall_utc"]):
        raise ValueError("Continuation freeze predates the clock exception")
    freeze_commit = v1.old_gate.first_add(REPO, FREEZE)
    if freeze_commit == source_commit:
        raise ValueError("Continuation freeze shares source review commit")
    v1.old_gate.ancestor(REPO, source_commit, freeze_commit)
    review_commit = v1.old_gate.first_add(REPO, FREEZE_REVIEW)
    if (review_commit == freeze_commit or FREEZE_GO not in FREEZE_REVIEW.read_text() or
            v1.sha(FREEZE) not in FREEZE_REVIEW.read_text()):
        raise ValueError("Independent continuation freeze review absent")
    v1.old_gate.ancestor(REPO, freeze_commit, review_commit)
    return freeze, upstream, reservation


def _signed_gaps(start: dict, finish: dict) -> tuple[float, float]:
    start_at = v1.old_gate.utc(start.get("wall_utc"))
    finish_at = v1.old_gate.utc(finish.get("wall_utc"))
    work_start = v1.old_gate.utc(finish.get("worker_started_utc"))
    work_finish = v1.old_gate.utc(finish.get("worker_finished_utc"))
    if not start_at < work_start <= work_finish <= finish_at:
        raise ValueError("Context UTC chronology differs")
    elapsed = finish.get("elapsed_seconds")
    work = finish.get("model_work_elapsed_seconds")
    if (type(elapsed) not in (int, float) or type(work) not in (int, float) or
            not math.isfinite(elapsed) or not math.isfinite(work) or
            elapsed < work or work < 0 or work > 300 or elapsed > 540):
        raise ValueError("Context elapsed seconds invalid")
    return ((finish_at - start_at).total_seconds() - elapsed,
            (work_finish - work_start).total_seconds() - work)


def _require_strict_timing(start: dict, finish: dict) -> None:
    if any(abs(gap) > 5 for gap in _signed_gaps(start, finish)):
        raise ValueError("Continuation five-second timing rule failed")


def _check_piper_one(reservation: dict, used0: float) -> tuple[float, datetime]:
    """The only timing exception, on byte-pinned original `_01` evidence."""
    index = 1
    label, attempt = v1.runner.CONTEXT_LABELS[index], reservation["attempt_ids"][index]
    capture = v1.RUNS / label / attempt
    output = v1.runner.RUNS / label / attempt
    start = v1._json(capture / "launch_start.json")
    finish = v1._json(capture / "launch_finish.json")
    claim = v1._json(capture / "worker_claim.json")
    result = v1._json(output / "result.json")
    audit = v1._json(output / "audit.json")
    expected_command = [sys.executable, str(Path(v1.runner.__file__).resolve()),
                        "--run-context", "1", "--attempt-id", attempt,
                        "--freeze-sha256", V1_FREEZE_SHA]
    base = reservation["prior_model_work_seconds"]
    if (attempt != f"{CAMPAIGN}_01" or
            start.get("status") != "context_launch_started" or
            start.get("context_index") != 1 or start.get("context_label") != label or
            start.get("attempt_id") != attempt or start.get("freeze_sha256") != V1_FREEZE_SHA or
            start.get("worker_command") != expected_command or
            start.get("worker_output") != str(output) or
            start.get("prior_model_work_seconds") != base or
            start.get("cumulative_prior_model_work_seconds") != base + used0 or
            start.get("process_cap_seconds") != 300 or
            start.get("max_mps_driver_bytes") != 30 * 1024**3 or
            start.get("max_output_bytes") != 2 * 1024**3 or
            start.get("max_worker_bytes") != v1.runner.MAX_WORKER_BYTES or
            start.get("max_capture_bytes") != 8 * 1024**2 or
            start.get("planned_counts") != v1.runner.COUNTS):
        raise ValueError("Original context 01 start differs")
    start_at = v1.old_gate.utc(start["wall_utc"])
    finish_at = v1.old_gate.utc(finish["wall_utc"])
    if not (v1.old_gate.utc(reservation["wall_utc"]) < start_at < finish_at):
        raise ValueError("Original context 01 order differs")
    gaps = _signed_gaps(start, finish)
    if (not 0 < gaps[1] <= gaps[0] or
            abs(gaps[0] - EXCEPTION_LAUNCH_GAP) > .001 or
            abs(gaps[1] - EXCEPTION_WORKER_GAP) > .001):
        raise ValueError("Original context 01 signed gap differs")
    if (claim.get("status") != "context_worker_claimed_no_retry" or
            claim.get("context_index") != 1 or claim.get("context_label") != label or
            claim.get("attempt_id") != attempt or claim.get("freeze_sha256") != V1_FREEZE_SHA or
            not start_at < v1.old_gate.utc(claim.get("wall_utc")) < finish_at or
            finish.get("status") != v1.FINISH_OK or
            finish.get("context_index") != 1 or finish.get("context_label") != label or
            finish.get("attempt_id") != attempt or finish.get("freeze_sha256") != V1_FREEZE_SHA or
            finish.get("prior_model_work_seconds") != base or
            finish.get("cumulative_prior_model_work_seconds") != base + used0 or
            finish.get("model_work_elapsed_seconds") != PRIOR_WORK[1] or
            finish.get("worker_returncode") != 0 or finish.get("auditor_returncode") != 0 or
            finish.get("post_run_source_verified") is not True or
            finish.get("error_sha256") is not None or (output / "error.json").exists() or
            finish.get("result_sha256") != v1.sha(output / "result.json") or
            finish.get("audit_sha256") != v1.sha(output / "audit.json") or
            finish.get("worker_output_bytes") != v1._tree_bytes(output) or
            type(finish.get("launcher_output_bytes_before_finish")) is not int or
            type(finish.get("combined_output_bytes_before_finish")) is not int):
        raise ValueError("Original context 01 finish differs")
    capture_pre_finish = v1._tree_bytes(capture) - (capture / "launch_finish.json").stat().st_size
    if (finish["combined_output_bytes_before_finish"] != v1._tree_bytes(output) + capture_pre_finish or
            v1._tree_bytes(capture) + v1._tree_bytes(output) > v1.MAX_OUTPUT_BYTES):
        raise ValueError("Original context 01 byte totals differ")
    if (result.get("status") != v1.runner.STATUS or result.get("context", {}).get("index") != 1 or
            result.get("context", {}).get("label") != label or result.get("attempt_id") != attempt or
            result.get("freeze_sha256") != V1_FREEZE_SHA or
            result.get("actual_counts") != v1.runner.COUNTS or
            audit.get("status") != "audited_voxtral_fp32_nonbeaver_saved_data_only" or
            audit.get("context") != result.get("context") or
            audit.get("attempt_id") != attempt or audit.get("freeze_sha256") != V1_FREEZE_SHA or
            audit.get("result_sha256") != v1.sha(output / "result.json") or
            audit.get("preflight_receipt_sha256") != v1.sha(v1.CPU_PREFLIGHT) or
            audit.get("protocol_v5_sha256") != v1.PROTOCOL_SHA or
            audit.get("worker_invalidity_disagreement") is not False):
        raise ValueError("Original context 01 saved result/audit differs")
    v1._success_files(output, result, audit)
    v1._rerun_saved_prior_audit(output)
    return finish["model_work_elapsed_seconds"], finish_at


def _v2_capture(index: int, reservation: dict) -> Path:
    return RUNS / v1.runner.CONTEXT_LABELS[index] / reservation["attempt_ids"][index]


def _actual_predecessor_finish(index: int, reservation: dict) -> datetime:
    if index not in (2, 3):
        raise ValueError("Continuation predecessor index invalid")
    earlier = index - 1
    label, attempt = v1.runner.CONTEXT_LABELS[earlier], reservation["attempt_ids"][earlier]
    root = v1.RUNS if earlier == 1 else RUNS
    receipt = v1._json(root / label / attempt / "launch_finish.json")
    if (receipt.get("status") != v1.FINISH_OK or
            receipt.get("context_index") != earlier or
            receipt.get("context_label") != label or
            receipt.get("attempt_id") != attempt):
        raise ValueError("Continuation predecessor finish differs")
    return v1.old_gate.utc(receipt.get("wall_utc"))


def _check_new_completed(index: int, reservation: dict, cumulative: float,
                         after: datetime, *, preview: dict | None = None) -> tuple[float, datetime]:
    label, attempt = v1.runner.CONTEXT_LABELS[index], reservation["attempt_ids"][index]
    capture, output = _v2_capture(index, reservation), v1.runner.RUNS / label / attempt
    start = v1._json(capture / "launch_start.json")
    finish = preview if preview is not None else v1._json(capture / "launch_finish.json")
    command = [sys.executable, str(WORKER.resolve()), "--run-context", str(index),
               "--attempt-id", attempt, "--continuation-freeze-sha256", v1.sha(FREEZE)]
    if (after != _actual_predecessor_finish(index, reservation) or
            start.get("status") != "context_launch_started" or
            start.get("context_index") != index or start.get("context_label") != label or
            start.get("attempt_id") != attempt or start.get("freeze_sha256") != V1_FREEZE_SHA or
            start.get("continuation_freeze_sha256") != v1.sha(FREEZE) or
            start.get("entrypoint_sha256") != v1.sha(WORKER) or
            start.get("worker_command") != command or start.get("worker_output") != str(output) or
            start.get("prior_model_work_seconds") != reservation["prior_model_work_seconds"] or
            start.get("cumulative_prior_model_work_seconds") != cumulative or
            start.get("process_cap_seconds") != 300 or
            start.get("max_mps_driver_bytes") != 30 * 1024**3 or
            start.get("max_output_bytes") != 2 * 1024**3 or
            start.get("max_worker_bytes") != v1.runner.MAX_WORKER_BYTES or
            start.get("max_capture_bytes") != 8 * 1024**2 or
            start.get("planned_counts") != v1.runner.COUNTS or
            not after < v1.old_gate.utc(start.get("wall_utc"))):
        raise ValueError("Continuation start receipt differs")
    _check_pre_host_record(capture, start)
    _require_strict_timing(start, finish)
    if (finish.get("status") != v1.FINISH_OK or
            finish.get("context_index") != index or finish.get("context_label") != label or
            finish.get("attempt_id") != attempt or finish.get("freeze_sha256") != V1_FREEZE_SHA or
            finish.get("continuation_freeze_sha256") != v1.sha(FREEZE) or
            finish.get("entrypoint_sha256") != v1.sha(WORKER) or
            finish.get("prior_model_work_seconds") != reservation["prior_model_work_seconds"] or
            finish.get("cumulative_prior_model_work_seconds") != cumulative or
            finish.get("worker_returncode") != 0 or finish.get("auditor_returncode") != 0 or
            finish.get("post_run_source_verified") is not True or
            finish.get("error_sha256") is not None or
            finish.get("result_sha256") != v1.sha(output / "result.json") or
            finish.get("audit_sha256") != v1.sha(output / "audit.json") or
            finish.get("worker_output_bytes") != v1._tree_bytes(output)):
        raise ValueError("Continuation finish receipt differs")
    claim = v1._json(capture / "worker_claim.json")
    if (claim.get("status") != "context_worker_claimed_no_retry" or
            claim.get("context_index") != index or claim.get("context_label") != label or
            claim.get("attempt_id") != attempt or claim.get("freeze_sha256") != V1_FREEZE_SHA or
            claim.get("continuation_freeze_sha256") != v1.sha(FREEZE) or
            claim.get("entrypoint_sha256") != v1.sha(WORKER) or
            not v1.old_gate.utc(start["wall_utc"]) < v1.old_gate.utc(claim.get("wall_utc")) <
                v1.old_gate.utc(finish["worker_finished_utc"])):
        raise ValueError("Continuation exclusive worker claim differs")
    result, audit = v1._json(output / "result.json"), v1._json(output / "audit.json")
    if (result.get("status") != v1.runner.STATUS or
            result.get("attempt_id") != attempt or result.get("freeze_sha256") != V1_FREEZE_SHA or
            result.get("context", {}).get("index") != index or
            result.get("context", {}).get("label") != label or
            result.get("actual_counts") != v1.runner.COUNTS or
            audit.get("status") != "audited_voxtral_fp32_nonbeaver_saved_data_only" or
            audit.get("attempt_id") != attempt or audit.get("freeze_sha256") != V1_FREEZE_SHA or
            audit.get("result_sha256") != v1.sha(output / "result.json") or
            audit.get("worker_invalidity_disagreement") is not False):
        raise ValueError("Continuation saved result/audit differs")
    if preview is None:
        if (finish.get("combined_output_bytes_before_finish") !=
                v1._tree_bytes(output) + v1._tree_bytes(capture) -
                (capture / "launch_finish.json").stat().st_size or
                v1._tree_bytes(capture) + v1._tree_bytes(output) > v1.MAX_OUTPUT_BYTES):
            raise ValueError("Continuation output bytes differ")
    elif v1._tree_bytes(capture) + v1._tree_bytes(output) + v1.FINISH_RESERVE_BYTES >= v1.MAX_OUTPUT_BYTES:
        raise ValueError("Continuation output cap cannot fit finish receipt")
    v1._success_files(output, result, audit)
    v1._rerun_saved_prior_audit(output)
    _check_post_host_record(capture, start, finish)
    return finish["model_work_elapsed_seconds"], v1.old_gate.utc(finish["wall_utc"])


def check_prior(index: int, reservation: dict) -> float:
    if type(index) is not int or index not in (2, 3):
        raise ValueError("Continuation index must be 2 or 3")
    if reservation.get("campaign_id") != CAMPAIGN:
        raise ValueError("Different campaign reservation")
    used0 = v1._check_prior_entries(1, reservation)
    if used0 != PRIOR_WORK[0]:
        raise ValueError("Original context 00 model-work changed")
    used1, after = _check_piper_one(reservation, used0)
    used = used0 + used1
    if index == 3:
        gained, after = _check_new_completed(2, reservation,
                                             reservation["prior_model_work_seconds"] + used, after)
        used += gained
    for later in range(index + 1, 4):
        if (_v2_capture(later, reservation).exists() or
                (v1.runner.RUNS / v1.runner.CONTEXT_LABELS[later] /
                 reservation["attempt_ids"][later]).exists()):
            raise ValueError("Later continuation attempt exists out of order")
    return used


def _run_command(argv: list[str]) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True)
    if not done.stdout:
        raise ValueError(f"Host command returned no evidence: {argv}")
    return done.stdout


def _power_events(log: str, start: datetime, end: datetime) -> list[str]:
    if not POWER_EVENT.search(log):
        raise ValueError("Host power log has no parseable timestamped events")
    found = []
    for match in POWER_EVENT.finditer(log):
        when = datetime.strptime(" ".join(match.group(1, 2, 3)), "%Y-%m-%d %H:%M:%S %z")
        event = match.group(4).strip()
        if start <= when <= end and (event in ("Sleep", "DarkWake", "Wake") or
                                      "Thermal" in event):
            found.append(match.group(0))
    return found


def _power_log_coverage(log: str, start: datetime, end: datetime,
                        collected: datetime) -> None:
    """Require a log that starts before the interval and ends at collection."""
    events = list(POWER_EVENT.finditer(log))
    snapshots = list(POWER_TERMINAL.finditer(log))
    if not events or not snapshots:
        raise ValueError("Host power-log coverage lacks events or terminal snapshot")
    parse = lambda match: datetime.strptime(
        " ".join(match.group(1, 2, 3)), "%Y-%m-%d %H:%M:%S %z")
    oldest = min(parse(match) for match in events)
    terminal = parse(snapshots[-1])
    if (oldest > start or terminal < end - timedelta(seconds=1) or
            terminal < collected - timedelta(seconds=5) or
            terminal > collected + timedelta(seconds=1)):
        raise ValueError("Host power-log coverage is stale or truncated")


def _host_probe(*, require_stable: bool = True) -> dict:
    batt = _run_command(["pmset", "-g", "batt"])
    lid = _run_command(["ioreg", "-a", "-r", "-c", "IOPMrootDomain", "-d", "1"])
    battery = _run_command(["ioreg", "-a", "-r", "-c", "AppleSmartBattery", "-d", "1"])
    log = _run_command(["pmset", "-g", "log"])
    collected = datetime.now(timezone.utc)
    if require_stable:
        if ("Now drawing from 'AC Power'" not in batt or
                plistlib.loads(battery.encode())[0].get("ExternalConnected") is not True):
            raise ValueError("Host is not on external power")
        if plistlib.loads(lid.encode())[0].get("AppleClamshellState") is not False:
            raise ValueError("Host clamshell is closed or unverifiable")
        _power_log_coverage(log, collected - timedelta(minutes=10), collected,
                            collected)
        if _power_events(log, collected - timedelta(minutes=10), collected):
            raise ValueError("Host had recent sleep, DarkWake, or thermal transition")
    return {"collected_utc": collected.isoformat(),
            "commands": HOST_COMMANDS,
            "outputs": {"power": batt, "clamshell": lid, "battery": battery,
                        "power_log": log}}


def _store_host_probe(capture: Path, probe: dict, name: str) -> dict:
    """Keep exact raw pmset output losslessly without bloating a JSON receipt."""
    log = probe["outputs"].pop("power_log")
    raw = log.encode()
    packed = gzip.compress(raw, mtime=0)
    if len(packed) >= v1.MAX_CAPTURE_BYTES or len(raw) > 128 * 1024**2:
        raise ValueError("Host power log cannot fit the capture cap")
    path = capture / f"{name}_power_log.txt.gz"
    with path.open("xb") as stream:
        stream.write(packed)
        stream.flush()
    probe["power_log_archive"] = {
        "file": path.name, "compressed_sha256": v1.sha(path),
        "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw)}
    return probe


def _load_power_log(capture: Path, probe: dict) -> str:
    archive = probe.get("power_log_archive")
    if not isinstance(archive, dict) or archive.get("file") not in (
            "pre_power_log.txt.gz", "post_power_log.txt.gz"):
        raise ValueError("Host power log archive missing")
    path = capture / archive["file"]
    if (not path.is_file() or path.is_symlink() or
            v1.sha(path) != archive.get("compressed_sha256") or
            path.stat().st_size >= v1.MAX_CAPTURE_BYTES):
        raise ValueError("Host power log archive changed")
    raw = gzip.decompress(path.read_bytes())
    if (len(raw) != archive.get("raw_bytes") or len(raw) > 128 * 1024**2 or
            hashlib.sha256(raw).hexdigest() != archive.get("raw_sha256")):
        raise ValueError("Host raw power log changed")
    return raw.decode()


def _check_pre_host_record(capture: Path, start: dict) -> None:
    probe = start.get("pre_host")
    if not isinstance(probe, dict) or not isinstance(probe.get("outputs"), dict):
        raise ValueError("Pre-run host evidence absent")
    if probe.get("commands") != HOST_COMMANDS:
        raise ValueError("Pre-run host command differs")
    outputs = probe["outputs"]
    if ("Now drawing from 'AC Power'" not in outputs.get("power", "") or
            plistlib.loads(outputs["battery"].encode())[0].get("ExternalConnected") is not True or
            plistlib.loads(outputs["clamshell"].encode())[0].get("AppleClamshellState") is not False):
        raise ValueError("Pre-run host was not plugged in with lid open")
    when = v1.old_gate.utc(probe.get("collected_utc"))
    log = _load_power_log(capture, probe)
    if (not when <= v1.old_gate.utc(start["wall_utc"]) or
            (v1.old_gate.utc(start["wall_utc"]) - when).total_seconds() > 30 or
            _power_events(log, when - timedelta(minutes=10), when)):
        raise ValueError("Pre-run host stability differs")
    _power_log_coverage(log, when - timedelta(minutes=10), when, when)


def _check_post_host_record(capture: Path, start: dict, finish: dict) -> None:
    host = finish.get("post_host")
    if not isinstance(host, dict) or not isinstance(host.get("outputs"), dict):
        raise ValueError("Post-run host evidence absent")
    if host.get("commands") != HOST_COMMANDS:
        raise ValueError("Post-run host command differs")
    if ("Now drawing from 'AC Power'" not in host["outputs"].get("power", "") or
            plistlib.loads(host["outputs"]["battery"].encode())[0].get("ExternalConnected") is not True or
            plistlib.loads(host["outputs"]["clamshell"].encode())[0].get("AppleClamshellState") is not False):
        raise ValueError("Post-run host was not plugged in with lid open")
    log = _load_power_log(capture, host)
    collected = v1.old_gate.utc(host.get("collected_utc"))
    finished = v1.old_gate.utc(finish["wall_utc"])
    if not finished <= collected <= finished + timedelta(seconds=30):
        raise ValueError("Post-run host collection time differs")
    _power_log_coverage(log, v1.old_gate.utc(start["wall_utc"]) - timedelta(seconds=1),
                        finished + timedelta(seconds=1), collected)
    if _power_events(log, v1.old_gate.utc(start["wall_utc"]) - timedelta(seconds=1),
                     finished + timedelta(seconds=1)):
        raise ValueError("Host slept, DarkWoke, or had thermal transition during slot")


def verify_worker_start(index: int, attempt_id: str, continuation_freeze_sha256: str) -> dict:
    """Validate then atomically create the sole no-retry claim before model load."""
    freeze, upstream, reservation = verify_freeze()
    used = check_prior(index, reservation)
    label = v1.runner.CONTEXT_LABELS[index]
    capture = _v2_capture(index, reservation)
    output = v1.runner.RUNS / label / attempt_id
    if not (capture / "launch_start.json").is_file():
        raise ValueError("Continuation worker admission failed: launch absent")
    start = v1._json(capture / "launch_start.json")
    expected = [sys.executable, str(WORKER.resolve()), "--run-context", str(index),
                "--attempt-id", attempt_id, "--continuation-freeze-sha256",
                continuation_freeze_sha256]
    if (attempt_id != reservation["attempt_ids"][index] or
            continuation_freeze_sha256 != v1.sha(FREEZE) or
            start.get("worker_command") != expected or
            start.get("continuation_freeze_sha256") != continuation_freeze_sha256 or
            start.get("entrypoint_sha256") != v1.sha(WORKER) or
            start.get("freeze_sha256") != V1_FREEZE_SHA or
            start.get("cumulative_prior_model_work_seconds") !=
                upstream["prior_model_work_seconds"] + used or
            start.get("context_index") != index or start.get("context_label") != label or
            start.get("attempt_id") != attempt_id or start.get("worker_output") != str(output) or
            start.get("prior_model_work_seconds") != upstream["prior_model_work_seconds"] or
            start.get("planned_counts") != v1.runner.COUNTS or
            start.get("process_cap_seconds") != 300 or
            start.get("max_mps_driver_bytes") != 30 * 1024**3 or
            start.get("max_output_bytes") != 2 * 1024**3 or
            start.get("max_worker_bytes") != v1.runner.MAX_WORKER_BYTES or
            start.get("max_capture_bytes") != 8 * 1024**2 or
            output.exists() or (capture / "worker_claim.json").exists() or
            (v1.RUNS / label / attempt_id).exists() or
            v1.old_gate.utc(start["wall_utc"]) <= _actual_predecessor_finish(index, reservation) or
            v1.old_gate.utc(start["wall_utc"]) <= v1.old_gate.utc(freeze["reviewed_utc"])):
        raise ValueError("Continuation worker admission failed")
    _check_pre_host_record(capture, start)
    v1._put(capture / "worker_claim.json", {
        "status": "context_worker_claimed_no_retry", "context_index": index,
        "context_label": label, "attempt_id": attempt_id,
        "freeze_sha256": V1_FREEZE_SHA,
        "continuation_freeze_sha256": continuation_freeze_sha256,
        "entrypoint_sha256": v1.sha(WORKER), "wall_utc": _utc_now()})
    return upstream


def launch_context(index: int) -> dict:
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Continuation requires pinned second_architecture environment")
    freeze, upstream, reservation = verify_freeze()
    used = check_prior(index, reservation)
    predecessor_finish = _actual_predecessor_finish(index, reservation)
    label, attempt = v1.runner.CONTEXT_LABELS[index], reservation["attempt_ids"][index]
    capture, output = _v2_capture(index, reservation), v1.runner.RUNS / label / attempt
    if (capture.exists() or output.exists() or
            (v1.RUNS / label / attempt).exists()):
        raise ValueError("Reserved continuation attempt already started; no retry")
    if (shutil.disk_usage(HERE).free < 12 * 1024**3 or
            v1._tree_bytes(v1.RUNS) + v1._tree_bytes(RUNS) +
            v1._tree_bytes(v1.runner.RUNS) >= 8 * 1024**3):
        raise ValueError("Continuation disk or campaign output cap reached")
    cumulative = upstream["prior_model_work_seconds"] + used
    if cumulative + (4 - index) * 300 > 4500:
        raise ValueError("Cumulative model-work cap would be exceeded")
    host = _host_probe()  # all host checks precede the exclusive attempt directory
    if len(gzip.compress(host["outputs"]["power_log"].encode(), mtime=0)) >= v1.MAX_CAPTURE_BYTES:
        raise ValueError("Pre-run host log exceeds capture cap")
    capture.mkdir(parents=True, exist_ok=False)
    host = _store_host_probe(capture, host, "pre")
    continuation_sha = v1.sha(FREEZE)
    command = [sys.executable, str(WORKER.resolve()), "--run-context", str(index),
               "--attempt-id", attempt, "--continuation-freeze-sha256", continuation_sha]
    start = {"status": "context_launch_started", "context_index": index,
             "context_label": label, "attempt_id": attempt,
             "freeze_sha256": V1_FREEZE_SHA,
             "continuation_freeze_sha256": continuation_sha,
             "entrypoint_sha256": v1.sha(WORKER), "wall_utc": _utc_now(),
             "worker_command": command, "worker_output": str(output),
             "prior_model_work_seconds": upstream["prior_model_work_seconds"],
             "cumulative_prior_model_work_seconds": cumulative,
             "process_cap_seconds": 300, "max_mps_driver_bytes": 30 * 1024**3,
             "max_output_bytes": 2 * 1024**3, "max_worker_bytes": v1.runner.MAX_WORKER_BYTES,
             "max_capture_bytes": 8 * 1024**2, "planned_counts": v1.runner.COUNTS,
             "pre_host": host}
    if v1.old_gate.utc(start["wall_utc"]) <= predecessor_finish:
        raise ValueError("Continuation start predates predecessor finish")
    v1._put(capture / "launch_start.json", start)
    finish = {"status": v1.FINISH_FAILED, "context_index": index,
              "context_label": label, "attempt_id": attempt,
              "freeze_sha256": V1_FREEZE_SHA,
              "continuation_freeze_sha256": continuation_sha,
              "entrypoint_sha256": v1.sha(WORKER),
              "worker_returncode": None, "auditor_returncode": None,
              "result_sha256": None, "audit_sha256": None, "error_sha256": None,
              "post_run_source_verified": False, "model_work_elapsed_seconds": 0.0,
              "prior_model_work_seconds": upstream["prior_model_work_seconds"],
              "cumulative_prior_model_work_seconds": cumulative}
    began = time.monotonic()
    try:
        finish["worker_started_utc"] = _utc_now()
        try:
            code, elapsed, reason = v1._bounded(command, capture,
                ("worker_stdout.txt", "worker_stderr.txt"), output, 300)
        finally:
            finish["worker_finished_utc"] = _utc_now()
        finish["worker_returncode"] = code
        finish["model_work_elapsed_seconds"] = elapsed
        if reason or code != 0 or elapsed >= 300:
            raise ValueError(reason or "Continuation worker failed or exceeded cap")
        claim = v1._json(capture / "worker_claim.json")
        if (claim.get("status") != "context_worker_claimed_no_retry" or
                claim.get("continuation_freeze_sha256") != continuation_sha or
                claim.get("entrypoint_sha256") != v1.sha(WORKER)):
            raise ValueError("Continuation worker claim absent")
        result_path = output / "result.json"
        result = v1._json(result_path)
        if ((output / "error.json").exists() or
                result.get("status") != v1.runner.STATUS or
                result.get("attempt_id") != attempt or
                result.get("freeze_sha256") != V1_FREEZE_SHA or
                result.get("actual_counts") != v1.runner.COUNTS):
            raise ValueError("Frozen worker result differs")
        finish["result_sha256"] = v1.sha(result_path)
        if verify_freeze()[0] != freeze:
            raise ValueError("Continuation freeze changed after worker")
        finish["post_run_source_verified"] = True
        audit_command = [sys.executable, str(v1.AUDITOR), "--run-dir", str(output),
                         "--output", str(output / "audit.json")]
        code, _, reason = v1._bounded(audit_command, capture,
                                      ("audit_stdout.txt", "audit_stderr.txt"), output, 120)
        finish["auditor_returncode"] = code
        audit_path = output / "audit.json"
        if reason or code != 0 or not audit_path.is_file():
            raise ValueError(reason or "Frozen CPU auditor rejected context")
        audit = v1._json(audit_path)
        if (audit.get("status") != "audited_voxtral_fp32_nonbeaver_saved_data_only" or
                audit.get("result_sha256") != finish["result_sha256"] or
                audit.get("freeze_sha256") != V1_FREEZE_SHA or
                audit.get("worker_invalidity_disagreement") is not False):
            raise ValueError("Frozen CPU audit binding differs")
        finish["audit_sha256"] = v1.sha(audit_path)
        finish["status"] = v1.FINISH_OK
    except Exception as exc:
        finish["failure_reason"] = str(exc)
        finish["exception_type"] = type(exc).__name__
        error = output / "error.json"
        finish["error_sha256"] = v1.sha(error) if error.is_file() else None
    finally:
        finish["wall_utc"] = _utc_now()
        finish["elapsed_seconds"] = time.monotonic() - began
        try:
            post = _host_probe(require_stable=False)
            finish["post_host"] = _store_host_probe(capture, post, "post")
            _check_post_host_record(capture, start, finish)
        except Exception as exc:
            finish["status"] = v1.FINISH_FAILED
            finish["failure_reason"] = f"Post-run host state invalid: {exc}"
        try:
            _require_strict_timing(start, finish)
        except ValueError:
            finish["status"] = v1.FINISH_FAILED
            finish["failure_reason"] = "Continuation five-second timing rule failed"
        if finish["status"] == v1.FINISH_OK:
            finish["worker_output_bytes"] = v1._tree_bytes(output)
            try:
                _check_new_completed(index, reservation, cumulative,
                    predecessor_finish,
                    preview=finish)
            except Exception as exc:
                finish["status"] = v1.FINISH_FAILED
                finish["failure_reason"] = f"Pre-finish validation failed: {exc}"
        v1._write_finish(capture, output, finish)
        if finish["status"] == v1.FINISH_OK:
            _check_new_completed(index, reservation, cumulative,
                predecessor_finish)
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--run-context", type=int)
    args = parser.parse_args()
    if args.check_freeze:
        verify_freeze()
        print(json.dumps({"status": "continuation_freeze_verified_read_only",
                          "freeze_sha256": v1.sha(FREEZE)}))
        return
    result = launch_context(args.run_context)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != v1.FINISH_OK:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
