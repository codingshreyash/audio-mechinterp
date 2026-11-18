"""One-shot external launcher for the frozen old-FIT Stage-B replay.

This source alone cannot run a model. An independent launcher review, a later
committed external freeze, and exact Stage-A/Stage-B receipts are required.
"""

from __future__ import annotations

import argparse
import hashlib
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

import voxtral_old_fit_handoff_stage_b_runner_v1 as driver
import voxtral_old_fit_handoff_stage_b_audit_v1 as auditor
import voxtral_old_fit_handoff_replay_stage_b_v1 as replay


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs/voxtral_old_fit_handoff_stage_b_launch_v1"
FREEZE = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_EXTERNAL_FREEZE_V1.json"
FREEZE_REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_EXTERNAL_FREEZE_INDEPENDENT_REVIEW_V1.md"
FP32_FREEZE = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_FP32_CONTINUATION_FREEZE_V1.json"
FP32_FREEZE_REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_FP32_CONTINUATION_FREEZE_INDEPENDENT_REVIEW_V1.md"
PROTOCOL = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_LAUNCH_PROTOCOL_V1.md"
REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_LAUNCH_INDEPENDENT_SOURCE_REVIEW_V1.md"
DRIVER_REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_DRIVER_CONTINUATION_INDEPENDENT_REVIEW_V1.md"
AUDITOR_REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_AUDITOR_INDEPENDENT_SOURCE_REVIEW_V1.md"
WORKER = HERE / "voxtral_old_fit_handoff_stage_b_worker_v1.py"
AUDIT_ADAPTER = HERE / "voxtral_old_fit_handoff_stage_b_audit_adapter_v1.py"
TESTS = HERE / "test_voxtral_old_fit_handoff_stage_b_launch_gate_v1.py"
BF16_B_MANIFEST = driver.STAGE_B_BF16_MANIFEST
BF16_B_REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_BF16_RESULT_INDEPENDENT_REVIEW_V1.md"
PHASES = driver.PHASES
AUDITOR_SHA = "45cbf7f1b1767d8a736ab5569a06a152b8c2705c010fbfb9cc45f7d039575cf0"
AUDITOR_TEST_SHA = "9d8b1e777706420c1deef6ee9a719eba8a7d589403252db513baac07df8b7810"
DRIVER_SHA = "f685a920a61bb1885ce86df44e41899adca886f42b90f5a6577ae7e6cffaf71e"
DRIVER_TEST_SHA = "517c136a45e84ca7a5173d0424e56fb63eda668807b97f6e7cbc54632b651724"
DRIVER_AMENDMENT_SHA = "392c5e061c1c1fdad8610eab96d46840073d4a50099c02d98cc4e93065895409"
AUDITOR_REVIEW_GO = "Decision: GO for repinned Stage-B auditor source only; live Stage B remains HOLD pending its separate admission, chronology, freeze and review gates."
DRIVER_REVIEW_GO = "Decision: GO for Stage-B continuation driver source; live model HOLD until launcher review and freeze."
LAUNCHER_REVIEW_GO = "Decision: GO for Stage-B launcher source; model run HOLD until external freeze."
BF16_B_REVIEW_GO = "Decision: GO for Stage-B BF16 saved-result provenance; FP32 Stage-B HOLD pending separate gate."
FREEZE_REVIEW_GO = "Decision: GO for Stage-B BF16 external freeze provenance; BF16 model may run under the pinned gate."
FP32_FREEZE_REVIEW_GO = "Decision: GO for Stage-B FP32 continuation freeze provenance; FP32 model may run under the pinned gate."
STATUS = "frozen_old_fit_handoff_stage_b_bf16_v1"
FP32_STATUS = "frozen_old_fit_handoff_stage_b_fp32_continuation_v1"
AUDIT_SCOPE = "old_fit_one_case_saved_data_arithmetic_only_no_execution_attestation"
LAUNCH_SUCCESS = {
    "bf16_direct_lm": "launch_completed_audited_old_fit_handoff_stage_b_bf16_direct_lm_only",
    "fp32_replay": "launch_completed_audited_old_fit_handoff_stage_b_fp32_replay_only",
}
PROCESS_GRACE_SECONDS = 120
AUDIT_SECONDS = 120
MAX_CAPTURE_BYTES = 8 * 1024**2
PARENT_OUTPUT_CAP_BYTES = 24 * 1024**2
FINAL_RECEIPT_RESERVE_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 5 * 1024**3
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")

BOUND_FILES = {
    "prospective_protocol_sha256": auditor.PROTOCOL,
    "replay_primitives_sha256": HERE / "voxtral_old_fit_handoff_replay_stage_b_v1.py",
    "driver_sha256": Path(driver.__file__).resolve(),
    "driver_tests_sha256": HERE / "test_voxtral_old_fit_handoff_stage_b_runner_v1.py",
    "driver_amendment_sha256": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_CONTINUATION_ADMISSION_AMENDMENT_V1.md",
    "driver_review_sha256": DRIVER_REVIEW,
    "auditor_sha256": Path(auditor.__file__).resolve(),
    "auditor_tests_sha256": HERE / "test_voxtral_old_fit_handoff_stage_b_audit_v1.py",
    "auditor_review_sha256": AUDITOR_REVIEW,
    "bf16_stage_a_manifest_sha256": driver.STAGE_A_MANIFESTS["bf16"],
    "bf16_posthoc_receipt_sha256": driver.continuation.POSTHOC_RECEIPT,
    "bf16_posthoc_review_sha256": driver.continuation.POSTHOC_REVIEW,
    "fp32_stage_a_manifest_sha256": driver.STAGE_A_MANIFESTS["fp32"],
    "fp32_stage_a_review_sha256": driver.STAGE_A_FP32_RESULT_REVIEW,
    "fp32_stage_a_freeze_sha256": driver.continuation.FREEZE,
    "stage_b_launch_protocol_sha256": PROTOCOL,
    "stage_b_launch_gate_sha256": Path(__file__).resolve(),
    "stage_b_launch_tests_sha256": TESTS,
    "stage_b_worker_sha256": WORKER,
    "stage_b_audit_adapter_sha256": AUDIT_ADAPTER,
    "stage_b_launcher_review_sha256": REVIEW,
}
REVIEWED_SOURCE_KEYS = ("stage_b_launch_protocol_sha256", "stage_b_launch_gate_sha256",
                        "stage_b_launch_tests_sha256", "stage_b_worker_sha256",
                        "stage_b_audit_adapter_sha256", "stage_b_launcher_review_sha256")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Git check failed: {' '.join(args[:2])}")
    return result.stdout


def regular(repo: Path, path: Path) -> str:
    """Use the stated repository path; reject symlinked path components."""
    relative = path.absolute().relative_to(repo.absolute())
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Bound path invalid")
    current = repo.absolute()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Bound path contains symlink: {path.name}")
    if not path.is_file():
        raise ValueError(f"Bound file absent: {path.name}")
    return relative.as_posix()


def committed_same(repo: Path, path: Path, ref: str = "HEAD") -> None:
    rel = regular(repo, path)
    if path.read_bytes() != _git(repo, "show", f"{ref}:{rel}"):
        raise ValueError(f"Bound file differs from {ref}: {path.name}")


def first_add(repo: Path, path: Path) -> str:
    committed_same(repo, path)
    rel = regular(repo, path)
    lines = _git(repo, "log", "--diff-filter=A", "--format=%H", "HEAD", "--", rel).decode().splitlines()
    if len(lines) != 1 or HEX40.fullmatch(lines[0]) is None:
        raise ValueError(f"First-add history invalid: {path.name}")
    if path.read_bytes() != _git(repo, "show", f"{lines[0]}:{rel}"):
        raise ValueError(f"Bound receipt changed after first-add: {path.name}")
    return lines[0]


def ancestor(repo: Path, earlier: str, later: str) -> None:
    _git(repo, "merge-base", "--is-ancestor", earlier, later)


def utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("UTC timestamp invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("UTC timestamp invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("UTC timestamp invalid")
    return parsed


def same_stage_a() -> dict:
    """The reviewed driver rehashes all 24 raw files and validates both arms."""
    stage_a = driver.verify_stage_a_manifests()
    if (stage_a.get("original_bf16_launch_status") != "launch_failed" or
            set(stage_a.get("arms", {})) != {"bf16", "fp32"} or
            not math.isclose(stage_a.get("stage_a_model_work_seconds", math.nan),
                             63.245136791840196 + 155.3182365419343,
                             rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Stage-A saved failure and continuation binding invalid")
    return stage_a


def verify_freeze() -> tuple[dict, dict]:
    """Require independently reviewed code and a later first-add-immutable freeze."""
    stage_a = same_stage_a()
    for path in BOUND_FILES.values():
        committed_same(REPO, path)
    for key, digest in (("auditor_sha256", AUDITOR_SHA),
                        ("auditor_tests_sha256", AUDITOR_TEST_SHA),
                        ("driver_sha256", DRIVER_SHA),
                        ("driver_tests_sha256", DRIVER_TEST_SHA),
                        ("driver_amendment_sha256", DRIVER_AMENDMENT_SHA),
                        ("prospective_protocol_sha256", auditor.PROTOCOL_SHA),
                        ("replay_primitives_sha256", auditor.PRIMITIVES_SHA)):
        if sha(BOUND_FILES[key]) != digest:
            raise ValueError(f"Stage-B reviewed source pin differs: {key}")
    if DRIVER_REVIEW_GO not in DRIVER_REVIEW.read_text():
        raise ValueError("Stage-B driver continuation review lacks GO")
    if AUDITOR_REVIEW_GO not in AUDITOR_REVIEW.read_text():
        raise ValueError("Stage-B auditor source review lacks GO")
    if LAUNCHER_REVIEW_GO not in REVIEW.read_text():
        raise ValueError("Stage-B launcher source review lacks GO")
    committed_same(REPO, FREEZE)
    receipt = read_json(FREEZE)
    expected = {"status", "permitted_phases", "audit_scope", "launcher_review_commit",
                "reviewed_utc", "stage_a_model_work_seconds", "max_total_bytes",
                "max_total_seconds", *BOUND_FILES}
    if (set(receipt) != expected or receipt["status"] != STATUS or
            receipt["permitted_phases"] != ["bf16_direct_lm"] or
            receipt["audit_scope"] != AUDIT_SCOPE or
            receipt["max_total_bytes"] != MAX_TOTAL_BYTES or
            receipt["max_total_seconds"] != driver.MAX_TOTAL_SECONDS or
            receipt["stage_a_model_work_seconds"] != stage_a["stage_a_model_work_seconds"]):
        raise ValueError("Stage-B external freeze schema/scope/caps invalid")
    utc(receipt["reviewed_utc"])
    review_commit = receipt["launcher_review_commit"]
    if not isinstance(review_commit, str) or HEX40.fullmatch(review_commit) is None:
        raise ValueError("Launcher review commit malformed")
    for key, path in BOUND_FILES.items():
        value = receipt[key]
        if not isinstance(value, str) or HEX64.fullmatch(value) is None or value != sha(path):
            raise ValueError(f"Stage-B freeze bound hash differs: {key}")
        if key in REVIEWED_SOURCE_KEYS:
            rel = regular(REPO, path)
            if hashlib.sha256(_git(REPO, "show", f"{review_commit}:{rel}")).hexdigest() != value:
                raise ValueError(f"Launcher review commit lacks reviewed {key}")
    if receipt["auditor_sha256"] != AUDITOR_SHA:
        raise ValueError("Stage-B auditor pin differs")
    review_rel = regular(REPO, REVIEW)
    last_review = _git(REPO, "log", "-1", "--format=%H", "--", review_rel).decode().strip()
    if review_commit != last_review:
        raise ValueError("Launcher review commit differs")
    for path in (DRIVER_REVIEW, AUDITOR_REVIEW, driver.STAGE_A_FP32_RESULT_REVIEW):
        rel = regular(REPO, path)
        commit = _git(REPO, "log", "-1", "--format=%H", "--", rel).decode().strip()
        ancestor(REPO, commit, review_commit)
    for review, keys in ((DRIVER_REVIEW, ("driver_sha256", "driver_tests_sha256",
                                         "driver_amendment_sha256")),
                         (AUDITOR_REVIEW, ("auditor_sha256", "auditor_tests_sha256"))):
        review_rel = regular(REPO, review)
        source_review_commit = _git(REPO, "log", "-1", "--format=%H", "--", review_rel).decode().strip()
        for key in keys:
            bound_rel = regular(REPO, BOUND_FILES[key])
            if hashlib.sha256(_git(REPO, "show", f"{source_review_commit}:{bound_rel}")).hexdigest() != receipt[key]:
                raise ValueError(f"Source review commit lacks reviewed {key}")
    freeze_add = first_add(REPO, FREEZE)
    if freeze_add == review_commit:
        raise ValueError("Stage-B freeze was self-authored in launcher review")
    ancestor(REPO, review_commit, freeze_add)
    committed_same(REPO, FREEZE_REVIEW)
    freeze_review_add = first_add(REPO, FREEZE_REVIEW)
    if freeze_add == freeze_review_add or FREEZE_REVIEW_GO not in FREEZE_REVIEW.read_text():
        raise ValueError("BF16 external freeze independent review absent")
    ancestor(REPO, freeze_add, freeze_review_add)
    if driver.LIVE_STAGE_B_SOURCE_PINS is not None:
        # The driver CLI must remain closed; the reviewed worker is the only live path.
        raise ValueError("Stage-B driver CLI unexpectedly enabled")
    return receipt, stage_a


def verify_fp32_freeze(stage_a: dict, bf16_prior: dict) -> dict:
    """A second phase-specific freeze binds the later BF16 B result and review."""
    base_sha = sha(FREEZE)
    committed_same(REPO, FP32_FREEZE)
    receipt = read_json(FP32_FREEZE)
    expected = {"status", "permitted_phase", "audit_scope", "base_freeze_sha256",
                "bf16_b_manifest_sha256", "bf16_b_review_sha256",
                "bf16_b_manifest_commit", "bf16_b_review_commit", "reviewed_utc",
                "max_total_bytes", "max_total_seconds", "stage_a_model_work_seconds",
                "bf16_b_model_work_seconds"}
    if (set(receipt) != expected or receipt["status"] != FP32_STATUS or
            receipt["permitted_phase"] != "fp32_replay" or
            receipt["audit_scope"] != AUDIT_SCOPE or
            receipt["base_freeze_sha256"] != base_sha or
            receipt["bf16_b_manifest_sha256"] != sha(BF16_B_MANIFEST) or
            receipt["bf16_b_review_sha256"] != sha(BF16_B_REVIEW) or
            receipt["bf16_b_manifest_commit"] != first_add(REPO, BF16_B_MANIFEST) or
            receipt["bf16_b_review_commit"] != first_add(REPO, BF16_B_REVIEW) or
            receipt["max_total_bytes"] != MAX_TOTAL_BYTES or
            receipt["max_total_seconds"] != driver.MAX_TOTAL_SECONDS or
            receipt["stage_a_model_work_seconds"] != stage_a["stage_a_model_work_seconds"] or
            receipt["bf16_b_model_work_seconds"] != bf16_prior["elapsed_seconds"]):
        raise ValueError("FP32 Stage-B continuation freeze schema/binding differs")
    utc(receipt["reviewed_utc"])
    freeze_add = first_add(REPO, FP32_FREEZE)
    ancestor(REPO, receipt["bf16_b_review_commit"], freeze_add)
    if freeze_add == receipt["bf16_b_review_commit"]:
        raise ValueError("FP32 freeze predates independent BF16 B review")
    committed_same(REPO, FP32_FREEZE_REVIEW)
    review_add = first_add(REPO, FP32_FREEZE_REVIEW)
    if review_add == freeze_add or FP32_FREEZE_REVIEW_GO not in FP32_FREEZE_REVIEW.read_text():
        raise ValueError("FP32 external freeze independent review absent")
    ancestor(REPO, freeze_add, review_add)
    return receipt


def verify_phase_freeze(phase: str) -> tuple[dict, dict, dict | None]:
    base, stage_a = verify_freeze()
    if phase == "bf16_direct_lm":
        return base, stage_a, None
    if phase != "fp32_replay":
        raise ValueError("Unknown Stage-B phase")
    prior = _prior_bf16_b(stage_a, sha(FREEZE))
    return verify_fp32_freeze(stage_a, prior), stage_a, prior


def phase_freeze_path(phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError("Unknown Stage-B phase")
    return FREEZE if phase == "bf16_direct_lm" else FP32_FREEZE


def _dirs(folder: Path) -> list[Path]:
    if folder.is_symlink():
        raise ValueError("Stage-B phase tree is symlinked")
    if not folder.exists():
        return []
    if not folder.is_dir():
        raise ValueError("Stage-B phase tree is not a regular directory")
    children = sorted(folder.iterdir())
    if any(child.is_symlink() or not child.is_dir() for child in children):
        raise ValueError("Stage-B phase tree has unexpected file or symlink")
    return children


def _bytes(folder: Path) -> int:
    total = 0
    if folder.is_symlink():
        raise ValueError("Stage-B output root is symlinked")
    if folder.exists():
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Stage-B output contains symlink: {path.name}")
            if path.is_file():
                total += path.stat().st_size
    return total


def _prior_bf16_b(stage_a: dict, freeze_sha: str) -> dict:
    """FP32 admission requires a later committed, independently reviewed BF16 B."""
    committed_same(REPO, BF16_B_MANIFEST)
    committed_same(REPO, BF16_B_REVIEW)
    if BF16_B_REVIEW_GO not in BF16_B_REVIEW.read_text():
        raise ValueError("Stage-B BF16 result review lacks GO")
    manifest_add = first_add(REPO, BF16_B_MANIFEST)
    review_add = first_add(REPO, BF16_B_REVIEW)
    ancestor(REPO, manifest_add, review_add)
    if manifest_add == review_add:
        raise ValueError("BF16 B manifest and review share a commit")
    prior = driver.verify_bf16_stage_b_manifest(stage_a)
    attempt = prior["attempt"]
    attempt_id = attempt.name
    gate_attempt = RUNS / "bf16_direct_lm" / attempt_id
    reservation = RUNS / "bf16_direct_lm_reservation.json"
    for path in (reservation, gate_attempt / "launch_start.json",
                 gate_attempt / "worker_claim.json",
                 gate_attempt / "launch_finish.json", attempt / "audit.json",
                 attempt / "result.json"):
        committed_same(REPO, path)
    first_add(REPO, gate_attempt / "worker_claim.json")
    if (_dirs(RUNS / "bf16_direct_lm") != [gate_attempt] or
            _dirs(driver.RUNS / "bf16_direct_lm") != [attempt]):
        raise ValueError("Stage-B BF16 attempt tree differs")
    start, finish = (read_json(gate_attempt / name) for name in
                     ("launch_start.json", "launch_finish.json"))
    claim = read_json(gate_attempt / "worker_claim.json")
    reserved = read_json(reservation)
    value = read_json(BF16_B_MANIFEST)
    if (reserved != {"status": "phase_reserved_no_retry", "phase": "bf16_direct_lm",
                     "attempt_id": attempt_id, "stage_b_freeze_sha256": freeze_sha} or
            {k: claim.get(k) for k in ("status", "phase", "attempt_id", "stage_b_freeze_sha256")} !=
            {"status": "worker_claimed_no_retry", "phase": "bf16_direct_lm",
             "attempt_id": attempt_id, "stage_b_freeze_sha256": freeze_sha} or
            utc(start.get("wall_utc")) >= utc(claim.get("wall_utc")) or
            utc(claim.get("wall_utc")) >= utc(finish.get("wall_utc")) or
            value.get("stage_b_freeze_sha256") != freeze_sha or
            value.get("launch_start_sha256") != sha(gate_attempt / "launch_start.json") or
            value.get("launch_finish_sha256") != sha(gate_attempt / "launch_finish.json") or
            value.get("reservation_sha256") != sha(reservation) or
            start.get("status") != "launch_started" or
            start.get("attempt_id") != attempt_id or
            start.get("stage_a_manifest_sha256") !=
            {arm: stage_a["arms"][arm]["manifest_sha256"] for arm in ("bf16", "fp32")} or
            finish.get("status") != LAUNCH_SUCCESS["bf16_direct_lm"] or
            finish.get("phase") != "bf16_direct_lm" or
            finish.get("attempt_id") != attempt_id or
            finish.get("stage_b_freeze_sha256") != freeze_sha or
            finish.get("runner_returncode") != 0 or finish.get("auditor_returncode") != 0 or
            finish.get("post_run_source_verified") is not True or
            finish.get("runner_result_sha256") != value["result_sha256"] or
            finish.get("audit_sha256") != value["audit_sha256"] or
            finish.get("model_work_elapsed_seconds") != value["model_work_elapsed_seconds"] or
            start.get("stage_b_freeze_sha256") != freeze_sha or
            start.get("phase") != "bf16_direct_lm" or
            utc(start.get("wall_utc")) >= utc(finish.get("wall_utc"))):
        raise ValueError("BF16 Stage-B launch chronology/receipt mismatch")
    if (sha(BF16_B_MANIFEST) != prior["manifest_sha256"] or
            stage_a["stage_a_model_work_seconds"] + prior["elapsed_seconds"] +
            driver.CAP_SECONDS["fp32_replay"] >= driver.MAX_TOTAL_SECONDS):
        raise ValueError("No full FP32 Stage-B time reservation")
    return prior


def admission(phase: str, attempt_id: str) -> tuple[dict, dict, dict | None]:
    if phase not in PHASES or not re.fullmatch(r"[0-9]{8}T[0-9]{12}Z", attempt_id):
        raise ValueError("Stage-B phase or attempt ID invalid")
    freeze, stage_a, prior = verify_phase_freeze(phase)
    freeze_sha = sha(FREEZE)
    if _dirs(RUNS / "fp32_replay") or _dirs(driver.RUNS / "fp32_replay"):
        raise ValueError("Stage-B FP32 attempt already exists")
    if phase == "bf16_direct_lm":
        if (BF16_B_MANIFEST.exists() or BF16_B_REVIEW.exists() or
                (RUNS / "bf16_direct_lm_reservation.json").exists() or
                _dirs(RUNS / phase) or _dirs(driver.RUNS / phase)):
            raise ValueError("Stage-B BF16 attempt already reserved; no retry")
        remaining_caps = sum(driver.CAP_SECONDS.values())
    else:
        if (RUNS / "fp32_replay_reservation.json").exists():
            raise ValueError("Stage-B FP32 attempt already reserved; no retry")
        remaining_caps = driver.CAP_SECONDS[phase]
    prior_time = stage_a["stage_a_model_work_seconds"] + (prior["elapsed_seconds"] if prior else 0)
    if prior_time + remaining_caps >= driver.MAX_TOTAL_SECONDS:
        raise ValueError("No full Stage-B model-work reservation")
    a_bytes = sum(_bytes(stage_a["arms"][arm]["attempt"]) for arm in ("bf16", "fp32"))
    b_bytes = (_bytes(prior["attempt"]) if prior else 0) + _bytes(RUNS)
    caps = (2 if phase == "bf16_direct_lm" else 1) * driver.MAX_B_BYTES
    if a_bytes + b_bytes + caps >= MAX_TOTAL_BYTES:
        raise ValueError("No full Stage-B output-byte reservation")
    if (RUNS / phase / attempt_id).exists() or (driver.RUNS / phase / attempt_id).exists():
        raise ValueError("Stage-B attempt already exists")
    return freeze, stage_a, prior


def _checked_start(phase: str, attempt_id: str, freeze_sha: str,
                   freeze: dict, stage_a: dict, prior: dict | None) -> dict:
    """Require the exact parent receipt and a permanent one-shot reservation."""
    gate_attempt = RUNS / phase / attempt_id
    reservation = RUNS / f"{phase}_reservation.json"
    start = read_json(gate_attempt / "launch_start.json")
    reserved = read_json(reservation)
    expected_prior = stage_a["stage_a_model_work_seconds"] + (prior["elapsed_seconds"] if prior else 0)
    command = [sys.executable, str(WORKER), "--phase", phase,
               "--attempt-id", attempt_id, "--freeze-sha256", freeze_sha]
    expected_keys = {"status", "phase", "attempt_id", "wall_utc", "stage_b_freeze_sha256",
                     "worker_output", "worker_command", "driver_sha256", "auditor_sha256",
                     "prior_model_work_seconds", "process_cap_seconds",
                     "process_timeout_seconds", "process_grace_seconds",
                     "worker_output_cap_bytes",
                     "max_mps_driver_bytes", "max_capture_bytes_per_stream",
                     "total_output_cap_bytes", "total_model_work_cap_seconds",
                     "stage_a_manifest_sha256"}
    if (reserved != {"status": "phase_reserved_no_retry", "phase": phase,
                     "attempt_id": attempt_id, "stage_b_freeze_sha256": freeze_sha} or
            set(start) != expected_keys or
            start.get("status") != "launch_started" or start.get("phase") != phase or
            start.get("attempt_id") != attempt_id or
            start.get("stage_b_freeze_sha256") != freeze_sha or
            start.get("worker_command") != command or
            start.get("driver_sha256") != DRIVER_SHA or
            start.get("auditor_sha256") != AUDITOR_SHA or
            start.get("prior_model_work_seconds") != expected_prior or
            start.get("process_cap_seconds") != driver.CAP_SECONDS[phase] or
            start.get("process_timeout_seconds") != driver.CAP_SECONDS[phase] or
            start.get("process_grace_seconds") != PROCESS_GRACE_SECONDS or
            start.get("worker_output_cap_bytes") != driver.MAX_B_BYTES or
            start.get("max_mps_driver_bytes") != driver.MAX_MPS_DRIVER_BYTES or
            start.get("max_capture_bytes_per_stream") != MAX_CAPTURE_BYTES or
            start.get("total_output_cap_bytes") != MAX_TOTAL_BYTES or
            start.get("total_model_work_cap_seconds") != driver.MAX_TOTAL_SECONDS or
            start.get("stage_a_manifest_sha256") !=
            {arm: stage_a["arms"][arm]["manifest_sha256"] for arm in ("bf16", "fp32")} or
            start.get("worker_output") != str(driver.RUNS / phase / attempt_id) or
            utc(start.get("wall_utc")) <= utc(freeze["reviewed_utc"])):
        raise ValueError("Worker reservation/start receipt invalid")
    if _dirs(RUNS / phase) != [gate_attempt]:
        raise ValueError("Worker launch attempt tree differs")
    return start


def verify_worker_start(phase: str, attempt_id: str, freeze_sha: str) -> tuple[dict, dict]:
    """Called in the child immediately before the only model construction path."""
    freeze, stage_a, prior = verify_phase_freeze(phase)
    if sha(phase_freeze_path(phase)) != freeze_sha:
        raise ValueError("Worker freeze hash differs")
    start = _checked_start(phase, attempt_id, freeze_sha, freeze, stage_a, prior)
    if (driver.RUNS / phase / attempt_id).exists():
        raise ValueError("Worker output already exists; no retry")
    return start, stage_a


def verify_audit_inputs(phase: str, attempt_id: str, freeze_sha: str,
                        prior_model_work_seconds: float) -> None:
    """CPU auditor admission after worker exit, before parent finish receipt."""
    freeze, stage_a, prior = verify_phase_freeze(phase)
    if sha(phase_freeze_path(phase)) != freeze_sha:
        raise ValueError("Audit adapter freeze differs")
    _checked_start(phase, attempt_id, freeze_sha, freeze, stage_a, prior)
    expected_prior = stage_a["stage_a_model_work_seconds"] + (prior["elapsed_seconds"] if prior else 0)
    if (prior_model_work_seconds != expected_prior or
            not (driver.RUNS / phase / attempt_id / "result.json").is_file() or
            (driver.RUNS / phase / attempt_id / "error.json").exists()):
        raise ValueError("Audit adapter parent/worker receipts differ")


def exclusive_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bounded(command: list[str], *, capture_dir: Path, worker_output: Path | None,
             timeout: float, output_name: str, error_name: str,
             termination_grace: float = 5.0) -> tuple[int | None, float, str | None]:
    """Bound wall time, stdout/stderr, and both output trees; kill process group."""
    began = time.monotonic()
    reason = None
    code = None
    with (capture_dir / output_name).open("xb") as stdout, (capture_dir / error_name).open("xb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                                start_new_session=True)
        while True:
            code = proc.poll()
            elapsed = time.monotonic() - began
            if elapsed > timeout:
                reason = "child_process_timeout"
            elif max(stdout.tell(), (capture_dir / output_name).stat().st_size) > MAX_CAPTURE_BYTES or \
                    max(stderr.tell(), (capture_dir / error_name).stat().st_size) > MAX_CAPTURE_BYTES:
                reason = "capture_byte_cap"
            elif _bytes(capture_dir) > PARENT_OUTPUT_CAP_BYTES:
                reason = "launcher_receipt_byte_cap"
            elif worker_output is not None and (_bytes(worker_output) + _bytes(capture_dir) >
                                                driver.MAX_B_BYTES - FINAL_RECEIPT_RESERVE_BYTES):
                reason = "combined_phase_output_byte_cap"
            if reason and code is None:
                # A time-cap hit must stop model work immediately; grace is
                # for reaping and parent receipts after execution stops.
                os.killpg(proc.pid, signal.SIGKILL if reason == "child_process_timeout"
                          else signal.SIGTERM)
                try:
                    code = proc.wait(timeout=termination_grace)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait(timeout=5)
                break
            if reason or code is not None:
                break
            time.sleep(.1)
    return code, time.monotonic() - began, reason


def launch(phase: str, attempt_id: str) -> dict:
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Stage-B launch requires pinned second_architecture environment")
    freeze, stage_a, prior = admission(phase, attempt_id)
    frozen_sha = sha(phase_freeze_path(phase))
    prior_seconds = stage_a["stage_a_model_work_seconds"] + (prior["elapsed_seconds"] if prior else 0)
    gate_attempt = RUNS / phase / attempt_id
    worker_output = driver.RUNS / phase / attempt_id
    (RUNS / phase).mkdir(parents=True, exist_ok=True)
    (driver.RUNS / phase).mkdir(parents=True, exist_ok=True)
    exclusive_json(RUNS / f"{phase}_reservation.json", {
        "status": "phase_reserved_no_retry", "phase": phase,
        "attempt_id": attempt_id, "stage_b_freeze_sha256": frozen_sha})
    gate_attempt.mkdir(exist_ok=False)
    worker_cmd = [sys.executable, str(WORKER), "--phase", phase,
                  "--attempt-id", attempt_id, "--freeze-sha256", frozen_sha]
    exclusive_json(gate_attempt / "launch_start.json", {
        "status": "launch_started", "phase": phase, "attempt_id": attempt_id,
        "wall_utc": datetime.now(timezone.utc).isoformat(),
        "stage_b_freeze_sha256": frozen_sha,
        "worker_output": str(worker_output), "worker_command": worker_cmd,
        "driver_sha256": sha(Path(driver.__file__)), "auditor_sha256": AUDITOR_SHA,
        "prior_model_work_seconds": prior_seconds,
        "process_cap_seconds": driver.CAP_SECONDS[phase],
        "process_timeout_seconds": driver.CAP_SECONDS[phase],
        "process_grace_seconds": PROCESS_GRACE_SECONDS,
        "worker_output_cap_bytes": driver.MAX_B_BYTES,
        "max_mps_driver_bytes": driver.MAX_MPS_DRIVER_BYTES,
        "max_capture_bytes_per_stream": MAX_CAPTURE_BYTES,
        "total_output_cap_bytes": MAX_TOTAL_BYTES,
        "total_model_work_cap_seconds": driver.MAX_TOTAL_SECONDS,
        "stage_a_manifest_sha256": {arm: sha(driver.STAGE_A_MANIFESTS[arm]) for arm in ("bf16", "fp32")},
    })
    finish = {"status": "launch_failed", "phase": phase, "attempt_id": attempt_id,
              "stage_b_freeze_sha256": frozen_sha, "runner_returncode": None,
              "auditor_returncode": None, "runner_result_sha256": None,
              "runner_error_sha256": None, "audit_sha256": None,
              "post_run_source_verified": False, "model_work_elapsed_seconds": 0.0}
    began = time.monotonic()
    try:
        code, elapsed, reason = _bounded(worker_cmd, capture_dir=gate_attempt,
                                         worker_output=worker_output,
                                         timeout=driver.CAP_SECONDS[phase],
                                         output_name="worker_stdout.txt", error_name="worker_stderr.txt",
                                         termination_grace=PROCESS_GRACE_SECONDS)
        finish["runner_returncode"] = code
        finish["model_work_elapsed_seconds"] = elapsed
        if (reason or code != 0 or elapsed >= driver.CAP_SECONDS[phase] or
                prior_seconds + elapsed >= driver.MAX_TOTAL_SECONDS):
            raise ValueError(reason or "Stage-B worker failed or exceeded model-work cap")
        claim_path = gate_attempt / "worker_claim.json"
        claim = read_json(claim_path)
        if ({k: claim.get(k) for k in ("status", "phase", "attempt_id", "stage_b_freeze_sha256")} !=
                {"status": "worker_claimed_no_retry", "phase": phase,
                 "attempt_id": attempt_id, "stage_b_freeze_sha256": frozen_sha}):
            raise ValueError("Stage-B worker one-shot claim absent or invalid")
        result_path, error_path = worker_output / "result.json", worker_output / "error.json"
        finish["runner_result_sha256"] = sha(result_path) if result_path.is_file() else None
        finish["runner_error_sha256"] = sha(error_path) if error_path.is_file() else None
        if not result_path.is_file() or error_path.exists():
            raise ValueError("Stage-B worker result absent or error receipt present")
        if sha(phase_freeze_path(phase)) != frozen_sha or verify_phase_freeze(phase)[0] != freeze:
            raise ValueError("Stage-B source or freeze changed after worker")
        finish["post_run_source_verified"] = True
        audit_cmd = [sys.executable, str(AUDIT_ADAPTER), "--phase", phase,
                     "--attempt-id", attempt_id, "--freeze-sha256", frozen_sha,
                     "--prior-model-work-seconds", repr(prior_seconds)]
        code, _, reason = _bounded(audit_cmd, capture_dir=gate_attempt,
                                    worker_output=worker_output, timeout=AUDIT_SECONDS,
                                    output_name="audit_stdout.txt", error_name="audit_stderr.txt")
        finish["auditor_returncode"] = code
        if code != 0 or reason:
            raise ValueError(reason or "Stage-B CPU auditor rejected saved result")
        audit = read_json(gate_attempt / "audit_stdout.txt")
        if (audit.get("phase") != phase or audit.get("auditor_sha256") != AUDITOR_SHA or
                audit.get("result_sha256") != finish["runner_result_sha256"] or
                audit.get("status") != ("audited_completed_old_fit_handoff_stage_b_bf16_direct_lm_only"
                                         if phase == "bf16_direct_lm" else
                                         "audited_completed_old_fit_handoff_stage_b_fp32_replay_only")):
            raise ValueError("Stage-B CPU audit receipt invalid")
        (gate_attempt / "audit_stdout.txt").replace(worker_output / "audit.json")
        finish["audit_sha256"] = sha(worker_output / "audit.json")
        if sha(phase_freeze_path(phase)) != frozen_sha or verify_phase_freeze(phase)[0] != freeze:
            raise ValueError("Stage-B source or freeze changed after audit")
        if _bytes(worker_output) + _bytes(gate_attempt) >= driver.MAX_B_BYTES - FINAL_RECEIPT_RESERVE_BYTES:
            raise ValueError("Stage-B combined phase output reached byte cap")
        finish["status"] = LAUNCH_SUCCESS[phase]
    except Exception as exc:
        finish["failure_reason"] = str(exc)
        finish["exception_type"] = type(exc).__name__
    finally:
        finish["wall_utc"] = datetime.now(timezone.utc).isoformat()
        finish["elapsed_seconds"] = time.monotonic() - began
        finish["worker_output_bytes"] = _bytes(worker_output)
        finish["launcher_output_bytes_before_finish"] = _bytes(gate_attempt)
        exclusive_json(gate_attempt / "launch_finish.json", finish)
        if _bytes(worker_output) + _bytes(gate_attempt) >= driver.MAX_B_BYTES:
            raise ValueError("Stage-B phase output exceeds cap including finish receipt")
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--attempt-id")
    args = parser.parse_args()
    if args.check_freeze:
        if args.attempt_id is not None:
            parser.error("--check-freeze takes no attempt ID")
        phase = args.phase or "bf16_direct_lm"
        frozen, _, _ = verify_phase_freeze(phase)
        print(json.dumps({"status": "verified_stage_b_freeze_only", "phase": phase,
                          "freeze_sha256": sha(phase_freeze_path(phase))}, indent=2))
        return
    if args.phase is None:
        parser.error("--run requires --phase")
    attempt_id = args.attempt_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = launch(args.phase, attempt_id)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != LAUNCH_SUCCESS[args.phase]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
