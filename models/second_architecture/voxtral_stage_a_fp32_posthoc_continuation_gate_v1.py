"""One-arm Stage-A FP32 continuation after an external posthoc BF16 audit.

No model module is imported. The committed external continuation freeze and
independent reviews are mandatory; this source scaffold alone cannot launch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import voxtral_old_fit_handoff_stage_a_launch_gate_v1 as old


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL_ATTEMPT = old.RUNS / "bf16/20261118T131755298331Z"
FAILED_MANIFEST = HERE / "VOXTRAL_STAGE_A_BF16_FAILED_ATTEMPT_MANIFEST_V1.json"
POSTHOC_AUDIT = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_V2.json"
POSTHOC_RESULT_REVIEW = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RESULT_INDEPENDENT_REVIEW_V1.md"
POSTHOC_RECEIPT = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RECEIPT_V1.json"
POSTHOC_REVIEW = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RECEIPT_INDEPENDENT_REVIEW_V1.md"
FREEZE = HERE / "VOXTRAL_STAGE_A_FP32_POSTHOC_CONTINUATION_FREEZE_V1.json"
RUNS = HERE / "runs/voxtral_stage_a_fp32_posthoc_continuation_v1"
REVIEW = HERE / "VOXTRAL_STAGE_A_FP32_POSTHOC_CONTINUATION_INDEPENDENT_REVIEW_V1.md"
PROTOCOL = HERE / "VOXTRAL_STAGE_A_FP32_POSTHOC_CONTINUATION_PROTOCOL_V1.md"
AUDITOR = HERE / "voxtral_old_fit_handoff_stage_a_audit_v2.py"
AUDITOR_REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_AUDIT_V2_INDEPENDENT_REVIEW_V1.md"
AMENDMENT = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_AUDIT_ONLY_AMENDMENT_V2.md"
V2_SOURCE_REVIEW_COMMIT = "a8b6c248f285485702ef46dca08f91012e51dbbf"
POSTHOC_RESULT_REVIEW_COMMIT = "074fb1b941732bbc1235f24461f50f4295cfd6e1"
POSTHOC_AUDIT_SHA = "934a4517a0de2880051cc3be4f135cc475dc1098e742c7cc62a7e5990ac5f7f9"
POSTHOC_RESULT_REVIEW_SHA = "4de300de75c6402e1e9b6707eefca5eea42722a1f96a2747f8d7c7e14296f0ae"
ORIGINAL_FREEZE_SHA = "6397da3c8d5b9be7ce67e04464530fbab90ac595b5760e2b53dfdccfed62fc4e"
FAILED_MANIFEST_SHA = "ee04b2f7a3e319c89d4426c78b03ad77107afb0dac927a80ca005dd33529ee02"
ORIGINAL_FINISH_SHA = "15ed053c8ca4e619735cce3319b5d2326ff4b19216475754acc2cbf4e3bcdf9a"
ORIGINAL_RESULT_SHA = "f7b3906aff2b1bb9c7b8fb818402982a9020d8507fcba9cf51cc58821dddd325"
ORIGINAL_LAYOUT_SHA = "5eea970f04c218a8e444373935467cff4f592a62239a7813884a72a09912f01e"
ORIGINAL_DETAILS_SHA = "df9fe20d1fd961a0b3502e073f933eed658f7a15de3214087552987ee8e8e849"
ORIGINAL_PROBES_SHA = "dad65cd52beff26ed4f6dfcc5b541a9a0d504a207f23c2783f8ce71113f6b53e"
V1_AUDITOR_SHA = "4f800ed604295ae0e4fbd53c893fccd14b8968badb43b835e6ec975e49b0cf3d"
V2_AUDITOR_SHA = "8d5f8b23bd32add8464bd976d42cd0225fbc5ee86510adc441b714de5a8fd275"
EXPECTED_BF16_SECONDS = 63.245136791840196
RECEIPT_STATUS = "posthoc_saved_tensor_audit_passed_original_launch_failed"
FREEZE_STATUS = "frozen_stage_a_fp32_posthoc_continuation_v1"
SCOPE = old.SCOPE

BOUND_FILES = {
    "original_freeze_sha256": old.FREEZE,
    "failed_manifest_sha256": FAILED_MANIFEST,
    "posthoc_audit_sha256": POSTHOC_AUDIT,
    "posthoc_result_review_sha256": POSTHOC_RESULT_REVIEW,
    "posthoc_receipt_sha256": POSTHOC_RECEIPT,
    "posthoc_review_sha256": POSTHOC_REVIEW,
    "v2_auditor_sha256": AUDITOR,
    "v2_auditor_review_sha256": AUDITOR_REVIEW,
    "v2_amendment_sha256": AMENDMENT,
    "original_runner_sha256": old.FILES["runner_sha256"],
    "original_native_source_sha256": old.FILES["native_source_sha256"],
    "original_protocol_sha256": old.FILES["protocol_sha256"],
    "continuation_protocol_sha256": PROTOCOL,
    "continuation_launcher_sha256": Path(__file__).resolve(),
    "continuation_tests_sha256": HERE / "test_voxtral_stage_a_fp32_posthoc_continuation_gate_v1.py",
    "continuation_review_sha256": REVIEW,
}
FIXED_PINS = {
    "original_freeze_sha256": ORIGINAL_FREEZE_SHA,
    "failed_manifest_sha256": FAILED_MANIFEST_SHA,
    "posthoc_audit_sha256": POSTHOC_AUDIT_SHA,
    "posthoc_result_review_sha256": POSTHOC_RESULT_REVIEW_SHA,
    "v2_auditor_sha256": V2_AUDITOR_SHA,
    "v2_auditor_review_sha256": "907f752df2ff882e5dc50c32051f5ce50c537fda49f8c227e3c3e740166f4a03",
    "v2_amendment_sha256": "c67a3605d18e98eb043c0702c10ba123963bb1f6e2b40e7abb9ac6553fed2c14",
    "original_runner_sha256": old.LIVE_SOURCE_PINS["runner_sha256"],
    "original_native_source_sha256": old.LIVE_SOURCE_PINS["native_source_sha256"],
    "original_protocol_sha256": old.LIVE_SOURCE_PINS["protocol_sha256"],
}


def _hex(value: object, size: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{size}}}", value) is not None


def require_regular_path(repo: Path, path: Path) -> None:
    """Reject symlinks before the older Git helper resolves a bound path."""
    try:
        relative = path.absolute().relative_to(repo.absolute())
    except ValueError as exc:
        raise ValueError("Bound path outside repository") from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Bound path is not a regular repository path")
    current = repo.absolute()
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Bound path contains a symlink: {path.name}")
    if not path.is_file():
        raise ValueError(f"Bound file absent: {path.name}")


def committed_regular(repo: Path, ref: str, path: Path) -> None:
    require_regular_path(repo, path)
    old.committed_same(repo, ref, path)


def require_first_add_bytes(repo: Path, path: Path) -> str:
    """Reject a receipt or review changed in any later committed revision."""
    require_regular_path(repo, path)
    first_add = old.first_add_commit(repo, path)
    if path.read_bytes() != old.git_blob(repo, first_add, path):
        raise ValueError(f"Bound receipt/review changed after first-add: {path.name}")
    return first_add


def verify_preserved_bf16() -> dict:
    """Verify the original failure and every manifest byte; never reinterpret it."""
    for path in (old.FREEZE, *old.FILES.values(), FAILED_MANIFEST):
        require_regular_path(REPO, path)
    original = old.verify_freeze()
    if original["receipt_sha256"] != ORIGINAL_FREEZE_SHA:
        raise ValueError("Original freeze differs")
    for path, digest in ((FAILED_MANIFEST, FAILED_MANIFEST_SHA),
                         (ORIGINAL_ATTEMPT / "launch_finish.json", ORIGINAL_FINISH_SHA),
                         (ORIGINAL_ATTEMPT / "runner/result.json", ORIGINAL_RESULT_SHA)):
        require_regular_path(REPO, path)
        if old.sha(path) != digest:
            raise ValueError(f"Original evidence changed: {path.name}")
    committed_regular(REPO, "HEAD", FAILED_MANIFEST)
    manifest = old.read_json(FAILED_MANIFEST)
    expected_root = "runs/voxtral_old_fit_handoff_stage_a_launch_v1/bf16/20261118T131755298331Z/"
    if (manifest.get("schema") != "voxtral_stage_a_bf16_failed_attempt_manifest_v1" or
            manifest.get("status") != "frozen_auditor_rejected_no_scientific_interpretation" or
            manifest.get("arm") != "bf16" or manifest.get("attempt_id") != ORIGINAL_ATTEMPT.name or
            manifest.get("freeze_receipt_sha256") != ORIGINAL_FREEZE_SHA or
            manifest.get("local_large_sidecars_required") is not True or
            not isinstance(manifest.get("files"), list) or len(manifest["files"]) != 12):
        raise ValueError("Original failed-attempt manifest schema/status invalid")
    seen = set()
    for item in manifest["files"]:
        if set(item) != {"path", "bytes", "sha256", "git_committed"}:
            raise ValueError("Original file record schema invalid")
        relative = item["path"]
        if (not isinstance(relative, str) or not relative.startswith(expected_root) or
                ".." in Path(relative).parts or relative in seen or not _hex(item["sha256"], 64) or
                isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or
                item["bytes"] < 0 or not isinstance(item["git_committed"], bool)):
            raise ValueError("Original file record invalid")
        seen.add(relative)
        path = HERE / relative
        require_regular_path(REPO, path)
        if path.stat().st_size != item["bytes"] or old.sha(path) != item["sha256"]:
            raise ValueError(f"Original raw file changed: {relative}")
        if item["git_committed"]:
            committed_regular(REPO, "HEAD", path)
        elif path.suffix != ".pt" or "/runner/blobs/" not in relative:
            raise ValueError("Unexpected uncommitted raw file")
    if sum(not row["git_committed"] for row in manifest["files"]) != 2:
        raise ValueError("Expected exactly two local tensor sidecars")
    if old.directory_bytes(ORIGINAL_ATTEMPT) > old.MAX_ARM_BYTES:
        raise ValueError("Original attempt exceeds byte cap")
    start = old.read_json(ORIGINAL_ATTEMPT / "launch_start.json")
    finish = old.read_json(ORIGINAL_ATTEMPT / "launch_finish.json")
    result = old.read_json(ORIGINAL_ATTEMPT / "runner/result.json")
    if (start.get("arm") != "bf16" or start.get("freeze_receipt_sha256") != ORIGINAL_FREEZE_SHA or
            start.get("runner_sha256") != original["receipt"]["runner_sha256"] or
            start.get("auditor_sha256") != V1_AUDITOR_SHA or
            finish.get("status") != "launch_failed" or finish.get("arm") != "bf16" or
            finish.get("runner_returncode") != 0 or finish.get("auditor_returncode") != 1 or
            finish.get("audit_sha256") is not None or
            finish.get("runner_result_sha256") != ORIGINAL_RESULT_SHA or
            finish.get("post_run_source_verified") is not True or
            finish.get("model_work_elapsed_seconds") != EXPECTED_BF16_SECONDS or
            result.get("status") != old.RESULT_STATUS or result.get("arm") != "bf16" or
            result.get("script_sha256") != original["receipt"]["runner_sha256"] or
            result.get("native_source_sha256") != original["receipt"]["native_source_sha256"] or
            result.get("protocol_sha256") != original["receipt"]["protocol_sha256"] or
            result.get("model_layout_sha256") != ORIGINAL_LAYOUT_SHA or
            result.get("details_sha256") != ORIGINAL_DETAILS_SHA or
            result.get("probes_sha256") != ORIGINAL_PROBES_SHA):
        raise ValueError("Original launch must remain a failed v1 audit")
    if old._attempt_dirs(old.RUNS, "bf16") != [ORIGINAL_ATTEMPT] or old._attempt_dirs(old.RUNS, "fp32"):
        raise ValueError("Original launch tree has unexpected retry or FP32 attempt")
    return {"manifest": manifest, "result": result, "finish": finish, "original_freeze": original}


def verify_posthoc_receipt(preserved: dict) -> dict:
    """Bind externally committed v2 saved-tensor audit to the preserved bytes."""
    for path in (AUDITOR, AUDITOR_REVIEW, AMENDMENT, POSTHOC_AUDIT, POSTHOC_RESULT_REVIEW,
                 POSTHOC_RECEIPT, POSTHOC_REVIEW):
        committed_regular(REPO, "HEAD", path)
    if old.sha(AUDITOR) != V2_AUDITOR_SHA:
        raise ValueError("Reviewed v2 auditor differs")
    if old.sha_bytes(old.git_blob(REPO, V2_SOURCE_REVIEW_COMMIT, AUDITOR)) != V2_AUDITOR_SHA:
        raise ValueError("v2 source review did not contain auditor")
    old.ancestor(REPO, V2_SOURCE_REVIEW_COMMIT, "HEAD")
    if old.sha(POSTHOC_AUDIT) != POSTHOC_AUDIT_SHA or old.sha(POSTHOC_RESULT_REVIEW) != POSTHOC_RESULT_REVIEW_SHA:
        raise ValueError("Independently reviewed posthoc output differs")
    for path, digest in ((POSTHOC_AUDIT, POSTHOC_AUDIT_SHA),
                         (POSTHOC_RESULT_REVIEW, POSTHOC_RESULT_REVIEW_SHA)):
        if old.sha_bytes(old.git_blob(REPO, POSTHOC_RESULT_REVIEW_COMMIT, path)) != digest:
            raise ValueError("Posthoc result review commit did not contain reviewed evidence")
    old.ancestor(REPO, V2_SOURCE_REVIEW_COMMIT, POSTHOC_RESULT_REVIEW_COMMIT)
    audit_add = require_first_add_bytes(REPO, POSTHOC_AUDIT)
    if audit_add == POSTHOC_RESULT_REVIEW_COMMIT:
        raise ValueError("Posthoc audit output was added with its review")
    old.ancestor(REPO, audit_add, POSTHOC_RESULT_REVIEW_COMMIT)
    audit = old.read_json(POSTHOC_AUDIT)
    result = preserved["result"]
    if (audit.get("status") != old.AUDIT_STATUS or audit.get("arm") != "bf16" or
            audit.get("scope") != "saved_tensor_and_receipt_arithmetic_only_no_model_execution_attestation" or
            audit.get("result_sha256") != ORIGINAL_RESULT_SHA or
            audit.get("protocol_sha256") != old.LIVE_SOURCE_PINS["protocol_sha256"] or
            audit.get("runner_sha256") != old.LIVE_SOURCE_PINS["runner_sha256"] or
            audit.get("native_source_sha256") != old.LIVE_SOURCE_PINS["native_source_sha256"] or
            audit.get("details_sha256") != result.get("details_sha256") or
            audit.get("probes_sha256") != result.get("probes_sha256") or
            audit.get("auditor_sha256") != V2_AUDITOR_SHA or
            audit.get("audited_native_pairs") != 36 or audit.get("audited_patch_forwards") != 72 or
            audit.get("audited_vjps") != 44 or audit.get("audited_lm0_control_cells") != 24 or
            audit.get("observed_progress", {}).get("progress_lines") != 214):
        raise ValueError("Posthoc audit output does not bind exact BF16 result")
    receipt = old.read_json(POSTHOC_RECEIPT)
    expected = {
        "schema": "voxtral_stage_a_bf16_posthoc_audit_receipt_v1",
        "status": RECEIPT_STATUS, "arm": "bf16",
        "attempt_id": ORIGINAL_ATTEMPT.name,
        "original_launch_status": "launch_failed",
        "failed_attempt_manifest_sha256": FAILED_MANIFEST_SHA,
        "original_freeze_sha256": ORIGINAL_FREEZE_SHA,
        "original_launch_finish_sha256": ORIGINAL_FINISH_SHA,
        "original_result_sha256": ORIGINAL_RESULT_SHA,
        "original_layout_sha256": ORIGINAL_LAYOUT_SHA,
        "original_details_sha256": ORIGINAL_DETAILS_SHA,
        "original_probes_sha256": ORIGINAL_PROBES_SHA,
        "original_v1_auditor_sha256": V1_AUDITOR_SHA,
        "v2_auditor_sha256": V2_AUDITOR_SHA,
        "v2_auditor_source_review_commit": V2_SOURCE_REVIEW_COMMIT,
        "posthoc_audit_sha256": old.sha(POSTHOC_AUDIT),
        "model_work_elapsed_seconds": EXPECTED_BF16_SECONDS,
    }
    if receipt != expected:
        raise ValueError("Posthoc receipt schema, hash, or original failure differs")
    if "Decision: GO for receipt provenance; FP32 HOLD pending launcher review and freeze." not in POSTHOC_REVIEW.read_text():
        raise ValueError("Posthoc receipt review lacks required GO decision")
    receipt_add = require_first_add_bytes(REPO, POSTHOC_RECEIPT)
    review_add = require_first_add_bytes(REPO, POSTHOC_REVIEW)
    old.ancestor(REPO, POSTHOC_RESULT_REVIEW_COMMIT, receipt_add)
    if receipt_add == POSTHOC_RESULT_REVIEW_COMMIT or receipt_add == review_add:
        raise ValueError("Posthoc receipt/review chronology invalid")
    old.ancestor(REPO, receipt_add, review_add)
    return {"audit": audit, "receipt": receipt, "receipt_sha256": old.sha(POSTHOC_RECEIPT),
            "review_commit": review_add}


def verify_continuation_freeze(preserved: dict, posthoc: dict) -> dict:
    """Require a separately reviewed, later committed freeze before model load."""
    committed_regular(REPO, "HEAD", FREEZE)
    freeze = old.read_json(FREEZE)
    expected_keys = {"status", "permitted_arm", "audit_scope", "receipt_review_commit",
                     "launcher_review_commit", "reviewed_utc", *BOUND_FILES}
    if set(freeze) != expected_keys or freeze["status"] != FREEZE_STATUS or freeze["permitted_arm"] != "fp32" or freeze["audit_scope"] != SCOPE:
        raise ValueError("Continuation freeze schema or scope invalid")
    old.utc(freeze["reviewed_utc"])
    for key in ("receipt_review_commit", "launcher_review_commit"):
        if not _hex(freeze[key], 40):
            raise ValueError(f"Malformed {key}")
    if freeze["receipt_review_commit"] != posthoc["review_commit"]:
        raise ValueError("Posthoc receipt review commit differs")
    if "Decision: GO for FP32-only launcher source; model run HOLD until external freeze." not in REVIEW.read_text():
        raise ValueError("Continuation launcher review lacks required GO decision")
    for key, path in BOUND_FILES.items():
        if not _hex(freeze[key], 64):
            raise ValueError(f"Malformed {key}")
        committed_regular(REPO, "HEAD", path)
        if freeze[key] != old.sha(path) or (key in FIXED_PINS and freeze[key] != FIXED_PINS[key]):
            raise ValueError(f"Continuation freeze bound hash mismatch: {key}")
    if (freeze["posthoc_receipt_sha256"] != posthoc["receipt_sha256"] or
            freeze["posthoc_audit_sha256"] != old.sha(POSTHOC_AUDIT)):
        raise ValueError("Continuation freeze posthoc evidence differs")
    for key in ("posthoc_receipt_sha256", "posthoc_audit_sha256", "posthoc_result_review_sha256",
                "posthoc_review_sha256"):
        if old.sha_bytes(old.git_blob(REPO, freeze["receipt_review_commit"], BOUND_FILES[key])) != freeze[key]:
            raise ValueError("Receipt review commit did not contain reviewed evidence")
    for key in ("continuation_launcher_sha256", "continuation_tests_sha256",
                "continuation_protocol_sha256", "continuation_review_sha256"):
        if old.sha_bytes(old.git_blob(REPO, freeze["launcher_review_commit"], BOUND_FILES[key])) != freeze[key]:
            raise ValueError("Launcher review commit did not contain reviewed source")
    old.ancestor(REPO, V2_SOURCE_REVIEW_COMMIT, freeze["receipt_review_commit"])
    old.ancestor(REPO, freeze["receipt_review_commit"], freeze["launcher_review_commit"])
    old.ancestor(REPO, freeze["launcher_review_commit"], "HEAD")
    freeze_add = require_first_add_bytes(REPO, FREEZE)
    if freeze_add == freeze["launcher_review_commit"]:
        raise ValueError("Continuation freeze was self-authored in launcher review")
    old.ancestor(REPO, freeze["launcher_review_commit"], freeze_add)
    return {"receipt": freeze, "receipt_sha256": old.sha(FREEZE)}


def admission(output: Path) -> tuple[dict, dict, dict]:
    preserved = verify_preserved_bf16()
    posthoc = verify_posthoc_receipt(preserved)
    frozen = verify_continuation_freeze(preserved, posthoc)
    if output.parent != RUNS / "fp32" or not output.name or output.exists():
        raise ValueError("FP32 output path invalid or already exists")
    if old._attempt_dirs(RUNS, "fp32") or (RUNS / "fp32_reservation.json").exists():
        raise ValueError("FP32 attempt already reserved; no retry")
    if EXPECTED_BF16_SECONDS + old.RUNNER_SECONDS["fp32"] > old.CUMULATIVE_MODEL_SECONDS:
        raise ValueError("Cumulative model-work budget insufficient")
    return preserved, posthoc, frozen


def launch(output: Path) -> dict:
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Launch requires pinned second_architecture virtual environment")
    preserved, posthoc, frozen = admission(output)
    old.RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "fp32").mkdir(parents=True, exist_ok=True)
    old.exclusive_json(RUNS / "fp32_reservation.json", {
        "status": "arm_reserved_no_retry", "arm": "fp32", "attempt": output.name,
        "continuation_freeze_sha256": frozen["receipt_sha256"],
        "original_launch_status": "launch_failed"})
    output.mkdir(exist_ok=False)
    runner_dir = output / "runner"
    command = [sys.executable, str(old.FILES["runner_sha256"]), "--run", "--arm", "fp32",
               "--output-dir", str(runner_dir)]
    started = time.monotonic()
    old.exclusive_json(output / "launch_start.json", {
        "status": "launch_started", "arm": "fp32", "wall_utc": datetime.now(timezone.utc).isoformat(),
        "continuation_freeze_sha256": frozen["receipt_sha256"],
        "posthoc_receipt_sha256": posthoc["receipt_sha256"],
        "original_launch_status": "launch_failed", "runner_sha256": old.sha(old.FILES["runner_sha256"]),
        "auditor_sha256": V2_AUDITOR_SHA, "runner_command": command,
        "runner_cap_seconds": old.RUNNER_SECONDS["fp32"],
        "process_timeout_seconds": old.RUNNER_SECONDS["fp32"] + old.PROCESS_GRACE_SECONDS,
        "runner_mps_driver_cap_bytes": old.MAX_DRIVER_BYTES,
        "arm_output_cap_bytes": old.MAX_ARM_BYTES,
        "max_capture_bytes_per_stream": old.MAX_CAPTURE_BYTES_PER_STREAM,
        "prior_model_work_seconds": EXPECTED_BF16_SECONDS,
        "cumulative_model_work_cap_seconds": old.CUMULATIVE_MODEL_SECONDS, "audit_scope": SCOPE})
    finish = {"status": "launch_failed", "arm": "fp32",
              "continuation_freeze_sha256": frozen["receipt_sha256"],
              "original_launch_status": "launch_failed", "runner_returncode": None,
              "auditor_returncode": None, "auditor_status": None,
              "runner_result_sha256": None, "runner_error_sha256": None,
              "audit_sha256": None, "post_run_source_verified": False,
              "model_work_elapsed_seconds": 0.0}
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    try:
        code, elapsed, reason = old.run_bounded_child(
            command, output=output, env=env,
            timeout_seconds=old.RUNNER_SECONDS["fp32"] + old.PROCESS_GRACE_SECONDS,
            cap_bytes=old.MAX_ARM_BYTES - old.RUNNER_PARENT_RESERVE_BYTES)
        finish["runner_returncode"] = code
        finish["model_work_elapsed_seconds"] = elapsed
        if reason or elapsed >= old.RUNNER_SECONDS["fp32"] or EXPECTED_BF16_SECONDS + elapsed > old.CUMULATIVE_MODEL_SECONDS:
            raise ValueError(reason or "FP32 model-work time cap exceeded")
        result_file, error_file = runner_dir / "result.json", runner_dir / "error.json"
        finish["runner_result_sha256"] = old.sha(result_file) if result_file.is_file() else None
        finish["runner_error_sha256"] = old.sha(error_file) if error_file.is_file() else None
        if old.directory_bytes(output) >= old.MAX_ARM_BYTES:
            raise ValueError("Entire FP32 attempt byte cap reached before audit")
        if verify_continuation_freeze(verify_preserved_bf16(), verify_posthoc_receipt(preserved))["receipt_sha256"] != frozen["receipt_sha256"]:
            raise ValueError("Continuation evidence changed after model work")
        finish["post_run_source_verified"] = True
        if code != 0 or not result_file.is_file() or error_file.exists():
            raise ValueError("FP32 runner failed or lacked success receipt")
        audit_code, _, audit_reason = old.run_bounded_child(
            [sys.executable, str(AUDITOR), str(runner_dir)], output=output, env=env,
            timeout_seconds=old.AUDITOR_SECONDS,
            cap_bytes=old.MAX_ARM_BYTES - old.PARENT_RECEIPT_RESERVE_BYTES,
            stdout_name="audit_stdout.txt", stderr_name="audit_stderr.txt")
        finish["auditor_returncode"] = audit_code
        if audit_reason or audit_code:
            raise ValueError(audit_reason or "v2 CPU auditor rejected FP32 raw")
        audit = old.read_json(output / "audit_stdout.txt")
        if (audit.get("status") != old.AUDIT_STATUS or audit.get("arm") != "fp32" or
                audit.get("result_sha256") != finish["runner_result_sha256"] or
                audit.get("auditor_sha256") != V2_AUDITOR_SHA or
                audit.get("runner_sha256") != old.LIVE_SOURCE_PINS["runner_sha256"] or
                audit.get("native_source_sha256") != old.LIVE_SOURCE_PINS["native_source_sha256"] or
                audit.get("protocol_sha256") != old.LIVE_SOURCE_PINS["protocol_sha256"]):
            raise ValueError("FP32 v2 audit receipt invalid")
        (output / "audit_stdout.txt").rename(output / "audit.json")
        finish["audit_sha256"] = old.sha(output / "audit.json")
        finish["auditor_status"] = audit["status"]
        if old.directory_bytes(output) >= old.MAX_ARM_BYTES:
            raise ValueError("Entire FP32 attempt byte cap reached after audit")
        if verify_continuation_freeze(verify_preserved_bf16(), verify_posthoc_receipt(preserved))["receipt_sha256"] != frozen["receipt_sha256"]:
            raise ValueError("Continuation evidence changed after audit")
        finish["status"] = old.LAUNCH_STATUS
    except Exception as exc:
        finish["failure_reason"] = str(exc)
        finish["exception_type"] = type(exc).__name__
    finally:
        finish["elapsed_seconds"] = time.monotonic() - started
        finish["wall_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            finish["entire_attempt_bytes_before_finish"] = old.directory_bytes(output)
            if finish["entire_attempt_bytes_before_finish"] > old.MAX_ARM_BYTES - 65536:
                finish["status"] = "launch_failed"
                finish["failure_reason"] = "entire_attempt_byte_cap_reached_before_finish"
        except ValueError as exc:
            finish["entire_attempt_bytes_before_finish"] = None
            finish["attempt_size_error"] = str(exc)
            finish["status"] = "launch_failed"
        old.exclusive_json(output / "launch_finish.json", finish)
        if "attempt_size_error" not in finish and old.directory_bytes(output) > old.MAX_ARM_BYTES:
            raise ValueError("Entire FP32 attempt exceeds byte cap including finish receipt")
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.check_freeze:
        if args.output_dir is not None:
            parser.error("--check-freeze takes no output directory")
        _, _, frozen = admission(RUNS / "fp32" / "admission_probe")
        print(json.dumps({"status": "verified_fp32_continuation_freeze_only",
                          "freeze_sha256": frozen["receipt_sha256"]}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = launch(args.output_dir or RUNS / "fp32" / stamp)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != old.LAUNCH_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
