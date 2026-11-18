"""External committed-freeze launch gate for the old-FIT Stage-A native arms.

This module imports no model code. Live use still requires a separately
committed external freeze receipt after independent launcher review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs/voxtral_old_fit_handoff_stage_a_launch_v1"
FREEZE = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_EXTERNAL_FREEZE_V1.json"
BF16_MANIFEST = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_BF16_ATTEMPT_MANIFEST_V1.json"
ARMS = ("bf16", "fp32")
RUNNER_SECONDS = {"bf16": 900, "fp32": 1200}
PROCESS_GRACE_SECONDS = 120
AUDITOR_SECONDS = 120
CUMULATIVE_MODEL_SECONDS = 4500
MAX_DRIVER_BYTES = 30 * 1024**3
MAX_ARM_BYTES = int(1.5 * 1024**3)
MAX_CAPTURE_BYTES_PER_STREAM = 8 * 1024**2
PARENT_RECEIPT_RESERVE_BYTES = 1024**2
RUNNER_PARENT_RESERVE_BYTES = 2 * MAX_CAPTURE_BYTES_PER_STREAM + PARENT_RECEIPT_RESERVE_BYTES
SCOPE = "posthoc_one_old_fit_case_native_stage_only_no_security_claim"
AUDIT_STATUS = "audited_completed_old_fit_handoff_stage_a_native_only"
RESULT_STATUS = "completed_old_fit_handoff_stage_a_native_only"
LAUNCH_STATUS = "launch_completed_audited_old_fit_handoff_stage_a_native_only"

FILES = {
    "protocol_sha256": HERE / "VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md",
    "plan_review_sha256": HERE / "VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_INDEPENDENT_PLAN_REREVIEW_V1.md",
    "native_source_sha256": HERE / "voxtral_old_fit_handoff_native_stage_a_v1.py",
    "native_tests_sha256": HERE / "test_voxtral_old_fit_handoff_native_stage_a_v1.py",
    "runner_sha256": HERE / "voxtral_old_fit_handoff_stage_a_runner_v1.py",
    "runner_schema_sha256": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_SOURCE_SCHEMA_V1.md",
    "runner_review_sha256": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_INDEPENDENT_SOURCE_REREVIEW_V1.md",
    "auditor_sha256": HERE / "voxtral_old_fit_handoff_stage_a_audit_v1.py",
    "auditor_tests_sha256": HERE / "test_voxtral_old_fit_handoff_stage_a_audit_v1.py",
    "auditor_hold_review_sha256": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_AUDIT_INDEPENDENT_REVIEW_V1.md",
    "auditor_review_sha256": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_AUDIT_INDEPENDENT_REREVIEW_V1.md",
    "launcher_sha256": HERE / "voxtral_old_fit_handoff_stage_a_launch_gate_v1.py",
    "launcher_tests_sha256": HERE / "test_voxtral_old_fit_handoff_stage_a_launch_gate_v1.py",
    "launcher_protocol_sha256": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_LAUNCH_GATE_PROTOCOL_V1.md",
    "launcher_review_sha256": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_LAUNCH_INDEPENDENT_REVIEW_V1.md",
}
REVIEWED_LAUNCH_FILES = ("launcher_sha256", "launcher_tests_sha256",
                         "launcher_protocol_sha256", "launcher_review_sha256")
REVIEWED_SOURCE_FILES = ("native_source_sha256", "native_tests_sha256",
                         "runner_sha256", "runner_schema_sha256", "runner_review_sha256")
REVIEWED_AUDITOR_FILES = ("auditor_sha256", "auditor_tests_sha256",
                          "auditor_hold_review_sha256", "auditor_review_sha256")
# Independently reviewed Stage-A source and auditor bytes. These are explicit
# expected values, never populated from the external freeze self-report.
LIVE_SOURCE_PINS: dict[str, str] | None = {
    "protocol_sha256": "61a67ac87a4e4e17cb0b1f7ad972f25fda4e0d90e0b70798e7144b3fd6f2484b",
    "plan_review_sha256": "8377420a6ae632a7a4cce5b6f69790eb834ec6218e48fd53952c3445edf58e19",
    "native_source_sha256": "dc63b8c10ef605e1cc8538e3d37d8d2531e50ca7a46140ee1ee551054e3de978",
    "native_tests_sha256": "e2e3c1b51f72d54c573665c96d984ea73857af6560d2a38ef0d51be28b8a0a77",
    "runner_sha256": "9f8f6b52d3e218acc496b78b35247cfdf322672cd487e585b92d58f08205f1dc",
    "runner_schema_sha256": "b6a9674f65ee352ed6635a48a39ef9bffb932419844e52bb6d28db8c7e9d41d5",
    "runner_review_sha256": "f82b000baa4aaa817292860d72e4bb61dbc761f90d2d607e69215368186bf1ef",
    "auditor_sha256": "4f800ed604295ae0e4fbd53c893fccd14b8968badb43b835e6ec975e49b0cf3d",
    "auditor_tests_sha256": "ea9588c657b5dde9bfbb5963e68ed8512987e4371d05206d917f5c8803e23351",
    "auditor_hold_review_sha256": "e0f31b8bf4b13f537e3257df7e68bf55559c543f0e5ef046733df0beb4fb0fcb",
    "auditor_review_sha256": "055c02fda105f6c0bc3a8b295294833ef7f170b6206baae75b57e779e48a8f51",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def git_blob(repo: Path, ref: str, path: Path) -> bytes:
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Bound path outside repository") from exc
    result = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{relative}"],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Missing committed blob at {ref}: {relative}")
    return result.stdout


def committed_same(repo: Path, ref: str, path: Path) -> None:
    if not path.is_file() or path.read_bytes() != git_blob(repo, ref, path):
        raise ValueError(f"Uncommitted or changed bound file: {path.name}")


def ancestor(repo: Path, earlier: str, later: str) -> None:
    result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", earlier, later],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Review commit is not an ancestor of freeze")


def first_add_commit(repo: Path, path: Path) -> str:
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    result = subprocess.run(["git", "-C", str(repo), "log", "--diff-filter=A",
                             "--format=%H", "HEAD", "--", relative],
                            capture_output=True, text=True, check=False)
    additions = result.stdout.splitlines()
    if result.returncode or len(additions) != 1 or not re.fullmatch(r"[0-9a-f]{40}", additions[0]):
        raise ValueError("Bound receipt must have exactly one Git first-add commit")
    return additions[0]


def utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Invalid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Timestamp is not UTC")
    return parsed


def verify_freeze(freeze: Path = FREEZE, *, files: dict[str, Path] = FILES,
                  pinned: dict[str, str] | None = LIVE_SOURCE_PINS, repo: Path = REPO) -> dict:
    """Reject absent, dirty, unreviewed, or self-frozen source before model load."""
    if pinned is None:
        raise ValueError("Live source pins absent; source scaffold cannot launch")
    if set(files).difference(REVIEWED_LAUNCH_FILES).difference(pinned):
        raise ValueError("Source pins incomplete; launch cannot trust freeze self-report")
    if not freeze.is_file():
        raise ValueError("External freeze receipt absent")
    committed_same(repo, "HEAD", freeze)
    raw = freeze.read_bytes()
    receipt = read_json(freeze)
    expected = {"status", "permitted_arms", "audit_scope", "source_review_commit",
                "auditor_review_commit", "launcher_review_commit", "reviewed_utc", *files}
    if set(receipt) != expected:
        raise ValueError("External freeze receipt schema changed")
    if (receipt["status"] != "frozen_old_fit_handoff_stage_a_v1" or
            receipt["permitted_arms"] != list(ARMS) or receipt["audit_scope"] != SCOPE):
        raise ValueError("External freeze status, arms, or scope invalid")
    utc(receipt["reviewed_utc"])
    for field in ("source_review_commit", "auditor_review_commit", "launcher_review_commit"):
        if not isinstance(receipt[field], str) or not re.fullmatch(r"[0-9a-f]{40}", receipt[field]):
            raise ValueError(f"Malformed {field}")
    for name, path in files.items():
        value = receipt[name]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Malformed {name}")
        committed_same(repo, "HEAD", path)
        if sha(path) != value or (name in pinned and pinned[name] != value):
            raise ValueError(f"External freeze hash mismatch: {name}")
    for name in REVIEWED_SOURCE_FILES:
        if sha_bytes(git_blob(repo, receipt["source_review_commit"], files[name])) != receipt[name]:
            raise ValueError(f"Source review commit did not contain {name}")
    for name in REVIEWED_AUDITOR_FILES:
        if sha_bytes(git_blob(repo, receipt["auditor_review_commit"], files[name])) != receipt[name]:
            raise ValueError(f"Auditor review commit did not contain {name}")
    for name in REVIEWED_LAUNCH_FILES:
        if sha_bytes(git_blob(repo, receipt["launcher_review_commit"], files[name])) != receipt[name]:
            raise ValueError(f"Launcher review commit did not contain {name}")
    ancestor(repo, receipt["source_review_commit"], receipt["auditor_review_commit"])
    ancestor(repo, receipt["auditor_review_commit"], receipt["launcher_review_commit"])
    ancestor(repo, receipt["launcher_review_commit"], "HEAD")
    relative = freeze.resolve().relative_to(repo.resolve()).as_posix()
    premature = subprocess.run(["git", "-C", str(repo), "cat-file", "-e",
                                f"{receipt['launcher_review_commit']}:{relative}"],
                               capture_output=True, check=False)
    if premature.returncode == 0:
        raise ValueError("External freeze predates launcher review")
    freeze_add = first_add_commit(repo, freeze)
    if freeze_add == receipt["launcher_review_commit"]:
        raise ValueError("External freeze was added in launcher review commit")
    ancestor(repo, receipt["launcher_review_commit"], freeze_add)
    return {"receipt": receipt, "receipt_sha256": sha_bytes(raw), "freeze_path": freeze}


def exclusive_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Attempt directory contains a symlink")
        if path.is_file():
            total += path.stat().st_size
    return total


def run_bounded_child(command: list[str], *, output: Path, env: dict[str, str],
                      timeout_seconds: float, cap_bytes: int = MAX_ARM_BYTES,
                      stream_cap_bytes: int = MAX_CAPTURE_BYTES_PER_STREAM,
                      stdout_name: str = "runner_stdout.txt",
                      stderr_name: str = "runner_stderr.txt") -> tuple[int, float, str | None]:
    """Drain both pipes and kill on wall, log, or entire-attempt byte cap.

    The runner also enforces its own exact write cap. This independent parent
    watchdog includes stdout, stderr and launch receipts in the same ceiling.
    """
    begun = time.monotonic()
    process = subprocess.Popen(command, cwd=REPO, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    reason = None
    selector = selectors.DefaultSelector()
    counts = {"stdout": 0, "stderr": 0}
    with (output / stdout_name).open("xb") as stdout_file, \
            (output / stderr_name).open("xb") as stderr_file:
        targets = {"stdout": stdout_file, "stderr": stderr_file}
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        try:
            while selector.get_map() or process.poll() is None:
                elapsed = time.monotonic() - begun
                if elapsed >= timeout_seconds:
                    reason = "runner_process_timeout"
                elif directory_bytes(output) >= cap_bytes:
                    reason = "entire_attempt_byte_cap_reached"
                if reason:
                    process.kill()
                    break
                for key, _ in selector.select(timeout=0.2):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    label = key.data
                    left = stream_cap_bytes - counts[label]
                    if len(chunk) > left:
                        if left:
                            targets[label].write(chunk[:left])
                            counts[label] += left
                        reason = f"runner_{label}_capture_cap_reached"
                        process.kill()
                        break
                    targets[label].write(chunk)
                    counts[label] += len(chunk)
                if reason:
                    break
            process.wait(timeout=5)
            stdout_file.flush()
            stderr_file.flush()
            os.fsync(stdout_file.fileno())
            os.fsync(stderr_file.fileno())
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
    return process.returncode, time.monotonic() - begun, reason


def _attempt_dirs(runs: Path, arm: str) -> list[Path]:
    directory = runs / arm
    return sorted(path for path in directory.iterdir() if path.is_dir()) if directory.is_dir() else []


def verify_bf16_manifest(manifest: Path, *, frozen: dict, runs: Path = RUNS,
                         repo: Path = REPO) -> dict:
    """Require a separately committed audited BF16 attempt before FP32."""
    committed_same(repo, "HEAD", manifest)
    value = read_json(manifest)
    expected = {"status", "arm", "attempt_relative_path", "freeze_receipt_sha256",
                "launch_start_sha256", "launch_finish_sha256", "result_sha256",
                "audit_sha256", "model_work_elapsed_seconds"}
    if set(value) != expected or value["status"] != LAUNCH_STATUS or value["arm"] != "bf16":
        raise ValueError("Committed BF16 manifest schema/status invalid")
    if value["freeze_receipt_sha256"] != frozen["receipt_sha256"]:
        raise ValueError("BF16 manifest belongs to another freeze")
    freeze_add = first_add_commit(repo, frozen["freeze_path"])
    manifest_add = first_add_commit(repo, manifest)
    if manifest_add == freeze_add:
        raise ValueError("BF16 manifest was added with freeze, before run")
    ancestor(repo, freeze_add, manifest_add)
    relative = Path(value["attempt_relative_path"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError("BF16 attempt path invalid")
    attempt = runs / "bf16" / relative
    if _attempt_dirs(runs, "bf16") != [attempt] or _attempt_dirs(runs, "fp32"):
        raise ValueError("Unexpected Stage-A attempts or retries")
    fields = {"launch_start_sha256": attempt / "launch_start.json",
              "launch_finish_sha256": attempt / "launch_finish.json",
              "result_sha256": attempt / "runner/result.json",
              "audit_sha256": attempt / "audit.json"}
    for key, path in fields.items():
        if not path.is_file() or sha(path) != value[key]:
            raise ValueError(f"BF16 manifest file hash mismatch: {key}")
    start, finish, result, audit = (read_json(fields[k]) for k in fields)
    for key in ("meta", "model_layout", "progress"):
        path = attempt / "runner" / ("progress.jsonl" if key == "progress" else f"{key}.json")
        if not path.is_file() or path.is_symlink() or sha(path) != result.get(f"{key}_sha256"):
            raise ValueError(f"BF16 runner receipt file mismatch: {key}")
    for key in ("probes", "details"):
        relative_blob = result.get(f"{key}_path")
        digest = result.get(f"{key}_sha256")
        if (not isinstance(relative_blob, str) or not isinstance(digest, str) or
                not re.fullmatch(r"[0-9a-f]{64}", digest) or
                relative_blob != f"blobs/{digest}.pt"):
            raise ValueError(f"BF16 {key} sidecar path invalid")
        blob = attempt / "runner" / relative_blob
        if (not blob.is_file() or blob.is_symlink() or sha(blob) != digest or
                blob.stat().st_size != result.get(f"{key}_bytes") or
                audit.get(f"{key}_sha256") != digest):
            raise ValueError(f"BF16 {key} sidecar missing or changed")
    if directory_bytes(attempt) > MAX_ARM_BYTES:
        raise ValueError("BF16 entire attempt exceeds output cap")
    if (start.get("arm") != "bf16" or start.get("freeze_receipt_sha256") != frozen["receipt_sha256"] or
            start.get("runner_sha256") != frozen["receipt"]["runner_sha256"] or
            start.get("auditor_sha256") != frozen["receipt"]["auditor_sha256"] or
            finish.get("status") != LAUNCH_STATUS or finish.get("arm") != "bf16" or
            finish.get("freeze_receipt_sha256") != frozen["receipt_sha256"] or
            finish.get("post_run_source_verified") is not True or
            finish.get("runner_returncode") != 0 or finish.get("auditor_returncode") != 0 or
            finish.get("runner_result_sha256") != value["result_sha256"] or
            finish.get("audit_sha256") != value["audit_sha256"] or
            result.get("status") != RESULT_STATUS or result.get("arm") != "bf16" or
            result.get("script_sha256") != frozen["receipt"]["runner_sha256"] or
            result.get("native_source_sha256") != frozen["receipt"]["native_source_sha256"] or
            result.get("protocol_sha256") != frozen["receipt"]["protocol_sha256"] or
            audit.get("status") != AUDIT_STATUS or audit.get("arm") != "bf16" or
            audit.get("result_sha256") != value["result_sha256"] or
            audit.get("auditor_sha256") != frozen["receipt"]["auditor_sha256"] or
            audit.get("runner_sha256") != frozen["receipt"]["runner_sha256"] or
            audit.get("native_source_sha256") != frozen["receipt"]["native_source_sha256"] or
            audit.get("protocol_sha256") != frozen["receipt"]["protocol_sha256"]):
        raise ValueError("BF16 audited receipt chain invalid")
    elapsed = value["model_work_elapsed_seconds"]
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
            not 0 < elapsed < RUNNER_SECONDS["bf16"] or
            finish.get("model_work_elapsed_seconds") != elapsed):
        raise ValueError("BF16 elapsed time invalid")
    return value


def admission(arm: str, output: Path, *, frozen: dict, runs: Path = RUNS,
              manifest: Path = BF16_MANIFEST, repo: Path = REPO) -> float:
    """Read-only serial/no-retry and cumulative-cap check before reservation."""
    if arm not in ARMS or output.parent != runs / arm:
        raise ValueError("Arm or attempt output path invalid")
    if not output.name or output.exists():
        raise ValueError("Attempt output already exists")
    if arm == "bf16":
        if manifest.exists() or _attempt_dirs(runs, "bf16") or _attempt_dirs(runs, "fp32"):
            raise ValueError("Stage-A attempt already exists; no retry")
        spent = 0.0
    else:
        spent = float(verify_bf16_manifest(manifest, frozen=frozen, runs=runs, repo=repo)
                      ["model_work_elapsed_seconds"])
    if spent + RUNNER_SECONDS[arm] > CUMULATIVE_MODEL_SECONDS:
        raise ValueError("Cumulative model-work cap insufficient for full arm")
    return spent


def reserve_attempt(output: Path, arm: str, frozen: dict, command: list[str],
                    spent: float, *, runs: Path = RUNS) -> float:
    """Use permanent exclusive arm reservation; a failed launch still consumes it."""
    runs.mkdir(parents=True, exist_ok=True)
    (runs / arm).mkdir(parents=True, exist_ok=True)
    reservation = runs / f"{arm}_reservation.json"
    exclusive_json(reservation, {"status": "arm_reserved_no_retry", "arm": arm,
                                 "attempt": output.name, "freeze_receipt_sha256": frozen["receipt_sha256"]})
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    exclusive_json(output / "launch_start.json", {
        "status": "launch_started", "arm": arm,
        "wall_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_receipt_sha256": frozen["receipt_sha256"],
        "source_review_commit": frozen["receipt"]["source_review_commit"],
        "auditor_review_commit": frozen["receipt"]["auditor_review_commit"],
        "launcher_review_commit": frozen["receipt"]["launcher_review_commit"],
        "runner_sha256": frozen["receipt"]["runner_sha256"],
        "auditor_sha256": frozen["receipt"]["auditor_sha256"],
        "runner_command": command, "runner_cap_seconds": RUNNER_SECONDS[arm],
        "process_timeout_seconds": RUNNER_SECONDS[arm] + PROCESS_GRACE_SECONDS,
        "runner_mps_driver_cap_bytes": MAX_DRIVER_BYTES,
        "arm_output_cap_bytes": MAX_ARM_BYTES,
        "max_capture_bytes_per_stream": MAX_CAPTURE_BYTES_PER_STREAM,
        "runner_parent_reserve_bytes": RUNNER_PARENT_RESERVE_BYTES,
        "cumulative_model_work_cap_seconds": CUMULATIVE_MODEL_SECONDS,
        "prior_model_work_seconds": spent, "audit_scope": SCOPE})
    return started


def launch(arm: str, output: Path) -> dict:
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Launch requires pinned second_architecture virtual environment")
    frozen = verify_freeze()
    spent = admission(arm, output, frozen=frozen)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    runner_dir = output / "runner"
    command = [sys.executable, str(FILES["runner_sha256"]), "--run", "--arm", arm,
               "--output-dir", str(runner_dir)]
    started = reserve_attempt(output, arm, frozen, command, spent)
    finish = {"status": "launch_failed", "arm": arm,
              "freeze_receipt_sha256": frozen["receipt_sha256"],
              "runner_returncode": None, "auditor_returncode": None,
              "auditor_status": None, "runner_result_sha256": None,
              "runner_error_sha256": None, "audit_sha256": None,
              "post_run_source_verified": False, "model_work_elapsed_seconds": 0.0}
    try:
        returncode, process_elapsed, stop_reason = run_bounded_child(
            command, output=output, env=env,
            timeout_seconds=RUNNER_SECONDS[arm] + PROCESS_GRACE_SECONDS,
            cap_bytes=MAX_ARM_BYTES - RUNNER_PARENT_RESERVE_BYTES)
        finish["runner_returncode"] = returncode
        finish["model_work_elapsed_seconds"] = process_elapsed
        if stop_reason:
            raise ValueError(stop_reason)
        if finish["model_work_elapsed_seconds"] >= RUNNER_SECONDS[arm]:
            raise ValueError("Arm model-process wall cap exceeded")
        if spent + finish["model_work_elapsed_seconds"] > CUMULATIVE_MODEL_SECONDS:
            raise ValueError("Cumulative model-work cap exceeded")
        if directory_bytes(output) >= MAX_ARM_BYTES:
            raise ValueError("Entire attempt byte cap reached before audit")
        result_file, error_file = runner_dir / "result.json", runner_dir / "error.json"
        finish["runner_result_sha256"] = sha(result_file) if result_file.is_file() else None
        finish["runner_error_sha256"] = sha(error_file) if error_file.is_file() else None
        after = verify_freeze()
        if after["receipt_sha256"] != frozen["receipt_sha256"]:
            raise ValueError("External freeze changed during attempt")
        finish["post_run_source_verified"] = True
        if finish["runner_returncode"] != 0 or not result_file.is_file() or error_file.exists():
            raise ValueError("Runner failed or lacked exclusive success receipt")
        audit_command = [sys.executable, str(FILES["auditor_sha256"]), str(runner_dir)]
        audit_code, _, audit_stop_reason = run_bounded_child(
            audit_command, output=output, env=env, timeout_seconds=AUDITOR_SECONDS,
            cap_bytes=MAX_ARM_BYTES - PARENT_RECEIPT_RESERVE_BYTES,
            stdout_name="audit_stdout.txt", stderr_name="audit_stderr.txt")
        finish["auditor_returncode"] = audit_code
        if audit_stop_reason:
            raise ValueError(audit_stop_reason)
        if audit_code:
            raise ValueError("Frozen CPU auditor rejected runner result")
        audit_stdout = output / "audit_stdout.txt"
        audit = json.loads(audit_stdout.read_bytes(),
                           parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        if (not isinstance(audit, dict) or audit.get("status") != AUDIT_STATUS or
                audit.get("arm") != arm or audit.get("result_sha256") != finish["runner_result_sha256"] or
                audit.get("auditor_sha256") != frozen["receipt"]["auditor_sha256"] or
                audit.get("runner_sha256") != frozen["receipt"]["runner_sha256"] or
                audit.get("native_source_sha256") != frozen["receipt"]["native_source_sha256"] or
                audit.get("protocol_sha256") != frozen["receipt"]["protocol_sha256"]):
            raise ValueError("Frozen auditor receipt invalid")
        audit_stdout.rename(output / "audit.json")
        finish["audit_sha256"] = sha(output / "audit.json")
        finish["auditor_status"] = audit["status"]
        if directory_bytes(output) >= MAX_ARM_BYTES:
            raise ValueError("Entire attempt byte cap reached after audit")
        after_audit = verify_freeze()
        if after_audit["receipt_sha256"] != frozen["receipt_sha256"]:
            raise ValueError("External freeze changed during audit")
        finish["status"] = LAUNCH_STATUS
    except Exception as exc:
        finish["failure_reason"] = finish.get("failure_reason", str(exc))
        finish["exception_type"] = type(exc).__name__
    finally:
        finish["elapsed_seconds"] = time.monotonic() - started
        finish["wall_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            finish["entire_attempt_bytes_before_finish"] = directory_bytes(output)
            if finish["entire_attempt_bytes_before_finish"] > MAX_ARM_BYTES - 65536:
                finish["status"] = "launch_failed"
                finish["failure_reason"] = "entire_attempt_byte_cap_reached_before_finish"
        except ValueError as exc:
            finish["entire_attempt_bytes_before_finish"] = None
            finish["attempt_size_error"] = str(exc)
            finish["status"] = "launch_failed"
        exclusive_json(output / "launch_finish.json", finish)
        if "attempt_size_error" not in finish and directory_bytes(output) > MAX_ARM_BYTES:
            raise ValueError("Entire attempt byte cap exceeded including finish receipt")
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.check_freeze:
        if args.arm or args.output_dir:
            parser.error("--check-freeze takes no arm/output directory")
        frozen = verify_freeze()
        print(json.dumps({"status": "verified_external_freeze_only",
                          "freeze_receipt_sha256": frozen["receipt_sha256"]}, indent=2))
        return
    if not args.arm:
        parser.error("--run requires --arm")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or RUNS / args.arm / stamp
    result = launch(args.arm, output)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != LAUNCH_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
