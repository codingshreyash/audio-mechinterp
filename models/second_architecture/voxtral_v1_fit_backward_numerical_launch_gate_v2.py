"""External committed-freeze gate for amended old-FIT BF16 arithmetic arm.

This module imports no model code. No execution is permitted without the
externally authored, committed freeze receipt and independent launcher review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FREEZE = HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_EXTERNAL_FREEZE_RECEIPT_V2.json"
RUNS = HERE / "runs/voxtral_v1_fit_backward_numerical_launch_v2"
SCOPE = "amended_bf16_paired_edit_numerical_consistency_only; donor_and_no_grad_report_only"
ARMS = {"bf16": 900}
MAX_DRIVER_BYTES = 30 * 1024**3
PROCESS_GRACE_SECONDS = 120
AUDIT_TIMEOUT_SECONDS = 120

FILES = {
    "runner_sha256": HERE / "voxtral_v1_fit_backward_numerical_followup_v2.py",
    "runner_tests_sha256": HERE / "test_voxtral_v1_fit_backward_numerical_followup_v2.py",
    "protocol_sha256": HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_FOLLOWUP_PROTOCOL_V1.md",
    "amendment_sha256": HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_FOLLOWUP_AMENDMENT_V2.md",
    "diagnosis_sha256": HERE / "VOXTRAL_NUMERICAL_BF16_LOAD_FAILURE_DIAGNOSIS.md",
    "runner_review_sha256": HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_FOLLOWUP_V2_INDEPENDENT_REVIEW.md",
    "auditor_sha256": HERE / "voxtral_v1_fit_backward_numerical_audit_v2.py",
    "auditor_tests_sha256": HERE / "test_voxtral_v1_fit_backward_numerical_audit_v2.py",
    "auditor_review_sha256": HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_AUDIT_V2_INDEPENDENT_REVIEW.md",
    "fp32_v1_result_sha256": HERE / "runs/voxtral_v1_fit_backward_numerical_launch_v1/fp32/20261118T120909312905Z/runner/result.json",
    "fp32_v1_audit_sha256": HERE / "runs/voxtral_v1_fit_backward_numerical_launch_v1/fp32/20261118T120909312905Z/audit.json",
    "launcher_sha256": HERE / "voxtral_v1_fit_backward_numerical_launch_gate_v2.py",
    "launcher_tests_sha256": HERE / "test_voxtral_v1_fit_backward_numerical_launch_gate_v2.py",
    "launcher_protocol_sha256": HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_LAUNCH_GATE_PROTOCOL_V2.md",
    "launcher_review_sha256": HERE / "VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_LAUNCH_V2_INDEPENDENT_REVIEW.md",
}
PINNED = {
    "runner_sha256": "93d2c607f4c218b9e027ffb4c221347663efb0729cc6e419b9c940dffda89b48",
    "runner_tests_sha256": "25cfe97d30504e807cb8b23ef2f0b7fca308f88debc11ac6036ddbf55b18061c",
    "protocol_sha256": "5778a4b84e535f7182a3daa2b973637a1c9127118ce2434ce82ba80fee467a82",
    "amendment_sha256": "8c1515c7f967392213189aefea8d84df814e86c4a4ce96ad6bcc62e068e33ffa",
    "diagnosis_sha256": "ad8ac4423ef51a1e5e674e8bb4a91b2a588ad312e49aa519d5f5a885bb08db5e",
    "auditor_sha256": "d0902608d90b1040b7e54318964d0a0b5b747d4cbd5eaf8a3eaeeda71f220ec9",
    "auditor_tests_sha256": "0c4adc89336e82c0fb1e544bbd23179527c83487a731a38a3dc40796e42444c6",
    "fp32_v1_result_sha256": "b1a124c9df14c980af18dee46cc058a98cc167a1269f33252d0e0980fc7e37fb",
    "fp32_v1_audit_sha256": "214f682be2e9b219b2375d3cb140cd10cd5b2714f4343d9ba35ea1c484399eb3",
    "launcher_protocol_sha256": "9a0469b3238dea7e142cb21ad8a67513e516a9456d19c8af4e5dcbfd173dbbbc",
}
REVIEWED_LAUNCH_FILES = ("launcher_sha256", "launcher_tests_sha256",
                         "launcher_protocol_sha256", "launcher_review_sha256")
def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def git_blob(repo: Path, ref: str, path: Path) -> bytes:
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Bound path outside repository: {path}") from exc
    completed = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{relative}"],
                               capture_output=True, check=False)
    if completed.returncode:
        raise ValueError(f"Missing committed blob at {ref}: {relative}")
    return completed.stdout


def committed_same(repo: Path, ref: str, path: Path) -> None:
    if not path.is_file() or path.read_bytes() != git_blob(repo, ref, path):
        raise ValueError(f"Uncommitted or changed bound file: {path.name}")


def verify_freeze(freeze: Path = FREEZE, *, files: dict[str, Path] = FILES,
                  pinned: dict[str, str] = PINNED, repo: Path = REPO) -> dict:
    """Fail closed on absent, dirty or insufficiently reviewed external freeze."""
    if not freeze.is_file():
        raise ValueError("External freeze receipt absent")
    committed_same(repo, "HEAD", freeze)
    raw = freeze.read_bytes()
    receipt = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(receipt, dict) or set(receipt) != {"status", "permitted_arms", "audit_scope",
                                                 "launcher_review_commit", "reviewed_utc", *files}:
        raise ValueError("External freeze receipt schema changed")
    if (receipt["status"] != "frozen_old_fit_numerical_launch_v2" or
            receipt["permitted_arms"] != ["bf16"] or receipt["audit_scope"] != SCOPE):
        raise ValueError("External freeze status/arms/scope invalid")
    commit = receipt["launcher_review_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Launcher review commit missing or malformed")
    try:
        reviewed = datetime.fromisoformat(receipt["reviewed_utc"].replace("Z", "+00:00"))
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError("External review UTC timestamp invalid") from exc
    if reviewed.tzinfo is None or reviewed.utcoffset() != timedelta(0):
        raise ValueError("External review timestamp is not UTC")
    for name, path in files.items():
        value = receipt[name]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Malformed {name} in external freeze")
        committed_same(repo, "HEAD", path)
        if sha(path) != value or (name in pinned and pinned[name] != value):
            raise ValueError(f"External freeze hash mismatch: {name}")
    for name in REVIEWED_LAUNCH_FILES:
        if sha_bytes(git_blob(repo, commit, files[name])) != receipt[name]:
            raise ValueError(f"Launcher review commit did not contain {name}")
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                              capture_output=True, check=False)
    if ancestor.returncode:
        raise ValueError("Launcher review commit is not an ancestor of external freeze")
    relative_freeze = freeze.resolve().relative_to(repo.resolve()).as_posix()
    premature = subprocess.run(["git", "-C", str(repo), "cat-file", "-e",
                                f"{commit}:{relative_freeze}"], capture_output=True, check=False)
    if premature.returncode == 0:
        raise ValueError("External freeze receipt predates launcher review")
    return {"receipt": receipt, "receipt_sha256": sha_bytes(raw),
            "review_commit": commit}


def exclusive_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def reserve_attempt(output: Path, arm: str, frozen: dict, runner_command: list[str]) -> float:
    """Reserve a unique parent directory and start the immutable launch trail."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    exclusive_json(output / "launch_start.json", {
        "status": "launch_started", "arm": arm,
        "wall_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_receipt_sha256": frozen["receipt_sha256"],
        "launcher_review_commit": frozen["review_commit"],
        "launcher_sha256": frozen["receipt"]["launcher_sha256"],
        "runner_sha256": frozen["receipt"]["runner_sha256"],
        "auditor_sha256": frozen["receipt"]["auditor_sha256"],
        "fp32_v1_result_sha256": frozen["receipt"]["fp32_v1_result_sha256"],
        "fp32_v1_audit_sha256": frozen["receipt"]["fp32_v1_audit_sha256"],
        "runner_command": runner_command,
        "runner_cap_seconds": ARMS[arm], "runner_mps_driver_cap_bytes": MAX_DRIVER_BYTES,
        "launcher_process_timeout_seconds": ARMS[arm] + PROCESS_GRACE_SECONDS,
        "audit_scope": SCOPE})
    return started


def launch(arm: str, output: Path) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Launch requires the pinned second_architecture virtual environment")
    frozen = verify_freeze()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    runner_dir = output / "runner"
    command = [sys.executable, str(FILES["runner_sha256"]), "--run", "--arm", arm,
               "--output-dir", str(runner_dir)]
    started = reserve_attempt(output, arm, frozen, command)
    finish = {"status": "launch_failed", "arm": arm, "freeze_receipt_sha256": frozen["receipt_sha256"],
              "runner_returncode": None, "auditor_returncode": None, "auditor_status": None,
              "post_run_source_verified": False, "runner_result_sha256": None,
              "runner_error_sha256": None, "audit_sha256": None}
    try:
        with (output / "runner_stdout.txt").open("xb") as out, (output / "runner_stderr.txt").open("xb") as err:
            try:
                process = subprocess.run(command, cwd=REPO, env=env, stdout=out, stderr=err,
                                         timeout=ARMS[arm] + PROCESS_GRACE_SECONDS, check=False)
                finish["runner_returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                finish["failure_reason"] = "runner_process_timeout"
        result_file = runner_dir / "result.json"
        error_file = runner_dir / "error.json"
        finish["runner_result_sha256"] = sha(result_file) if result_file.is_file() else None
        finish["runner_error_sha256"] = sha(error_file) if error_file.is_file() else None
        after = verify_freeze()
        if after["receipt_sha256"] != frozen["receipt_sha256"]:
            raise ValueError("External freeze changed during attempt")
        finish["post_run_source_verified"] = True
        if result_file.is_file() == error_file.is_file():
            raise ValueError("Runner did not produce exactly one result/error receipt")
        audit_command = [sys.executable, str(FILES["auditor_sha256"]), str(runner_dir), "--arm", arm]
        audited = subprocess.run(audit_command, cwd=REPO, env=env, capture_output=True,
                                 timeout=AUDIT_TIMEOUT_SECONDS, check=False)
        finish["auditor_returncode"] = audited.returncode
        if audited.returncode:
            with (output / "audit_error.txt").open("xb") as stream:
                stream.write(audited.stderr)
            raise RuntimeError("Frozen CPU auditor rejected runner receipt")
        audit_value = json.loads(audited.stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if (not isinstance(audit_value, dict) or
                audit_value.get("auditor_sha256") != frozen["receipt"]["auditor_sha256"] or
                audit_value.get("runner_sha256") != frozen["receipt"]["runner_sha256"] or
                audit_value.get("protocol_sha256") != frozen["receipt"]["protocol_sha256"] or
                audit_value.get("amendment_sha256") != frozen["receipt"]["amendment_sha256"] or
                audit_value.get("arm") != arm or
                audit_value.get("status") not in ("audited_completed_numerical_consistency_only",
                                                  "audited_failed_attempt_receipt_only")):
            raise ValueError("Frozen auditor stdout invalid")
        if result_file.is_file() and audit_value.get("result_sha256") != finish["runner_result_sha256"]:
            raise ValueError("Frozen audit did not bind the runner result receipt")
        with (output / "audit.json").open("xb") as stream:
            stream.write(audited.stdout)
            stream.flush()
            os.fsync(stream.fileno())
        finish["audit_sha256"] = sha(output / "audit.json")
        finish["auditor_status"] = audit_value.get("status")
        if (finish["runner_returncode"] == 0 and
                finish["auditor_status"] == "audited_completed_numerical_consistency_only" and
                result_file.is_file()):
            finish["status"] = "launch_completed_audited_numerical_consistency_only"
        else:
            finish["failure_reason"] = finish.get("failure_reason", "runner_failed_or_audited_failure_only")
    except Exception as exc:
        finish["failure_reason"] = finish.get("failure_reason", str(exc))
        finish["exception_type"] = type(exc).__name__
    finally:
        finish["elapsed_seconds"] = time.monotonic() - started
        finish["wall_utc"] = datetime.now(timezone.utc).isoformat()
        exclusive_json(output / "launch_finish.json", finish)
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true", help="read-only committed freeze verification")
    mode.add_argument("--run", action="store_true", help="launch exactly one reviewed arithmetic arm")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.check_freeze:
        if args.arm is not None or args.output_dir is not None:
            parser.error("--check-freeze takes no arm/output directory")
        frozen = verify_freeze()
        print(json.dumps({"status": "verified_external_freeze_only",
                          "freeze_receipt_sha256": frozen["receipt_sha256"],
                          "launcher_review_commit": frozen["review_commit"]}, indent=2))
        return
    if args.arm is None:
        parser.error("--run requires --arm")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = launch(args.arm, args.output_dir or RUNS / args.arm / stamp)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "launch_completed_audited_numerical_consistency_only":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
