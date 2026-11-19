"""Held one-shot parent for the selected FIT Wav2Vec2 causal patch worker.

The parent creates no attempt until independently committed worker and parent
GO receipts bind exact source, preflight, host/deadline helper and output path.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import plistlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

import acoustic_fit_noise_causal_followup_v1 as causal
import acoustic_fit_noise_causal_worker_v1 as worker
import acoustic_fit_noise_parent_launch_v2 as base
import acoustic_fit_noise_pilot_v2 as pilot


HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "runs/acoustic_fit_noise_causal_v1/kokoro_teal_attempt_001"
PARENT_GO = HERE / "ACOUSTIC_FIT_NOISE_CAUSAL_PARENT_GO_V1.json"
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 128 * 1024**2


def source_binding() -> dict[str, Any]:
    paths = {
        "parent": Path(__file__), "worker": Path(worker.__file__),
        "helper": Path(causal.__file__), "protocol": worker.PROTOCOL,
        "preflight": worker.PREFLIGHT, "base_parent": Path(base.__file__),
    }
    return {
        "schema": "acoustic_fit_noise_causal_parent_source_v1",
        "sources": {name: {"sha256": pilot.sha_file(path),
                           "commit": pilot.source_commit(path.resolve())}
                    for name, path in paths.items()},
        "attempt_path": str(ATTEMPT.resolve()),
        "parent_deadline_seconds": MAX_SECONDS,
        "rss_bytes": MAX_RSS_BYTES, "total_output_bytes": MAX_OUTPUT_BYTES,
    }


def verify_parent_go(path: Path = PARENT_GO,
                     worker_review: Path = worker.WORKER_GO) -> dict[str, Any]:
    worker.verify_worker_go(worker.PREFLIGHT, worker_review)
    if not path.is_file() or path.is_symlink():
        raise ValueError("causal parent HOLD: independent GO receipt absent")
    binding = source_binding()
    review = json.loads(path.read_text())
    if (review.get("schema") != "acoustic_fit_noise_causal_parent_go_v1"
            or review.get("status") != "GO"
            or review.get("source_binding") != binding
            or review.get("preflight_sha256") != pilot.sha_file(worker.PREFLIGHT)
            or review.get("worker_go_sha256") != pilot.sha_file(worker_review)):
        raise ValueError("causal parent GO source/receipt binding differs")
    review_commit = pilot.source_commit(path.resolve())
    for item in (*binding["sources"].values(),
                 {"commit": pilot.source_commit(worker_review.resolve())}):
        prior = item["commit"]
        if (prior == review_commit or subprocess.run(
                ("git", "merge-base", "--is-ancestor", prior, review_commit),
                cwd=pilot.ROOT, check=False).returncode != 0):
            raise ValueError("causal parent review is not a later separate commit")
    return review


def _read_array(output: Path, descriptor: dict[str, Any],
                shape: tuple[int, ...]) -> np.ndarray:
    name = descriptor.get("path")
    if not isinstance(name, str) or "/" in name or ".." in name:
        raise ValueError("causal sidecar path invalid")
    path = output / name
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size != descriptor.get("bytes")
            or pilot.sha_file(path) != descriptor.get("file_sha256")):
        raise ValueError("causal sidecar bytes differ")
    value = np.load(path, allow_pickle=False)
    if (value.shape != shape or value.dtype != np.float32
            or descriptor.get("shape") != list(shape)
            or descriptor.get("dtype") != "float32"
            or not np.isfinite(value).all()
            or causal.array_sha(value) != descriptor.get("raw_sha256")):
        raise ValueError("causal sidecar array geometry or raw hash differs")
    return value


def audit_worker_output(output: Path) -> dict[str, Any]:
    """Read-only full 2+16 worker matrix, sidecar, edit and control audit."""
    from transformers import Wav2Vec2Processor
    import wav2vec2_fit_baseline_stage_capture_v1 as source

    records = output / "records.jsonl"
    if not records.is_file() or records.is_symlink():
        raise ValueError("causal worker records absent")
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    if len(rows) != 20:
        raise ValueError("causal worker row count differs")
    start, complete = rows[0], rows[-1]
    if (start.get("type") != "start"
            or start.get("schema") != "acoustic_fit_noise_causal_worker_v1"
            or start.get("receipt_sha256") != pilot.sha_file(worker.PREFLIGHT)
            or start.get("worker_go_sha256") != pilot.sha_file(worker.WORKER_GO)
            or start.get("source_sha256") != pilot.sha_file(Path(worker.__file__))
            or start.get("protocol_sha256") != pilot.sha_file(worker.PROTOCOL)
            or start.get("selected_case") != causal.CASE
            or start.get("selected_arms") != list(causal.ARMS)
            or complete.get("type") != "complete"
            or complete.get("baseline_rows") != 2
            or complete.get("patch_rows") != 16
            or complete.get("control_gate_pass") is not True):
        raise ValueError("causal worker start/completion identity differs")
    processor = Wav2Vec2Processor.from_pretrained(source.SNAPSHOT,
                                                  local_files_only=True)
    saved = causal._selected_rows()
    baseline: dict[str, dict[str, Any]] = {}
    used_files = {"records.jsonl"}
    previous_time = -1.0
    for row, arm in zip(rows[1:3], worker.BASELINE_ARMS, strict=True):
        if (row.get("type") != "baseline" or row.get("case") != causal.CASE
                or row.get("arm") != arm
                or row.get("pcm_sha256") != causal.PCM_HASHES[arm]
                or row.get("output") != saved[arm]["output"]
                or not isinstance(row.get("hook_counts"), dict)
                or set(row["hook_counts"]) != set(source.STAGES)
                or any(type(v) is not int or v != 1 for v in row["hook_counts"].values())):
            raise ValueError("causal baseline row differs from selected evidence")
        state = _read_array(output, row["state"], causal.STATE_SHAPE)
        logits = _read_array(output, row["logits"], (1, 286, 32))
        if (causal.array_sha(state) != causal.STATE_HASHES[arm]
                or causal.array_sha(logits) != saved[arm]["metrics"]["logits"]["sha256"]):
            raise ValueError("causal baseline state/logits differ from V2")
        ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
        decoded = processor.batch_decode([ids])[0]
        if row["output"] != {"argmax_ids": ids, "transcript": decoded}:
            raise ValueError("causal baseline complete CTC output differs from logits")
        baseline[arm] = {"state": state, "logits": logits}
        used_files.update((row["state"]["path"], row["logits"]["path"]))
        elapsed = row.get("elapsed_seconds")
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or elapsed < previous_time or elapsed > MAX_SECONDS
                or type(row.get("rss_bytes")) is not int
                or not 0 < row["rss_bytes"] <= MAX_RSS_BYTES):
            raise ValueError("causal baseline resource evidence invalid")
        previous_time = elapsed
    planned = [(recipient, arm) for recipient in worker.BASELINE_ARMS
               for arm in causal.ARMS]
    for row, (recipient_arm, arm) in zip(rows[3:-1], planned, strict=True):
        donor_arm = next(x for x in worker.BASELINE_ARMS if x != recipient_arm)
        if (row.get("type") != "patch" or row.get("case") != causal.CASE
                or row.get("recipient_arm") != recipient_arm
                or row.get("donor_arm") != donor_arm
                or row.get("arm") != arm or row.get("hook_count") != 1):
            raise ValueError("causal patch arm matrix differs")
        recipient, donor = baseline[recipient_arm], baseline[donor_arm]
        reconstructed, edit = causal.construct_patch(
            recipient["state"], donor["state"],
            recipient_arm=recipient_arm, arm=arm)
        if row.get("edit") != edit:
            raise ValueError("causal edit tensor hash/norm/seed differs")
        logits = _read_array(output, row["logits"], (1, 286, 32))
        ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
        decoded = processor.batch_decode([ids])[0]
        if row.get("output") != {"argmax_ids": ids, "transcript": decoded}:
            raise ValueError("causal patch complete CTC output differs from logits")
        if arm == "self_W":
            if (row.get("self_logits_exact") is not True
                    or not np.array_equal(logits, recipient["logits"])):
                raise ValueError("causal self logits parity differs")
        elif row.get("self_logits_exact") is not None:
            raise ValueError("causal non-self row claims self parity")
        if arm == "full_donor":
            if (row.get("full_donor_logits_exact") is not True
                    or not np.array_equal(logits, donor["logits"])):
                raise ValueError("causal full-donor logits parity differs")
        elif row.get("full_donor_logits_exact") is not None:
            raise ValueError("causal non-full row claims full parity")
        used_files.add(row["logits"]["path"])
        elapsed = row.get("elapsed_seconds")
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or elapsed < previous_time or elapsed > MAX_SECONDS
                or type(row.get("rss_bytes")) is not int
                or not 0 < row["rss_bytes"] <= MAX_RSS_BYTES):
            raise ValueError("causal patch resource evidence invalid")
        previous_time = elapsed
    if (set(p.name for p in output.iterdir()) != used_files
            or type(complete.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(complete["elapsed_seconds"])
            or not previous_time <= complete["elapsed_seconds"] <= MAX_SECONDS):
        raise ValueError("causal worker files or final elapsed time differ")
    total = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if total > MAX_OUTPUT_BYTES:
        raise ValueError("causal worker output cap exceeded")
    return {"status": "complete_worker_evidence_pass",
            "records_sha256": pilot.sha_file(records),
            "baseline_rows": 2, "patch_rows": 16,
            "output_bytes": total}


def execute(attempt: Path = ATTEMPT) -> dict[str, Any]:
    if attempt.resolve() != ATTEMPT.resolve() or attempt.exists():
        raise ValueError("causal attempt path differs or already exists")
    worker.verify_worker_go()
    verify_parent_go()
    attempt.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    started_mono = time.monotonic()
    errors: list[str] = []
    worker_rc: int | None = None
    timed_out = False
    worker_seconds: float | None = None
    worker_started = False
    worker_audit: dict[str, Any] | None = None
    try:
        base.probe_host(attempt, "pre")
        base.write_once(attempt / "launch_start.json", {
            "schema": "acoustic_fit_noise_causal_parent_start_v1",
            "attempt_path": str(attempt.resolve()), "start_utc": started_utc,
            "start_monotonic": started_mono,
            "source_binding": source_binding(),
            "worker_go_sha256": pilot.sha_file(worker.WORKER_GO),
            "parent_go_sha256": pilot.sha_file(PARENT_GO),
        })
        argv = [sys.executable, str(Path(worker.__file__)), "run",
                "--receipt", str(worker.PREFLIGHT.resolve()),
                "--worker-review", str(worker.WORKER_GO.resolve()),
                "--output", str((attempt / "worker_output").resolve())]
        worker_started = True
        worker_rc, timed_out, worker_seconds = base.run_bounded(
            argv, attempt / "worker_stdout.txt", attempt / "worker_stderr.txt",
            timeout_seconds=MAX_SECONDS)
        if (timed_out or worker_rc != 0 or type(worker_seconds) not in (int, float)
                or not math.isfinite(worker_seconds)
                or not 0 < worker_seconds <= MAX_SECONDS):
            errors.append("worker_failed_or_timed_out")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
    finally:
        if worker_started:
            try:
                base.probe_host(attempt, "post",
                                run_start=datetime.fromisoformat(started_utc))
            except BaseException as exc:
                errors.append(f"post_host_{type(exc).__name__}:{exc}")
        ended_utc = datetime.now(timezone.utc).isoformat()
        ended_mono = time.monotonic()
        wall = (datetime.fromisoformat(ended_utc) -
                datetime.fromisoformat(started_utc)).total_seconds()
        monotonic = ended_mono - started_mono
        if (not 0 < wall or not 0 < monotonic or abs(wall - monotonic) > 5):
            errors.append("utc_monotonic_gap_or_nonpositive")
        try:
            worker_audit = audit_worker_output(attempt / "worker_output")
        except BaseException as exc:
            errors.append(f"worker_audit_{type(exc).__name__}:{exc}")
        try:
            inventory = base.evidence_inventory(attempt)
        except BaseException as exc:
            inventory = {}
            errors.append(f"inventory_{type(exc).__name__}:{exc}")
        total = sum(entry["bytes"] for entry in inventory.values())
        if total > MAX_OUTPUT_BYTES:
            errors.append("total_output_over_128MiB")
        required = {"launch_start.json", "pre_host.json",
                    "pre_power_log.txt.gz", "post_host.json",
                    "post_power_log.txt.gz", "worker_stdout.txt",
                    "worker_stderr.txt", "worker_output/records.jsonl"}
        if (worker_started and (not required.issubset(inventory)
                                or not base.host_sidecars_valid(
                                    attempt, inventory))):
            errors.append("missing_or_unbound_host_worker_evidence")
        result = {
            "schema": "acoustic_fit_noise_causal_parent_result_v1",
            "status": "complete" if not errors else "invalid_or_partial",
            "attempt_path": str(attempt.resolve()), "errors": errors,
            "start_utc": started_utc, "end_utc": ended_utc,
            "start_monotonic": started_mono, "end_monotonic": ended_mono,
            "wall_seconds": wall, "monotonic_seconds": monotonic,
            "worker_started": worker_started,
            "worker_returncode": worker_rc,
            "worker_timed_out": timed_out,
            "worker_elapsed_seconds": worker_seconds,
            "worker_audit": worker_audit,
            "total_output_bytes_before_result": total,
            "source_binding": source_binding(),
            "preflight_sha256": pilot.sha_file(worker.PREFLIGHT),
            "worker_go_sha256": pilot.sha_file(worker.WORKER_GO),
            "parent_go_sha256": pilot.sha_file(PARENT_GO),
            "evidence_sha256": inventory,
        }
        if total + len(pilot.canonical(result).encode()) > MAX_OUTPUT_BYTES:
            result["errors"].append("total_output_including_result_over_128MiB")
            result["status"] = "invalid_or_partial"
        base.write_once(attempt / "launch_result.json", result)
    return result


def audit_saved_attempt(attempt: Path = ATTEMPT) -> dict[str, Any]:
    """Read-only full saved evidence; never loads Wav2Vec2 weights."""
    if attempt.resolve() != ATTEMPT.resolve():
        raise ValueError("causal saved attempt path differs")
    worker.verify_worker_go()
    verify_parent_go()
    result_path = attempt / "launch_result.json"
    result = json.loads(result_path.read_text())
    if (result.get("schema") != "acoustic_fit_noise_causal_parent_result_v1"
            or result.get("attempt_path") != str(attempt.resolve())
            or result.get("source_binding") != source_binding()
            or result.get("preflight_sha256") != pilot.sha_file(worker.PREFLIGHT)
            or result.get("worker_go_sha256") != pilot.sha_file(worker.WORKER_GO)
            or result.get("parent_go_sha256") != pilot.sha_file(PARENT_GO)):
        raise ValueError("causal saved parent identity differs")
    inventory = base.evidence_inventory(attempt, allow_result=True)
    if inventory != result.get("evidence_sha256"):
        raise ValueError("causal saved attempt inventory differs")
    total = sum(item["bytes"] for item in inventory.values())
    if (type(result.get("total_output_bytes_before_result")) is not int
            or total != result["total_output_bytes_before_result"]):
        raise ValueError("causal saved output byte count differs")
    if result.get("status") != "complete":
        return {"status": "bound_invalid_or_partial",
                "result_sha256": pilot.sha_file(result_path)}
    if total > MAX_OUTPUT_BYTES:
        raise ValueError("causal complete attempt exceeds output cap")
    if (result.get("errors") != [] or result.get("worker_started") is not True
            or type(result.get("worker_returncode")) is not int
            or result["worker_returncode"] != 0
            or result.get("worker_timed_out") is not False
            or type(result.get("worker_elapsed_seconds")) not in (int, float)
            or not 0 < result["worker_elapsed_seconds"] <= MAX_SECONDS):
        raise ValueError("causal saved worker outcome invalid")
    start = json.loads((attempt / "launch_start.json").read_text())
    if (start.get("schema") != "acoustic_fit_noise_causal_parent_start_v1"
            or start.get("attempt_path") != result["attempt_path"]
            or start.get("source_binding") != result["source_binding"]
            or start.get("start_utc") != result["start_utc"]
            or start.get("start_monotonic") != result["start_monotonic"]
            or start.get("worker_go_sha256") != result["worker_go_sha256"]
            or start.get("parent_go_sha256") != result["parent_go_sha256"]):
        raise ValueError("causal start/result identity differs")
    if not base.host_sidecars_valid(attempt, inventory):
        raise ValueError("causal host sidecars differ")
    hosts = {label: json.loads((attempt / f"{label}_host.json").read_text())
             for label in ("pre", "post")}
    if not base.valid_timeline(result, hosts["pre"]["collected_utc"],
                               hosts["post"]["collected_utc"]):
        raise ValueError("causal host UTC/monotonic chronology differs")
    start_dt = datetime.fromisoformat(result["start_utc"])
    for label, host in hosts.items():
        outputs = host["outputs"]
        if ("Now drawing from 'AC Power'" not in outputs["power"]
                or plistlib.loads(outputs["clamshell"].encode())[0].get("AppleClamshellState") is not False
                or plistlib.loads(outputs["battery"].encode())[0].get("ExternalConnected") is not True):
            raise ValueError("causal saved AC/lid state differs")
        collected = datetime.fromisoformat(host["collected_utc"])
        with gzip.open(attempt / f"{label}_power_log.txt.gz", "rt") as stream:
            log = stream.read()
        beginning = collected - timedelta(minutes=10) if label == "pre" else start_dt
        base.host_gate._power_log_coverage(log, beginning, collected, collected)
        if base.host_gate._power_events(log, beginning, collected):
            raise ValueError("causal saved sleep/thermal event")
    audited = audit_worker_output(attempt / "worker_output")
    if audited != result.get("worker_audit"):
        raise ValueError("causal saved worker audit differs")
    return {"schema": "acoustic_fit_noise_causal_parent_saved_audit_v1",
            "status": "complete_evidence_pass",
            "result_sha256": pilot.sha_file(result_path),
            "records_sha256": audited["records_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-only", "run", "audit"))
    parser.add_argument("--attempt", type=Path, default=ATTEMPT)
    args = parser.parse_args()
    if args.command == "validate-only":
        print(pilot.canonical({
            "source_binding": source_binding(),
            "worker_go_present": worker.WORKER_GO.is_file(),
            "parent_go_present": PARENT_GO.is_file(),
        }))
    elif args.command == "run":
        print(pilot.canonical(execute(args.attempt)))
    else:
        print(pilot.canonical(audit_saved_attempt(args.attempt)))


if __name__ == "__main__":
    main()
