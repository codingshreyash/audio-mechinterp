"""One-shot V2 host and timeout gate for Wav2Vec2 acoustic capture.

This parent loads no candidate weights. The pilot worker is a subprocess only
after independently committed worker and launcher GO receipts pass. A timed-out
worker process group is killed; an attempt directory is never reused.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import plistlib
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import acoustic_fit_noise_pilot_v2 as pilot
import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as host_gate

HERE = Path(__file__).resolve().parent
WORKER = HERE / "acoustic_fit_noise_pilot_v2.py"
PROTOCOL = HERE / "ACOUSTIC_FIT_NOISE_PILOT_PROSPECTIVE_V2.md"
TESTS = HERE / "test_acoustic_fit_noise_parent_launch_v2.py"
HANDOFF = HERE / "ACOUSTIC_FIT_NOISE_PARENT_LAUNCH_SOURCE_HANDOFF_V2.md"
MAX_WORKER_SECONDS = 1800
MAX_CAPTURE_BYTES = 16 * 1024**2
ATTEMPT = HERE / "runs/acoustic_fit_noise_parent_v2/wav2vec2_attempt_002"
PRIOR_ATTEMPT = HERE / "runs/acoustic_fit_noise_parent_v1/wav2vec2_attempt_001"
PRIOR_HASHES = {
    "launch_result.json": "b69468b110a06a55ae8b4b96be367dbc27d608bd1ecfc2b5ecd334547a7d6ebd",
    "worker_output/records.jsonl": "f1752019284dec76d8cd14203b1d8bc63f5b2890a6f140dbb42d3901c732d5cc",
    "worker_stderr.txt": "a23cc760bf3fd9e1b43ca7d9da914d2111bfcf8a6df2ae5256a59973100dd9b5",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
WHISPER_STAGES = ("mel", "post_conv_pos", "post_block0", "post_block1",
                  "post_block2", "post_block3", "post_norm")
WAV_STAGES = ("input_values", "first_conv_raw", "first_conv_norm",
              "final_feature_conv", "feature_projection", "encoder_pre_block0",
              "post_block0", "post_block5", "post_block11", "logits")


def finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(pilot.canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_binding() -> dict:
    return {
        "schema": "acoustic_fit_noise_parent_source_v2",
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
        "attempt_path": str(ATTEMPT.resolve()),
        "prior_failed_attempt_sha256": PRIOR_HASHES,
    }


def verify_prior_failure() -> None:
    for relative, digest in PRIOR_HASHES.items():
        path = PRIOR_ATTEMPT / relative
        if not path.is_file() or path.is_symlink() or pilot.sha_file(path) != digest:
            raise ValueError(f"V1 failed-attempt evidence changed: {relative}")
    result = json.loads((PRIOR_ATTEMPT / "launch_result.json").read_text())
    if (result.get("model") != "wav2vec2" or result.get("status") != "invalid_or_partial"
            or result.get("worker_complete") is not False
            or evidence_inventory(PRIOR_ATTEMPT, allow_result=True) != result.get("evidence_sha256")):
        raise ValueError("V1 prior result was not the preserved failed attempt")


def require_launcher_go(path: Path, binding: dict, preflight_sha: str,
                        worker_review: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing independent launcher GO")
    review = json.loads(path.read_text())
    if (review.get("schema") != "acoustic_fit_noise_parent_source_go_v2"
            or review.get("status") != "GO"
            or review.get("source_binding") != binding
            or review.get("preflight_sha256") != preflight_sha
            or review.get("worker_review_sha256") != pilot.sha_file(worker_review)):
        raise ValueError("independent launcher GO does not bind current source/receipts")
    review_commit = pilot.source_commit(path)
    for source in (binding["launcher_commit"], binding["tests_commit"],
                   binding["handoff_commit"], pilot.source_commit(worker_review)):
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
    elapsed = time.monotonic() - start
    return (process.returncode,
            timed_out or not math.isfinite(elapsed) or not 0 < elapsed <= timeout_seconds,
            elapsed)


def expected_geometry(model: str, key: str, stage: str) -> tuple[list[int], list[int]]:
    engine = key.split("_", 1)[0]
    spec = pilot.EXPECTED[engine]
    frames = spec["frames"]
    start, end = spec["slot"]
    if model == "whisper":
        word_window = {"piper": (90, 161), "kokoro": (101, 184)}[engine]
        if stage == "mel":
            return [1, 80, 3000], [2 * word_window[0], 2 * word_window[1]]
        return [1, 1500, 384], list(word_window)
    first = (frames - 10) // 5 + 1
    final = (frames - 400) // 320 + 1
    if stage == "input_values":
        return [1, frames], [start, end]
    if stage in ("first_conv_raw", "first_conv_norm"):
        return [1, 512, first], [max(0, math.ceil((start - 9) / 5)),
                                min(first, (end - 1) // 5 + 1)]
    window = [max(0, math.ceil((start - 399) / 320)),
              min(final, (end - 1) // 320 + 1)]
    if stage == "final_feature_conv":
        return [1, 512, final], window
    if stage == "logits":
        return [1, final, 32], window
    return [1, final, 768], window


def valid_worker_rows(rows: list[dict], model: str, cases_expected: dict,
                      receipt_sha: str, review_sha: str) -> bool:
    if model != "wav2vec2" or len(rows) != 22:
        return False
    start, complete = rows[0], rows[-1]
    if not (start.get("type") == "start" and start.get("model") == model
            and start.get("preflight_sha256") == receipt_sha
            and start.get("review_sha256") == review_sha
            and complete.get("type") == "complete"
            and complete.get("planned_cases") == 20):
        return False
    planned = [(key, arm) for key in sorted(cases_expected) for arm in pilot.ARMS]
    if len(planned) != 20:
        return False
    stages = WHISPER_STAGES if model == "whisper" else WAV_STAGES
    clean_by_case: dict[str, dict] = {}
    previous_time = -1.0
    for row, (key, arm) in zip(rows[1:-1], planned, strict=True):
        if (row.get("type") != "case" or row.get("model") != model
                or row.get("case") != key or row.get("arm") != arm
                or row.get("pcm_sha256") != cases_expected[key][arm]):
            return False
        elapsed, rss = row.get("elapsed_seconds"), row.get("rss_bytes")
        if (not finite_number(elapsed)
                or not previous_time <= elapsed <= MAX_WORKER_SECONDS
                or type(rss) is not int or not 0 < rss <= pilot.MAX_RSS_BYTES):
            return False
        previous_time = elapsed
        counts, metrics, output = row.get("hook_counts"), row.get("metrics"), row.get("output")
        expected_counts = stages[1:] if model == "whisper" else stages
        if (not isinstance(counts, dict) or set(counts) != set(expected_counts)
                or any(type(v) is not int or v != 1 for v in counts.values())
                or not isinstance(metrics, dict) or set(metrics) != set(stages)
                or not isinstance(output, dict) or not isinstance(output.get("transcript"), str)):
            return False
        if model == "whisper":
            ids = output.get("token_ids")
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 132
                    or any(type(x) is not int or x < 0 for x in ids)
                    or type(output.get("eos_seen")) is not bool
                    or type(output.get("censored")) is not bool
                    or output["censored"] == output["eos_seen"]):
                return False
        else:
            ids = output.get("argmax_ids")
            frame_count = expected_geometry(model, key, "logits")[0][1]
            if (not isinstance(ids, list) or len(ids) != frame_count
                    or any(type(x) is not int or not 0 <= x < 32 for x in ids)):
                return False
        for stage in stages:
            metric = metrics[stage]
            shape, window = expected_geometry(model, key, stage)
            if (not isinstance(metric, dict) or metric.get("shape") != shape
                    or metric.get("nominal_window") != window
                    or not isinstance(metric.get("sha256"), str)
                    or HEX64.fullmatch(metric["sha256"]) is None):
                return False
            values = [metric.get(name) for name in ("whole_l2", "word_l2",
                                                       "outside_l2", "outside_maxabs")]
            if any(not finite_number(v) or v < 0
                   for v in values):
                return False
            whole, word, outside, maxabs = values
            if (abs(whole**2 - word**2 - outside**2) > 1e-5 * max(1, whole**2)
                    or maxabs > outside + 1e-5):
                return False
            if arm in ("clean", "clean_repeat") and any(v != 0 for v in values):
                return False
        if arm == "clean":
            clean_by_case[key] = row
        elif arm == "clean_repeat":
            clean = clean_by_case.get(key)
            if (clean is None or output != clean["output"]
                    or any(metrics[s]["sha256"] != clean["metrics"][s]["sha256"]
                           for s in stages)):
                return False
    finish_time = complete.get("elapsed_seconds")
    return (finite_number(finish_time)
            and previous_time <= finish_time <= MAX_WORKER_SECONDS)


def valid_timeline(result: dict, pre_collected: str, post_collected: str) -> bool:
    names = ("start_monotonic", "end_monotonic", "wall_seconds",
             "monotonic_seconds")
    if any(not finite_number(result.get(name)) for name in names):
        return False
    try:
        start = datetime.fromisoformat(result["start_utc"])
        end = datetime.fromisoformat(result["end_utc"])
        pre = datetime.fromisoformat(pre_collected)
        post = datetime.fromisoformat(post_collected)
        if any(value.tzinfo is None for value in (start, end, pre, post)):
            return False
        wall = (end - start).total_seconds()
        monotonic = result["end_monotonic"] - result["start_monotonic"]
        return (start <= pre < post <= end
                and 0 < wall and 0 < monotonic
                and abs(wall - monotonic) <= 5
                and abs(wall - result["wall_seconds"]) <= 1e-3
                and abs(monotonic - result["monotonic_seconds"]) <= 1e-3)
    except (KeyError, TypeError, ValueError):
        return False


def evidence_inventory(attempt: Path, *, allow_result: bool = False) -> dict[str, dict]:
    inventory = {}
    for path in sorted(attempt.rglob("*")):
        if path.is_symlink():
            raise ValueError("attempt evidence contains a symlink")
        if not path.is_file():
            continue
        relative = str(path.relative_to(attempt))
        if relative == "launch_result.json":
            if allow_result:
                continue
            raise ValueError("result already exists before immutable final write")
        inventory[relative] = {"sha256": pilot.sha_file(path), "bytes": path.stat().st_size}
    return inventory


def host_sidecars_valid(attempt: Path, inventory: dict) -> bool:
    for label in ("pre", "post"):
        host_name = f"{label}_host.json"
        log_name = f"{label}_power_log.txt.gz"
        if host_name not in inventory or log_name not in inventory:
            return False
        try:
            host = json.loads((attempt / host_name).read_text())
            archive = host["power_log_archive"]
            if (archive.get("file") != log_name
                    or archive.get("compressed_sha256") != inventory[log_name]["sha256"]
                    or not isinstance(archive.get("raw_sha256"), str)
                    or HEX64.fullmatch(archive["raw_sha256"]) is None
                    or type(archive.get("raw_bytes")) is not int
                    or archive["raw_bytes"] <= 0):
                return False
            digest = hashlib.sha256()
            raw_bytes = 0
            with gzip.open(attempt / log_name, "rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    raw_bytes += len(block)
                    if raw_bytes > 128 * 1024**2:
                        return False
                    digest.update(block)
            if raw_bytes != archive["raw_bytes"] or digest.hexdigest() != archive["raw_sha256"]:
                return False
        except (ValueError, KeyError, TypeError, OSError, EOFError):
            return False
    return True


def execute(model: str, receipt: Path, worker_review: Path,
            launcher_review: Path, attempt: Path) -> dict:
    if model != "wav2vec2":
        raise ValueError("V2 admits only the failed Wav2Vec2 slot")
    if attempt.resolve() != ATTEMPT.resolve():
        raise ValueError("V2 admits only the frozen attempt_002 path")
    verify_prior_failure()
    if attempt.exists():
        raise FileExistsError("attempt directory already exists; no retry or overwrite")
    current = pilot.preflight()  # no candidate model load
    pilot.require_review(worker_review, receipt, current)
    binding = source_binding()
    require_launcher_go(launcher_review, binding, pilot.sha_file(receipt), worker_review)

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
            "schema": "acoustic_fit_noise_launch_start_v2", "model": model,
            "attempt_path": str(attempt.resolve()),
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
        if (timed_out or worker_seconds is None or not math.isfinite(worker_seconds)
                or not 0 < worker_seconds <= MAX_WORKER_SECONDS):
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
        try:
            inventory = evidence_inventory(attempt)
        except (ValueError, OSError) as exc:
            inventory = {}
            errors.append(f"evidence_inventory_{type(exc).__name__}:{exc}")
        capture_bytes = sum(entry["bytes"] for name, entry in inventory.items()
                            if not name.startswith("worker_output/"))
        if capture_bytes > MAX_CAPTURE_BYTES:
            errors.append("parent_capture_over_16MiB")
        required = {"launch_start.json", "pre_host.json", "pre_power_log.txt.gz",
                    "post_host.json", "post_power_log.txt.gz", "worker_stdout.txt",
                    "worker_stderr.txt", "worker_output/records.jsonl"}
        if worker_started and (not required.issubset(inventory)
                               or not host_sidecars_valid(attempt, inventory)):
            errors.append("missing_or_unbound_host_worker_evidence")
        if inventory.get("worker_output/records.jsonl", {}).get("bytes", 0) > pilot.MAX_OUTPUT_BYTES:
            errors.append("worker_jsonl_over_32MiB")
        worker_output = attempt / "worker_output" / "records.jsonl"
        worker_complete = False
        if worker_output.is_file() and "worker_output/records.jsonl" in inventory:
            try:
                rows = [json.loads(line) for line in worker_output.read_text().splitlines()]
                worker_complete = valid_worker_rows(
                    rows, model, current["cases"], pilot.sha_file(receipt),
                    pilot.sha_file(worker_review))
            except (ValueError, OSError) as exc:
                errors.append(f"worker_record_parse_{type(exc).__name__}")
        if worker_started and not worker_complete:
            errors.append("worker_matrix_incomplete")
        result = {
            "schema": "acoustic_fit_noise_parent_result_v2", "model": model,
            "attempt_path": str(attempt.resolve()), "source_binding": binding,
            "preflight_sha256": pilot.sha_file(receipt),
            "worker_review_sha256": pilot.sha_file(worker_review),
            "launcher_review_sha256": pilot.sha_file(launcher_review),
            "status": "complete" if not errors and worker_complete else "invalid_or_partial",
            "errors": errors, "worker_started": worker_started,
            "worker_returncode": worker_rc, "worker_timed_out": timed_out,
            "worker_elapsed_seconds": worker_seconds, "worker_complete": worker_complete,
            "start_utc": started_utc, "end_utc": ended_utc,
            "start_monotonic": started_mono, "end_monotonic": ended_mono,
            "wall_seconds": wall_seconds, "monotonic_seconds": monotonic_seconds,
            "capture_bytes_before_result": capture_bytes,
            "evidence_sha256": inventory,
        }
        write_once(attempt / "launch_result.json", result)
    return result


def audit_saved_attempt(model: str, receipt: Path, worker_review: Path,
                        launcher_review: Path, attempt: Path) -> dict:
    """Read-only evidence check; never imports or loads candidate checkpoints."""
    if model != "wav2vec2" or attempt.resolve() != ATTEMPT.resolve():
        raise ValueError("V2 saved audit requires frozen Wav2Vec2 attempt_002")
    verify_prior_failure()
    result_path = attempt / "launch_result.json"
    result = json.loads(result_path.read_text())
    current = json.loads(receipt.read_text())
    binding = source_binding()
    pilot.require_review(worker_review, receipt, current)
    require_launcher_go(launcher_review, binding, pilot.sha_file(receipt), worker_review)
    if (result.get("schema") != "acoustic_fit_noise_parent_result_v2"
            or result.get("model") != model
            or result.get("attempt_path") != str(attempt.resolve())
            or result.get("source_binding") != binding
            or result.get("preflight_sha256") != pilot.sha_file(receipt)
            or result.get("worker_review_sha256") != pilot.sha_file(worker_review)
            or result.get("launcher_review_sha256") != pilot.sha_file(launcher_review)):
        raise ValueError("result source/review/attempt identity mismatch")
    actual_inventory = evidence_inventory(attempt, allow_result=True)
    if actual_inventory != result.get("evidence_sha256"):
        raise ValueError("saved evidence SHA-256/size inventory mismatch")
    parent_bytes = sum(entry["bytes"] for name, entry in actual_inventory.items()
                       if not name.startswith("worker_output/"))
    worker_bytes = actual_inventory.get("worker_output/records.jsonl", {}).get("bytes")
    if (type(result.get("capture_bytes_before_result")) is not int
            or result["capture_bytes_before_result"] != parent_bytes):
        raise ValueError("saved parent capture byte count mismatch")
    if result.get("status") == "complete" and (
            parent_bytes > MAX_CAPTURE_BYTES or type(worker_bytes) is not int
            or worker_bytes > pilot.MAX_OUTPUT_BYTES):
        raise ValueError("saved complete attempt exceeds parent/worker byte cap")
    if result.get("status") != "complete":
        return {"schema": "acoustic_fit_noise_parent_audit_v2",
                "status": "bound_invalid_or_partial", "attempt_path": str(attempt.resolve()),
                "result_sha256": pilot.sha_file(result_path)}
    if not host_sidecars_valid(attempt, actual_inventory):
        raise ValueError("host sidecar binding mismatch")
    start = json.loads((attempt / "launch_start.json").read_text())
    if (start.get("schema") != "acoustic_fit_noise_launch_start_v2"
            or start.get("model") != model
            or start.get("attempt_path") != result["attempt_path"]
            or start.get("source_binding") != binding
            or start.get("receipt_sha256") != result["preflight_sha256"]
            or start.get("worker_review_sha256") != result["worker_review_sha256"]
            or start.get("launcher_review_sha256") != result["launcher_review_sha256"]
            or start.get("start_utc") != result["start_utc"]
            or start.get("start_monotonic") != result["start_monotonic"]):
        raise ValueError("launch start and result identity mismatch")
    worker_time = result.get("worker_elapsed_seconds")
    if (result.get("errors") != [] or result.get("worker_started") is not True
            or type(result.get("worker_returncode")) is not int
            or result.get("worker_returncode") != 0
            or result.get("worker_timed_out") is not False
            or not finite_number(worker_time) or not 0 < worker_time <= MAX_WORKER_SECONDS
            or result.get("worker_complete") is not True):
        raise ValueError("completed result has invalid worker outcome")
    rows = [json.loads(line) for line in (attempt / "worker_output/records.jsonl").read_text().splitlines()]
    if not valid_worker_rows(rows, model, current["cases"], result["preflight_sha256"],
                             result["worker_review_sha256"]):
        raise ValueError("saved worker matrix fails complete-row audit")
    host_rows = {label: json.loads((attempt / f"{label}_host.json").read_text())
                 for label in ("pre", "post")}
    if not valid_timeline(result, host_rows["pre"].get("collected_utc"),
                          host_rows["post"].get("collected_utc")):
        raise ValueError("saved UTC/monotonic/host chronology mismatch")
    start_dt = datetime.fromisoformat(result["start_utc"])
    for label in ("pre", "post"):
        host = host_rows[label]
        outputs = host["outputs"]
        if ("Now drawing from 'AC Power'" not in outputs["power"]
                or plistlib.loads(outputs["clamshell"].encode())[0].get("AppleClamshellState") is not False
                or plistlib.loads(outputs["battery"].encode())[0].get("ExternalConnected") is not True):
            raise ValueError("saved host power/lid check failed")
        collected = datetime.fromisoformat(host["collected_utc"])
        with gzip.open(attempt / f"{label}_power_log.txt.gz", "rt") as stream:
            log = stream.read()
        check_start = collected - timedelta(minutes=10) if label == "pre" else start_dt
        host_gate._power_log_coverage(log, check_start, collected, collected)
        if host_gate._power_events(log, check_start, collected):
            raise ValueError("saved host power log has sleep/thermal transition")
    return {"schema": "acoustic_fit_noise_parent_audit_v2", "status": "complete_evidence_pass",
            "attempt_path": str(attempt.resolve()),
            "result_sha256": pilot.sha_file(result_path),
            "worker_jsonl_sha256": actual_inventory["worker_output/records.jsonl"]["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-only", "run", "audit"))
    parser.add_argument("--model", choices=("wav2vec2",))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--worker-review", type=Path)
    parser.add_argument("--launcher-review", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.command == "validate-only":
        verify_prior_failure()
        print(pilot.canonical(source_binding()))
        return
    if not all((args.model, args.receipt, args.worker_review,
                args.launcher_review, args.attempt)):
        parser.error("run/audit requires model, receipt, both reviews and attempt")
    if args.command == "audit":
        result = audit_saved_attempt(args.model, args.receipt,
                                     args.worker_review, args.launcher_review,
                                     args.attempt)
    else:
        result = execute(args.model, args.receipt,
                         args.worker_review, args.launcher_review,
                         args.attempt)
    print(pilot.canonical(result))


if __name__ == "__main__":
    main()
