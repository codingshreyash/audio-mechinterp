"""Source-only VoiceChat parity arm supervisor; no production model entry point.

This executes only explicitly injected synthetic workers. It exercises the
fixed sample/adapter/stock sequence and evidence checks while human QA,
external attestation, executable freeze and independent source review are
pending. A later production worker must be separately implemented/reviewed.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .live_parity_source_plan_v1 import (
    ARMS, CASE_ID, VOICE_ID, SAMPLE_WAV, INJECTION_FIELDS, canonical_bytes,
    fixed_response_bytes,
)
from .official_adapter_parity import compare_stock_adapter

ORDER = ("official_sample", *ARMS)
FIXTURE_ARMS = ARMS
REQUIRED_FIXTURE_ARTIFACTS = frozenset((
    "pass1_raw.bin", "pass2_raw.bin", "pass_calls.bin", "input_tokens.bin",
))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def _evidence_bytes(root: Path, ceiling: int) -> int:
    """Stat all run files before any content read; fail on excess or links."""
    total = 0
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "Evidence link is not allowed")
        if path.is_dir():
            continue
        _require(path.is_file(), "Evidence path is not a regular file")
        total += path.stat().st_size
        _require(total <= ceiling, "Total evidence cap exceeded")
    return total


def _write_bounded(root: Path, path: Path, data: bytes, ceiling: int) -> None:
    _require(len(data) <= ceiling, "Serialized evidence exceeds total cap")
    _require(_evidence_bytes(root, ceiling) + len(data) <= ceiling,
             "Total evidence cap exceeded before write")
    _write_exclusive(path, data)


def _bounded_file_sha(path: Path, root: Path, ceiling: int) -> str:
    """Check aggregate/individual sizes before a bounded streaming digest."""
    _evidence_bytes(root, ceiling)
    _require(path.is_file() and not path.is_symlink(),
             "Frozen response is not a regular file")
    size = path.stat().st_size
    _require(size <= ceiling, "Frozen response exceeds total cap before read")
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as stream:
        while block := stream.read(min(65536, ceiling - read + 1)):
            read += len(block)
            _require(read <= ceiling, "Frozen response grew beyond total cap")
            digest.update(block)
    _require(read == size and path.stat().st_size == size,
             "Frozen response changed while hashing")
    _evidence_bytes(root, ceiling)
    return digest.hexdigest()


def _artifact_hashes(directory: Path, ceiling: int) -> tuple[dict[str, str], int]:
    """Stat before reading, then stream-hash regular files in one arm."""
    result: dict[str, str] = {}
    total = 0
    for path in sorted(directory.iterdir()):
        _require(path.is_file() and not path.is_symlink(),
                 "Arm artifact is not a regular file")
        size = path.stat().st_size
        total += size
        _require(total <= ceiling, "Arm artifact exceeds output cap before read")
        digest = hashlib.sha256()
        read = 0
        with path.open("rb") as stream:
            while block := stream.read(min(65536, ceiling - total + size - read + 1)):
                read += len(block)
                _require(total - size + read <= ceiling,
                         "Arm artifact grew beyond output cap")
                digest.update(block)
        _require(read == size and path.stat().st_size == size,
                 "Arm artifact changed while hashing")
        result[path.name] = digest.hexdigest()
    return result, total


def _validate_synthetic_plan(plan: dict[str, Any]) -> bytes:
    _require(plan.get("schema") == "voicechat_live_parity_source_plan_v1"
             and plan.get("status") == "synthetic_test_only_no_model_work"
             and plan.get("synthetic_test_injection") is True,
             "Source-only launcher requires an explicitly synthetic plan")
    selected = plan.get("selected", {})
    _require(selected.get("case_id") == CASE_ID and selected.get("voice_id") == VOICE_ID,
             "Fixed case-B Piper selection differs")
    _require(plan.get("arm_order") == list(ORDER)
             and plan.get("fresh_process_per_arm") is True
             and plan.get("external_tool_execution") is False,
             "Frozen VoiceChat arm order or tool boundary differs")
    _require(type(plan.get("seed")) is int and 0 <= plan["seed"] < 2**32,
             "Fixed VoiceChat seed invalid")
    hashes = plan.get("input_hashes")
    _require(isinstance(hashes, dict) and bool(hashes)
             and hashes.get("selected_wav") == selected.get("wav_sha256")
             and f"nemo:{SAMPLE_WAV}" in hashes
             and "checkpoint_weight" in hashes and "checkpoint_config" in hashes,
             "Bound plan inputs incomplete")
    comparisons = plan.get("required_comparisons", {})
    _require(comparisons.get("exact_discrete_and_replay") is True
             and comparisons.get("stock_repeat_required") is True
             and comparisons.get("tts_no_wrapper_control") is True
             and comparisons.get("raw_unspliced_pcm_max_abs") == 1e-4,
             "Predeclared parity comparison differs")
    return canonical_bytes(plan)


def run_synthetic_protocol(
    *, plan: dict[str, Any], output_dir: Path,
    worker: Callable[[str, Path, Path | None], dict[str, Any]],
    rehash_inputs: Callable[[], dict[str, str]],
    caps: dict[str, int],
    compare: Callable[..., dict[str, Any]] = compare_stock_adapter,
) -> dict[str, Any]:
    """Exercise fixed orchestration with fake workers; never authorize live use.

    The worker receives one exclusive arm directory and, for stock arms, the
    immutable response file. It must return fake raw captures and a provenance
    receipt. It is called serially. There is no CLI or model-loading code.
    """
    plan_bytes = _validate_synthetic_plan(plan)
    frozen_plan = json.loads(plan_bytes)
    expected_hashes = frozen_plan["input_hashes"]
    frozen_caps = dict(caps)
    _require(set(frozen_caps) == {"wall_seconds", "vram_bytes", "output_bytes",
                                  "total_output_bytes"}
             and all(type(value) is int and value > 0 for value in frozen_caps.values()),
             "Prospective per-arm or total caps invalid")
    total_cap = frozen_caps["total_output_bytes"]
    _require(len(plan_bytes) <= total_cap,
             "Serialized source plan exceeds total evidence cap")
    _require(rehash_inputs() == expected_hashes, "Bound input changed before sample")
    scope_bytes = canonical_bytes({
        "status": "synthetic_test_only_no_model_work",
        "source_plan_sha256": _sha(plan_bytes), "caps": frozen_caps,
        "arm_order": list(ORDER), "created_utc": _now(),
    })
    _require(len(plan_bytes) + len(scope_bytes) <= total_cap,
             "Source plan and scope exceed total evidence cap")
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "source_plan_exact.json"
    _write_bounded(output_dir, plan_path, plan_bytes, total_cap)
    _write_bounded(output_dir, output_dir / "synthetic_scope.json",
                   scope_bytes, total_cap)
    response_path = output_dir / "api_response.json"
    response_sha: str | None = None
    no_call: bool | None = None
    prompt_sha: str | None = None
    process_ids: set[str] = set()
    captures: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}

    for arm in ORDER:
        _require(rehash_inputs() == expected_hashes,
                 f"Bound input changed before {arm}")
        if arm in FIXTURE_ARMS and arm != "adapter":
            _require(response_sha is not None and response_path.is_file()
                     and _bounded_file_sha(response_path, output_dir, total_cap) == response_sha,
                     "Stock response absent or changed")
        arm_dir = output_dir / arm
        arm_dir.mkdir(exist_ok=False)
        started = _now()
        start = time.monotonic()
        _write_bounded(output_dir, arm_dir / "started.json", canonical_bytes({
            "arm": arm, "started_utc": started,
            "source_plan_sha256": _sha(plan_bytes),
            "response_sha256": response_sha if arm.startswith("stock_") else None,
        }), total_cap)
        try:
            result = worker(arm, arm_dir, response_path if arm.startswith("stock_") else None)
            elapsed = time.monotonic() - start
            _require(isinstance(result, dict) and result.get("returncode") == 0,
                     f"{arm} worker failed")
            _require(elapsed <= frozen_caps["wall_seconds"]
                     and type(result.get("peak_vram_bytes")) is int
                     and 0 <= result["peak_vram_bytes"] <= frozen_caps["vram_bytes"],
                     f"{arm} wall or VRAM cap exceeded")
            provenance = result.get("provenance", {})
            expected = {"source_plan_sha256": _sha(plan_bytes),
                        "input_hashes": expected_hashes,
                        "seed": frozen_plan["seed"],
                        "checkpoint_weight_sha256": expected_hashes["checkpoint_weight"],
                        "checkpoint_config_sha256": expected_hashes["checkpoint_config"]}
            _require(isinstance(provenance, dict)
                     and all(provenance.get(key) == value for key, value in expected.items()),
                     f"{arm} source/checkpoint/seed provenance differs")
            pid = provenance.get("process_id")
            _require(isinstance(pid, str) and pid and pid not in process_ids,
                     f"{arm} did not report a fresh process")
            process_ids.add(pid)
            if arm != "official_sample":
                _require(provenance.get("wav_sha256") == expected_hashes["selected_wav"],
                         f"{arm} fixture WAV differs")
                this_prompt = provenance.get("prompt_token_sha256")
                _require(isinstance(this_prompt, str) and len(this_prompt) == 64,
                         f"{arm} prompt token hash absent")
                prompt_sha = prompt_sha or this_prompt
                _require(this_prompt == prompt_sha, f"{arm} prompt token hash differs")
                if arm.startswith("stock_"):
                    _require(provenance.get("response_sha256") == response_sha,
                             f"{arm} injected response differs")
            _evidence_bytes(output_dir, total_cap)
            hashes, output_bytes = _artifact_hashes(arm_dir, frozen_caps["output_bytes"])
            required = ({"sample_raw.bin"} if arm == "official_sample"
                        else REQUIRED_FIXTURE_ARTIFACTS)
            _require(required <= hashes.keys(), f"{arm} raw artifact missing")
            _require(output_bytes <= frozen_caps["output_bytes"], f"{arm} output cap exceeded")
            capture = result.get("capture")
            _require(isinstance(capture, dict), f"{arm} capture missing")
            if arm == "adapter":
                trace, calls = capture.get("trace"), capture.get("calls")
                _require(isinstance(trace, dict) and isinstance(calls, list),
                         "Adapter raw trace or calls missing")
                _require(all(isinstance(trace.get(phase), dict)
                             and "raw_result" in trace[phase]
                             for phase in ("pass1", "pass2")),
                         "Adapter pass capture missing")
                response_bytes, no_call = fixed_response_bytes(trace, calls)
                _write_bounded(output_dir, response_path, response_bytes, total_cap)
                response_sha = _sha(response_bytes)
            elif arm.startswith("stock_"):
                stock = capture.get("stock")
                _require(isinstance(stock, dict)
                         and all(isinstance(stock.get(phase), dict)
                                 and "result" in stock[phase]
                                 and "kwargs" in stock[phase]
                                 for phase in ("pass1", "pass2")),
                         f"{arm} stock capture missing")
                _require(all(isinstance(stock[phase]["kwargs"], dict)
                             for phase in ("pass1", "pass2")),
                         f"{arm} stock argument capture missing")
                first_injection = {key for key in INJECTION_FIELDS
                                   if key in stock["pass1"]["kwargs"]}
                second_injection = {key for key in INJECTION_FIELDS
                                    if key in stock["pass2"]["kwargs"]}
                _require(not first_injection and second_injection ==
                         (set() if no_call else set(INJECTION_FIELDS)),
                         f"{arm} replay injection differs")
                _require(_bounded_file_sha(response_path, output_dir, total_cap) == response_sha,
                         f"{arm} changed the frozen response")
            captures[arm] = capture
            receipts[arm] = {"arm": arm, "returncode": 0,
                             "started_utc": started, "finished_utc": _now(),
                             "elapsed_monotonic_seconds": elapsed,
                             "peak_vram_bytes": result["peak_vram_bytes"],
                             "artifact_sha256": hashes,
                             "response_sha256": provenance.get("response_sha256")}
            _write_bounded(output_dir, arm_dir / "complete.json",
                           canonical_bytes(receipts[arm]), total_cap)
            _, final_bytes = _artifact_hashes(arm_dir, frozen_caps["output_bytes"])
            _require(final_bytes <= frozen_caps["output_bytes"],
                     f"{arm} output cap exceeded after receipt")
        except BaseException as error:
            try:
                _write_bounded(output_dir, arm_dir / "failure.json", canonical_bytes({
                    "arm": arm, "status": "synthetic_arm_failed_no_retry",
                    "reason": str(error), "finished_utc": _now(),
                }), total_cap)
            except (ValueError, FileExistsError):
                # A worker may already have filled the cap. Its exclusive arm
                # directory and partial bytes still consume this run ID.
                pass
            raise
    _require(rehash_inputs() == expected_hashes,
             "Bound input changed before comparison")
    _require(response_sha is not None
             and _bounded_file_sha(response_path, output_dir, total_cap) == response_sha,
             "Frozen stock response changed before comparison")
    comparison = compare(captures["stock_a"]["stock"],
                         captures["stock_b"]["stock"],
                         captures["stock_no_tts"]["stock"],
                         captures["adapter"]["trace"],
                         captures["adapter"]["calls"], waveform_tolerance=1e-4)
    _require(comparison.get("status") ==
             "source_comparison_passed_pending_live_provenance_review",
             "Parity comparator did not pass")
    summary = {"status": "synthetic_test_only_no_model_work",
               "source_plan_sha256": _sha(plan_bytes),
               "response_sha256": response_sha, "no_call": no_call,
               "prompt_token_sha256": prompt_sha,
               "arm_order": list(ORDER), "arm_receipts": receipts,
               "comparison": comparison,
               "external_tool_execution": False}
    _write_bounded(output_dir, output_dir / "synthetic_comparison.json",
                   canonical_bytes(summary), total_cap)
    return summary
