"""One-attempt, one-fresh-process adapter observation for fixed VoiceChat B.

The default external-anchor admission is closed. This source does not run the
official sample or three stock arms and cannot claim parity. A child starts
only after parent admission; the child repeats admission before model load.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .live_parity_production_admission_v1 import (
    admit_fixture, file_hash, read_record, rehash_bound_inputs, source_hashes,
)
from .live_parity_saved_capture_v1 import load_capture
from .live_parity_source_plan_v1 import canonical_bytes, fixed_response_bytes
from .public_sample_smoke_v1 import evidence_inventory, execute_official


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _closed_sample(_admission: dict[str, Any]) -> str:
    raise RuntimeError("Reviewed official sample receipt binding not installed")


def fixed_command(plan_path: Path, freeze_path: Path, review_path: Path,
                  paths: dict[str, Path]) -> list[str]:
    # The pinned preflight sets PYTHONPATH to NeMo only, and the bounded
    # executor uses the run directory as cwd. Add this reviewed checkout to
    # sys.path explicitly so the fixed local worker is importable.
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import sys; sys.path.insert(0, " + repr(str(repo_root)) + "); "
        "from models.second_architecture.voicechat_calendar_fixture_v1."
        "live_parity_fixture_worker_v1 import main; main()"
    )
    command = [sys.executable, "-c", bootstrap]
    for name, path in (("plan", plan_path), ("freeze", freeze_path),
                       ("freeze-review", review_path),
                       *((name.replace("_", "-"), value) for name, value in paths.items())):
        command.extend(("--" + name, str(path)))
    return command


def launch_adapter_once(
    *, plan_path: Path, freeze_path: Path, review_path: Path,
    paths: dict[str, Path], freeze_review_key: bytes, controller_key: bytes,
    admit: Callable[..., dict[str, Any]] = admit_fixture,
    sample_verify: Callable[[dict[str, Any]], str] = _closed_sample,
    execute: Callable[..., dict[str, Any]] = execute_official,
    rehash: Callable[..., dict[str, str]] = rehash_bound_inputs,
    source_digest: Callable[[], dict[str, str]] = source_hashes,
) -> dict[str, Any]:
    """Reserve one run ID, start one bounded child, derive response from disk.

    Injection points exist for CPU tests only. A production CLI must pass no
    replacements and must receive external anchor attestation separately.
    """
    admission = admit(plan_path=plan_path, freeze_path=freeze_path,
                      review_path=review_path, paths=paths,
                      freeze_review_key=freeze_review_key,
                      controller_key=controller_key)
    require(admission.get("status") == "fixture_admitted_for_fresh_process_source_only",
            "One-process parent admission incomplete")
    sample_sha = sample_verify(admission)
    require(isinstance(sample_sha, str) and len(sample_sha) == 64
            and sample_sha == admission.get("official_sample_receipt_sha256"),
            "Reviewed official sample receipt hash absent")
    caps = dict(admission["caps"])
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=False)
    scope = {"schema": "voicechat_adapter_single_process_scope_v1",
             "status": "run_id_consumed", "source_plan_sha256": admission["source_plan_sha256"],
             "freeze_sha256": admission["freeze_sha256"], "seed": admission["seed"],
             "official_sample_receipt_sha256": sample_sha,
             "input_hashes": admission["input_hashes"],
             "source_sha256": admission["source_sha256"], "caps": caps,
             "arm_order": ["official_sample", "adapter", "stock_a", "stock_b",
                           "stock_no_tts"], "completed_arms": []}
    scope_bytes = canonical_bytes(scope)
    require(len(scope_bytes) <= caps["total_output_bytes"], "Scope exceeds evidence cap")
    (output / "scope.json").write_bytes(scope_bytes)
    try:
        require(rehash(read_record(plan_path)[0], paths) == admission["input_hashes"]
                and source_digest() == admission["source_sha256"],
                "Fixture changed before child process")
        command = fixed_command(plan_path, freeze_path, review_path, paths)
        environment = os.environ.copy()
        child_caps = {"wall_seconds": caps["wall_seconds"],
                      "vram_mib": caps["vram_bytes"] // (1024 * 1024),
                      "total_evidence_bytes": caps["total_output_bytes"]}
        require(child_caps["vram_mib"] > 0, "VRAM cap below one MiB")
        run = execute(command, output, child_caps, environment)
        require(run.get("returncode") == 0 and run.get("cap_reason") is None
                and run.get("logs_complete") is True,
                "Adapter child failed or exceeded cap")
        require(type(run.get("process_id")) is int and run["process_id"] > 0
                and type(run.get("peak_vram_mib")) is int
                and run["peak_vram_mib"] * 1024 * 1024 <= caps["vram_bytes"]
                and run.get("elapsed_monotonic_seconds", float("inf")) <= caps["wall_seconds"],
                "Adapter child process/resource receipt invalid")
        inventory, total = evidence_inventory(output, caps["total_output_bytes"])
        arm_total = sum(row["bytes"] for name, row in inventory.items()
                        if name.startswith("adapter/"))
        require(arm_total <= caps["arm_output_bytes"],
                "Adapter nested output exceeds arm cap")
        result, _ = read_record(output / "adapter" / "arm_result.json")
        require(result.get("status") == "fixture_admitted_for_fresh_process_source_only"
                and result.get("arm") == "adapter"
                and result.get("process_id") == run["process_id"]
                and result.get("seed") == admission["seed"]
                and result.get("source_plan_sha256") == admission["source_plan_sha256"]
                and result.get("freeze_sha256") == admission["freeze_sha256"]
                and result.get("input_hashes") == admission["input_hashes"],
                "Adapter child result/provenance differs")
        capture = result.get("capture")
        require(isinstance(capture, dict), "Adapter saved capture receipt absent")
        require((output / "adapter" / "raw_capture").is_dir()
                and not (output / "adapter" / "raw_capture").is_symlink(),
                "Adapter saved capture directory absent or linked")
        saved = load_capture(output / "adapter" / "raw_capture",
                             byte_cap=caps["arm_output_bytes"],
                             expected_manifest_sha256=capture["manifest_sha256"],
                             trusted_root=output)
        require(saved.get("arm") == "adapter", "Adapter saved arm differs")
        response_bytes, no_call = fixed_response_bytes(saved["trace"], saved["calls"])
        require(total + len(response_bytes) <= caps["total_output_bytes"],
                "Frozen response exceeds total evidence cap")
        response_path = output / "api_response.json"
        with response_path.open("xb") as stream:
            stream.write(response_bytes)
        response_sha = hashlib.sha256(response_bytes).hexdigest()
        require(file_hash(response_path, cap=caps["total_output_bytes"]) == response_sha,
                "Frozen stock response changed")
        require(rehash(read_record(plan_path)[0], paths) == admission["input_hashes"]
                and source_digest() == admission["source_sha256"],
                "Fixture changed after child process")
        inventory, _ = evidence_inventory(output, caps["total_output_bytes"])
        receipt = {"status": "adapter_capture_only_no_parity_claim",
                   "source_plan_sha256": admission["source_plan_sha256"],
                   "freeze_sha256": admission["freeze_sha256"],
                   "official_sample_receipt_sha256": sample_sha,
                   "process_id": run["process_id"],
                   "capture_manifest_sha256": capture["manifest_sha256"],
                   "response_sha256": response_sha,
                   "no_call": no_call, "artifact_sha256": inventory}
        receipt_bytes = canonical_bytes(receipt)
        require(sum(row["bytes"] for row in inventory.values()) + len(receipt_bytes)
                <= caps["total_output_bytes"], "Adapter receipt exceeds evidence cap")
        with (output / "adapter_capture_receipt.json").open("xb") as stream:
            stream.write(receipt_bytes)
        return receipt
    except BaseException:
        # The exclusive output directory remains consumed, including partial
        # evidence. No automatic retry occurs under this run ID.
        raise
