"""Fixture-specific pre-load admission source; external anchor defaults closed.

No NeMo import or model load occurs here. A future externally reviewed process
launcher must call this before each fresh process and recheck after each arm.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .audio_qa_v2 import check_audio_qa_v2
from .live_parity_source_plan_v1 import (
    ARMS, CASE_ID, VOICE_ID, _check_phase_pair, _input_hashes,
    _preload_receipt, _ready_receipt, _source_gate, canonical_bytes,
    PHASE_IDENTITY_FIELDS,
)
from .two_pass_adapter import read_bound_qa_receipt

HERE = Path(__file__).resolve().parent
SOURCE_FILES = {
    "admission": HERE / "live_parity_production_admission_v1.py",
    "admission_tests": HERE / "test_live_parity_production_admission_v1.py",
    "admission_handoff": HERE / "VOICECHAT_PRODUCTION_ADMISSION_SOURCE_HANDOFF_V1.md",
    "worker": HERE / "live_parity_fixture_worker_v1.py",
    "worker_tests": HERE / "test_live_parity_fixture_worker_v1.py",
    "worker_handoff": HERE / "VOICECHAT_FIXTURE_WORKER_SOURCE_HANDOFF_V1.md",
    "adapter": HERE / "two_pass_adapter.py",
    "qa": HERE / "audio_qa_v2.py",
    "calendar": HERE / "calendar_stub.py",
    "recorder": HERE / "passive_dual_head_recorder.py",
    "official_parity": HERE / "official_adapter_parity.py",
    "capture_codec": HERE / "live_parity_saved_capture_v1.py",
    "source_plan": HERE / "live_parity_source_plan_v1.py",
    "single_process_supervisor": HERE / "live_parity_single_process_supervisor_v1.py",
    "single_process_tests": HERE / "test_live_parity_single_process_supervisor_v1.py",
    "sample_executor": HERE / "public_sample_smoke_v1.py",
    "preflight": HERE.parent / "voicechat_h100_local_preflight.py",
    "protocol": HERE / "VOICECHAT_LIVE_PARITY_ORCHESTRATOR_PROSPECTIVE_V1.md",
}
MAX_RECORD_BYTES = 65_536
CAP_LIMITS = {"wall_seconds": 1_800, "vram_bytes": 80_000 * 1024 * 1024,
              "arm_output_bytes": 1 << 30, "total_output_bytes": 5 << 30}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def file_hash(path: Path, *, cap: int | None = None) -> str:
    require(path.is_file() and not path.is_symlink(), f"Missing or linked file: {path}")
    size = path.stat().st_size
    require(cap is None or size <= cap, "Admission file exceeds cap")
    seen = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(65_536):
            seen += len(block)
            require(cap is None or seen <= cap, "Admission file grew beyond cap")
            digest.update(block)
    require(seen == size and path.stat().st_size == size,
            "Admission file changed while hashing")
    return digest.hexdigest()


def read_record(path: Path) -> tuple[dict[str, Any], bytes]:
    digest = file_hash(path, cap=MAX_RECORD_BYTES)
    with path.open("rb") as stream:
        data = stream.read(MAX_RECORD_BYTES + 1)
    require(len(data) <= MAX_RECORD_BYTES
            and hashlib.sha256(data).hexdigest() == digest,
            "Admission record changed during read")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate admission JSON key")
            value[key] = item
        return value
    value = json.loads(data, object_pairs_hook=unique)
    require(type(value) is dict and canonical_bytes(value) == data,
            "Admission record is not canonical JSON object")
    return value, data


def utc_record(value: Any) -> datetime:
    require(isinstance(value, str), "Missing UTC review time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid UTC review time") from error
    require(parsed.utcoffset() is not None
            and parsed.utcoffset().total_seconds() == 0,
            "Review time is not UTC")
    return parsed


def _closed_anchor(_plan: dict[str, Any], _cohort: dict[str, Any]) -> str:
    raise RuntimeError("Independent external QA/anchor/grant attestation not installed")


def _real_qa(plan: dict[str, Any], paths: dict[str, Path],
             controller_key: bytes) -> dict[str, Any]:
    cohort = check_audio_qa_v2(paths["qa_manifest"], paths["qa_ledger"],
                               controller_key=controller_key,
                               trusted_controller_id=plan["controller_id"])
    require(cohort.get("human_eligible") == 12,
            "Full 12-WAV human cohort absent")
    receipt, _, receipt_hash = read_bound_qa_receipt(
        paths["qa_receipt"], case_id=CASE_ID, wav_path=paths["wav"],
        observed_sha256=plan["input_hashes"]["selected_wav"],
        confirmation_state=None, qa_manifest_path=paths["qa_manifest"],
        qa_ledger_path=paths["qa_ledger"], controller_key=controller_key,
        trusted_controller_id=plan["controller_id"])
    require(receipt.get("voice_id") == VOICE_ID
            and receipt_hash == plan["input_hashes"]["qa_receipt"]
            and cohort["qa_manifest_sha256"] == plan["input_hashes"]["qa_manifest"]
            and cohort["qa_ledger_sha256"] == plan["input_hashes"]["qa_ledger"],
            "Selected QA cohort differs from plan")
    return cohort


def _real_ready_preload(plan: dict[str, Any], paths: dict[str, Path]) -> tuple[dict, dict]:
    kwargs = {"nemo_dir": paths["nemo"], "checkpoint_dir": paths["checkpoint"],
              "nano_home": paths["nano"], "qa_manifest": paths["qa_manifest"],
              "qa_ledger": paths["qa_ledger"],
              "controller_id": plan["controller_id"],
              "gpu_policy": plan["ready"]["reported"]["gpu_policy"]}
    ready = _ready_receipt(**kwargs)
    preload = _preload_receipt(**kwargs)
    _check_phase_pair(ready, preload, plan["input_hashes"])
    return ready, preload


def _paths(plan: dict[str, Any], paths: dict[str, Path]) -> None:
    require(set(paths) == {"nemo", "checkpoint", "nano", "qa_manifest",
                           "qa_ledger", "qa_receipt", "wav", "output"},
            "Fixture path set differs")
    require(all(isinstance(path, Path) and path.is_absolute()
                and path == path.resolve() for path in paths.values()),
            "Fixture paths must be absolute and resolved")
    require(str(paths["nemo"]) == plan["source"]["nemo_dir"]
            and str(paths["checkpoint"]) == plan["source"]["checkpoint_dir"]
            and str(paths["nano"]) == plan["source"]["nano_home"]
            and str(paths["wav"]) == plan["selected"]["wav_path"],
            "Fixture path differs from source plan")


def rehash_bound_inputs(plan: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    """Recompute every source-plan file including system, tools and checkpoint."""
    actual = _input_hashes(nemo_dir=paths["nemo"],
                           checkpoint_dir=paths["checkpoint"],
                           nano_home=paths["nano"],
                           qa_manifest=paths["qa_manifest"],
                           qa_ledger=paths["qa_ledger"],
                           qa_receipt=paths["qa_receipt"], wav_path=paths["wav"])
    require(actual == plan["input_hashes"], "Complete fixture source-plan input changed")
    return actual


def source_hashes() -> dict[str, str]:
    return {role: file_hash(path) for role, path in SOURCE_FILES.items()}


def verify_source_commit(commit: str, expected: dict[str, str]) -> None:
    """Require every reviewed role to exist with identical bytes in one commit."""
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            "Source review commit invalid")
    repo = HERE.parents[2]
    for role, path in SOURCE_FILES.items():
        relative = path.relative_to(repo).as_posix()
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=repo,
            capture_output=True, timeout=15, check=False)
        require(result.returncode == 0
                and hashlib.sha256(result.stdout).hexdigest() == expected[role],
                f"Source review commit lacks exact {role} bytes")


def admit_fixture(
    *, plan_path: Path, freeze_path: Path, review_path: Path,
    paths: dict[str, Path], freeze_review_key: bytes,
    controller_key: bytes,
    anchor_attest: Callable[[dict[str, Any], dict[str, Any]], str] = _closed_anchor,
    qa_verify: Callable[[dict[str, Any], dict[str, Path], bytes], dict[str, Any]] = _real_qa,
    ready_preload: Callable[[dict[str, Any], dict[str, Path]], tuple[dict, dict]] = _real_ready_preload,
    rehash: Callable[[dict[str, Any], dict[str, Path]], dict[str, str]] = rehash_bound_inputs,
    source_digest: Callable[[], dict[str, str]] = source_hashes,
    commit_verify: Callable[[str, dict[str, str]], None] = verify_source_commit,
) -> dict[str, Any]:
    """Admit fixed fixture only after external attestation and exact freeze.

    Injection points are for CPU tests. The eventual CLI must expose none.
    The default external-anchor verifier is closed until controller integration.
    """
    require(len(freeze_review_key) >= 32 and len(controller_key) >= 32,
            "External review/controller keys absent")
    plan, plan_bytes = read_record(plan_path)
    require(plan.get("schema") == "voicechat_live_parity_source_plan_v1"
            and plan.get("status") == "source_plan_only_no_model_work"
            and plan.get("synthetic_test_injection") is False
            and plan.get("selected", {}).get("case_id") == CASE_ID
            and plan["selected"].get("voice_id") == VOICE_ID
            and plan.get("arm_order") == ["official_sample", *ARMS]
            and plan.get("fresh_process_per_arm") is True
            and plan.get("external_tool_execution") is False,
            "Fixture plan is synthetic, incomplete, or altered")
    require(type(plan.get("seed")) is int and 0 <= plan["seed"] < 2**32
            and isinstance(plan.get("input_hashes"), dict),
            "Fixture plan seed or hash map invalid")
    _check_phase_pair(plan["ready"], plan["preload"], plan["input_hashes"])
    _source_gate()
    _paths(plan, paths)
    require(rehash(plan, paths) == plan["input_hashes"],
            "Fixture input rehash differs")
    cohort = qa_verify(plan, paths, controller_key)
    require(cohort.get("human_eligible") == 12,
            "Full 12-WAV human cohort absent")
    anchor_id = anchor_attest(plan, cohort)
    require(isinstance(anchor_id, str) and anchor_id
            and anchor_id == plan.get("external_anchor_audit_id"),
            "External anchor attestation differs")
    current_source = source_digest()
    freeze, freeze_bytes = read_record(freeze_path)
    expected = {"source_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "input_hashes": plan["input_hashes"],
                "source_sha256": current_source,
                "paths": {key: str(path) for key, path in paths.items()},
                "seed": plan["seed"], "arm_order": ["official_sample", *ARMS],
                "external_anchor_audit_id": anchor_id}
    require(freeze.get("schema") == "voicechat_fixture_executable_freeze_v1"
            and freeze.get("status") == "externally_frozen_fixture"
            and isinstance(freeze.get("run_id"), str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}",
                             freeze["run_id"]) is not None
            and freeze["run_id"] == paths["output"].name
            and freeze.get("expected") == expected
            and isinstance(freeze.get("caps"), dict)
            and set(freeze["caps"]) == set(CAP_LIMITS)
            and all(type(value) is int and 0 < value <= CAP_LIMITS[name]
                    for name, value in freeze["caps"].items())
            and isinstance(freeze.get("official_sample_receipt_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}",
                             freeze["official_sample_receipt_sha256"]) is not None,
            "Fixture executable freeze scope or caps differ")
    freeze_time = utc_record(freeze.get("created_utc"))
    source_review = freeze.get("source_review")
    require(type(source_review) is dict and set(source_review) == {"path", "sha256"},
            "Independent fixture source review absent")
    source_path = Path(source_review["path"])
    require(source_path.is_absolute()
            and file_hash(source_path, cap=MAX_RECORD_BYTES) == source_review["sha256"],
            "Independent fixture source review changed")
    source_record, _ = read_record(source_path)
    require(source_record.get("schema") == "voicechat_fixture_source_review_v1"
            and source_record.get("decision") == "GO"
            and source_record.get("source_sha256") == current_source
            and isinstance(source_record.get("reviewer_id"), str)
            and source_record["reviewer_id"]
            and isinstance(source_record.get("source_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", source_record["source_commit"]),
            "Independent fixture source review does not bind current code")
    require(utc_record(source_record.get("reviewed_utc")) < freeze_time,
            "Source review must precede executable freeze")
    commit_verify(source_record["source_commit"], current_source)
    review, _ = read_record(review_path)
    require(review.get("schema") == "voicechat_fixture_freeze_review_v1"
            and review.get("status") == "independently_reviewed_executable_freeze"
            and review.get("freeze_sha256") == hashlib.sha256(freeze_bytes).hexdigest()
            and review.get("source_review_sha256") == source_review["sha256"]
            and isinstance(review.get("reviewer_id"), str)
            and review["reviewer_id"]
            and review["reviewer_id"] != source_record["reviewer_id"]
            and utc_record(review.get("reviewed_utc")) >= freeze_time,
            "Fixture freeze review scope differs")
    unsigned = {key: value for key, value in review.items() if key != "hmac_sha256"}
    expected_mac = hmac.new(freeze_review_key, canonical_bytes(unsigned),
                            hashlib.sha256).hexdigest()
    require(isinstance(review.get("hmac_sha256"), str)
            and hmac.compare_digest(review["hmac_sha256"], expected_mac),
            "Fixture freeze review HMAC differs")
    ready, preload = ready_preload(plan, paths)
    _check_phase_pair(ready, preload, plan["input_hashes"])
    require(all(ready["reported"].get(field) == plan["ready"]["reported"].get(field)
                for field in PHASE_IDENTITY_FIELDS),
            "Current preflight differs from planned source/checkpoint/runtime")
    require(rehash(plan, paths) == plan["input_hashes"]
            and source_digest() == current_source,
            "Fixture bytes changed during admission")
    return {"status": "fixture_admitted_for_fresh_process_source_only",
            "source_plan_sha256": expected["source_plan_sha256"],
            "freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
            "official_sample_receipt_sha256": freeze["official_sample_receipt_sha256"],
            "source_sha256": current_source,
            "input_hashes": plan["input_hashes"],
            "seed": plan["seed"], "caps": dict(freeze["caps"]),
            "paths": paths, "ready": ready, "preload": preload}
