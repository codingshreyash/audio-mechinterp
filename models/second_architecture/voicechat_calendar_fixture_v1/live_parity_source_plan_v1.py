"""Build a hash-bound VoiceChat parity plan without importing or running a model.

This is a source-only admission artifact. It does not execute the official
sample or any of the four fixture arms. A later launcher requires separate
source and external-freeze reviews and must recheck all bound bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .audio_qa_v2 import RENDER, check_audio_qa_v2
from .two_pass_adapter import read_bound_qa_receipt

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOICECHAT_LIVE_PARITY_ORCHESTRATOR_PROSPECTIVE_V1.md"
CONCEPT_REVIEW = HERE / "VOICECHAT_LIVE_PARITY_ORCHESTRATOR_CONCEPTUAL_REREVIEW_V2.md"
PREFLIGHT = HERE.parent / "voicechat_h100_local_preflight.py"
PROTOCOL_SHA256 = "cccf011664a17382e23edfa390ea97ce56f28102d2e0970a74fb5f471bc880f9"
CONCEPT_REVIEW_SHA256 = "c67ea8419e2ae762375cfc43667482ffb59ff03fb4a2fdc8889d637a441d3c48"
CONCEPT_GO = "Decision: GO for implementing the VoiceChat live parity orchestrator source only; live model work remains on HOLD until a separate source review and external freeze."
CASE_ID = "B"
VOICE_ID = "piper_amy_en_us"
ARMS = ("adapter", "stock_a", "stock_b", "stock_no_tts")
INJECTION_FIELDS = ("function_calls", "function_call_lengths", "function_call_steps",
                    "function_responses", "function_response_lengths", "function_response_steps")
NO_CALL_BYTES = b'{"tool_name":"list_calendar_events","response":{"status":"no_call","executed":false}}\n'
NO_CALL_SHA256 = "a1e210ab7434dca112d7fa273ba6de5ee42886f532bc1b1aa39416b0146c9212"
SAMPLE_SCRIPT = "examples/speechlm2/offline_voicechat_fc_infer.py"
SAMPLE_WAV = "examples/speechlm2/sample_audio/sample_fc.wav"
SAMPLE_RESPONSE = "examples/speechlm2/function_calling/random_number_response.json"
TEMPLATE = "examples/speechlm2/function_calling/template.jinja"
HELPER = "nemo/collections/speechlm2/inference/utils/offline_voicechat.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")


def fixed_response_bytes(trace: dict[str, Any],
                         adapter_calls: list[dict[str, Any]]) -> tuple[bytes, bool]:
    """Pure prospective response rule; never inspect output quality to select it."""
    if (trace.get("external_tool_execution") is not False or "replay" not in trace or
            len(adapter_calls) != 2 or
            adapter_calls[0].get("decode_audio") is not False or
            adapter_calls[1].get("decode_audio") is not True or
            any(field in adapter_calls[0] for field in INJECTION_FIELDS)):
        raise ValueError("Adapter capture or pass order differs")
    replay = trace.get("replay")
    if replay is None:
        if any(field in adapter_calls[1] for field in INJECTION_FIELDS):
            raise ValueError("No-call adapter injected function tensors")
        if hashlib.sha256(NO_CALL_BYTES).hexdigest() != NO_CALL_SHA256:
            raise RuntimeError("Frozen no-call placeholder bytes changed")
        return NO_CALL_BYTES, True
    if any(field not in adapter_calls[1] for field in INJECTION_FIELDS):
        raise ValueError("Parsed call lacks complete second-pass injection")
    selector = replay["response_selector"]
    value = {"tool_name": selector["tool_name"],
             "response": selector["response"]}
    if not isinstance(value["tool_name"], str) or not value["tool_name"]:
        raise ValueError("Selected model call lacks an official response tool name")
    return json.dumps(value, ensure_ascii=True).encode("ascii") + b"\n", False


def _source_gate() -> None:
    if (sha256(PROTOCOL) != PROTOCOL_SHA256 or
            sha256(CONCEPT_REVIEW) != CONCEPT_REVIEW_SHA256 or
            CONCEPT_GO not in CONCEPT_REVIEW.read_text()):
        raise ValueError("Reviewed VoiceChat parity protocol changed")
    if hashlib.sha256(NO_CALL_BYTES).hexdigest() != NO_CALL_SHA256:
        raise ValueError("Frozen no-call response changed")


def _controller_key() -> bytes:
    try:
        key = bytes.fromhex(os.environ.get("VOICECHAT_CONTROLLER_KEY_HEX", ""))
    except ValueError as error:
        raise ValueError("Invalid external controller key encoding") from error
    if len(key) < 32:
        raise ValueError("External controller key is missing or too short")
    return key


def _preload_receipt(*, nemo_dir: Path, checkpoint_dir: Path,
                     nano_home: Path, qa_manifest: Path, qa_ledger: Path,
                     controller_id: str, gpu_policy: str) -> dict[str, Any]:
    """Existing preflight performs source/checkpoint/Nano/GPU reads, no model load."""
    command = [sys.executable, str(PREFLIGHT), "--phase", "preload",
               "--nemo-dir", str(nemo_dir), "--checkpoint-dir", str(checkpoint_dir),
               "--qa-v2-manifest", str(qa_manifest), "--qa-v2-ledger", str(qa_ledger),
               "--trusted-controller-id", controller_id,
               "--nano-hf-home", str(nano_home), "--gpu-policy", gpu_policy]
    done = subprocess.run(command, capture_output=True, timeout=300)
    if done.returncode != 0:
        raise ValueError("Pinned VoiceChat preload preflight failed")
    try:
        receipt = json.loads(done.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Pinned preload returned invalid JSON") from error
    if receipt.get("status") != "local_preload_preflight_passed_no_model_or_network":
        raise ValueError("Pinned preload did not certify the staged source")
    return {"command": command, "stdout_sha256": hashlib.sha256(done.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(done.stderr).hexdigest(),
            "stdout_utf8": done.stdout.decode("utf-8"),
            "stderr_utf8": done.stderr.decode("utf-8"),
            "reported": receipt}


def build_source_plan(*, nemo_dir: Path, checkpoint_dir: Path,
                      nano_home: Path, qa_manifest: Path, qa_ledger: Path,
                      qa_receipt: Path, wav_path: Path, controller_id: str,
                      gpu_policy: str, seed: int,
                      qa_checker: Callable[..., dict[str, Any]] = check_audio_qa_v2,
                      preload: Callable[..., dict[str, Any]] = _preload_receipt,
                      source_gate: Callable[[], None] = _source_gate) -> dict[str, Any]:
    """All read-only source/QA gates complete before emitting a future plan.

    The injectable checkers exist only for CPU fake tests. The CLI passes no
    replacement checker, so it always invokes the real v2 gate and preflight.
    """
    synthetic_test_injection = (qa_checker is not check_audio_qa_v2 or
                                preload is not _preload_receipt or
                                source_gate is not _source_gate)
    if gpu_policy not in ("h100", "a100-80gb") or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("GPU policy or fixed seed invalid")
    if not controller_id or not controller_id.isascii():
        raise ValueError("External controller ID missing")
    source_gate()
    key = _controller_key()
    cohort = qa_checker(qa_manifest, qa_ledger, controller_key=key,
                        trusted_controller_id=controller_id)
    if cohort.get("human_eligible") != 12 or not cohort.get("qa_manifest_sha256") or \
            not cohort.get("qa_ledger_sha256"):
        raise ValueError("Full 12-WAV human QA cohort absent")
    render = json.loads(RENDER.read_bytes())
    matches = [row for row in render["rows"]
               if row["case_id"] == CASE_ID and row["voice_id"] == VOICE_ID]
    if len(matches) != 1 or matches[0]["final_sha256"] != sha256(wav_path):
        raise ValueError("Fixed case-B Piper WAV differs from rendered cohort")
    receipt, exact_receipt, receipt_hash = read_bound_qa_receipt(
        qa_receipt, case_id=CASE_ID, wav_path=wav_path,
        observed_sha256=matches[0]["final_sha256"], confirmation_state=None,
        qa_manifest_path=qa_manifest, qa_ledger_path=qa_ledger,
        controller_key=key, trusted_controller_id=controller_id,
    )
    if receipt.get("voice_id") != VOICE_ID or receipt_hash != sha256(qa_receipt):
        raise ValueError("Fixed case-B Piper QA receipt differs")
    for relative in (SAMPLE_SCRIPT, SAMPLE_WAV, SAMPLE_RESPONSE, TEMPLATE, HELPER):
        if not (nemo_dir / relative).is_file():
            raise ValueError(f"Pinned NVIDIA sample/source file absent: {relative}")
    preflight = preload(nemo_dir=nemo_dir, checkpoint_dir=checkpoint_dir,
                        nano_home=nano_home, qa_manifest=qa_manifest,
                        qa_ledger=qa_ledger, controller_id=controller_id,
                        gpu_policy=gpu_policy)
    if preflight.get("reported", {}).get("status") != \
            "local_preload_preflight_passed_no_model_or_network":
        raise ValueError("Pinned preload receipt differs")
    return {
        "schema": "voicechat_live_parity_source_plan_v1",
        "status": ("synthetic_test_only_no_model_work" if synthetic_test_injection else
                   "source_plan_only_no_model_work"),
        "synthetic_test_injection": synthetic_test_injection,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": PROTOCOL_SHA256,
        "concept_review_sha256": CONCEPT_REVIEW_SHA256,
        "selected": {"case_id": CASE_ID, "voice_id": VOICE_ID,
                     "wav_path": str(wav_path.resolve()),
                     "wav_sha256": sha256(wav_path),
                     "qa_receipt_sha256": hashlib.sha256(exact_receipt).hexdigest()},
        "qa_manifest_sha256": cohort["qa_manifest_sha256"],
        "qa_ledger_sha256": cohort["qa_ledger_sha256"],
        "external_anchor_audit_id": json.loads(qa_ledger.read_bytes())["external_anchor_audit_id"],
        "controller_id": controller_id,
        "source": {"nemo_dir": str(nemo_dir.resolve()),
                   "checkpoint_dir": str(checkpoint_dir.resolve()),
                   "nano_home": str(nano_home.resolve()),
                   "sample_script_sha256": sha256(nemo_dir / SAMPLE_SCRIPT),
                   "sample_wav_sha256": sha256(nemo_dir / SAMPLE_WAV),
                   "sample_response_sha256": sha256(nemo_dir / SAMPLE_RESPONSE),
                   "official_helper_sha256": sha256(nemo_dir / HELPER),
                   "template_sha256": sha256(nemo_dir / TEMPLATE)},
        "preload": preflight,
        "seed": seed,
        "arm_order": ["official_sample", *ARMS],
        "fresh_process_per_arm": True,
        "response_rule": {"discovery_arm": "adapter",
                          "no_call_placeholder_sha256": NO_CALL_SHA256,
                          "call_branch": "default_json_dumps_insertion_order_lf",
                          "stock_response": "hash_after_adapter_before_stock_a"},
        "required_comparisons": {"exact_discrete_and_replay": True,
                                 "raw_unspliced_pcm_max_abs": 1e-4,
                                 "stock_repeat_required": True,
                                 "tts_no_wrapper_control": True},
        "external_tool_execution": False,
        "unimplemented_live_gates": [
            "source_review_and_external_executable_freeze",
            "unchanged_official_sample_success_and_raw_capture",
            "four_fresh_processes_with_identical_model_rng_state",
            "bounded_wall_gpu_memory_and_output",
            "saved_raw_stock_and_adapter_arm_hashes",
            "independent_real_model_parity_review",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--nano-home", required=True, type=Path)
    parser.add_argument("--qa-manifest", required=True, type=Path)
    parser.add_argument("--qa-ledger", required=True, type=Path)
    parser.add_argument("--qa-receipt", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--gpu-policy", choices=("h100", "a100-80gb"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build_source_plan(
        nemo_dir=args.nemo_dir, checkpoint_dir=args.checkpoint_dir,
        nano_home=args.nano_home, qa_manifest=args.qa_manifest,
        qa_ledger=args.qa_ledger, qa_receipt=args.qa_receipt,
        wav_path=args.wav, controller_id=args.controller_id,
        gpu_policy=args.gpu_policy, seed=args.seed,
    )
    if plan["synthetic_test_injection"]:
        raise RuntimeError("CLI cannot emit a synthetic parity plan")
    with args.output.open("xb") as stream:
        stream.write(canonical_bytes(plan))
    print(json.dumps({"status": plan["status"], "plan": str(args.output),
                      "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
