"""CPU-only Stage-B audit adapter over the separately reviewed tensor auditor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import voxtral_old_fit_handoff_stage_b_audit_v1 as auditor
import voxtral_old_fit_handoff_stage_b_launch_gate_v1 as gate
import voxtral_old_fit_handoff_stage_b_runner_v1 as driver


def audit_existing(phase: str, attempt_id: str, freeze_sha256: str,
                   prior_model_work_seconds: float) -> dict:
    # No model is constructed here. Both Stage-A sidecars are rehashed before
    # the saved Stage-B tensor arithmetic is independently recomputed.
    gate.verify_audit_inputs(phase, attempt_id, freeze_sha256,
                             prior_model_work_seconds)
    bf16 = auditor.verify_bf16_posthoc()
    stage_a = driver.verify_stage_a_manifests()
    if bf16["receipt_sha256"] != gate.sha(driver.continuation.POSTHOC_RECEIPT):
        raise ValueError("BF16 posthoc receipt changed")
    binding = {
        "manifest": {arm: stage_a["arms"][arm]["manifest_sha256"] for arm in ("bf16", "fp32")},
        "result": {arm: stage_a["arms"][arm]["manifest"]["result_sha256"] for arm in ("bf16", "fp32")},
        "audit": {arm: stage_a["arms"][arm]["manifest"]["audit_sha256"] for arm in ("bf16", "fp32")},
    }
    output = driver.RUNS / phase / attempt_id
    return auditor.check_attempt(output, bf16["native"], bf16["probes"],
                                 stage_a_binding=binding,
                                 prior_model_work_seconds=prior_model_work_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=gate.PHASES, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--prior-model-work-seconds", type=float, required=True)
    args = parser.parse_args()
    result = audit_existing(args.phase, args.attempt_id, args.freeze_sha256,
                            args.prior_model_work_seconds)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
