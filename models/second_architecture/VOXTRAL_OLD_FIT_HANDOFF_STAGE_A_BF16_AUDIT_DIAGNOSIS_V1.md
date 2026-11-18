# Stage-A BF16 audit failure diagnosis

18 November 2026. Scope: the single saved BF16 Stage-A attempt at `runs/voxtral_old_fit_handoff_stage_a_launch_v1/bf16/20261118T131755298331Z`. This note diagnoses an auditor failure. It does not change the frozen auditor, certify the attempt, authorize another model run, or change the launch receipt's `launch_failed` status.

## Observed failure and cause

The runner exited 0 and wrote a completed receipt (`result.json` SHA-256 `f7b3906aff2b1bb9c7b8fb818402982a9020d8507fcba9cf51cc58821dddd325`). The frozen auditor (`voxtral_old_fit_handoff_stage_a_audit_v1.py` SHA-256 `4f800ed604295ae0e4fbd53c893fccd14b8968badb43b835e6ec975e49b0cf3d`) exited 1 at `validate_layout`, with `model parameter dtype/device/shape differs`. Consequently `launch_finish.json` records `launch_failed`, `auditor_returncode: 1`, and no audit hash.

All 762 saved parameter rows have `device: "mps:0"`; `model_layout.json` also records `valid: true` and dtype/device counts of 761 BF16 MPS plus one FP32 MPS parameter. The runner uses the existing `parameter_layout` helper: it tests `param.device.type == "mps"`, serializes each row with `str(param.device)`, and aggregates counts using `param.device.type`. The auditor expects each serialized row's device to equal the literal string `"mps"`. The saved `"mps:0"` spelling therefore fails even though it denotes MPS device index zero. This is an exact serialization-contract mismatch between the reviewed runner and the reviewed auditor; it is not evidence that a parameter was loaded on CPU or an unintended device.

## Read-only diagnostic

I imported the unmodified frozen auditor into a separate Python process and wrapped only `validate_layout`. The wrapper copied the parsed layout in memory, changed parameter-row device strings equal to `"mps:0"` to `"mps"`, and passed that copy to the original `validate_layout`. It did not alter any saved file, source file, tensor, receipt, or other validation function. The complete `audit(attempt)` then returned `audited_completed_old_fit_handoff_stage_a_native_only` in approximately 1.2 seconds. Its content-addressed hashes were details `df9fe20d1fd961a0b3502e073f933eed658f7a15de3214087552987ee8e8e849` and probes `dad65cd52beff26ed4f6dfcc5b541a9a0d504a207f23c2783f8ce71113f6b53e`. It reported 36 audited native pairs, 72 patch forwards, 44 VJPs, 24 LM0-control cells, and 214 progress lines. Thus no further rejection appeared in the complete saved-tensor and receipt audit after that one normalization.

This was a diagnostic bypass in memory, **not** an accepted independent audit. The original frozen auditor still rejects the original saved attempt. Even a successful CPU audit attests only saved-tensor and receipt arithmetic; it cannot independently attest that model execution produced those tensors.

## Prospective resolution

1. Preserve the original raw directory, v1 auditor, freeze receipt, and failed launch receipt unchanged. Do not rerun the model; the protocol forbids retries and the saved runner result is available for a separate audit.
2. Add a versioned audit-only amendment that accepts exactly the MPS row spellings `"mps"` and `"mps:0"`, while retaining the existing dtype, count, shape, hash, chronology, tensor, and numerical checks. Add a CPU regression showing both accepted spellings and rejection of `"cpu"` and nonzero MPS indices. Review the code and amendment independently before use.
3. Produce a separate, content-bound posthoc audit receipt. Bind it to the original attempt path and exact result, details, probes, original auditor, amended auditor, and amendment-review hashes. State explicitly that the original launch failed its v1 audit and that the later v2 audit is an amendment, not a retroactive rewrite of `launch_finish.json`.
4. Admit FP32 or Stage B only through a newly reviewed external gate that explicitly consumes the amended audit receipt and unchanged original attempt, and checks all prior protocol gates. Keep the no-retry and cumulative model-work accounting intact.

The current result remains `launch_failed` until that separate process completes.
