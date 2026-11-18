# Stage-A FP32 continuation after the preserved BF16 audit failure

18 November 2026. This is a prospective amendment to the **admission and audit process**, written after the one BF16 model process and before any FP32 Stage-A process. It does not change the old-FIT estimand, model, input, native runner, source edits, probes, thresholds, arithmetic, or resource limits in `VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md`. It does not authorize a BF16 retry or Stage B.

## Preserved observation

The original BF16 launch remains `launch_failed`: the runner exited 0, the frozen v1 auditor exited 1, and `launch_finish.json` SHA-256 `15ed053c8ca4e619735cce3319b5d2326ff4b19216475754acc2cbf4e3bcdf9a` records both. Its completed raw result and both tensor sidecars remain unchanged. The separately reviewed v2 auditor changes exactly the accepted spelling of MPS device zero in parameter-layout rows, from `mps` to `mps` or `mps:0`. Its independent source review is commit `a8b6c248f285485702ef46dca08f91012e51dbbf` and grants GO **only** for posthoc CPU audit of saved bytes. It cannot retroactively make the original launch successful or attest the live model execution.

## External posthoc BF16 audit receipt

An actor outside the launcher runs the reviewed v2 auditor against the original `runner/` directory without rewriting any file in that attempt. Save the exact auditor stdout bytes as `VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_V2.json`. Independently rehash all 12 files in `VOXTRAL_STAGE_A_BF16_FAILED_ATTEMPT_MANIFEST_V1.json`, including both local large tensor sidecars, and require their recorded sizes and hashes. Require the ten committed files to match their Git blobs, the saved original freeze to validate under its frozen v1 gate, the runner result to have the original completed status, and the original finish to retain `launch_failed`, runner return code 0, auditor return code 1, and no audit hash. Check the v2 audit JSON's source hashes, result and sidecar hashes, exact expected counts, and its saved-tensor-only scope. A shell exit code alone is insufficient.

Then commit a separate `VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RECEIPT_V1.json` with exactly these keys:

| Key | Required value |
| --- | --- |
| `schema` | `voxtral_stage_a_bf16_posthoc_audit_receipt_v1` |
| `status` | `posthoc_saved_tensor_audit_passed_original_launch_failed` |
| `arm` | `bf16` |
| `attempt_id` | `20261118T131755298331Z` |
| `original_launch_status` | `launch_failed` |
| `failed_attempt_manifest_sha256` | SHA-256 of the committed failed-attempt manifest |
| `original_freeze_sha256` | SHA-256 of the original external freeze |
| `original_launch_finish_sha256` | SHA-256 of the unmodified original `launch_finish.json` |
| `original_result_sha256` | SHA-256 of the unmodified BF16 `runner/result.json` |
| `original_layout_sha256` | SHA-256 of the unmodified BF16 `runner/model_layout.json` |
| `original_details_sha256` | SHA-256 of the retained BF16 details tensor sidecar |
| `original_probes_sha256` | SHA-256 of the retained BF16 probes tensor sidecar |
| `original_v1_auditor_sha256` | SHA-256 of the frozen v1 auditor |
| `v2_auditor_sha256` | SHA-256 of the independently reviewed v2 auditor |
| `v2_auditor_source_review_commit` | Full commit `a8b6c248f285485702ef46dca08f91012e51dbbf` |
| `posthoc_audit_sha256` | SHA-256 of exact saved v2 auditor stdout |
| `model_work_elapsed_seconds` | `63.245136791840196`, copied from original finish |

An independent reviewer then checks this receipt, raw manifest and exact v2 audit output, including the two retained local sidecars, and writes a separate GO/HOLD review. That review must distinguish saved-tensor arithmetic from live model execution. The posthoc receipt and review are evidence of a later audit, never a replacement for the original failure receipt.
For an executable GO, the review must contain the exact line `Decision: GO for receipt provenance; FP32 HOLD pending launcher review and freeze.` A HOLD review cannot satisfy the gate.
The gate reads the committed receipt-specific review `VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RECEIPT_INDEPENDENT_REVIEW_V1.md`. It requires the audit output, receipt, and receipt review bytes to equal their own first-add Git blobs. A later edit and recommit cannot silently redefine those admission facts.

The already committed independent posthoc **result** review at `074fb1b941732bbc1235f24461f50f4295cfd6e1` pins the exact audit JSON SHA-256 `934a4517a0de2880051cc3be4f135cc475dc1098e742c7cc62a7e5990ac5f7f9` and review-file SHA-256 `4de300de75c6402e1e9b6707eefca5eea42722a1f96a2747f8d7c7e14296f0ae`. The later receipt review is still a separate prerequisite. The continuation gate binds both reviews and checks their Git chronology.

## FP32-only admission

Only a **new** externally frozen and independently reviewed gate can start one FP32 process. The gate must bind the original freeze, failed-attempt manifest, v2 auditor/amendment/source review, posthoc audit and receipt, independent receipt review, the unchanged native runner/protocol, its own code/tests/protocol and launcher review. Its freeze must be committed only after all reviews, must verify each bound file against its committed blob, and must reject a missing, dirty, stale, malformed, or self-authored freeze. The gate verifies the original failure and every raw file again immediately before reservation, after model work, and after FP32 audit. If any check fails, no new model process starts, or the reserved attempt finishes as failed; the BF16 process is never repeated.
The continuation freeze must also equal the exact bytes of its **first-add Git blob**. Recommitting a changed `reviewed_utc` or any other field makes admission fail even if the new bytes match HEAD.
The gate rejects symlinks in every bound path, including receipt, audit output, reviews, freeze, original source and raw files. A symlink to a committed same-byte copy is still a changed path and cannot satisfy the freeze.
For an executable GO, the independent launcher review must contain the exact line `Decision: GO for FP32-only launcher source; model run HOLD until external freeze.` This is a source-review marker only; the later committed external freeze remains mandatory.

The FP32 process uses the original runner `--run --arm fp32` in the pinned environment and offline cache, with an exclusive fresh output directory and permanent reservation. The runner and parent keep the original 1,200-second FP32 process ceiling, 120-second parent grace, observed 30-GiB MPS driver cap, 1.5-GiB whole-attempt cap, 8-MiB per-stream capture cap and v2 CPU audit timeout of 120 seconds. The original BF16 model-process time `63.245136791840196` counts toward the original 4,500-second total, even though its launch failed. The gate must verify enough budget remains for the *full* FP32 arm. A failed FP32 attempt consumes its one allowed slot and all its measured model-process time; no retry or workload reduction is allowed. The v2 auditor also audits FP32, because its single device-spelling fix applies to both arms. FP32 success requires a new separately audited and reviewed result/manifest before Stage B is considered.

This continuation can support the original one-case numerical handoff diagnostic. It is not a security finding, a general audio mechanism claim, or permission to inspect a fresh corpus. The large local sidecars are essential evidence and must remain available for independent re-audit.
