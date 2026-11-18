# Independent review: failed graph/no-grad diagnostic attempt

18 November 2026. I reviewed the committed raw evidence and manifest for the
single V1 attempt `20261118T151548690094Z`. I did not run the model, alter the
V1 runner or gate, or reclassify this attempt as a scientific result.

**Decision: GO for failed-attempt provenance only; V1 remains closed with no retry.**

The exact manifest is
`VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_FAILED_ATTEMPT_MANIFEST_V1.json`, first
committed at `738be7b`, SHA-256
`0a2335dcab3ce5bb4cd66f5d0095074058c37537743b1c42764da06438811280`.
Its ten raw files were separately committed first at `9b8b4d5` and total
163,803 bytes. I independently rehashed every current file against its
manifest row and the blob in that raw commit; all ten match. The two attempt
directories contain exactly the manifest's files, plus the launcher root's
one reservation. The V1 result, audit and tensor sidecar are absent. The
launcher and worker run directories contain only this attempt.

The freeze was first committed at `349dfa6` and independently reviewed at
`05e9247`, both before the reservation and raw attempt. The manifest's exact
freeze, freeze-review, V1 runner and gate SHA-256 values match the committed
files: `70601c9d95da4553237a48af258044db25a2e9fc70c40eda9e44a3f745cde06a`,
`0f6bd05f8a10263e504a844cef39167f828a4311fa752281caa87f2866690b08`,
`1f92935f219b6b6a86c5307ec57ad69ddabcb1c0de2387e8c2c99b49888c3d91`,
and `93bd26d235b6a60eb296b951850f40ac41d06f29fb57ab5ffce906e32cebfa59`.
The launch, worker claim and finish UTC timestamps are strictly ordered. All
three receipts use the same attempt ID and freeze hash. The result is
`diagnostic_launch_failed_no_retry`; worker return code is 1, auditor return
code is null, and the parent measured 28.450194334145635 seconds of model
process time. The worker error and progress hashes match the actual files.

Progress recorded `preflight_complete_before_model_load` and then
`model_loaded`, with no later phase. The worker raised `ValueError: Old-FIT
batch is not a tensor mapping`. In the frozen V1 runner, this check follows
model loading and precedes the first full-forward loop. A fresh CPU-only
`native.preflight_cpu()` inspection returned a
`transformers.feature_extraction_utils.BatchFeature`, not a Python `dict`,
with tensor keys `input_ids` int64 `[1,396]`, `attention_mask` int64 `[1,396]`
and `input_features` float32 `[1,128,3000]`. Thus V1's plain-`dict` guard
explains the recorded exception. The receipts and source support zero
**observed diagnostic forwards**; they do not provide tensor traces. No
result can be inferred about graph/no-grad parity or an audio mechanism.

This review verifies saved evidence and the control-flow explanation. It is
not independent attestation of device execution or proof of absence of any
unrecorded process. A later study must use a new prospective question,
source review and freeze; it cannot reuse the V1 reservation.
