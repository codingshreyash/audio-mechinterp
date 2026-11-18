# Independent Stage-B CPU auditor source review — HOLD

18 November 2026. Reviewed `voxtral_old_fit_handoff_stage_b_audit_v1.py` at SHA-256 `436891f9e2509bf8c789a218b683d9d118bbe50ec5b9d9122efbf100f5c30d24` (source commits `912dcc4`, `a73be84`) and its test file at SHA-256 `7647abc863698ef996fd296d31b09de5fe5ca52fa8b4194565eafe6796bb47e7`. The pinned prospective protocol, driver, and replay-primitives SHAs match the auditor's constants: `61a67ac87a4e4e17cb0b1f7ad972f25fda4e0d90e0b70798e7144b3fd6f2484b`, `f82f99e9d0e705cff682ffcf388986b772bf1db7b92699fa8052b0cb7c31b709`, and `bef337c90a024e034e60818761b4be79371ca0767f83d28efe29269ba0cc89e1`.

**Decision: HOLD for Stage-B auditor source. No Stage-B model execution or result interpretation.**

## Verified

- All 12 CPU unit tests pass. The real preserved BF16 failed attempt, its later posthoc audit, 17-key receipt and two local tensor sidecars rehash successfully. `verify_bf16_posthoc()` returns 12 cycle-zero pairs, 63.245136791840196 seconds of prior model work and receipt SHA-256 `df4e16989a67a4cc3e4584aa74c99f3b0a2601e82d8226adc7beb2d41bd38ba7`. This is a saved-data provenance check; the original BF16 launch remains failed.
- The CLI exits before reading a Stage-B attempt, with `Stage-B external Stage-A/launcher chain verifier is not yet implemented`. No FP32 Stage-A manifest or Stage-B external chronology is available. The auditor does not attest that model forwards or backward calls happened.
- Source checks include the fixed 68 VJP tensor shapes, exact archived BF16 source inputs, selected-row effective edits, projector self-derivative identity, direct-LM parity, four FP32 override parities, progress order and resource receipts. These checks do not cure the blocker below.

## Blocking mismatch

The frozen protocol §27 defines resolution as `max(3 × maximum paired-zero projected L2 spread, 1e-6)`. The Stage-A CPU auditor computes the maximum pairwise distance across all paired zero outputs. Stage-B `check_source()` instead uses only `parity.repeat − parity.clean` for `zero_repeat_spread_l2` and `resolution_l2`; it ignores the twelve source-replay paired zero outputs.

An independent synthetic example makes the difference concrete. Keep clean and repeat identical, change one later paired-zero LM29 probe by 1, and set the corresponding saved terminal delta consistently. `check_source()` accepts the record and reports spread `0`, resolution `1e-6`, measured norm `1`, and `resolved=True`. The paired-zero spread is `1`, so the protocol's resolution is `3` and this cell is unresolved. A result could therefore be called resolved when it is below observed replay variation. The current 12 tests do not cover that case.

Compute the maximum pairwise projected spread over the source-replay paired zeros for each site and target stage, use it for every source-replay cell's resolution, and add a negative or positive regression with discordant paired zeros. Review the amended source and tests before any live Stage-B freeze. Any future auditor output still needs a separate result review and must distinguish saved-tensor arithmetic from model-execution attestation.

## Superseding source rereview — 18 November 2026

The producer changed only `check_source()`'s source-replay resolution calculation and added a discordant paired-zero regression in commit `4ab9bf888a090c2cbfed3d7618a7942d8d92dde9`. New auditor SHA-256: `f3027134c91ed22c7ffa946b24d779af4932269205c32f703099efe23f7f59b9`; new auditor-test SHA-256: `9d8b1e777706420c1deef6ee9a719eba8a7d589403252db513baac07df8b7810`. The protocol, driver and replay-primitives source pins listed above remain unchanged.

I independently reran the failing synthetic case, without relying on the added test: with paired-zero projected spread `1`, the corrected auditor now reports resolution `3`, measured norm `1`, `resolved=False`, and no relative error. It computes the maximum pairwise distance across all six source-replay paired zeros for each site and target stage, as required by protocol §27. The new regression covers that case. All **29 CPU tests** across the auditor (13), driver (7), and replay primitives (9) pass. The preserved BF16 posthoc chain still rehashes to the receipt above with 12 cycle-zero pairs. The audit CLI still exits before a live result because the external Stage-A/Stage-B chronology verifier is absent.

**Decision: GO for corrected Stage-B CPU auditor source only; live Stage-B HOLD pending FP32 Stage-A manifest, external chronology verifier, separate freeze and review.** This GO approves the corrected saved-data arithmetic source. It does not attest model forwards/backwards or authorize Stage-B execution. The original HOLD and its reproducer remain above as the audit history.
