# Independent V5 non-beaver saved-data auditor source review — HOLD

18 November 2026. I reviewed the CPU auditor and tests committed at
`6a92ed14b7713733a4397adec3419d98d2353e99` without editing them or
loading model weights. The reviewed SHA-256 hashes are:

| File | SHA-256 |
|---|---|
| `voxtral_fp32_nonbeaver_linearity_audit_v1.py` | `471d9494108a071c1880791b56fc3f41e755e58dddad45aa06095deaa843dca9` |
| `test_voxtral_fp32_nonbeaver_linearity_audit_v1.py` | `b1685746c3bd7c46db7231da81e5354056cb09eff4b5a858f5f74e4f324f4025` |

**Decision: HOLD for V5 CPU saved-data auditor source. No model launch.**

## Verified checks

- All 20 CPU tests pass. The complete synthetic fixture has 96 ordered
  forwards per context; four contexts aggregate to 24 planned cells per site.
  The tests exercise 66 unordered zero comparisons, the worst of two cycle
  errors, signed-zero byte equality, passive donor/recipient direction,
  invalid cells retained in the denominator, sidecar/result rehashing,
  malformed and nonfinite tensors, and a tampered metadata receipt.
- I independently ran the V5 full-size serialization probe. Its Piper
  sidecar was 1,302,437,429 bytes and the Kokoro sidecar was 1,318,364,789
  bytes. I mmap-loaded the Piper sidecar, checked finiteness and SHA-256 for
  all 933 retained tensors (1,302,066,560 tensor bytes), and recomputed 288
  clean/zero boundary byte comparisons. The full run took 2.102 seconds on
  this Mac. This is a synthetic size/path check, not a model result.
- The auditor's source binds the V2–V5 protocol hashes, exact four WAV and
  batch hashes, fixed basis, selected mask rows, sidecar digest, 2 GiB
  worker cap, result/meta receipts, and fixed four-context order. It does
  not independently attest that a claimed model forward occurred.

## Blocking issue

`checked_trace` checks the existence and Boolean type of
`runtime_checks["projector_lm_input_bitwise"]` and
`runtime_checks["lm_input_lm0_pre_bitwise"]`, but it does not require either
recorded handoff check to be true on every trace. For full captures, the
projector flag is compared to recomputed equality, so a false handoff with
a matching false flag passes. The LM0 pre-input flag is not checked at all.
For edited captures, neither full tensor is retained, and a false flag can
pass without even invalidating the cell. The reviewed runner asserts both
handoffs before saving, but the independent saved-data auditor must not
accept contrary saved evidence as a valid numerical cell.

I mutated one otherwise complete 96-forward synthetic payload at a time:

| Mutation | Auditor result |
|---|---|
| Full zero: `lm_input_lm0_pre_bitwise=false` | Accepted; 0 invalidities, cell valid |
| Edited trace: `projector_lm_input_bitwise=false` | Accepted; 0 invalidities, cell valid |
| Edited trace: `lm_input_lm0_pre_bitwise=false` | Accepted; 0 invalidities, cell valid |

Require both flags true on every trace. For full captures, recompute the
projector handoff from retained tensors and compare LM input with the saved
LM0 pre-input digest. For edited traces, the flags and digests remain runner
claims because the full tensors are not retained; require their consistency
and state that limit. Add negative mutation tests for both full and edited
records. Preserve this HOLD review when a corrected source is rereviewed.
