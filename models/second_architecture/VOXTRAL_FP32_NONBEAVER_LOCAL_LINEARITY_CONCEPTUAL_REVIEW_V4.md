# Independent conceptual review: V4 signed-zero baseline amendment

18 November 2026. I independently reviewed committed source-only amendment
`VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V4.md`, first added in
commit `1cb79ec`, SHA-256
`edff669ab0c611a50160896b5b4b75e7621dde743c4e769a7b10af08f92a9d94`.
The V1 HOLD (`de236e5`), V2 design GO (`69b8b27`), and V3 output-budget
design GO (`4f01214`) remain immutable. V4 amends their direction and
clean-reference order only.

**Decision: GO for V4 non-beaver FP32 local-linearity protocol design only; model launch HOLD until source and external freeze reviews.**

## Independent checks

- I reproduced the risk in the existing FP32 `realized_edit` helper on CPU.
  With source `-0.0` and alpha zero, a `+0.0` direction yields an edited
  `+0.0` bit pattern; a negative direction yields an edited `-0.0` bit
  pattern. Both compare numerically equal with `torch.equal`, but their raw
  FP32 bytes differ. A clean reference built with a provisional zero
  direction and a later paired zero using the final signed direction can
  therefore fail a bytewise clean/zero parity rule for an implementation
  reason.
- V4 adds one passive N0 recipient capture per context, alongside the
  already specified passive N0 donor capture. It derives both site deltas
  from those passive N0 source states. Each site-specific N0 clean
  reference and G1 clean-VJP graph then uses the **same final signed
  delta**, alpha-zero hook operation, row list, batch, hook order, mode and
  leaf flag as that site's paired zeros. The V2 within-mode bitwise
  clean/zero rule remains mandatory.
- The passive N0 recipient's **full** source at each site must match the
  site's N0 clean-reference full **prepatch** source bitwise. V4 retains
  both tensors and hashes. A mismatch invalidates the directional
  numerical gate without choosing a different recipient state or
  direction. This does not assert that an alpha-zero hook preserves every
  signed-zero bit from the passive natural output; the derivative and
  paired edit are defined around the identical hooked baseline.
- Counts are unchanged for paired model work: four contexts × 96 = **384
  patched forwards**, eight VJPs per context = **32 VJPs**, eight G1
  clean graphs, eight N0 clean references, and eight natural generations.
  V4 adds exactly **four passive recipient** captures to the existing
  four passive donor captures. No fresh corpus or outcome-based context
  choice is added.
- Independently recomputed V4's added per-context tensor terms: zero
  selected-row pre/post/effective captures are 35,389,440 bytes (Piper)
  or 42,835,968 (Kokoro); the float64 basis plus two FP32 direction
  vectors are 115,712; the new passive recipient full capture with three
  LM terminal vectors is 17,715,200. The revised **partial** payload is
  1,303,541,120 bytes for Piper and 1,318,847,872 for Kokoro, totaling
  5,244,777,984 across the four contexts. This remains below V3's
  2-GiB/context and 8-GiB/campaign hard caps, with implementation-level
  serialization and omitted-field estimates still required before freeze.
- The V3 12-GiB free-disk admission, 300-second process deadline,
  30-GiB MPS-driver limit, capped streams, independent CPU audit deadline,
  one-shot failure rules, and claim limits remain in force. V4 states
  explicitly that draft V3 source and preflight receipts cannot authorize
  the amended method.

## Review limit

This is a prospective design review only. The V4 runner must be changed
and independently source-reviewed, including a CPU test for signed zero,
then externally frozen and independently reviewed before any weight load.
The source review must measure full-size synthetic serialization against
the revised cap. No model weights were loaded or run for this review.
