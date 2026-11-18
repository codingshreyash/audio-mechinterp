# Independent conceptual review: V5 exact-zero identity amendment

18 November 2026. I independently reviewed committed source-only
`VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V5.md`, first added
in commit `e31045d`, SHA-256
`99de95003190de6b5fb55c5ba8bed446dfea7cdd8cf53c0569d304bcb942b49b`.
The V1 HOLD (`de236e5`) and V2–V4 design GO reviews (`69b8b27`,
`4f01214`, `61ea5da`) remain immutable. V5 amends only the alpha-zero
source-hook operation.

**Decision: GO for V5 non-beaver FP32 local-linearity protocol design only; model launch HOLD until source and external freeze reviews.**

## Independent checks

- The existing borrowed edit helper computes `source + alpha*direction`
  even at alpha zero and checks with `torch.equal`. On CPU I reproduced
  its signed-zero weakness: a `-0.0` FP32 source element plus
  `0.0*positive_direction` became `+0.0`; `torch.equal` still returned
  true, while raw FP32 bytes differed.
- I reproduced the V5 operation on CPU: detach a FP32 source leaf, clone
  the leaf, clone selected rows bitwise and write them back at the same
  indices. The full returned tensor retained both negative-zero bit
  patterns exactly. With the source leaf requiring grad, the returned
  tensor retained a connected gradient path; a sum of all returned
  elements had a unit gradient to each source element. This checks the
  proposed operation's basic identity and connectivity, not a full
  Voxtral hook implementation.
- V5 specifies that every site-specific G1/N0 clean reference and paired
  zero uses the **same** alpha-zero clone-and-replace branch, final signed
  delta in its descriptor, row list, detached-leaf construction, hook
  order, batch, grad context and leaf flag. It requires bytewise full
  pre/post source identity, including signed zeros, and an explicitly
  allocated all-zero effective edit. Recorded zero metrics are fixed at
  norm/fidelity/nonuniformity `0/0/0` and unchanged fraction `1`.
  The G1 branch must retain its source-leaf gradient path, while N0 must
  remain graph-free.
- Nonzero signed-alpha edits keep the V2–V4 `source + alpha*delta` FP32
  arithmetic and realized-difference measurements. The frozen borrowed
  helper is not modified; the future runner must call it only for
  nonzero edits. The CPU fake-model and saved-data auditor must test
  negative-zero identity, clean-to-paired-zero parity, and G1 VJP
  connectivity before a live model run.
- Model work remains 96 paired patch forwards, eight VJPs, two G1 clean
  graphs, two N0 clean references, one passive donor, one passive
  recipient, and two natural generations per context; four contexts still
  mean **384 patch forwards and 32 VJPs**. The saved zero tensors retain
  their shapes/dtypes, so V4's partial payload and V3's 2-GiB/context,
  8-GiB/campaign, 12-GiB free-disk, 300-second/process, 30-GiB MPS and
  120-second CPU-audit limits remain the prospective controls. Full-size
  synthetic serialization against the **implemented V5 schema** remains
  a mandatory pre-freeze check.

## Review limit

This is design approval only. No V5 implementation, model weights, or
live outputs were reviewed here. The updated runner, auditor, preflight
and byte-budget receipts require independent source review and a new
external freeze with its own independent review before any weight load.
