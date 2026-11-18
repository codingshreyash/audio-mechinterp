# Independent conceptual review: non-beaver FP32 local-linearity pilot V1

18 November 2026. Reviewed the committed source-only protocol
`VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V1.md` from commit
`acdfe62`. Its SHA-256 is
`c3ae7038ff1c5a00645430db9cd5db67a0b746c4400f02ef7b853d6952966718`.
This review is of the design only. It authorizes no runner, freeze, or model run.

**Decision: HOLD for protocol design. Do not implement or launch this V1 plan as written.**

## Checks that passed

- The committed FIT selection SHA-256 is
  `f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b`;
  the committed causal-mask SHA-256 is
  `a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e`.
  Both contain `color_05|fit_chosen` for Piper and Kokoro.
- Rehashed all four selected WAVs against the protocol. Each is mono, 16 kHz,
  signed 16-bit PCM. Within each voice, the paired WAV PCM is byte-identical
  before and after the frozen word slot and differs inside it. The frozen mask
  has whole-word rows of 60 encoder and 15 projector rows for Piper, and 73
  encoder and 18 projector rows for Kokoro.
- The older FP32 numerical follow-up uses `grad=True` for its clean derivative
  graph **and** its 72 paired zero/edit captures. The separate Stage-A handoff
  uses `grad=True` for its clean VJP but defaults to `grad=False` for paired
  traces. The protocol describes these different comparisons correctly.
- The four-condition factorial review reports a small post-LM0 difference
  only in the gradient-carrying condition, with bitwise-identical captured
  LM input and unchanged first-token argmax. The new protocol does not call
  this an audio mechanism or security finding.
- The planned paired-forward arithmetic is four contexts × two sites × two
  modes × two cycles × six signed steps × zero/edit = 384 forwards. The
  per-context count is 96; the four-probe VJP count is 16 overall. The
  300-second/30-GiB/1-GiB limits are specified per context, not inferred
  from the earlier 80.70-second run.

## Required changes before a new GO review

1. **Tie the derivative graph to its patched zero baseline.** The clean VJP
   graph keeps the projector connected to the encoder source, while a
   projector-site G1 zero/edit call replaces the projector output with a new
   leaf. Matching selected prepatch source rows does not prove the two
   forwards have the same LM input or target. Predeclare and save clean-versus-
   G1-zero comparisons at full source, projector, LM input, LM29 target and
   first logits for each site/context. Define a fixed parity rule or observed
   floor before model work, and mark the cell invalid if it fails. The same
   requirement must cover any encoder-site graph difference introduced by
   its source hook.
2. **Define how repeated cycles become scored cells.** Each site/alpha has
   two paired zero/edit cycles, but the advance gate counts 24 site-alpha
   cells per site. Specify per-cycle validity and resolution, whether both
   cycles must pass, how their two predictions/errors combine, and how one
   failed or malformed cycle affects the all-planned denominator. Define
   `maximum L2 spread among twelve zeros` as an exact all-pairs or other
   deterministic calculation. Without these rules the gate can be changed
   after observing the data.
3. **Define the N0 clean reference.** The source-row parity rule refers to
   `that mode's clean source rows`, but the protocol only specifies one G1
   clean graph. Add an explicit N0 clean capture or define the N0 paired-zero
   reference and cross-mode comparison. State which saved tensors the CPU
   auditor uses to verify it.

Also specify the exact SHA-to-seed byte order, PRNG, Rademacher sign mapping,
QR implementation and sign convention for the frozen four-probe basis. A
later executable freeze can bind the actual tensor bytes, but the prospective
recipe should not permit multiple valid bases.

No model weights were loaded for this review. No source file was changed.
