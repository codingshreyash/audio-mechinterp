# Independent conceptual review: V3 output-budget amendment

18 November 2026. I independently reviewed the committed source-only
`VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V3.md`, first added in
commit `71ed672`, SHA-256
`a222ce0f06a0e82b2c22bf6231e605983d726610b2f51dfbe7299de3ea8d79bd`.
The immutable V1 HOLD (`de236e5`) and V2 design-only GO (`69b8b27`)
remain separate. V3 is an additive resource amendment to V2.

**Decision: GO for V3 non-beaver FP32 local-linearity protocol design only; model launch HOLD until source and external freeze reviews.**

## Independent arithmetic and scope checks

- Per FP32 tensor, encoder-31 full source is 7,680,000 bytes, projector
  output 4,608,000 bytes, LM input 4,866,048 bytes, and 131,072-wide
  first logits 524,288 bytes. For 24 encoder and 24 projector zero calls,
  the stated full-capture expression independently evaluates to
  **1,069,744,128 bytes**. The old 1-GiB cap leaves just 3,997,696 bytes
  for every other required record and is therefore infeasible as written.
- I independently recomputed each table component: four site-specific
  clean references 89,145,344 bytes; one donor full capture 22,286,336;
  48 edited selected-row and logit captures 60,555,264 Piper or 68,001,792
  Kokoro; three LM terminal vectors on 96 patch calls 3,538,944; eight
  selected-row VJPs 1,966,080 Piper or 2,379,776 Kokoro; two CPU batches
  3,084,672. Including the full zero captures gives **1,250,320,768
  bytes** for Piper and **1,258,180,992 bytes** for Kokoro before serialization
  and other retained fields. Two contexts per voice sum to
  **5,017,003,520 bytes** of estimated tensor payload. The table
  deliberately double-counts projector source and passive projector output
  for projector zero calls; that makes this budget conservative for those
  fields, but it remains only a partial output estimate.
- The new combined cap is 2 GiB = 2,147,483,648 bytes **per context** and
  8 GiB = 8,589,934,592 bytes across all four contexts. The protocol
  requires a reviewed implementation-level byte estimate including all
  saved fields and serialization overhead before freeze. It reserves a
  lower worker cap for launcher captures/receipts and requires the parent
  to enforce the combined cap. A cap failure stops the campaign; it does
  not permit dropping tensors, compression-only fallback, or a same-plan
  retry.
- Before campaign reservation, V3 requires at least 12 GiB available on
  the selected filesystem and absence of any existing campaign directory
  or reservation. This is a **run-time** check; current free space does
  not substitute for the frozen prelaunch gate. All four context
  reservations precede the first model process.
- The 300-second process cap, 30-GiB MPS-driver cap, 8-MiB individual
  stdout/stderr caps, and 120-second CPU audit cap are unchanged. V3
  explicitly inherits V2's four fixed contexts, two sites, two modes,
  two cycles, six signed steps, **384 paired patch forwards, 32 VJPs,
  eight G1 clean graphs, eight N0 clean references, four donor captures,
  eight natural generations**, acceptance rules, failure handling, and
  bounded claims. No fresh-v2 inference or security claim is authorized.

## Review limit

This is a design-only resource review. The eventual runner's exact
serialized record schema, output estimate, disk reservation and cap
enforcement must receive independent source review; an external freeze and
its independent review are still required before any model weight load.
No model was loaded or run for this review.
