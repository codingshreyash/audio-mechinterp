# Independent executable review: Whisper Tiny cross-family tail-state probe

19 November 2026. **GO for one bounded multilingual band-noise attempt, subject to a separate committed V3 GO receipt and the live host gate.** This is a source decision. I did not load the checkpoint or run candidate inference.

## Identity and frozen selection

- V3 source was frozen in `f4604c4`; the source-only admission was committed in `f7629eb`.
- The V3 plan SHA-256 is `f60c2eb425dc21100f383acbd549ace272aab4189b34537857b44a80fce4ab4e`; the admission SHA-256 is `85c8ed4557d5aceb4e7384080b566feefc8947cdff43b716a9e962100357c162`. I rebuilt both from the committed source and obtained exact equality. The fixed V3 attempt and GO path were absent during review.
- The selected V4 contrast is `d08_band_noise_low`, equal-duration `z32000` versus `n32000_s0`, both 160,000 PCM samples. The saved multilingual outputs are complete `you` versus `Thank you.` with the same English prefix. The selection was made after V4 behavior was known; this is a test of whether the earlier *intervention locations* transfer to another sound family, not held-out discovery of a behavioral difference.
- V3 retains the three V1/V2 primary noise-to-zero rescue cells: post-convolution boundary, post-convolution tail interior, and post-block-0 boundary. Its source-row control indices, three sites, and two row regions equal V1 exactly. The two convolution kernels imply local mel support `2r-2..2r+2`; source/tail and tail/padding boundaries are encoder rows 400 and 500. Attention can spread information after the first block.

## Executable checks

The frozen plan schedules two fresh baselines and 54 patch calls: three sites, two regions, two directions, four local arms per cell, and six full-donor positives. The worker and auditor use a distinct V3 attempt, admission, GO, and output schema. Baselines must reproduce the V4 token paths, mel, mask, language prefix/detection, and first-step vectors. Patch evidence includes lossless native and edited encoder states; the saved auditor reconstructs edits, checks untouched rows and the 394-sidecar census, and qualifies each target with identity, source-row donor, equal-norm displaced-delta and full-donor controls. It reports execution `PASS` separately from the primary transfer count, which may legitimately be zero through three.

The V3 worker, parent, and auditor differ from reviewed V2 scientific functions only in versioned paths/schemas, prior-source bindings, the V3 target metric, and a stronger GO-time provenance check. That check requires current tracked source bytes to equal the admission and original GO tree, and evaluates source/admission/review ordering as of the original GO commit. Four Git fault tests cover byte-identical later restoration, modified current bytes, a GO-time source mismatch, and an admission committed after review.

The bounded parent retains the live nonce and heartbeat gate before weight import, a 1,800-second whole deadline, process-tree RSS below 4 GiB, evidence below 1 GiB, and fail-closed host monitoring. At each start it requires battery at least 80%, open lid and no thermal warning; it aborts at battery 70% or less, unknown/sleep/thermal state, or a power-sample gap over 15 seconds. Power probing runs asynchronously while RSS, heartbeat and deadline checks continue. Post-run power evidence and all sidecar/output invariants are checked by the saved auditor.

I independently ran all four V3 no-weight test modules: **15/15 passed**. I also reran the two inherited independent host/parent fault-test modules: **9/9 passed**. I recomputed the V3 admission, confirmed the attempt was unused, and rehashed the complete prior V1 and V2 raw inventories: **806/806** and **403/403** files matched their pinned sizes and SHA-256 digests. Those prior raw attempts remain local and untracked; neither their contents nor the earlier compact results were edited for this review.

## Interpretation boundary

This GO authorizes only one already-observed, engineered non-speech contrast in multilingual Whisper Tiny. A complete donor-answer rescue at a primary cell, if observed, would support conditional transfer of that internal patch location to band noise; a failed or partial result is equally valid evidence. It would not establish human-speech behavior, a unique neuron, attacker-reachable perturbation, action violation, or a security vulnerability. The full EOS answer and controls matter more than a first-logit score change.
