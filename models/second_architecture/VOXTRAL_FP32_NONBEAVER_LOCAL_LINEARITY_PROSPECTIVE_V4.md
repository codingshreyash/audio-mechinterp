# Prospective FP32 non-beaver local-linearity pilot — V4 signed-zero baseline amendment

18 November 2026. **Source-only amendment; no model run or executable freeze.** The independently reviewed [V2 protocol](VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V2.md) and [V3 output-budget amendment](VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V3.md) remain immutable. V4 changes how the donor-minus-recipient direction is obtained before constructing each site's N0 clean reference. It requires a new independent conceptual GO, implementation/source review, CPU preflight receipt, and external freeze before any checkpoint weight load.

## Why V4 is required

V2 defined `δ_s` from the donor's passive N0 source and the recipient's **site-specific N0 clean-reference prepatch** source. The N0 clean reference must therefore run before `δ_s` is known. The draft V3 runner used a zero direction for that clean reference and the final signed `δ_s` for its later alpha-zero paired baselines. In FP32, `0.0 * δ_s` can have a different signed-zero bit pattern from `0.0 * 0`; adding those values to a negative-zero source can produce different returned source bytes. Thus the two alpha-zero hooks need not be bitwise identical even when their mathematical edits are both zero. This violates the frozen clean-to-zero parity requirement and could invalidate a site for an avoidable implementation reason. No numerical result from this draft runner exists.

## Revised direction and clean-reference order

For each fixed recipient context, run **one additional passive N0 recipient prompt forward** with the same no-grad, no-cache, FP32 model settings and passive hook topology as the already specified donor capture. Capture and retain the full encoder-31 source, full projector output, full language-model input, prompt-final LM0/14/29 states, first-answer logits, audio placeholder positions, and runtime flags. This recipient capture is natural and unedited; it introduces no fitted parameter or new choice of source rows. Its sole numerical purpose is to supply the recipient prepatch source before `δ_s` is constructed.

Derive both fixed site directions from the passive N0 donor and passive N0 recipient captures:

`δ_s = mean_R(H_donor,passive,N0) − mean_R(H_recipient,passive,N0)`.

Require each `δ_s` to be finite and nonzero. Then run the site-specific N0 clean reference using this **final signed `δ_s` and alpha zero**, with exactly the same hook operation, row list, direction tensor values, no-grad setting, leaf flag, batch, and logits settings as every later N0 paired zero at that site. Run the site-specific G1 clean-VJP graph with the same final signed `δ_s` and alpha-zero hook as every later G1 paired zero at that site. The G1 and N0 modes retain their distinct graph/leaf settings and separate within-mode references.

Before the VJPs and paired edits, compare the passive N0 recipient's **full** source at each site bitwise with that site's N0 clean-reference **prepatch full** source. Save both full tensors and hashes. A mismatch invalidates that site's directional numerical gate; record it as raw evidence and do not substitute a different recipient state or direction. Continue to require all V2 within-mode clean-to-zero full-tensor bitwise parity checks, including prepatch source, returned source, projector, LM input, prompt-final LM29, and first logits. At alpha zero, the clean reference and paired zero now receive the same final signed direction, removing the avoidable signed-zero mismatch. Do not use a later zero as a replacement clean reference or relax bitwise parity.

## Fixed work, evidence, and limits

The four added passive recipient captures make **one donor plus one recipient passive N0 capture per context**: four donor and four recipient captures campaign-wide. Each context still has 96 patched forwards, eight VJPs, two G1 clean graphs, two N0 clean references, and two natural generations. The four-context campaign still has 384 patched forwards, 32 VJPs, eight G1 clean graphs, eight N0 clean references, and eight natural generations. The added passive recipient capture does not count as one of the 96 patched forwards, and it is saved in full. The prior two-cycle, six signed-alpha paired schedule, four-probe basis, fixed old-FIT `color_05` pair, source sites, modes, target, audit arithmetic, failed-cell handling, claim limits, and fresh-corpus exclusion remain unchanged.

The V3 tensor-payload estimates were 1,250,320,768 bytes for a Piper context and 1,258,180,992 bytes for a Kokoro context. Source review found that this draft estimate omitted the selected pre/post/effective row tensors on the 48 zero records, the fixed basis and two deltas, and the new passive recipient capture. Add the following explicit tensor budget before serialization:

| Additional retained component | Piper bytes | Kokoro bytes |
|---|---:|---:|
| 48 zero-record selected pre/post/effective row tensors | 35,389,440 | 42,835,968 |
| Fixed float64 `[3072,4]` basis and two FP32 direction vectors | 115,712 | 115,712 |
| Full passive recipient encoder/projector/LM-input/logits and three LM terminal vectors | 17,715,200 | 17,715,200 |
| **Revised partial tensor payload** | **1,303,541,120** | **1,318,847,872** |

The four-context partial payload is **5,244,777,984 bytes**. The old V3 estimate already conservatively double-counted one projector-sized tensor in its donor component, so these are not exact serialized byte counts. They still omit some small clean/donor terminal vectors, metadata, and torch serialization and launcher overhead. The implementation review must recompute the retained schema and measure full-size synthetic serialization before freeze. The V3 **2 GiB combined per-context** and **8 GiB combined campaign** hard output caps, 12 GiB free-disk admission, 300-second per-model-process deadline, 30 GiB MPS-driver cap, capped stdout/stderr, and 120-second saved-data audit deadline remain unchanged. If the extra forward or bytes prevent completion within those caps, retain the failed context and stop according to the existing no-retry rule; do not change the schedule or limits after seeing results.

The V3 source commit and preflight receipt are draft implementation artifacts. Update and independently review their source and receipt for this V4 method, then create a new external freeze before any live model work. An earlier V3 freeze, if one exists, does not authorize this amended experiment.
