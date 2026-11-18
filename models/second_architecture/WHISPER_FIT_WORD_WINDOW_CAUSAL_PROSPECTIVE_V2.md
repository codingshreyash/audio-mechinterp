# Whisper Tiny FIT word-window causal assay — prospective v2 correction

18 November 2026. This corrects two implementation details in the [v1 plan](WHISPER_FIT_WORD_WINDOW_CAUSAL_PROSPECTIVE_V1.md) identified by the [independent v1 review](WHISPER_FIT_WORD_WINDOW_CAUSAL_INDEPENDENT_PLAN_REVIEW_V1.md). All input, split, window, direction, count, output, and stop rules in v1 remain in force. No model inference or intervention result has been run under either plan.

## Exact causal sites

The three intervention sites are (1) the tensor passed to `encoder.layers[0]` immediately after `conv2`, positional addition, and dropout; (2) the output of `encoder.layers[1]`; and (3) the output of `encoder.layer_norm`, after `encoder.layers[3]`. The first site uses a forward **pre-hook** on `encoder.layers[0]`; the second and third use forward hooks. The final site is the actual normalized encoder state consumed by decoder cross-attention. Capture the unedited tensors at all three sites. Capture post-block outputs 0, 2, and 3 as additional observation boundaries so the report can name the first captured numerical divergence accurately. A post-block-3 tensor is **before** final layer normalization and is not described as the decoder input.

For each arm, exactly one intervention hook must fire on the one encoder pass made for that complete greedy decode. The hook must preserve all other arguments and output tuple members, and replacement is `recipient.clone()` with only the frozen `W` or `P` rows assigned from the donor tensor at the **same site**. Record shape, dtype, finite checks, hook count, and `L2`/max absolute difference at every captured observation boundary. A self-patch must reproduce the unedited baseline token IDs exactly. The post-final-norm all-row donor patch must reproduce the unedited donor baseline token IDs exactly, because the decoder sees an identical encoder state.

## Exact random control

For each engine, direction, and final-normalized-state site, let `D = donor[:, W, :] - recipient[:, W, :]`. For seeds `20261118` and `20261119`, draw an independent standard-normal tensor `R` of the same shape using a local CPU `torch.Generator`, normalize it so `||R||_F = ||D||_F`, and add it only to recipient rows `W`. If `||D||_F` is zero or nonfinite, stop the case as invalid. Save the actual per-case seed, both norms, and a hash of the resulting edit tensor. This matches the **total** edit norm over the window; it does not match per-row norms or semantic direction. The planned decode total remains 52.

The independent source review must verify these corrected site and norm semantics before the first live model pass. The V1 HOLD remains in the record.
