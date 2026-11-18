# Independent review: Qwen FIT-only conditional routing diagnostic

**18 November 2026. Verdict: numerical audit passed; the proposed audio-state router did not show a useful advantage over the pair-excluded global map in this FIT diagnostic.** This is an analysis of already measured FIT tensors. It does not change the failed Qwen v1 development gate or expose the locked test.

## Provenance and accounting

The committed result is `fit_conditional_routing_diagnostic_v1.result.json`, SHA-256 `2e53b6e7550fbe99c9b1b3045c914a00be66cb5b464a9b633808cea9311803b9`. Its embedded source SHA-256 (`a4cca7d759098531a7d504859f1dd86402b116a0cfbf33ba4dccb06be1081a65`) and protocol SHA-256 (`e69bb992a753df568c77e4c56d8444f543c24d9b93e2a0af0d6f0241d99e211d`) match the committed files. All three pinned input hashes match the saved FIT artifacts.

I independently loaded the saved stage-one basis, local Jacobians, and calibration edits with `torch.load(weights_only=True)`. Their ordered case IDs match. The result contains exactly 48 unique contexts at each site, grouped into 12 lexical pairs of four contexts. For each outer pair, my separate calculation used only the other 44 contexts to fit the feature mean, standard deviation, PCA8, Gaussian bandwidth, and candidate derivative pool. The saved wrong-router case uses the next sorted lexical pair with the same voice and sorted-word ordinal; the evaluated edit and derivative exclusion remain unchanged.

I recomputed every candidate's projected and full-terminal L2 errors from the saved tensors, including the conditional, global, zero, wrong-router, and local-oracle predictions. Across all 480 predictions, the largest absolute difference from the result was `2.18e-6` for projected L2 and `1.10e-7` for full-terminal L2, consistent with floating-point contraction order; maximum captured-energy difference was `8.86e-11`. Independent aggregation reproduced both mean ratios and the 10,000-resample, whole-pair bootstrap intervals exactly from the stated hash-derived seed. No case was dropped.

## What the result says

Primary metric: mean rank-16 **projected terminal L2**, lower is better. Intervals are the fixed 95% percentile bootstrap interval for the conditional/global mean-error ratio, sampled by lexical pair.

| Site | Conditional | Pair-excluded global | Zero | Wrong router | In-sample local oracle | Conditional/global ratio [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder block 31 | 0.547016 | 0.546933 | 0.729879 | 0.547071 | 0.035202 | 1.000152 [0.999186, 1.001322] |
| Connector | 0.716604 | 0.716685 | 0.966364 | 0.716766 | 0.056756 | 0.999887 [0.998957, 1.000786] |

The conditional map beats the pair-excluded global map in 20/48 encoder cases and 24/48 connector cases. It beats the fixed wrong-router control in 22/48 and 29/48 cases. Both conditional/global intervals include 1, and the mean differences are around `8e-5` L2 in opposite directions. This run therefore provides no evidence that the chosen unpatched source-state features, PCA8 distance, bandwidth, and fixed 0.5 shrinkage route the saved derivatives better than a global map. Both shared maps are better than predicting zero on average, while the own-context local derivative is much better in the projected space; that oracle uses the evaluated context's Jacobian and is not a deployable comparator.

The mean rank-16 captured-energy fraction is 0.2224 at encoder block 31 and 0.2235 at the connector. Full-terminal errors for the conditional map are 1.4710 and 1.9489, respectively. These figures matter because a strong projected-space oracle need not explain the full terminal change.

## Limits and next decision

The target basis was fitted on **all 48 FIT contexts**, including each outer held-out pair. Pair exclusion here tests derivative pooling and feature preprocessing, not independent basis learning. The result predicts a small internal change, not a nonlinear answer score or generated response. It uses no new audio, no development or test pairs, and no model forward. It cannot establish transfer, an audio-time circuit, or a security finding. The result is a useful negative diagnostic: this particular simple router does not close the shared-map gap, so any next router should be proposed and tested on the fresh, basis-disjoint v2 corpus without relabeling this FIT outcome as success.
