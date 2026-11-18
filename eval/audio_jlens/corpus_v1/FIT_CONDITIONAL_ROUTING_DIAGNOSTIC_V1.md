# FIT-only diagnostic: can an unpatched audio state route Jacobians?

**Prospective analysis rule, 18 November 2026.** This is a diagnostic on the already-inspected **48 FIT contexts**, not a repair of Qwen v1 Stage A, a v2 fit, or a held-out claim. The v1 development gate failed and its locked test remains sealed. No model forward, new VJP, development/test audio, or Qwen answer is used here.

## Exact inputs and estimand

Read only these local, `torch.load(weights_only=True)` artifacts, verifying SHA-256 before use:

| Artifact | SHA-256 |
| --- | --- |
| `fit_rank16_transport_v1.stage1.pt` | `8276aca0e24fcaa1397693e0353f9c7ee59190036f3516c0c13ebcbc6208b641` |
| `fit_jacobian_heterogeneity_v1.result.pt` | `336a2589743c06a2e5be71f3fce694e7a01c78450dc81dc07c6368a01bc58903` |
| `fit_uniform_calibration_v1.raw.pt` | `76085f0dab8f37b6052024b582ff34533cac3d1c579b27b39a631c536d9cd110` |

Require identical ordered case IDs and audio hashes across all three, exactly 12 lexical pairs with four contexts each, finite six-bin source means, local Jacobians, edit vectors, rank-16 target basis, and measured terminal changes. At each site, the target for context `i` is `Uᵀ Δh_i`, where the saved measured terminal change is `α × terminal_delta_per_alpha` at **α=.025**. A candidate map predicts `α Σ_b J_b δ_{i,b}` in the same 16 coordinates, where `δ` is the saved uniform donor-minus-recipient bin-mean edit. Primary error is Euclidean distance between predicted and measured **rank-16 projected** terminal changes. Report full-terminal error and captured-energy fraction separately; this analysis does not predict the nonlinear class score or a complete answer.

## Fixed pair-excluded router

For each of 12 outer word-pair folds, exclude **all four** contexts of that pair from its derivative pool and feature preprocessing. Fit coordinatewise mean/standard deviation on the 44 training contexts' flattened six-bin **unpatched recipient** source means, replacing a zero standard deviation by one. PCA is the first eight right singular components of that training matrix, with a deterministic sign choice (largest absolute loading positive); a rank below eight fails. Project train and held-out source means using that fit only. Set Gaussian bandwidth `h` to the median **nonzero between-pair** train-distance in this eight-dimensional space; a nonfinite or zero median fails. For each held-out query, weights over all 44 training contexts are proportional to `exp(-||φ_query−φ_train||²/(2h²))` and sum to one; a zero/nonfinite weight sum fails. The sole candidate map is **0.5 × pair-excluded global mean + 0.5 × weighted local-Jacobian mean**. There is no label tuning, candidate search, or best-site selection.

Compare its projected-terminal L2 error to (a) the pair-excluded global mean on the same 44 training Jacobians, (b) zero predicted change, and (c) the recipient's own local Jacobian as an explicitly **in-sample oracle**. Also run a fixed wrong-router control: use the next lexical pair's query feature in sorted pair order for the same voice/direction slot, while retaining the evaluated pair's edit and excluding its Jacobians. This permutation is outcome-independent and only a negative control. Report mean and median errors at both sites, all 48 case-level rows, conditional/global ratios, and percentile 95% confidence intervals from 10,000 whole-pair bootstrap resamples seeded from the concatenated three input hashes. Missing, nonfinite, or mismatched cases abort rather than shrink denominators. No threshold or pass label is attached to this exploratory FIT diagnostic.

The fixed target basis was itself learned from all 48 FIT contexts, including each outer held-out pair. Pair exclusion therefore tests **router/derivative pooling**, not fully independent representation learning. Even a lower FIT pair-excluded error would only motivate the fresh, basis-disjoint v2 protocol; it would not demonstrate transfer, a semantic lens, audio-time causal specificity, or a security finding.
