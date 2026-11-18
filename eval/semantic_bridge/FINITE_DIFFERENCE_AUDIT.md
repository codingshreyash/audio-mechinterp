# Full-model finite-difference check of the terminal Jacobian

**Scope:** the saved `association:interview` Piper **fit anchor** only. I made no development/test pass or corpus fit. The [audit script](finite_difference_audit.py) loaded the pinned Qwen2.5-Omni-3B Thinker on local MPS and compared full downstream forwards after patching a terminal text residual against the [saved one-prompt Jacobian](artifacts/terminal_interview_piper.pt). The [machine-readable result](FINITE_DIFFERENCE_AUDIT.json) records the exact checkpoint/audio hashes, seed, perturbation sizes, per-direction metrics, and memory.

## Method

At source layer 0 or 17, I patched only the **last prompt position**. For each of three fixed random unit directions and each of two magnitudes (1% and 3% of that layer's anchor-state L2 norm), I ran both `+εd` and `-εd` through the actual remaining decoder blocks. The patched state was rounded to the model's BF16 dtype. The comparison used the **effective rounded perturbation**, not the nominal float32 vector: `J(δ_plus - δ_minus)` versus `h_35(plus) - h_35(minus)`. I recorded cosine similarity and `||observed - predicted|| / ||observed||`. This central comparison removes the common anchor state, but it cannot remove BF16 rounding effects in the forward computation.

The input audio SHA-256 was `c1317b32eab0c56e86d29f3ead175e72fc57410cd92f0dae338022d1b34ef99e`, exactly the checkpoint's fit input. The checkpoint SHA-256 was `caa1f346247ee0c0c5cb39a09bcd5a51acf564e287e2dd0d4b7c9c02f7c2ba78`. Repeating the unpatched forward gave **bitwise-identical** layer-0, layer-17, and layer-35 terminal residuals. A zero-magnitude hook patch at either source layer also gave **exactly zero** source and target change, so the hook's clone/return path by itself did not change the output.

## Result

Means over the three fixed directions (all individual measurements are in the JSON):

| Source layer | ε / source-state L2 | Cosine, predicted vs observed | Relative L2 error | Predicted central L2 | Observed central L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1% | −0.105 | 1.008 | 0.207 | 3.401 |
| 0 | 3% | 0.228 | 0.977 | 0.635 | 3.345 |
| 17 | 1% | 0.543 | 0.845 | 1.728 | 3.801 |
| 17 | 3% | 0.835 | 0.559 | 5.215 | 6.879 |

The layer-17 matrix predicts a substantial **directional** component at the larger perturbation, but its amplitude and residual error remain poor. The layer-0 matrix does **not** pass this finite-difference check for these random directions. The final BF16 residual had L2 norm 305 and maximum absolute coordinate 90. Observed central output changes stayed around L2 3–4 even where the layer-0 Jacobian predicted only 0.2–0.6. This pattern is consistent with coarse BF16 rounding or other non-smooth downstream numerical effects overwhelming small signals; it does **not** establish that rounding is the only cause. A real saved-matrix or graph-cut discrepancy remains possible. Larger perturbations improve signal-to-rounding ratio but weaken the local-linear approximation, so the 3% alignment is not a precision validation.

## Interpretation and next validation

The previous [fitter review](FITTER_REVIEW.md) established the checkpoint's row orientation and that the graph cut is mathematically valid for a derivative **from the hooked source residual onward**. This full-model check adds an independent intervention path, but under BF16 it does not validate the saved Jacobian quantitatively at layer 0, and only weakly validates layer 17. Do not describe either matrix as a verified causal mechanism on this evidence alone.

To separate a checkpoint/gradient problem from forward rounding, compare a few saved rows with a **fresh autograd VJP** on the same anchor and hook boundaries; that comparison is within the differentiable graph and avoids finite differences. Then, if memory permits, repeat a small high-precision downstream check or compare response over a finer perturbation-size curve, reporting the effective BF16 source changes and final residual quantization. A later direction-specific patch should use held-out matched pairs and measure complete-answer behavior, not just token rank.
