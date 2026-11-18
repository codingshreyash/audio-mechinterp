# Independent review: fit-only audio-Jacobian results

I audited [all-routes results](fit_gate_v1_all_routes_results.json) (SHA-256 `71a3f6bfece8d1ed17b69cf818c6619a6e31e66c3b5759a58cf9af8cfbd1aa85`) and [encoder-31 results](fit_gate_v1_encoder31_results.json) (SHA-256 `be1eeb4c9458bcf3080d30616fcfb905dca6611b68797ee6a7835acd96f94a28`) against [PLAN.md](PLAN.md) and [fit_gate_v1.py](fit_gate_v1.py). I did not run MPS, alter results, or open the new locked test.

## Provenance and saved-case checks

Both results match the current plan and split hashes, the pinned fit manifest, checkpoint/source revisions, candidate token IDs, alpha grid, and the current WAV bytes for their recorded source/donor hashes. All-routes records script SHA `a8ae2045…`; the current script SHA `18c64823…` matches encoder-31. I independently reconstructed the earlier script in memory by reversing only the later encoder-31 site/selector addition; its SHA equals the recorded all-routes hash. The saved executions are therefore attributable to those two source revisions without modifying either result.

There are **12 unique fit cases**: two voices × two directions × three source sites. All 12 clean answers are the expected complete word followed by EOS. All 60 patched generations reached EOS and recorded one patch hook call. All 12 alpha-zero conditions have exactly zero effective edit, prediction, and measured score change, and unchanged complete token IDs. At alpha one, all **12/12** complete answers flip to the donor with EOS, and each effective norm equals the saved donor-minus-recipient word-row norm. Cached and uncached source-state maximum differences are zero in every case. The per-row contribution sums reproduce each saved prediction within `2.0×10⁻⁷`; recorded prediction errors and patched-score contrasts also recompute. The two memory logs contain 65 and 33 records, no over-limit entry, and peak driver allocations of about 11.13 and 11.03 GiB.

## Small-edit metrics from the saved scores

These are raw descriptive results pooled across alpha `0.05` and `0.10`, four fit cases per site. The zero predictor's error is the magnitude of the measured score change. All eight measured changes per site are nonzero as stored; no numerical-noise floor was applied.

| Audio source site | Matching signs | Median absolute error: VJP / zero | Median error reduction | Mean absolute error: VJP / zero | Mean error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Connector | 8/8 | 0.0888 / 0.1753 | 49.3% | 0.0993 / 0.1972 | 49.6% |
| Encoder 15 | 5/8 | 0.0964 / 0.0879 | −9.7% | 0.0937 / 0.1110 | 15.6% |
| Encoder 31 | 6/8 | 0.0783 / 0.1793 | 56.3% | 0.1516 / 0.1756 | 13.6% |

The connector clears the plan's raw sign and 25% error-reduction thresholds. Encoder 15 does not. Encoder 31 meets the raw **median** reading of the plan (6/8 signs, 56.3% lower median absolute error) but misses a **mean** absolute-error reading (13.6% reduction). The plan says to record median absolute error, yet its threshold says “lower absolute error” without fixing whether reduction is computed from medians or means. Either way, a robust connector-plus-encoder locality pass is **unresolved**, given the numerical issue below; do not treat the alpha-one flips as evidence that the local VJP predicts the nonlinear full swap. At alpha one, the donor answer flips 4/4 per site while the prediction has the wrong sign in two connector cases and two encoder-31 cases.

## Execution-mode limitation

The first attempt used a grad-enabled VJP score as the baseline for no-grad patched scores. Its zero edit changed the measured contrast by `0.00792` despite no state change. The revised script correctly uses fresh **no-grad unpatched scores** as the finite-difference baseline, making alpha zero exactly zero, and separately records the grad-enabled VJP baseline. The two score paths still differ by up to **0.0726 log-probability units** across these cases, comparable to several alpha-0.05 measured changes. Thus the gradient is taken along a grad-enabled forward path while the saved finite differences are measured on a no-grad path. Alpha-zero equality alone does not show that those paths have the same local derivative. A matched-path small-step check and a prespecified numerical-noise floor are needed before calling the observed sign/error agreement a validated local Jacobian prediction.

This is an original red/blue fit-only assay, with four related voice/direction routes per site. It has not calibrated the plan's norm-only comparator or run the wrong-time, shuffled, random, development, or locked-test controls. It does not constitute a reusable corpus-fitted audio Jacobian lens or a security finding.
