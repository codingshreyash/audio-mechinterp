# Independent FP32 old-FIT numerical result review

18 November 2026. **Verdict: the frozen FP32 arm is internally consistent as a one-case numerical diagnostic.** This review does not establish a reusable Jacobian lens, audio-time localization, model-behavior efficacy, or a security finding.

## Provenance and recomputation

I reviewed the committed [FP32 attempt](runs/voxtral_v1_fit_backward_numerical_launch_v1/fp32/20261118T120909312905Z) from commits `d6e1e70` and `6eadd6e`, including the 6.6 MB tensor sidecar. All nine attempt files matched their committed `HEAD` blobs. The external freeze receipt SHA-256 is `a487b199b28d80b70c58c4732c8d2d53e64c423226e15d76bf5c91c624bc5df8`; the launch start and finish both bind it. The launcher review commit `5c4997d57373ba3656a05659413169edc1b463bd` is an ancestor of `HEAD` and did not itself contain the freeze receipt. The pinned-venv `--check-freeze` returned `verified_external_freeze_only` with the same receipt hash.

I reran the frozen CPU auditor on the committed `runner` directory. It returned `audited_completed_numerical_consistency_only`, with result SHA-256 `b1a124c9df14c980af18dee46cc058a98cc167a1269f33252d0e0980fc7e37fb` and tensor sidecar SHA-256 `5ceb99186098360351a673b430e4bd9108eafc7457ce6ca892134340fcfdaac9`, matching the saved audit and launch finish. It verified the pinned old-FIT donor direction, shape/dtype/finiteness, source edit arithmetic, gradient contractions, all 36 paired zero/edit observations (72 patch forwards), three cycles per signed step, progress order, and reported resource caps. The runner finished in 80.70 seconds; the observed MPS driver peak was 28,793,487,360 bytes, below the 30 GiB cap.

## Measured numerical behavior

The paired zero terminal states had zero observed variation across 18 comparisons at each site, both on the selected four coordinates and on the full terminal vector. All six signed step groups at each site had nonzero measured change above the protocol's repeatability floor. The table gives the median-over-three-repeats **relative L2 error** of the rowwise local-Jacobian prediction on the four selected LM29 coordinates. The denominator is the measured change norm, with the protocol's noise/epsilon floor.

| Site | +0.00625 | −0.00625 | +0.025 | −0.025 | +0.1 | −0.1 |
|---|---:|---:|---:|---:|---:|---:|
| Encoder 31 | 0.176 | 0.165 | 0.089 | 0.056 | 0.037 | 0.035 |
| Projector | 0.447 | 0.544 | 0.130 | 0.201 | 0.083 | 0.060 |

The median relative errors across the six step groups were 0.072 at encoder 31 and 0.165 at the projector. Realized FP32 edit-mean error ratios were about `9.2×10⁻⁷` to `1.57×10⁻⁵` at encoder 31 and `5.3×10⁻⁷` to `8.6×10⁻⁶` at the projector; these values measure **deviation from the intended edit**, so smaller is better. The projector kept six of 9,216 selected source elements unchanged by FP32 addition rounding; encoder 31 changed every selected element. At the smallest step, the measured even component of the ±step response was 0.160 of the odd component at encoder 31 and 0.567 at the projector, so the local-linear approximation is imperfect even though repeats are deterministic.

Compared with the earlier one-step BF16 pilot's roughly 100% four-coordinate relative prediction error, this FP32 result is **consistent with finite-precision effects being a major contributor**. It does not isolate precision as the sole cause: the whole model, intermediate states, and backward path were run in a different arithmetic mode, the donor direction still originates from the old BF16 capture, and the smallest-step projector error remains 0.45–0.54. The separate BF16 arm under this follow-up froze before any interventions because Voxtral intentionally retains one audio positional embedding in FP32; its failed launch is preserved in commit `74f4f58`. A narrowly amended BF16 run is needed for a matched comparison.

The auditor recomputes arithmetic from saved tensors and checks their provenance fields; it does **not** independently rerun model forwards. Its donor and no-grad terminal tensors are shape/dtype/finiteness checked but not independently content bound, and the clean/donor source hash strings are report-only. The result applies to one fit-informed beaver→monkey direction, two internal sites and four raw target coordinates. It must not be used to reopen the failed Voxtral development gate or the sealed locked test.
