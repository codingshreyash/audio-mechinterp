# Fit-only precision and audio-Jacobian microstep findings

The question was whether a Jacobian at an audio representation predicts how a **complete answer** changes when that representation is edited. All data here are the original, known `red`/`blue` Piper fit pair in the pinned Qwen2.5-Omni-3B Thinker. The unseen audio-Jacobian test clips were not run. This is a numerical and causal-method validation, not a security finding or a corpus-fitted Jacobian Lens.

## What the runs established

The BF16 fit gate used exact-PCM paired WAVs and the same nominal spoken-word rows at each source site. In [all four fit routes](fit_gate_v1_all_routes_results.json) and the separate [layer-31 routes](fit_gate_v1_encoder31_results.json), a full donor-state patch changed the **complete, EOS-terminated** answer at connector, encoder layer 15, and encoder layer 31: **4/4 per site**, 12/12 total. All alpha-zero patches reproduced the same answer and no-grad complete-answer score. The [independent fit review](FIT_RESULT_INDEPENDENT_REVIEW.md) checked the saved outputs, per-row sums, source hashes, and memory logs.

That full-swap effect does not make the local derivative reliable. At BF16, grad-enabled and no-grad score paths differ at alpha zero; the first script incorrectly mixed those baselines and stopped. Its failure and retry memory logs are preserved. The corrected no-grad intercept removes the false zero-edit effect, but [same-path finite differences](fit_fd_diagnostic_v1_results.json) still disagree with the VJP at several small edits. For example, at connector alpha 0.01 the prediction is +0.0343, while grad-mode and no-grad measured changes are +0.0081 and +0.1184. Repeated unedited scores are bitwise stable, so this is not sampling noise. BF16 rounding and downstream execution-mode numerics are material confounds; these observations alone do not apportion every discrepancy between them and curvature.

An [FP32 rerun](fit_fp32_diagnostic_v1_results.json) sharply reduces the execution-mode gap. For connector alpha 0.01, predicted and measured no-grad changes are +0.03081 and +0.03073, respectively. The subsequent [FP32 microstep run](fit_fp32_microstep_v1_results.json) saved its full source, donor, and gradient tensors in a [state sidecar](fit_fp32_microstep_v1_results.states.pt), SHA-256 `fac209e1cea22dedd0c2cb3554efa9954f3a390b630366110a86412124acadb7`. An independent CPU reload recomputed every reported gradient–edit dot product and edit norm from those tensors.

| FP32 site, Piper red→blue | Edit fraction | Jacobian-predicted Δ complete-answer score | Measured no-grad Δ score |
| --- | ---: | ---: | ---: |
| Connector | 0.01 | +0.03081 | +0.03073 |
| Encoder layer 15 | 0.001 | −0.00855 | −0.00903 |
| Encoder layer 15 | 0.01 | −0.08550 | −0.06294 |
| Encoder layer 15 | 0.025 | −0.21374 | −0.06624 |
| Encoder layer 15 | 0.10 | −0.85497 | +0.05413 |
| Encoder layer 31 | 0.001 | +0.003118 | +0.003092 |
| Encoder layer 31 | 0.01 | +0.031181 | +0.030753 |
| Encoder layer 31 | 0.025 | +0.077952 | +0.074836 |

The layer-15 FP32 derivative converges at the tiniest measured step but quickly loses quantitative accuracy and eventually even its sign. Layer 31 stays locally predictive over the measured microsteps on **this one fit route**. The result is consistent with increasing curvature or interaction depth early in the encoder; a microstep curve cannot by itself identify a specific neuron, attention head, or circuit. Casting FP32 states to BF16 in the saved sidecar is labeled hypothetical and must not be confused with activations from the BF16 model.

## Decision

The original BF16 locality gate is **unresolved**, not a transfer success. The FP32 connector and late-encoder behavior motivates the separately frozen [FP32 validation plan](PLAN_FP32_V1.md), committed as `dbef46b` and clarified in `b171278` before any development or locked-test Jacobian run. That plan fixes fit, development, and locked-test splits; alpha grid; both primary sites; baselines; complete-answer scoring; controls; and pass criteria. The next step is four-route FP32 fit calibration. Any failed gate remains a result; it does not justify inspecting the locked test or retuning against it.
