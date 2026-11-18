# One-prompt J-lens cost benchmark

**Status:** The loaded-model gate and bounded 16-coordinate pilot passed locally on MPS; see [JACOBIAN_BENCHMARK.md](JACOBIAN_BENCHMARK.md). No full 2,048-coordinate transport or corpus lens has been fit. The earlier 0.153-second single-logit backward is not directly comparable to the reference estimator's batched final-residual coordinate backwards.

## Purpose and prerequisites

Determine actual time, peak memory, and numerical behavior for an audio-conditioned Jacobian estimate on the pinned [3B Thinker](README.md). This is a cost and plumbing gate, **not** a fitted or validated interpretability result. The pilot used local MPS after the scaffold review; it did not require a cloud lease.

## One controlled input

Use one 2–5 second, 16 kHz mono WAV with a known transcript and a fixed, recorded system instruction. Record its byte hash, exact processor batch shapes, model and library revisions, prompt template, random seed, device, precision, attention implementation, and whether `use_cache` is false. Use the Thinker only; no Talker or speech decoder is needed to fit a text-vocabulary readout.

## Measurements

1. Load model/processor from the pinned revision. Freeze all parameter gradients, use `eval()`, and make one ordinary audio-to-text forward pass. Confirm the answer, finite logits, actual audio placeholder count, and no-hook baseline. Repeat with detached observation hooks; complete logits should match within the chosen BF16 tolerance.
2. Check encoder, projection, and text-layer activation shapes and whether gradients from a selected output token reach each boundary. A zero or missing gradient is a failed gate, not a runtime estimate.
3. **Completed:** two selected source layers (0, 17), 16 target coordinates at `dim_batch=8`, with explicit MPS synchronization and a 30 GiB driver-memory guard. The audio-conditioned source mask selected 137 placeholder positions. The script resumes after each saved eight-coordinate batch. This is a cost and plumbing probe; its position average is not a terminal-position semantic lens.
4. **Pending:** fix a source/target position policy and decide whether a full one-prompt fit for width 2,048 is worth the measured cost. `ceil(2048/8) = 256` batched backward passes. The 500-second naive extrapolation is upper-bound-ish because the pilot repeats forward passes, while a graph-reuse fitter can do repeated backwards from one forward. Benchmark `dim_batch=16` only if memory permits, then compare measured full fit to the pilot estimate.
5. **Pending:** compare audio-conditioned fitting and readouts against a text-only lens and held-out audio tasks. The stock Anthropic fitter accepts text prompts, so the completed audio pilot used a local adaptation rather than `jlens.fit()` directly.

On local MPS, use `torch.mps.synchronize()` around timings and record allocated, driver, and recommended memory. On a future CUDA run, use `torch.cuda.synchronize()` and reset peak stats. Record exact torch, Transformers, platform, precision, attention implementation, input SHA-256, and command. Never infer the full corpus cost solely from parameter count.

## Scale decision

The [Anthropic reference fitter](https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py) estimates one forward plus `ceil(d_model / dim_batch)` backward passes per prompt. At width 2,048 and `dim_batch=8`, 100 prompts would entail about 25,600 batched backward passes. Fitting all 36 full-rank float32 transport matrices would require roughly 604 MiB for those matrices alone, before model weights and activations. These are arithmetic estimates, **not** a measured full fit. Scale to multiple prompts and layers only after a full one-prompt measurement and the [readout/causality gates](../../eval/PROTOCOL.md) are reviewed.
