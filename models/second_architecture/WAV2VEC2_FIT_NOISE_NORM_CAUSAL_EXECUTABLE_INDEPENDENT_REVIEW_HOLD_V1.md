# FIT normalization executable: independent review HOLD

19 November 2026. I independently reviewed committed executable source `7d09a74` against the prospective study and its reviewed 88-forward amendment. **HOLD before admission or candidate-model inference.** No pinned checkpoint was loaded or run in this review.

## What passed

- The fixed order covers eight Stage I pairs and four Stage II word-noise pairs. The four clean waveforms have one no-hook/read-only stock pair each, and all eight distinct noisy waveforms have their own pair: 24 Stage I control forwards, 32 Stage I scored forwards, 16 Stage II controls, and 16 Stage II scored forwards, totaling **88**. The saved auditor checks this exact name order and pair boundaries.
- The Stage I diagonal uses the pinned processor formula and byte-equality check; stock and read-only logits must match bitwise. The Stage II raw-convolution and GroupNorm statistic donors come from the same pair's shared-input clean/noisy path. The worker requires exact outside-slot raw equality, natural-diagonal GroupNorm/logit tolerances, exact full CTC path parity, and crossed clean-statistic outside equality.
- The saved auditor reads all 88 complete logits files, recomputes CTC paths, reconstructs first convolution from pinned weights and normalized PCM, verifies raw donor hashes, and reconstructs GroupNorm statistics/edits. Later Stage I feature-projection and post-block-0 traces are only hashed summaries, and the auditor says so.
- Source/weights/processor version binding and later committed GO ancestry fail closed in the inspected code. The one-shot output path is exclusive; the parent has a 1,200-second timeout and archives pre/post power evidence.
- `models/second_architecture/.venv/bin/python -m unittest -q models.second_architecture.test_wav2vec2_fit_noise_norm_causal_worker_v1 models.second_architecture.test_wav2vec2_fit_noise_norm_causal_88_plan_v1` passed 7 no-weight tests, including a fake 88-forward saved audit and altered-file rejection. `pytest` is absent from this venv; the tests are `unittest` tests.

## Required fixes before GO

1. **Enforce the stated 4-GiB process RSS ceiling during each forward.** `wav2vec2_fit_noise_norm_causal_launcher_v1.py:22-49` blocks in `process.wait(timeout=seconds)` and monitors time only. The worker's `Budget.check` runs before/after a forward and reserves 768 MiB, but cannot observe a transient or sustained peak inside a forward. Add a parent process-tree RSS poll, record the peak and limit reason, and kill the process group at the ceiling. Add a fake-child cap test and require the saved auditor to verify this parent measurement. This is a resource-admission defect, not a claim that the current small model would exceed 4 GiB.
2. **Name the censored calls after early failure.** `wav2vec2_fit_noise_norm_causal_worker_v1.py:518-525` records only the aggregate `remaining_censored`. The frozen prospective protocol requires all remaining cells censored in the ledger. Add the ordered remaining call names from the 88-name plan, distinguish a failed attempted call from never-attempted calls, and test a mid-pair failure. Preserve the existing one-shot failure and no-retry rule.

After those fixes, freeze a new source commit and request independent executable rereview. Only then create a fresh no-weight admission receipt and a separately committed GO binding its exact hash. This HOLD does not authorize the real run.
