# FIT normalization causal study: 88-forward control amendment

19 November 2026. **Prospective source-only amendment; no candidate weights run.**
This document supersedes only the 80-forward schedule in
`WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_PROSPECTIVE_V1.md` and its
independent source review `35a5fcc`. The interventions, frozen eight FIT
pairs, expected CTC endpoints, tolerances, model pin, resource ceilings,
claim limits, and no-retry rule remain as specified there. A new independent
conceptual and source review is required before any model inference.

The original count treated all clean/noisy controls in the eight Stage I
pairs as 16 forwards. There are 12 **unique** Stage I stock waveforms:
four source-case clean recordings, four word-noise versions, and four
outside-noise versions. Each must have both a no-hook stock forward and
a read-only input-hook forward to validate both natural diagonals and hook
parity. The corrected fixed count is:

| Forward class | Count |
| --- | ---: |
| Stage I 2×2 scored cells, eight pairs | 32 |
| Stage I no-hook stock, twelve unique waveforms | 12 |
| Stage I read-only stock, twelve unique waveforms | 12 |
| Stage II 2×2 scored cells, four word-noise pairs | 16 |
| Stage II no-hook shared-input controls | 8 |
| Stage II read-only raw/GN captures | 8 |
| **Total hard ceiling** | **88** |

For each source case, run clean stock controls once at the first
`word_noise` pair and reuse their **saved complete logits** for that case's
`outside_noise` clean diagonal. Run each noisy variant's controls
separately: `word_noise` and `outside_noise` have different PCM and cannot
share a stock result. Thus each first pair makes eight forwards, each
second pair six, and each Stage II pair eight. Four × (8+6) + four × 8
= 88. The worker records all controls and all scored cells, including
the six pairs with unchanged prior complete transcripts. No saved
historical capture substitutes for these prospective controls.

The worker, launcher, admission receipt, and auditor bind this amendment
hash and enforce 88 attempted forwards. Any error censors the remaining
scheduled cells under the one-shot attempt. A new attempt requires a new
prospective protocol and review. The previous 80-forward source plan and
its `35a5fcc` review remain immutable evidence of the detected counting
mistake and do not authorize execution.
