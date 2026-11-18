# Independent result audit: full Voxtral v2 FIT causal assay

**18 November 2026.** Read-only CPU audit of [the completed eight-record result](runs/voxtral_fit_v2_causal_full/20261118T0928Z/result.json) with the [frozen independent auditor](audit_voxtral_fit_v2_causal_full.py), SHA-256 `747b9aa1c6340badd82562345c226964f907d56066466ac28b76ed73a8cb80d2`. I did not run Voxtral inference or open development/locked-test WAVs or answers. This result uses four FIT lexical pairs, two synthetic TTS voices, and both word directions: **16 planned cases**.

## Integrity and resource audit

The auditor passed at result SHA-256 `e29a686e30505a69a3e4b26bddde606a02db753a3a5410c38432cdd14ac3baaf` and metadata SHA-256 `90a46945c2036efd286f9c842b5900bd5fdf5a484c55253da0ff4f9cb18e1d21`. It verified the committed runner, frozen design/masks/selection/index, audited first-record result, selected WAV bytes, local model checkpoint and installed source hashes, prompt IDs, and CPU-reconstructed native/blank/six-condition feature hashes. Across eight immutable checkpoints, every one of the **864 planned slots** is present exactly once: **64 source captures, 800 complete generations, 1600 teacher-forced candidate-score forwards, zero invalid controls**. Hook calls, BF16 shapes, selected rows, realized control norms, random seeds, strict full-answer classifications, saved score subtraction, and natural/blank passive and self-patch token-ID parity all match the frozen rules. The auditor independently recomputed every eligibility, answer, score-comparison, pair/voice, and gate count from the saved records; the recomputed summary equals the terminal summary.

The last checkpoint records **670.76 seconds** of cumulative work, below the **2700-second** cap. Peak saved MPS driver allocation is **10,671,669,248 bytes**, below **32,212,254,720 bytes** (30 GiB). All checkpoints, attempt hashes, and the invocation memory log are present; this run has no failed-attempt debt receipt to reconcile. The saved scalar candidate log probabilities cannot be independently recomputed without another model forward; this audit checks their arithmetic and the pinned teacher-forcing code rather than rerunning the model.

## Prespecified FIT gates

| Endpoint | Encoder31 post-block | Post-projector connector |
| --- | ---: | ---: |
| Blanked bases eligible for restoration | 16/16 | 16/16 |
| Whole-word restoration to exact recipient + EOS | **16/16; effect gate passes** | **16/16; effect gate passes** |
| Whole-word lexical loss to valid other/opposite + EOS | **15/16; effect gate passes** | **15/16; effect gate passes** |
| Opposite-word clean donor yields opposite + EOS | 16/16 | 16/16 |
| Restoration time-preference gate | **Fails** | **Fails** |
| Loss time-preference gate | **Passes** | **Passes** |

The lexical-loss output was `None` in **15/16** directions at each site; the remaining case produced a quoted, malformed `"chose"`/`"Chose"`. Thus the 15 successes are valid **other** answers under the frozen classifier, not transitions to the opposite target word. The opposite-donor control separately demonstrates content-specific opposite-word changes in all 16 directions at both hooks. All pre/post/padding/random controls caused **0/16 complete restorations** at either site. For loss, pre/post/padding controls caused **0/16** valid lexical losses; random caused **1/16** at encoder31 and **2/16** at the connector.

The restoration time-preference gate is false for a specific preregistered reason: the blank/control outputs are often quoted or otherwise malformed, leaving only **1–9 valid finite paired score comparisons of 16** per comparator; the gate requires at least **14/16**. Whole-word answer success still exceeds every control, but the continuous-score comparison cannot be labeled as passing. For loss, each comparator has **15/16 valid comparisons and 15/16 strict whole-word score wins**, so the loss time-preference gate passes at both sites. The four quarter-word bins remain descriptive: `word_1` alone restored **15/16** at each site; no bin was selected post hoc or promoted to a primary endpoint.

**Verdict:** No provenance, counting, parity, or resource blocker found. The FIT assay establishes strong causal effects of the specified **constructed mel blanking and hidden-state patches** on complete lexical answers in this one Voxtral model and two synthetic voices. It does not establish a naturally reachable waveform perturbation, a unique or exhaustive word circuit, tool-call integrity failure, a security vulnerability, or held-out generalization. A separately frozen development protocol is needed before applying these observations beyond FIT.
