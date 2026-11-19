# Independent result review: Whisper natural-tail V4

19 November 2026. **PASS for both complete V4 attempts and their saved audits. This is an engineered non-speech model-behavior result, not a security vulnerability or proof of padding mediation.** I reviewed both attempt directories read-only, using the frozen V4 source, final `_V4_2.json` admissions, and model-specific GO receipts. No candidate inference was run for this review. The invalid V3 multilingual attempt remains invalid and unchanged.

## Evidence and replay

Both fixed attempts have 160 ordered calls, 320 ordered progress entries, an explicit 160-call terminal record, 800 lossless sidecars, zero base-replay failures, worker exit code 0, parent `completed_pending_saved_audit`, and saved-audit `PASS`. I reran the V4 saved auditor read-only for both models. It reverified source WAVs, all sidecar SHA-256/value hashes and shapes, full first-step vectors, IDs/EOS/censors, host windows and archives, process-tree RSS samples, evidence ceilings, replay parity, and 120 planned pair comparisons per checkpoint. The recomputed comparisons match the saved audits exactly **by** `(case_id, kind, length, seed)`; their JSON row order varies because the auditor iterates a set of case IDs. This order difference is not a changed result.

The independently generated [raw-file inventory](WHISPER_NATURAL_TAIL_V4_RAW_FILE_INVENTORY.json) records SHA-256 and byte length for every file: 812 files per model, including all 800 sidecars and the saved audit. I rehashed all 1,624 inventoried files after committing the compact inventory and the two saved audit JSONs at `1aea23d`; all matched. The raw sidecars remain in their immutable local attempt directories. Multilingual parent-result SHA-256 is `da42b6791022f5e01e971898bd263756b6aea654cf1baa543c34965f0f9004ba`; English is `f9ee8c2ac65b6c992fd1fe586634a06fd57b16cf75ff5b1aff6430b211c553a2`. Peak sampled tree RSS was 1,011,924,992 and 846,528,512 bytes; whole-attempt evidence before the parent result was 228,318,232 and 227,197,189 bytes; whole elapsed time was 31.09 and 23.26 seconds, respectively. Both remained below the frozen 4-GiB, 1-GiB, and 1,800-second ceilings, with host evidence passing.

## What changed

| Comparison | Multilingual Whisper Tiny | English Whisper Tiny |
| --- | ---: | ---: |
| Zero tail versus original, complete sequence changes | 0/40 | 0/40 |
| Noise tail versus equal-duration zero tail, complete sequence changes | 12/80 | 4/80 |
| Noise/zero changes in primary 18 non-silence cases | 10/72 | 4/72 |
| Noise/zero changes in two silence controls | 2/8 | 0/8 |
| 128-token length censors among 160 calls | 3 | 0 |

All 40 zero-tail pairs per model have identical 30-second padded PCM and identical 80×3000 mel tensors, even though their processor attention masks differ. None changed the complete output sequence. For all 80 equal-duration noise/zero pairs per model, the mask stayed the same; mel differences occurred in the appended-tail and boundary regions, with **zero changed frames** in the original-source interior and nominal-padding interior. The derived global log-spectrogram floor moved in 16/80 noise/zero pairs per model (silence and dither), yet nominal-padding mel still stayed identical. This ordinary-WAV run therefore gives no evidence that a changed nominal-padding feature caused the observed answer changes.

The multilingual model's 12 sequence changes include two punctuation-only changes, three 128-token bounded repetitions, and three detected-language-prefix changes. Five source/length groups changed with both noise seeds; two more changed with only one seed. The English model's four changes are capitalization only, in two two-seed groups. The multilingual silence-control pair at 8 seconds plus a 2-second noise tail changed text and detected language for both seeds, showing that the appended noise can matter without source speech. A 128-token censor is a bounded output, not an endless loop.

## Interpretation limit

The same edited WAVs produce different sensitivity across the two pinned checkpoints, but checkpoint training and language configuration differ; this does not isolate an architecture cause. The feature comparison localizes the changed ordinary input to active tail/boundary mel frames for these cases. It does not establish a causal internal circuit or an externally exploitable effect. These are synthetic, non-speech sounds; the ±32-LSB tails and added durations have not been shown imperceptible. A next mechanistic test would intervene on the active tail/boundary representation while holding other model states fixed and measure complete EOS-qualified output, with fresh preregistration and an independent review.
