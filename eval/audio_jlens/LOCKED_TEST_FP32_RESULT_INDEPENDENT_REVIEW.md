# Independent saved-result review: one-shot Qwen FP32 locked test

Reviewed 2026-11-18 against frozen `PLAN_FP32_V1.md` (`b171278`) and locked-test runner (`31b9fb5`). I audited the saved result, progress, memory, test manifest and prerequisite hashes. I did **not** run model inference or rerun the test.

## Verdict and denominator

The result **passes the prespecified gate at both sites among clean-eligible directions**. The all-planned result is narrower: **10/16 directions per site** were clean-eligible, and **10/16** full α=1 patches produced the exact donor word plus EOS. All **10/10 clean-eligible** patches did so. The local Jacobian comparison covers **30/48 planned small-step conditions per site**, all of which were determinate and had the predicted sign.

| Site | Clean / planned directions | Correct determinate signs | Jacobian MAE | Zero MAE | Frozen norm MAE | Complete donor flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Connector | 10/16 | 30/30 | 0.001358 | 0.040740 | 0.040740 | 10/16 planned; 10/10 eligible |
| Encoder layer 31 | 10/16 | 30/30 | 0.001783 | 0.032969 | 0.032969 | 10/16 planned; 10/10 eligible |

The frozen fit floor is `0.0001` at both sites. Jacobian MAE is 3.33% of the comparator at the connector and 5.41% at encoder31, below the required 75% ceiling. The frozen norm coefficient is zero at both sites, so its comparator is numerically the same as the zero-change comparator. I independently recomputed these metrics, medians, sign counts, and flip denominators from the raw case rows; they agree with the saved gate summaries.

**Why six directions are excluded:** Three of the eight clean pairs answered the number words as digits `7` and `8`, each with EOS: Piper `number/codeword`, Piper `number/term`, and Kokoro `number/codeword`. The frozen rule requires exact decoded `Seven` and `Eight` word forms plus EOS, so both directions of each pair are clean-ineligible. These digit answers may be semantically correct, but they do not satisfy the precommitted answer surface or the corresponding teacher-forced word-plus-EOS readout. The eligible set consists of all eight direction-word directions plus two Kokoro `number/term` directions. Do not present the result as 16/16 transfer or as broad number-category transfer.

## Integrity checks

- Final result SHA-256 is `5ab129d1f77f68811a9c4014ff677e6f4e6ea1926ff77869b90a11a25f038f84`. Script, plan, fit result/progress/baseline, development result/progress/code, test manifest, and imported helper SHA values all match the saved report and current frozen files. The 32-line progress JSONL exactly matches the 32 result case dictionaries; its SHA-256 `49f0f967c17fe6eefe4ab5327f35b2ace2bdff7e65c6150ce73aab41d0767b51` is recorded in the result.
- Every case's recipient/donor WAV digest and candidate-plus-EOS token IDs match the locked manifest and frozen token table. All 16 category × phrasing × voice × site groups contain both patch directions; saved category/phrasing/voice denominators agree with the raw cases.
- Every alpha-zero patch has zero effective edit and reproduces the clean generated token IDs and score within the frozen tolerance. All alpha-grid generations end on EOS and report one patch hook call. Score contrast and measured score-delta arithmetic are exact; the largest difference between predicted delta and its saved per-row sum is `3.71e-7`. Cached/uncached prefill state difference is zero; the largest grad/no-grad intercept gap is `4.24e-5` log-prob units.
- The memory sidecar contains 449 observations, with a maximum MPS driver allocation of **21.186 GiB** and no reported 30 GiB limit violation.

## Controls and limits

All seven prespecified control names ran on each of eight `codeword` directions per site: **112/112 control rows eligible**. No control produced the *exact donor word* plus EOS. For clean-eligible `codeword` directions, all 56 control generations retained the recipient word plus EOS. The other 56 controls belong to clean-ineligible number/codeword directions and often retained digit answers; some produced other digit strings. They cannot be counted as selective word-form control successes. The one clean-eligible number pair uses the `term` phrasing, for which controls were not planned.

This result validates a local derivative predictor of the **complete word-form answer score** on the eligible locked-test inputs and reproduces donor answers under full state swaps there. It does not identify a reusable corpus-fitted Jacobian Lens, show cross-model transfer, prove a unique word-local circuit, or demonstrate a security violation. The clean failure on three number pairs is a material generalization limit and must accompany any headline pass rate.
