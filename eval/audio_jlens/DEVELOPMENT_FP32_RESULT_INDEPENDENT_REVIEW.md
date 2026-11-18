# Independent saved-result review: Qwen FP32 development

Reviewed 2026-11-18 against `PLAN_FP32_V1.md` at `b171278` and the frozen fit baseline. I audited only saved JSON/progress/memory artifacts and source hashes; I did not run Qwen inference or inspect locked-test inputs.

## Verdict

The saved development result **passes the prespecified locality gate at both primary sites**. I independently recomputed every gate denominator and metric from the 32 saved site cases. This supports a local audio-state-to-complete-answer score predictor across these new words and phrasings in the two previously used voice families. It is not yet a reusable corpus-fitted Jacobian Lens or a security finding; locked-test transfer remains unmeasured in this review.

| Site | Planned / clean-eligible directions | Determinate signs | Jacobian MAE | Zero MAE | Frozen norm MAE | Complete α=1 donor flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Connector | 16 / 16 | 48 / 48 | 0.000814 | 0.022941 | 0.022941 | 16 / 16 |
| Encoder layer 31 | 16 / 16 | 48 / 48 | 0.002662 | 0.019875 | 0.019875 | 16 / 16 |

The frozen numerical floor is `0.0001` at each site; all 48 clean small-step effects per site exceed it. Jacobian MAE is 3.55% of either comparator at the connector and 13.39% at encoder31, below the required 75% ceiling. Both frozen norm coefficients are zero, so the norm and zero baselines coincide here. Report them as two prespecified comparisons with identical numerical predictions, not as independent baselines.

## Integrity and arithmetic

- Final result SHA-256: `0eb01bab4eff1dfaa2c4c2082b25e51fae8d5b91909afaf5fcd222076a48f769`. The saved development script, plan, fit result, fit progress, fit baseline, development manifest, and imported helper hashes all match their files. The completed 32-line progress JSONL exactly matches the final result's 32 case dictionaries; its SHA-256 `9ddfaa974b174d8adbdfd9efa76e7280e1611d29fcf1dff205aee5843bcbbc07` is recorded in the result.
- All eight development pairs have two clean exact color/category answers under case folding and EOS, yielding 16 clean-eligible directions per site. Source/donor audio hashes and candidate-plus-EOS token IDs in every case match the bound development manifest and token table. All category × phrasing × voice × site groups contain both directions; the saved breakdown counts agree.
- Alpha-zero cases retain the clean generated token IDs and zero effective edit; cached/uncached prefill difference is zero. Every alpha-grid generation ends on EOS and reports one patch hook call. Contrast and measured-delta arithmetic are exact; the largest difference between a saved predicted delta and the sum of its row contributions is `2.87e-7`. The largest grad/no-grad intercept gap is `2.03e-5` log-prob units.
- The memory sidecar contains 441 records; maximum observed MPS driver allocation is **21.217 GiB**, with no reported violation of the 30 GiB cap.

## Controls and interpretation

Controls ran on the fixed `chosen` phrasing only: eight directions per site, seven planned control names each. Of the 112 planned control rows, 104 were eligible. The eight ineligible rows are the norm-scaled off-word synthetic control where its natural off-word delta is zero (four per site). All eligible control generations ended on EOS with one hook call.

Only one control produced a donor answer: the connector row-shuffled donor edit for Piper `animal/chosen`, dog→cat. Natural off-word, self, random same-row, and eligible scaled off-word controls did not produce donor answers in these development cases. This supports selectivity for these inputs but does not prove that only a precise word-row pattern can cause the behavior; the shuffled case is a concrete counterexample. The fit controls also had donor flips under noncanonical edits and remain a limit on localization claims.

The two voices were already present in fit, while the words and phrasings are new. The result tests transfer across lexical items and prompt wording within this model/task family; it does not establish cross-architecture transfer, natural perturbation robustness, or downstream security impact.
