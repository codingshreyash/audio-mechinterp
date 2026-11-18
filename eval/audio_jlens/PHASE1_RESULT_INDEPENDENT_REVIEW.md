# Independent saved-result review: Qwen FP32 phase-1 fit

Reviewed 2026-11-18 against `PLAN_FP32_V1.md` at commit `b171278b51e3ee577a31b83161a9b9b2dcc502ed`. I inspected only the saved fit result, progress, baseline, memory sidecars, and code. I did not run MPS or open development or locked-test inputs.

## Verdict

The saved result **passes the frozen fit-only stop rule** at both primary sites. This permits the planned development evaluation; it does not establish transfer, a reusable audio Jacobian Lens, a localized word circuit, or a security finding.

| Site | Clean fit routes | Complete donor flips at α=1 | Determinate small effects | Correct local signs | Numerical floor | Fit mean absolute local error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Connector | 4/4 | 4/4 | 12/12 | 12/12 | 0.0001 | 0.000834 |
| Encoder layer 31 | 4/4 | 4/4 | 12/12 | 12/12 | 0.0001 | 0.004323 |

All four clean generated answers are the expected exact color word with EOS. Every α=1 condition generated the donor color with EOS. I independently recomputed floors, flip counts, sign denominators, mean absolute local errors, and norm-baseline coefficients from raw case rows; they agree with the saved summaries. `fit_only_gate_passed` and `development_permitted_by_fit_gate=true` are therefore consistent with the frozen fit rule.

## Integrity and arithmetic

- Eight distinct planned cases are present: Piper and Kokoro, red→blue and blue→red, connector and encoder31. The progress JSONL contains the same eight case dictionaries, in order, as the final JSON. Its SHA-256 `2de5082c6e4b954938f02615469e096e48ed3d9d1a6ae87cc78aba6e1a66b538` matches both the result and baseline sidecar. The result's script, plan, fit manifest, imported helper, and baseline hashes also match their current files. The runner SHA-256 is `f1ae3e697221dc68ef21df847304b38adb1b303a7c65ddf336e2f2e3050cf113`.
- Alpha zero reproduces unpatched score and generated token IDs in all eight cases, with zero effective edit. Repeated-zero score deviation is zero in all cases. Cached/uncached prefill state difference is zero. The largest grad-minus-no-grad intercept difference is `1.90e-5` log-prob units.
- Score contrast and measured score-delta arithmetic agree exactly with the saved component values. The largest difference between saved predicted delta and the sum of saved row contributions is `2.74e-7`, consistent with float conversion/summation order. All 56 alpha condition generations and all 56 eligible control generations end on EOS and report exactly one patch hook call. The memory sidecar has 165 records; maximum observed MPS driver allocation is **20.705 GiB**, with no 30 GiB limit violation.
- The fit-only norm baseline recomputes to `β=0` at both sites because the signed `Σ n·z` numerator is negative (`−0.07634` connector; `−3.25149` encoder31). Thus this baseline predicts zero on later splits and is identical to the zero-change comparator there. Preserve that fact when evaluating the plan's separate development error gate.

## Control and locality limits

All seven planned controls ran on each of the eight site/routes; no control was ineligible. Natural off-word patches and self patches **never** produced a complete donor answer. Every selected off-word window is wholly in a chunk disjoint from the word-row chunk, as the frozen plan requires.

Other controls **did** produce complete donor answers:

| Site and control | Donor answers |
| --- | ---: |
| Connector, row-shuffled donor word edit | 3/4 |
| Encoder31, norm-scaled off-word synthetic edit | 1/4 |
| Encoder31, random same-row seed 1 | 1/4 |
| Encoder31, random same-row seed 2 | 1/4 |

All six listed flips ended on EOS. The connector row-shuffle result means the exact order of donor row vectors is often unnecessary for the full-swap flip on this pair. The encoder31 synthetic off-word and random flips show that large, norm-matched interventions can drive the answer without the natural donor word-row pattern. These results limit a narrow word-local or uniquely semantic interpretation, even though the natural off-word control did not flip.

The small-step fit is genuinely local: for Kokoro, all three small measured deltas in each direction are negative at both sites, while the full α=1 replacement produces a large positive donor-oriented score change and donor answer. The frozen sign gate checks derivative agreement with the **local** measured change, not monotonic movement toward the donor answer over the whole edit path. No full-swap effect should be attributed to the local derivative alone.
