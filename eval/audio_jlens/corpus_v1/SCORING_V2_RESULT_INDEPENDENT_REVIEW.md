# V2 fit-only scoring result: independent audit

**Verdict:** The saved V2 fit comparison is internally consistent. Its 48/48 positive scores are restricted expected-word versus paired-foil comparisons, not complete-answer accuracy. This review used saved data only; no Qwen inference or development/test input was opened.

The saved result's runner, V2 policy, original V1 score, frozen spec, rendered manifest, progress, and memory SHA-256 bindings all match files on disk. All 48 report rows exactly equal the 48 progress rows and match the preselected 12 pairs × two voices × two directions. The 49 memory samples include post-load plus one per case; peak MPS driver allocation was **18.593 GiB**, with no recorded limit breach.

For every row, V1 first-step fields exactly match the prior V1 result; the maximum V1 log-odds rerun difference is **0**. V2 first-token ID sets and complete answer-plus-EOS sequences match `SCORING_V2.json`. I recomputed each serialized prefix mass from its log mass, each prefix expected-minus-foil margin, each complete-class `logsumexp` of the four sequence log probabilities, and each complete-class margin. No discrepancy exceeded floating-point serialization tolerance. Individual token log probabilities and the full-vocabulary denominator cannot be independently replayed without saved logits, so this checks aggregation and provenance rather than a second model evaluation.

| Fit-only measure | Expected exceeds paired foil | Median expected-class mass |
| --- | ---: | ---: |
| V1 single-token first-step score | 48/48 | 0.26381 |
| V2 expanded-prefix first-step score | 48/48 | 0.95559 |
| V2 complete canonical-sequence-plus-EOS score | 48/48 | 0.92348 |

The V2 prefix and complete canonical sequences cover all **44/44 naturally generated correct** map-fit answers. The remaining four natural complete answers are wrong despite positive margins on all three paired scores:

| Case | Natural answer | V2 expected complete-sequence mass | V2 complete expected-minus-foil log odds |
| --- | --- | ---: | ---: |
| `animal_06/fit_chosen/kokoro/deer` | `Dear` | 0.2348 | 10.490 |
| `animal_06/fit_chosen/piper/deer` | `Dear` | 0.0588 | 11.463 |
| `color_06/fit_chosen/piper/cyan` | `sand` | 0.000191 | 9.258 |
| `nature_10/fit_chosen/piper/grove` | `growth` | 0.1454 | 12.729 |

The foil is only the paired corpus word, not every possible output. In these four cases, the expected class can beat that foil while another answer is generated. The V2 complete score also depends on later decoder positions, so it cannot be substituted into the prompt-last residual transport gate. V2 is a disclosed, fit-informed secondary score; this result provides no held-out lens or security finding.
