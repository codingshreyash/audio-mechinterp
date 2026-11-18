# Independent saved-result review: Voxtral preprocessing decomposition

Reviewed 2026-11-18 against the fit-only protocol frozen in commit `087b0d4`. Source: `runs/voxtral_preprocessing_decomposition_fit_v1/piper_red_blue_20261118T064322754029Z.json` (SHA-256 `7cba026aba0f81ca1edc97ed138b6fb9b9e0e85168c08e38f6bba8d3fdaba9c8`). I used the saved result and a separate CPU-only preflight; I did not run MPS or development/test inputs.

## Result

All six conditions produced a complete one-word color answer followed by EOS. The crossed feature hybrids followed the **word-window source**; swapping only the nominal padded frames kept the native answer. The full candidate-plus-EOS log-probability contrast had the same color sign in each condition.

| Condition | Complete generated IDs | Answer | `LL(Blue+EOS) − LL(Red+EOS)` |
| --- | --- | --- | ---: |
| Native red | `[12846, 2]` | Red | −5.7111 |
| Native blue | `[24851, 2]` | Blue | +6.8136 |
| Red word, blue outside | `[12846, 2]` | Red | −5.4183 |
| Blue word, red outside | `[24851, 2]` | Blue | +6.8979 |
| Red with blue padding | `[12846, 2]` | Red | −5.1022 |
| Blue with red padding | `[24851, 2]` | Blue | +6.8524 |

## Independent checks

- The result's script, smoke helper, requirements freeze, manifest, and both WAV hashes match the current frozen files. It records model revision `3060fe34b35ba5d44202ce9ff3c097642914f8f3` and the three expected installed Transformers source digests. The pair has equal 16 kHz mono PCM outside samples `[18082,26627)`, as checked by the CPU preflight.
- All six saved float32 feature SHA-256 values exactly match an independent CPU-only preflight from the same frozen WAVs and processor. The native red and blue feature tensors are the processor endpoints. Independent raw mel reconstruction has 0 maximum absolute error for both, and raw/shared-maximum off-word frame L2 maxima are 0. The distinct clip maxima are 1.3852976561 and 1.6545178890.
- Each teacher-forced candidate has one word token plus EOS ID 2. The saved per-token log probabilities sum exactly to each complete-reply score, and the saved blue-minus-red contrast equals the difference of those two scores in all six rows.
- The model recorded one `[375,3072]` projector tensor per condition. Every per-reference profile has 375 L2 values partitioned into 9 nominal word rows, 30 other real-audio rows, and 336 nominal padding rows. I independently recomputed all profile means, maxima, nonzero counts, squared-L2 totals, and shares from the row arrays; maximum arithmetic difference was `2.22e-16`. Both native self-reference profiles are exactly zero.
- The sampled peak MPS driver allocation was **9.932 GiB**, below the 30 GiB bound; six generations plus two candidate-score forwards per condition are represented in the result. The output is marked `complete`, fit-only.

## Claim boundary

This is a **feature-space intervention** on one Piper red/blue pair. It shows that, for these six tensors, the word-window source dominates the complete color answer, while changing the normalized outside or padding features changes score margins and post-projector activations without changing that answer. It does not isolate a natural waveform intervention or identify a hidden model circuit. The projector profiles can differ in nominal padding rows because the audio encoder is bidirectional; those rows are indexing groups, not independent causal locations. The crossed and padding-swapped tensors may be off-manifold.
