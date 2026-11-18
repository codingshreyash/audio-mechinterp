# Independent review: Voxtral Kokoro Sarah fit-voice decomposition

**Reviewed 18 November 2026.** The single saved Kokoro result [JSON](runs/voxtral_kokoro_preprocessing_fit_v1/kokoro_sarah_red_blue_20261118T070011156066Z.json) is complete. Its SHA-256 is `a7a3d12705f2fa5bf9920228db8c8c76abe23a3ce0f043a64c5f151b231ae945`. The [voice-specific protocol](VOXTRAL_KOKORO_PREPROCESSING_PROTOCOL.md) was committed as `294c325` before this MPS run. I audited the result against a fresh **CPU-only** execution of `prepare_kokoro()` and recomputed every saved projector-profile partition summary. I did not run model inference.

## Provenance and numerical checks

| Check | Result |
| --- | --- |
| Frozen inputs | Result's manifest SHA-256 `318807c92971babacad93b69b7d8478e29f397219da0cb9ed6e3134649aea98b`, TTS pins, both WAV hashes, PCM equality outside `[35920,61120)`, and the six input-feature SHA-256 values match the protocol and fresh CPU preparation. Both prompts have the same 390 token IDs and 375 audio placeholders. |
| Code and model | Kokoro runner SHA-256 `acaf7a81b0d1dddd4329f258609dd1945286720c608dd913abd2e4bddddf1856`, committed Piper helper SHA-256 `026445da1ac2c4c2f4b8f45ab2f8d7a1f4679924774714951b16d5cd29de0788`, pinned Voxtral revision `3060fe34b35ba5d44202ce9ff3c097642914f8f3`, installed Transformers source hashes, dependency versions, and local snapshot metadata all match CPU preparation. |
| Preprocessing | Independent Whisper reconstruction has maximum absolute error **0.0** for both clips. Raw log-mel maxima are red **1.4036763906**, blue **1.5550248623**. Raw off-word and shared-max-normalized off-word differences are **0.0**. Shared-max difference is confined to the conservative mel word window, frames **224–383**; nominal feature padding starts at frame **637**. |
| Memory and bounds | Six declared conditions, each with one greedy generation capped at 32 tokens and two teacher-forced candidate passes. Observed MPS driver high-water mark **9.93 GiB**, below the 30 GiB per-process cap; saved cap fraction is consistent with the 37.44 GiB recommended maximum. No cloud GPU or dev/test audio appears in this result. |
| Scores and profiles | All six generated suffixes end in EOS ID `2`; complete answers and raw token IDs agree. Each saved `Blue − Red` score equals the difference between the two stored candidate-plus-EOS log likelihoods; each candidate likelihood equals the sum of its stored token log probabilities. Every condition has a `[375,3072]` projector capture and two finite, nonnegative 375-row L2 profiles. Recomputed word, other-real-audio, and padding summaries match the saved values; native self-distance profiles are zero. |

## Complete-answer result

| Feature condition | Full answer (generated IDs) | `LL(Blue+EOS) − LL(Red+EOS)` |
| --- | --- | ---: |
| Word red / outside red, native | `Red` (`[12846,2]`) | −6.893 |
| Word blue / outside blue, native | `Blue` (`[24851,2]`) | +8.778 |
| Word red / outside blue | `Red` (`[12846,2]`) | −6.532 |
| Word blue / outside red | `Blue` (`[24851,2]`) | +8.961 |
| Red with blue padding | `Red` (`[12846,2]`) | −7.004 |
| Blue with red padding | `Blue` (`[24851,2]`) | +8.888 |

The complete answer followed the word-window source in **6/6** conditions. The teacher-forced contrast agreed in sign for all six. Swapping outside frames or only padding moved the score but did not change the generated answer. The Kokoro native red/blue projector contrast had nonzero L2 in all **22** nominal word rows, all **58** other real-audio rows, and all **295** nominal padded rows. Their mean L2 values were **6.819**, **2.311**, and **0.670**, respectively. This broad activation difference is compatible with bidirectional encoder mixing and preprocessor normalization; it does not localize a discrete word circuit.

## Comparison and limits

The [committed Piper result](runs/voxtral_preprocessing_decomposition_fit_v1/piper_red_blue_20261118T064322754029Z.json) also produced the word-source answer in all six conditions, with matching score signs. Thus the *fit-only feature-splice behavior* replicated across the Piper and Kokoro Sarah renderings of this one English red/blue question. Kokoro's word interval is 1.575 seconds versus Piper's 0.534 seconds, and their nominal word windows are 22 versus 9 projector rows. Absolute projector magnitudes and score margins should not be treated as a voice-only effect or compared as calibrated quantities.

These are feature-level interventions, not natural waveforms. The model's complete answer is stable to the tested outside and padding swaps, but the off-manifold splice boundary and one lexical pair limit interpretation. The result does **not** establish a security vulnerability, a general refusal or authorization mechanism, or transfer to a different model architecture, multilingual speech, human voices, or held-out wording.
