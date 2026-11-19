# A local word state controls transcription despite widespread audio-state differences

19 November 2026. This case study combines two independently reviewed, prospectively frozen **FIT-only** internal-state experiments on `facebook/wav2vec2-base-960h`. The paired recordings differed only in one int16 PCM word slot. The question was whether the model's widespread hidden-state differences outside that slot could, by themselves, switch the **complete CTC transcript**. This is a model-mechanism result, not a security finding.

## The four-state comparison

At each of two hook sites—feature projection, before the transformer, and after encoder block 11—we divided the saved hidden state into a nominal word window **W** and every other unpadded audio row **O**. For each of 16 directed comparisons (two word pairs × two synthetic voices × two directions × stock/shared preprocessing modes), we held the recipient waveform fixed and transplanted the paired recording's states:

| Word rows | Outside rows | Complete transcript at feature projection | Complete transcript after block 11 |
| --- | --- | ---: | ---: |
| Recipient | Recipient | Recipient **16/16** | Recipient **16/16** |
| Donor | Recipient | Donor **16/16** | Donor **16/16** |
| Recipient | Donor | Recipient **16/16** | Recipient **16/16** |
| Donor | Donor | Donor **16/16** | Donor **16/16** |

The donor-word arms replayed the earlier saved word-patch logits exactly in **32/32** site/direction checks. Self and full-donor arms matched clean recipient/donor logits exactly in **32/32** each. Two independently seeded, equal-*outside*-norm random edits in O preserved the recipient transcript in **64/64** arms total. The [independent complement result review](WAV2VEC2_FIT_COMPLEMENT_RESULT_INDEPENDENT_REVIEW_V1.md) rehashed all 384 new tensor sidecars, reconstructed all 192 patches bitwise, and manually decoded every complete CTC output. The earlier [word-patch result review](WAV2VEC2_FIT_CAUSAL_RESULT_INDEPENDENT_REVIEW_V1.md) checked another 224 arms, including immediately preceding/following rows and random word-window edits.

**The denominator matters.** Both clean endpoints had the intended spoken word in only Piper `plate`↔`cup` under the two correlated preprocessing modes: **4/16 directed comparisons per site**. On that strict subset, donor W yielded the intended target-only word change in **4/4**, whereas donor O yielded **0/4**. The other 12 comparisons per site are descriptive, because the model transcribed a paired word as `TEEL`, `TIL` or `CUB`. Across all 16, the outcome above is transfer of the **model's donor transcript**, including those errors; it is not 16/16 intended-word accuracy.

## Why the outside rows were worth testing

With stock per-utterance waveform normalization, the paired model inputs differ outside the raw PCM word slot before entering Wav2Vec2. With shared outside-word input statistics, the first raw convolution stays local, but its time-wide GroupNorm spreads the numerical difference across frames before transformer attention. An [independent GroupNorm reconstruction](WAV2VEC2_FIT_GROUPNORM_RECONSTRUCTION_INDEPENDENT_REVIEW_V1.md) reproduced the saved postnorm tensors from the pinned weights and raw convolution; all four pair/voice cells had exact zero outside-word difference before that norm and a 2.15–4.80 maximum after it. Thus O genuinely contains differing values at the tested downstream sites. The four-state experiment tests what those values can do **when transplanted**, not whether they originated at GroupNorm or are read in a natural forward.

In these exact patch contexts, donor W was sufficient for the donor transcript without donor O, while donor O alone was not sufficient with recipient W. That is a sharper conclusion than “the activation map lights up outside the word.” It does **not** establish that W is the unique necessary representation, that O contains no word information, or that GroupNorm spread is irrelevant. The natural donor-minus-recipient O edit norms were **13.33–28.66**, much smaller than W's **40.95–221.68**; position, magnitude and interactions are confounded. The earlier word-window random controls also changed an unrelated spelling in **11/64** arms, so arbitrary hidden-state edits are not generally transcript stable.

## Scope and next test

These were two previously inspected lexical pairs in one English CTC ASR checkpoint, rendered by Piper and Kokoro. The two modes reuse the same WAVs, and both hook sites reuse the same directions; they are correlated, not independent replications. There was no human voice, multilingual input, assistant response, speech output, tool action, attacker-controlled perturbation or security-property violation.

The next strong check is word-disjoint speech with an independently specified voice and quality rule. A speech-assistant security case would then need to trace its audio encoder through its language, speech/text and native action outputs and demonstrate an attacker-reachable change in a consequential decision under a specified harness. The [offline explorer](explorer/wav2vec2_fit_v1.html) lets a reader hear the paired FIT WAVs and inspect the first 224-arm word-patch experiment and its time traces; the complement run is documented in the raw records and independent review linked above.
