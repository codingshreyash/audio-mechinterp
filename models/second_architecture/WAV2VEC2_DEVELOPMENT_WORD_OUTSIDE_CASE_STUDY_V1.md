# A spoken-word state swap across 12 word pairs: exploratory Wav2Vec2 development result

19 November 2026. This case study interprets the [independently reviewed saved run](WAV2VEC2_DEV_WORD_DISJOINT_RESULT_INDEPENDENT_REVIEW_V1.md). It adds no model inference or new effect estimate. The corpus-v1 development set was already inspected in earlier work, and its Piper and Kokoro voices are the same engines as FIT. This is an exploratory cross-word check, not a fresh held-out or independent-voice result.

## The causal question

For a paired spoken sentence that differs only in its middle word, hold the **recipient waveform** fixed and replace selected rows of its internal Wav2Vec2 state with rows from the paired donor. The nominal word window W is computed from the known PCM edit slot and the checkpoint's convolutional receptive field, before seeing model outcomes. O is every other active row. The swap is at the output of encoder block 11, before this checkpoint's CTC head. At this late site, the CTC head reads frame states to produce a transcript; a positive result does not identify earlier computations that created those states.

The [frozen V2 plan](WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md), [runner source review](WAV2VEC2_DEV_WORD_DISJOINT_SOURCE_INDEPENDENT_REVIEW_V1.md), [preflight/freeze](runs/wav2vec2_dev_word_disjoint_v1/FREEZE.json) and [saved-data audit](runs/wav2vec2_dev_word_disjoint_v1/20261119T0103Z/audit.json) bind 12 development word pairs, one `dev_listen` sentence template, two synthetic engines, both directions, and two correlated input-normalization modes S/C. That yields 96 directed mode cells and six arms per cell: self, donor W, donor O, full donor, and independent norm-matched random W/O edits. The auditor reconstructed every patch from saved clean states, re-decoded full CTC outputs, and verified 864 sidecar hashes. All 192 self/full controls matched their clean complete logits exactly.

## What changed, and what did not

| Internal patch | Complete output across 96 planned directed mode cells | Strict intended-word result |
| --- | --- | --- |
| Donor word rows W | Donor model transcript **96/96** | Intended-word-only transcript **56/56 eligible** |
| Donor outside rows O | Recipient model transcript **96/96** | Intended-word-only donor transcript **0/56 eligible** |
| Full donor state | Donor model transcript **96/96** | Intended-word-only transcript **56/56 eligible** |
| Random W, matched to natural W edit norm | Recipient transcript 83/96; other transcript 13/96; donor 0/96 | Donor-word transfer 0/56 eligible |
| Random O, matched to natural O edit norm | Recipient transcript **96/96** | Donor-word transfer 0/56 eligible |

The strict denominator matters. Only **74/96 clean model/mode outputs** had the exact intended full template, representing **37/48 distinct WAVs** and **14/24 pair/voice cells** with both words correctly transcribed. The resulting 56 eligible directed mode cells repeat the same recordings under correlated S/C processing. The 96/96 donor-model figure includes clean transcription errors such as `gray` becoming `GREY`; it is **not** 96/96 correct intended-word transfer. The W/O pattern appears in all 12 pairs, both voices and both modes, but these are not independent samples of each dimension.

The positive W result and negative O result show **relative sufficiency under these exact internal transplants at this late site**. They do not show that O contains no word information, that W is a unique circuit, or that the same transfer survives a new voice or earlier encoder layer. Natural W edits were larger (median L2 49.47) than O edits (median 18.45), and region widths differ. Random W changed 13 complete transcripts to other spellings, so generic state disturbance is a real control outcome even though it never produced the donor transcript here.

## Why this advances the audio question

The earlier [FIT factorial](WAV2VEC2_FIT_WORD_OUTSIDE_FACTORIAL_CASE_STUDY_V1.md) used only two word pairs and had four strict intended-word directions per site. This wider word-disjoint exploratory screen repeats the *complete-output* W-positive/O-negative pattern across 12 development pairs, while retaining the ASR misses and random-control collateral. It demonstrates a reproducible way to distinguish numerical spread of an acoustic edit from the rows sufficient to change a downstream transcript. The next claim-bearing check is the [new 32-WAV Piper/Samantha corpus](WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md): blinded human quality review first, then both early and late Wav2Vec2 sites plus Whisper Tiny. An earlier-site effect or failure will be more informative about how the model builds the word representation than this near-head swap alone.

Nothing here is an attacker-reachable audio perturbation or speech-assistant action. A security finding would need an ordinary input that changes an unmodified model's consequential decision, with the complete audio-to-action path and harness traced separately.
