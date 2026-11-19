# Engineered non-speech through three audio paths

19 November 2026. This is a bounded mechanistic case study on pinned small checkpoints. The [source protocol](NON_SPEECH_MECHANISM_PROSPECTIVE_V1.md), [independent descriptive audits](INDEPENDENT_MULTILINGUAL_DESCRIPTIVE_V2_RESULT_REVIEW_20261119.md), [comparator audit](INDEPENDENT_COMPARATOR_V2_RESULT_REVIEW_20261119.md), and [independent causal audit](INDEPENDENT_CAUSAL_ATTEMPT_002_RESULT_REVIEW_20261119.md) carry the exact inputs, hashes, controls, and failure ledgers. The first multilingual descriptive attempt, first English/Wav2Vec2 comparator attempts, and first causal attempt are preserved invalid and excluded.

## What the same 20 sounds produced

| Pinned path | Silence output | Planned 20-sound result | Difference from matched silence |
| --- | --- | --- | --- |
| Multilingual Whisper Tiny | ` you`, then EOS | One English lexical token and EOS on 20/20 | Four beep clips changed the complete token sequence; three changed capitalization and one changed the word to ` Fuck`. All 20 selected the English language prefix. |
| English Whisper Tiny | ` you`, then EOS | ` you` and EOS on 20/20 | 0/18 non-silence clips. |
| Wav2Vec2-base CTC | Empty | Nonempty on 3/20, all beep clips | 3/18 non-silence clips; the strings were repeated or nonsensical letter patterns. |

Whisper's silence response means the repeated `you` is a decoder/output baseline in this setup. It cannot be attributed to the beeps. In multilingual Whisper, 16/20 raw first-step argmax tokens were on the pinned generation suppression list; the chosen text token is a *processed* generation decision. The Wav2Vec2 audit independently decoded the saved complete CTC paths, but the run did not save full logits for an independent argmax recomputation. All of these are descriptive observations on engineered sounds, not speech intelligibility or security tests.

## Phase-separated internal intervention

The prospectively amended study ran **328/328** planned calls on all 18 non-silence sounds against same-duration silence donors. The independent checker rehashed 812 binary sidecars, reconstructed 80 native states and the saved edits, verified exact no-hook/read-only/self/full-donor and raw-logit parity, and recorded five length-censored arms. A prior attempt stopped before model load and is not pooled with this run.

Only four beep clips had a recipient output different from their silence donor. At the final encoder state, substituting the silence donor during the **transcription pass** gave the donor's complete output on 4/4 of these informative cases. Substituting it during the separate **language-detection pass** changed 0/4 complete outputs, even though all 99 recorded language scores became the donor's scores; English remained selected. Detection-only therefore had an internal effect without crossing the output decision boundary. One of eight informative equal-norm random Stage A edits also produced the donor output, so the donor effect is not a unique direction or uniquely specific edit.

An early state transplant over the whole nominal sound-bearing interval also produced the donor output on 4/4 informative beep cases. That is a broad intervention, not localization to an individual beep or a neuron. The displaced-window control had one changed, length-censored output and no verified donor transfer among the four informative cases.

## Why the nominal padding matters

The pinned Whisper feature extractor floors log-mel values using the **maximum over the whole padded spectrogram**. Beep and silence inputs therefore differ in every nominally padded mel frame in the checked cases, despite having zeros in the padded waveform tail. The pinned encoder accepts an audio attention mask but does not use it; the decoder calls cross-attention with no encoder mask. Saved pre-block-0 states differ throughout the 1,500-row encoder window. A padded row is not automatically an information-free control here. The [prospective padding-channel follow-up](WHISPER_PADDING_CHANNEL_PROSPECTIVE_V1.md) tests whether those padded values *causally* affect complete outputs; this case study does not infer that from activation differences alone.

## Claim boundary and next check

This establishes sufficiency of a **white-box donor encoder state** for the four complete-output differences in this pinned Whisper Tiny setup. Three differences are capitalization, one is lexical; five other arms are length-censored. It does not establish a narrow acoustic circuit, a model-family generalization, a waveform an attacker can supply to recreate the patched state, or a confidentiality, integrity, or availability violation. [Prior work](NONSPEECH_MECHANISM_PRIOR_ART_20261119.md) already documents non-speech transcription and internal interventions in audio models.

The next decisive test is the frozen active-versus-padded factorial with exact full-donor and random controls, followed by independent audio and checkpoint transfer. A security study then needs an unmodified waveform to change a consequential complete answer or action under an explicit system boundary.
