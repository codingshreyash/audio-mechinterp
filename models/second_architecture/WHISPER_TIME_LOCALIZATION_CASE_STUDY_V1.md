# Where a spoken word is represented and used in Whisper Tiny

18 November 2026. This case study synthesizes three independently audited, CPU-only experiments on `openai/whisper-tiny.en`. It asks a mechanistic question: when two recordings differ only in one spoken word, where does that difference first appear in the model's audio pathway, and which internal states can change the **complete transcription** when replaced? It is an ASR study, not a security finding.

## Controlled inputs and readout

The `silver`/`teal` and `plate`/`cup` pairs each have the same surrounding waveform within a synthetic voice. Only the declared word slot differs in PCM. Each pair was rendered by Piper and Kokoro, and both directions were tested. All eight stock Whisper transcriptions contained exactly one intended word, excluded its alternative, and ended normally. The first pair was already open FIT material; the second was chosen as a follow-up after the first result. Neither is a locked test or natural human speech.

The experiments saved complete token sequences and decoded text, not just a selected token score. All 232 saved outputs had exactly one final EOS, as checked in the independent review; the later two runners enforced that condition directly, while the first runner checked only that an EOS appeared. The primary word-change outcome also required substitution of the donor word exactly once, removal of the recipient word, and every other normalized word unchanged. State patches replaced donor rows in a fixed word window at the post-convolution/position input, an encoder block output, or the final normalized encoder state. Self patches, off-word patches, full-donor final-state patches, and equal-norm random edits checked whether the effect was specific to the donor word and intervention location.

## What the evidence shows

| Experiment | Word-window donor patch | Off-word donor patch | Other controls |
|---|---:|---:|---|
| [`silver`/`teal` FIT assay](WHISPER_FIT_WORD_WINDOW_CAUSAL_RESULT_INDEPENDENT_REVIEW_V1.md) | 12/12 target-only complete-transcript changes across three sites | Prefix 0/12 | Self 0/12; full donor 4/4 exact donor tokens. Random edits gave 0/8 donor changes, but only four distinct noise tensors were used across reverse directions and one edit shortened a transcript. |
| [`plate`/`cup` FIT replication](WHISPER_OBJECT02_FIT_CAUSAL_RESULT_INDEPENDENT_REVIEW_V1.md) | 12/12 target-only changes across three sites | Prefix 0/12 | Self 0/12; full donor 4/4 exact donor tokens; eight distinct random edits gave 0/8 donor changes. |
| [Spread-versus-use follow-up](WHISPER_FIT_SPREAD_CAUSAL_RESULT_INDEPENDENT_REVIEW_V1.md) on both pairs | 16/16 target-only changes at post-block 0 and final norm | Active-audio complement 0/16; equal-width prefix 0/16; equal-width postword 0/16 | Self 16/16 exact recipient tokens; full donor 8/8 exact donor tokens; 32 distinct norm-matched complement edits preserved recipient normalized words 32/32. |

The input difference is local at the measured early boundaries: in the spread assay, donor and recipient post-convolution/position states were bitwise equal outside the frozen word window. Their log-mel differences first appeared at frames 195 (Piper) and 233 (Kokoro), and their post-convolution differences at rows 97 and 116, within that window. After encoder block 0, donor-minus-recipient differences were nonzero even at row 0. The block contains bidirectional self-attention and a feed-forward network; the capture identifies the first **observed** global difference, not which operation inside the block caused it.

The intervention result is more selective than the activation difference. At the two tested later sites, replacing the word-window rows changed the word in the complete transcript; replacing every active row **outside** that window did not. A wide numerical difference therefore did not, by itself, establish that those outside rows were sufficient to control the decoded word in these contexts.

## What this does not establish

- These are internal state edits made with model access. There is no attacker-controlled waveform that produces the same edit, no assistant tool action, and no violated security property.
- The same Tiny English ASR checkpoint, two already-inspected FIT word pairs, and two synthetic voice engines underlie all three experiments. The follow-ups were designed after seeing earlier results. Counts across arms are correlated and are not a population success rate or a statistical significance claim.
- A failed outside-window patch does not prove that those states contain no word information. The decoder may use interactions between positions, a different patch magnitude, or another trajectory. The causal claim is limited to the fixed patch contexts and complete transcriptions tested here.
- The first `silver`/`teal` random control deviated from its independent-draw specification; the later `plate`/`cup` and spread assays corrected that issue. The original result remains unchanged in the record.

## Research use

This supplies a practical audio mechanistic-interpretability check: track the waveform contrast through preprocessing and internal stages, then test whether candidate states alter a complete output under location, self, full-donor, and random controls. A transcript-level Jacobian or colorful activation map alone cannot answer the causal question. The next cross-architecture step must separately measure preprocessing effects: a model that normalizes each utterance globally can create off-word differences **before** any transformer block. A speech assistant requires the same discipline across its audio encoder, language state, generated speech/text, and native action path before making a security claim.

The [research ledger](../../STATUS.md) and linked independent reviews bind the frozen plans, code, no-weight preflights, raw JSONL, tensor captures, and audits.
