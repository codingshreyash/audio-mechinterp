# Independent result review: engineered non-speech, multilingual Whisper Tiny

19 November 2026. **GO for a descriptive saved-evidence result; no causal or security finding.** I reviewed the fresh V2 attempt `runs/descriptive_whisper_multilingual_20261119_v2` independently of the runner and its own auditor. I did not load candidate model weights or run another model forward. My read-only recomputation is `independent_v2_result_recompute_20261119.py`; it returns `GO_SAVED_EVIDENCE` against `records.jsonl` SHA-256 `588165986eadcfc7df04f6a8869de78ccd656f5b9be578fe41b3e1395abeec42`.

## Admission and completeness

- The separate V2 parent exited with `worker_exited_pending_audit`, exit code 0, no resource-limit reason, and `host_after_ok=true`. Its post-host log covers launch through **worker exit** at `2026-11-19T08:19:02.288558+00:00`; collection followed 2.354 seconds later. The saved pre/post logs, AC, open lid, ten-minute prior history, absence of sleep/wake/thermal transitions, source receipt, and GO hashes passed independent checks. The parent sampled a 936,902,656-byte peak RSS under its 4-GiB limit. Its saved structural audit passed.
- All 20 rendered WAV and PCM SHA-256 hashes match the pinned manifest `a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90`; files are 16-kHz, mono, PCM16 with the planned 2- or 8-second frame count. All 20 case rows, two repeated silence rows, 18 same-duration-silence difference rows, start, and completion are present in the exact 42-record inventory. The two silence repeats exactly match their original complete outputs and saved trace summaries.
- The earlier V1 attempt remains `invalid_or_partial`: its parent result SHA-256 `806bfd0c84899e88cec47b24b0966274561cdd155908c52369d0699496c2404a` and records SHA-256 `984625850ac33cd4c5d039304162a9b4396bd2a2ad3d31513a231eb452e708a3` are unchanged. Its rows are **excluded** from these counts.

## Complete output census

The pinned tokenizer independently decodes every saved full sequence and generated segment as shown below. Each row's prefix is `[50258, 50259, 50359, 50363]` (`start`, detected English, transcribe, no timestamps). The English token came from a separate recorded language-detection forward and was not counted as generated speech. Every row then generated exactly one lexical token and EOS `50257`; all 20 are complete, none length-censored. All emitted letters are Latin. `raw top` is the independently recomputed argmax of the saved **pre-processing first decoder-step logit vector**; `generated` is the actual first text token.

| Engineered input | Raw top | Generated | Complete lexical transcript |
| --- | ---: | ---: | --- |
| `d02_band_noise_high` | 522 | 291 | ` you` |
| `d02_band_noise_low` | 542 | 291 | ` you` |
| `d02_beep_train_high` | 522 | 10965 | ` Fuck` |
| `d02_beep_train_low` | 522 | 509 | ` You` |
| `d02_click_train_high` | 542 | 291 | ` you` |
| `d02_click_train_low` | 542 | 291 | ` you` |
| `d02_dither` | 291 | 291 | ` you` |
| `d02_silence` | 291 | 291 | ` you` |
| `d02_steady_tone_high` | 522 | 291 | ` you` |
| `d02_steady_tone_low` | 542 | 291 | ` you` |
| `d08_band_noise_high` | 542 | 291 | ` you` |
| `d08_band_noise_low` | 542 | 291 | ` you` |
| `d08_beep_train_high` | 522 | 509 | ` You` |
| `d08_beep_train_low` | 522 | 509 | ` You` |
| `d08_click_train_high` | 522 | 291 | ` you` |
| `d08_click_train_low` | 542 | 291 | ` you` |
| `d08_dither` | 291 | 291 | ` you` |
| `d08_silence` | 291 | 291 | ` you` |
| `d08_steady_tone_high` | 542 | 291 | ` you` |
| `d08_steady_tone_low` | 542 | 291 | ` you` |

The two silence baselines themselves generate ` you`. Against the matched-duration silence token sequence, **only four beep rows differ**: one emits ` Fuck` (`d02_beep_train_high`), while three emit capitalized ` You`. The other 14 nonsilence rows match their duration's silence output. These are generated text observations from engineered nonspeech, with existing Whisper hallucination prior art; the table does not establish that these sounds contained speech or that a person would hear those words.

## Decoder processing and trace limits

The raw first-step argmax was token 542 (` [`) in 9/20 cases, token 522 (` (`) in 7/20, and token 291 (` you`) in 4/20. The checkpoint's pinned `generation_config.json` includes both 542 and 522 in `suppress_tokens`; neither is in `begin_suppress_tokens`. Thus 16/20 raw top IDs are on a configured suppression list, whereas the recorded processed top ID equals the generated first token in 20/20. This establishes a visible distinction between raw decoder preferences and the final generation choice. It does **not** isolate one generation processor as the sole cause of any transcript, and it does not locate the origin of the beep-specific choices in the audio encoder.

I recomputed the SHA-256 of every saved first-step raw logit vector and first-step final decoder-layer state, their shapes, raw argmax and selected raw logit; checked the actual 200/3,000 and 800/3,000 mel active/padded masks and nominal encoder row map for 2- and 8-second cases; and checked one capture at each declared detect/generate stage per row. The remaining encoder traces and later decoder logits are **summaries and hashes without raw tensors**. Their actual values, the reported first-step processed logit hashes/log probabilities, and the 18 same-duration-silence difference curves cannot be independently recomputed from the saved evidence. The difference rows pass shape and numerical self-consistency checks only. A future causal patch needs a separately frozen intervention plan and raw states at its selected interface.

**Interpretation:** This small, synthetic, single-checkpoint descriptive run shows a language-detection decision, a one-token text hallucination even for silence, and a decoder-processing distinction. It does not show a multilingual script outcome, a verified audio-to-internal-state causal pathway, a deployed assistant action, or a security violation.
