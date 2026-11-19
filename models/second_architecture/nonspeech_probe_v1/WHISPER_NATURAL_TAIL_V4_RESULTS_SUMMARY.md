# Whisper natural-tail V4: descriptive saved-result summary

19 November 2026. **Read-only summary of two saved audit receipts; independent result review is separate.** The [multilingual receipt](runs/natural_tail_waveform_v1/whisper_multilingual_attempt_v4_audit.json) (SHA-256 `09d7a1aa8ecdd798de71df5d3218826d3c903c091be9cea1a9cab4826d6ffd2a`) and [English-only receipt](runs/natural_tail_waveform_v1/whisper_english_attempt_v4_audit.json) (SHA-256 `5fa4fc150c693be4c56b550a8b29188372a79a3d4912ea44bf9b51b45e98c900`) each report `PASS`, 160 calls, 40 zero-tail/base comparisons, 80 noise/zero-tail comparisons, and no base-replay failures. The 18 non-silence cases contribute 36 zero and 72 noise comparisons per checkpoint; two silence cases contribute four zero and eight noise comparisons. I read the saved `records.jsonl` and referenced first-step `.npy` sidecars to compare full returned token sequences, EOS status, prefixes, and FP32 raw and processed vectors exactly. No model inference was run for this summary.

| Checkpoint | Zero tail versus base: changed complete answers | Noise versus matching zero tail: changed saved token paths | Noise pairs with both answers ending in EOS: changed complete answers |
| --- | ---: | ---: | ---: |
| Multilingual, all 20 cases | 0/40 | 12/80 | 9/77 |
| Multilingual, 18 non-silence cases | 0/36 | 10/72 | 7/69 |
| English-only, all 20 cases | 0/40 | 4/80 | 4/80 |
| English-only, 18 non-silence cases | 0/36 | 4/72 | 4/72 |

Three multilingual noise outputs at the 32,000-sample tail hit `max_new_tokens=128` without EOS: both seeds of `d08_beep_train_high` and one seed of `d08_dither`. All three saved paths differ from their zero-tail counterparts, but they are **length censored**, so they are excluded from the complete-answer denominator. The other 157 multilingual calls and all 160 English-only calls ended in EOS. For each checkpoint, all 20 exact base replays matched the base sequence and both first-step vectors **bit for bit**; the auditor's recorded replay-failure list is empty.

**Where the measured differences appear.** For every one of the 40 zero-tail/base comparisons per model, the 30-second padded PCM is identical, the full mel sidecar and both first-step vectors are exactly identical, and the returned token sequence is identical. The attention mask differs in all 40 because the explicit clip duration changed. For every one of the 80 noise/zero comparisons per model, the masks match, the mel differs in the appended-tail and boundary regions, and both full first-step raw and processed vectors differ exactly. The mel's source interior and nominal padding interior have zero changed frames in all 80 comparisons. First-step raw argmax changes in 28/80 multilingual and 23/80 English pairs; processed argmax changes in 10/80 and 4/80. The maximum absolute **finite** first-step logit difference is 12.3476 multilingual and 4.4848 English; the processed vectors retain the same infinity masks. Multilingual language-prefix IDs differ in three 32,000-sample noise pairs; English-only prefixes never differ. These observations place the first measured noise effect at the tail mel features and show a downstream logit effect, but do not locate a causal circuit inside the encoder or decoder.

| Source case and added tail | Multilingual changed paths / two seeds | English-only changed paths / two seeds |
| --- | ---: | ---: |
| `d02_beep_train_high`, 8,000 samples | 2/2 | 0/2 |
| `d02_beep_train_low`, 8,000 samples | 0/2 | 2/2 |
| `d02_beep_train_low`, 32,000 samples | 2/2 | 0/2 |
| `d08_band_noise_low`, 8,000 samples | 0/2 | 2/2 |
| `d08_band_noise_low`, 32,000 samples | 1/2 | 0/2 |
| `d08_beep_train_low`, 32,000 samples | 2/2 | 0/2 |
| `d08_beep_train_high`, 32,000 samples | 2/2, both censored | 0/2 |
| `d08_dither`, 32,000 samples | 1/2, censored | 0/2 |
| `d08_silence`, 32,000 samples | 2/2 | 0/2 |

All omitted case/tail combinations have 0/2 changed saved paths. `d02` and `d08` source clips are two and eight seconds; 8,000 and 32,000 samples are 0.5- and 2-second tails at 16 kHz. Seed pairs share the same source case and are not independent examples. These inputs are engineered **non-speech** sounds. The outputs include transcription hallucinations and the three censored continuations. The results show sensitivity of two specific Whisper checkpoints to these tails under the frozen generation settings. They do not establish a security vulnerability, behavior on human speech, transfer to other audio architectures, or the internal causal mechanism. The next mechanism test would require a separately specified activation measurement or intervention with the same frozen waveform controls.
