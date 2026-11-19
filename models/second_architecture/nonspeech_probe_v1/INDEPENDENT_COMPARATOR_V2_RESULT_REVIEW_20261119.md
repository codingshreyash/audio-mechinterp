# Independent result review: English Whisper and Wav2Vec2 non-speech controls

19 November 2026. **GO for the saved descriptive evidence; no causal or security finding.** I independently recomputed the two completed V2 controls with [this read-only checker](independent_comparator_v2_result_recompute_20261119.py). It loads revision-pinned tokenizer/processor metadata and the saved files, never candidate weights or another model forward. The source runners, structural auditors, and failed V1 attempts were not modified.

## Evidence integrity

| Attempt | Parent result SHA-256 | Records SHA-256 | Saved structural audit |
| --- | --- | --- | --- |
| English Whisper V2 | `89dc84e4d8a3fdb8733c0c8f17a137beabd5414cd680cd2004039740cd61339a` | `4fd1709d85ebe13ed6a46a7996491fbb227463df6592537435e81c4af91ee923` | Corrected V3 audit `PASS_STRUCTURAL`; SHA `b744d05ff7f1f5f4c5f43c3a0a90350f90aadd3972799cebdd2fbd9b75bff499` |
| Wav2Vec2 V2 | `457ebd98407ce7a33ff227e8e5c038f409f7c967a3b65622bc7ebf399b9c875d` | `96880742d7762b1ec4517fd5f869ec37b39f6d9195e6c590e93825219cf060ff` | V2 audit `PASS_STRUCTURAL`; SHA `70ac4181893d505f124a0993b01c4f9fd5d618057cb41b08bd3e26607093d28c` |

For each model I rehashed all 20 engineered WAV and PCM files against manifest SHA `a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90`; verified mono PCM16, 16 kHz and 2/8-second lengths; checked source/asset identities against the model-specific receipt and review; and checked the exact 42-record inventory (20 cases, two silence repeats, 18 matched-silence difference records, start, complete). Both silence repeats matched their original complete output and saved trace summaries. The earlier English and Wav2Vec2 V1 attempts remain `invalid_or_partial`; their parent/record bytes match the hashes carried by the V2 parents and are excluded from every count below.

Both V2 parents exited cleanly within their 1,800-second, 4-GiB RSS and 128-MiB evidence limits. I rehashed saved pre/post power-log archives and checked AC power, open lid, event coverage through each recorded worker exit and post-probe collection, and absence of saved sleep/wake/thermal events. The post-probe lags were 2.335 seconds (English) and 2.974 seconds (Wav2Vec2). I also checked saved trace stage names, geometry, per-row curve lengths and finite norms, and the internal whole-L2 arithmetic of the 18 difference rows per model. These checks cannot reconstruct omitted activations.

## Complete output results

| Model | Nonempty lexical outputs | Different from same-duration silence | 2-second / 8-second nonempty |
| --- | ---: | ---: | ---: |
| English Whisper Tiny | 20/20 | 0/18 nonsilence inputs | 10/10 / 10/10 |
| Wav2Vec2-base-960h CTC | 3/20 | 3/18 nonsilence inputs | 1/10 / 2/10 |

English Whisper's pinned tokenizer independently decoded every complete sequence. All 22 saved rows generated ` you` followed by its **checkpoint-specific EOS ID 50256**, with no length censoring. Both silence durations generated the same ` you`; every nonsilence case matched its duration's silence output. I rehashed every saved first-step raw logit vector and decoder state, checked their shapes and first-step raw argmax, and independently checked prefix, generated IDs, EOS and text decoding. The output alone therefore does not show an audio-dependent word decision for this English-only checkpoint.

For Wav2Vec2, I independently checked every complete CTC frame-ID path length and repeated-label/blank collapse against the pinned 32-token vocabulary, then compared a manual character decode with the pinned processor decode and saved full transcript. Both silence durations and all non-beep inputs decoded empty. Three beep inputs decoded nonempty strings: `d02_beep_train_low` → `ABA B BA`; `d08_beep_train_high` → `AAAAAAAAAAAAAAAA`; `d08_beep_train_low` → `A AATATATATATATATATATATATATATA`. The other beep input, `d02_beep_train_high`, decoded empty. All three deviations are in one engineered sound family, not three independent model or dataset replications.

The separately [audited multilingual Whisper V2 result](INDEPENDENT_MULTILINGUAL_DESCRIPTIVE_V2_RESULT_REVIEW_20261119.md) had a different pattern: all 20 inputs, including silence, produced a one-token English output, and four beep cases differed from the matched silence. Across these **three specific small checkpoints and 20 engineered WAVs**, the recurring silence output in both Whisper models and beep-sensitive deviations in multilingual Whisper/Wav2Vec2 are useful leads for a phase-specific causal study. The counts do not identify an audio circuit or establish transfer to other models.

## Limits

The recordings were engineered without speech by construction, but this study did not collect blinded human listening. English Whisper cannot test multilingual-script selection. The Wav2Vec2 run saved complete frame IDs and a logits hash, **not the full logits tensor**; I verified CTC decoding of saved IDs, but cannot independently prove that those IDs are the argmax of omitted logits. Neither comparator saved raw encoder tensors or all-step logits, so the trace hashes and reported sound-versus-silence differences cannot be fully recomputed from saved evidence. No patch, fresh audio, human voice, assistant action, attacker-controlled exploit, or security property was tested here.
