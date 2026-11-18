# Prospective two-recognizer QA of VoiceChat calendar audio

Status: source proposal; no ASR inference authorized until an independent source review and exact freeze receipt. This screen is solely about whether the saved audio plausibly contains the intended English words. It cannot establish what VoiceChat heard or did.

## Frozen inputs

- Render commit `cdb893b`; independent file audit commit `085c532`.
- Manifest `rendered_v1/20261118T1145Z/render_manifest.json`, SHA-256 `bfafef903eaff14a444c015d0c38886e3bedde928ef8c5ca9100aac00370ca7c`.
- `scripts.json`, SHA-256 `c3416890395c89c03bdbf2bad9e62e0215bf46d6d9a749f8cb043af1cbe2f99d`.
- Exactly six cases (`A`, `B`, `C`, `D`, `A_literal`, `B_literal`) × two voices (`apple_samantha_en_us`, `piper_amy_en_us`): 12 final WAVs. Use the manifest's `final_path` and `final_sha256`; do not discover candidates by glob or choose recordings after ASR.
- Two locally cached, exact-revision ASR models and dependency/source file hashes are those in `eval/audio_jlens/corpus_v2/ASR_MODEL_PINS_PROSPECTIVE_V1.json`, SHA-256 `1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b`: `openai/whisper-tiny.en` and `facebook/wav2vec2-base-960h`. Use CPU, four threads, greedy decoding. Do not download or update weights.

## Order and outputs

1. Validate every pinned source and all 12 final WAV hashes, mono 16-kHz PCM16 geometry, exact 80,000-frame zero tail, and complete case–voice matrix. Build opaque input IDs before inference. No script/reference text enters the ASR model call.
2. After source review, create `ASR_QA_FREEZE_V1.json` with the exact `status`, `runner_sha256`, `protocol_sha256`, `manifest_sha256`, `scripts_sha256`, `pins_sha256`, and `v2_asr_source_sha256` specified by the runner. The runner refuses inference without this receipt.
3. Run Whisper and Wav2Vec2 serially, in separate new output directories. Each writes metadata, then one raw JSON line per input (opaque ID, WAV/PCM hashes, transcript, token IDs, decoding metadata), flushing it before the next input. The raw file contains no case, target wording, normalized text, or score. An interruption leaves an error receipt; there is no resume or cherry-picking.
4. Only after **both** 12-row raw logs and completion receipts exist, run `--compare-only` to join them to the frozen scripts. Preserve raw logs unchanged. Report exact normalized full-script match and word error rate for each of 12 × 2 observations. Normalization is Unicode NFKC, `casefold`, punctuation to spaces, whitespace collapse. Report counts with denominator 12 per recognizer and 24 jointly, including errors; do not remove a failed case from the denominator.
5. A separate blinded human listening check must record wording, title, date, and confirmation phrase before VoiceChat testing. ASR agreement is a screen, not a substitute for human listening. Any unclear/misworded audio is reported as such; replacement requires a new prospective fixture and review, not a selective rerender after seeing VoiceChat output.

No ASR score or automated threshold is declared a model-security gate here. The comparison is descriptive, and it must not select cases based on later VoiceChat results. A clean screen only permits the planned listening QA. It does not authorize or establish a VoiceChat finding.
