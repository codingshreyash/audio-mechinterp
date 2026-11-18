# CPU ASR audit of the 12 fit-only synthetic clips

Date: 2026-11-18. Audited the 24 **processed** WAVs in [`outputs/semantic_fit_smoke_12/render_manifest.json`](outputs/semantic_fit_smoke_12/render_manifest.json), SHA-256 `aabfe2060d2f295b80faca77fcaddbaa088913f60e2adb59bf01f384983c20e0`. The source subset was the frozen [`../semantic_bridge/smoke.jsonl`](../semantic_bridge/smoke.jsonl), SHA-256 `12f4f6ab4d80542e075422ed17a5c3edbc2b7fb52d13852e96374be76fdb4c96`; all 12 records are `split=fit`. No TTS file or source item was changed, and no development or test item was audited.

## Method

Whisper Tiny English, [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en/tree/87c7102498dcde7456f24cfd30239ca606ed9063) revision `87c7102498dcde7456f24cfd30239ca606ed9063`, was loaded from the local cache and run on **CPU only** with PyTorch `2.15.0`, Transformers `5.12.1`, four CPU threads, greedy decoding, `max_new_tokens=128`, and the processor's feature attention mask. It received only the rendered audio, never the reference script. Each raw and processed WAV was SHA-256 checked against the render manifest before transcription.

WER and CER use the [Whisper pilot's normalization](https://github.com/kennethli319/audio-jacobian-lens/blob/1784e62e4caaf801bfc8dc1208860703b5f3d074/jlens/evaluate_whisper_cli.py): casefold; retain Unicode letters/numbers and internal apostrophes; strip punctuation; join normalized words with spaces for CER. The reference is the **unchanged** `audio_input.script`, so digit transcription and American/British spelling differences count as errors. Word alignments and complete transcripts are in [`ASR_AUDIT.json`](ASR_AUDIT.json), SHA-256 `9376d1ca7e44b8dcc9040d305f41514cdeb6fe0384e5f235bec411503933e82b`.

```bash
cd /Users/shrranjan/audio-mechinterp
reproductions/audio_prior/.venv/bin/python eval/tts/audit_asr.py \
  eval/tts/outputs/semantic_fit_smoke_12/render_manifest.json \
  --output eval/tts/ASR_AUDIT.json > eval/tts/asr_audit.log 2>&1
```

## Results

| Engine | Word errors / reference words | WER | Character errors / reference characters | CER | Possible severe omissions |
|---|---:|---:|---:|---:|---:|
| Piper VITS | 11 / 228 | 4.82% | 29 / 1,230 | 2.36% | 0 / 12 |
| Kokoro | 6 / 228 | 2.63% | 21 / 1,230 | 1.71% | 0 / 12 |

The automated omission flag is WER ≥30% or at least three consecutive deleted reference words. No clip met it. The largest per-clip mismatch was `association:code-h`: Piper WER 24%, Kokoro WER 20%. Both transcripts rendered “distrusted” as “distressed” and “fixed” as “fix”; both also wrote “forty-third”/“forty-two” as digits. This is a **manual listening candidate**, since a single ASR model can make the same error on two intelligible voices. Piper also wrote “realised” for “realized” in `association:interview`, an orthographic difference.

All 24 WAVs were 16 kHz mono, finite, nonempty, and hash verified. Durations were 3.39–7.67 s for Piper and 4.68–9.45 s for Kokoro. RMS was 0.070–0.093 and 0.078–0.084 respectively; peaks were 0.540–0.688 and 0.670–0.962. No near-silence, hard clipping, large DC offset, >2 s near-zero dropout, or raw-to-processed duration shift above 10 ms was flagged. The longest near-zero run was 0.38 s (Piper) and 0.92 s (Kokoro). These are automated waveform checks; no human playback judgment is claimed.

The unfinished multihop prompts remained lexically recognizable, but Whisper supplied end punctuation such as `is.` or `is...`. A spoken waveform cannot carry the typed trailing-space completion cue. Four of these 12 source records differ from their typed prompts only by that trailing space. The ASR audit supports that the main words are present; it cannot establish that audio and typed prompts create the same continuation task.

**Interpretation:** This is a useful quality screen for the fit subset. Whisper's transcription is another model output, so low WER does not prove exact spoken content or absence of a subtle acoustic artifact. Hold manual review of `association:code-h` and any future flagged clips before treating them as clean evaluation inputs.
