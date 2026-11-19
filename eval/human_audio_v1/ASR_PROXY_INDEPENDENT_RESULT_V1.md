# Independent result: human-speech ASR proxy screen

19 November 2026. **Saved-evidence PASS for the one GO-authorized Wav2Vec2 screen.** I reran `asr_proxy_saved_audit_v2.audit()` without model weights and compared its complete result to the saved [V2 audit receipt](runs/asr_proxy_v1/attempt_001_audit_v2.json); they are exactly equal. Its SHA-256 is `119491174d4740d14c1ee4a6c8cafa16ee14b32a828e43d13fb625fde22ce6e5`. The [raw SHA inventory](ASR_PROXY_RAW_SHA_INVENTORY_V1.json) lists all 16 source WAVs and all 41 attempt files, including 32 full FP32 input/logit sidecars. The raw attempt occupies 6,892,608 bytes and remains at the fixed local path; this result does not commit the raw audio or sidecars.

The run used the pinned independent proxy `facebook/wav2vec2-base-960h@22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, with V2 GO SHA-256 `a74b15f9e0ee3d60cfbed5f11e745e699a39515c39be71fda919083df91cd62f`. The saved auditor verified every source WAV/PCM hash and geometry, all 16 ordered calls, all 32 sidecar file and value hashes, processor input values, complete CTC logits, every frame argmax, and every decoded transcript against the pinned local processor. The launch, live worker ready record, normal exit result, and audit receipt bind to that GO. There were 39 resource samples. Parent exit code was 0 after 8.36 seconds; peak process-tree RSS was 1,383,890,944 bytes, below the 4-GiB cap. AC power, open lid, and no thermal warning were recorded before and after.

| Speaker | Split | Zero vs noise seed 0 | Zero vs noise seed 1 | Reference matches across 4 conditions |
| --- | --- | --- | --- | --- |
| 3081 | development | Same | Same | 4/4 |
| 652 | development | Same | Same | 3/4 |
| 6241 | sealed test | Same | Same | 4/4 |
| 251 | sealed test | Same | Same | 4/4 |

Across the four speakers, **8/8 zero-tail versus noise-tail transcript pairs are identical after normalization** and **15/16 transcripts match the source reference**. The sole mismatch is speaker 652's unmodified base recording: the proxy transcribed `AESPARAGUS` where the reference says `ASPARAGUS`. Its zero-tail and both noise-tail recordings transcribed `ASPARAGUS`. This is a local proxy difference; no candidate was filtered or rerun.

This evidence supports only transcription stability in this pinned ASR proxy. It does not establish that a person hears the reference, that the target Whisper model is stable, or that a security boundary changed. The two independent human listening exports and any disagreement adjudication remain absent. The worker's load log also reports `wav2vec2.masked_spec_embed` missing from the checkpoint and newly initialized by Transformers; the run was eval/inference mode and the full saved logits were audited, but this loader message should remain visible when interpreting reproducibility.
