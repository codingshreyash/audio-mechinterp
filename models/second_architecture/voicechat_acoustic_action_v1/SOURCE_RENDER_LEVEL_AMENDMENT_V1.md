# Pre-output source-render correction: uniform level matching

19 November 2026. **Prospective correction before any VoiceChat candidate inference; corrected render HOLD pending independent review.** This amends only the source-audio conversion rule in [VoiceChat acoustic action V1](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V1.md) and its [V2 endpoint amendment](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V2.md). The eight scripts, 12 planned source recordings, voice assignments, 20-dB noise arms, model-output endpoints, split, and human gates remain fixed. No candidate VoiceChat weights were loaded or answers observed.

## Preserved first attempt

The first TTS-only command used `render-sources` with output `runs/sources_20261119_0129`. It rendered eight complete source rows (four Piper development and four Samantha locked) and stopped at locked `A2__kokoro_bella_en_us` before its converted WAV was written. The preserved [failure record](runs/sources_20261119_0129/failure.json), SHA-256 `3f61dad0119cf7243c20d50e87fb7deddf2aad38de15b6b521deeda684c1fa35`, reports `TTS conversion would clip; preserve failed source`. The Kokoro raw WAV was retained with SHA-256 `8c9a5fa08391a4fc8026fbb97189fd4304b35bb6e895f3ac2a0042733ec01cc3`. A direct read of that raw audio found 24-kHz samples, with peak `1.0342888832`; the frozen 16-kHz resampler produced maximum `1.0849668240`, and one rounded PCM16 sample exceeded `32767`. This is a source-render feasibility failure, not a speech-intelligibility or model-behavior observation. The attempted directory must remain immutable and is not a completed corpus.

## One rule for every engine and source

For **every** Piper, Samantha, and Kokoro source in the next exclusive attempt, resample first using `scipy.signal.resample_poly` with Kaiser beta `8.6`, constant zero padding, to 16 kHz. Let `r` be the resampled float64 RMS and `p` its absolute peak. Require finite nonzero `r` and `p`. Apply the single deterministic gain

`g = min(0.1 / r, 0.85 / p)`

to the entire resampled utterance before PCM16 rounding. The `0.1` target corresponds to −20 dBFS RMS when the peak constraint does not bind; the `0.85` peak cap leaves some headroom for the separately frozen nominal 20-dB additive-noise experiment. Both constants apply to every source, with no engine-specific or case-specific override. Record `r`, `p`, `g`, post-gain peak, raw/converted/clean hashes and source frames. Reject any PCM16 clipping after the gain; do not silently saturate or substitute another recording. Append 80,000 exact zero frames after quantization as before.

This rule is chosen because the first unnormalized Kokoro source could not be represented without clipping, and because the new study requires an identical conversion policy across its three engines. It is fixed **before** the corrected TTS attempt and before any candidate answer. The per-source numeric gain is determined only by that source's waveform, never by VoiceChat output or human QA. It may change absolute loudness relative to the failed first attempt, so only a fresh complete corpus under this one rule may enter the later human gate; the eight earlier converted files are not mixed in.

## Admission

The corrected [`spec.json`](spec.json) pins the level rule and [`corpus.py`](corpus.py) applies it uniformly. An independent reviewer must verify the exact new source/spec bytes and this amendment, check the first failure evidence and fake tests, and issue a separate source-only GO before any corrected TTS render. The corrected render needs a new exclusive output directory and a full 12/12 byte audit. Timing annotations and 48 noise variants remain blocked until the clean source render is complete; human intelligibility, model inference, and security claims remain blocked after rendering until their own gates pass.
