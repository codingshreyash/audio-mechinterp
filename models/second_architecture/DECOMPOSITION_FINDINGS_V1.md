# Voxtral: a local waveform change becomes a global feature change

**Scope:** one synthetic Piper `red`/`blue` fit pair in pinned `mistralai/Voxtral-Mini-3B-2507`; six feature-level interventions under the [committed protocol](VOXTRAL_PREPROCESSING_DECOMPOSITION_PROTOCOL.md) (`087b0d4`). The [saved run](runs/voxtral_preprocessing_decomposition_fit_v1/piper_red_blue_20261118T064322754029Z.json) has SHA-256 `7cba026aba0f81ca1edc97ed138b6fb9b9e0e85168c08e38f6bba8d3fdaba9c8`. This is a mechanistic fit case, not a security vulnerability or a held-out transfer result.

The two 16 kHz WAVs have exactly identical PCM outside spoken-word samples `[18082,26627)`. Their raw log-mel arrays also match at every frame whose STFT window misses that interval. Voxtral's pinned Whisper frontend then clips every log-mel value below `clip_max−8`. Red and Blue have different clip maxima, `1.385298` and `1.654518`, so even silent padded frames become different after the clip-wide floor. Recomputing the frontend from PCM matches both processor tensors exactly. Using one shared maximum for both clips makes every off-word and padded feature equal. The [passive spread audit](SPREAD_INDEPENDENT_REVIEW.md) independently verified the WAVs, STFT, max-floor calculation, and all 3,000 frame differences.

In native processor features, **2,689/2,689** nominal padded frames differ. They account for **59.3% of the total squared feature difference** because there are so many padded frames; their per-frame L2 difference is `0.761470`, versus a mean `3.649577` in the 56-frame nominal word window. After the bidirectional encoder and projector, all 336 nominal padded output rows also differ, but the passive hook cannot separate preprocessing, attention, and projection effects.

The causal feature experiment held audio prompt tokens fixed and crossed the mel word window (frames 112–167) with all other frames. Every run stopped on EOS. Its secondary score is `log P(Blue+EOS)−log P(Red+EOS)`; the full generated answer is the primary outcome.

| Feature condition | Complete answer | Blue-minus-Red complete-reply score |
| --- | --- | ---: |
| Native Red word + Red outside | Red | −5.7111 |
| Native Blue word + Blue outside | Blue | +6.8136 |
| Red word + Blue outside | Red | −5.4183 |
| Blue word + Red outside | Blue | +6.8979 |
| Native Red with Blue padding only | Red | −5.1022 |
| Native Blue with Red padding only | Blue | +6.8524 |

The complete answer followed the **word-window source** in both crossed hybrids. Swapping padding alone did not flip it. Thus the global padding difference is a real representation confound, but it was not sufficient to determine this color answer. The score shifts show that background and padding can still influence the model's preference without changing the complete reply.

The crossed and padding-swapped mel tensors can be off the distribution of natural audio; discontinuities introduced by feature splicing may affect the model. This result concerns these exact feature interventions on one pair. The next checks should use a second voice/phrasing and, if a security hypothesis develops, a physically realizable waveform change with a human/ASR invariant. A heatmap showing nonzero padded activations alone would have overstated the role of model-internal spread.
