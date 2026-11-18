# Independent code review: Voxtral preprocessing decomposition

Reviewed 2026-11-18 before MPS inference. I read `VOXTRAL_PREPROCESSING_DECOMPOSITION_PROTOCOL.md`, the runner, its imported fit-pair/processor helpers, and the installed pinned Whisper and Voxtral forward code. I ran CPU-only preflight and two CPU tests. I did not load model weights or run development/test inputs.

## Decision

**No blocking mismatch found for the six-condition fit run.** The causal claim must remain limited to the specified mel-feature interventions on one exact Piper pair.

## Checks

- The input path uses the Piper red/blue entries from the controlled-pair manifest. `select_clip` verifies each WAV hash against that manifest; `prepare_pair` verifies equal length, 16 kHz mono, identical PCM before sample 18,082 and from sample 26,627 onward, and identical audio-instruct prompt IDs. The model loader uses the local Voxtral revision `3060fe34b35ba5d44202ce9ff3c097642914f8f3` with no download. The runner checks SHA-256 of the installed Voxtral model, processor, and Whisper feature-extractor source files.
- `independent_raw_log_mel` matches the pinned Whisper math: 480,000-sample right padding, centered Hann STFT (FFT 400, hop 160), squared magnitude excluding the final STFT frame, the checkpoint mel filters, log floor at `1e-10`, clip-wide `max−8` floor, and `(x+4)/4`. The checkpoint sets dither to zero. CPU preflight measured **0.0 maximum absolute error** against both processor mel tensors. Raw off-word and shared-maximum off-word frame L2 maxima were also **0.0**; the pair's raw maxima were 1.3852976561 (red) and 1.6545178890 (blue). This validates the proposed decomposition at the input-feature level, before any model inference.
- Six tensor constructions match the protocol and fixed order: two byte-identical processor endpoints, two crossed word/outside hybrids over frames 112–167, and two swaps beginning at frame 311. Every condition has a float32 tensor SHA-256 in preflight. The runner keeps the native prompt batch, replacing only `input_features`, and iterates exactly over `CONDITION_ORDER`.
- Teacher forcing appends the candidate word token to the prompt and reads the word logit at `prompt_len−1` and EOS at `prompt_len`; the pinned `logits_to_keep=0` returns all positions. It verifies `Red=[12846]`, `Blue=[24851]`, and generation EOS ID 2. Generation is deterministic with cache and at most 32 new tokens; it stores full generated IDs, decoded answer and EOS. The two native endpoints run first, and a failed complete color answer or EOS stops before hybrid inference.
- The passive hook is on `model.model.multi_modal_projector` and requires one `[375,3072]` capture per generation. Per-row L2 profiles compare each condition to both native captures and summarize nominal word rows, other real-audio rows, and padded rows. The hook returns no replacement.
- MPS inference requires `--run`; the default preflight is CPU-only. The 30 GiB allocator fraction, periodic memory sampler, and after-load/after-condition driver checks are present. Offline settings, local weights, and exclusive output creation prevent network fetching and silent overwrite. `py_compile` passed; CPU unit tests passed **2/2**. The unit tests cover six-factor construction and candidate-plus-EOS score indexing, while the preflight checked the real fit WAVs and source hashes.

## Reporting limits

- The script compares WAV hashes with the *current* manifest and records that manifest's SHA-256 (`318807c92971babacad93b69b7d8478e29f397219da0cb9ed6e3134649aea98b` in this preflight), but does not bind an expected manifest SHA literal. This is a provenance limitation if the manifest and WAVs were both edited before a rerun; the saved run report still permits later audit against the frozen Git revision.
- The crossed and padding-swapped mel tensors may be off-manifold. A changed answer supports a causal effect of that exact feature tensor in this model. It does not by itself show that ordinary speech, a particular hidden circuit, or a security attack has the same effect.
- The native-answer gate accepts the color word followed by an optional period or exclamation mark. It requires EOS and rejects extra prose or Markdown; report this acceptance rule when quoting complete-answer rates.
