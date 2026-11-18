# Independent source and design review: Whisper FIT word-window causal assay

18 November 2026. Review target: `WHISPER_FIT_WORD_WINDOW_CAUSAL_PROSPECTIVE_V1.md` as first written, before any assay model inference. **Decision: HOLD.** This is a plan/source review, not a result review.

## Checks that passed

- The frozen v1 manifest SHA-256 named in the plan matches `rendered/BINDING.json`. Its `color_05`, `fit_chosen` records are Piper and Kokoro `silver`/`teal` exact-PCM matched pairs, with equal within-engine durations, identical PCM outside the word slot, and different PCM inside it.
- Piper's slot is `[30527,49847)` at 16 kHz. Its seconds are `[1.9079375,3.1154375)`, giving `W=[90,161)` under the plan's floor/ceil plus five-frame rule and `P=[10,81)`. Kokoro's slot is `[33920,57120)`, seconds `[2.12,3.57)`, giving `W=[101,184)` and `P=[10,93)`. Both prefix controls precede the word slot.
- The existing independent Whisper ASR audit records normalized-exact transcripts, exactly one intended target word, and no paired alternative for all four files. The plan correctly requires fresh unmodified baselines and does not promote that old screen to intervention evidence.
- Four baseline decodes plus three arms at three sites in four directional cases (`36`) plus four whole-state donor controls plus eight random controls total `52` decodes, of which `48` are interventions. The stated cap is consistent.
- The pinned Whisper checkpoint exists locally at the stated revision. The isolated `reproductions/audio_prior/.venv/bin/python` has Torch 2.15.0, Transformers 5.12.1, and SoundFile 0.14.0. The earlier negative aligned-encoder Jacobian readout did not test this clean-to-corrupt causal replacement.

## Corrections required before GO

1. In the installed Transformers 5.12.1 `modeling_whisper.py`, `WhisperEncoder.forward` applies `self.layer_norm(hidden_states)` **after** `self.layers` (`line 642`). A replacement at the output of block 3 therefore passes through a final normalization before decoder cross-attention. The plan currently says this post-block-3 replacement is "at the final encoder state consumed by decoder cross-attention." Correct the claim to "before final encoder layer normalization," or predeclare the final-normalization output as the site instead. If the latter, update the capture and control definitions consistently.
2. Define the exact first intervention module and hook contract. The state after convolution, positional addition, and dropout is the *input* to `model.model.encoder.layers[0]`, accessible through a forward pre-hook; a hook on `conv2` is earlier than positional addition. State how tuple/keyword arguments are preserved when replacing rows. This matters because the plan's first site otherwise has more than one possible implementation.
3. Define what the Gaussian control's "equal norm" means. Match each time row's perturbation L2 norm to that row's donor-minus-recipient difference before normalizing seeded noise, or explicitly predeclare a single whole-window norm. The choice affects the comparison, and zero-norm rows need a stated zero edit.

Additional interpretation caution: identical waveform samples in `P` do not guarantee identical log-mel or late encoder activations there. Feature extraction and bidirectional encoder attention can spread the word difference. Treat `P` as an off-slot **intervention-location control**, not an assumed zero-difference state.

No model was loaded or run for this review, and no held-out corpus or Voxtral pilot outcomes were opened.
