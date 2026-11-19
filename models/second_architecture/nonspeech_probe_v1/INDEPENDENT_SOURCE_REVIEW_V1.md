# Independent source review: non-speech mechanism pilot V1

19 November 2026. Reviewed immutable source commit `bb0a5bd55e63c396c5408fd9714d1d528c9c9792`. **No candidate model weights were fetched or loaded; no candidate output was generated.**

## Decision

**GO for the frozen 20-case source and descriptive design. HOLD for model inference and causal intervention.** The source produces the declared engineered non-speech WAVs, and the ten pinned multilingual Whisper metadata/tokenizer files match the remote revision byte for byte. The installed Transformers 5.20.2 path can select a language automatically with this generation config. The descriptive runner, weight identity, resolved runtime config, and separate intervention source still require prospective implementation and review. In particular, Whisper language detection is a **separate forward pass**; a decoder-only or main-generation-only patch cannot establish full donor parity for the language decision.

## Exact source identities and independent checks

| Artifact | SHA-256 |
| --- | --- |
| `spec.json` | `2dd69b687b00d90ed4e2d8bbc266039943498c31532634440f2d3bd0a0ca0951` |
| `render.py` | `8c20404965c30b11c347e562f1bb8b399e433c58e64d8674f62dab98dcf98cb1` |
| `test_render.py` | `4ac6925fc4af0a896dc4295c7cf39ec1f51d3f81469d7fac390c5c2526b2d069` |
| Prospective protocol | `2815d24b7f05aea778aec71c5a76746f952eca92a680d640c3544ab57d832b32` |
| Metadata pin | `1482d83b52380186d5adac196fb02100167dfccc13b6fd2ad56a0f8b2d942675` |
| Installed `generation_whisper.py` | `e1531401f2cc2e389f13a7a55fa22542132164fefe9d431644c9368f3147583a` |
| Installed `feature_extraction_whisper.py` | `dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb` |
| Installed `modeling_whisper.py` | `bf4f022d37ed523d09a472dae267b86007aaae0cd39d2ce37927de44bfc5b9e6` |

Under local NumPy 2.3.5, Torch 2.15.0, and Transformers 5.20.2, the four source unit tests passed. I rendered the source twice into new temporary directories and independently read each RIFF/WAVE payload with Python's `wave` reader. All 20 case IDs and WAV hashes matched across renders; each file is mono, 16-bit PCM, 16 kHz, exactly 32,000 or 128,000 frames, with no extra payload frames or clipping. Both silence arrays contain only zero; both dither arrays contain only -1, 0, +1 LSB with nonzero samples. The 16 positive-level rows reach their target full-duration RMS within one 16-bit LSB (largest absolute error `2.36e-6`). The 700-Hz tone peaks at 700 Hz, the beep-off intervals are exact zero, samples outside each 128-sample click window are zero, and the band-noise rows have more than 99.9% of PCM spectral energy in 300–3400 Hz. Highest observed peak was `0.57953`, under the frozen `0.8` ceiling. These checks validate signal construction; they do not prove that listeners can never perceive speech-like patterns.

Adversarial no-weight checks rejected an existing output directory, a relative output path, an absent parent, invalid PCM dtype/shape, and a changed NumPy version. An injected renderer failure left `started.json` and `failure.json` and did not produce a completed manifest. The planned inference order is **lexicographic case ID**; the renderer's manifest order is different, so the future runner must sort explicitly.

## Pinned checkpoint and language path

I independently fetched only the ten small metadata/tokenizer files named in `whisper_tiny_metadata_pin.json` from `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; every SHA-256 matched the pin. No weight file was fetched. The remote `generation_config.json` has `is_multilingual=true`, `forced_decoder_ids=[[1,null],[2,50359]]`, and no fixed `language` or `task`. The remote `config.json` carries an older English forced prefix `[[1,50259],[2,50359],[3,50363]]`. The installed `WhisperGenerationMixin._retrieve_init_tokens` uses the generation config first; with `task='transcribe'` and `language=None`, it ignores `forced_decoder_ids`, invokes `detect_language`, and constructs `[50258, detected_language_id, 50359, 50363]`. A no-weight test using the actual pinned JSON, a fake model, and a stubbed Russian result `50263` produced `[50258,50263,50359,50363]` and exactly one detection call. This establishes the **code path**, not what language the checkpoint will choose for any sound.

The detected language token is chosen in a separate `detect_language()` model forward and then placed in the decoder prefix. It is not one of the main transcription decoding steps, and its score is not a normal per-step transcription score. The runner must record the detection-pass scores or logits and the resolved prefix separately from the subsequent generated text token scores. In the checked configs, `pad_token_id=eos_token_id=50257`; count actual generated steps and inspect the full returned sequence when distinguishing EOS from censor or padding. Do not infer completion solely from a trailing `50257`. The generation config has no `no_speech_threshold`; preserve that resolved setting rather than calling an empty transcript the model's explicit no-speech classification.

For a full final-encoder donor intervention, freeze whether the hook applies to the detection pass, the transcription pass, or both. Full donor parity on the **complete** output, including selected language, requires donor-consistent treatment of both passes or an equivalent precomputed donor encoder output. If the intervention covers only transcription, score that as a narrower intervention. Confirm the exact final-encoder site and that the hook fires the expected number of times before treating a failed parity check as a substantive negative.

A useful **future, separately frozen** 2×2 causal design would put the same-duration silence donor's final encoder state into (a) neither pass, (b) language-detection pass only, (c) transcription pass only, and (d) both passes, on all 18 non-silence cases. Save the selected language ID and full prefix separately from lexical tokens and EOS for every arm. This can test whether the detection prepass controls the language choice while the transcription pass controls later text; it cannot be added after choosing cases from the observed outputs. First verify that the two passes expose compatible encoder-state geometry and identical hook semantics, and keep self/read-only/full-donor and random controls.

## Outcome and novelty limits

The protocol correctly requires all 20 descriptive rows, all 18 non-silence intervention cases, complete outputs, EOS/censor status, and self/read-only/random controls. Its stated interpretation is appropriately narrow: a successful full-state transplant shows sufficiency at an artificial interface for a case; it does not identify a unique neuron or an attacker-suppliable vulnerability. No candidate-model behavior or causal effect is established by this review.

Non-speech Whisper transcription is prior art. The [OpenAI Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md) warns about text not spoken in the audio; [Barański et al., ICASSP 2025](https://arxiv.org/abs/2501.11378) directly studied hallucinations induced by non-speech sounds; and [Wang et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.html) tested decoder-head masking and fine-tuning on non-speech hallucination. A tiny-checkpoint observation or a single state transplant cannot by itself establish a new general audio mechanism or a security finding. A stronger claim needs held-out sounds, a larger checkpoint, and a behavior with a defined security consequence.

## Gates before a live attempt

1. Pin candidate weight bytes and every resolved checkpoint/tokenizer/config file; record package and relevant source hashes at runtime.
2. Implement and independently review bounded one-sample descriptive runners. Assert `task='transcribe'`, `language=None`, no translation, resolved prefix behavior, and automatic detection in the actual loaded model before opening the 20-case outputs.
3. Record language detection-pass logits and inserted prefix separately from main generated IDs/scores; save complete sequences, step counts, EOS/censor, and failure rows. Check that every case reaches the report denominator.
4. Freeze and separately review the intervention implementation, including hook firings in the language-detection and transcription passes, full donor parity, exact self parity, equal-norm random controls, and active/padded row geometry.

The current prospective protocol already says **HOLD** before candidate inference. This review does not lift that gate.
