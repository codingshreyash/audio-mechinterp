# Voxtral preprocessing decomposition: fit-only protocol v1

**Question.** The exact Piper red/blue PCM pair differs only in the spoken word, but the pinned Whisper feature extractor yields differences throughout the 30-second feature tensor. Determine how much of that input-side difference is caused by clip-wide log-mel normalization, and whether the word-window, other real-audio frames, or padded frames causally alter Voxtral's *complete* answer. This is a mechanistic fit experiment on one pair, not a security finding or a generalization test.

## Frozen sources and preprocessing check

Use the exact [Piper pair manifest](../../eval/tts/outputs/controlled_pair_v1/controlled_pair_manifest.json): 16 kHz mono WAVs, same length and sample-identical outside word samples `[18082, 26627)`. The script rechecks both WAV SHA-256 values and outside-word PCM equality. Model checkpoint: `mistralai/Voxtral-Mini-3B-2507@3060fe34b35ba5d44202ce9ff3c097642914f8f3`, local split safetensors only. Python environment is [frozen](requirements.freeze.txt). The script records its own SHA-256 and checks the installed Transformers source SHA-256 against the official v5.20.2 tag commit `18329f1065382e54db698e242cf98bd3f3e5a766`: [Voxtral model](https://github.com/huggingface/transformers/blob/18329f1065382e54db698e242cf98bd3f3e5a766/src/transformers/models/voxtral/modeling_voxtral.py), [Voxtral processor](https://github.com/huggingface/transformers/blob/18329f1065382e54db698e242cf98bd3f3e5a766/src/transformers/models/voxtral/processing_voxtral.py), and [Whisper feature extractor](https://github.com/huggingface/transformers/blob/18329f1065382e54db698e242cf98bd3f3e5a766/src/transformers/models/whisper/feature_extraction_whisper.py).

Before any MPS run, reconstruct each 128×3,000 mel tensor independently from PCM: zero-pad to 480,000 samples; centered Hann STFT with FFT 400 and hop 160; magnitude squared; the pinned mel filterbank; `log10(max(power, 1e-10))`; clip-specific floor at `raw_max − 8`; then `(value + 4) / 4`. Require maximum absolute error ≤1e-6 against the processor output. Save each raw log-mel maximum. Repeat the floor/normalization with a **shared maximum** across the pair; require off-word and padded-frame differences ≤1e-6. The conservative mel word window is frames 112–167 inclusive; nominal padding starts at frame 311. These tests distinguish preprocessing from model-internal spread.

## Six predeclared model inputs

All six use **identical** audio-instruct prompt token IDs and differ only in the `[1,128,3000]` `input_features` tensor. `W` is mel frames 112–167, and `O` is every other frame, including padding. For each condition the script saves a SHA-256 of the resulting float32 tensor. Native endpoints are byte-identical to the processor tensors.

| Condition | Word window | Other frames | Interpretation |
| --- | --- | --- | --- |
| `word_red_outside_red` | red | red | Native red endpoint |
| `word_blue_outside_blue` | blue | blue | Native blue endpoint |
| `word_red_outside_blue` | red | blue | Crossed feature hybrid |
| `word_blue_outside_red` | blue | red | Crossed feature hybrid |
| `red_with_blue_padding` | red | red through frame 310; blue from 311 onward | Padding-only swap |
| `blue_with_red_padding` | blue | blue through frame 310; red from 311 onward | Padding-only swap |

The crossed and padding-swapped features may not correspond to any natural waveform. Discontinuities at the splice and at the 30-second padded tail can themselves affect the model. Treat a changed answer as an effect of the *specified feature intervention*, not proof that a naturally recorded word or hidden direction works the same way. If either native endpoint fails to produce the expected complete color answer and EOS, stop interpretation of the hybrids.

## Frozen readouts and bounds

For each condition, record greedy `max_new_tokens=32` generated suffix token IDs, full decoded answer, and EOS. Score complete `Red` and `Blue` replies **including EOS** by teacher forcing: sum log probabilities of each candidate token at its preceding position and EOS after the candidate; report `LL(Blue + EOS) − LL(Red + EOS)`. The pinned tokenizer gives `Red=[12846]`, `Blue=[24851]`, EOS `2`; fail if this changes. This contrast is a secondary continuous readout, not a substitute for a changed complete answer.

A passive hook captures the post-projector `[375,3072]` tensor during generation for each condition. Save per-row L2 distance to both native endpoints across all 375 rows, with summaries for nominal word rows 13–21, other real-audio rows, and nominal padded rows 39–374. The encoder is bidirectional, so these are indexing groups, not isolated word representations.

Run at most six generations plus two teacher-forced candidate passes per condition, batch size 1, BF16 MPS, with the existing 30 GiB per-process MPS guard. No cloud GPU, no dev/test pairs, no weight downloads, and no edits to existing results. The default command performs **CPU-only preflight**; only `--run` enables model inference. Results go to a new `runs/voxtral_preprocessing_decomposition_fit_v1/` artifact with no overwrite.
