# Independent ASR proxy screen, version 1

19 November 2026. **Source-only plan before any ASR proxy inference.** This screen records how an already-pinned Wav2Vec2 CTC model transcribes the 16 authentic human-speech WAVs. The target mechanism checkpoint is Whisper Tiny, so Wav2Vec2 is a separate proxy. ASR output is a quality flag, not a human intelligibility verdict.

## Frozen input and model

- Input is the four-speaker [source manifest](source_v1/manifest.json), SHA-256 `c0593a4631191b7643714651c751fde29ebc6323bf1d59680d3a4c1d1bd38051`, exactly 4 rows × `base`, `z32000`, `n32000_s0`, `n32000_s1` = **16 WAVs**. Preserve manifest order and all outputs, including errors and mismatches. The independent [technical review](SOURCE_INDEPENDENT_TECHNICAL_REVIEW_V1.md) passed before this plan.
- Use `facebook/wav2vec2-base-960h` at revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, with the checkpoint and processor hashes in the existing [descriptive pins](../../models/second_architecture/nonspeech_probe_v1/descriptive_pins_v1.json). Require NumPy `2.3.5`, Torch `2.15.0`, and Transformers `5.20.2` in `models/second_architecture/.venv`, plus the pinned installed Wav2Vec2 source hashes.
- Decode each canonical WAV as mono PCM16 16 kHz. Convert to FP32 by division by 32768, run the stock `Wav2Vec2Processor` once with `sampling_rate=16000`, `padding=False`, then one CPU FP32 `Wav2Vec2ForCTC` forward in eval/inference mode with no attention mask. Greedy frame argmax and the pinned processor's CTC decoder give raw text. No prompt, correction, language model, text normalizer, candidate-model state, or transcript-informed filtering enters inference.

## Evidence and bounds

One fresh attempt has 16 ordered call records. For each call save the entire contiguous FP32 CTC logits as a `.npy` sidecar and its SHA-256, full framewise argmax IDs, decoded raw string, source WAV/PCM hash, input-values hash, model/source identity, frame and vocabulary geometry, and elapsed time. The saved-only auditor must rehash every sidecar and recompute every argmax and transcript using the pinned processor without loading model weights. Keep raw case outcomes visible; do not report only an aggregate word-error rate.

The parent runs one local CPU worker, stops it at **600 seconds** whole elapsed, **4 GiB** process-tree RSS, or **256 MiB** attempt evidence, and preserves any partial attempt as invalid. Require AC power and no thermal emergency at start; record power before and after. No automatic retry under the same attempt path. Render and audit the blinded listening bundle separately. Both independent human listening records over all 16 clips remain necessary before a perception-invariant claim. Target Whisper inference is a later, separately frozen and reviewed step.

For descriptive analysis, compare normalized zero-tail and noise-tail proxy text for each speaker/seed while retaining all sixteen raw outputs. A proxy mismatch is a flag for human review. Proxy agreement does not establish that a human hears the reference, and a proxy match to the source transcript does not validate an unheard clip.
