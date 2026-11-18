# Frozen source candidate: two recognizer v2 ASR screen

This is a **prospective source candidate**, not an ASR result or an input-quality pass. No v2 ASR inference was run while this candidate was prepared. The [QA protocol](ASR_LISTENING_QA_PROTOCOL_PROSPECTIVE_V1.md) still requires both complete 480-clip ASR passes, blinded human listening, an independent auditor, and a separate quality receipt before Qwen v2 model inference.

## Exact inputs and models

The runner reads [ASR_INPUTS_PROSPECTIVE_V1.json](ASR_INPUTS_PROSPECTIVE_V1.json), a reference-free projection of the existing case map. Each of its 480 rows contains only an opaque ID, clip path, WAV and PCM SHA-256 values, and frame count. A synthetic test checks its byte-for-byte field projection from the frozen case map. The runner does not open that map, and neither model receives a transcript, target word, candidate list, filename, or Qwen output. Filename words are used only by the host to locate the WAV. The runner rechecks the frozen manifest, binding, input list, every WAV, and every decoded PCM buffer before proceeding.

The two checkpoints are pinned by exact Hugging Face revision and SHA-256 of **every locally used checkpoint/processor/tokenizer file** in [ASR_MODEL_PINS_PROSPECTIVE_V1.json](ASR_MODEL_PINS_PROSPECTIVE_V1.json):

| Recognizer | Exact revision | Safetensors SHA-256 | Decoder |
| --- | --- | --- | --- |
| [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en/tree/87c7102498dcde7456f24cfd30239ca606ed9063) | `87c7102498dcde7456f24cfd30239ca606ed9063` | `db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad` | Greedy, beam 1, no sampling, max 128 new tokens; log EOS and token IDs. |
| [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h/tree/22aad52d435eb6dbaf354bdad9b0da84ce7d6156) | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a` | Framewise argmax and CTC collapse; log all argmax IDs. |

The Wav2Vec2 model card specifies 16 kHz English ASR and demonstrates greedy CTC decoding. These are different recognizer and decoding architectures, but their agreement is not a human intelligibility finding. Both snapshots are locally cached, and the runner uses offline, local-only loading. The pins also record Python, Torch, Transformers, Hugging Face Hub, NumPy, SoundFile, and the SHA-256 of nine relevant installed Transformers source files.

## Execution boundary

`asr_screen_v2.py --validate-only` verifies the corpus, cached checkpoints, and installed source without loading either ASR model. It does not write an ASR result. The subsequent `--run whisper` and `--run wav2vec2` modes require a separately committed `ASR_SCREEN_FREEZE_V1.json` with `status: independently_reviewed_asr_screen_frozen`, the exact runner SHA-256, and the expected `manifest`, `binding`, `cases`, `inputs`, `pins`, and `protocol` SHA-256 values. That receipt is **absent**. The current source cannot start either inference pass through its CLI.

After independent review and a freeze receipt, run the two passes serially on CPU, with four intra-op threads and one inter-op thread. Use a fresh output directory per recognizer; the runner refuses an existing directory and cannot resume or overwrite it. It writes a raw transcript and token sequence to an append-only JSONL file, flushing and syncing each row, before reference text is read by any comparator. A complete receipt is written only after all 480 rows pass identity, order, hash, and schema checks. An exception writes an incomplete receipt; a hard stop leaves the absence of a complete receipt. No output is normalized or compared by this runner.

Before any run, an independent reviewer should inspect the source and tests, independently confirm the Hugging Face revision/file pins, and freeze an auditor and listening interface in accordance with the QA protocol. Do not overlap this ASR inference with the ongoing Voxtral development run. Once both raw passes are complete, the separate auditor must normalize and score all 480 transcripts, retain every failure, then incorporate human listening. No partial ASR result can satisfy the go gate.

## Prepared checks, with no model inference

- `reproductions/audio_prior/.venv/bin/python eval/audio_jlens/corpus_v2/asr_screen_v2.py --validate-only`
- `cd eval/audio_jlens/corpus_v2 && ../../../reproductions/audio_prior/.venv/bin/python -m unittest -v test_asr_screen_v2.py`

These checks validate 480 WAV/PCM inputs, both cached model snapshots and installed source, exact reference-free case projection, missing-freeze rejection, and synthetic malformed raw logs. They do not test actual ASR transcripts or model loading, which are intentionally deferred.
