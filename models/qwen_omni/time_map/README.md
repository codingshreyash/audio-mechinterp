# Pinned Qwen Omni audio-row extraction

This directory captures audio encoder and audio-to-language projection activations for **fit-only** WAVs from the [frozen 12-item render manifest](../../../eval/tts/outputs/semantic_fit_smoke_12/render_manifest.json). It records exact packed tensor indices and audio placeholder positions. It makes no claim about recognized words, security behavior, or causality.

## Source and row meaning

The checkpoint is `Qwen/Qwen2.5-Omni-3B` revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`; Transformers source commit is `e7e568dc7673ac99a87660c7c50e84830268285c`. `extract.py` validates the installed commit, local model config hash, rendered WAV hash, and processor batch shape. It uses the already reviewed [Thinker adapter](../adapter.py), with no Talker/speech decoder.

The pinned [audio encoder source](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L717-L912) splits valid mel frames into chunks of `2 × n_window = 200` frames. The feature extractor uses 16 kHz and a 160-sample hop, so a full chunk spans a **nominal** two seconds. Conv1 has kernel 3, stride 1; conv2 has kernel 3, stride 2. Pinned utilities pack valid conv rows, then pool adjacent encoder rows before projecting them to 2048 dimensions. Each projected row is inserted into one `<|AUDIO|>` placeholder of the text sequence by `masked_scatter` ([source](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1891-L1898)). `index.py` mirrors those indexing formulas and its three CPU tests compare cumulative encoder row boundaries and pool indices with the installed pinned source across chunk-edge lengths.

**A row is not a word time.** Even though each conv row has a small pre-attention receptive field, the encoder layer uses self-attention over its entire chunk. The row map therefore records both the local pre-attention conv span and the entire attention-scope chunk. Its seconds are nominal mel-hop coordinates, not independent word alignments.

## Run

```bash
cd /Users/shrranjan/audio-mechinterp/models/qwen_omni/time_map
/Users/shrranjan/audio-mechinterp/models/qwen_omni/.venv/bin/python -m unittest -v test_index.py test_extract.py

cd /Users/shrranjan/audio-mechinterp
models/qwen_omni/.venv/bin/python models/qwen_omni/time_map/extract.py \
  eval/tts/outputs/semantic_fit_smoke_12/render_manifest.json \
  --device mps --engines piper --limit 1 \
  --output-dir models/qwen_omni/time_map/outputs/one_clip_piper_smoke
```

The one-clip run succeeded on local MPS, and no cloud instance was used. Its [manifest](outputs/one_clip_piper_smoke/extraction_manifest.json) has SHA-256 `f4d7b1a7ac7761bcfcae91e7ca7f0088ce3a4442b710396a7eede917daba5606`. The 7.047-second Piper clip produced 705 valid mel frames, four chunks (`200, 200, 200, 105` frames), 353 packed encoder rows, and 176 projected rows/placeholders. The last encoder row (352) has no projection partner and is not inserted into a text placeholder. Layers `0, 7, 15, 23, 31` were captured as finite `353 × 1280` matrices; the audio projection was captured as `176 × 2048`. All saved `.npy` hashes were verified. `cpu_tests.log` reports **6/6 passed**.

To extract all 12 fit items with both TTS engines later, omit `--limit` and `--engines`, choose a new empty output directory, and run after any concurrent MPS fit completes. No development or test record is accepted: the render manifest and every source record must say `split=fit`.

## Output contract

`extraction_manifest.json` gives source and checkpoint pins, source render hashes, selected item/engine records, fixed system prompt (`Answer the spoken request briefly.`), selected layers, device/dtype, and a row-timing warning. Each record points to `metadata.json` and `row_map.json` under a clip directory. `metadata.json` contains exact input token IDs and audio-placeholder positions, mel tensor hash, one `.npy` per layer and connector, each file SHA-256, source/saved dtypes, and shape. Load row `i` with `np.load(path, mmap_mode="r")[i]` and join to `row_map.json`'s `encoder_rows[i]` or `projected_rows[i]`. BF16 hook outputs are saved as float32 to allow NumPy access; this conversion adds no precision. Every clip is processed separately to avoid cross-sample packing ambiguity.

The fixed system prompt makes this an *audio-path extraction context*. It is not the same as the raw completion prompt in the semantic-bridge task. A future experiment should specify and verify task framing before comparing audio decoder states to typed decoder states; the audio tower and connector records here are observational.

See [ALIGNMENT_PLAN.md](ALIGNMENT_PLAN.md) before attaching lexical labels to any rows.
