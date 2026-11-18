# Frozen English audio renderer

This renderer reads the existing semantic item JSONL unchanged and synthesizes each selected `audio_input.script` twice, with two independent neural TTS families on CPU:

| Engine | Family and voice | Pinned source |
|---|---|---|
| [Piper](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md) | VITS, `en_US-lessac-medium`, native 22,050 Hz | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices/tree/0ca208afc2a2ee102b1e961e72e0a7e4907be3d0/en/en_US/lessac/medium) at `0ca208afc2a2ee102b1e961e72e0a7e4907be3d0` |
| [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/play.py) | Kokoro StyleTTS2 + iSTFTNet, `af_sarah`, native 24,000 Hz | [`speaches-ai/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/speaches-ai/Kokoro-82M-v1.0-ONNX/tree/dc196c76d64fed9203906231372bcb98135815df) at `dc196c76d64fed9203906231372bcb98135815df` |

The engines share some English phonemization dependencies, so they are independent **synthesis model families**, not independent grapheme-to-phoneme systems. [Model and voice file hashes and fixed synthesis parameters](pins.json) are checked before rendering. The ONNX sessions are required to use `CPUExecutionProvider`; the script makes no MPS or paid-cloud call.

## Install and run

```bash
cd /Users/shrranjan/audio-mechinterp
/opt/homebrew/bin/python3.12 -m venv eval/tts/.venv
eval/tts/.venv/bin/python -m pip install -r eval/tts/requirements.freeze.txt
eval/tts/.venv/bin/python eval/tts/render.py eval/semantic_bridge/manifest.jsonl \
  --split fit --limit 2 --allow-download \
  --output-dir eval/tts/outputs/semantic_fit_smoke_2_v2
```

`--allow-download` fetches only the four files named and hashed in `pins.json` from immutable revisions. Omit it once cached. The output directory must be empty. To render the complete fit split, omit `--limit` and choose a new output directory. The renderer never edits, translates, selects by meaning, or rewrites the frozen items. The `--split` and `--limit` options select an existing split and its first N rows for bounded runs; their values are recorded in the output manifest.

Input can be the frozen semantic JSONL with `id`, `split`, `audio_input.script`, and `typed_input.prompt`. A simple JSONL with `item_id` and `text` is accepted for renderer smoke tests. Every source record is copied verbatim into the output manifest. The exact input file bytes, source line bytes, spoken text, rendered file bytes, and decoded PCM are separately SHA-256 hashed.

## Output contract

`<output-dir>/render_manifest.json` contains the input manifest hash, selected rows, full `pins.json`, package versions, ONNX providers, and one `records[]` entry per rendered item. Each entry has its original `source_item`, exact spoken-text hash, and `outputs.piper` and `outputs.kokoro`. Each output contains `raw` and `processed` paths relative to the output directory, WAV metadata, file and decoded-PCM hashes, duration, peak, and RMS. Processed audio is 16 kHz mono PCM16. The resampler is `scipy.signal.resample_poly` with exact reduced integer ratio, Kaiser beta 8.6, constant zero padding; there is no loudness normalization or silence trim. The processed record also gives any pre-quantization clipping count.

The current fit-only smoke is [`outputs/semantic_fit_smoke_2_v2/render_manifest.json`](outputs/semantic_fit_smoke_2_v2/render_manifest.json): two frozen fit items, four raw WAVs, four processed WAVs. A separate process rendered the same items into `outputs/semantic_fit_smoke_2_v2_repeat`; **all eight WAV file SHA-256 hashes matched**. The first Kokoro implementation used `soundfile` to write float WAV, which inserted a timestamped `PEAK` header chunk despite identical decoded PCM. The current renderer writes a minimal deterministic IEEE-float WAV header; the earlier smoke outputs remain local but should not be used.

The 12-item fit-only subset at [`../semantic_bridge/smoke.jsonl`](../semantic_bridge/smoke.jsonl) has also been rendered with both engines. Its [render manifest](outputs/semantic_fit_smoke_12/render_manifest.json) has SHA-256 `aabfe2060d2f295b80faca77fcaddbaa088913f60e2adb59bf01f384983c20e0`; its input JSONL SHA-256 is `12f4f6ab4d80542e075422ed17a5c3edbc2b7fb52d13852e96374be76fdb4c96`. All 12 original item objects and IDs were preserved, all 48 raw/processed WAV file hashes verified, and no sample clipped during resampling. Eight of the 12 typed/audio strings are byte-identical; the other four differ only by a trailing typed space. This is a rendering check, not intelligibility validation.

## Interpretation limits

- The frozen text prompts include 27 trailing-space differences between `typed_input.prompt` and `audio_input.script`. All differences disappear after removing trailing whitespace. Both original strings remain in `source_item`; `typed_audio_bytes_equal` flags the distinction. A space at a text completion boundary has no spoken equivalent, and some prompts are unfinished fragments. The paired task must report this limitation, especially if the model continues a sentence.
- A synthetic render is an input artifact, not a verified transcript. Before scientific scoring, audit intelligibility with an independent ASR and/or human listeners and retain the exact WAVs. Do not silently regenerate a difficult clip or alter the prompt text.
- The two voices differ as well as the model families. Any engine effect also includes voice, duration, prosody, and phonemization differences. Attribution to a single acoustic factor needs further controls.
