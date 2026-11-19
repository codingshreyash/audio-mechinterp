# VoiceChat acoustic action corpus builder (source only)

This is the two-stage input builder for [V1](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V1.md) plus the [V2 endpoint/admission amendment](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V2.md). It never loads VoiceChat, runs ASR, assigns human quality, grants cancellation authority, or scores a model. The 12 source recordings and all noise variants are new. They do not reuse the earlier VoiceChat calendar WAVs.

## Stage 1: render the 12 exact clean sources

Use the pinned `eval/tts/.venv` Python, Piper Amy 1.8.0, Kokoro ONNX 0.6.1 `af_bella`, and macOS Samantha. The program verifies package/model/voice and OS bytes. It converts each separately rendered utterance to 16-kHz mono PCM16 and applies the same frozen level rule to every source: gain is the lesser of `0.1 / resampled RMS` and `0.85 / resampled peak`. It then appends 80,000 exact zero frames. A clipped conversion fails and preserves the attempted source; no replacement is selected. The failed first TTS-only attempt, before this rule, remains in `runs/sources_20261119_0129`.

```sh
eval/tts/.venv/bin/python -m models.second_architecture.voicechat_acoustic_action_v1.corpus \
  render-sources --output-dir /absolute/new/source-run-id
```

`source_manifest.json` binds the script, engine, raw, converted and clean WAV hashes, package/OS identity, source code and protocol hashes. `annotation_template.json` contains **blank** interval fields. If any step fails, the exclusive run directory retains `failure.json` and completed files. Do not reuse an attempted output directory.

## Independent timing annotation

Give two independent, English-fluent annotators the 12 `*.clean.wav` files and the target decision phrase in `annotation_template.json`. They must listen without model traces or candidate answers. Each records two `[start,end)` **sample-frame** intervals for every source:

- `decision`: the listed phrase, including the authorization wording.
- `displaced`: an equally wide speech interval elsewhere in the same utterance, separated from `decision` by at least 2,560 frames (160 ms).

Only content frames before the five-second zero tail are valid. Each annotator enters a distinct `reviewer_id` and their own proposed `decision` and `displaced` arrays. The coordinator then enters the final intervals in `adjudication`, with an `adjudication.by` ID and nonempty `note`. If the proposals differ, a third person must adjudicate. Preserve both initial proposals. This JSON file is a provenance record, but the program cannot prove identities or that humans listened; the coordinator must attest independence. **Do not auto-align, infer, or fill intervals from a transcript or model output.**

Make a copy of the blank template; do not edit the renderer's original template. The exact `source_manifest_sha256`, `clean_wav_sha256`, `clean_wav_path`, `decision_phrase` and `content_frames` fields must stay intact. If a source lacks a valid displaced interval, this version stops before creating a completed corpus. A later revised design needs its own review.

## Stage 2: make the 48 fixed acoustic arms

After the independent annotations are complete, run:

```sh
eval/tts/.venv/bin/python -m models.second_architecture.voicechat_acoustic_action_v1.corpus \
  make-variants --source-dir /absolute/source-run-id \
  --annotations /absolute/completed-intervals.json \
  --output-dir /absolute/new/variant-run-id
```

For each source, this writes clean, full-content noise, decision-window noise, and displaced-window noise. All three noisy arms use the same source-indexed Gaussian vector, globally standardized over the content interval and scaled to nominal 20-dB SNR. The terminal zero tail stays unchanged; the builder rejects any PCM16 clipping. `variant_manifest.json` records every file/PCM hash, noise seed, exact mask and realized SNR. A missing or invalid annotation leaves `failure.json` and no completed manifest. All 48 WAVs still require blinded two-listener wording/intelligibility review before any candidate VoiceChat inference.

Fake tests use a tone renderer in temporary directories and mark the source `synthetic_test_only`. The public stage-2 command rejects that source identity. Run them with:

```sh
eval/tts/.venv/bin/python -m unittest \
  models.second_architecture.voicechat_acoustic_action_v1.test_corpus -v
```

The scripts and engine plan are in [`spec.json`](spec.json); its byte hash is pinned in [`corpus.py`](corpus.py). Both stage manifests explicitly set `candidate_inference_authorized=false`. Independent source review and a separate human-QA/audit chain remain required before these bytes are experimental inputs.
