# Non-speech descriptive runner V1: source review handoff

19 November 2026. **HOLD: no candidate-model inference has run.** This adds a bounded descriptive runner for the reviewed 20-waveform source. It does not execute a causal intervention or make a security claim. The 20 cases and two exact-silence repeats are fixed before any model output is seen. The source renderer remains at commit `bb0a5bd`; the local render is audited separately.

## Candidate inputs and checkpoints

The [source spec](spec.json) and [manifest](runs/render_20261119_0740Z/manifest.json) identify every WAV by bytes. The worker independently checks the spec, all 20 WAV and PCM SHA-256s, and the sorted case set. [Checkpoint pins](descriptive_pins_v1.json) identify three local snapshots and every file the loader needs. Multilingual `openai/whisper-tiny` uses revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, 151,061,672-byte safetensors SHA-256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`. English-only Tiny and Wav2Vec2 have separately pinned revisions and local weight hashes. The same pin file fixes NumPy 2.3.5, Torch 2.15.0, Transformers 5.20.2, and relevant installed source bytes.

The **source-only preflight** hashes these local assets without deserializing weights. It produces a receipt with snapshot paths, revisions, weight byte counts, all model/config/tokenizer hashes, installed package versions and source hashes, source file hashes, case IDs, and limits:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.descriptive_parent_v1 \
  --preflight-only --receipt /absolute/new/preflight.json
```

No live attempt is allowed until a reviewer independently commits a `nonspeech_descriptive_source_review_v1` JSON with `status=GO`, the exact receipt SHA-256, manifest SHA-256, and `source_identity` object. The worker and parent both recompute the receipt and require the review commit to follow the source commit. No such GO exists at this handoff.

## Runner evidence and bounds

The [worker](descriptive_worker_v1.py) runs one model in one fresh process, one-sample batches, CPU FP32, at most four CPU threads. The [parent](descriptive_parent_v1.py) uses a new attempt directory, a 1,800-second wall cap, 4 GiB sampled process-tree RSS cap, 128 MiB output cap, and an existing pinned Mac AC/lid/power-log gate before and after. It kills the process group on a cap breach. A process killed inside a forward cannot write all remaining case rows; the parent retains the partial attempt, and the [auditor](descriptive_audit_v1.py) rejects it. A worker that catches a model or case failure writes all 22 planned rows with explicit failure/not-run statuses and stops. No retry is automatic. Parent monitoring samples RSS every 250 ms; the worker also checks its own peak RSS between cases. Neither can bound a sub-interval peak inside a single forward.

The worker records all generated Whisper token IDs, explicit prefix IDs, per-step scores/hashes, actual generated step count, selected language detection ID and **separate detection-pass language logits**, full decoded text, lexical text, Unicode script, EOS/censor status, and per-case elapsed time. Pad and EOS are the same token ID in these Whisper configs; the step count comes from the score tuple, not a trailing token. CTC rows record every frame ID, blank/collapsed IDs, logits hash, text, and timing. An unexpected short sequence without EOS is marked `incomplete_unknown` and retained. Both Whisper checkpoints use transcribe, no forced language, greedy 128-token decoding; only multilingual Whisper is expected to call `detect_language` once. The English-only and Wav2Vec2 rows cannot test a Cyrillic/Devanagari-script outcome.

Read-only hooks record per-time-row norms, full-tensor hashes, shapes and counts at the declared model stages, separately for multilingual detection and transcription. Once the same-duration silence row is observed, the worker computes per-time-row differences for **all 18** non-silence cases. It retains earlier tensors in worker memory until that silence row and saves only hashes and curves to stay within 128 MiB. The saved-data auditor can validate coverage and consistency but **cannot independently recompute activation hashes or model forwards from the saved curves**. A stronger mechanism claim would require a separate raw-capture design. This descriptive stage records no intervention effects.

## Current verification and next gate

Only fake-model and no-weight tests have run. They exercise three local checkpoint hashes, package/source pins, exact source case coverage, detection/prefix versus generated-step accounting, EOS and length censor, CTC collapse, all-case failure preservation, complete fake-case/difference coverage, and process-group wall timeout. Source review should inspect the installed Whisper `generate` return contract, hook count/topology for both passes, Wav2Vec2 layer return types, and whether the saved evidence is adequate before issuing GO. The run decision remains **HOLD**.
