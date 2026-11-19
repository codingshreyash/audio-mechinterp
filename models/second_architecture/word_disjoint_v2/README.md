# Word-disjoint English audio fixtures, source stage

The [V2 prospective plan](../WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md) fixes eight new English word pairs, Piper Amy and macOS Samantha, and a single script: `I said {word} and then stopped.` The renderer creates 32 synthetic WAVs in 16 paired cells. The technical auditor checks that each pair has exactly equal 16-kHz PCM outside a saved word slot. The queue builder copies the verified WAVs under opaque names for two blinded listeners. None of these programs loads ASR or assistant weights.

Source-only checks:

```sh
python3 -m unittest models.second_architecture.word_disjoint_v2.test_fixture_source -v
python3 -m models.second_architecture.word_disjoint_v2.render --validate-only
```

The full render is disabled until an independent review accepts the V2 plan, the renderer, auditor, queue builder and tests. An independently created `RENDER_FREEZE_V2.json` must then bind the exact SHA-256 of the V2 plan, `render.py`, `audit.py`, `queue.py`, the source TTS plan and its helper, with status `independently_reviewed_renderer_frozen`. The renderer checks that receipt before creating any audio and refuses an existing output directory. The `--render` path does not download Piper weights.

After that source gate, the intended sequence is:

```sh
python3 -m models.second_architecture.word_disjoint_v2.render --render
python3 -m models.second_architecture.word_disjoint_v2.audit
python3 -m models.second_architecture.word_disjoint_v2.queue --write
```

The listener-facing directory is `rendered_v2/listening/` and contains only opaque WAV names plus `QUEUE.json`. Keep `rendered_v2/QA_PRIVATE_CASES.json` away from reviewers; it maps IDs to words and original paths. Every listener records a verbatim full transcript, the middle word, a clear/unclear judgment and confidence for every ID. The queue is **not** a listening receipt, and the technical binding explicitly leaves candidate-model inference unauthorized. The full 32/32 independently audited human QA gate and later runner review/freeze are separate requirements.

macOS `say` and `afconvert` binary hashes, OS build, conversion command, Piper package and checkpoint/config hashes, raw and converted fragment hashes, exact little-endian PCM hashes, and final WAV hashes are saved in the render manifest. Because Samantha can change with OS updates, the frozen WAV bytes, rather than the voice name alone, identify the test stimuli.
