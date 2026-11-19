# Word-disjoint English audio fixtures, source stage

The [V2 prospective plan](../WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md) fixes eight new English word pairs, Piper Amy and macOS Samantha, and a single script: `I said {word} and then stopped.` The renderer creates 32 synthetic WAVs in 16 paired cells. The technical auditor checks that each pair has exactly equal 16-kHz PCM outside a saved word slot. The queue builder copies the verified WAVs under opaque names for two blinded listeners. None of these programs loads ASR or assistant weights.

Source-only checks:

```sh
python3 -m unittest models.second_architecture.word_disjoint_v2.test_fixture_source -v
python3 -m models.second_architecture.word_disjoint_v2.render --validate-only
```

The full render is disabled until an independent review accepts the V2 plan and the exact renderer, auditor, queue builder, tests, package initializer, and this README. The independent reviewer must write and separately commit `RENDERER_SOURCE_INDEPENDENT_REREVIEW_GO_V2.md` after the reviewed source commit. It must contain `**Decision: GO` and exactly one fenced `json` block with this binding:

```json
{
  "schema": "word_disjoint_renderer_source_go_v2",
  "reviewed_source_commit": "FULL_40_CHARACTER_SOURCE_COMMIT",
  "source_sha256": {
    "render.py": "SHA256", "audit.py": "SHA256", "queue.py": "SHA256",
    "test_fixture_source.py": "SHA256", "__init__.py": "SHA256",
    "README.md": "SHA256"
  },
  "plan_sha256": "SHA256", "plan_review_sha256": "SHA256",
  "source_tts_plan_sha256": "SHA256", "source_tts_helper_sha256": "SHA256",
  "source_tts_renderer_sha256": "SHA256"
}
```

An externally written, separately committed `RENDER_FREEZE_V2.json` must have status `independently_reviewed_renderer_frozen` and bind the GO review file SHA-256 and commit, its `reviewed_source_commit`, the same `source_sha256` object, the plan and plan-review hashes, and the three TTS source hashes. `render.freeze_receipt()` verifies current files are committed and clean, the source blobs in the reviewed commit match, the GO review is a later separate commit, and the receipt matches all hashes. The renderer checks this before creating output and refuses an existing directory. The `--render` path does not download Piper weights.

After that source gate, the intended sequence is:

```sh
python3 -m models.second_architecture.word_disjoint_v2.render --render
python3 -m models.second_architecture.word_disjoint_v2.audit
python3 -m models.second_architecture.word_disjoint_v2.queue --write
```

The listener-facing directory is `rendered_v2/listening/` and contains only opaque WAV names plus `QUEUE.json`. Keep `rendered_v2/QA_PRIVATE_CASES.json` away from reviewers; it maps IDs to words and original paths. Before making that directory, the queue rechecks the source and manifest binding, rereads all 36 saved fragments and 32 final WAVs, independently reconstructs every paired PCM stream, and requires the saved technical report to match the recalculated report exactly. Every listener records a verbatim full transcript, the middle word, a clear/unclear judgment and confidence for every ID. The queue is **not** a listening receipt, and the technical binding explicitly leaves candidate-model inference unauthorized. The full 32/32 independently audited human QA gate and later runner review/freeze are separate requirements.

macOS `say` and `afconvert` binary hashes, OS build, conversion command, Piper package and checkpoint/config hashes, raw and converted fragment hashes, exact little-endian PCM hashes, and final WAV hashes are saved in the render manifest. Because Samantha can change with OS updates, the frozen WAV bytes, rather than the voice name alone, identify the test stimuli.
