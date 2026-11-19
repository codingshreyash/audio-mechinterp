# Independent rereview: word-disjoint renderer source V2

19 November 2026. I independently reviewed source commit `8c65d8ebdc18fde10e21774721bb65f4d790f4b1` against the [V1 HOLD](RENDERER_SOURCE_INDEPENDENT_REVIEW_HOLD_V1.md) and the independently accepted [V2 prospective plan](../WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2_INDEPENDENT_REVIEW.md). **Decision: GO for the frozen source-level 32-WAV render and technical audit, subject to a separately committed freeze receipt; no GO for candidate-model inference.** I did not render the corpus, listen to clips, open locked-test WAVs, or run a candidate model.

The forged positive freeze tested in the HOLD no longer passes: `freeze_receipt()` requires a separately committed GO review containing this exact JSON binding, verifies the reviewed source blobs against the earlier source commit, checks current source and receipt paths are committed and clean, and requires the external freeze receipt to bind the review's hash and commit. The listening queue now reruns the saved-fragment and final-WAV technical audit and compares the complete report with `TECHNICAL_BINDING.json` before copying clips. The six no-model tests passed, including forged-freeze and forged-technical-binding rejection. `render --validate-only` passed with 16 pair/voice cells and 32 WAVs planned; all six reviewed source blobs matched the specified commit byte for byte.

The renderer retains the fixed 8 pairs and 16 distinct words with metadata exclusions, Piper Amy and Samantha pins, mono 16-kHz PCM16 conversion, a common outside-word PCM stream per pair, raw/converted/final hashes, and opaque listener filenames. The technical audit verifies exact reconstructed PCM; it does not establish that a human hears the intended sentence. The 32/32 independent blinded human listening gate, candidate-runner review and no-weight freeze are separate prerequisites for model inference.

```json
{
  "schema": "word_disjoint_renderer_source_go_v2",
  "reviewed_source_commit": "8c65d8ebdc18fde10e21774721bb65f4d790f4b1",
  "source_sha256": {
    "render.py": "f6524f30f2655c9f62fdc4898be5e480c20c5e0ba84c5e19427ad437656d044c",
    "audit.py": "6411c2961ba31700f3c3e099124644c80c38e52c925c4ef02cb9d01173cd831f",
    "queue.py": "1744be6270352f86fb1990f9ef7223734514dd05a9ec14b717876c8b28d15e3c",
    "test_fixture_source.py": "1228f0c60b6ccb1154739eab376546ce74a50564d05c9e29357880d917f578e2",
    "__init__.py": "46050500ba491a71951c429e0bd5e69eadb4136118974c5b8475fc04b43aee6c",
    "README.md": "4025fdaf0fd0ab9107eaaa17e0cfdc5b8253b5b53ed5a5a7996f9354971e6a24"
  },
  "plan_sha256": "a2065c2bf74b9acb56c511f89142e86aa653e3b2a910215ab42a1c478034d9e6",
  "plan_review_sha256": "2d96fc8104a81bcd3dda4dfdc9f84e5a773eca7481415bfdfd6492f2c747d4cd",
  "source_tts_plan_sha256": "13cea31e2e4843d70a6ab26b060812f43691147435c1e10ea9ebaaa0b24efc78",
  "source_tts_helper_sha256": "bbbd7931dd93121d32b32a9c7898d0a14a3e884f7a421a01d4ecdfd3bd3d063c",
  "source_tts_renderer_sha256": "c3f4f9f71d9c7db8db3af414dbde72ebe7e0a9bc1a0c354ed0d86b8dbe3d9783"
}
```
