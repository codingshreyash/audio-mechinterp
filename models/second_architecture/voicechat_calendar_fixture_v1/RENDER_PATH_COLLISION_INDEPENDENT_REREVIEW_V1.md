# VoiceChat synthetic render path collision: independent rereview

Verdict: **source-level go for one new, exclusive output directory**. This review authorizes no model inference and does not turn the failed render into a complete fixture.

The anchored attempt `b1fc8ca` stopped on the first Piper case: its source and final were both `A__piper_amy_en_us.wav`, and the padding helper correctly raised `FileExistsError`. Its `error.json` says `incomplete_synthetic_render`; it has no complete manifest.

In patch `551d5be`, `output_paths()` assigns distinct `.source.aiff` or `.source.wav`, `.converted.wav`, and `.final.wav` paths. I independently inspected the 6 frozen case IDs × 2 voices × 3 stages: all **36 paths are unique**. The source and final for the previously failing Piper case now differ. The main entry point creates the output directory with `exist_ok=False`, so the failed attempt cannot be resumed or overwritten by a normal run. Its metadata and manifest filenames are also distinct from the 36 planned audio files. The padding helper still refuses an existing final path.

Verification on the patched tree: `checked_inputs()` passed through the new synthetic unit test, which checks the frozen dependencies and all 36 paths; the test passed (1/1). `py_compile` passed for the renderer and test. The reviewed renderer SHA-256 is `149a568fb06d73a57eddbbd64a7a9d48b99d4d46387762e04c4633bb6fed38c2`. No TTS, audio render, ASR, or VoiceChat inference was run for this review.

The remaining completion check is a fresh, exclusive render followed by a 12-row manifest and per-file hash audit. A partial render remains incomplete even if some audio files exist.
