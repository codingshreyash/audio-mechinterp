# Independent rendered-corpus audit

**Verdict:** No rendering or allocation blocker found. This was a CPU-only inspection of saved files; I did not run Qwen, ASR, or any answer scorer.

I independently checked `SPEC_V1.json`, `rendered/manifest.json`, `rendered/BINDING.json`, the referenced source files, and every rendered WAV. The manifest and binding hashes agree with the files on disk. Source hashes agree for `render.py`, `make_spec.py`, the held-out assembly helper, the underlying TTS renderer, and the pinned TTS plan.

| Check | Observed result |
| --- | --- |
| Fragment files | 216 raw plus 216 processed; all 432 WAV SHA-256 and decoded float32 PCM SHA-256 values match the manifest. Sample rates and frame counts match. |
| Final clips | 384 WAVs; all file SHA-256 and decoded int16 PCM SHA-256 values match. Every clip has 16 kHz audio and the recorded frame count. |
| Paired audio | All 192 pairs have equal int16 PCM before and after the recorded word slot, and differing PCM within it. Slot bounds are valid for both clips. |
| Allocation | Exact frozen set of 48 word pairs × two split-specific templates × two voices. Fit pool has 24 word pairs, development 12, locked test 12. Each category has 6/3/3 pairs across those splits; the 12 map-fit pairs have 3 per category. |
| Separation | 96 unique words across pairs and splits; no historical excluded word appears. Six template IDs are split-specific. All 816 artifact paths are unique. |
| Text metadata | All 216 fragment IDs and texts and all 384 assembled spoken-script strings match the frozen spec. Voice labels match Piper Amy and Kokoro Bella as specified. |

These checks establish that the saved stimulus bytes and allocation match the frozen design. They do not establish what a human listener, ASR system, or candidate model hears, or whether any candidate answer is correct. Those are separate model and perceptual gates.
