# Model-blind timing bundle source handoff

19 November 2026. **Source-only; human marks absent.** This bundle is a convenience for collecting the two independent interval records required by the [VoiceChat acoustic action protocol](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V2.md). It contains no model output, inferred alignment, completed annotation, noise variant, or candidate inference result.

The standalone bundle is `/Users/shrranjan/VoiceChat-Timing-20261119`. It has exactly 12 audited clean WAVs plus `QUEUE.json`, `TIMING_FORM.html`, and `README.md` (15 files). `QUEUE.json` SHA-256 is `7bae82a5e42c7fcb7db99829c753adf19c47bc979d5e1daacab68a3935e01260`. The builder pins the independently audited source manifest SHA-256 `3d7714430bfca4c5888b7142be0d6ca575841f3cad0e74163650f02e0d8b1ca5` and original timing-template SHA-256 `ebac0ab9ab1804eb7d225313f47ad0fd9e417b1ab6a80316a8a6dd13fefe3a86`, then rechecks the stage-1 production admission and every clean WAV hash before copying it.

The correction after the [independent V1 HOLD](TIMING_BUNDLE_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md) makes assembly verify the exact **15-file regular-file inventory** and reconstruct the precise rendered `QUEUE.json`, `TIMING_FORM.html`, and `README.md` bytes from the pinned form template and audited queue. It rejects a changed form, extra file, link, or directory before accepting reviewer exports. The existing bundle's 15 files and three generated text files matched the corrected reconstruction byte for byte. This is a source-integrity check, not proof of what a reviewer saw or heard.

Each reviewer opens the HTML form in a separate browser profile, uses a distinct ID, listens to each clip, and marks `[start,end)` frame indices for the decision phrase and an equally wide speech interval at least 2,560 frames away. The player buttons copy the **human-selected** current time to a frame input; no waveform analysis or alignment runs. The form locks each submitted first impression in that profile and exports one JSON after 12 clips. An unmarkable clip requires a note and stops stage-2 assembly. The coordinator retains both raw exports. A third distinct reviewer uses the same form if any interval pair differs. The assembler chooses the third's independently marked pair for those disagreements and writes the existing `voicechat_acoustic_intervals_v1` schema; it records hashes of all reviewer exports. Do not edit the raw exports or fill missing marks from scripts or model output.

Source-only verification: seven fake-export `unittest` cases passed, including exact stage-2 annotation-validator acceptance, differing-window third-review requirement, tampered WAV and queue rejection, altered HTML rejection, and extra-file rejection. `node --check` passed on the inline JavaScript. In a separate `agent-browser` smoke profile, the local page opened, displayed clip 1/12 and all four frame controls, and loaded the exact first WAV (`12.325938` s); the test profile was cleared and closed without submitting a review. These tests do not establish human identity or that a person listened. A coordinator must verify those facts and independently review completed exports before allowing stage 2.

To rebuild the bundle without model work:

```sh
eval/tts/.venv/bin/python -m models.second_architecture.voicechat_acoustic_action_v1.timing_bundle build --output-dir /absolute/new/bundle
```

After two real exports, with a third if needed, assemble to a new path:

```sh
eval/tts/.venv/bin/python -m models.second_architecture.voicechat_acoustic_action_v1.timing_bundle assemble --bundle /absolute/bundle --first /absolute/reviewer-a.json --second /absolute/reviewer-b.json --third /absolute/reviewer-c.json --output /absolute/new/intervals.json
```

Omit `--third` if the two initial records agree on every clip. The local code can check IDs and bytes; it cannot certify that IDs correspond to different humans. The assembled intervals are still not a 48-WAV listening QA receipt or permission for VoiceChat inference.
