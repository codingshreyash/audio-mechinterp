# Independent review: render path fix `551d5be`

18 November 2026. Source and local tests only; no TTS render, ASR, VoiceChat inference, or cloud use.

The preserved [first-attempt error](rendered_v1/20261118T1143Z/error.json) in `b1fc8ca` is `FileExistsError` at `A__piper_amy_en_us.wav`, after `rendered: 6` Apple clips and while Piper case A was active. In the old code, that path was **both** Piper's raw `source` and padded `final`; `prepare(converted, final)` refused to overwrite the raw source. The failed directory has no complete manifest and must remain excluded from a subsequent fixture run.

`output_paths` in `551d5be` names each recording with a case/voice stem and three explicit stages. This is the full 12-recording, 36-path matrix (`{stem}` is the exact value in the first column):

| Stem | Raw source | Converted | Final model input |
| --- | --- | --- | --- |
| `A__apple_samantha_en_us` | `A__apple_samantha_en_us.source.aiff` | `A__apple_samantha_en_us.converted.wav` | `A__apple_samantha_en_us.final.wav` |
| `B__apple_samantha_en_us` | `B__apple_samantha_en_us.source.aiff` | `B__apple_samantha_en_us.converted.wav` | `B__apple_samantha_en_us.final.wav` |
| `C__apple_samantha_en_us` | `C__apple_samantha_en_us.source.aiff` | `C__apple_samantha_en_us.converted.wav` | `C__apple_samantha_en_us.final.wav` |
| `D__apple_samantha_en_us` | `D__apple_samantha_en_us.source.aiff` | `D__apple_samantha_en_us.converted.wav` | `D__apple_samantha_en_us.final.wav` |
| `A_literal__apple_samantha_en_us` | `A_literal__apple_samantha_en_us.source.aiff` | `A_literal__apple_samantha_en_us.converted.wav` | `A_literal__apple_samantha_en_us.final.wav` |
| `B_literal__apple_samantha_en_us` | `B_literal__apple_samantha_en_us.source.aiff` | `B_literal__apple_samantha_en_us.converted.wav` | `B_literal__apple_samantha_en_us.final.wav` |
| `A__piper_amy_en_us` | `A__piper_amy_en_us.source.wav` | `A__piper_amy_en_us.converted.wav` | `A__piper_amy_en_us.final.wav` |
| `B__piper_amy_en_us` | `B__piper_amy_en_us.source.wav` | `B__piper_amy_en_us.converted.wav` | `B__piper_amy_en_us.final.wav` |
| `C__piper_amy_en_us` | `C__piper_amy_en_us.source.wav` | `C__piper_amy_en_us.converted.wav` | `C__piper_amy_en_us.final.wav` |
| `D__piper_amy_en_us` | `D__piper_amy_en_us.source.wav` | `D__piper_amy_en_us.converted.wav` | `D__piper_amy_en_us.final.wav` |
| `A_literal__piper_amy_en_us` | `A_literal__piper_amy_en_us.source.wav` | `A_literal__piper_amy_en_us.converted.wav` | `A_literal__piper_amy_en_us.final.wav` |
| `B_literal__piper_amy_en_us` | `B_literal__piper_amy_en_us.source.wav` | `B_literal__piper_amy_en_us.converted.wav` | `B_literal__piper_amy_en_us.final.wav` |

All 36 full paths are distinct within the output directory. The explicit `.source`, `.converted`, and `.final` markers remove the prior raw/final ambiguity, and `output.mkdir(..., exist_ok=False)` prevents reusing the failed attempt. `prepare` still rejects an existing destination. No remaining path collision or overwrite is apparent in the frozen 12-case matrix. A downstream script that recursively globs all WAVs under `rendered_v1` could nevertheless mix the incomplete first attempt with a new one; select only `final_path` entries from a *complete* `render_manifest.json`, and keep the attempt directory in every record.

Local verification: `/Users/shrranjan/Garak/garak/.venv/bin/python -m unittest discover -s models/second_architecture/voicechat_calendar_fixture_v1 -p 'test_*.py' -v` passed **9/9**, including `checked_inputs()` and the 36-path uniqueness test. An independent enumeration reproduced all 12 rows above and asserted 36 unique paths. System `/usr/bin/python3` cannot import `numpy` here, so that interpreter cannot run the renderer test; the pinned project venv can. The fix only clears this source-level path blocker. Fresh audio still requires render, independent ASR, and listening QA before inference.
