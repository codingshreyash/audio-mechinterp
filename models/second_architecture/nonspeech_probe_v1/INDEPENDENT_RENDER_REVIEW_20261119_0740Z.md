# Independent render receipt: 20261119 0740Z

**Decision: GO for this exact no-model source render only.** I audited `runs/render_20261119_0740Z/` against the frozen prospective source at commit `bb0a5bd55e63c396c5408fd9714d1d528c9c9792`. No model weights were loaded and no model output was generated. This receipt does not lift the separate inference or intervention HOLD in `INDEPENDENT_SOURCE_REVIEW_V1.md`.

## Byte binding

| Item | SHA-256 |
| --- | --- |
| `runs/render_20261119_0740Z/manifest.json` | `a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90` |
| `runs/render_20261119_0740Z/started.json` | `23df9dc31730db837b4fefd957559b356471d120aaab47c79069a48b268985ed` |
| `render.py` | `8c20404965c30b11c347e562f1bb8b399e433c58e64d8674f62dab98dcf98cb1` |
| `spec.json` | `2dd69b687b00d90ed4e2d8bbc266039943498c31532634440f2d3bd0a0ca0951` |

The manifest and start record agree on these source/spec hashes, NumPy `2.3.5`, and `macOS-26.7.1-arm64-arm-64bit`. The complete manifest has exactly 20 unique declared case IDs. The directory contains exactly those 20 WAVs plus `started.json` and `manifest.json`, with no extra or missing file. Every WAV SHA-256 and PCM SHA-256 in the manifest matches the corresponding on-disk bytes. The manifest is the bound list of all 20 individual WAV and PCM digests.

## Independent signal audit

I opened every WAV with Python's standard `wave` reader, independently checked RIFF/WAVE format, mono PCM16, 16 kHz, exact 2- or 8-second frame count, no trailing audio frames, and manifest RMS/peak against decoded samples. Both silence files are exact zero. Both dither files contain only -1, 0, or +1 LSB and reproduce the frozen PCG64 seeds. All 16 other files reach their declared full-duration RMS within one PCM16 LSB; the largest absolute RMS error is `2.358858236364836e-06`. The highest absolute sample peak is `0.57952880859375`, below the frozen `0.8` ceiling.

For the signal families, the steady-tone spectral maximum is 700 Hz; the beep-off windows are exact zero and their on-window carrier repeats at 1 kHz; the clicks occur only within 128-sample windows every 6,400 samples with alternating polarity; and at least `0.9999981029591264` of band-noise PCM spectral energy lies in 300–3400 Hz. I independently reconstructed both band-noise durations from their frozen PCG64 seeds and FFT band filter and matched the quantized PCM arrays exactly at both levels. These are engineered non-speech waveforms by construction; this audit makes no claim about every listener's perception.

The reported first CLI attempt at 0738Z was rejected because its parent directory did not exist. I did not observe that command directly; I confirmed that no 0738Z render directory or partial receipt is present under `runs/`. The 0740Z directory is the first complete on-disk render reviewed here. The renderer's manifest order is not lexicographic, so the future inference runner must explicitly sort case IDs as the prospective protocol requires.
