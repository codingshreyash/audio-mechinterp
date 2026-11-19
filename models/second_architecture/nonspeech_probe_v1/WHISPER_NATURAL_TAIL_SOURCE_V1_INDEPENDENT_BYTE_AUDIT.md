# Independent byte audit: Whisper natural-tail source V1

19 November 2026. **PASS for the rendered source WAVs only; no candidate-model inference.** I audited `runs/natural_tail_waveform_v1/source_v1` against the frozen 20 original WAVs using a separate, read-only Python check. I did not call or import `natural_tail_waveform_v1.py` in the check, modify the renderer or its output, or load Whisper weights. The rendered manifest SHA-256 is `2ae99cfa279c3fd95e4552fd8f59ac0294afb445ddb92c2081b201661b9e0d11`.

The audit recomputed the source manifest SHA-256 (`a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90`) and each source WAV and PCM SHA-256. It parsed each rendered WAV's RIFF, `fmt `, and `data` fields directly, requiring a 44-byte PCM header, one channel, 16-bit integer samples, 16,000 samples per second, exact byte count, and no trailing data. It independently constructed the expected PCM as the complete original data bytes followed by either exact zero samples or seeded two-valued PCM16 samples. The four seed streams were recreated with NumPy 2.3.5 `Generator(PCG64(seed)).integers(0, 2, size=n, dtype=int8)`, using frozen seeds 9405, 9406, 9420, and 9421. No constructor function was reused.

| Check | Result |
| --- | --- |
| Manifest rows and directory inventory | 140 distinct rows; exactly 140 WAVs plus `manifest.json`, with no symlinks or extra files |
| Schedule labels | 20 rows for each of `base`, `z8000`, `n8000_s0`, `n8000_s1`, `z32000`, `n32000_s0`, `n32000_s1`; source order and condition order match the frozen design |
| Source identity | All 20 original WAV/PCM hashes match; all 20 rendered `base` WAVs are byte-identical to their originals |
| Source prefix and tails | All 140 rendered PCM byte strings equal an independent construction; all row WAV, PCM, source-prefix, and source-WAV hashes match; every zero sample is zero and every noise sample is exactly +32 or -32 PCM16 units |
| Format and length | All 140 WAV headers are exact PCM16 mono 16 kHz; all retain their full original 2- or 8-second source and have only the declared 0-, 0.5-, or 2-second tail; no file reaches Whisper's 30-second limit |
| Zero-tail null | For all 20 sources at both zero-tail lengths, the 30-second zero-padded PCM bytes equal those of `base`: **40/40 identities** |

The render manifest names the expected builder hash `a3166ff18aa79dcf0e7edf75dd277854cd0181b99835ffb7667013e2762317d5`. Its status is `complete_source_only_no_model_inference`. The directory occupied about 26.4 MiB at review. It was untracked in Git at review time; the prospective protocol requires committing or otherwise freezing the render manifest and source bytes before a candidate run. Preserve this manifest hash when doing so.

This audit establishes the edited audio bytes and source inventory. It does **not** establish processor feature parity, generated-output changes, padding mediation, perceptual invisibility, or a security vulnerability. The [design review](WHISPER_NATURAL_TAIL_INDEPENDENT_DESIGN_REVIEW_V1.md) still holds model-runner work pending the explicit English-only generation-call correction; inference also needs its separate executable GO.
