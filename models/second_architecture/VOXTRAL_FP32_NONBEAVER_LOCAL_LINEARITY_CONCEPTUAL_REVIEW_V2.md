# Independent conceptual review: non-beaver FP32 local-linearity pilot V2

18 November 2026. I independently reviewed committed source-only protocol
`VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V2.md`, first added in
commit `b803c79`. Exact SHA-256:
`04f99648f5483fa7f32da80515c31e3f8472511c954cd95711d8eba09bb5b42d`.
The V1 protocol (`c3ae7038ff1c5a00645430db9cd5db67a0b746c4400f02ef7b853d6952966718`)
and its immutable HOLD review (`de236e5`) remain separate.

**Decision: GO for V2 protocol design only; no runner or model launch authorized.**

## Independent checks

- The FIT selection and mask files match the protocol's SHA-256 values
  `f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b`
  and `a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e`.
  I rehashed the four `color_05|fit_chosen` WAVs: Piper silver
  `fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9`,
  Piper teal `112ed36abf5fb2be1ce77dcfffa9d1bb6deba10b4ecb2d14fc5960b8106626e9`,
  Kokoro silver `bcad702fa8b55c453286cb0e83ee3336a4bb6ade5e5ac9d4a42dc82c1a97c26e`,
  Kokoro teal `5038dbe167ae2ba6c9317f74c17bd9bb8ef7d07bbdee9da49f6dc0731e119365`.
  The mask has 60/15 Piper and 73/18 Kokoro encoder/projector whole-word
  rows. The V1 review established matched-voice WAV equality outside the
  frozen word slot and 16-kHz mono signed PCM.
- V2 fixes the first HOLD item: each site has its own G1 clean VJP graph with
  the same source-leaf replacement and hook topology as its G1 zero/edit
  calls. It requires saved, bitwise clean-versus-every-zero parity for full
  returned source, projector, LM input, LM29 target, and first logits, plus
  full prepatch source. A failure invalidates that context/site gate without
  choosing a new baseline or relaxing tolerance.
- V2 fixes the second HOLD item: zero drift is the maximum float64 projected
  L2 over all `C(12,2)=66` unordered zero pairs per context/site/mode.
  Both cycles in both modes must be valid for a site-alpha cell. A cell is
  resolved only when both G1 cycle deltas exceed the fixed floor, and its
  primary error is the **maximum** of its two cycle errors. Invalid and
  unresolved cells remain in the 24-cell-per-site denominator. The site
  advancement rule requires all 24 valid, at least 18 resolved, and the
  fixed error thresholds.
- V2 fixes the third HOLD item: it defines a site-specific N0 clean reference
  with the same alpha-zero hook operation and saves its tensors. The donor
  and recipient source states used for delta are both N0. G1 and N0 paired
  changes use separate zero baselines.
- I independently executed the specified CPU probe recipe with PyTorch
  `2.15.0` and NumPy `2.3.5`: seed `3693039655029897049`, digest
  `10c9132454a01e034886a3e14bf9b9670e75bb3befd58300b9b45f8d9ba7751f`.
  QR orthogonality passed the specified `1e-12` absolute check.
- The corrected counts are four contexts × two sites × two modes × two
  cycles × six signed steps × paired zero/edit = **384 patch forwards**;
  four contexts × two sites × four probes = **32 VJPs**; plus eight G1
  clean graphs, eight N0 clean references, four donor captures and eight
  natural generations. The protocol caps each context process at 300
  seconds, 30 GiB MPS driver memory and 1 GiB combined local output. It
  explicitly requires a shape-based serialized-byte estimate before a
  launcher freeze and stops rather than shrinking a workload that exceeds
  these limits.
- Prior-result claims remain correct: the old FP32 numerical follow-up used
  gradient-enabled measured edits; the separate Stage-A handoff did not.
  The factorial result is a tiny one-case numerical contrast, not evidence
  of a specific kernel, audio mechanism, or security issue.

## What this GO does not establish

This is a prospective **design** approval. It does not check a runner, CPU
auditor, resource estimate, model outputs, or reproducibility on the second
FIT pair. The fresh v2 corpus and its locked test remain gated by independent
human listening and a later executable lens protocol. Any implementation
must be independently source-reviewed and externally frozen before the first
model process. No model weights were loaded for this review.
