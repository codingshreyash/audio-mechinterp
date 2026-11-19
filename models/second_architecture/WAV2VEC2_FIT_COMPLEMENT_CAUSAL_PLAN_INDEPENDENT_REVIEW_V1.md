# Independent plan review: Wav2Vec2 FIT outside-word complement v1

19 November 2026. **Decision: HOLD for freezing or running V1.** I reviewed the committed [prospective plan](WAV2VEC2_FIT_COMPLEMENT_CAUSAL_PROSPECTIVE_V1.md), pinned Wav2Vec2 baseline and prior causal records/audits, sidecar manifests, source topology and corpus manifest. I loaded no model weights and ran no new model forward.

## Correction required before inference

The noncontiguous complement is defined as `[0,W_start) ∪ [W_end,T)`, but the seeded random-control instruction calls `standard_normal(C_shape)` without defining `C_shape` or the order in which those two row intervals receive draws. A single `[1,T-|W|,768]` draw scattered in increasing time differs bitwise from two separate draws for the left and right intervals, even with the same seed. Both are plausible readings of V1. This matters because the plan requires exact reproducibility of 64 seeded random-C controls and bitwise auditor reconstruction. Fix the single draw shape, concatenation order and scatter rule in a prospectively committed V2 amendment, and re-review it before runner preflight or model inference. The letter **C** also names preprocessing Mode C; using `C_off` for the complement in code would reduce ambiguity, though the current prose distinguishes the two.

## Checks that passed

- The pinned corpus manifest, baseline records/audit/sidecar manifest, and prior causal records/audit/sidecar manifest all exist and match the six SHA-256 values written in V1. The prior causal review independently established 224 saved arms, exact W/self/full controls, and the strict 4/16 intended-word denominator per site. This new matrix is 16 directed mode cells × two sites × six arms = **192** planned forwards. The seed formula assigns 64 distinct values across 16 direction indices, two sites and two random arms.
- Each stock and shared-stat baseline is unpadded and has the same feature-projection, post-block-11 and CTC frame length for a pair. The frozen W and complement geometry is valid at both sites:

  | Pair/voice | T | W | Complement rows |
  | --- | ---: | --- | ---: |
  | `color_05` Piper | 271 | `[95,156)` | 210 |
  | `color_05` Kokoro | 286 | `[105,179)` | 212 |
  | `object_02` Piper | 249 | `[95,134)` | 210 |
  | `object_02` Kokoro | 278 | `[105,170)` | 213 |

  The complement includes **all** CTC time rows outside W, including quiet regions and other words. It does not mean only voice-active frames. I independently computed all 16 pair/mode/site donor-minus-recipient complement L2 norms from the saved float32 captures; each is finite and nonzero, ranging from **13.33 to 28.66**. Reverse directions have the same norm, so these represent 32 directed site cells.
- The hook sites match the corrected pinned source path: feature-projection tuple member 0 enters the encoder; post-block-11 tuple member 0 enters CTC dropout/head without a final encoder norm. Requiring exact prior W logits/IDs, exact recipient self-C logits/IDs, and exact donor full-state logits/IDs is a strong empirical gate. If any W replay fails on a fresh run, the plan correctly holds that direction/site's C interpretation rather than treating a transcript match alone as a pass.
- V1 keeps the two-endpoint intended-word outcome separate from model-donor transcript equality, retains the 12 descriptive directions per site, and limits any positive or negative complement result to this already-inspected FIT context. The 30-minute, 4-GiB RSS and 512-MiB output limits are consistent with the prior CPU run's observed footprint, subject to a newly reviewed implementation.

These passing checks do not authorize model inference. A V2 random-control clarification, independently reviewed runner/auditor, no-weight preflight and bound freeze receipt remain necessary. The experiment would still be internal ASR analysis, not an attacker-controlled waveform or security finding.
