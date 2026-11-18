# Stage B implementation and verification checklist

**Status:** Design review against the frozen `PRE_DEV_EVAL_PROTOCOL_V1.md`; no Stage B model evaluation. Use FIT data and toy tensors to implement and test. Freeze any necessary clarification as a new version **before** inspecting development Stage A item outcomes. Do not change the Stage A runner or its primary gates.

## Entry and provenance

- [ ] Enter Stage B only if the frozen Stage A summary has both sites passing all four primary conditions over all 96 directions. If either fails, report that result and stop Stage B, as the protocol directs.
- [ ] Bind protocol, Stage A script/result, corpus spec/manifest, V2 canonical policy, fitted lens/basis, FIT calibration, baselines, model and Transformers revisions, and exact Stage B code by SHA-256. Preserve the 96 fixed case IDs and 12 pair groups; no case selection by clean/edited answer. Keep the locked-test split unopened.
- [ ] Use the same FP32 Thinker, no-grad execution, system prompt, audio preparation, post-encoder31/post-connector hooks, prompt-last post-block35 residual, six row lists, donor matching, and 30 GiB MPS stop as Stage A. Reuse Stage A clean captures only after verifying their hashes and tensor/row shapes. Record one source hook call for each patched forward and generation. Save full token IDs, decoded text, terminal EOS, special-token validity, and both strict and semantic one-word classifications.

## Fixed time controls at α=.025

For every one of the **96 directions × two sites**, derive the six `δ_b` from the opposite-word donor in the same pair, voice, and template. Keep source row sets and full source shapes equal. Save actual edited tensor support, six row lists, effective edit Frobenius norm, measured donor-oriented V2 prefix score change relative to the same clean anchor, terminal residual change, and complete generated answer. Use a separate hook call for each control; never choose a control based on its outcome.

| Control | Exact construction to verify | Required record |
| --- | --- | --- |
| Self | Replace donor source with recipient source, giving six zero vectors on the same rows. | Zero effective norm, clean score and complete answer parity within the established α=0 tolerance. |
| Reverse | Apply `δ_(5-b)` to recipient bin `b`, with fixed permutation `(5,4,3,2,1,0)`. | Actual norm, score, answer, and support; do not permute row positions. |
| Word only | Apply original `δ_b` only for bins `1,2,3,4`; leave bins `0,5` untouched. | Word-only norm, score, answer, and support. |
| Preword only | Apply original `δ_0` only on bin `0`, multiplied by `||ΔH_word||_F / ||ΔH_pre||_F`. | Pre/post-normalization norms; mark ineligible if denominator is zero. |
| Postword only | Apply original `δ_5` only on bin `5`, scaled analogously. | Pre/post-normalization norms; mark ineligible if denominator is zero. |
| Gaussian seeds 0,1,2 | Draw a fixed-seed normal tensor on the **same selected rows** as the control being matched, normalize to the declared effective Frobenius norm, and leave all other rows unchanged. | Seed, generator/device/dtype, target and realized norm, support, score, answer. The frozen protocol does not specify which support/norm is the reference; resolve this before implementation. |

- [ ] Unit-test each support mask with small tensors, including disjoint bins, reversal order, zero pre/post norm, exact target norm after scaling, deterministic Gaussian replay, and no edit outside selected rows. The source hook must return a tensor/tuple of the same shape and preserve all unrelated outputs.
- [ ] Verify that clean and self controls remain stable and that α=.025 uniform Stage A replay produces the same saved score and complete token suffix within the frozen numerical tolerance. Do not tune control strength against development outputs.
- [ ] Compute time selectivity only as the protocol's **secondary label**: word-only `|ΔS₂|` strictly exceeds each available preword, postword, and reversed-bin `|ΔS₂|` in at least 75% of predeclared clean-eligible directions, and fewer than 10% of the specified controls produce an **exact donor word with valid terminal EOS**. Report counts separately by site/control and over all 96 planned directions. Encoder rows may already contain earlier acoustic context; this label describes intervention location, not exclusive information origin.

## Natural donor, full source patch, and complete-sequence endpoint

- [ ] For each recipient, run the paired donor WAV naturally under the identical prompt and record full answer/EOS, prompt-last residual, V2 prefix score, and SHA. The natural difference is donor minus recipient. Do not count it as a uniform-edit calibration hit.
- [ ] Make the `α=1` **full rowwise** patch from the saved donor source tensor into the recipient at the chosen site, after verifying identical complete tensor shape. Save exactly which rows are overwritten (the text suggests the entire source tensor, but this should be explicit), one hook call, full answer/EOS, terminal change, V2 score, and effective edit norm. Treat any comparison with `α UΣJδ` as descriptive because a rowwise replacement is a different intervention.
- [ ] Teacher-force every frozen unique `SCORING_V2.json` canonical sequence **including EOS**, with no length normalization or answer-selected variant. For sequence `c_0…c_(n−1)`, append `c_0…c_(n−2)` to the prompt and sum `log softmax(logits[prompt_len−1 : prompt_len−1+n])[c_0…c_(n−1)]`. The audio-source patch must fire exactly once on the extended forward and affect only the prompt's audio path. Aggregate each word's unique sequence log probabilities with `logsumexp`; subtract recipient class from donor class for donor-oriented contrast. Save per-variant values and class contrasts, plus the generated full answer. The complete-sequence score is a later-decoder behavioral endpoint and **must not enter** prompt-last map MAE.
- [ ] On FIT cases only, reproduce the existing `fit_pool_scoring_v2.py` clean sequence scores from the new scorer. Toy tests must catch off-by-one prompt indexing, omitted EOS, duplicate sequences, wrong donor orientation, hidden special tokens, and hooks firing during decode. Specify in advance whether sequence scores cover just clean/uniform `.025`, all six Stage A alphas, all eight time controls, natural donor, and full rowwise patch; the current protocol says every direction but does not enumerate every condition.

## Reporting and interpretation

- [ ] Preserve all 96 directions per site in reports. Show all-planned and clean-eligible denominators, invalid/zero-norm counts, time-control score and complete-answer distributions, exact donor-answer rates, natural versus full-patch differences, and canonical complete-sequence contrasts. Aggregate uncertainty by the 12 independent word pairs, retaining all eight related contexts together.
- [ ] Distinguish three outcomes: Stage A primary transport gate, Stage B time-selectivity **label**, and complete-answer behavioral effects. Stage B controls cannot repair a failed Stage A gate. A passing prefix transport gate alone does not establish a changed complete answer, authorization decision, or security vulnerability.

## Ambiguities requiring a versioned clarification before a positive Stage B claim

1. **No positive full-answer gate.** The protocol requires Stage B to run but sets no threshold for canonical complete-sequence contrast or actual generated donor answers. A positive claim about complete-answer transport therefore has no prespecified success rule. Report those as descriptive unless a new version fixes an endpoint and threshold before development outcomes are inspected.
2. **Time-selectivity denominator.** Define `clean-eligible` exactly (for example, recipient clean semantic word+EOS, and whether donor clean correctness is also required). Define whether zero-norm pre/post controls are excluded, failures, or reported as a separate unavailable category, plus a minimum eligible coverage. Without that, a tiny selected subset could earn the ≥75% label.
3. **Control threshold and norm.** Specify whether `<10%` exact-donor rate applies separately to preword/postword/reversed controls or to their pool; whether Gaussian/self are included; and whether ties or changes below the frozen V2 floor count. Specify the Gaussian support and whether its target norm is the full six-bin or word-only effective edit norm.
4. **Complete-sequence condition matrix.** Enumerate which patched conditions require teacher-forced scores and whether the required contrast is donor-minus-recipient or expected-minus-foil. Save both orientations only if clearly labeled. State whether natural donor and full-row patch are included.
5. **Full patch scope.** Define whether “full rowwise donor patch” overwrites every source row or only the six selected bins. These interventions differ, so the source mask and norm must be frozen.

If these choices are made after reading Stage A item outcomes, they are exploratory labels. A frozen version made before those outcomes can support a prospective development interpretation; the locked test must use the identical choices once.
