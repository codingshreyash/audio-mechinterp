# Saved-logit CTC reference-margin diagnostic, v1

19 November 2026. This is a **FIT and previously inspected development diagnostic**, not a new model run, held-out test, safety result, or security finding. Commit this protocol and the analyzer source before reading any reference-margin result. Do not load model weights, run inference, synthesize audio, modify source runs, open locked-test material, or select a successful subset after scoring.

## Fixed inputs and population

Use only the independently audited Wav2Vec2 `facebook/wav2vec2-base-960h` saved arrays and records:

| Set | Clean source | Patch source | Planned directed cells |
| --- | --- | --- | ---: |
| FIT | `runs/wav2vec2_fit_baseline_stage_capture_v1/20261118T2352Z` | `runs/wav2vec2_fit_complement_patch_v1/20261119T0035Z` | 16 × 2 sites |
| Previously inspected development | `runs/wav2vec2_dev_word_disjoint_v1/20261119T0103Z` | same | 96 × post-block-11 site |

FIT arms are `self_C`, `W`, `C`, `R1_C`, `R2_C`; report the two independently seeded random outside arms separately. Development arms are `self`, `W`, `O`, `random_O`. The same case appears under correlated S/C preprocessing modes, directions and sites; never count them as independent speech samples. Every planned cell stays in the output, including incorrect clean ASR. The previously reviewed saved-data audits must pass with their frozen file hashes. Verify each **used** state and logit tensor against its row hash, shape, dtype and finiteness before scoring. Also verify rows, directions, arm coverage, exact clean self/full controls as recorded, and the pinned `vocab.json`/`config.json` bytes. No patched model forward is permitted.

## Fixed reference strings and score

The two alternatives are the **complete scripted recipient and donor utterances**, not the model's clean transcripts. For FIT, take the frozen `fit_chosen` prefix/suffix from `SPEC_V1.json` and insert the recipient or donor word. For development, use each clean row's frozen complete `reference_tokens`. Canonicalize both with Python NFKC + casefold + `re.sub(r"[\W_]+", " ", ...)`, then join tokens with single spaces, uppercase, and map letters through the pinned vocabulary with `|` for each space. Require every mapped symbol to be an ordinary single-character vocabulary entry. CTC blank is pinned config `pad_token_id=0`; special-token targets are invalid.

For each saved `(1,T,32)` float32 logit tensor, compute the **exact CTC sequence log probability** of each complete reference string by the standard blank-interleaved forward recursion in float64 log space, with log-softmax frame probabilities. Let `M_arm = log P(donor reference | arm logits) − log P(recipient reference | arm logits)`. Report `ΔM_arm = M_arm − M_self`, where self is the matching direction/site's saved self arm. A positive change favors the donor reference relative to the recipient; it need not change the greedy decoded transcript. An arm passes a descriptive doubling threshold when `ΔM ≥ ln(2)` nats. Preserve finite raw `M`, `ΔM`, the two complete reference strings, strict baseline eligibility, observed complete greedy transcript/IDs hash, and the original row references for every arm.

Recompute the natural donor-minus-recipient W and outside edit L2 from hashed clean states using the frozen windows. Report `ΔM/natural_W_L2` for W and `ΔM/natural_O_L2` for O and random-O **only as descriptive scale checks**. W/O region widths and distributions differ; dividing by L2 is not a causal equalization. No p-values or confidence intervals are justified by these correlated cases. Report all-planned and separately strict full-template-clean denominators by set, site, mode, pair, voice and arm: count positive `ΔM`, count at least `ln(2)`, median and range. Report paired `ΔM_O − ΔM_random_O` as a descriptive control contrast. Do not drop nonfinite scores silently; they fail the run and remain an error, not a selected denominator.

## Interpretation and falsifier

The specific question is whether donor outside-state patches that left the complete greedy transcript unchanged also move the *relative complete-reference score*. A systematic positive O margin shift, especially larger than matched-norm random O, would falsify the strong claim that outside states are behaviorally inert; no such shift would strengthen only the narrower statement that these interventions have little measured influence on this score. Either result leaves the observed W/O complete-transcript findings intact. A CTC score change is neither a spoken-action effect nor evidence that an attacker can create the internal patch.

The analyzer must save a bounded JSON result (at most 2 MiB) in this directory, with protocol/source hashes, exact source record/audit hashes, all cell rows, group summaries and a complete count audit. A separate reviewer should check input hashes, reference encoding, CTC recursion on a synthetic example, all planned denominators, raw score arithmetic and interpretation before any claim is promoted.
