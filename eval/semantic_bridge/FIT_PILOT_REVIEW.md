# Independent review: one-anchor Omni bridge readout pilot

Date: 2026-11-18. Reviewed [`evaluate_fit_pilot.py`](evaluate_fit_pilot.py), [`fit_pilot_scores.json`](fit_pilot_scores.json), the saved one-prompt Jacobian checkpoint, source [render manifest](../tts/outputs/semantic_fit_smoke_12/render_manifest.json), and frozen [SCORING.md](SCORING.md). I did not rerun the model, fit another map, or inspect any development/test model result.

## Verdict

The runner is a credible **fit-only plumbing test** of the pinned Thinker hooks, a completed full-width Jacobian, the same terminal text position, and the frozen bridge-token candidates. It gives a **negative result for `unembed(J_l h_l)` from one audio prompt**: no bridge label reaches full-vocabulary top 100 in the 12 fit items under typed, Piper, or Kokoro inputs, for either source layer or the raw logit baseline. This does **not** show that Omni lacks the bridge concepts. The current calculation omits the affine anchor required for a one-point Jacobian, and the source labels are not yet validated as Omni chat-task labels.

## Checks and measured outcomes

- The report hashes match the actual `fit_pilot_scores.json` source: render manifest SHA-256 `aabfe2060d2f295b80faca77fcaddbaa088913f60e2adb59bf01f384983c20e0`; checkpoint SHA-256 `caa1f346247ee0c0c5cb39a09bcd5a51acf564e287e2dd0d4b7c9c02f7c2ba78`; scorable manifest SHA-256 `926e53ed66a9f5162e1981a786cd2e51231a273a5ce4b1713c3dea7fd10ac1a8`. The report itself is SHA-256 `32a95ce3544258403eb1608218685716a044259b36591b4556324a219e3ab552`.
- The checkpoint's 2,048 completion bits are all true. Its two 2,048×2,048 row matrices are keyed to layers 0 and 17 and contain finite completed values. Metadata pins Qwen2.5-Omni-3B revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`, final block 35, the exact continuation system prompt, and first-item Piper audio SHA-256 `c1317b32eab0c56e86d29f3ead175e72fc57410cd92f0dae338022d1b34ef99e`.
- The render manifest has exactly the 12 fit-only smoke items; its first Piper SHA matches the fit checkpoint. Every scored item ID is fit-only. All 12 report items have typed, Piper, and Kokoro routes, with both raw and J readouts at layers `[0,17]`.
- I independently checked all saved candidate IDs against the frozen scorable manifest and all saved min-over-layer ranks against their two per-layer ranks. They agree. The fit code stores target-coordinate **rows** of `∂h_35/∂h_l`, so `J_l @ h_l` has the correct orientation for this implementation.
- Reconstructing the final Thinker logits from the captured final residual, final norm, and LM head differs by at most `0.03125` across the 36 route/item forwards. That supports the hook position and final-unembedding plumbing. The runner does not save raw logits, so I cannot independently recompute each vocabulary rank from its final scores; the check above is of report structure and the scoring code, not a second model-forward reproduction.

| Category | Route | Raw median bridge rank (6 items) | One-anchor J median rank (6 items) | J better than raw |
|---|---|---:|---:|---:|
| Association | typed | 89,591 | 44,756 | 5/6 |
| Association | Piper | 74,240 | 43,895 | 5/6 |
| Association | Kokoro | 82,297 | 47,572 | 5/6 |
| Multihop | typed | 57,630 | 41,147 | 4/6 |
| Multihop | Piper | 53,280 | 36,332 | 5/6 |
| Multihop | Kokoro | 51,999 | 35,645 | 5/6 |

The table rounds half ranks to integers for readability; exact numbers are in the JSON. Across all 36 route/item scores, J improves the rank over raw in 29, but **0/36** J and **0/36** raw scores reach top 100. These are 12 correlated items rendered in three routes, not 36 independent prompts. The smallest J ranks are 119 typed, 177 Piper, and 152 Kokoro on a multihop item; none pass the frozen top-100 threshold. In particular, the fitting item `association:interview` Piper has J min rank 92,107 and raw 61,373, so the current uncentered J readout does not even recover its own anchor label.

## Main mathematical limitation

The checkpoint is a **Jacobian at one Piper waveform and one prompt state**. For an evaluation residual `h_l`, the local first-order approximation to the final residual is

`h_35(anchor) + J_l(anchor) · [h_l − h_l(anchor)]`.

The runner instead computes `J_l(anchor) · h_l`, then applies final norm and head. It omits both the source centering and target intercept. This can produce a large constant offset even on the very prompt used for fitting. The completed matrix demonstrates that backpropagation worked; it is not yet a valid test of whether the model's bridge concept is linearly recoverable from the terminal state. Anthropic's published corpus-averaged `J_l h_l` is a different estimator, whose average may behave differently; one local Jacobian should not inherit its interpretation. The checkpoint stores matrices and metadata but **does not store the anchor source and target residual vectors**, so the current artifacts cannot compute the affine correction without a new exact-hash forward capture.

Other limitations: matrices are stored float32 but cast to BF16 for evaluation, so a precision check is useful; this is unlikely by itself to explain ranks in the tens of thousands. The runner passes no `allowed_ids` to `score_item`, leaving the frozen secondary semantic-token ranks `null`. The primary full-vocabulary metric is implemented as specified. There is no held-out score here and no audio-encoder localization: only terminal text residuals are inspected.

## Affine follow-up completed on the same fit-only items

After the caveat above was reported, the runner was amended in place to recapture source/target anchor residuals from the exact fitted waveform and to compare four methods: raw logit, uncentered `J h`, local affine `h_T(anchor)+J(h-h_l(anchor))`, and the constant `h_T(anchor)` alone. The new [affine report](fit_pilot_affine_scores.json) is SHA-256 `f21fae5397d7588851a1bf8f9eb030c841b7f1c4425668b220e99e348fb6602e`. Its pinned semantic-token ID hash `03910c8e7e6c66aca614867f024bcb7d72041b2c21c659b34ed35e0340cc2718` matches the frozen scoring artifact; both full-vocabulary and secondary semantic ranks are now populated. The current amended evaluator is SHA-256 `6f70a37e2c5d4b2c45d713804a0700ae95b13043342af0141499dff95958fb07`.

| Route | Raw median full rank | `J h` median | Affine median | Anchor-constant median | Top-100 hits, any method |
|---|---:|---:|---:|---:|---:|
| typed | 72,147 | 44,756 | 13,599 | 13,980 | 0/12 |
| Piper | 58,418 | 43,058 | 14,133 | 13,980 | 0/12 |
| Kokoro | 63,059 | 40,626 | 13,973 | 13,980 | 0/12 |

The affine map substantially improves rank over uncentered `J h`, confirming that the missing intercept mattered. But the **constant anchor baseline explains almost all of that gain**: affine beats constant on 8/12 typed, 9/12 Piper, and 10/12 Kokoro items, yet the median **paired per-item** rank advantages are only 21, 44.5, and 59 positions respectively in a 151,936-row vocabulary. The fit anchor's Piper affine rank equals the anchor-constant rank exactly by construction (`5,730` for `interview`), and it remains nowhere near top 100. Secondary semantic ranks also have zero top-100 hits for every method and route. Therefore this follow-up is still negative for a useful bridge readout; a lower median rank than raw by itself would be misleading.

The affine report includes the original uncentered method, but the earlier `fit_pilot_scores.json` was produced by a prior version of the evaluator that has since been edited in place. The original output remains hash-pinned, while its exact prior script bytes were not preserved. Future runs should save a script revision/hash with each result. The amended evaluator checks the anchor audio hash against checkpoint metadata, but it recaptures anchor states at runtime rather than storing their vector hashes separately; its own reproducibility relies on the pinned model/processor/audio/system and runtime.

## Task-label validity

The original Anthropic association prompts are prose vignettes with a **latent** concept label, not a question asking the assistant to say that label. The multihop prompts are often incomplete facts. Under Omni's user/assistant chat wrapper, the first assistant prediction site is a different task from raw text continuation. The two-item fit-only [generation pilot](task_framing_smoke_2item_snapshot.json) produced related prose in typed, Piper, and Kokoro routes but no explicit bridge word. One typed continuation stopped at the 80-token cap without EOS. This is insufficient to decide whether the bridge is represented internally; it also cannot validate all 12 fit items or source labels for Omni. The [v2 fit-only framing gate](framing_gate_v2.json) remains unmet. A token-capped output is censored, not a complete-answer pass.

## Next experiment: a discriminating audio mechanism assay, still fit-only

I recommend **pivoting before a 12-prompt corpus Jacobian fit**. Twelve full 2,048-coordinate fits would spend substantial compute while this task may never require the Omni chat model to form the released latent label. A corpus average could improve the readout, but the affine-versus-constant comparison currently offers almost no evidence that a bridge-specific signal drives the score.

1. Construct a versioned **fit-only, matched audio pair assay** with 6–8 complete spoken questions. Each pair changes one answer-determining content word while keeping syntax, voice, duration, and delivery as close as possible (for example, a country cue that changes a capital answer). Use Piper and Kokoro renderings, plus matched typed inputs. Freeze both expected full answers and the changed word's audio interval from synthesis timing or independent alignment. Keep the current 60-item source manifest intact; this is a separate positive-control task.
2. Generate full Thinker answers first. Require the clean typed and both audio routes to answer each pair correctly before that pair is used for a mechanistic **behavior-change** test. Report failures rather than silently replacing pairs. Capture audio-encoder, post-projection, and terminal text states, then test whether the known changed word can be decoded or discriminated at each stage with a fit-only linear/contrastive probe and matched wrong-time, silence, and speaker controls.
3. For a few correctly answered pairs, patch the known changed-word audio projection states or corresponding residual segment from answer A into answer B. Record the **complete** generated answer, self-patch, reverse patch, wrong-time patch, and equal-norm random patch. An answer flip localized to the content interval would show a causal audio-to-decision path; a changed rank alone would not. Only after that positive-control path is established should a multi-prompt Jacobian fit be considered as a cheaper readout of the same mechanism.

In parallel, finish the 12-item fit-only source-task framing gate if the released bridge eval remains a priority. For association, validate source-concept plausibility independently even when the generated continuation does not say the label. For multihop, judge full answers against source targets and record EOS. No development/test model outputs are needed for either decision.

This next experiment separates a mathematical readout failure from an unsupported task label before spending more compute or making a model-level claim.
