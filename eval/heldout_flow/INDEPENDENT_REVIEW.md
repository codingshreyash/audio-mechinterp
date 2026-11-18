# Independent review of held-out audio-flow test v1

I audited the [frozen plan](plan_v1.json), [renderer](render_v1.py), [stimulus manifest](stimuli_v1/manifest.json), [rule addendum](evaluator_rule_addendum_v1.json), [evaluator](evaluate_v1.py), and [saved results](results_v1.json) without rerunning the model or changing the results.

## Artifact and implementation checks

- The result hashes match the current plan, addendum, and stimulus manifest. The manifest matches the plan and renderer hashes. The plan's three TTS source/dependency hashes also match. All 16 fragment raw and processed WAV hashes and all 16 final clip WAV and decoded int16 PCM hashes match their manifest entries.
- I reopened the eight matched WAV pairs as mono 16 kHz 16-bit PCM. Every pair has equal frame length and exact sample equality before and after the recorded word-slot interval; every word slot differs. The renderer assembles each pair from shared prefix/suffix PCM, shared gaps, and right-padded word fragments. These slot intervals are splice intervals, not verified phoneme boundaries.
- The plan names two new word pairs (`cat`/`dog`, `Monday`/`Friday`), two new phrasings, and new Piper Amy and Kokoro Bella voices. The same two TTS families and the same Qwen checkpoint were used in the fit-only pilot. File modification times are consistent with plan → renderer → manifest → addendum → evaluator → results, and the addendum's hash binds it to the rendered stimuli. Because this workspace has no Git commits or immutable timestamp record, the claimed pre-answer freeze cannot be independently proved from files alone. I found no answer-dependent selection in the renderer or evaluator.
- The addendum fixes the row selection rule for slots that cross a chunk boundary before the recorded model run. I independently reprocessed all 16 WAVs with the pinned processor and source-matched row index. All recorded word-row, wrong-time-row, and terminal-token positions recompute exactly. Wrong-time windows are disjoint and have the same row count as the word windows for all eight pairs.
- `resolve_hook_points` selects the post-output of the requested decoder blocks. The evaluator captures donor and recipient residuals on uncached full prompts, replaces only chosen positions on the cached generation prefill, and leaves decode calls unchanged. Its self-patch check compares complete token IDs to clean generation. The saved cached-versus-uncached full-prompt states match exactly at all five tested layers for all 16 clips (80 comparisons, zero changed elements). This check does not by itself compare a normal `generate` prefill state with a standalone forward state, though self patches stayed behaviorally identical.

## Recomputed outcomes

All 16 clean clips answer their own word with normal EOS, so all eight pairs and all 16 directions meet the registered clean-eligibility gate. The intention-to-test and eligible denominators are therefore identical. The full-answer donor flip requires exact normalized donor word and EOS; no generated answer was malformed or lacked EOS.

| Post-block word-row donor patch | Donor answer / 16 directions | Piper / 8 | Kokoro / 8 | Animal / 8 | Weekday / 8 | Chosen / 8 | Remember / 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer 29 | 16 | 8 | 8 | 8 | 8 | 8 | 8 |
| Layer 30 | 16 | 8 | 8 | 8 | 8 | 8 | 8 |
| Layer 31 | 16 | 8 | 8 | 8 | 8 | 8 | 8 |
| Layer 32 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |
| Layer 35 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The only layer-32 donor flip was Kokoro, animal, remember phrasing, dog recipient → cat donor. The **registered primary layer-30 minus layer-31 contrast is 16/16 − 16/16 = 0**. The apparent cliff from layer 31 to layer 32 is a secondary observation discovered from the planned layer sweep, not a successful transfer of the pilot's exact 30→31 boundary. Both directions of a matched pair reuse the same two clips, so 16 directions are not 16 independent stimuli.

Across the five layers, self-word patches preserve the recipient answer in 80/80 cases; same-width wrong-time donor patches preserve it in 80/80; and three equal-norm random word-row edits preserve it in 240/240. All wrong-time controls were eligible. The random edits' effective BF16 norms are 0.999966–1.000040 times the requested donor word-row norm. Wrong-time donor differences are only 0.218–0.624 times the word-row edit norm, so that control tests location but is **not norm matched**. Terminal donor patches give the donor answer in 0/16 at layer 31 and 16/16 at layer 35. These terminal patches alter the whole terminal residual vector; they do not localize a specific feature.

There are 512 patched generations plus 16 clean generations. All 512 patched generations reached EOS; each recorded exactly one modified prefill hook call and two unmodified decode hook calls. All 16 clean generations reached EOS. No implementation bug or denominator discrepancy was found in this audit.

## Claim boundary

The held-out words, voices, and phrasings support transfer of **word-local audio information flow through at least block 31** under this particular state-swap intervention. They do not support a fixed layer-30→31 handoff, an exact natural first-use layer, a unique circuit, speech-to-speech behavior, or a security violation. The new 31→32 boundary is worth testing in another independently frozen set. The sample is small: eight matched synthetic-audio pairs, two TTS families, one checkpoint, and one prompted single-word task.

## Posthoc prompt crossover diagnostic

I also read the later [crossover plan](../controlled_pairs/prompt_crossover_plan_v1.json), [code](../controlled_pairs/prompt_crossover_v1.py), and [results](../controlled_pairs/prompt_crossover_results_v1.json). The plan and result hashes match the original red/blue manifest and this held-out result. The diagnostic reuses the **fit-only red/blue WAVs** but changes their system prompt to the held-out prompt; it was designed after inspecting the held-out boundary. Twelve cases (four voice/direction combinations × three layers) have clean correct answers, all patched outputs reach EOS, and each patch records one prefill edit and two unchanged decode calls. Self patches keep the clean complete answer.

| Layer | Word-row donor answers / 4 | Terminal donor answers / 4 |
| --- | ---: | ---: |
| 30 | 4 | 0 |
| 31 | 1 | 1 |
| 32 | 0 | 4 |

The changed prompt shifts the original stimuli's layer-31 result toward a mixed transition, but it does not reproduce the held-out set's 16/16 word-row flips at layer 31. Prompt wording can affect the measured boundary; this posthoc diagnostic cannot explain the whole difference or serve as a new held-out replication.

## Posthoc block-32 sublayer mediation

I audited the later [mediation plan](block32_mediation_plan_v1.json), [hook code](block32_mediation_v1.py), and [results](block32_mediation_results_v1.json) against the pinned decoder forward path. The result's plan, stimulus-manifest, and prior held-out-result hashes match; the plan's prior held-out-plan hash matches; and all 16 input WAV hashes match their manifest entries. The grid contains each of the eight frozen pairs in both directions, with no duplicate case or excluded failure. Word-row and terminal-token positions match the earlier held-out results.

| Intervention | Complete donor answers / 16 |
| --- | ---: |
| Layer-31 word-row donor patch | 16 |
| Block-32 terminal postblock donor state only | 10 |
| Block-32 terminal attention output donor only | 0 |
| Block-32 terminal MLP output donor only | 0 |
| Layer-31 word patch plus restore clean block-32 terminal attention output | 2 |
| Layer-31 word patch plus restore clean block-32 terminal MLP output | 16 |
| Self terminal attention or MLP output patch | 0 / 32 |
| Equal-norm random terminal attention edits, three seeds | 0 / 48 |

The layer-31 word-patch token IDs reproduce the prior held-out run in all 16 cases; both self-patch types reproduce the prior clean token IDs. All **176 condition generations** end on EOS. Each registered hook has one prefill edit and two unmodified decode calls (208 hook records). Effective BF16 random-edit norms are 0.999780–1.000141 times the requested donor attention-output difference. The two surviving donor answers after clean-attention restoration are Kokoro animal dog→cat, in both phrasings; their donor-oriented first-token margins are 0 and +0.375, close to the greedy decision boundary.

In the pinned block-32 forward, the attention output is added to the incoming residual; that sum is normalized and fed to the MLP, whose output is added next. Restoring clean terminal attention output therefore also changes the terminal MLP **input**. For this posthoc intervention, the terminal attention-output route is needed for the donor flip in **14/16** cases, while restoring only the clean MLP output prevents **0/16**. The result does not identify one head, prove that attention alone is sufficient, or show exclusive natural information flow. Donor activations came from separate clean uncached forwards and were patched into cached generation; the earlier held-out prefill comparison and self controls support compatibility, but do not independently equate every sublayer value across those modes. This run reused already-inspected held-out inputs after the boundary was seen, so it is a mechanistic follow-up, not a fresh held-out confirmation or security finding.
