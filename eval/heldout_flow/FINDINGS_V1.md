# Small held-out spoken-word transfer test

Date: 2026-11-18. The [evaluation plan](plan_v1.json), [row-selection addendum](evaluator_rule_addendum_v1.json), [16-clip manifest](stimuli_v1/manifest.json), and [complete result](results_v1.json) identify every input and intervention. Result SHA-256: `1346d01ed5fbc23227d8fbba7078205afa25de3ef0e9f1a864ab4d678766c0a2`. This tests the **Qwen2.5-Omni-3B Thinker text answer**, not its Talker speech output.

Eight matched audio pairs combine two new word contrasts (`cat/dog`, `Monday/Friday`), two new spoken phrasings, and two voices (`Piper Amy`, `Kokoro Bella`) from the same two TTS families as the fit pilot. Both words in each pair have identical int16 PCM outside the padded word slot. All **16/16 clean clips** produced their expected complete word answer and EOS; therefore all 16 reciprocal directional cases per layer are clean-eligible. Cached and uncached full-prompt post-block activations were exactly equal at all five tested layers in the saved comparison.

The first evaluator invocation stopped at a checkpoint-config attribute check **before processing any clip**; the evaluator was corrected to compare its pinned adapter revision to the frozen plan, then the full run succeeded. The plan and audio files did not change. The independent audit found no answer-dependent filtering, but this workspace had no immutable pre-run Git commit, so file timestamps alone cannot prove the plan was frozen before the result.

## Frozen primary outcome

The primary preregistered contrast was the donor-answer flip rate for nominal word-slot audio-placeholder patches at post-block layer 30 **minus** layer 31. It was **0/16 difference**: both layers flipped **16/16** complete answers. Therefore the exact layer-30/31 transition found in the fit-only red/blue example **did not transfer** to these fresh words, voices, and phrasings.

| Post-block Thinker layer | Donor answer after word-position patch | Recipient answer after word-position patch |
| --- | ---: | ---: |
| 29 | 16/16 | 0/16 |
| 30 | 16/16 | 0/16 |
| 31 | 16/16 | 0/16 |
| 32 | 1/16 | 15/16 |
| 35 | 0/16 | 16/16 |

The sole layer-32 flip is Kokoro Bella, `cat/dog`, “Please remember this word,” recipient `dog` → donor `cat`. This later 31→32 drop is an **observed secondary pattern**, not the frozen primary contrast. The terminal-position donor positive control gave **0/16** donor answers at layer 31 and **16/16** at layer 35. Terminal position was not preregistered for layer 32, so this run does not locate the exact handoff within that block.

Across the five layers, self patches retained the recipient answer **80/80**, direct same-width wrong-time donor patches **80/80**, and three equal-norm random word-position edits **240/240**. All **512 patched continuations** reached normal EOS, and every hook modified one full-prompt prefill call while leaving subsequent decode calls unchanged. The largest relative realized BF16 norm deviation among random edits was below 0.005%. Direct wrong-time donor differences were smaller than word-position differences (at layer 30, median norms 153.73 versus 468.61), so the wrong-time control alone is not strength matched.

This small test supports transfer of a **late, position-specific answer influence** across two new lexical contrasts, voices, and phrasings in one model. It falsifies a fixed layer-31 boundary. The 16 directional cases are correlated within two word pairs and two TTS families; they are not 16 independent semantic replications. There is no human-voice, second-model, native speech-output, or security-decision result here. The [independent result audit](INDEPENDENT_REVIEW.md) recomputed the outcomes and found no implementation or denominator discrepancy.

## Posthoc block-32 follow-up

After inspecting the 31→32 drop, a separate [plan](block32_mediation_plan_v1.json) tested the terminal attention and MLP outputs inside block 32 on the **same 16 already-inspected directions**. Its [complete result](block32_mediation_results_v1.json), SHA-256 `eaa7a8456f1e68b08822080fe016c2c1cc2e27fd02b01ca111b3565489ddd1fa`, reproduces the upstream layer-31 word-patch donor answer in **16/16**. Restoring the recipient's terminal block-32 **attention output** under that patch kept the recipient answer in **14/16**; restoring the terminal MLP output kept the recipient answer in **0/16**. Donor attention or MLP output alone flipped **0/16** each. Donor post-block-32 terminal residual swapped alone flipped **10/16**, so a terminal position is not uniformly sufficient at this precise layer. Self and 48 equal-norm random attention-output edits kept recipient answers; all 176 patched continuations reached EOS. The two attention-restoration exceptions are Kokoro Bella, `cat/dog`, recipient `dog` → donor `cat`, under both spoken phrasings.

This is consistent with terminal attention being an important path for the induced audio-position influence across these cases. It is **not** complete mediation, a unique block or head, or new held-out confirmation. The block-32 probe was selected after viewing the result above and uses the same audio. The [independent review](INDEPENDENT_REVIEW.md) checked its hashes, cases, hook calls, outcomes, and limits.
