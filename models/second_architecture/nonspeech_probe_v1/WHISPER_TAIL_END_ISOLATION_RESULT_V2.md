# Whisper Tiny isolated tail/end follow-up, V2

19 November 2026. **The prespecified isolated-window timing claim failed: 0/2 offsets qualified.** This is a white-box test on one engineered non-speech zero/noise pair, not a security finding. The [V2 protocol](WHISPER_TAIL_END_ISOLATION_PROSPECTIVE_V2.md) was written after the V1 combined-window result and before any V2 model inference. The [frozen plan](WHISPER_TAIL_END_ISOLATION_PLAN_V2.json), [executable review](WHISPER_TAIL_END_ISOLATION_EXECUTABLE_REVIEW_V2.md), source-only admissions, and stage-specific GOs preceded the three attempts. The [empirical receipt](WHISPER_TAIL_END_ISOLATION_EMPIRICAL_RECEIPT_V2.json) records the exact joint result and hashes of the saved raw attempts; the [independent raw-result audit](WHISPER_TAIL_END_ISOLATION_INDEPENDENT_RAW_RESULT_AUDIT_V2.md) recomputed and confirmed them.

## What happened

All three bounded attempts completed and passed the saved-result auditor: **7 original, 24 half-second, and 24 one-second calls**, with no skipped arms. Every generated path ended in EOS. The original `498..502` tail/end window passed its clean contrast and controls, including a nonzero donor-minus-recipient state difference (L2 **8.1954**). Both translated prepend and append zero/noise pairs retained complete, distinct answers and equal masks. Their saved full mel/mask tensors were checked against the frozen WAVs; the prespecified interior mel shift matched exactly.

| Offset | Prepend: aligned old tail/end | Append: old tail/end | Strict offset result |
|---|---|---|---|
| +0.5 s | Direct patch copied the donor answer; displaced equal-norm control made a small but real complete-path change, not a donor switch. | Aligned patch and controls qualified. | **Fails** exact-path specificity |
| +1 s | Aligned patch and its local controls qualified; relocating its equal-norm delta to the competing rows changed the recipient answer. | Aligned patch and controls qualified. | **Fails** cross-window specificity |

At +1 second, the competing direct patch on prepend also produced a third complete answer rather than preserving the recipient; the protocol records this separately. At both offsets the append layout's *new waveform end* had native donor-minus-recipient delta **0.0**, while the old noise-tail/silence interface at the original rows had a substantial delta. A native donor patch at the new end is therefore a zero edit and cannot show that the model ignores the new end. The nonzero old-interface delta relocated to the new-end rows preserved the recipient in these tests.

The unrelated-row donor edit also had zero measured state delta at this site, so that control alone has little force. The decisive +0.5-second prepend displaced edit and +1-second prepend relocation used nonzero deltas; their failures determine the strict result under the frozen rule.

The source pair's zero/noise baselines transcribed as `You`/`I`. Appending silence kept those outputs; prepending silence changed the noisy baseline to a longer hallucinated sentence. The V2 comparisons use complete donor and recipient token paths **within each layout**. They do not compare the prepend answer to the append answer as if those contexts were identical.

## What this means

The earlier V1 **1/4 combined-window crossover remains its own qualified observation**. V2 does not turn it into an isolated tail/end mechanism: one shifted isolated patch failed a norm-matched control, and the other failed a cross-window relocation control. This leaves positional context, nearby representations, and interaction between the two V1 windows open. The V2 window was selected using V1 evidence, and neither result generalizes to speech, other checkpoints, or an attacker-controlled action.

The 25 source-only tests passed before GO and inference. One direct-worker test assumes the fixed attempt path has not yet been created; rerunning that test after the completed attempts fails on its expected error message because the live pipe guard now rejects the direct launch first. A post-run no-weight check confirmed that direct launch still fails before the model loader is called. The frozen pre-run source and GO records remain unchanged.

The next decisive work should move to a separately selected sound family and then to reviewed speech in an audio-language model, with complete native answer/action endpoints. Additional edits to this same selected non-speech pair would be exploratory unless frozen as a new study. The broader goal remains understanding where audio features enter the model's decisions and whether a causal intervention changes a security-relevant outcome.
