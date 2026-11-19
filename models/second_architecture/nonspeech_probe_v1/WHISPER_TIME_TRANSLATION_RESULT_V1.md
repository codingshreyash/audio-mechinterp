# Whisper Tiny time-translation result, V1

19 November 2026. This is a **white-box mechanism result on one engineered non-speech zero/noise pair**, not a security finding. The [prospective protocol](WHISPER_TIME_TRANSLATION_PROSPECTIVE_V1.md) and [frozen plan](WHISPER_TIME_TRANSLATION_PLAN_V1.json) preceded all three model attempts. The [source-only executable review](WHISPER_TIME_TRANSLATION_EXECUTABLE_INDEPENDENT_REVIEW_V1.md) passed before the three stage-specific GOs. The [empirical receipt](WHISPER_TIME_TRANSLATION_EMPIRICAL_RECEIPT_V1.json) records the exact joint decision and raw-attempt file hashes; an [independent raw-result audit](WHISPER_TIME_TRANSLATION_INDEPENDENT_RAW_RESULT_AUDIT_V1.md) recomputed and confirmed them. Raw attempt folders are retained locally under `runs/whisper_time_translation_v1/`.

## Question and result

The prior donor-to-recipient rescue used the **union of two 5-row encoder windows** near the source/tail and tail/end transitions. We asked whether that combined rescue follows the waveform when 0.5 or 1 second of silence is inserted before it, compared with placing the same silence after it. All three one-shot attempts completed and passed their saved-result audits: **28 original calls, 62 half-second calls, and 62 one-second calls**, with no skipped arms. All ten no-hook baselines produced complete EOS-terminated outputs. Each translated zero/noise pair retained a distinct answer and an equal attention mask. Saved full mel/mask arrays were recomputed from their frozen WAVs, and the predeclared interior mel translation matched exactly in the checked windows.

| Prespecified combined-window cell | Original window viable? | Strict translated crossover? | Main failing condition |
|---|---:|---:|---|
| Post-convolution/position, +0.5 s | Yes | **Yes** | None under the frozen rule |
| After encoder block 0, +0.5 s | Yes | No | Prepend aligned direct patch did not transfer the full donor path |
| Post-convolution/position, +1 s | Yes | No | Prepend equal-norm relocation changed the recipient answer |
| After encoder block 0, +1 s | Yes | No | Prepend equal-norm relocation changed the recipient answer |

Thus **1/4 combined-boundary cells qualified**. In that cell, a donor-state patch at the waveform-aligned shifted rows for prepend and at the original rows for append reproduced the complete donor token path through EOS. Direct patches at the opposite windows did not. Identity, unrelated-row, displaced equal-norm, and cross-window equal-norm relocation controls preserved the recipient path; full-donor controls reproduced the donor path. The unrelated-row donor delta happened to be zero at this site, so that control alone is weak; the nonzero displaced and cross-window relocation controls carry the specificity check. This argues against a simple explanation in which only the original absolute row matters for that cell. It does not establish that the same location will work at other shifts or on other sounds.

The isolated original source/tail window failed its viability check at both sites, so **0/2 isolated source/tail cells qualified** across the two offsets. The original isolated tail/end window was viable only at the post-convolution/position site. The successful combined patch therefore cannot tell us which boundary, or interaction between them, caused the rescue. In the +1-second prepend, the equal-norm relocation changed a complete answer even though it did not reproduce the donor answer; the prespecified specificity rule correctly excludes that cell.

The baselines also show why complete-answer controls matter. The original zero/noise pair transcribed as `You`/`I`; adding silence after the pair retained those answers. Adding silence before the noisy pair yielded longer hallucinated sentences, while the zero-tail answer remained `You`. The reported crossover compares donor and recipient **within each matched layout**, using exact full token sequences including EOS. These outputs are hallucinations from engineered non-speech WAVs, not intelligible spoken requests.

## Interpretation boundary and next experiment

This is a causal intervention on saved encoder states, under one model, one synthetic sound family, and one decoding path. It is not an attacker-supplied waveform effect, a unique circuit, a proof of necessity, or a confidentiality, integrity, or availability violation. The pre-attention site is more locally interpretable than the post-block-0 site, but even the combined pre-attention window spans two transitions. Appending silence preserves the old noise-tail/silence interface while creating a new waveform end, so the combined result could follow the old tail boundary or depend on two windows together.

A new prospective test should isolate the originally viable tail/end window. For append, it should compare the old tail boundary with the new waveform end and include union-minus-one-window ablations. Any positive result should then be replicated on a separately selected sound pair before making a general audio-mechanism claim. Human speech and audio-language action paths remain separate required steps for the broader security objective.
