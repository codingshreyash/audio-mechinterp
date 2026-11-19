# Qwen action method: spoken-input bridge candidate V1

19 November 2026. **Source only. No Qwen audio inference or candidate GO.**
`audio_candidate_v1.py` joins the already audited typed F_L1 measurement method
to the pinned Qwen audio processor path. It accepts only the six development
Piper WAVs after the real 18/18 two-listener review, coordinator statement,
saved V2 QA report and separate independent review receipt all agree. Under
the reviewed V2 intake, a critical first-review disagreement remains HOLD
even if a third reviewer documents it. The candidate cannot select an `H_*`
locked WAV. The original blind exports must be inspected
by a real independent reviewer; the unit-test mock does not count.

The candidate re-runs the existing 18-WAV technical source and blind-bundle
audits, hashes the selected final WAV and verifies mono/16-bit/16-kHz/16-second
geometry. It uses the frozen system prompt and a **single audio content block**
with no text transcription in the user message. A future processor-only call
verifies nine small files from the exact Qwen checkpoint revision, then
constructs the processor internally from that verified snapshot. A caller
cannot supply a separately constructed processor. It checks tokenizer EOS,
then records prompt IDs, the final
prompt position, audio placeholder positions, 128-channel mel features, masks,
full tensor hashes and source-matched encoder/projector row counts. It does not
load weights or invoke Thinker. Those row counts describe tensor geometry, not
human word timing. Phrase timing requires separate alignment review.

I exercised the audio processor geometry on the **existing non-study**
`models/qwen_omni/sample_apple_5p5s.wav` only. The pinned local processor
returned input IDs `[1,236]`, features `[1,128,30000]`, a matching feature
mask, 550 valid mel frames, 275 encoder rows and 137 projected/audio-token
rows. This was a processor call without model weights, scores, gradients or
generation. No D_* or H_* candidate WAV was passed to Qwen or its processor
in this check. The first source draft incorrectly expected 80 mel channels;
this live no-weight check caught that and the code now requires 128.

Run the no-weight source plan and tests from the repository root:

```sh
eval/tts/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.audio_candidate_v1 --source-plan
eval/tts/.venv/bin/python -m unittest models.second_architecture.qwen_action_method_v1.test_audio_candidate_v1 -v
```

The optional `--prepare-development D_L1__piper_en_US_amy_medium` entrypoint
requires paths to two actual blind exports, coordinator attestation, saved QA
report and independent review receipt. It fails before importing Transformers
when they are absent. The future independent review receipt must have schema
`qwen_action_audio_human_qa_independent_review_v1`, status
`PASS_REAL_HUMAN_QA`, exact hashes for both exports, attestation, optional
third export, report,
source manifest and bundle mapping, and true independent-person/original-export
review statements. The candidate recomputes the V2 human audit from the raw
files and requires all 18 initial pairs to pass with zero critical disagreement.
It always reports `candidate_model_inference_authorized: false`.

Before using this bridge for **model** inference, independently review its
source and the actual human records, freeze a measured audio resource pilot,
verify exact processor/hook/no-hook parity, design a bounded worker and saved
auditor, and issue a separate executable GO. Reuse the typed F_L1 score/VJP
method at the final prompt position, while preserving each audio case's own
prompt length and the full complete-answer endpoint. A three-label score is
not a native function call. A model-origin security finding still requires an
attacker-controlled input and a consequential action boundary.
