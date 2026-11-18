# Fit-only controlled audio-state intervention: observed result

Date: 2026-11-18. Model: pinned Qwen2.5-Omni-3B **Thinker**. These are text answers generated from speech input; the Talker and native speech output were not tested. The [protocol](PROTOCOL.md), [v1 patch plan](patch_plan_v1.json), [v2 controls plan](patch_plan_v2_controls.json), code, full token outputs and hashes remain available. The v2 controls were specified **after** seeing the v1 answer flips.

## Input and clean behavior

Within each of two independent TTS families, a red/blue pair has the same length and identical 16 kHz PCM outside a fixed word slot. Piper's slot is 1.130–1.664 seconds, covering 27 of 100 encoder rows in its self-attention chunk. Kokoro's slot is 2.245–3.820 seconds, covering 78 of 100 rows in its chunk. The slot includes TTS silence and padding, not only voiced color phonemes. The [assembled-waveform manifest](../tts/outputs/controlled_pair_v1/controlled_pair_manifest.json) records exact samples, sources and hashes.

The model's **log-mel input is not strictly identical outside that slot**. An independent processor comparison found changed valid mel frames far from it (219 for Piper, 348 for Kokoro, excluding a ±5-frame boundary), with maximum absolute differences 0.0673 and 0.0378. The feature extractor uses a clip-wide maximum when flooring log-mel values. Thus byte-identical waveform context does not imply byte-identical model features or a perfectly isolated internal word signal.

The Thinker answered `Red` or `Blue` correctly on all four audio clips in two identical greedy runs, with normal end-of-turn in every run ([full answers](../tts/outputs/controlled_pair_v1/thinker_answers.json)). A pinned Whisper Tiny screen recognized the intended color on the two Piper clips and Kokoro red; it transcribed Kokoro blue as `Blues`. That is an ASR observation, not proof the waveform said a different word ([screen](../tts/outputs/controlled_pair_v1/asr_screen.json)).

## Encoder patch result

The [v1 output](patch_results_v1.json), SHA-256 `6cfcd39fb57914559ee34fc9e21e45124d28692b9e49081ba9a51fea13e5368d`, contains 12 engine × layer × direction cases: two engines, encoder post-block layers 0/15/31, and reciprocal red→blue/blue→red edits. Donor and recipient WAVs were processed with the same task frame. A forward hook replaced selected recipient encoder rows with the corresponding donor rows during generation.

| Condition | Complete donor-color answers | Complete recipient-color answers |
|---|---:|---:|
| Known word-slot rows | 12/12 | 0/12 |
| Entire changed attention chunk | 12/12 | 0/12 |
| Same-count rows in another chunk, direct donor swap | 0/12 | 12/12 |
| Self-patch at word-slot rows | 0/12 | 12/12 |

All outputs reached normal end-of-turn. The donor-oriented first-token Blue-minus-Red margin changed from a median **−4.38 to +4.41** for Piper and **−5.72 to +5.63** for Kokoro under the word-row swap. The full decoded answer changed as well; the result is not merely a moved score.

The original wrong-chunk donor difference is much smaller than the word-row difference because the waveform is unchanged there. It is a useful intact-input check, but by itself does not match intervention strength.

## Equal-magnitude and position controls

The [v2 output](patch_controls_v2.json), SHA-256 `79767585596fd22a9830110b809a16af84f6e69ae73b83fd263eb99f70d6e4ea`, uses the same 12 cases. The requested random edit has the same Frobenius norm as the donor-minus-recipient word-row difference; the realized BF16 edit norms agree within displayed rounding. All outputs reached normal end-of-turn.

| Post-v1 control | Donor-color answers | Recipient-color answers |
|---|---:|---:|
| Five fixed, norm-matched random word-row edits per case | 0/60 | 60/60 |
| Actual donor word-row difference added in another chunk | 2/12 | 10/12 |
| Actual donor difference shuffled across word rows | 1/12 | 11/12 |

The two wrong-chunk flips and one shuffled-row flip prevent a claim that the signal works *only* in the nominal word rows or that every row's temporal order is necessary. The random controls argue against a generic effect of perturbation magnitude on this one pair. The patch swaps a **post-attention** state; each row may already contain information from its entire 2-second chunk. This identifies an effective intervention window in this model/task, not a phoneme-localized circuit or a deployable attack.

## Log-mel input control after the preprocessing caveat

The clip-wide mel-floor difference prompted a separately fixed [feature-splice plan](feature_splice_plan_v1.json). In the [feature-splice output](feature_splice_results_v1.json), SHA-256 `47bdc8b6b25d0f09f742d9056f5c790271b6f9422574e011c3fcbf0487c7a67a`, we copied only donor log-mel frames in the word slot plus five frames of margin on either side into a recipient feature tensor. All token IDs, masks, and all other features came from the recipient. This **word-only** hybrid produced the donor color in 4/4 reciprocal engine cases, with normal EOS. The opposite hybrid, containing donor features only *outside* that interval, kept the recipient color in 4/4. Piper used valid mel frames 108–172; Kokoro used 219–387. Other valid mel frames still differed between clean clips, but this experiment shows those far differences were not required to switch the answer in these four cases.

This is a model-input intervention on a hybrid feature tensor, not a physically replayable WAV. Its result supports sufficiency of the selected word-adjacent mel interval for this simple decision, while the encoder-row result remains a post-attention state intervention.

## Connector-stage continuation of the path

The separately fixed [connector plan](connector_patch_plan_v1.json) targets the output of `audio_tower.proj`, where adjacent audio encoder rows are pooled and projected into the 2,048-wide language-model input space. The [connector results](connector_patch_results_v1.json), SHA-256 `81ce863c0cac1dbc7ef47e9a55b498341a812e8e165d3305c8303991ded3ff0d`, used 13 projected rows in Piper's word slot and 40 in Kokoro's. Reciprocal donor swaps at those rows changed the complete answer in **4/4** cases, with normal EOS. Whole changed-chunk swaps also changed 4/4; direct wrong-chunk and self swaps changed 0/4. Five equal-norm random word-row edits per case changed 0/20 complete answers. The random edits' realized BF16 norms matched the donor edit within displayed rounding.

This extends the causal assay from processed mel input and encoder state to the audio-to-language connector for this pair. It does not yet trace a unique concept direction through the Thinker residual stream or the Talker. The direct wrong-chunk differences have smaller norms than the word-slot differences, so that control alone is not strength-matched.

## Thinker residual influence curve

The [three-layer Thinker probe](thinker_patch_results_v1.json) patched the projected word-slot **audio placeholder token positions** after text decoder blocks 0, 17, and 35. The donor answer appeared in all 8 early/mid-layer cases and none of the 4 final-layer cases; a donor terminal-position swap had the opposite pattern (0/8 early/mid, 4/4 final). Self, wrong-chunk and 60 equal-norm random word-position edits kept the recipient answer. All generation calls patched only the full-prompt prefill, leaving subsequent one-token decoding calls unchanged; all outputs reached EOS.

The [fixed 36-layer word-position sweep](thinker_layer_sweep_results_v1.json), SHA-256 `cc2b02b134373e12e200d2ac85b54b4092c221de04998a1c6808d7c8261ffe89`, gave donor-color answers for **all four** engine/direction cases at every post-block layer 0–30 (124/124 cases), then recipient-color answers at layers 31–35 (0/20 donor answers). The donor-oriented first-token margin still moved at layer 31, but remained below the answer-flip threshold. No configured attention mask or layer type changes at layer 31 in this pinned Thinker; see [the source audit](LAYER_BOUNDARY_ANALYSIS.md). This is a sharp *patch-sufficiency boundary* on one pair, not proof that the decision first forms in exactly one block.

The [posthoc handoff probe](thinker_handoff_results_v1.json), SHA-256 `abd1742e936065f0801894bbbb2a989ac845989fbe6f6e0adace39c572179358`, checked the other side of that boundary. Donor terminal-position swaps gave no donor answer at layers 0–30 and donor answers in all four cases at each layer 31–35. Swapping **all** audio placeholder positions gave donor answers at layers 28–30 and none at 31–35. Self terminal patches and 40 equal-norm random terminal edits at layers 30/31 gave no donor answers. All outputs reached EOS. Together these interventions are consistent with a transfer of answer-relevant influence from audio placeholders into the terminal state by the late decoder, with a threshold crossover at post-block 31 in this task. They do not identify a unique attention head or rule out a gradual accumulation crossing the answer threshold.

The further [preplanned block-31 sublayer probe](block31_mediation_results_v1.json), SHA-256 `3d780aac04da2db3cfd3f30f563fc7eb2fb74614ea8213012be6086ac1d5ae06`, tests the specific crossing within the block. A post-block-30 word-audio-position donor patch flipped all four full answers. Restoring the recipient's **terminal self-attention output** inside block 31 while keeping that upstream patch prevented the answer flip in **4/4**. Restoring its terminal MLP output instead prevented **0/4**. The donor attention output alone flipped **2/4**, both blue-to-red directions; donor MLP output alone flipped **0/4**. Self patches and 20 equal-norm random attention-output edits kept the recipient answer. All 48 patched continuations reached EOS, and each hook modified one full-prompt prefill call while leaving two decode-token calls unchanged. This supports terminal self-attention as a necessary route *for this particular intervention's answer flip*, but restoring that output did not restore the original logit margin, so it is not a claim of complete mediation or a uniquely identified attention head. The donor-only asymmetry also limits sufficiency.

The [posthoc 16-head scan](block31_head_scan_results_v1.json), SHA-256 `60af7c0e5610c8c7c9e1b5b5b147434412185207c3724c048a262b3aa0203d2f`, patched each 128-dimensional pre-`o_proj` head-output slice at the terminal prompt position. No single donor head flipped a complete answer (**0/64**), and restoring no single clean head under the layer-30 word patch prevented its flip (**0/64**). Head 3 moved the first-token margin most in both engines/directions, but that is a score observation from a scanned fit-only pair, not a selected circuit. All-head controls reproduced the earlier full attention-output patches exactly; all 144 scanned continuations reached EOS. The [independent source and result audit](INDEPENDENT_REVIEW.md) checked the slice semantics, hashes, and hook calls.

A [posthoc prompt-only crossover](prompt_crossover_results_v1.json), SHA-256 `2fb908ae4bf990486e041530bfff7f12cf4ea40e3d9aecf7c8b90c48a8d713b7`, reused the original red/blue WAVs with the generic system prompt of the fresh-word transfer set. Word-position swaps flipped **4/4** at post-block 30, **1/4** at 31, and **0/4** at 32; terminal swaps flipped **0/4**, **1/4**, and **4/4** respectively. Thus the generic system instruction alone did not recreate the later 31→32 transfer-set boundary in these old WAVs. This is an after-results diagnostic, not another held-out test; lexical content, spoken wording, voice, and their interactions remain possible explanations.

## Evidence boundary and next test

This is one red/blue contrast, two synthetic voice models, three correlated layers, and a short one-word answer. The 12 v1 cases are not 12 independent lexical replications. Audio encoder patching and Qwen causal patching have [published prior art](PRIOR_ART_AND_VALIDATION.md). The result is a verified fit-only mechanism assay, **not** a new security vulnerability or a general claim about speech models.

The next gate is the [locked validation matrix](PRIOR_ART_AND_VALIDATION.md): unseen word pairs, phrasings, independent voices and a separate model architecture, with clean full-answer accuracy, matched perturbation controls, row-level stage maps, and analysis clustered by pair and voice. A security case then needs a protected decision and realistic attacker-controlled audio, with tools inert until the end-to-end behavior is justified.
