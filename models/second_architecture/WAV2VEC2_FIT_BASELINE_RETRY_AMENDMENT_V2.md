# Wav2Vec2 FIT baseline: prospective retry amendment V2

This amendment preserves the original 16-case clean-baseline and eight-trace
protocol in `WAV2VEC2_FIT_BASELINE_STAGE_CAPTURE_PROSPECTIVE_V1.md`. It applies
only to a fresh retry after the first attempt's source failure. It does not
authorize a causal patch, a held-out assay, a changed input, or reuse of a
partial result as a successful baseline.

## Historical attempt

The first attempt at
`models/second_architecture/runs/wav2vec2_fit_baseline_stage_capture_v1/20261118T2345Z`
has `records.jsonl` SHA-256
`9625b3596da2ce7041c1f8dc394cf226734d33c0168102f2dd17889a0c8391f4`.
It contains a preflight row and a failure row, with no capture sidecars. The
first stock-mode forward completed and produced CTC frame IDs. The runner then
raised `NameError` while selecting the other word because `target` was read
inside a generator before assignment. The failure row preserves the computed
frame IDs and hook counts. The attempt has no scientific baseline outcome.
Its frozen preflight receipt remains
`WAV2VEC2_FIT_BASELINE_PREFLIGHT_V1.json`, SHA-256
`bf6db18d379d649905c7cfb37ebad572bd5de0d773ec479e33e69f67edfcd30b`.

## Frozen correction and load gate

The retry runner assigns `target = case.word` before selecting the alternative
word. A no-checkpoint fake execution must persist the first baseline row, so
the original failure path is exercised before any retry.

The pinned checkpoint omits `wav2vec2.masked_spec_embed`. The installed pinned
source constructs this parameter because `mask_time_prob` is 0.05, but it only
uses it for a supplied time mask or generated training masks. This assay calls
the model in evaluation mode with no mask indices; `mask_feature_prob` is 0.0.
The no-weight preflight checks this pinned configuration and checkpoint key
inventory. The retry's model-load gate must report exactly this missing key,
with no unexpected keys, mismatched keys, or load errors. Completion records
that load report. A different report fails the attempt before a forward.

## Retry boundary

Freeze a new `WAV2VEC2_FIT_BASELINE_PREFLIGHT_RETRY_V2.json` after committing the
corrected runner and fake test. Independently review the corrected source and
new receipt before loading weights. Execute into a new timestamped directory;
never overwrite or append to the historical attempt. Save and independently
audit all 16 baselines, eight traces, and the final completion row before
interpreting any result. The same eight FIT WAVs, both normalization modes,
stage definitions, budget caps, and complete CTC/eligibility rule apply.
