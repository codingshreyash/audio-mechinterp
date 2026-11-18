# Independent VoiceChat parity source rereview — V3

18 November 2026. Reviewed corrective commit `605f118`, with
`official_adapter_parity.py` SHA-256
`c9bc38da67f41ca42c4635e10ad9bf8660f2d4745f86078615d034f462279046`,
against the [V1](OFFICIAL_ADAPTER_PARITY_INDEPENDENT_SOURCE_REVIEW_V1.md)
and [V2](OFFICIAL_ADAPTER_PARITY_INDEPENDENT_SOURCE_REREVIEW_V2.md) HOLDs and
the pinned NVIDIA helper. **Source-only verdict: GO for this bounded capture
and comparison utility. Live model parity remains untested.** I loaded no
VoiceChat weights and ran no real WAV, calendar tool, or model inference.

The new result gate requires nonempty text tokens, token lengths, and text in
both passes. For a replaying first pass, it selects function tokens with the
same `.get("tokens_function_pred", fallback)` semantics as the pinned helper
and requires the selected value to be nonempty. A no-call run may omit both
function-token fields. `tokens_audio` is compared exactly when present,
including presence across arms, but it is not treated as a mandatory model
output. The wrapped official and adapter pass-2 arms must capture nonempty
TTS codes; a complete six-field injection is required only when replay
occurs. The original official-wrapper restoration and fixed `1e-4` raw PCM
control remain in place.

I ran the five related unittest modules: **25/25 passed**. These include the
three V1 missing-evidence rejection tests, the two V2 valid-schema tests, a
no-function-stream no-call control, and the original fake official/adapter
tests. I also separately mutated the fake arms: an optional `tokens_audio`
presence mismatch, a function-token mismatch, a partial replay injection,
and an empty adapter TTS capture were each rejected with `ValueError`.

This GO is limited to source logic under fake outputs. It does not establish
that the actual checkpoint exposes the assumed nonempty text or TTS code
fields, nor that an official sample, 12-WAV human-QA gate, source/checkpoint
pin, reset model/RNG state, audited WAV and prompt origin, or durable stock
capture has been satisfied. The live parity protocol still requires those
steps and an independent review of actual raw captures before using a
VoiceChat output as model-behavior evidence. No security finding follows from
this source rereview.
