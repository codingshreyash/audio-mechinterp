# Independent VoiceChat parity source rereview — V2

18 November 2026. Reviewed corrective commit `2e17bd7`, whose
`official_adapter_parity.py` SHA-256 is
`80618fd3266a1987d0f95e130f5f8028ce51fdc3e06ee477e0bfc836434fcc1d`,
against the [V1 HOLD](OFFICIAL_ADAPTER_PARITY_INDEPENDENT_SOURCE_REVIEW_V1.md),
the pinned NVIDIA helper, and the source-only parity protocol. **Verdict: HOLD
for live parity use, with a narrower schema correction required.** No model
weights, real inference, listening, or calendar action was used.

The V1 missing-evidence findings are fixed. The comparator now requires core
result fields, all six pass-2 injection arguments when replay occurs, no
injection for no-call runs, and nonempty captured pass-2 TTS codes. The three
preserved V1 negative tests now pass without `expectedFailure`; all **22**
related fake source tests pass. This closes the originally reproduced
vacuous-comparison paths.

The new `REQUIRED_RESULT_KEYS` is too strict for the pinned helper's own
contract. The official function selects
`result1.get("tokens_function_pred", result1.get("tokens_function", None))`,
so it supports a result with either function-token field. The comparator
instead requires **both** keys, each nonempty, in **both** passes. It also
requires `tokens_audio` even though the pinned helper does not reference that
key; the independently captured TTS codes are the appropriate required
codec evidence. A valid fallback-only result or a result without the optional
`tokens_audio` key is therefore rejected before any comparison. This may
create a false failure on a real checkpoint, and no live result schema has
yet been observed to justify the stronger requirement.

The two exact fake-output reproductions are preserved as `expectedFailure`
tests in `test_official_adapter_parity_schema_hold_v1.py`. Against `2e17bd7`
they report `Ran 2 tests ... OK (expected failures=2)`. A correction should
require at least one nonempty function-token field and exact per-key presence
and values across arms; keep the pinned helper's first-pass fallback order.
Compare `tokens_audio` exactly **when present**, without claiming it was
present when absent. Retain the V1 fixes for mandatory text/length, replay
arguments, raw pass-2 audio/length and nonempty observed TTS codes.

This is still a low-level source utility. Human QA, checkpoint and prompt
provenance, official-sample run, model/RNG resets, and immutable stock capture
storage remain separate live gates even after a source GO.
