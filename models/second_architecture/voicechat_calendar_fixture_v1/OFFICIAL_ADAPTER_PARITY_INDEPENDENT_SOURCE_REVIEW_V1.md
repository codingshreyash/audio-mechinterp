# Independent source review of VoiceChat stock/adapter parity V1

18 November 2026. I reviewed implementation commit `546757a` and source SHA-256
`4292cc50ec9f5ea7380ff789049ee6ca6ccc249526f8367935a159f4e4eb429c`
against `VOICECHAT_OFFICIAL_HELPER_STATIC_PARITY_V1.md` and the pinned official
helper. **Verdict: HOLD for using this comparator to certify live parity.** This
is a source-only review: no VoiceChat weights, real inference, listening record,
calendar action, or security outcome was produced.

The new capture wrapper calls the original official `run_offline_inference`
with unchanged arguments for two passes, clones arguments and results, checks
pass order, and restores the global reference in `finally`. The comparator
checks common input values, available discrete outputs, parser calls, replay
steps and response IDs, available injected arguments, captured TTS codes, and
raw pass-2 audio against a fixed maximum absolute PCM tolerance of `1e-4`.
It requires stock repeat and no-TTS-wrapper controls. The 19 related fake
source tests passed. These properties are useful, but three omissions let a
comparison return `source_comparison_passed_pending_live_provenance_review`
when evidence required by the parity protocol has been removed from every arm.

1. `_compare_discrete` treats a field absent from both dictionaries as equal.
   Removing `tokens_text`, `tokens_len`, `text`, and `tokens_audio` from every
   captured result still returns a pass. A passing result therefore need not
   contain the text or token sequence the method purports to compare. Require
   the pass-specific core fields and their expected shape/value presence;
   preserve optional function fields only where the official model really
   permits them to be absent.
2. `_compare_injection` likewise accepts absence from both dictionaries.
   With a parsed and replayed tool call, removing all six `function_*`
   arguments from the official and adapter pass-2 call records still returns
   a pass. Require the complete injection set when `replay` exists and require
   its absence for a no-call pass.
3. The wrapped stock arms and adapter may all report empty TTS-code lists while
   pass 2 contains generated audio; the comparator still returns a pass. For
   this checkpoint, establish that the instrumentation observed the expected
   TTS path and at least one code, or withhold a codec-identity conclusion.

The exact fake-output reproductions are preserved as three `expectedFailure`
tests in `test_official_adapter_parity_hold_v1.py`. Running that module against
`546757a` reports `Ran 3 tests ... OK (expected failures=3)`. Remove those
decorators only when a corrected comparator rejects all three cases. The
original 19 fake tests still pass, which shows they did not cover these
missing-evidence states.

This low-level utility also does not itself establish 12/12 human QA,
checkpoint/helper identity, a real official sample run, identical model and
RNG resets, exact audited WAV/prompt origin, or immutable saved stock captures.
Those are declared external prerequisites in its docstring; this review does
not treat their absence as a new bug in the utility. An independently reviewed
QA-gated live harness and saved raw outputs are still needed before any model
behavior or tool-call finding can be attributed to VoiceChat.
