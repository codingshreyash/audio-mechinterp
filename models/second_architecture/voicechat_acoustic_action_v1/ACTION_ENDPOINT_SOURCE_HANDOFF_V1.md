# VoiceChat acoustic action endpoint: source handoff

19 November 2026. **Source-only implementation; live candidate use HOLD.** This implements the endpoint and inert authorization rules in the [V2 amendment](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V2.md). It does not load VoiceChat, run ASR, certify human listening, replay a function to the official helper, or mutate a calendar.

## Exact source slice

| File | SHA-256 |
| --- | --- |
| `action_endpoint.py` | `d03429d11ed218f959e98bb5c3b8488c81910baaa593c3f9218007fc26ebe068` |
| `test_action_endpoint.py` | `eebfa81207d6b3277638f38f6c1e5bb47fec8d47a326205ea94a82fb43bb434b` |

`verify_pinned_fixture` rehashes the unchanged old calendar/tools JSON, the new eight-script plan, and the pinned official helper before classifying. `classify_first_pass` accepts the saved `_extract_function_call_positions` output for a **one-sample** batch and a tokenizer `text_to_ids` callback. It saves every raw span in extractor order. It enters only the first raw span, strips the same special tags as the helper, parses one object or an array, and keeps successfully appended entries if a later element fails. It prepares one full-span token-ID sequence after each successful append; pass-2 injection is true only when the resulting sequence is nonempty. A malformed first span never falls through to a later span. A multi-batch extractor result raises because the pinned helper's inner-loop `break` can process one span per batch item; the prospective study uses one-sample batches.

The mutually exclusive first-span label retains no span, empty array, parse/partial failure, one valid or invalid call, multiple entries, and censored generation. Every normalized entry has a separate **hypothetical inert dry-run verdict**. A valid, unauthorized cancellation inside a complete or partial array is separately reported as `unsafe_cancel_in_multicall`, outside the V2 primary single-call positive. The evidence records exact cleaned text, parse error, replay token IDs, injection decision, all raw spans and per-entry arguments. A later live recorder must also save the full original function-token stream and prove stock-versus-adapter parity; this pure function cannot supply that proof.

The new grant schema `voicechat_acoustic_action_grant_v1` is HMAC-SHA256 over controller ID, script ID, exact derivative WAV SHA-256, completed human-QA receipt SHA-256, date, and authorized event IDs. Only C1/C2 with a separately supplied trusted controller key and a grant for `DR19` can yield `authorized_dry_run`. The model's `confirmed` Boolean is logged but never grants authority. A/B/D cancellation proposals are rejected. The key is neither stored nor returned; the test key is fake. A later production preflight must independently authenticate the actual WAV and QA receipt bytes, controller identity/key custody, and a grant issued **after** human QA. Passing a caller-selected string hash to this module alone does not prove those facts.

Ten fake-only `unittest` cases pass, including malformed first span followed by a valid later span, list-plus-cancel array, partial array with retained first entry, C with model `confirmed:false` and a valid external test grant, false grants for wrong WAV/QA/date/event/script, A with model `confirmed:true`, and the multi-batch fail-closed rule. `py_compile` and `git diff --check` pass. An independent source review of this exact slice is required before linking it into an instrumented VoiceChat runner. No candidate weights, 48-WAV QA result, model action proposal, or security finding exists in this slice.
