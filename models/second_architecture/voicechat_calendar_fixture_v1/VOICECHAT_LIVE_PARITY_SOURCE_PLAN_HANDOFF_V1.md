# VoiceChat parity source-plan handoff

18 November 2026. `live_parity_source_plan_v1.py` is a **no-model plan builder** under the [prospective orchestration protocol](VOICECHAT_LIVE_PARITY_ORCHESTRATOR_PROSPECTIVE_V1.md) and its [independent conceptual GO](VOICECHAT_LIVE_PARITY_ORCHESTRATOR_CONCEPTUAL_REREVIEW_V2.md). It has not loaded VoiceChat, run NVIDIA's sample, emitted a model call or constructed human listening evidence. Its CPU tests use explicit synthetic dependency injection and mark their output `synthetic_test_only_no_model_work`.

The production CLI requires the real v2 12-clip human QA manifest, ledger and fixed case-B/Piper receipt, an external controller key, the exact final WAV, pinned NeMo checkout, checkpoint and Nano cache on an H100/A100 host. It invokes the existing `ready` check followed by the `preload` source/checkpoint/Nano/GPU check, verifies phase chronology and matching staged identity, and rehashes its bound inputs after each phase. It then writes an exclusive, hash-bound **plan only**. The plan fixes official sample → adapter discovery → stock A → stock B → stock without TTS wrapper, the selected case/voice/WAV, seed, response-selection rule and required comparisons. It records both preflight receipts, including parsed and exact UTF-8 outputs and hashes. This CLI does not accept test-injected checkers.

`fixed_response_bytes` is a pure rule for the later launcher. It requires an inert adapter trace with exactly two pass labels; a no-call trace must have zero function-injection fields and returns the protocol's exact 86-byte placeholder. A parsed call requires all six second-pass injection fields and returns the selected inert response with the helper's default insertion-order JSON encoding. It never chooses a response using observed parity or answer quality.

Before any VoiceChat model load, a separately reviewed **live launcher** must still:

1. Revalidate external human/anchor authenticity and all plan-bound bytes; obtain independent source review and an external executable freeze/review.
2. Run the unchanged official sample successfully and preserve its stdout, generated files and hashes.
3. Use a fresh checkpoint-loading process and identical seed/settings for each of the four fixed fixture arms; capture the adapter's raw pass 1 and pass 2, official stock A/B and no-wrapper outputs and every exact input/replay argument.
4. Enforce frozen wall-time, GPU-memory and output caps, no retry after an arm starts, and hash every raw artifact. Run the independently source-approved `compare_stock_adapter` and arrange independent saved-result review.

The local QA validator checks record consistency, not that people listened or that the external store is immutable. The controller's external attestation remains necessary. A future comparator pass would establish bounded instrumentation identity only; it would not establish a model security issue.

Source-only tests:

```sh
python3 -m unittest models.second_architecture.voicechat_calendar_fixture_v1.test_live_parity_source_plan_v1 models.second_architecture.voicechat_calendar_fixture_v1.test_live_parity_source_plan_review_hold_v1 -v
```
