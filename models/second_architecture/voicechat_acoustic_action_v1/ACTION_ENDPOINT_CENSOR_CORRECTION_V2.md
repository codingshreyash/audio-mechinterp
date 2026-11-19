# VoiceChat action endpoint: censored no-span correction

19 November 2026. **Source-only correction; live use remains HOLD.** This addresses the single blocking case in the [independent V1 HOLD](ACTION_ENDPOINT_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md). The prior implementation chose `no_span` before checking whether a generation was censored. An incomplete generation with no extracted span could therefore look like a completed no-call turn.

The label now gives `censored` precedence for every incomplete generation, including an empty span list and a malformed first span. The output also records `generation_censored` explicitly. The raw-span list, parse error, replay-token decision and response step retain their original values. A complete empty result remains `no_span`; a censored empty result is `censored`, with no injection and response step zero. The caller must separately retain the full original function-token stream and the actual timeout/stop reason; this pure source slice receives only the censor Boolean and extracted positions.

| Corrected file | SHA-256 |
| --- | --- |
| `action_endpoint.py` | `f125df5b7727546024af4d1dbcaefa06ca557b4be47f449a33038ab20c10589f` |
| `test_action_endpoint.py` | `5e2bfd2a5e37ebb3510be9e482b30f14eb5e07239b939e1de3e136a7c326763a` |

Eleven fake-only tests pass, including the new censored/empty versus complete/empty and censored/malformed cases. `py_compile` and `git diff --check` pass. No VoiceChat weights, candidate audio, human-QA record, grant or real calendar action were used. An independent source rereview must approve these exact bytes before live integration.
