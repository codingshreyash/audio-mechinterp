# VoiceChat QA v2 builder independent source review V1

18 November 2026. **HOLD for use on real listening records.** This review is
of source only. No real human listening record, controller grant, external
anchor, ledger, or VoiceChat model output was produced or accepted.

## Reviewed source

The three files were first committed in `bf36dd3e1493bcddadbba0d0214d5f231901e1ad`:

| File | SHA-256 |
| --- | --- |
| `audio_qa_v2_build.py` | `71ac2a55f70f1f86c341bcbd930096aa12395cc2a87b73840400e8ccf72eea22` |
| `test_audio_qa_v2_build.py` | `0ea82c13038f461594fbc5a4e0dcf7056b75e35811ca8a24e34a710b763ba28a` |
| `AUDIO_QA_V2_BUILDER_SOURCE_ONLY.md` | `9dfedfeb4e581488b59ea44428e93e33fd09507f7a58fcf2efe4745e5bd53802` |

I read the accepted v2 amendment and existing `audio_qa_v2.py` validator,
traced the builder's source and tests, and ran all seven builder tests. They
pass. A complete synthetic cohort built by the source passes the existing
validator with synthetic grants and ledger. The builder derives all 12 rows,
rechecks frozen WAV/ASR sources and copied listener stages, demands secondary
and tertiary records under the same trigger expressions as the validator,
and leaves the external anchor and ledger to a separate process.

## Blocking findings

1. `finalize_manifest` does not require unique C `grant_id` values. I supplied
   two correctly signed synthetic C grants for different WAV/receipt pairs,
   both with `grant_id="same-id"`. The builder emitted a manifest (SHA-256
   `f8940d6fbe1c21b9ba35f33ecc22e78d1d3f9bd2ca6d93f09498c8b1f9334f3c`),
   although `check_audio_qa_v2` explicitly rejects reused C grant IDs. The
   builder must reject this before copying grants or writing a manifest, with
   a regression test.
2. Final grant copying is not transactional. After complete verification, the
   builder creates `prepared/grants`, copies grants sequentially, then writes
   `manifest.json`. In a synthetic injected failure on the second copy, it
   left one copied grant and no manifest. A repeat invocation failed with
   `FileExistsError` because `prepared/grants` already existed. Finalization
   should leave the prepared human evidence untouched and retryable after a
   copy or write failure, while never replacing an existing valid manifest.

These are builder defects, not evidence that any human judgment or VoiceChat
behavior is wrong. They do not weaken the existing v2 validator; they prevent
this builder from being treated as a reviewed way to produce a valid, stable
manifest. After a new source commit and independent rereview, actual listening,
two independent signed C grants, an external immutable hash anchor, a
pre-inference ledger, and the separate live parity gates are still required.
