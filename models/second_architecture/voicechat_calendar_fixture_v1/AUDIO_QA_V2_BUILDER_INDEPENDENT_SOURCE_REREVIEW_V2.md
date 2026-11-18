# VoiceChat QA v2 builder independent source rereview V2

18 November 2026. **GO for the source-only QA evidence builder at the exact
hashes below.** This clears the two defects in the immutable [V1 HOLD](AUDIO_QA_V2_BUILDER_INDEPENDENT_SOURCE_REVIEW_V1.md).
It does not certify actual human listening, a controller grant, an external
anchor, a live VoiceChat model run, or the planned calendar assay.

## Reviewed source and checks

The fix was committed as `7ea8c9f9c29893dc875d4c526cb214596fb7d8d4`.
The reviewed `audio_qa_v2_build.py` SHA-256 is
`61256c8543911b3002f7cdeef7aab71aa7f51b0d4ff9e4256b4c5ed52fa4946e`;
`test_audio_qa_v2_build.py` is
`78d339f64a98de2f991374cfd0b87e77608a5fa4865dcc38e21f82dc31926689`.
The unchanged source-only usage document is
`9dfedfeb4e581488b59ea44428e93e33fd09507f7a58fcf2efe4745e5bd53802`.
The prior HOLD review is unchanged at
`212ff87af2e02fea5ec1ac6f3966a6e42773739d19172111fb1bb274c2726943`.

I read the source diff and accepted v2 validator and ran all nine synthetic
builder tests: 9/9 passed. A complete synthetic 12-WAV cohort with 11 required
secondary reviews and two distinct synthetic signed C grants still passes
`check_audio_qa_v2`. The builder continues to copy human stage bytes without
rewriting them, derives secondary/tertiary triggers from the pinned ASR and
listener records, rejects missing/duplicate/conflicting stages, and does not
create a grant, external anchor, ledger, or model output.

I independently repeated both V1 negative probes:

1. Two correctly signed C grants with the same `grant_id` now raise
   `ValueError: C grant ID reused or missing` **before any final evidence is
   written**. The prepared evidence stays byte-identical; no `grants/` or
   `manifest.json` appears.
2. An injected exception during the second grant copy leaves the prepared
   evidence byte-identical with no `grants/` or `manifest.json`. A later
   finalization using the same valid grants succeeds.

The new staging and cleanup cover ordinary Python exceptions. They do not
promise crash-atomic installation across a process kill or power loss between
the final grants rename and manifest write; operators must inspect for partial
evidence after such an event. This is a source review, not evidence that a
human remained blind or that any claimed external storage is immutable.

Actual listening records are still absent. Before live VoiceChat inference,
the accepted gate still requires 12/12 eligible human-reviewed WAVs, the
required independent secondary and tertiary reviews, two externally signed C
grants, an externally anchored manifest, a separate pre-inference ledger, and
the pinned official-sample and stock-versus-adapter parity checks.
