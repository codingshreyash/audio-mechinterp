# Independent source review: VoiceChat acoustic corpus builder

19 November 2026. **HOLD for a corrected stage-1 render or stage-2 variant corpus.** I reviewed exact commit `be1f35d` against the [V1 protocol](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V1.md), [V2 amendment](../VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V2.md), and [pre-output level correction](SOURCE_RENDER_LEVEL_AMENDMENT_V1.md). I did not edit the builder, rerender corrected source audio, load VoiceChat weights, or inspect any candidate answer.

Reviewed SHA-256: `corpus.py` `6e4efc4aeb1f0ffa916a28e6a4d91cd47c0219c49791f16bd2be72d8d27dce5b`; `spec.json` `3325f4f693461cf48cf994bfda486e81b2bc6535d0c855a3ac838601644d414d`; `test_corpus.py` `fe02c63e281769c8e06efa30c9c4b2b9cb2c95f4da460d330a37486d572c4a98`; level amendment `1ee563e409f2bf8fbb62a3fb23ef04ff719ab6260ff4346d91066f84914f92ea`. The preserved first-attempt `failure.json` hash is `3f61dad0119cf7243c20d50e87fb7deddf2aad38de15b6b521deeda684c1fa35`. Its eight completed rows and Kokoro A2 clipping failure remain separate from any corrected corpus.

## Blocking source-admission gap

The stage-2 public path treats `source_manifest.json`'s `renderer_identity.synthetic_test_only` as authoritative but does not compare it to the stage-1 `started.json` or validate every manifest row against the frozen source plan, script, phrase, voice, split, and original render provenance. In an isolated temporary fake-tone run, I changed only the completed manifest label to `synthetic_test_only=false`, changed A1's script to an unrelated sentence, recomputed that local manifest hash in the annotation template, and supplied otherwise valid fake timing fields. The **public** `make_variants` call (without `allow_synthetic_test_only`) returned `complete_48_wavs_no_human_qa_or_model_inference`, 48 rows, and `synthetic_test_only=false`. It had created variants from fake audio under a claimed real identity. This does not show real study contamination, but the current source cannot certify its own production-vs-test boundary.

Before GO, stage 2 should verify the immutable stage-1 start/complete identity and exact row population/content against `spec.json`, reconstruct or independently audit raw-to-converted-to-clean bytes, and require an externally pinned complete stage-1 manifest hash for the production path. A self-repinned local manifest and annotation file must not suffice to promote fake data. Add an adversarial fake test that edits the source label and/or exact A1 script and requires rejection before any `variant_manifest.json`.

## Blocking adjudication gap

The two annotators' window proposals are validated, but when they agree, the coordinator's final window need only be a valid equal-width pair; it need not equal the agreed proposals. In another isolated temporary fake-tone run, both annotators marked decision `[1200,2000)` and displaced `[6000,6800)`, while the final record changed them to `[3000,3800)` and `[7000,7800)` with a coordinator note. Stage 2 still returned a complete 48-WAV corpus. This violates the model-blind two-person timing rule and lets a final window shift without an independent dissent or adjudication basis.

Require final intervals to equal both agreeing proposals. If the final differs from either initial proposal, require a distinct third reviewer with recorded independent timing and a reason before accepting it. Preserve all initial proposals and the final choice. Add a regression test for consensus drift.

## Checks that passed

The six existing fake `unittest` cases passed; `py_compile` and `git diff --check` passed. The frozen scripts, 4 Piper development/8 Samantha-Kokoro locked population, level rule, 16-kHz conversion, five-second zero tail, one per-source Gaussian field, and no-clipping checks agree with the reviewed plan on source inspection. I ran the real TTS **identity preflight only**: pinned Piper/Kokoro package versions and model/voice hashes, `say` binary, Samantha voice line, and macOS build matched. No corrected render was started. The first unnormalized partial attempt's failure hash matches the level amendment. Human interval quality and source authenticity still require later human and byte-level reviews; fake tests do not certify them.

**Decision:** preserve the partial first attempt, correct these two admission paths, obtain an independent source rereview, and only then use a fresh exclusive run ID for corrected stage 1. This review reports no VoiceChat action, mechanistic effect, or security finding.
