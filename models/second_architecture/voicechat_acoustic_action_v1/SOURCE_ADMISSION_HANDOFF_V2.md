# VoiceChat acoustic source admission: correction handoff

19 November 2026. This source-only correction addresses the two blocking findings in [the independent V1 HOLD](CORPUS_BUILDER_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md). It does not rerun the corrected TTS render or authorize VoiceChat inference. The first 8/12 partial render at `runs/sources_20261119_0129` remains untouched.

## Stage-2 production source binding

The public `make-variants` path now reads the complete stage-1 `source_manifest.json` and `started.json` together. The start receipt must be exactly the same render identity, versions, source/spec/protocol hashes, and renderer identity as the completed manifest, except for `status` and the completed rows. The complete manifest must match the **current exact reviewed builder hash** and the 12-row frozen script/voice/split/phrase/path population in [`spec.json`](spec.json). Every raw, converted, and clean WAV is rehashed; conversion is reconstructed from the raw samples using the frozen resampler and uniform level rule; the reconstructed PCM must equal the saved converted PCM, and clean PCM must be that sequence plus 80,000 zeros. Production renderer identity and per-row engine command must match the pinned Piper Amy, Samantha, or Kokoro Bella configuration.

An additional **external, exact stage-1 manifest pin** is required for production stage 2. Two fixed files must exist in this directory, be committed in Git, and be clean relative to `HEAD`:

1. `SOURCE_RENDERED_INDEPENDENT_REVIEW_GO_V1.md` must contain an independent `**Decision: GO` and the literal SHA-256 hashes of the complete `source_manifest.json` and `started.json` after a real 12/12 source byte audit.
2. `SOURCE_RENDERED_INDEPENDENT_ADMISSION_GO_V1.json` must be a separate committed receipt with the exact fields below. Its `review_sha256` binds the review file. No local annotation file, completed manifest, or caller argument can replace this receipt.

```json
{
  "schema": "voicechat_acoustic_source_admission_go_v1",
  "status": "independent_source_byte_audit_go",
  "source_dir": "/absolute/exact/source-run-id",
  "source_manifest_sha256": "64-lowercase-hex",
  "started_sha256": "64-lowercase-hex",
  "source_spec_sha256": "3325f4f693461cf48cf994bfda486e81b2bc6535d0c855a3ac838601644d414d",
  "source_builder_sha256": "exact-reviewed-current-builder-sha256",
  "review_sha256": "sha256-of-committed-independent-review-file"
}
```

The reviewer must calculate these values from the actual final files and review the full 12/12 inventory. The receipt's absolute path and SHA-256 protect against a local self-repin of only `source_manifest.json` and annotations. Git commitment and hashes do not prove reviewer identity by themselves; the independent review process supplies that assurance. The receipt cannot exist before the corrected source run, so **production stage 2 remains HOLD** now. A future change to the builder or spec after the source run also causes HOLD and needs a new versioned admission.

Fake tests can call the internal Python function with `allow_synthetic_test_only=True`, but only when both the immutable start and complete records still say `synthetic_test_only`. Their variant manifest retains `synthetic_test_only=true`. The public CLI exposes no such option. A regression for the prior exploit changes A1's script and both local identities, then locally rehashes the annotation template; it now rejects the frozen script. A second fake-tone self-repin with the exact scripts/commands rejects because the external review/receipt is absent.

## Timing adjudication correction

If the two timing annotators agree, the final decision and displaced sample windows must equal their agreed windows. If the proposals disagree or the final differs from either proposal, a distinct third reviewer must record independent `third_annotation` windows, and `adjudication.by` must equal that third reviewer's ID. The final must equal one recorded proposal. A previous fake test that moved both agreed windows by coordinator fiat now fails before any completed `variant_manifest.json`.

After independent source rereview, a new exclusive corrected TTS stage-1 attempt can be considered. The corrected render still needs its own 12/12 byte audit, actual human timing annotations, 48-WAV listening QA, and later model gates. This source correction is no action or security result.
