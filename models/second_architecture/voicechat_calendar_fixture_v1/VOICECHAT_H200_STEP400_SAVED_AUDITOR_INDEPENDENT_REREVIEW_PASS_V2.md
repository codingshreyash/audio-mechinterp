# Independent saved-auditor rereview: source-only PASS

19 November 2026. Supersedes the saved-auditor [V1 HOLD](VOICECHAT_H200_STEP400_SAVED_AUDITOR_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md)
for commit `202b3d4`. Rereview target: saved-auditor fix commit `cb8a52a`,
against the actual probe core, worker receipt, parent supervisor schema, and
the V1/V2 diagnostic protocol. **PASS for this source-only saved-evidence
auditor constituent.** No model or GPU was used.

## Independent verification

I reran the original HOLD mutations after updating and resealing the ledger
hash and parent inventory. All now return `NO_GO`:

| Previously false PASS | Rereview result |
| --- | --- |
| 100-second NVML query inside an approximately 0.001-second checkpoint lag | `NO_GO: checkpoint interval or telemetry outside parent coverage` |
| Parent elapsed 0.001 seconds, shorter than the 0.36-second poll span | `NO_GO: parent elapsed shorter than GPU polling span` |
| Checkpoint CUDA allocation above 130,000 MiB, with ordered allocator fields | `NO_GO: observed CUDA peak exceeds GPU cap` |
| Carried CUDA storage of 1 TiB while PyTorch allocation is 100 bytes | `NO_GO: enumerated CUDA storage exceeds PyTorch allocated bytes` |
| Float step label `100.0` in place of integer `100` | `NO_GO: checkpoint order incomplete` |

The revised code checks the baseline NVML query duration as well, bounds
every baseline/checkpoint CUDA peak by the policy cap, requires exact integer
step and `T` fields, and compares enumerated CUDA storage with live PyTorch
allocation. The parent result must still be clean and present: a child-only
ledger/receipt cannot pass because the six-file census and parent schema/hash
checks are mandatory.

The four-module no-model suite passes with `ResourceWarning` treated as an
error: **45 tests** across the probe core, worker, parent supervisor, and
saved auditor. The added fake-process integration exercises the actual probe
wrapper at 400 ordered STT/TTS steps, actual worker completion serialization,
actual parent polling/receipt, and saved-audit acceptance. Its model, CUDA,
and NVML are mocks; it demonstrates schema compatibility, not runtime
behavior or identity attestation.

## Scope and remaining admission boundary

The auditor establishes **internal consistency of a saved, parent-complete
diagnostic bundle** against an independently supplied expected identity. It
cannot authenticate that the binary/checkout/host actually produced the
bytes, that a worker boolean proves no second pass, or that unsampled GPU
peaks did not occur. Query timestamps and CUDA/NVML measurements are not
perfectly simultaneous; the auditor reports the lag without claiming exact
cross-counter equality.

This PASS does not lift the parent supervisor's executable HOLD or authorize
a new H200 attempt. The exact launcher, clean checkout/ignored-file census,
Nano metadata, package and native-runtime pins, one-shot host freeze,
authenticated receipts, and live poll cadence require their own review and
GO. The stopped H200 was not restarted.
