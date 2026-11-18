# Prospective source freeze for the old-FIT numerical audit

**18 November 2026. Proposed gate; not an execution receipt.** The independent CPU auditor is `voxtral_v1_fit_backward_numerical_audit_v1.py` SHA-256 `c619576f3daf188940c4fee67ecf798a2bc12ea3876b317673a12f964e916c5e`. Its synthetic test source SHA-256 is `238998077b489a38cd32a3ba2d21d18bd4c5812f4c0954fc4c65f4abb21c29a8`; 7/7 tests pass. The auditor pins runner `ae9d2620ad4e06d01ea01c920f1f8f4aee8c21764598bac343e383869c2940d2` and protocol `5778a4b84e535f7182a3daa2b973637a1c9127118ce2434ce82ba80fee467a82`.

Before either live arithmetic arm, obtain an independent source review and write an immutable external freeze receipt. The producer or launch gate should verify this receipt and all named file hashes **before model load**, then start only the reviewed runner. Do not derive the accepted auditor hash from a live run's output. A suggested JSON schema is:

```json
{
  "status": "frozen_old_fit_numerical_audit_v1",
  "auditor_sha256": "c619576f3daf188940c4fee67ecf798a2bc12ea3876b317673a12f964e916c5e",
  "audit_tests_sha256": "238998077b489a38cd32a3ba2d21d18bd4c5812f4c0954fc4c65f4abb21c29a8",
  "runner_sha256": "ae9d2620ad4e06d01ea01c920f1f8f4aee8c21764598bac343e383869c2940d2",
  "protocol_sha256": "5778a4b84e535f7182a3daa2b973637a1c9127118ce2434ce82ba80fee467a82",
  "independent_review_commit": "<review commit after approval>",
  "reviewed_utc": "<UTC timestamp>",
  "permitted_arms": ["bf16", "fp32"]
}
```

The auditor itself writes no files. Its JSON stdout receipt binds `auditor_sha256`, `runner_sha256`, `protocol_sha256`, `result_sha256`, and `details_sha256` after a completed attempt. An error receipt is audited separately and cannot be reported as a completed numerical result. This is an arithmetic/provenance audit on one old-FIT case; it does not authorize a v2 efficacy or security claim.
