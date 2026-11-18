# CUDA old-FIT CPU auditor: canonical independent source review V1

18 November 2026. This review concerns saved-data auditor source only. It does not authorize a CUDA worker, load a checkpoint, or establish a model result.

Decision: GO for CUDA CPU auditor source only; live model HOLD until external freeze.

## Reviewed identity and review history

- Corrected auditor commit: `d62bffeaa5341fe0b933d98b120d9a9bb799e729`.
- Corrected auditor source SHA-256: `9c0fde688fbd195da70a83d8db9e4ca723eb7209a4647c96c596c35376c240b6`.
- Original auditor review HOLD commit: `6fab9cd`, recorded in `VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_AUDITOR_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md`.
- Corrected auditor rereview GO commit: `f5c7e3b`, recorded in `VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_AUDITOR_INDEPENDENT_SOURCE_REREVIEW_GO_V1.md`.
- Corrected original tests SHA-256: `b1c8dd4ca887282eee20e10c27d4cd7524f8f6cfa2f02a14144002d599f306b0`; independent negative tests SHA-256: `d51d729d00579bb81fb8baa20167ac6947ce83b8f5967d71343adfd26fb73067`.

## Independent verification

From `models/second_architecture`, I reran `./.venv/bin/python -m unittest test_voxtral_old_fit_grad_leaf_cuda_audit_v1 test_voxtral_old_fit_grad_leaf_cuda_audit_independent_review_v1 -v` on 18 November 2026. All **13 CPU synthetic tests passed**. The tests cover the two-case, 66-planned-forward, four-cell denominator; 96 required raw-byte direct/full comparisons; signed-zero mismatch; source/passive, schedule and hook tampering; saved receipt hashes; and rejection of the previously accepted forged runner/checkpoint/environment identities and zero-parameter layout. They also check rejection after a pinned environment receipt changes or runtime GPU name disagrees with it.

The production CLI requires an externally expected freeze SHA-256 and verifies the freeze and environment receipt bytes before auditing case data. The corrected auditor compares the worker's recorded freeze, runner, checkpoint, and environment hashes with those trusted pins, and checks the ordered model parameter structure against the independently audited prior MPS layout digest. The synthetic compact layout is available only through an explicit test fixture path, not a CLI option.

The expected freeze digest must originate from the independently reviewed freeze decision. This source GO does not replace reviews of the runner, gate, environment receipt, or executable freeze. Saved-data checks cannot attest that the model ran or prove an audio mechanism or security effect.
