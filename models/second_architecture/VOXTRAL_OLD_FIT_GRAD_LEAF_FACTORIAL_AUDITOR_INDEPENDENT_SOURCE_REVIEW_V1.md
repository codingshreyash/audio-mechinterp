# Independent source review: grad-context × leaf-flag factorial CPU auditor

Reviewed 2026-11-18 UTC, before an external freeze or any V2 model attempt.

**Decision: GO for grad-leaf factorial CPU auditor source only; live model HOLD until external freeze.**

Reviewed committed source `1c48480c728e925f58d0e2383e092cda0ffe4cc6`:

| File | SHA-256 |
| --- | --- |
| `voxtral_old_fit_grad_leaf_factorial_audit_v1.py` | `653d081e903fb8c87bbfd32c92867655c14d37d2e9154ce664cb7543ce240259` |
| `test_voxtral_old_fit_grad_leaf_factorial_audit_v1.py` | `2e86665177b3962075edb57e59cc2d91e29dd093d08eeca8fcc7ca2035a8cbc3` |

I read the auditor and its inherited, previously frozen V1 tensor helpers. I reran its six CPU tests with the pinned local Python environment; all six passed. I also independently injected a NaN into a saved LM state and confirmed rejection before metric computation.

The auditor requires exactly 16 full and 16 direct calls in the registered order, fixed BF16 source and batch tensor hashes, correct tensor shapes and finite values, projector-to-LM placeholder parity, and bitwise-fixed direct LM inputs. It rehashes every retained tensor and checks the content-addressed sidecar before loading it. It computes all four repeat pairings for each of six condition contrasts, within-condition spreads, separate first bitwise and resolved differences, and the audio/non-audio LM-input partitions. The factor flags are checked against condition labels and preserved in the report. Symlink, malformed sidecar, changed schedule, and file-size guards are present. The output path must be a new file in the attempt directory.

The readout is a saved-data audit. It cannot prove that a model forward occurred or that a reported runtime flag was honestly captured; the separately reviewed launcher and source provenance are needed for those claims. Two repeats give only an observed resolution floor, and a resolved contrast is not a causal or security result. I did not load model weights or run a V2 diagnostic.
