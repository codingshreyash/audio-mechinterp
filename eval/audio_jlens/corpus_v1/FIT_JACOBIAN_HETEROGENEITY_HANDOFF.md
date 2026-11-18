# FIT-only Jacobian heterogeneity assay: static handoff

**Status:** Protocol and code ready for independent static review; no MPS run or development/locked-test WAV or candidate model output was used. The frozen corpus spec/manifest include split metadata. This is a diagnostic after the frozen negative Stage A v1 result, not a repair or reclassification of it.

Frozen draft hashes:

| File | SHA-256 |
| --- | --- |
| `FIT_JACOBIAN_HETEROGENEITY_PROTOCOL_V1.md` | `69ea797ee956d16f78b014df18621f206d1279ebe978c96a005e041675e51ba9` |
| `fit_jacobian_heterogeneity_v1.py` | `b1b9e023394a517aabe4e56404ed83ba215c0ad8e8665a771aee3915f07df3e4` |
| `test_fit_jacobian_heterogeneity_v1.py` | `c7e8328c26a8c3bf3a2a2f5ec9a99c1b952ea1ccd8141d0b0f99d29b5837a59a` |

The runner reads only frozen map-fit FIT artifacts and their 48 audio cases. It pins literal SHA-256 values for the FIT lens, stage-1 checkpoint, fitter report, calibration raw/report, and V2 policy, plus the pinned spec/manifest/model/source revisions; it verifies donor deltas and row geometry. It computes 16 VJPs per case using the original retained-graph hook semantics with an added check before and after **each** VJP, saves one local `[6,16,width]` Jacobian per site/context, and checkpoints each context. A first-case smoke uses a separate prefix. A 30 GiB MPS driver cap and 20-minute per-invocation wall check apply to capture and score analysis; resume is bound to script/protocol/artifact hashes. The final CPU/MPS readout compares local, frozen 48-context average, and pair-excluded 44-context mean on the same measured FIT α=.025 uniform edit. It reports score/terminal errors, fixed pair/category/voice Frobenius-cosine strata both as raw uniform-edit derivatives and per-row-mean gradients, 12 pair summaries, and the reconstructed average-versus-frozen map gap.

CPU preflight passed: 4/4 toy tests (uniform-bin row-sum derivative versus centered finite difference with two rows per bin; leave-one-pair-out exclusion; fixed 1,128-pair similarity strata; fail-closed deadline) plus `--validate-only` and `py_compile`. The preflight loads no Qwen model weights. Estimated VJP work is 48×16 backward coordinates, comparable to the earlier full FIT lens run; result scoring adds lightweight norm/head evaluations. The assay retains the original all-FIT target basis, so its pair-held-out Jacobian means are conditional on that basis and should not be called fully independent held-out transport.

After review and GPU coordination, run a fresh-prefix `--smoke` first; inspect one-case hook counts, graph parity, matrix shapes/norms, memory and timing. Then run `--run` only if the smoke passes. Do not start either while another MPS job is active.
