# Independent result audit: Qwen FIT Jacobian heterogeneity v1

**18 November 2026.** Read-only CPU audit of the completed 48-context FIT result, checkpoint, report, and memory log with [the auditor](audit_fit_jacobian_heterogeneity_v1.py). No Qwen MPS inference or development/locked-test input, activation, or answer inspection. This diagnostic cannot relabel the negative Stage A v1 result.

## Provenance and completeness

The result SHA-256 is `336a2589743c06a2e5be71f3fce694e7a01c78450dc81dc07c6368a01bc58903`; checkpoint `f9d710708f06832133dedc0ee0f9776e13c78c4626e2596c0292b765139d42b1`; memory log `fadeb8d678724db4f4b83de9adbe24b5f7021d13f2fc3fddb9c1813a19ef5309`. The report binds all three. The checkpoint, result, and report metadata match the frozen script/protocol and FIT lens/stage-1/calibration hashes. All **48** ordered case IDs/audio hashes match the FIT source artifacts: 12 lexical pairs × four contexts, with no dropped case. Every record has **16** finite nonnegative VJP times (**768** total), stage-1-consistent six-bin row counts and source shapes, and grad/no-grad terminal gap ≤0.001. Each encoder `[6,16,1280]` and connector `[6,16,2048]` FP32 matrix is finite and nonzero; checkpoint and final result matrices match exactly.

The saved 48-context mean reconstructs each frozen corpus Jacobian exactly at the stored precision (`max_abs=0`). The auditor independently reconstructed each leave-one-pair-out mean from the other **44** matrices, predicted terminal changes and full/projected terminal errors for local, frozen-average, and pair-excluded maps, all 48 score-error subtractions and pair/global summaries, and all **1,128** cosine pairs at both raw uniform-edit and per-row-mean-gradient scales. The fixed strata have counts **24/48/192/864** by pair/category and **552/576** by same/cross voice, exactly as preregistered. The saved scalar local/LOPO prefix-score deltas require the pinned final norm/head weights for an independent numeric forward; this audit checks their error arithmetic, frozen-calibration cross-check, and pinned generating code, not a second logit computation.

The memory log has every coordinate **0–15** and checkpoint for each of 48 contexts, plus model-load and final-analysis phases. Peak driver allocation was **22,819,700,736 bytes**, below the **32,212,254,720-byte** cap. The longest logged post-load invocation segment was **166.50 s**, below the 1,200-second operation-boundary deadline. Pre-load time and any single uninterruptible kernel cannot be bounded from this log alone; the frozen runner checks the deadline around operations.

## What the FIT comparison says

| Site | Map | V2 prefix-score MAE | Projected-terminal L2 mean | Full-terminal L2 mean |
| --- | --- | ---: | ---: | ---: |
| Encoder31 | Local in-sample oracle | 0.01758 | 0.0352 | 1.3559 |
| Encoder31 | Frozen 48-context mean | 0.02974 | 0.5304 | 1.4644 |
| Encoder31 | Leave-one-pair-out mean | 0.03101 | 0.5469 | 1.4711 |
| Connector | Local in-sample oracle | 0.02523 | 0.0568 | 1.8006 |
| Connector | Frozen 48-context mean | 0.03997 | 0.6957 | 1.9409 |
| Connector | Leave-one-pair-out mean | 0.04168 | 0.7167 | 1.9491 |

The local derivative predicts each case's **rank-16 projected** terminal change much more closely than the corpus mean. It also improves the FIT-only prefix-score error. But this is an **in-sample oracle** computed on the recipient itself, not a reusable estimator. The frozen mean includes the evaluated pair, so its slight advantage over pair-excluded mean is expected. The mean fraction of measured terminal-change energy in the fixed rank-16 basis is only about **0.222** at either site; full-terminal errors remain large. Thus the small projected error does not imply that the full hidden-state change is explained. The pair-excluded mean still uses a basis fitted on all 48 FIT contexts, so it is not a fully held-out representation test. Raw-matrix cosine distributions overlap substantially across pair/category/voice strata; the similarity summary supports heterogeneity but does not identify a unique voice or category mechanism.

**Verdict:** No provenance, arithmetic, geometry, or resource blocker found in the completed FIT diagnostic. It supplies a plausible explanation for why a fixed corpus-mean Jacobian underperformed the local perturbation while leaving Stage A's negative development result intact. It does not establish a complete-answer effect, cross-model transfer, or a security vulnerability.
