# FIT-only uniform calibration and ridge baselines: independent saved-artifact review

**Verdict:** No provenance, arithmetic, or gate-invariant discrepancy found. This is a CPU-only review of saved FIT artifacts; no Qwen forward pass or development/test input was opened. The executable audit is [`audit_fit_uniform_artifacts.py`](audit_fit_uniform_artifacts.py).

The calibration report authenticates its raw tensor and 433-line memory log. The raw metadata binds to the current frozen protocol, script, V2 policy, rank-16 lens, stage-1 checkpoint, and FIT map report. The 48 checkpoint records equal the final raw records tensor by tensor. Baseline JSON authenticates its tensor artifact and binds to the same calibration raw/report, lens, protocol, and baseline script. All 48 unique case IDs match stage 1: **12 FIT pairs × four voice/direction contexts**. All donor-minus-recipient six-bin tensors, row sets, source shapes, and audio hashes match stage 1. Each site records one source and one target hook call, three zero-edit scores, and an effective edit norm matching `0.025 sqrt(Σ_b n_b ||δ_b||²)`.

The maximum clean residual norm/head versus full-forward logit difference is **3.72e-5**; saved source means match stage 1 exactly. The largest zero-edit V2 score departure from its unedited baseline is **2.39e-5**, below `1e-4`; recomputed site floors are both **1e-4**. All 433 memory samples are under the 30 GiB guard; peak MPS driver allocation is **18.593 GiB**. I recomputed projected map vectors and the saved capture/error/cosine diagnostics. Independently applying the pinned checkpoint's final norm and token-head rows to saved terminal states reproduced each saved map-predicted V2 score change within **5.8e-6** (CPU versus MPS).

| Site | Map V2 MAE | Zero MAE | Determinate sign matches |
| --- | ---: | ---: | ---: |
| Encoder31 | 0.0297417 | 0.0420943 | 46/48 |
| Connector | 0.0399701 | 0.0520030 | 46/48 |

The two wrong-sign contexts at **both** sites are `animal_06/fit_chosen/{kokoro,piper}/deer`, the previously disclosed `deer`/`dear` homophone item. They remain in all 48 denominators. These FIT metrics are calibration, not held-out transport evidence.

For each site and comparator, I rebuilt the saved global recipient-row-weighted or flattened six-bin input matrix and rank-16 projected terminal target from raw records. I recomputed all six γ candidates across 12 leave-one-pair-out folds (four related contexts held out together); saved pair MSEs match. Training-only RMS and fold-specific `n_train=44` were used in the independent computation, followed by a 48-context refit. Chosen γ is **0.1** for global ridge and **0.0001** for six-bin ridge at both sites; saved full-refit coefficients, scales, and reported CV RMSEs agree. The zero comparator is zero terminal change. The ridge model is a CPU fit to measured FIT responses and has not yet been evaluated on development or locked test.
