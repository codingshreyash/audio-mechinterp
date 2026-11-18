# Independent review: fit-only precision diagnostics

I audited the saved [BF16 directional finite differences](fit_fd_diagnostic_v1_results.json) (SHA-256 `bb5682d328057a18dbe18c59b1e660cd3facf5c570cb93c88ba161f756491b99`) and [FP32 rerun](fit_fp32_diagnostic_v1_results.json) (SHA-256 `5551d7638f8b052ca941f86bcd1773fd0cfd7636f883513050f34c9578c42983`), their [shared scoring code](fit_fd_diagnostic_v1.py), [FP32 driver](fit_fp32_diagnostic_v1.py), and both memory logs. I did not run Qwen/MPS or inspect the locked test. The recorded script, fit-gate, split, fit-manifest, model, and Transformers revision hashes match current sources. The saved source/donor WAV hashes match the original Piper red/blue files. CPU diagnostic tests pass (one shared-path test, two FP32-driver tests).

Both scripts use the original Piper red→blue fit pair at connector and encoder layer 15. The grad-mode patch path creates the same detached, requires-grad audio-site leaf used for the VJP, then replaces the selected rows; the no-grad path uses the prior patch scorer. Each path takes three zero-edit score repeats before the five alpha values. All repeated zero contrasts are exactly stable within a path. All four site/dtype results record zero effective edit and zero finite difference at alpha zero. I recomputed each stored contrast as `log P(Blue,EOT) − log P(Red,EOT)` and each finite difference against its **own path's** three-repeat zero mean; all agree with the saved values. The MPS memory logs have 21 records each, no over-limit entry, and peak driver allocations of 10.12 GiB (BF16) and 19.98 GiB (FP32), below the 30 GiB limit. The logs are separate mutable sidecars; their hashes are not embedded in the result JSON.

The table reports the error against the **matched grad-mode** finite difference, with relative error defined as `|prediction − grad FD| / |grad FD|`. Relative error is unstable when a measured change is close to zero, so the actual values are shown alongside it. No relative error is assigned to alpha zero.

| Dtype | Site | α | VJP prediction | Grad-path FD | No-grad FD | Abs. error vs grad FD | Relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | Connector | .01 | +.03429 | +.00814 | +.11835 | .02615 | 321% |
| BF16 | Connector | .025 | +.08131 | +.16717 | +.18047 | .08586 | 51% |
| BF16 | Connector | .05 | +.16758 | +.24985 | +.24187 | .08227 | 33% |
| BF16 | Connector | .10 | +.33004 | +.35387 | +.55507 | .02383 | 7% |
| BF16 | Encoder 15 | .01 | +.00252 | −.06245 | −.00627 | .06496 | 104% |
| BF16 | Encoder 15 | .025 | +.01377 | +.03333 | +.15941 | .01956 | 59% |
| BF16 | Encoder 15 | .05 | +.02895 | +.12527 | +.18087 | .09632 | 77% |
| BF16 | Encoder 15 | .10 | +.07191 | +.19857 | +.24066 | .12665 | 64% |
| FP32 | Connector | .01 | +.03081 | +.03071 | +.03073 | .00010 | 0.3% |
| FP32 | Connector | .025 | +.07703 | +.07601 | +.07601 | .00101 | 1.3% |
| FP32 | Connector | .05 | +.15405 | +.14679 | +.14679 | .00726 | 4.9% |
| FP32 | Connector | .10 | +.30810 | +.26404 | +.26402 | .04406 | 16.7% |
| FP32 | Encoder 15 | .01 | −.08550 | −.06295 | −.06294 | .02255 | 35.8% |
| FP32 | Encoder 15 | .025 | −.21374 | −.06625 | −.06624 | .14749 | 223% |
| FP32 | Encoder 15 | .05 | −.42748 | −.07828 | −.07830 | .34920 | 446% |
| FP32 | Encoder 15 | .10 | −.85497 | +.05409 | +.05413 | .90906 | 1,681% |

The zero-contrast grad/no-grad gap is **−.007923 and +.019994** for BF16 connector and encoder 15, but only **+.00000459 and +.00000721** in FP32. At alpha .01, grad versus no-grad finite differences differ by .11021 at the BF16 connector, while the FP32 paths differ by about .00002. Thus three-repeat stability does **not** make the BF16 modes interchangeable. BF16 rounding also removes many small edit coordinates: at alpha .01, 21,794/26,624 connector coordinates and only 6,822/34,560 encoder-15 coordinates change, versus all coordinates in FP32. The FP32 connector gives a credible local directional match at the smallest step, with error increasing by alpha .10. FP32 encoder 15 still diverges under matched execution paths and even reverses the measured sign at alpha .10. That failure cannot be explained solely by the BF16 mode mismatch; the data are consistent with substantial nonlinearity along this donor direction, though they do not isolate its cause.

These are two sites on **one fit route**, not a transferred lens or a security result. The JSON stores scalar gradients/norms and predictions, not the full gradient or edited tensor, so the VJP dot products cannot be reconstructed independently from the saved data alone; the audit verifies source code and all stored score arithmetic. A next fit-only check should use matched execution modes and smaller FP32 steps at encoder 15 before treating its derivative as a predictive readout.
