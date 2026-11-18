# One-prompt terminal-residual Jacobian fitter

**Implementation ready for review; full-width run not started.** The [script](fit_terminal_jacobian.py) targets the pinned Qwen2.5-Omni-3B Thinker and one fixed audio prompt. It computes, for source layers 0 and 17, all 2,048 rows of the derivative of the **final text decoder block residual at the last prompt position** with respect to each source layer's residual at that same position. This position policy differs intentionally from the [audio-placeholder average cost pilot](JACOBIAN_BENCHMARK.md). A terminal residual is audio-conditioned because its causal attention can read the preceding audio placeholders, but this derivative does not directly localize which audio frame carried information.

The script freezes parameters, hooks the pinned text decoder blocks, and returns a detached but numerically identical layer-0 output with `requires_grad=True` to start the retained graph there. One batch of eight copies of the same prompt is forwarded once. Each backward uses a one-hot cotangent on a separate target coordinate for each batch copy, at the last prompt position only. The graph is retained until the final missing coordinate batch. A resume process rebuilds one graph and skips fully saved batches. Rows, completion mask, metadata, and per-batch measurements are atomically saved after each backward. A checkpoint whose input hash, pins, layer choices, dimension batch, or position policy differ is rejected.

The default local MPS guard is 30 GiB of driver allocation, checked after loading, forward, and each backward, with explicit `torch.mps.synchronize()` around timing. The guard cannot preempt an instantaneous kernel out-of-memory error. The script refuses a cap above 90% of the device's recommended MPS maximum and currently supports batch widths 1–8. It does not download weights if the pinned checkpoint is absent. A completed checkpoint exits without a new forward.

## Verification completed without a full model run

- [CPU tests](test_terminal_jacobian.py) prove row orientation and values against two analytic matrices, including a short final coordinate batch, simulated interruption after an atomic checkpoint, and exact completion after rebuilding the graph. They also check that the graph-start hook preserves output values while exposing source gradients.
- The prior [loaded-model smoke](loaded_smoke_verified.json) established the exact module paths and gradient reachability on this pinned checkpoint. The earlier [16-coordinate pilot](jacobian_16coords.json) established local MPS memory headroom for a related but different position reduction.
- These checks **do not measure** the full-rank fitter's memory, runtime, numerical conditioning, vocabulary readout quality, or causal utility. The full fit should be reviewed before running and then compared with a text-only baseline and held-out audio evaluations under [the protocol](../../eval/PROTOCOL.md).

Example future invocation from this directory, **only after review**:

```sh
HF_HOME=.hf_cache HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv/bin/python -u fit_terminal_jacobian.py sample_apple_5p5s.wav \
  --checkpoint terminal_jacobian.pt --report terminal_jacobian.json \
  --dim-batch 8 --max-driver-gib 30
```
