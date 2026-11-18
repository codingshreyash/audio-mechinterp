# Independent static review: fit-only FP32 corpus cost probe

Reviewed 2026-11-18 against commit `4a2e6eb` and its revised `NEXT_PHASE_CORPUS_LENS_PROPOSAL.md`. I ran only CPU syntax/toy tests; no Qwen/MPS inference or development/test input was opened.

## Decision

**No blocker found for the bounded fit-only MPS cost probe.** The script measures one original Piper red fit prompt and 16 fixed target-coordinate VJPs through both source sites in one retained graph. It does not fit or evaluate a lens.

## Checks

- The fixed fit helper checks the controlled-pair manifest and WAV hashes. This probe selects only `piper/red` from that fit pair. It verifies the pinned Transformers source revision and loads the exact Qwen Thinker revision locally in FP32. The report records the script, proposal, helper, adapter and row-map source hashes. A source search found no development or locked-test path in the probe.
- The six time intervals are the exact swapped-word interval split into four quarters, plus one equal-duration pre and post interval. `bin_rows` assigns each nominal source-row center once, with boundary ties sent to the earlier bin; it rejects a shape mismatch or an empty bin. For the pinned Piper red clip, the pre/post intervals (about 0.997–1.798 s overall) fit inside its 3.108 s WAV. These are nominal row assignments, not acoustic localization after attention.
- The encoder-31 hook creates a detached differentiable FP32 leaf. The connector hook retains the original output computed from that leaf. The terminal hook captures post-block text residual 35; cotangents select the **last prompt position**, before final norm. `autograd.grad(target, [encoder, connector], grad_outputs=...)` therefore gives both source-site gradients from one backward without severing the encoder→connector path. The CPU toy test checks this orientation, including the encoder's indirect path through connector.
- Six-bin gradient norms sum each source-row VJP over the selected rows, matching the proposed derivative for a uniform additive edit. The 16 cotangents are fixed standard coordinates `0..15` as a **cost proxy**, not the future fit-only rank-16 SVD basis. Source/target hook counts must each equal one; gradient shapes and finiteness are checked.
- One synchronized forward and each of 16 synchronized backward calls have separate elapsed times. A memory sidecar is sampled after load, forward and every VJP, with the established 30 GiB MPS allocator fraction and driver limit. Exceptions, including memory failures, set `status=failed` and persist the report in `finally`; existing result/sidecar paths are refused.
- The revised proposal explicitly fixes **48** map-fitting recipient contexts: 12 fit pairs × two voices × two directions × one fit template. Thus `projected_48_example_model_work_seconds` is the correct first extrapolation **if** this fit prompt is representative and the two-site shared graph remains usable. The other 12 fit-pool pairs and second fit template are diagnostics, not additions to the saved map. The projected model work excludes the separately recorded initial model load.
- `py_compile` passed, and both CPU toy tests passed (**2/2**). Reviewed source SHA-256 `77b56fe1883443e2a85b89cc9b746da77b65e2f1b6a4f30c56bf7efd9edf00e2`.

## Limits for interpreting the cost

One short original fit clip cannot establish a worst-case time or memory bound for the planned new corpus. Compare its audio row count and prompt length with the frozen 48 fitting contexts before treating `48×` as a six-hour go/no-go estimate; longer clips and different feature counts can raise both costs. The fixed coordinate VJPs benchmark backward mechanics, not the later SVD construction, baseline fits, feature processing, or development/test passes. Keep those costs separate from the six-hour **model-work** cap.
