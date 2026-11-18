# Old-FIT Voxtral backward result auditor: pre-run handoff

**18 November 2026.** The separate [`voxtral_v1_fit_backward_result_audit.py`](voxtral_v1_fit_backward_result_audit.py) is a CPU-only, read-only checker for the *future* one-shot `animal_08|piper|fit_chosen` backward feasibility result. It pins producer script SHA-256 `cc48437b96d47c766f5ba9f86eb5789481fecd7d1469c09a503d225a42ed3be1` and protocol SHA-256 `8fa6aa224e150b2579e335d14f7dafddd542eacdf733bc92ab7f3ff281384148`. It neither loads a model nor opens the active development run or fresh v2 corpus.

## Before and after the live pilot

Run source-only validation now or immediately before the old-FIT pilot:

```bash
cd models/second_architecture
.venv/bin/python voxtral_v1_fit_backward_result_audit.py --validate-only
.venv/bin/python -m unittest -v test_voxtral_v1_fit_backward_result_audit.py
```

After the development run has ended and a separate one-shot old-FIT pilot has produced its immutable directory, audit **that exact directory**:

```bash
.venv/bin/python voxtral_v1_fit_backward_result_audit.py --run-dir runs/voxtral_v1_fit_backward_feasibility/ATTEMPT_DIRECTORY
```

The public auditor accepts paths only under the old-FIT backward run root. It emits JSON to stdout and does not modify the attempt. An error receipt yields `audited_failed_attempt_no_numerical_result`, never a numerical success. Missing, mixed, unbound, nonfinite, or out-of-cap receipts fail closed.

For a completed attempt, it rehashes pinned source files, frozen old-FIT mask/index/tree and saved baseline; verifies the start/progress/meta/result/sidecar SHA chain, snapshot and installed-source provenance, exact recipient/donor/WAV/word-row binding, suffix parity, hook shapes and BF16/FP32 dtype claims, phase order, elapsed time and MPS peaks. It loads the tensor sidecar on CPU with `weights_only=True`. It checks full source/target tensor shapes and finiteness, the four selected coordinate gradients at both sites, exact BF16 representability of stored VJPs, and hashes of recipient, donor, edit, zero-patch and alpha-patch tensors.

It independently reconstructs the intended uniform edit and BF16-realized row edits from saved recipient/donor states; recomputes all **4 coordinates × 2 sites × 3** contraction series (realized-mean uniform, rowwise, ideal uniform), the BF16 fidelity/nonuniformity diagnostics, measured `alpha−zero` and `zero−clean` vectors from full saved terminal states, and all absolute/relative four-coordinate error metrics. The [`test_voxtral_v1_fit_backward_result_audit.py`](test_voxtral_v1_fit_backward_result_audit.py) suite uses synthetic receipts and sidecars only; its six tests passed, including successful recomputation and fail-closed prediction, terminal-hash, nonfinite, resource, and path tampering.

This audit establishes **internal numerical consistency and provenance of the saved engineering pilot**. It cannot independently rerun the model graph, verify unselected gradient rows, prove that recorded tensors came from the claimed forward pass, or establish v2 transfer, acoustic localization, model behavior under ordinary speech, tool behavior, or a security finding. A failed or incomplete pilot remains a measured feasibility outcome, not a reason to substitute a new case or silently relax the frozen protocol.
