# Full FIT-only rank-16 transport: independent saved-artifact audit

**Verdict:** The saved fit is internally consistent and complete for its **48 preselected fit contexts**. This audit verifies construction and bookkeeping, not predictive usefulness. It ran only CPU checks on saved data; no Qwen inference or development/locked-test input was used.

I ran the independent [CPU audit script](audit_fit_rank16_saved_v1.py) against the completed stage-1 checkpoint, accumulator checkpoint, lens, report, progress, and memory log. It returned `passed: true`, `errors: []`. The report's SHA-256 bindings agree with the local stage-1 (`8276aca0…`), checkpoint (`b8af2db2…`), and lens (`fef84a20…`) files, and with the progress and memory logs. Metadata binds the corrected fitter, frozen spec and manifest, V2 scoring artifact, proposal, case list, model revision, and source/target sites.

| Check | Independent result |
| --- | --- |
| Fit coverage | 48/48 exact case IDs and audio hashes; 48 matching progress/checkpoint rows; 16 timings per case = 768 VJPs. |
| Basis | `2048 × 16`, finite and orthonormal (max `UᵀU−I` error `8.3e−7`); independently reconstructed from 24 undirected fit-only terminal differences. Singular values 1/16/17: `380.37 / 65.90 / 53.66`; rank-16 relative ratio `0.173`, above the frozen `1e−6` threshold. |
| Time bins | For all 48 contexts and both sites, six row lists are nonempty, disjoint, and in range. Per-bin row counts range `9–20` at encoder31 and `4–10` at connector. |
| Matrix reconstruction | Saved encoder31 `[6,16,1280]` and connector `[6,16,2048]` maps exactly equal the corresponding checkpoint gradient sums divided by 48 (`max abs difference = 0`), and their report hashes match. Every element is finite and nonzero; Frobenius norms are `18.65` and `25.46`. |
| Numerical/runtime | Maximum grad/no-grad terminal-state gap `0.000717`, below the frozen `0.001` tolerance. VJP times sum to `139.69 s`; the memory-log elapsed span is `178.52 s`. |
| Memory | 818 expected phases present (load, 48 captures, post-stage1, 768 VJPs); peak MPS driver allocation `21.25 GiB`; no sample exceeds 30 GiB. |

The full run's stage-1 **tensor contents** (target states, source means, basis, metadata) equal those in the successful corrected smoke; the serialized `.pt` bytes have a different file hash. The `.pt` stage-1, checkpoint, and lens files are ignored by Git (`*.pt`); the report, progress, and memory logs are versioned and hash-bound. Reproducing or independently evaluating the lens from another checkout therefore requires the local `.pt` files or a rerun. The saved report establishes that a corpus-averaged rank-16 map was fitted, **not** that it predicts held-out interventions, complete answers, or a security failure. Those conclusions require the still-unrun, versioned evaluation protocol and controls.
