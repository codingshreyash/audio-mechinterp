# Independent V2 source review: old-FIT graph/no-grad parity diagnostic

18 November 2026. Decision: GO for graph/no-grad diagnostic source only; live model HOLD until external freeze.

This new review supersedes the runner/gate V1 source decision for launch admission. It preserves the V1 review and the auditor V1 HOLD as immutable history. The V2 gate amendment is commit `607a2053f920ced4d1f8b710753c14c3afbe1159`; the corrected auditor and tests are commit `770851bd15f974d2bf7e48cb858ea831074dd5ba`, followed by the separate auditor V2 GO review `a06f3b0`. The V2 runner/gate review occurs after both. No model or freeze was run or created for this review.

| Gate-bound source | SHA-256 |
| --- | --- |
| `VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_PROSPECTIVE_V1.md` | `06d587f4b4742b26882c8a99bd901966f7b68aea7957777d9e226e854ed08f79` |
| `voxtral_old_fit_grad_mode_parity_runner_v1.py` | `1f92935f219b6b6a86c5307ec57ad69ddabcb1c0de2387e8c2c99b49888c3d91` |
| `test_voxtral_old_fit_grad_mode_parity_runner_v1.py` | `9985e2a060136e528288a3a68e71317cb75fb8f5f37c3d4ba7da9da1a85cfd6a` |
| `voxtral_old_fit_grad_mode_parity_launch_gate_v1.py` | `1db5c9a789083ae3a2bf6b4b99886a8ffb6a2ee9c0fe7772adae00f27481cb60` |
| `test_voxtral_old_fit_grad_mode_parity_launch_gate_v1.py` | `107bba86e2c945f367fde32e485aa586fa9adf23da1a401fd5e1e6e28e1f5b75` |
| `voxtral_old_fit_grad_mode_parity_audit_v1.py` | `3b5294853f1692a9d18598e9ab352057fa206e8a43d20072d968b45a48174ecf` |
| `test_voxtral_old_fit_grad_mode_parity_audit_v1.py` | `2e93258833a3667caf8381c4c2df6419fe970a38f404beceb4c01f8882ed3bf9` |
| `VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_AUDITOR_INDEPENDENT_SOURCE_REVIEW_V2.md` | `f8671d76aecf95d71de0e33c5fca65b5691dc2b9397c660512a339ca7e4ce1ec` |

The only runner/gate amendment since V1 is the gate's V2 review paths and reviewed-source chronology check, plus a test for those paths. For each runner/protocol/gate source and test, the gate now requires current working bytes to match `HEAD`, bytes at this V2 review commit to match current bytes, and the last source-changing commit to precede the review commit. It applies the same rule to the corrected auditor source and tests at the auditor V2 review commit. First-add immutability remains required for reviews, failed-attempt receipts and the future freeze. This avoids treating corrected source as if it had never changed; the earlier source and HOLD/GO reviews remain in Git history.

The other source checks from V1 still hold: distinct one-shot diagnostic rather than a Stage-B FP32 retry; exact failed-attempt and Stage-A/B provenance; full BF16 encoder31/projector source hashes; eight full `N,G,G,N`/`G,N,N,G` calls and sixteen fixed-origin direct calls; capture before any cross-forward equality assertion; exact source/projector/LM handoff checks; a joint graph-mode/leaf-gradient contrast; and the 180-second, 30 GiB, 1 GiB, 8 MiB stream and 120-second CPU-audit caps. A first bitwise difference must be reported separately from a difference above the observed two-repeat floor. A one-case numerical result cannot establish an audio-specific mechanism or a security issue.

I reran all three committed CPU suites in the pinned `.venv`: **24/24 passed** (ten runner/gate and fourteen auditor tests). The auditor's corrected frozen-batch and pre-deserialization size-cap tests are included. In a separate temporary Git repository, the gate's real Git binding primitives rejected both an uncommitted source mutation and a later committed source change after review. The live `--check-freeze` command failed closed because this V2 runner review was absent at test time. An external freeze and independent freeze review are still necessary before one diagnostic attempt can run.
