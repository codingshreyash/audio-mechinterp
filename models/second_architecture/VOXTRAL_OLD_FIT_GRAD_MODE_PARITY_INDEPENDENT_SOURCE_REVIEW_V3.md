# Independent V3 source review: old-FIT graph/no-grad parity diagnostic

18 November 2026. Decision: GO for graph/no-grad diagnostic source only; live model HOLD until external freeze.

This V3 review is a new launch-admission review. The V1 and V2 runner reviews, the auditor V1 HOLD and auditor V2 GO remain immutable. The only source amendment since runner review V2 is commit `55e1720de7c611f05ada683beda304dde4ab2c88`: it changes the gate's runner-review path to this V3 file and matches the exact GO sentence in the already committed auditor V2 review. A CPU gate test checks that exact sentence and the corrected auditor source hash in the committed auditor review. This fixes the real V2 gate failure `Independent diagnostic source review absent` caused by a mismatched auditor GO marker. The auditor V2 review at `a06f3b0` precedes this gate amendment and this review.

| Gate-bound source or review | SHA-256 |
| --- | --- |
| `VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_PROSPECTIVE_V1.md` | `06d587f4b4742b26882c8a99bd901966f7b68aea7957777d9e226e854ed08f79` |
| `voxtral_old_fit_grad_mode_parity_runner_v1.py` | `1f92935f219b6b6a86c5307ec57ad69ddabcb1c0de2387e8c2c99b49888c3d91` |
| `test_voxtral_old_fit_grad_mode_parity_runner_v1.py` | `9985e2a060136e528288a3a68e71317cb75fb8f5f37c3d4ba7da9da1a85cfd6a` |
| `voxtral_old_fit_grad_mode_parity_launch_gate_v1.py` | `93bd26d235b6a60eb296b951850f40ac41d06f29fb57ab5ffce906e32cebfa59` |
| `test_voxtral_old_fit_grad_mode_parity_launch_gate_v1.py` | `a95f703f6b5483af5ea58586eecb3f3ef8d7c14ed22a3eb521315226ac3fb232` |
| `voxtral_old_fit_grad_mode_parity_audit_v1.py` | `3b5294853f1692a9d18598e9ab352057fa206e8a43d20072d968b45a48174ecf` |
| `test_voxtral_old_fit_grad_mode_parity_audit_v1.py` | `2e93258833a3667caf8381c4c2df6419fe970a38f404beceb4c01f8882ed3bf9` |
| `VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_AUDITOR_INDEPENDENT_SOURCE_REVIEW_V2.md` | `f8671d76aecf95d71de0e33c5fca65b5691dc2b9397c660512a339ca7e4ce1ec` |

I reviewed the full gate amendment and verified its marker is an exact substring of the immutable auditor V2 review. The gate still requires committed byte equality, independent source-review chronology, first-add-immutable reviews and freeze, exact failed Stage-B FP32 provenance, the new one-shot reservation and worker claim, fixed 8 full plus 16 direct calls, and the process, MPS, output, stream and CPU-audit caps. The corrected auditor independently pins all frozen input batch tensors and checks actual attempt and sidecar byte counts before deserialization. The runner still records boundaries before cross-forward assertions and treats N/G as a joint grad-context/leaf-gradient contrast; any later result must separate first bitwise from first above-floor differences and avoid mechanistic or security overclaim.

With the pinned `.venv`, I reran the three CPU suites: **25/25 passed** (four runner, seven gate, fourteen auditor). In a separate temporary Git repository, the gate's Git binding logic rejected an uncommitted source mutation and a later committed change after review. Before this V3 review was added, live `--check-freeze` failed at the absent V3 review, as intended. I did not create a freeze, load model weights or run the model. A later external freeze and independent freeze review are required before one live diagnostic attempt.
