# Independent source review: old-FIT graph/no-grad parity diagnostic

18 November 2026. Decision: GO for graph/no-grad diagnostic source only; live model HOLD until external freeze.

This review covers the prospective protocol, runner, launch gate, and their CPU tests committed in `2c8e8b828fc532722d1e1f9ba14b1e30c3878bf0`. It does not approve the separate CPU auditor, a freeze, or a model run. The auditor has a separate HOLD review at the time of this source review; the gate requires a later auditor GO and an independently reviewed external freeze before it can launch.

| Committed file | SHA-256 |
| --- | --- |
| `VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_PROSPECTIVE_V1.md` | `06d587f4b4742b26882c8a99bd901966f7b68aea7957777d9e226e854ed08f79` |
| `voxtral_old_fit_grad_mode_parity_runner_v1.py` | `1f92935f219b6b6a86c5307ec57ad69ddabcb1c0de2387e8c2c99b49888c3d91` |
| `test_voxtral_old_fit_grad_mode_parity_runner_v1.py` | `9985e2a060136e528288a3a68e71317cb75fb8f5f37c3d4ba7da9da1a85cfd6a` |
| `voxtral_old_fit_grad_mode_parity_launch_gate_v1.py` | `1c296bf03bbb693480304025fdb77012a9f0d38da5b4dfaed11e58015d59ca89` |
| `test_voxtral_old_fit_grad_mode_parity_launch_gate_v1.py` | `8bd87a18bd61085a37a1f3788be249452d83fa0e064051e61e89a1d61ffacc88` |

## Source checks

The new diagnostic has its own namespace, one-shot reservation, outcome, and estimand. It preserves the failed Stage-B FP32 receipt `20261118T143734566440Z`, its `launch_failed` status and `41.18471045792103` seconds of model time. The prior `Direct LM replay mismatch at lm0` was the first failed assertion in code order; it was not proof that LM0 was the first divergent computation. No Stage-B retry is authorized.

The runner pins the two **full** BF16 cycle-zero clean-source tensor hashes, `e0abb561ad91255d1945c81c1dead1a75266f09d0e7f3695a123040fef47b2b6` for encoder31 and `a96b5e19ffefe48e537b30ddecb9de11c8c40becf4b4dd3f2813ab5583fce100` for the projector, before casting to FP32. The archive checks cover both signed-step rows at each site. The gate rechecks the committed failed-attempt manifest, review, ten raw files, Stage-A provenance, Stage-B BF16 result, and existing FP32 freeze before admission.

The fixed full-model schedule is encoder31 `N,G,G,N` followed by projector `G,N,N,G` (eight calls). Each site's first N and first G **complete** LM inputs become two fixed direct-LM origins. Each origin runs `N,G,G,N` (sixteen calls total). The runner records source, projector, complete LM input, LM0 pre-block input, five later terminal vectors and full first-answer logits without asserting equality between separate forwards. It checks each call's hook count, frozen source override, and projector-to-audio-slot and LM-input-to-LM0 handoffs. N uses a non-gradient leaf under `torch.no_grad()`; G uses a differentiable leaf under `torch.enable_grad()`. These are a joint mode/leaf contrast, not a test that attributes any difference to one factor.

The gate requires a committed source review with the exact GO marker above, a separately committed auditor GO, a later first-add-immutable freeze that hashes all bound source and provenance files, and a still-later independent freeze review. The worker claims the single reservation before model construction. The parent watches an isolated child process group with a 180-second model deadline, observed 30 GiB MPS cap, 1 GiB combined output cap, 8 MiB per-stream cap and separate 120-second CPU audit deadline. If polling catches an output overshoot, the gate records a failed attempt and retains each trimmed stream's original byte count, SHA-256 and first 512 bytes; that exceptional path does not preserve the complete stream.

I ran the committed runner, gate and auditor CPU suites with the pinned `.venv`: **20/20 tests passed** (nine runner/gate, eleven auditor). The gate's `--check-freeze` entry point failed closed before this review existed. I did not load Voxtral, create a freeze or launch the diagnostic. Saved-tensor arithmetic, any first differing boundary, and any mechanistic or security conclusion remain unobserved. A later result must distinguish the first bitwise difference from the first difference above the two-repeat observed floor; this single MPS case cannot identify a kernel, an audio-specific mechanism, or a security issue.
