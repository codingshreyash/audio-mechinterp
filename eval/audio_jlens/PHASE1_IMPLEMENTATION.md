# FP32 phase-1 fit-only runner handoff

Date: 2026-11-18. The Qwen candidate has **not** been run with this script. No development or locked-test audio is read.

Implementation: `fit_fp32_phase1_v1.py` SHA-256 `f1ae3e697221dc68ef21df847304b38adb1b303a7c65ddf336e2f2e3050cf113`. CPU checks: `test_fit_fp32_phase1_v1.py` SHA-256 `d64b20c51a32693bae59819d0812e7d09578ae6ee633d1c298dceaf818d0b844`. Frozen plan: `PLAN_FP32_V1.md` SHA-256 `c5cc77a37ebaa69c0d231c83cd19e0956c7a5c8ab87e1aae067f3812b6a5b34b` (bound in the runner).

Run from the repository root after review:

```text
models/qwen_omni/.venv/bin/python eval/audio_jlens/fit_fp32_phase1_v1.py
```

The runner fixes Piper and Kokoro, red→blue and blue→red, connector and encoder31, and `α={0,.005,.01,.025,.05,.1,1}`. Each score is the donor answer **plus EOS** versus recipient answer **plus EOS**. It saves complete generated answers, EOS flags, no-grad intercepts, grad/no-grad gaps, cache parity, effective FP32 edits, per-row VJP terms, and outside-word state differences. It checks all four clean answers before patching.

Every route/site also evaluates self, natural off-word, three fixed-seed same-row random, NumPy-seed-0 row-shuffled donor, and norm-scaled off-word controls. The off-word window is the lowest-start contiguous nonoverlapping candidate that lies wholly in a different attention chunk if one exists; otherwise the lowest-start nonoverlapping candidate. If none exists, off-word controls are marked ineligible. If its natural edit norm is zero, the scaled control is marked `ineligible_zero_offword_delta`. No replacement direction is invented. Controls report natural and applied norms, score changes, full answers, and hook counts when eligible.

The 30 GiB MPS limit has a hard allocator fraction and a measured driver-allocation sidecar after load, each VJP, cached prefill, each zero repeat, each alpha, and each control. A progress JSONL records completed route/site cases. After all eight fit cases, the script checks that every saved JSONL case equals its in-memory result and hashes the closed file. That `fit_progress_sha256` is written into both the final result and the fit-only norm-baseline sidecar, without a circular result hash. Existing paths are refused.

The final fit report applies the frozen stop rule separately at connector and encoder31. It computes a site floor from three repeated alpha-zero contrasts per route, counts all four complete alpha-one donor flips, and reports sign agreement on every determinate `.005`, `.01`, and `.025` effect. A site fails if no small effects are determinate, fewer than four full donor flips occur, or sign agreement is below 75%. The final status is `fit_only_gate_passed` or `fit_only_gate_failed`; `development_permitted_by_fit_gate` records the corresponding decision. The baseline sidecar remains available for audit even if the fit gate fails.

Checks run without Qwen inference:

```text
python -m py_compile eval/audio_jlens/fit_fp32_phase1_v1.py eval/audio_jlens/test_fit_fp32_phase1_v1.py   PASS
python eval/audio_jlens/test_fit_fp32_phase1_v1.py                                               6 tests PASS
```

The tests cover complete-answer teacher-forced score contrast and VJP orientation with a CPU toy model, deterministic control placement/norms, and the exact fit-only norm-baseline formula. This runner evaluates the **fit** split only. Its output does not establish the plan's development or locked-test gates.
