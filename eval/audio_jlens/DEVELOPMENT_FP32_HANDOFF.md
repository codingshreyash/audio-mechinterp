# FP32 development runner handoff

Date: 2026-11-18. This is a **development-only implementation**. No Qwen/MPS development inference has been run by this agent, and no locked-test manifest or audio is opened by the script.

Files and pins:

- `development_fp32_v1.py` SHA-256 `4c2ac2c702126494f3d1d50ceaeddf69912a044720b8f216fda4e99d6083d77c`
- `test_development_fp32_v1.py` SHA-256 `2557952db01f29868c9359ec7cec1efa185c21040c4fd69292dc678103fa6d6e`
- Frozen FP32 plan SHA-256 `c5cc77a37ebaa69c0d231c83cd19e0956c7a5c8ab87e1aae067f3812b6a5b34b`
- Frozen fit result SHA-256 `3cf12f5e85891d1f804111384c8fab3d09242980d44c2e3854c875310b98d520`; fit progress `2de5082c6e4b954938f02615469e096e48ed3d9d1a6ae87cc78aba6e1a66b538`; fit baseline `076346729817170d488cd81ad549b0ac7f9e4daf5a402c5c21bdb244a315727c`
- Frozen development manifest SHA-256 `ae7cfd2e08a09e44847bd7a2e0d5775b8017a6f5e072f3f5d3dc007b9f759a01`

The script verifies those SHA values and the exact fit helper source hashes before loading Qwen. It requires `fit_only_gate_passed` and `development_permitted_by_fit_gate=true`. It verifies all 16 development WAV hashes and the exact title-case canonical answers **including EOS** with the pinned tokenizer: `Cat=[26801,151645]`, `Dog=[48940,151645]`, `Monday=[35239,151645]`, `Friday=[34620,151645]`. Generated answers are judged by case-folded decoded text plus EOS.

Planned evaluation: eight matched development pairs × two patch directions × two primary sites = **32 site cases**, 16 directions/site. The fixed α grid is `0,.005,.01,.025,.05,.1,1`. Each case records the complete-answer teacher-forced contrast, local VJP and row contributions, no-grad intercept and grad/no-grad gap, FP32 edit norm, full greedy answer/EOS, cache parity and outside-word state differences. Controls run only on `chosen` phrasing: eight directions/site. They reuse the frozen fit control placement, seeds and ineligibility rules. MPS has a 30 GiB allocator cap and measured driver sidecar. Completed cases are appended to a progress JSONL; output paths cannot be overwritten.

The development gate uses the fit-only floor and norm coefficient at each site. It reports all-planned and clean-eligible denominators, determinate sign rate, mean absolute error versus both zero and frozen norm baselines, median absolute and relative error, and complete α=1 donor-answer flip rates. Indeterminate effects stay in error metrics. It passes a site only with at least 75% sign agreement on determinate clean-eligible small steps and Jacobian MAE ≤0.75 times **each** nonzero comparator MAE. Both sites must pass for `locked_test_permitted_by_development_gate=true`. The result retains category, phrasing and TTS family rows.

Checks completed without model inference:

```text
models/qwen_omni/.venv/bin/python eval/audio_jlens/test_development_fp32_v1.py
# 5 CPU tests PASS
models/qwen_omni/.venv/bin/python -m py_compile eval/audio_jlens/development_fp32_v1.py eval/audio_jlens/test_development_fp32_v1.py
# PASS
models/qwen_omni/.venv/bin/python eval/audio_jlens/development_fp32_v1.py --validate-inputs-only
# PASS: hashes, manifest shape/WAVs, tokenizer IDs, 8 pairs/16 directions per site
```

After independent review, the default command is `models/qwen_omni/.venv/bin/python eval/audio_jlens/development_fp32_v1.py`. It produces a new development result, memory log and progress log. It does not start a locked-test run.
