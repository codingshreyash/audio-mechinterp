# Frozen FP32 locked-test runner handoff

Date: 2026-11-18. The locked-test candidate has **not** been run by this agent. The input-only check read file hashes and tokenizer IDs; it performed no Qwen/MPS forward and inspected no test model answers.

`locked_test_fp32_v1.py` SHA-256 `dd9ea047caf3284844d2bb1226bedd807822eb66b11bd9d1e6312d0813a928d1` is an auditable copy of the frozen development runner (`development_fp32_v1.py` SHA-256 `4c2ac2c702126494f3d1d50ceaeddf69912a044720b8f216fda4e99d6083d77c`). The differences are the frozen input/answer allocation, names of outputs, and a passed-development prerequisite. `test_locked_test_fp32_v1.py` SHA-256 `bd3f7db3b31785beb2a595be8d8f42c581dfc419fc879800ed69a9ade5c0e62f` contains CPU tests.

Before model weights load, the runner verifies:

- Frozen plan SHA-256 `c5cc77a37ebaa69c0d231c83cd19e0956c7a5c8ab87e1aae067f3812b6a5b34b`.
- Fit result SHA-256 `3cf12f5e85891d1f804111384c8fab3d09242980d44c2e3854c875310b98d520`, progress `2de5082c6e4b954938f02615469e096e48ed3d9d1a6ae87cc78aba6e1a66b538`, baseline `076346729817170d488cd81ad549b0ac7f9e4daf5a402c5c21bdb244a315727c`, and passed fit gate.
- Development result SHA-256 `0eb01bab4eff1dfaa2c4c2082b25e51fae8d5b91909afaf5fcd222076a48f769`, progress `9ddfaa974b174d8adbdfd9efa76e7280e1611d29fcf1dff205aee5843bcbbc07`, frozen development code hash, and passed development gate.
- Test manifest SHA-256 `88bb12c02bc000ffee7dee8329ac3aeddd57d5b44898786f8e6c88d63454f845`, all 16 WAV file hashes, exact voice/word/phrasing allocation, and identical outside-slot PCM declarations.
- Pinned title-case complete answers with EOS: `North=[25221,151645]`, `South=[25018,151645]`, `Seven=[59528,151645]`, `Eight=[59085,151645]`. Greedy answers are judged by case-folded decoded text and EOS.

The one-shot evaluation fixes eight matched test pairs × two directions × connector/encoder31, α `0,.005,.01,.025,.05,.1,1`, and `codeword`-phrasing controls. It uses the fit-only floor and β without tuning. It records all-planned and clean-eligible denominators, local VJP, full-answer+EOS score contrast, row contributions, grad/no-grad gap, cache parity, controls, complete generated answers, MPS allocation, and categorical breakdowns. The gate requires ≥75% determinate sign agreement and Jacobian MAE ≤0.75 times **each** nonzero zero/norm comparator MAE, separately at both sites. Results, progress JSONL, and memory log are new files and cannot overwrite existing paths.

Checks completed without candidate inference:

```text
models/qwen_omni/.venv/bin/python eval/audio_jlens/test_locked_test_fp32_v1.py
# 5 CPU tests PASS
models/qwen_omni/.venv/bin/python -m py_compile eval/audio_jlens/locked_test_fp32_v1.py eval/audio_jlens/test_locked_test_fp32_v1.py
# PASS
models/qwen_omni/.venv/bin/python eval/audio_jlens/locked_test_fp32_v1.py --validate-inputs-only
# PASS: prerequisite/result hashes, 16 test WAV hashes, tokenizer IDs
```

After independent review and a code commit, the one-shot model command is `models/qwen_omni/.venv/bin/python eval/audio_jlens/locked_test_fp32_v1.py`. No run is triggered by this handoff.
