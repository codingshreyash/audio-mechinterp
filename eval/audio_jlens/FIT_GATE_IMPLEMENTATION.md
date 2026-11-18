# FIT-only scalar audio Jacobian gate: implementation handoff

Date: 2026-11-18. The Qwen candidate has **not** been run for this gate. No development or new-test answers were inspected.

Files:

- `fit_gate_v1.py` SHA-256 `d8d708bc734763c0f341d2c4a1466f47310bda390da0ba01cd9bd80f478084da`
- `test_fit_gate_v1.py` SHA-256 `f6ce3ab6f944bf1577d2aea5edb0703459e5ca2c0a017a907fc68606be9641b5`
- `PLAN.md` SHA-256 `18670b7a8b49ef1eef693c6daf784eb30ba18d89886fa6eebbbcea0469589863`; this revision fixes the exact fit-only norm-baseline formula before a candidate run.
- Frozen `SPLIT_V1.json` SHA-256 `2b48246361a629fbb0fa80c79a3d061b6977c8cb8af71cdc2275b1e4b0b4d45a` (unchanged).

Verified fit source manifest: `318807c92971babacad93b69b7d8478e29f397219da0cb9ed6e3134649aea98b`. The script verifies the selected Piper/Kokoro WAV file hashes and matched outside-slot PCM. The pinned processor yields `Red=[6033]`, `Blue=[10331]`, and end-of-turn `151645`. Teacher forcing scores `[answer token, end-of-turn]` from logits at prompt-last and next-token positions. The system prompt is `Answer the spoken question with one color word.`; patch fractions are `0, 0.05, 0.10, 0.25, 1`. Default run is Piper red recipient, blue donor at connector and encoder layer 15.

Checks run from repository root with `models/qwen_omni/.venv/bin/python`:

```text
python -m py_compile eval/audio_jlens/fit_gate_v1.py eval/audio_jlens/test_fit_gate_v1.py    PASS
python eval/audio_jlens/test_fit_gate_v1.py                                              4 tests PASS
python eval/audio_jlens/fit_gate_v1.py --validate-inputs-only --engines piper kokoro      PASS
```

The CPU toy tests cover both teacher-forced positions, EOS contribution, audio-row patch orientation, candidate-specific contrast gradient, zero edit, hook cleanup, and the memory guard's record-before-abort behavior. The validator loads only the pinned processor and fit files, not the Qwen weights. The processor emitted existing configuration warnings about token IDs outside the audio vocabulary; token IDs were verified against the actual tokenizer.

Independent reviewer should check Qwen-specific forward/generation behavior before running the default command. If the fit clean answer is not the requested word with EOS, the script aborts that positive-control case. It records effective BF16 deltas, per-row VJP terms, measured score changes, full generated answers, hashes and model pins. It refuses to overwrite an existing result.

At model load, after each VJP, after cached prefill, and after each patch fraction, the MPS current/driver allocations are appended to a sidecar `*.memory.jsonl`. A measured driver allocation above 30 GiB is persisted and then aborts the run. The Qwen run has not yet occurred, so this document contains no actual MPS driver measurements for it.
