# Development Stage A implementation handoff

**Status:** code draft for independent review; no development candidate inference has run. It must remain unexecuted until the FIT calibration and ridge artifact are completed, audited, and hash-bound. The protocol's V2 first-token endpoint was adapted after FIT outputs; it is not evidence of a complete answer by itself.

Files: `development_transport_stage_a_v1.py`, `development_transport_metrics_v1.py`, and `test_development_transport_stage_a_v1.py`. The runner opens only `split == "development"` corpus rows and verifies all 96 planned directions, clip hashes, and exact outside-word PCM. It has no locked-test input path. It pins the committed FIT lens, calibration raw/report, ridge baseline/report, protocol, model, and V2 scoring artifact hashes before loading development inputs.

Stage A first records 96 clean source/terminal states and complete greedy answer+EOS outputs, then measures both sites at six frozen uniform-edit alphas. Each direction uses the opposite word in the same voice/template as donor. It compares actual V2 score movement with the corpus map, zero, row-count-weighted global ridge, and six-bin ridge; V1 movement, full generated answers, rank-16 projection capture, effective edit norms, and absolute category-anchor scores are saved separately. The clean and measurement stages have prefix checkpoints; an early failure writes a numbered failure JSON with the 96-planned denominator. The completed result is a single raw `.result.pt` plus `.summary.json` with pair-level bootstrap and category/voice/template/pair rows. No case is chosen using development answers.

CPU checks (no candidate model inference):

```sh
cd /Users/shrranjan/audio-mechinterp
models/qwen_omni/.venv/bin/python -m unittest eval.audio_jlens.corpus_v1.test_development_transport_stage_a_v1 -v
models/qwen_omni/.venv/bin/python -m py_compile eval/audio_jlens/corpus_v1/development_transport_stage_a_v1.py eval/audio_jlens/corpus_v1/development_transport_metrics_v1.py
```

The five toy tests check donor-oriented score sign, rank-16 comparator shape, all-planned gate and pair-resampling mechanics, a toy generation patch, and rejection of hidden special tokens or early EOS. The **FIT-only** Qwen generation-patch smoke in `fit_generation_patch_smoke_v1.py` still needs an MPS run before opening development. Then run `--validate-input-only`, review the complete manifest and hashes, commit the exact runner, and execute the one-shot development Stage A. Do not alter the runner based on item outcomes. If either site fails Stage A, report the negative result; if both pass, Stage B time controls and canonical full-sequence scoring are required before a positive development claim or locked test.

Stage B is specified in `PRE_DEV_EVAL_PROTOCOL_V1.md` but not implemented in this handoff. The Stage A result alone cannot support time selectivity, complete-sequence prediction, or a security finding.
