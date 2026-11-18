# FIT-only rank-16 transport handoff

`fit_rank16_transport_v1.py` is a candidate fitter, not evidence that a reusable lens works. It reads only the frozen fit-pool manifest and selects the 48 `map_fit` contexts (12 word pairs, two TTS engines, two word directions, first fit template). It has no development/test input path or evaluator. The code is ready for independent review before MPS execution.

## Method

1. Capture unedited post-encoder31, post-connector, and post-text-block35 prompt-last states for the 48 fit contexts. Partition each frozen word slot into four equal time bins plus one preword and one postword bin. The source state for a bin is the mean over its disjoint mapped rows.
2. Compute 24 undirected fit pair/voice terminal-state differences, then an orthonormal rank-16 target basis by SVD. Save the basis, singular values, fit-only category source means, and terminal means in a stage-1 checkpoint.
3. For every fit context, make one grad-enabled forward and 16 target-basis VJPs through the same graph. Sum gradient rows within each bin and average across the 48 contexts. Save separate `[6,16,1280]` encoder and `[6,16,2048]` connector maps. Their values predict the rank-16 terminal response to a uniform additive edit of all rows in each bin; they do not predict a full donor rowwise swap by definition.

Stage 2 checkpoints after every complete context and verifies that the saved case IDs remain an exact prefix of the frozen list. The checkpoint is authoritative if the progress JSONL missed its last line. The code records MPS driver allocation after model load, each stage-1 capture, and every VJP; it aborts above 30 GiB. It rejects changed script/spec/manifest/scoring hashes on resume. Use distinct output prefixes for smoke and full fit.

## Verified without candidate model execution

Python: `/Users/shrranjan/audio-mechinterp/models/qwen_omni/.venv/bin/python`.

```sh
cd /Users/shrranjan/audio-mechinterp
models/qwen_omni/.venv/bin/python -m unittest eval.audio_jlens.corpus_v1.test_fit_rank16_transport_v1 -v
models/qwen_omni/.venv/bin/python eval/audio_jlens/corpus_v1/fit_rank16_transport_v1.py --validate-only
```

The three CPU toy tests pass: simultaneous encoder/connector VJP orientation, basis orthogonality plus checkpoint serialization, and fit-only category means. `--validate-only` checks all 48 fit inputs and source hashes without loading weights or making a model forward. It binds the frozen corpus, V2 scoring artifact, model revision, and current proposal; it does not score model behavior.

## Independent review and first model check

Review hook identity, prompt-last target, six-bin row mapping, checkpoint behavior, and resource guard. Then run the FIT-only integration smoke on a unique prefix; it does all 48 no-grad basis captures and one VJP on the first case, without saving a fitted map:

```sh
models/qwen_omni/.venv/bin/python eval/audio_jlens/corpus_v1/fit_rank16_transport_v1.py --smoke --output-prefix eval/audio_jlens/corpus_v1/fit_rank16_smoke_v1
```

The smoke is expected to save `.stage1.pt`, `.memory.jsonl`, and `.smoke.json`. Check the rank-16 singular spectrum, nonempty bins, finite first-coordinate norms, grad/no-grad terminal-state gap, measured memory, and hook counts before committing to the full 768-VJP fit. A smoke failure should be recorded, not fixed by consulting development/test results.

After review and a successful smoke, the full fit can use another unique prefix:

```sh
models/qwen_omni/.venv/bin/python eval/audio_jlens/corpus_v1/fit_rank16_transport_v1.py --output-prefix eval/audio_jlens/corpus_v1/fit_rank16_transport_v1
```

The stage-1 checkpoint is written after all 48 captures; only stage 2 resumes per context. A stage-1 interruption requires restarting that stage with a fresh prefix. The proposal is still a design document: a versioned development gate and baselines must be frozen before development evaluation.
