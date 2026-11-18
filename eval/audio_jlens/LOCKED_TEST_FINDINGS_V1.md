# FP32 audio-to-answer Jacobian: locked-test findings

The one-shot Qwen2.5-Omni-3B locked test passed its frozen local-prediction gate at the audio connector and post-block audio encoder layer 31. **Only 10 of 16 planned directions per site were clean-eligible.** Three `seven/eight` pairs generated `7` and `8` with EOS instead of the precommitted `Seven` and `Eight` answer surfaces, and the exact-answer rule excluded both directions of each pair. The [independent saved-result review](LOCKED_TEST_FP32_RESULT_INDEPENDENT_REVIEW.md) recomputes the gate and checks hashes, progress, controls, and memory.

| Site | Eligible / planned | Small-step sign | Jacobian MAE | Zero MAE | Exact α=1 donor answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Connector | 10/16 | 30/30 | 0.001358 | 0.040740 | 10/10 eligible; 10/16 planned |
| Encoder layer 31 | 10/16 | 30/30 | 0.001783 | 0.032969 | 10/10 eligible; 10/16 planned |

The fit-only norm baseline has coefficient zero at both sites, so it is numerically identical to the zero-change baseline. The signed local derivative predicts changes in a **complete canonical answer-plus-EOS log-probability contrast**, not the full generated response. At α=1, full donor-state patches produced the exact donor word and EOS for every eligible direction. The two sites' local prediction errors were 3.33% and 5.41% of the zero comparator. These are new word pairs and prompt phrasings in the same two synthetic voice families, not independent speaker or architecture transfer.

The precommitted seven controls ran on eight `codeword` directions per site. For the four clean-eligible `codeword` directions at each site, all 56 control generations across both sites retained the recipient word. The other 56 control generations are on clean-ineligible number cases and cannot be counted as exact-word selectivity successes. Exploratory normalization of `7`↔`seven` and `8`↔`eight` makes the full patches semantically donor-directed on all 16 directions per site, but it also exposes one connector row-shuffle control that outputs the donor digit. That normalization was not the frozen endpoint and is not included in the gate.

This establishes a reproducible **local, per-input derivative and causal state-swap effect** on a narrow speech word-identification task. It does not establish a corpus-fitted transport that can be reused on unseen audio without computing a new gradient, a unique word-local circuit, a speech-output or tool-action mechanism, or a security vulnerability. The next method step is to fit an audio-time-to-output transport over a separate corpus and compare it on genuinely held-out inputs with transcript/anchor baselines and interventions; the answer-surface failure here must be handled by an explicitly frozen evaluation rule.

Protocol: [FP32 plan](PLAN_FP32_V1.md). Code: [locked-test runner](locked_test_fp32_v1.py). Raw result: [locked-test JSON](locked_test_fp32_v1_results.json). The fit, development, and locked-test code/results were committed in that order before each next split was run.
