# Qwen action method: typed-only resource/VJP pilot backend V1

19 November 2026. **Source only; no independent executable GO and no candidate weights loaded by this backend.** The [method V3 source review](INDEPENDENT_METHOD_V3_SOURCE_REVIEW_GO_20261119.md) approved a target-restricted three-class method, while the [runner contract review](INDEPENDENT_RUNNER_CONTRACT_SOURCE_REVIEW_GO_V1.md) approved only a no-weight evidence contract. This backend implements just the first typed FIT resource pilot, `F_L1__typed`, under a new one-shot gate. It has no audio, development, locked-test, or native tool-call entrypoint. The 18 technically audited WAVs still have no human semantic review, so their candidate path remains closed.

`typed_pilot_backend_v1.py` and `typed_pilot_parent_v1.py` reject a missing independently committed `TYPED_PILOT_GO_V1.json` **before the attempt directory or model load**. The GO must bind the exact committed hashes of all source, adapter, frozen spec, readout, host, parent-monitor and audit files; package versions, the installed Transformers commit, the fixed `runs/typed_pilot_v1/attempt_001`, and an independent executable review memo. These GO files do not yet exist. Source review must test the gate and add a receipt only after any fixes. A GO permits exactly one typed pilot, not the full FIT or audio batch.

The parent uses a 300-second whole-attempt deadline, a sampled 30-GiB process-tree RSS ceiling, a 64-MiB total evidence ceiling, and pre/post AC, open-lid, sleep and thermal checks. The worker sets an MPS allocation fraction and records sampled driver allocation below 24 GiB after load, each VJP, and generation. Resource sampling can miss shorter transients, so the result reports measured peaks. A failed or killed attempt stays in place as partial evidence and cannot be silently reused. The parent writes and fsyncs a terminal receipt after its result, then checks the deadline again; the separate auditor rejects missing/late terminal proof.

The worker loads only pinned `Qwen2_5OmniThinkerForConditionalGeneration` in FP32 and prepares one typed request with the frozen system prompt. For each canonical `LIST`, `ASK`, `CANCEL` plus EOS target, it appends the label token and reads full-vocabulary logits at the last prompt and appended-label positions. The scalar is the sum of the two log probabilities. The model/logits remain FP32; vocabulary normalization uses FP64 to match the saved NumPy log-sum-exp audit. It captures post-block residuals at layers 17 and 34 at the last prompt token and obtains one VJP per target through both sites. Frozen parameters and integer input IDs provide no autograd leaf by themselves: the layer-17 hook returns a **numerically identical detached leaf** with `requires_grad`, while layer 34 stays connected to that leaf. This isolates the VJP to the layer-17 state without changing its forward value. One fake-model test caught and fixed this detail before any real run.

The worker separately generates a complete response with no hook and with a read-only layer-34 hook. Exact generated suffix IDs must match. It saves three full target pairs of FP32 vocabulary logits, six FP32 prompt states, six FP32 VJPs, prompt IDs, hook counts, complete generation IDs, endpoint records, MPS samples and source identity. `typed_pilot_audit_v1.py` runs without candidate weights, rehashes all 18 binary sidecars, reconstructs full-sequence scores, verifies identical prompt-last states across suffix passes, re-decodes generation with the pinned tokenizer, and checks parent, host, resource, terminal and read-only parity evidence. A positive pilot establishes feasibility for one typed FIT item only, not a reusable lens, complete audio behavior, or security finding.

No-weight checks:

```sh
models/qwen_omni/.venv/bin/python -m unittest models.second_architecture.qwen_action_method_v1.test_typed_pilot_v1 -q
models/qwen_omni/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.typed_pilot_parent_v1 --source-preflight
```

Only after independent executable GO, the fixed candidate command would be:

```sh
models/qwen_omni/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.typed_pilot_parent_v1 --run
```

The saved auditor would then use the fixed exclusive destination `runs/typed_pilot_v1/attempt_001_audit.json`. **Do not execute candidate inference from this handoff alone.** Any failure of stock/instrumented parity, cross-suffix state, human QA, source hash, host, MPS, RSS, deadline, or saved-byte audit stays a failure. Later FIT/development work requires its own reviewed job and cannot inherit this pilot's GO.
