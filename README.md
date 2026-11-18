# Audio mechanistic interpretability research

Objective: reproduce and independently verify strong text and audio interpretability methods, then develop and validate audio-aware readouts and causal interventions that explain information flow through open audio models. The [research plan](../Garak/garak/knowledge/multilingual_mechinterp_pilot/AUDIO_MECHINTERP_FRONTIER_PLAN.md) and [codebase audit](../Garak/garak/knowledge/multilingual_mechinterp_pilot/AUDIO_MECHINTERP_CODEBASE_AUDIT.md) define the initial scope.

## First milestone

1. Reproduce Anthropic's Jacobian Lens readout and a causal swap on an exact open-model checkpoint; record the original method's evaluation metric and the local result.
2. Reproduce one existing audio Jacobian-lens or audio-SAE result, including its negative controls or negative outcome.
3. Build and validate a Qwen2.5-Omni-3B adapter exposing the audio encoder, audio-to-language boundary, and Thinker residual stream. Benchmark one prompt before any large Jacobian fit.
4. Freeze the audio-extension evaluation protocol before comparing candidate readouts.

Artifacts live in `reproductions/`, `models/`, and `eval/`. Each result must name its model revision, source revision, input data, run command, hardware, and exact output. A source audit or unit test is not an experimental reproduction.

The prior NemotronLabs VoiceChat intervention is [documented separately](../Garak/garak/knowledge/multilingual_mechinterp_pilot/voicechat_public_20261117/INTERVENTION_FINDINGS.md). It is a future streaming-system test case, not a general audio-lens validation.

## Current evidence

- [Status ledger](STATUS.md): achieved, negative, and unresolved results with their limits.
- [Reproductions](reproductions/anthropic_jlens/REPRODUCTION.md): upstream Jacobian Lens checks and a bounded open-model causal test; [independent audio review](eval/INDEPENDENT_REVIEW.md) records the Whisper pilot and its limits.
- [Qwen2.5-Omni adapter](models/qwen_omni/README.md): pinned checkpoint/source, hook semantics, local full-rank Jacobian benchmark, and source-position time map.
- [Fit-only causal path](eval/controlled_pairs/FINDINGS_V1.md): matched red/blue audio through processed features, audio encoder, connector, and Thinker; [independent review](eval/controlled_pairs/INDEPENDENT_REVIEW.md) checks saved outcomes.
- [Small held-out transfer](eval/heldout_flow/FINDINGS_V1.md): two new word pairs, voices, and phrasings. The frozen layer-30 versus layer-31 boundary failed to transfer; a later boundary appears in the planned sweep. See the [independent audit](eval/heldout_flow/INDEPENDENT_REVIEW.md).
- [Inert security decision plan](eval/security_case/PLAN.md): a future protected-action assay that separates audible output, native tool-call intent, and stub enforcement. No security finding has been established.

The local tests use only the Thinker's **text answer** from speech input. The Talker, human voices, second model architecture, and deployable attacker-controlled audio are open work. The initial terminal Jacobian readout on a separate semantic-bridge task was negative; see [pilot review](eval/semantic_bridge/FIT_PILOT_REVIEW.md).
