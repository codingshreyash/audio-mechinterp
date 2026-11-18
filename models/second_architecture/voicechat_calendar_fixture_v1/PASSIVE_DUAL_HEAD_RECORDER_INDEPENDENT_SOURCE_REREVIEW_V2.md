# Independent source rereview: passive VoiceChat dual-head recorder V2

18 November 2026. **Decision: GO for corrected source-only passive recorder; live use remains gated.**
This rereview covers correction commit `e6cd42f` and clears the three source
issues in the immutable [V1 HOLD](PASSIVE_DUAL_HEAD_RECORDER_INDEPENDENT_SOURCE_REVIEW_V1.md)
(`675f79a`). It does not load VoiceChat weights or attest model behavior.

## Exact reviewed files

| File | SHA-256 |
|---|---|
| `passive_dual_head_recorder.py` | `11f6b3e60652eaa886e65daac1b927bf1b5cb4521c7385dc1f297dcd0ab54527` |
| `test_passive_dual_head_recorder.py` | `25cc7179cb25be70ecbdff899a38a9c4282a70971f94a30444ccf6b393b16281` |
| `PASSIVE_DUAL_HEAD_RECORDER_SOURCE_ONLY.md` | `e933e7fa89006ec9d997fdaad1788873c463964e858955e1807d43ea8454a210` |

The pinned NVIDIA-NeMo/Speech commit
`097dfe9e2f55baf653b83035868bdc89849f1b47` has the source hashes
claimed by the recorder: `duplex_stt_model.py`
`415e1e01584bab96d9c059d5b0fe9162adaf19677ef43e565c3afad6e073e74b`
and `nemotron_voicechat.py`
`fe11dcdde3c31d7144469dab0c6ec9629ff8f7e4dbf881d8d1db258243a67f80`.
I independently fetched and hashed those exact files. The pinned
`_init_inference` uses no cache for Nemotron, and `_step_inference` then passes
a growing input prefix to the language model.

## Rereview of the V1 blockers

1. Hooks now copy only `[:, -1:, :]` from the shared state, raw text head,
   raw function head, and returned logits. They record each original prefix
   length, require all five captured outputs to agree, and require length
   `t+1` in the pinned no-cache path (one in a cache path). This preserves
   the final-position pre-boost and post-boost values used for selection
   without copying the entire growing-prefix logits.
2. Each STT frame must have the next contiguous ordinal. Each TTS code step
   must immediately follow the corresponding STT frame. Before the pinned
   postprocessing call, the recorder requires exactly `T` STT frames and
   `T-1` ordered TTS steps. Fake tests reject a skipped STT frame, a
   duplicate STT frame, a missing TTS step, and a wrong no-cache prefix.
3. The untrimmed state capture now reads `function_response_spans`, the key
   used by the pinned postprocessor. A fake pass-2 test saves and reloads a
   nonempty span.

I ran the seven recorder tests and eight existing two-pass adapter tests:
`15/15` pass. The fake stock-versus-instrumented comparison checks text and
function result channels, unspliced audio, and parser/replay decisions. The
recorder separately captures the fake TTS code steps. The wrapper still calls original methods once and
returns their original objects; all hooks return `None`, and method attributes
are restored after success or failure. These tests cannot establish identity
for the 11B checkpoint.

## Remaining live gate

No human listening records are present for the 12-WAV cohort, and no live
VoiceChat inference was run. Complete the independent human QA, controller C
grants, and official-sample stock-versus-instrumented identity checks before
using this recorder on fixture audio. It records last-position head values
and per-step TTS codes; it does not independently extract the outer method's
local `gen_codes` tensor or prove that the audible response caused a parsed
function proposal. The real run must retain raw outputs and parser decisions
and compare unspliced waveform bytes or samples against stock execution.
