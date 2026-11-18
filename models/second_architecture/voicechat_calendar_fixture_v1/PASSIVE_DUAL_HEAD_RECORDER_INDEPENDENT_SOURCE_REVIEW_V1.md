# Independent source review: passive VoiceChat dual-head recorder V1

18 November 2026. **Decision: HOLD for live use and source-only identity claims.**
This review concerns the source committed as `94e3448`; it does not load the
VoiceChat checkpoint, listen to audio, or attest a live model result.

## Exact source and checks

| File | SHA-256 |
|---|---|
| `passive_dual_head_recorder.py` | `dff21efa4b23a598482b8cd9506343462903380bd51da6f1eb903993c581bbd8` |
| `test_passive_dual_head_recorder.py` | `67292629c20f1917a4b979f3d03b9b2e25daa96586c706a5abf7cae0ed9c0fed` |
| `PASSIVE_DUAL_HEAD_RECORDER_SOURCE_ONLY.md` | `d715c13e608801859ad9916ef0b0ee2ee620ec480057b27e9d8432374cbc13c5` |

I independently fetched the exact NVIDIA-NeMo/Speech commit
`097dfe9e2f55baf653b83035868bdc89849f1b47` and verified that its
`duplex_stt_model.py` and `nemotron_voicechat.py` SHA-256 values match the
recorder's pinned constants, respectively
`415e1e01584bab96d9c059d5b0fe9162adaf19677ef43e565c3afad6e073e74b`
and `fe11dcdde3c31d7144469dab0c6ec9629ff8f7e4dbf881d8d1db258243a67f80`.
The four recorder tests plus eight existing adapter tests pass with fake models:
`12/12` under the local test environment. They establish fake-model output
identity and cleanup only.

## Blocking findings

1. **Actual Nemotron frames have a growing sequence dimension.** The pinned
   `DuplexSTTModel._init_inference` selects `use_cache=False` when
   `pretrained_llm` contains `Nemotron`; `_step_inference` then passes
   `input_embeds[:, :t+1]` to `self`. The recorder copies full hook outputs
   and its `finish` function requires every captured tensor's sequence length
   to equal one. At `t=1`, the official path produces a two-position output,
   so this recorder would reject a valid frame. Copying full-prefix text and
   function logits before that check also incurs avoidable, growing memory.
   Capture the last time position from each head and shared state, preserve
   the observed prefix length, and check the expected frame relation. The
   pre-boost and post-boost copies must refer to that same final position.
2. **Frame completeness is not verified.** `start(frame)` checks that no frame
   is open and that the cap is not exceeded, but does not require
   `frame == len(record["frames"])`. A skipped or duplicate frame index could
   pass. The post hook does not compare the frame count with `state["T"]`,
   and the number/order of TTS code calls is not checked against the STT
   frames. Add these checks and fake tests with missing/duplicate frames and
   missing TTS code calls before claiming a complete untrimmed trajectory.
3. **The response-span state key is wrong.** The recorder saves
   `state.get("response_spans")`, while the pinned `_post_inference` reads
   `function_response_spans`. The current field silently records `None` on
   the relevant official path. Capture the actual key and test a nonempty
   injected-response span.

The two-pass adapter's existing QA and inert dispatcher remain separate.
The 12-WAV human listening and controller grants are absent; no VoiceChat
weights were run in this review. After a corrected source commit, perform a
fresh independent source review. A live stock-versus-instrumented identity
comparison remains necessary after the human QA gate.
