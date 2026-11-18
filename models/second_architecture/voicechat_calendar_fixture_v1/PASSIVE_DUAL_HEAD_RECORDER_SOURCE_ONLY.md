# VoiceChat passive dual-head recorder — source-only handoff

18 November 2026. This is instrumentation source and fake-model evidence, not
a VoiceChat-11B result. The full 12-WAV blinded listening cohort, external
controller grants, official-sample parity, and stock-versus-instrumented live
identity gate in `../VOICECHAT_DUAL_HEAD_ASSAY_PROTOCOL.md` remain required.

`passive_dual_head_recorder.py` wraps the existing QA-gated `run_two_pass`
inference runner. Its default verifier rejects a VoiceChat or DuplexSTT class
whose loaded source differs from pinned NeMo commit `097dfe9e2f55baf653b83035868bdc89849f1b47`.
It checks that `llm`, `lm_head`, and `function_head` are separate hookable modules,
then observes each official `_step_zero`, `_step_inference`, and
`_post_inference` call. It calls each original method with unchanged arguments
and returns its original object. Hooks return `None` and copy outputs to CPU.
The adapter still owns QA, parser selection, inert tool responses, and pass-2
replay. The recorder never executes a tool.

Per untrimmed frame and pass, it stores the shared LLM state, raw text-head and
function-head logits, text logits after the model's inference boosts, the
returned function logits, selected text/function IDs, and whether a function
ID was model-selected, prompted, or pre-injected. The raw head copy occurs
before the pinned forward's in-place text boost. It also saves the untrimmed
token arrays before postprocessing, each TTS step's input text IDs and output
code, both raw result channels (including unspliced audio), and the adapter's
raw parser positions, selected first call, injected response, and dry-run
decisions. The source-level data alone cannot prove that a generated function
script caused a spoken statement; pass 1 has `decode_audio=False`, while
pass-2 replay can change the trajectory.

The capture is bounded to one sample, 4,096 frames, and at most 2 GiB of
tensor payload per pass. It fails closed on missing heads, unexpected output
shape, missing/duplicate frame calls, changed pass order, or a cap breach.
The cap counts tensor payload, not Python object overhead. A real run must
budget host RAM and disk separately. `save()` writes two independent `.pt`
files after the caller binds the returned adapter trace; it never splices a
waveform. Future live runs should hash and content-address those files in a
separate reviewed receipt.

Source-only use with the existing adapter:

```python
from models.second_architecture.voicechat_calendar_fixture_v1.two_pass_adapter import (
    _default_inference_runner, run_two_pass,
)

recorder = PassiveDualHeadRecorder(model)
trace = run_two_pass(
    model,
    # Supply the reviewed all-12 v2 QA manifest, ledger, exact case WAV,
    # signed C grant when applicable, and signal loaded from that same WAV.
    inference_runner=recorder.wrap_runner(_default_inference_runner),
    **reviewed_case_inputs,
)
recorder.bind_adapter_trace(trace)
paths = recorder.save(output_dir / "passive_frames")
```

The direct `run_two_pass` API cannot itself prove `input_signal` came from the
WAV. The live launcher must retain `load_bound_wav` and its manifest/ledger
checks. Before interpreting any recorded frame, compare stock versus
instrumented text/function/codec token IDs, parser calls, and unspliced
waveforms on the pinned official sample and fixture under fixed decoding.
An observed difference fails the identity gate. The fake-model tests establish
only that the wrapper preserves their deterministic outputs.

Test command:

```sh
models/second_architecture/.venv/bin/python -m unittest \
  models.second_architecture.voicechat_calendar_fixture_v1.test_passive_dual_head_recorder -v
```
