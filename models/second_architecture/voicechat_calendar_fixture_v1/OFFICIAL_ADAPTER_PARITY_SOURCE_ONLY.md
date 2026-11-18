# VoiceChat stock/adapter parity: source-only capture and comparison

18 November 2026. `official_adapter_parity.py` is a low-level identity-control
utility. It has **not** run NVIDIA VoiceChat. It cannot certify a human-audited
WAV, checkpoint, model reset, or security finding by itself.

`capture_stock_helper` invokes the pinned official two-pass function and
temporarily wraps its `run_offline_inference` reference to save the untouched
pass-1 and pass-2 result dictionaries, exact call arguments, and optional TTS
codes. The wrapper is restored in `finally`, including on failure. The official
helper still receives a prewritten `api_response`; for fixture parity the caller
must supply the inert response selected by the QA-gated adapter's pass-1 call.
It must not use a live calendar or interpret that response as a model decision.

`compare_stock_adapter` requires two TTS-wrapped stock runs, a third stock run
without that wrapper, the QA-gated adapter's trace, and its two captured
`inference_runner` kwargs dictionaries. It checks both passes' common inputs,
exact discrete text/function/audio-token results, exact injected call/response
token tensors and steps, the official first-call parser result, and all TTS
codes available in the wrapped arms. It compares the **unspliced** pass-2 PCM
samples. Stock repeat, stock wrapper versus no-wrapper, and stock versus
adapter must each have maximum absolute PCM difference no greater than the
predeclared tolerance of `1e-4`; a larger stock-repeat difference fails the
control rather than widening the threshold. Any discrete difference raises.

The caller still has to enforce, in this order: complete 12/12 human listening
QA and controller grants; immutable pre-inference anchor; pinned NeMo and
checkpoint preflight; unchanged official-example run on NVIDIA's sample;
one exact QA-approved fixture WAV and prompt; identical model state and RNG
reset before each of the four arms; and independent review of the saved raw
captures. No local function can prove a listener's identity, external anchor,
model-state reset, or that the official sample ran. This utility deliberately
has no model-loading CLI and does not bypass the existing QA-gated live entry.

Run the source-only fake-output tests from the repository root:

```sh
python3 -m unittest models.second_architecture.voicechat_calendar_fixture_v1.test_official_adapter_parity -v
```

The tests execute the exact pinned NVIDIA helper function by AST against a fake
model and the real local `run_two_pass` adapter against the same fake outputs.
They cover a valid list call, a no-call case, discrete and injection changes,
stock audio instability, a TTS-wrapper effect, and restoration after failure.
