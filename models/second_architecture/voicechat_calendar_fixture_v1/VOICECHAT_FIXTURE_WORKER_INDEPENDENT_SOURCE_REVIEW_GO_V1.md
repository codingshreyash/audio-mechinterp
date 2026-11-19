# VoiceChat fixture-worker independent source review v1

Decision: **GO for the bounded source slice only; live VoiceChat model work remains HOLD.**

Reviewed commits: `49f9df1` and its wording-only follow-up `de9bf7d`. No model weights, NeMo import, or live inference were used in this review.

## Exact source reviewed

| File | SHA-256 |
| --- | --- |
| `live_parity_fixture_worker_v1.py` | `66a400e229b6469f037d6f8226481d6bf472ff72b118c725dfbb367139783b53` |
| `test_live_parity_fixture_worker_v1.py` | `0e5427bd20707a273ffbf5b50c90cb2b3e2481132724e0f821f4a609129066f6` |
| `two_pass_adapter.py` | `a5e82273c254996e42c4a80cd458e313b982442fdc0ffc09b926f0ae2eea90c1` |
| `VOICECHAT_FIXTURE_WORKER_SOURCE_HANDOFF_V1.md` | `2652a1138230a98c3d73cfdfabfc62d09dd14270a321a79cc24123d283a4dd6d` |

## Source checks

- Omitting the new `inference_runner` parameter in `run_live_case_from_audited_wav` still supplies `_default_inference_runner` to `run_two_pass`; the old default path did the same through that function's default argument. The added hook only changes behavior when explicitly supplied.
- The worker fixes arm, case `B`, Piper voice, seed type/range, and a new child for adapter high-level output. The default admission callback raises before `_qa_before_factory`, NeMo import, either factory, or output creation. An injected test-only admission must precede QA; `_qa_before_factory` runs the full 12-WAV v2 check and exact bound B/Piper receipt before either factory. The real high-level adapter repeats the QA gate before its NeMo import and model call.
- The selected WAV, QA receipt, manifest, and ledger digests returned by the precheck must equal the four corresponding entries in caller-supplied `input_hashes`. This establishes internal equality to fresh QA observations. **It does not authenticate a saved source plan:** the current admission callback is closed, and a future production admission must verify that the request hashes came from the reviewed plan and recheck other source/checkpoint bytes.
- The stock response is bounded, non-symlink, one-line canonical JSON with `executed: false`, and its observed SHA-256 must equal the caller's `response_sha256` before either factory. The worker does **not** derive that expected hash from the saved adapter decision. The future supervisor must do so; `de9bf7d` correctly leaves this as a live blocker.
- Adapter mode wraps the pinned default runner in `PassiveDualHeadRecorder`, passes it to the QA-gated high-level adapter, requires exactly two observed calls with `decode_audio=False` then `True`, binds parser trace to the recorder, and saves raw trace/call/frames through the reviewed lossless capture codec under the arm root. Stock modes call `capture_stock_helper`; its source wraps the unchanged official `run_fc_offline_inference` helper and records both passes. The `stock_no_tts` mode disables only the TTS wrapper.
- The prepared arm cannot already contain `raw_capture` or `adapter_high_level`. The saved capture codec uses `trusted_root=arm_dir`. Whole-arm exclusive creation, ancestor protection, and total output budget remain supervisor duties.

## Independent negative probe and CPU checks

The seven new worker tests and seven saved-capture tests passed (`python -m unittest ...test_live_parity_fixture_worker_v1 ...test_live_parity_saved_capture_v1`), with 14 total tests, no model weights. `py_compile` and `git diff --check 49f9df1^ 49f9df1` passed.

An additional fake-only probe set `input_hashes["tools"]` to 64 `f` characters while replacing the request's tool file with different JSON. The test-only admission echoed that request and the fake QA returned matching selected-QA values; the stock arm reached the fake official and model factories and returned `synthetic_test_only_no_model_work`. This confirms that the worker itself does not validate non-QA plan hashes. It is **not** a bypass of the default closed gate or the real human QA checker; it is a concrete obligation for production admission and pre-use byte checking.

## Remaining live blockers

Before any real model load, independently review and bind a production admission and supervisor that authenticate the full source plan, anchor/grants, checkpoint/Nano identity, official sample success, fixed seed and arm order; derive each stock expected-response hash from the saved adapter decision; recheck prompt/tool/helper/template bytes before use and after use; launch fresh serial processes; cap the whole output tree, runtime, and VRAM; and compare reloaded stock/adapter captures including final trimmed codec codes. This review grants no live model admission and makes no security finding.
