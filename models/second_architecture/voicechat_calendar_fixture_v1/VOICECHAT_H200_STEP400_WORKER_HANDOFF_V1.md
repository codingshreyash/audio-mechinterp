# VoiceChat step-400 diagnostic worker: source handoff

19 November 2026. **Candidate constituent only. No executable GO.** No model
was loaded for this handoff. The stopped H200 was not restarted.

`voicechat_step400_worker_v1.py` follows the pinned official public
`offline_voicechat_fc_infer.py` example at NeMo Speech commit
`097dfe9e2f55baf653b83035868bdc89849f1b47`. It imports the official
example's `DEFAULT_SYSTEM_MESSAGE`, `DEFAULT_TOOLS`, `DEFAULT_TEMPLATE`,
`build_model`, `load_wav_16k_mono`, `render_fc_system_prompt`, and
`encode_system_prompt`; it loads only official `sample_fc.wav`. It verifies
the official response JSON bytes but never sends them to the model, because
the official helper uses that response only after the first pass completes.

The exact first-pass call is in
`nemo/collections/speechlm2/inference/utils/offline_voicechat.py` lines
188–195 at that commit: `run_offline_inference(model, input_signal=...,
input_signal_lens=..., prompt_tokens=..., prompt_token_lens=...,
decode_audio=False)`. The worker supplies the pinned helper's
`run_offline_inference` to `run_pinned_pass1_only`, which makes this call and
stops after the TTS forward at step 400. It does not call
`run_fc_offline_inference`; therefore it cannot parse a tool call, inject a
response, run pass 2, or save an answer/audio from this path.

The worker starts a byte-1 heartbeat on the inherited parent pipe before
source/weight verification or model load. It hashes the five pinned NeMo
source files, public WAV and response, config, and 44,382,749,892-byte
weight. It writes only `checkpoints.jsonl` and, after all six checkpoints,
`worker_completion.json` in the parent-created attempt directory. The
completion binds source/input/checkpoint hashes, hashes of the exact default
tool/system/rendered prompt, and the core completion. It includes no prompt,
token, tensor value, decoded text, audio, or tool proposal. A failed child
does not write a success receipt; the parent must issue the failure receipt.

## Boundaries for executable review

The worker checks the pinned NeMo commit and exact five source files, but
does **not** independently establish the entire clean worktree, Nano cached
snapshot/metadata, package inventory, native FFmpeg runtime, one-GPU host,
free disk, parent command/environment authenticity, or resistance to a
same-UID malicious child. A fresh external host freeze and independently
reviewed exact launcher must do that. The existing parent supervisor remains
a source-only constituent with an executable HOLD in its V2 review; this
worker does not lift that HOLD. A child could be run directly by fabricating
an inherited pipe, so this file must not be treated as admission control.

No-model verification:

```sh
models/second_architecture/.venv/bin/python -m unittest \
  models.second_architecture.voicechat_calendar_fixture_v1.test_voicechat_step400_worker_v1 -v
```

The four tests check consumed evidence rejection, exact official helper
preparation arguments, wrong-tool rejection before model load, and a single
completion receipt with the exact first-pass callback. They do not establish
runtime parity or a complete memory trajectory.
