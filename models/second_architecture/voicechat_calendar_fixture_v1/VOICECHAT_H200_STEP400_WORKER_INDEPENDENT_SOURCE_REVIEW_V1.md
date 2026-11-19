# Independent source review: VoiceChat step-400 diagnostic worker V1

19 November 2026. **PASS as a source-only constituent at `f621096`; HOLD executable GO.** I reviewed the worker, its four no-model tests, the V1 protocol and V2 amendment, and the official NeMo Speech example/helper at pinned commit `097dfe9e2f55baf653b83035868bdc89849f1b47`. I did not load the checkpoint, launch a GPU, or verify a live worker trace.

The four worker tests pass in `models/second_architecture/.venv` with warnings treated as errors:

`models/second_architecture/.venv/bin/python -W error -m unittest models.second_architecture.voicechat_calendar_fixture_v1.test_voicechat_step400_worker_v1 -q`

## Source comparison

The [official FC example](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/offline_voicechat_fc_infer.py#L90-L142) defaults to its local template, system message, one `generate_random_number` tool, `sample_fc` identifier and CUDA device. It builds the model, renders the prompt, loads the WAV, encodes the prompt, and calls the two-pass FC helper. The worker imports those same pinned definitions and helper functions, accepts no prompt/tool/sample override, checks `sample_fc.wav` and the response JSON by size/hash, and prepares the same model/WAV/prompt arguments. The worker renders the prompt *before* building the model, unlike the example; rendering is CPU template work, but this is a preparation-order difference and not a byte-identical replay. The checkpoint path, official template, WAV, prompt tokens and lengths remain the same arguments to pass 1.

The [official FC helper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/utils/offline_voicechat.py#L155-L262) starts with `run_offline_inference(model, input_signal=..., input_signal_lens=..., prompt_tokens=..., prompt_token_lens=..., decode_audio=False)`. The worker passes that pinned function to the reviewed probe core, which invokes precisely those first-pass arguments with the helper's default temperature, top-p and repetition penalty. It never calls the two-pass FC helper. The core's dedicated stop is after the original TTS forward at `t=400`, before pass 1 can return. Thus this path cannot reach the helper's tool-call parsing, response injection, audio decode or pass 2. No partial user answer or tool proposal is claimed.

The official example parses the 82-byte response JSON before model load. The worker verifies its exact pinned bytes but does not parse it. That is sufficient for this **pass-1-only memory diagnostic**: the official FC helper cannot inspect the response until after pass 1 returns, and the planned stop prevents that return. It is not a reproduction of the complete two-pass example.

The worker checks the NeMo HEAD and five source-file hashes, sample/response bytes, config hash, and full 44,382,749,892-byte weight hash before model import. Its completion stores only hashes/metadata and the core's six-step ledger receipt; it does not serialize prompt text, audio, tokens, tensor values or generated outputs. The inherited pipe heartbeat starts before the weight hash and model load, while the separate parent enforces liveness and GPU/resource caps. A child failure before the core completes produces no worker success receipt.

## Limits before executable GO

- The receipt fields `voicechat_revision` and `nano_revision` are pinned **expected constants**, not observed verification of the local VoiceChat/Nano snapshot. The worker does not verify the six Nano metadata hashes, clean NeMo worktree, ignored files, package/native FFmpeg inventory, one-GPU host, actual disk floor, or exact launch command/environment. The fresh external freeze and independent review must do so; a saved auditor must not treat these two constants alone as runtime attestation.
- The worker module has a callable CLI and can be run with a fabricated pipe. Its docstring is a warning, not an admission barrier. An exact launcher must bind the worker path and hash, arguments, PYTHONPATH, inherited descriptor, parent supervisor, output ID, and one-attempt freeze. The supervisor's source-only review remains an executable HOLD.
- The four fake tests establish argument plumbing and receipt shape but do not exercise the real NeMo import, actual `T=859`, step-400 stop, CUDA/NVML timing, checkpoint-state inventory, cap coverage or full model output. The saved trace requires independent audit before any result claim. A synchronous NVML call can still hang the parent; no hard wall-time claim follows from source inspection.

**Decision:** Accept this worker as a source-only candidate for the fixed public-sample diagnostic. Keep executable and scientific-result gates closed until the exact launcher/freeze, reviewed parent, live host checks, and saved-result audit exist. This review does not establish a VoiceChat model behavior or security finding.
