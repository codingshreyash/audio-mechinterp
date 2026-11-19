# VoiceChat step-400 probe core handoff V1

19 November 2026. **Source-only, fake-tested core. HOLD GPU inference.**

The implementation is [`voicechat_step400_probe_core_v1.py`](voicechat_step400_probe_core_v1.py), with no-model tests in [`test_voicechat_step400_probe_core_v1.py`](test_voicechat_step400_probe_core_v1.py). It implements the measurement portion of the reviewed [V1 prospective design](VOICECHAT_H200_STEP400_MEMORY_DIAGNOSTIC_PROSPECTIVE_V1.md) and [V2 amendment](VOICECHAT_H200_STEP400_MEMORY_DIAGNOSTIC_AMENDMENT_V2.md). It does not launch a model or an instance.

## Implemented

- `run_pinned_pass1_only` calls an injected `run_offline_inference` once with the exact six arguments in the pinned official helper's first pass: the model; identical `input_signal`, `input_signal_lens`, `prompt_tokens`, and `prompt_token_lens` objects; and `decode_audio=False`. It never calls the helper's function-call parsing or second pass.
- `Step400Probe` wraps only the supplied model instance's STT and TTS step methods. It requires sequential steps and `T=859`, calls each original once with unchanged arguments, synchronizes CUDA before measurements, captures the pre-step-1 baseline and peak reset, and snapshots steps 1, 50, 100, 200, 300, and 400 after the TTS call. It resets peak counters after each intermediate snapshot. A dedicated `DiagnosticStop` follows the fsynced step-400 record. The context manager restores both instance methods.
- Each checkpoint records allocated/reserved and interval peak CUDA bytes, process-tree NVML bytes and timestamps, device and interval timing, and a bounded metadata-only walk of the existing STT state and returned TTS cache. Storage identity remains in memory only. Family totals, shared storage, and the global union deduplicate tensor views across both families. Unknown containers are counted and make `inventory_complete` false.
- `CheckpointLedger` creates a new JSONL path, enforces per-record and ledger size limits, fsyncs every record, and produces a digest on completion. The core receipt distinguishes a complete inventory from an incomplete one and makes no model-behavior claim.
- Fake tests cover call count, argument and return identity, order, six checkpoints, timestamp ordering, peak resets, planned stop, method restoration, no retained step result, cache metadata, cross-family storage sharing, unknown cache handling, and ledger reuse/order rejection.

## Missing before any executable GO

1. Build a separate launcher using the pinned NeMo checkout and the official WAV, prompt preparation, model setup, and `run_offline_inference`. Verify every call argument and the pinned runtime/cache classes against the exact hosted revision. The core deliberately accepts a callback and therefore cannot attest to the callback's origin.
2. Build an independent parent supervisor with process-group kill, 130,000 MiB sampled GPU cap, NVML polling targeted at intervals no longer than 0.2 seconds, 30 GiB actual disk floor, 1,800-second wall cap, heartbeat and telemetry failure rules, and 64 MiB evidence cap with terminal reserve. The checkpoint NVML sampler is not this watchdog.
3. Seal partial failures, bounded stdout/stderr, exact file inventory, hashes, child environment, command, runtime package inventory, source pins, and host freeze. The core's ledger alone is insufficient evidence.
4. Obtain an independent executable source review, a fresh host freeze and independent freeze review, then a one-shot GO. The previous H200 freeze and approval cannot authorize this diagnostic.

No GPU/model inference, instance start, or alteration of prior attempts was performed while building this core. A complete core receipt only means the fake-tested instrumentation reached its planned stop; it is not permission to run the diagnostic or evidence of a memory mechanism.
