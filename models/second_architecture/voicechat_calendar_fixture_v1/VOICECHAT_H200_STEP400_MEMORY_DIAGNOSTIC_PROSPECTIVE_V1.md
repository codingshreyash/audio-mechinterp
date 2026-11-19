# VoiceChat H200 step-400 memory diagnostic: prospective design V1

19 November 2026. **Source-only design; HOLD GPU inference.** This is a bounded
diagnostic of the pinned public random-number example, not a calendar test,
official-sample completion, model-behavior result, or security finding. No
instance is to be started from this document. A new executable, no-model tests,
independent source review, fresh host freeze, independent freeze review, and a
one-shot GO are required before an H200 launch.

## Why this experiment exists

The independently audited H200 attempt 002 loaded 1,617 checkpoint tensors and
reached pass-1 step 650 of 859. Its supervisor sampled 130,540 MiB, above the
reviewed 130,000-MiB cap, and killed the child. It retained no complete model
response. The pinned STT path recomputes a full prefix without cache, while the
TTS path carries `past_key_values`; neither source fact diagnoses a leak. The
two full-prefix 131,072-vocabulary heads account for under 0.7 GiB of simple
FP32 payload at step 650 and cannot alone explain the sampled peak. See the
[attempt audit](VOICECHAT_H200_PUBLIC_SAMPLE_ATTEMPTS001_002_INDEPENDENT_RESULT_20261119.md),
[memory source audit](VOICECHAT_H200_MEMORY_TRAJECTORY_SOURCE_AUDIT_20261119.md),
and [shape budget](VOICECHAT_H200_SOURCE_ONLY_SHAPE_BUDGET_20261119.md).

The question is whether **live PyTorch allocations**, **interval peaks**,
**carried cache/state tensors**, or **NVML-reported process memory outside the
PyTorch accounting** rise by different amounts during the first 400 steps. This
is an instrumented same-input trajectory, not a byte-identical execution of
the prior uninstrumented child.

## Fixed input and source boundary

- Use the pinned NeMo Speech checkout at
  `097dfe9e2f55baf653b83035868bdc89849f1b47`, VoiceChat snapshot
  revision `443794ea956ef0065f001967ffd00e77f519cb39`, single checkpoint
  blob 44,382,749,892 bytes/SHA-256
  `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`,
  and config SHA-256
  `2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5`.
  Revalidate the Nano revision and six metadata hashes, environment, native
  FFmpeg, Python package inventory, actual disk floor, and model-card pins as
  in the independently reviewed H200 V5 preflight. The previous freeze and
  review key are consumed and cannot authorize this attempt.
- Use only the official `sample_fc.wav` (384,652 bytes; SHA-256
  `265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136`),
  official random-number response JSON (82 bytes; SHA-256
  `e36ae164cdc14a0b9bbed7b4be2b37223dc3730b39de1edafb99e73638e2b60f`),
  default random-number tool definition, prompt template, system message,
  `sample_id`, greedy/default inference arguments, and the first pass with
  `decode_audio=False`. No calendar WAV, dispatcher, real tool, altered
  checkpoint, or different prompt is admitted. The fixture JSON is a supplied
  response, not a model-generated function call.
- Leave the pinned NeMo checkout and official helper unchanged on disk. Build a
  separately source-reviewed wrapper around `build_model`, official WAV/prompt
  preparation, and `run_fc_offline_inference` for **pass 1 only**. A reviewer
  must compare every argument to the official helper's pass-1 invocation.
  The wrapper may replace only this model instance's bound
  `stt_model._step_inference` and `tts_model.infer_codes_one_step` with probes
  that call the originals exactly once with unchanged arguments and return
  their unchanged outputs until the planned stop. No cache toggles, precision
  changes, inference shortcut, training mode, or altered model code are allowed.
  Fake-model tests must prove call count, argument identity, return identity,
  order, and stop behavior. No loaded-model parity claim is possible from this
  partial diagnostic alone.

The [pinned loop](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/models/nemotron_voicechat.py#L627-L665)
calls STT and then TTS once at each `t` from 1 through `T-1`; the official
[two-pass helper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/utils/offline_voicechat.py#L171-L289)
does not parse/replay a function response before pass 1 returns. The STT probe
records the current `t`, verifies monotone one-call steps and `T=859`, and
borrows the existing inference-state reference without copying or retaining
its tensors. The TTS probe observes the unchanged result immediately after the
original TTS call at steps **1, 50, 100, 200, 300, 400**. Thus each checkpoint
is *after both forwards at that step, before the caller writes that step's
`gen_codes`*. At step 400 the probe writes and fsyncs its final checkpoint,
then raises a dedicated diagnostic-stop exception. The wrapper catches only
that exception, proves the six-checkpoint sequence and no second pass, and
exits with a distinct `diagnostic_early_stop_complete` receipt. A normal
sample-success label or partially decoded answer is invalid for this run.

## Exact measurements

At every checkpoint, synchronize the active CUDA device before reading
`torch.cuda.memory_allocated`, `memory_reserved`,
`max_memory_allocated`, and `max_memory_reserved`. The maxima cover the interval
since the prior checkpoint; reset peak stats **after** writing the current
checkpoint. Record all values in bytes, device ID, `t`, `T`, monotonic elapsed
time, and a bounded NVML per-process used-memory sample for the model process
and its descendants, with the sample timestamp and lag from the CUDA counters.
The runner must record whether the NVML sample includes this PID; missing or
ambiguous telemetry fails the diagnostic rather than silently writing zero.
The parent independently polls process-tree NVML memory at least every 0.2 s
for the cap; the six checkpoint samples answer a different, time-aligned
question. No CUDA allocator cache clearing is allowed during the trajectory.

At each checkpoint, count **metadata only** for tensors in the existing
inference state and returned TTS `past_key_values`: container path, shape,
dtype, device, logical bytes, and de-duplicated storage bytes per named
family. Support the pinned cache/container types explicitly, bound traversal
depth and node count, detect cycles, and mark any unvisited/unknown container
as `incomplete_inventory`; never infer cache size from an incomplete walk.
Do not serialize tensor values, storage addresses, token IDs, decoded text,
audio, weights, prompts, or model outputs. Inspection must not clone, detach
to CPU, retain a new tensor reference beyond the checkpoint, or mutate a
tensor. A no-model allocation test must check that probes do not retain
per-step outputs. Save a compact JSONL checkpoint ledger and bounded standard
logs with hashes; use `fsync` and an exact file inventory before terminal
status. Every failed/partial run remains visible with its reason and last
valid checkpoint.

## Resource and admission gates

Use a **new** output ID and nonexistent attempt directory. Pin the wrapper,
tests, source hashes, runtime package inventory, exact sample/response bytes,
checkpoint, command, child environment, six sample steps, and caps in a new
external freeze. A different reviewer must verify the executable and host
freeze, including ignored-file inventory (the ignored `.pyc` caused attempt
001), before signing a one-attempt GO. A signed V5 public-sample freeze is not
reusable. The diagnostic's independent auditor must rehash every record and
reconstruct checkpoint order, cap coverage, `T`, and the terminal/partial
status without model weights.

Keep the H200 policy's **130,000-MiB sampled GPU cap**, **30-GiB actual free
disk floor after staging**, and **1,800-second wall cap**. Tighten retained
evidence to at most **64 MiB**, with a protected terminal reserve. A separate
parent watchdog must kill the entire process group on cap breach, telemetry
failure, child failure, missing heartbeat, or excessive checkpoint/log bytes;
the child's cooperative step-400 stop alone is insufficient. Record sampled
peak, timestamps, process return code, checkpoint count, and whether logs are
complete. A transient unsampled peak remains possible. Do not raise the GPU
cap or run past step 400 under this protocol. A pre-400 resource stop is a
failed diagnostic, never a censored success. Preserve the prior H200 archive
and attempt directories unchanged.

## Decision rule and limits

Report all six checkpoints and every interval, including numerical deltas
from step 1. Compare live `allocated` with interval `max_allocated`, reserved
memory, NVML memory and the de-duplicated carried-state estimate. Rising
interval peaks with roughly flat live allocation would support a temporary
work hypothesis; rising live allocation plausibly accounted for by growing
cache/state would support retained-state growth; an unexplained live rise
requires a separate bounded ownership/allocator study. NVML-minus-PyTorch is
only a rough accounting gap because contexts, libraries, allocator behavior,
sampling lag and other direct allocations differ. Neither trajectory proves a
leak, predicts step 859, explains pass 2, or warrants a larger memory cap.
No function proposal, audio output, user-facing action, or security outcome is
scored here. Decide a next compute configuration only after an independent
review of the saved trace and its measurement completeness.
