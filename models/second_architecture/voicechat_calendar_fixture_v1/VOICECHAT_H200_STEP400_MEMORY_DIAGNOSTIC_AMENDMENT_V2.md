# VoiceChat H200 step-400 diagnostic: pre-inference measurement amendment V2

19 November 2026. **Source-only amendment; HOLD GPU inference.** This resolves two
measurement ambiguities in the [V1 prospective design](VOICECHAT_H200_STEP400_MEMORY_DIAGNOSTIC_PROSPECTIVE_V1.md)
before an executable or host freeze exists. V1 remains unchanged. All V1
source, input, early-stop, resource, review, and interpretation boundaries
remain in force except where this amendment makes the measurement definitions
more precise. This is not an executable GO.

## Define the first peak interval after model setup

The pinned VoiceChat method initializes the STT state, performs `_step_zero`,
sets up and warms the TTS model, then enters the generation loop at `t=1`.
The instrumented STT method wrapper must require that its **first** invocation
is `t=1` and must act **before** it calls the original `_step_inference`:

1. Synchronize the CUDA device. Record a `pre_step_1_baseline` object containing
   timestamp, `memory_allocated`, `memory_reserved`, NVML process memory and
   whether the monitored PID was present. Record `T` and verify `T=859`.
2. Call `torch.cuda.reset_peak_memory_stats` on the same device and record a
   second timestamp. The reset must occur after checkpoint/model loading and
   TTS warmup but before the first STT generation forward. It must neither
   empty the allocator cache nor modify any model/inference tensor.
3. Call the original `_step_inference` exactly once with its unchanged arguments
   and return its original result. The TTS wrapper makes the already specified
   post-forward `t=1` snapshot. Its interval peak therefore covers the first
   STT and TTS generation forwards, **not** checkpoint loading, STT setup,
   `_step_zero`, or TTS warmup.

The baseline object is stored in the `t=1` checkpoint record; it is **not a
seventh checkpoint**. The exact checkpoint sequence remains `1, 50, 100, 200,
300, 400`. Subsequent peak intervals begin with the reset **after** writing
the preceding checkpoint and end at the next checkpoint snapshot. Record both
reset and snapshot monotonic timestamps for every interval. If no pre-step-1
reset is recorded, label the t=1 peak `preload_inclusive` and mark the
diagnostic **incomplete**; do not interpret it as a generation-step peak or
compare it with the later intervals. A failure before reaching t=1 is an
environment/setup failure with no generation memory trajectory.

## Count shared storage across the whole carried-state union

V1's per-family de-duplicated storage totals are diagnostic views, **not** an
additive estimate of all live carried state. At every checkpoint, inventory
the union of tensors reachable from the existing STT `inference_state` and
the TTS `past_key_values` returned by that step. Compute:

- per-family tensor count, sum of logical `numel × element_size` bytes and
  unique-storage bytes;
- a **global union unique-storage byte count**, deduplicating storage shared
  across the two families, plus the intersection/shared-storage byte count;
- the number and types of unknown/unvisited containers and whether the
  inventory is complete.

Identify an underlying storage in memory by device plus storage identity and
its full capacity (for example `untyped_storage().data_ptr()` and
`untyped_storage().nbytes()`), not by tensor shape or Python object ID alone.
The pointer/identity is used only for in-process de-duplication and must
**never be serialized**. Account for tensor views using the storage's full
capacity once, while retaining logical tensor bytes separately. Handle
zero-sized or unbacked storages explicitly; unsupported storage types mark
the inventory incomplete instead of being silently omitted. The global union
is a *lower-bound inventory of the enumerated carried tensors*, not an
estimate of all GPU allocations: model parameters, workspaces, allocator
reservation and unreachable native allocations are outside it. Never sum the
two per-family unique-storage totals and call that live memory.

The independent saved auditor must check `global_unique = stt_unique +
tts_unique − shared_unique` for complete inventories, all nonnegative sizes,
the baseline/reset ordering, all six checkpoint intervals, and the actual
parent NVML polling timestamps and largest gap. The parent still targets the
V1 0.2-second cap-monitor interval; any larger observed gap must be reported
as a coverage failure or limitation and cannot be hidden by the six
time-aligned checkpoint samples. A bounded fake-cache test must include two
tensor views sharing one storage **across** STT and TTS families and verify
that the union counts it once. None of these accounting checks establishes a
leak or a safe end-to-end memory cap.
