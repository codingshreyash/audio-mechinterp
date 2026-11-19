# Independent saved-auditor source review: HOLD

19 November 2026. Review target: `voicechat_step400_saved_auditor_v1.py`
at commit `202b3d4`, compared with the pinned step-400 probe core, exact
worker receipt, parent supervisor poll/terminal schema, and prospective
V1/V2 diagnostic protocol. **Source-only HOLD.** No model was loaded and no
GPU instance was started.

## Checks that passed

The auditor requires the six expected regular files, a clean parent result,
untruncated evidence, the six ordered checkpoints, a pre-step-1 baseline,
the worker/core completion receipt, ledger and parent file hashes, a pinned
identity equal to an independently supplied expectation, and parent poll
samples with gaps no greater than 0.2 seconds. It rejects an external parent
failure receipt and does not call NeMo or run the model. The project's 14
no-model auditor tests pass with `ResourceWarning` treated as an error.

These are *saved-file consistency checks*. The worker's
`no_second_pass: true` and identity fields are self-reports; the auditor
cannot independently attest that the pinned code ran or that no second pass
occurred. A reviewed executable launcher, host freeze, and authenticated
parent/worker provenance remain separate prerequisites. The existing parent
supervisor executable HOLD is unchanged.

## Reproducible false PASS cases

Starting from the auditor test module's `write_fixture`, I mutated one field,
updated the ledger SHA/byte count when needed, and resealed the parent file
inventory exactly as that module's `reseal_parent` does. Every case below
returned `status: PASS`:

| Case | Mutation | Why a genuine source trace cannot have it |
| --- | --- | --- |
| NVML duration | At checkpoint 100, set `nvml.query_seconds=100.0`, leaving `nvml_lag_seconds≈0.001`. | The probe timestamps the NVML query after the CUDA snapshot and at query completion; query duration must fit within the snapshot-to-sample lag. |
| Parent wall time | Set `elapsed_monotonic_seconds=0.001`, while first parent poll begins at 0.9 and last ends at 1.26. | The parent starts before the first poll and finishes after the last; elapsed time must be at least the 0.36-second poll span. |
| CUDA cap/accounting | At checkpoint 100, set allocated to 140,000 MiB, reserved to 141,000 MiB, max allocated to 140,500 MiB, and max reserved to 141,500 MiB. | The checkpoint reports live CUDA allocation above the 130,000-MiB policy cap while its own NVML sample claims only 1 KiB and the parent poll peak claims 123 MiB. The values are internally ordered but mutually contradictory. |
| Exact step type | Change checkpoint `t` from integer 100 to float 100.0. | `tuple(row.get("t")) == STEPS` accepts numeric equality, but the core constructs integer step indices. The saved schema is not exact. |

I also changed the carried-state unique CUDA storage to 1 TiB while keeping
CUDA allocated at 100 bytes; after arithmetic resealing it returned PASS.
This is an accounting contradiction if all traversed CUDA storages are
PyTorch-allocated. That assumption should be made explicit before adding a
strict cross-counter rule. It is not needed for the HOLD decision: the NVML
duration, parent elapsed span, CUDA cap, and step type cases suffice.

The auditor's owned tests do not exercise these contradictions. Their passing
baseline fixture itself has `nvml.used_bytes=1024` while
`cuda.allocated_bytes=100`, so it remains a synthetic schema example rather
than a realistic GPU telemetry fixture.

## Minimum changes for rereview

1. Check `nvml.query_seconds <= nvml.sample_monotonic - snapshot_monotonic`
   at each checkpoint and the analogous baseline interval, allowing only a
   documented clock tolerance. Check parent elapsed time against the first
   poll start and last poll end.
2. Bound every checkpoint and baseline CUDA allocation/reservation/peak by
   the explicit 130,000-MiB policy, and establish a defensible relation to
   same-time process NVML usage. If allocator counters can exceed the sampled
   parent cap transiently, record that as a cap violation rather than PASS.
3. Require exact integer types for `t`, `T`, and other discrete fields before
   comparing values; add mutation tests that rehash and reseal every affected
   receipt.
4. Keep the saved-audit conclusion scoped to internally consistent evidence.
   Authentication of the code/parent/host remains an external executable
   review requirement, not something a worker boolean can prove.

Rerun the full no-model tests and the four false-PASS mutations after fixes.
Do not use this auditor to declare a complete diagnostic trace until it
rejects the contradictory bundles.
