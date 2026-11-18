# Independent source review — Voxtral non-beaver continuation gate

18 November 2026. I independently reviewed corrective commit `50aacbd83f107f46e110c40af3d6ca137b41c747` against the prospective continuation protocol `7d8d340`, its conceptual GO `3461d1f`, the original source HOLD `5a5cd9f`, and the narrow rereview HOLD `becf753`. The exact source SHA-256 values are:

- V2 continuation gate `1fbc29e7851491d53c1c8245a7acd385563a1491435d21790a4370a7aa3492be`.
- V2 worker entrypoint `db2289aefa99c0820485ffcfbc2db751da95e22913a6d8f81aac216739f5a185`.
- Main V2 gate tests `1c76a06dfa10f51eae4ccdc9a6d3ceb6acede28dcddd724db232768bec636e66`.
- Preserved independent regression tests `22e3376cf5890b21ea7bdf1a0a6679caf99c925fc5c725c95ab3eb8091d81c9c`.

Decision: GO for non-beaver FP32 continuation gate and worker entrypoint source only; model HOLD until continuation freeze.

## Independent checks

`./.venv/bin/python -m unittest test_voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 test_voxtral_fp32_nonbeaver_continuation_source_hold_v1 -v` passed **15/15** CPU tests. The four original fail-open cases now reject stale live/saved pre-host logs, a stale post-host log, and a backdated new start; the final-slot variant also rejects a backdated start. The last test-only correction replaced assertions about the *current* absence of source-review and freeze files with controlled missing-dependency fixtures, so creating those required files no longer invalidates the suite.

The V2 gate revalidates the original one-shot reservation, V1 source/freeze hashes, 27 exact Piper raw files, the signed-gap exception on byte-pinned `_01` only, and both saved-data audits. My read-only rerun of `_verify_manifest()` and `check_prior(2, reservation)` returned `raw_file_count=27` and completed model work `290.7225837910082` seconds. It did not inspect the pilot's efficacy metrics. The original V1 gate and receipts remain unchanged.

The new host check requires external power and an open lid. It stores the exact `pmset` output in a content-hashed, lossless archive and requires a timestamped event before each audited interval plus a recent terminal assertion snapshot. It then scans the interval for Sleep, DarkWake, Wake, or thermal transitions. The revised parser accepted a real contemporaneous `pmset -g log` snapshot in a read-only source check; that check did **not** assert the Mac was ready to run. Runtime pre- and post-slot probes still have to pass independently for each slot. The local OS log is trusted evidence, not cryptographic proof that the OS could never omit an event.

For `_02` and `_03`, the gate compares the start UTC to the **actual predecessor's** finish UTC at launch, worker admission, preview completion, and final completion. The saved-data auditor is rerun on a new completed slot, including `_03`; both new slots retain the five-second UTC/monotonic checks. The worker shim calls V2 admission exactly once, creating one exclusive no-retry claim, then invokes the frozen V1 `runner.run_context`. No numerical runner code was changed. The original process, MPS, disk, stream, worker-output, campaign-output, audit-time, and cumulative model-work caps remain enforced in the continuation gate.

## Remaining gates and scope

This is source-only approval. The gate must still bind this review in a separately committed external continuation freeze, receive an independent freeze review, and verify a stable externally powered Mac before `_02`. Neither `_02` nor `_03` was started by this review. The two completed Piper contexts are a numerical FIT pilot on already inspected speech, not a held-out audio lens result or a security finding.
