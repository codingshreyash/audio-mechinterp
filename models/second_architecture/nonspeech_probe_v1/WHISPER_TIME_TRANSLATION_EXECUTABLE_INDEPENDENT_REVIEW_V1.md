# Independent executable review: Whisper time translation V1

19 November 2026. **PASS for a separately committed GO to run the three fixed, bounded attempts under the trusted local-owner assumption.** This is a source-only review. I did not load model weights or run this experiment. A GO and a saved audit are still required before any empirical result can be claimed.

## Exact source and admission reviewed

- The prospective protocol and frozen plan remain at `ab71f52`; the plan SHA-256 begins `b1aa22f0b0c9`. The rendered source manifest SHA-256 is `eec732fb24a4b238ad0666d52d168f5ba41ec38c9d315a057a39e0d53be1435b`.
- I reviewed the bounded parent and gate at `4ae260a`, the worker and saved auditor at `24dc67b`, the direct-worker test at `bd98aa7`, the full baseline WAV/mel binding and reaudited joint decision at `974ac4e`, and the import-closure fixes at `9d3ccff` and `970098e`.
- The three stage admissions were refreshed and committed at `784ffc7`. Recomputing `gate.admission(stage)` in the admitted environment matched each committed JSON exactly, including all 25 source hashes, the frozen plan, the pinned checkpoint and processor assets, and the inherited runtime source and version pins. The named source files were tracked and clean at review time.
- The admitted Python environment is `models/second_architecture/.venv` with NumPy 2.3.5, PyTorch 2.15.0, and Transformers 5.20.2. I independently rebuilt both original V4 zero/noise mels with the pinned processor in this environment; each full tensor and mask matched its saved V4 counterpart bitwise. A different local Garak environment has different processor numerics and is rejected by the inherited runtime pin before weights load.

## No-model checks

The six new test modules passed **27/27** tests in the admitted environment. The inherited independent host and parent-monitor modules passed **9/9** tests. The new suite covers the frozen plan and WAVs, fixed paths, source census, direct worker refusal without a parent-created attempt, heartbeat and process-group monitoring, RSS and evidence limits, stage call caps, partial progress rejection, and the saved decision rule.

I also ran independent fake-process probes without model inference:

1. Invoking the worker CLI directly with no fixed attempt stopped before weights. Each current stage GO is absent, and `validate_go` rejected it. In a temporary mocked-path check it rejected a stale review hash, a stale source identity, and an uncommitted forged GO.
2. A simulated failure after attempt creation wrote `parent_result.json` as `invalid_or_partial` and cleared the lock. A simulated thermal probe error and evidence-cap hit killed fake child processes and recorded `host_probe_failed` and `evidence_limit`, respectively.
3. The gate's source census now includes the prior causal patch verifier, its imported parent, and both prior plan modules used by the new worker or auditor. The saved auditor recomputes every baseline's full mel and mask from its frozen WAV with the pinned processor; a changed boundary mel or WAV is rejected. The claim-bearing joint CLI reaudits all three fixed attempts rather than trusting editable PASS summaries.

## Scope of PASS and remaining limits

The schedule is fixed at 28 original calls and at most 62 calls for each offset. The parent uses one attempt path and GO per stage, a 1,800-second whole-attempt deadline, process-tree RSS below 4 GiB, retained evidence below 1 GiB, lid-open and thermal gates, battery at least 80% initially and above 70% thereafter, and host probes no more than 15 seconds apart. The saved auditor rejects incomplete progress, missing EOS when complete output is required, incorrect row patches, failed controls, and incomplete resource receipts. The strict joint decision requires original viability and both translated layouts; no single stage can establish event tracking.

The heartbeat proves that a live parent process supplied the expected pipe and nonce. It **does not authenticate the bounded parent against a deliberate same-UID local launcher**. I demonstrated that an arbitrary local process can fabricate a launch receipt and matching pipe to pass the inherited guard. The gate, one-shot paths, and saved audit protect the intended trusted local-owner workflow; they are not a privilege boundary against a local owner who can forge files and processes. Reports must not describe this as cryptographic parent authentication. This caveat does not change the source-only PASS for the intended execution route.

Sampling cannot prove that an unsampled, very short memory or host-state excursion never occurred. The review and no-model tests do not establish that any translated pair retains its clean contrast, that any window transfers a complete donor path, that a timing-linked result will appear, or that any security problem exists. Those questions require the bounded attempts and independent saved audits.
