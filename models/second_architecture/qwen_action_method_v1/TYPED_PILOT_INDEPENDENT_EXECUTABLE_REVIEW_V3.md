# Independent executable review: typed Qwen pilot V3

19 November 2026. **GO for source protocol only**, limited to the single fixed `F_L1__typed` pilot at `runs/typed_pilot_v3/attempt_001`. This review does not claim a model result. I did not load Qwen weights or run candidate inference. V1 and V2 remain HOLD.

## Review and no-weight evidence

- Reviewed committed V3 source `ead78ad` against V2 HOLD `0a989cc`. `source_identity()` passed on 22 clean, tracked source files with the pinned package and Transformers revision; parent source preflight returned `independent_go_required`, with no attempt or GO present.
- All eight `test_typed_pilot_v3` no-weight tests passed. The fake model exercises the complete `LIST`, `ASK`, `CANCEL` plus EOS teacher-forced paths, 36 FP32 sidecars, unchanged no-hook/read-only/leaf logits and states, generation-token parity, and deliberate saved-logit corruption rejection.
- Independent fake direct-worker negatives rejected a missing parent start, wrong parent PID, stale deadline, wrong pre-host hash, and missing inherited pipe before candidate import. The V3 regression test additionally intercepts any attempted torch import and confirms that GO/pre-host and even a valid-looking start without the live pipe cannot reach it.
- Exercised the reviewed parent with harmless Python sleep workers. A normal child yielded positive RSS samples and heartbeats; a forced 30-GiB RSS sample caused `rss_limit`; a short fake deadline caused `wall_limit`. No model was imported. A fake saved parent result with zero RSS samples/heartbeats failed the read-only auditor's resource/completion gate.

## Source assessment

`parent.launch()` validates the bound GO before creating the one-shot attempt. It collects and validates pre-host evidence, then writes a start receipt binding GO and committed-source identity, pre-host hash, parent PID, nonce digest, and whole-attempt monotonic deadline. `_run_worker()` owns the inherited pipe, sends the nonce and ongoing heartbeats, samples process-tree RSS and total evidence size, and kills the worker group on the 30-GiB sampled RSS, 64-MiB evidence, or whole-attempt time gate. `parent_admission()` validates the fixed attempt and all receipt fields, requires the inherited pipe and correct nonce before importing torch, arms a worker-local deadline alarm, and runs a pipe-loss watcher throughout inference. The worker repeats PID/deadline checks before model loading.

The teacher-forced method saves full-vocabulary FP32 logits for all three frozen label plus EOS suffixes on no-hook, read-only-hook, and differentiable-leaf paths. It checks both hook sites (residual 17 and 34), numerically identical forward states/logits, and full-label-plus-EOS autograd scores. The saved auditor independently reconstructs and rechecks the 36 arrays, prompt/target IDs, score and state parity, generation tokens, worker start linkage, host ordering, sampled MPS/RSS limits, and fixed FIT case. The typed-only QA amendment is bound by hash and allows no audio, other FIT, development, locked, or native tool action inference before separate human QA and GO.

Each evidence file and gzip host archive is fsynced, with parent/worker-output directories fsynced before the terminal receipt and the terminal directory fsynced after. A missing or failed terminal cannot be credited. The resource claims are explicitly **sampled peaks**, not continuous maxima. A failed attempt remains consumed.

The handshake is a local execution guard, not attestation against an operator who can forge local files, spawn a fake parent and pipe, or run arbitrary Python. I demonstrated a fake-parent admission with mocked GO validation and a forged nonce pipe; it reached `parent_admission()` but made no model import and produced no auditable parent RSS/terminal evidence. That threat is outside this source protocol. An ordinary direct `--worker` launch with GO/pre-host/start and no inherited live pipe fails before candidate import; the reviewed parent enforces caps in its normal path, and the saved auditor rejects an unmonitored run.

**Decision:** issue one source-bound executable GO for this V3 fixed typed pilot only. A future candidate must pass the separate saved auditor before any scientific claim. No audio or security finding is authorized by this GO.
