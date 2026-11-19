# VoiceChat fixture worker source handoff v1

Status: **source increment for independent review; live model work remains HOLD**.

This increment adds `live_parity_fixture_worker_v1.py` and an optional
`inference_runner` hook to the high-level two-pass adapter. The hook defaults
to the existing runner. In the adapter arm, the worker wraps that runner with
the passive recorder, sends it through the existing QA-gated adapter, and
saves the raw result and observer frames with the reviewed lossless capture
codec. The high-level adapter writes to a *new child directory* beneath the
supervisor's already prepared arm directory. The stock arms use the pinned
official helper with a bounded, hash-bound, inert response file. The two
TTS-wrapped arms and the no-TTS arm use the same helper path.

The default admission callback always raises before NeMo import or model
creation. A separate QA precheck, which calls the real 12-WAV v2 cohort gate
and exact case-B receipt verifier by default, also runs before either
factory. The tests inject fake gate, QA, model, helper, adapter and recorder
objects. They create no actual human listening record, anchor, grant or model
output. A test-only status is the only status this worker accepts.

## Verified in this increment

- Default admission fails before any factory or artifact write.
- Admission and QA checks run before NeMo/model factories; wrong QA source-plan
  hashes and wrong stock response hash fail at that boundary.
- Adapter capture observes exactly two passes in the fixed decode order and
  persists re-loadable raw tensors and observer frames.
- Three stock arm variants use an inert response and persist raw stock helper
  results. Non-inert response bytes are rejected before model creation.
- A used arm directory cannot be silently reused.
- 7 worker tests and 7 saved-capture tests passed on CPU; no weights loaded.

## Required before a live VoiceChat experiment

1. Independent source review of this exact worker, optional callback path and
   tests; bind their hashes into a new executable freeze. Any source change
   after that review requires renewed review.
2. A production admission implementation that authenticates the external
   human listening QA, independent anchor and grants, source plan, checkpoint
   hashes, exact local source closure, fixed seed, official sample success,
   CUDA/VRAM policy, and frozen arm order *before* a model factory. The current
   default admission deliberately raises.
3. A supervisor that launches the official sample and then each of four arms
   in a fresh process with identical model/RNG state, one process at a time.
   It must protect the official helper's module-global wrapper from
   concurrency, enforce time, VRAM and total output budgets, and bind the
   complete child artifact tree. The current worker is a callable, not a
   production CLI or process launcher.
4. A pre-use check of exact system/tool/template/helper/checkpoint bytes at
   each arm, and post-use checks for mutation. This worker compares the
   selected QA hashes to the source plan and pins the stock response hash;
   other source identity is still an admission/supervisor responsibility.
5. Verify that the recorder's per-step TTS codes correspond to the official
   final trimmed `gen_codes`, and verify stock versus zero-edit adapter
   identity using *reloaded* saved captures. No such comparison is claimed
   here.
6. Confirm that the high-level adapter's files and raw capture together fit
   the total output budget. `save_capture` caps its own directory, while the
   prospective supervisor must cap the whole arm. Also verify bounded reads
   for all prompt and QA artifacts before enabling production entry.

No calendar operation is performed by this worker. The synthetic tests use
only `executed: false` response data. No result here is a security finding.
