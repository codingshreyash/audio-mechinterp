# Independent review: Stage-B FP32 continuation freeze

18 November 2026. I reviewed the separately committed FP32-only freeze at commit `761549069963218567e68f3fa8809ea2727120ac`, SHA-256 `bc2025eb6beab51965a468544bec53b598919b4a4aea7d4f00d8dbe7ca3c462c`. This was a read-only evidence review. No model was loaded or run.

Decision: GO for Stage-B FP32 continuation freeze provenance; FP32 model may run under the pinned gate.

## Checks

- Before this review file existed, the real CPU `--check-freeze --phase fp32_replay` reached the missing FP32 freeze-review file and stopped before model work. The base BF16 freeze and its independent review passed the gate; its SHA-256 remains `18a8602ab5eac1a5099f98349ef6a0ef6456c721faca6cabeccbc351f9e3bb55`.
- The FP32 freeze has exactly the phase, audit-scope, cap, source, BF16-result and time fields required by the reviewed launcher. Its current bytes equal both the HEAD Git blob and its first-add blob. The BF16 manifest first-add commit `efaa17e2df5293c7f12b9a1a3e0b76dc5c62684d` precedes its separate result-review commit `2bbfaee211e1809d00a31ede90330daa6bb04b00`, which precedes this freeze's first-add commit. The BF16 result review contains its required GO marker and the exact manifest SHA-256 `c5bab618791e2f2c9c7255b60032cf6f8624aa844ecea7ad6d5eb3c0658dc4b5`. The freeze's recorded UTC time is after the BF16 result-review commit time.
- I independently rehashed all 24 Stage-A raw files: each arm has ten files equal to its committed Git blob and two required local tensor sidecars. BF16 Stage A remains `launch_failed`; FP32 Stage A retains its separately audited continuation. The Stage-B BF16 replay has exactly one reserved attempt, the committed reservation/start/worker-claim/finish/result/audit receipts, and a 65,472,751-byte local tensor with SHA-256 `e148b94455e9a5533582c68d1b26c1b7a013f080b96e139f319f230b5531a3d2`. Its parent model-work time is `28.278460667002946` seconds, no less than the worker time.
- The freeze binds `218.5633733337745` seconds of total Stage-A model work plus the BF16 Stage-B time, `246.84183400077745` seconds before FP32. A full 1,800-second FP32 reservation remains below the 4,500-second cap. I found no FP32 Stage-B reservation or attempt. The freeze permits only `fp32_replay`.

The four Stage-A tensors and one BF16 Stage-B tensor remain local-only and are required for independent re-audit from a checkout. The saved-tensor and receipt checks do not independently attest that the recorded model forwards or backward calls occurred. This approval covers the freeze and the gate's one-shot FP32 admission, not a numerical mechanism, held-out transfer, or security finding.
