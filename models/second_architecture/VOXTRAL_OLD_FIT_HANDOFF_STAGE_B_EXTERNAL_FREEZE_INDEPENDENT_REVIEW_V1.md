# Independent review: Stage B BF16 external freeze

18 November 2026. I reviewed the committed BF16-only external freeze at commit `f5263304441dbc85e499c5942cf169f5b91e9498`. Its SHA-256 is `18a8602ab5eac1a5099f98349ef6a0ef6456c721faca6cabeccbc351f9e3bb55`. This review does not run the model.

Decision: GO for Stage-B BF16 external freeze provenance; BF16 model may run under the pinned gate.

I independently checked all 21 bound files against their SHA-256 values and current Git blobs. The six launcher source, test, worker, adapter, protocol and review blobs match the source-review commit `dd9132d57f3d304733a4877e4f7ed562fff41269`. The driver, auditor and Stage A FP32 reviews precede that commit. The freeze's bytes match its sole first-add Git blob, and its commit follows the launcher review. The receipt permits only `bf16_direct_lm`, preserves the stated audit scope, and binds 4,500 seconds and 5 GiB aggregate caps.

I separately rehashed all 12 BF16 Stage A attempt files (718,170,997 bytes) and all 12 FP32 Stage A attempt files (1,414,419,896 bytes). In each arm, ten files match committed Git blobs and two content-addressed tensor sidecars remain local and hash-bound. The original BF16 `launch_finish.json` still says `launch_failed`; the FP32 continuation has an audited successful finish. Their measured model-process times sum to `218.5633733337745` seconds, matching the freeze. No Stage B phase reservation or attempt existed at review.

Before this review file existed, the real CPU `--check-freeze --phase bf16_direct_lm` command reached the missing independent freeze review and exited without model work. That is the expected fail-closed state. The gate must recheck this review and all bound bytes at launch. This GO establishes receipt provenance and admission for one bounded BF16 Stage B attempt; it does not independently attest past model forwards/backwards, establish a mechanism across inputs, or establish a security finding.
