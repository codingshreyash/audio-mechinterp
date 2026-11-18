# Stage-B external launcher independent source review — HOLD

18 November 2026. I independently reviewed launcher source commit `f96a1f8` and the CPU-test follow-up `8bc2f55`. This is a source review; no Stage-B model process or external Stage-B freeze was created.

**Decision: HOLD for Stage-B launcher source. No external freeze or model run.**

The 12 CPU tests pass. The real `--check-freeze` stops before model load because this independent launcher source review was absent. The source binds the preserved Stage-A BF16 failed launch and posthoc audit, the separate FP32 continuation, reviewed driver and auditor hashes, and two phase-specific future freezes. A synthetic Git fixture accepts a valid reviewed freeze and rejects an edited and recommitted freeze; direct and parent symlinks are rejected. Those checks do not resolve the following resource gate defect.

`_bounded` polls the child and immediately exits its loop if the child has finished, before checking the final stdout/stderr sizes, parent output bytes, worker output bytes, or elapsed time. In an independent synthetic subprocess, a child wrote **9,437,184 bytes** to stdout (over the **8,388,608-byte** per-stream cap), exited with code 0 in about 0.11 s, and `_bounded` returned `reason=None`. The same ordering can miss a child that completes just after its watchdog deadline. The launcher must check the final sizes and elapsed time after reaping every child, and a CPU regression must cover a fast successful child whose output exceeds the cap. This is a blocker because the prospective launch protocol promises hard output and time ceilings.

After a corrected source commit, independently rerun the fast-child regression and the full CPU suite, check `--check-freeze` still fails before model load, then append a separate GO/HOLD rereview. The current HOLD history must remain visible. Saved Stage-A tensors and the later Stage-B audit can establish arithmetic and provenance, but they cannot independently attest that the model calls occurred.

## Corrected source rereview — GO for source only

The producer corrected the fast-exit cap bypass in `4843880` and the hard phase deadline in `0c3ab02`. I inspected both diffs and independently reran all **15 CPU tests**. A separate fast child exited 0 after writing 9,437,184 stdout bytes; `_bounded` now returned `capture_byte_cap` against the 8,388,608-byte limit. A separate slow child was killed with signal 9 at the test deadline and returned `child_process_timeout`. A direct worker reinvocation with a fake model body called that body once and then failed on its exclusive `worker_claim.json`. The revised launcher also requires that claim to retain its first-add Git bytes when later admitting FP32.

The final reviewed source hashes are launcher `46e669a0eec6df56573a572ee22bc9e7e19d50683e7535e51031ce2779245d42`, worker `f4128106dccb07510b7be75437941ff3dc1ff2d0fd36c9181155842821ae1a7c`, audit adapter `69b8e064fbcda35cdd9b93fdd16a9fdfd5cd2d523e17fb0104847d171da6d703`, CPU tests `cb5134fbe040170f9eb4388d7bc853ec3405ab229485be5fb0fcbbd0e1c0fe18`, and launch protocol `03195b3fd45ee2a46c23b7b4c23fc20c49f441c993ade4882216b5c93c3ae0d4`. The real `--check-freeze` still rejects before model construction because the required external freeze is absent. The exact BF16 failed-launch/posthoc and FP32 continuation source chain was checked by the CPU admission test; the first-add freeze regression accepts a valid reviewed freeze and rejects its later recommit.

**Decision: GO for Stage-B launcher source; model run HOLD until external freeze.**

This GO does not create or approve either phase freeze. The BF16 external freeze requires a separate independent review before its one permitted model process. FP32 requires its own later freeze after a reviewed BF16 Stage-B result. These source checks establish bounded execution logic and saved-evidence arithmetic admission; future result reviews must verify the actual receipts and tensor sidecars.

## Exact-review-hash amendment rereview — GO for source only

Commit `60e70ef` changed only the launcher gate, its CPU test, and the launch protocol. I inspected the diff: the BF16 freeze review must name the exact BF16 freeze SHA-256, the FP32 freeze review must name the exact FP32 freeze SHA-256, and the BF16 Stage-B result review must name the exact BF16 manifest SHA-256. Existing first-add and Git chronology checks still apply. The synthetic reviewed-freeze fixture now includes the bound freeze hash. All **15 CPU tests** pass, and the real `--check-freeze` stops at the absent BF16 external freeze before a model process. The new gate SHA-256 is `8bf750e1f79095c714a57cb78a4c0793f6b98122fb64d5453398c820e2058858`; tests `a8968ad034457674135725c15618d9e3355395497088d514eecc75c1a5043bdc`; protocol `b0d8f46905f83be14cdbaadef47ee8a351e49c8b67f594b4a4d305135a0f2701`. Worker and adapter hashes remain `f4128106dccb07510b7be75437941ff3dc1ff2d0fd36c9181155842821ae1a7c` and `69b8e064fbcda35cdd9b93fdd16a9fdfd5cd2d523e17fb0104847d171da6d703`.

**Decision: GO for Stage-B launcher source; model run HOLD until external freeze.**

This source GO supersedes the preceding source GO only for the changed bytes. It is not a freeze review or an authorization of a Stage-B result. The external BF16 freeze must be committed after this review and checked by an independent reviewer before the one-shot BF16 replay.
