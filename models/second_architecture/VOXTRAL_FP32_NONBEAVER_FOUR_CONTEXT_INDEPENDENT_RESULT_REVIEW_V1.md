# Independent review: four-context Voxtral FP32 non-beaver local-linearity pilot

19 November 2026. **GO for the frozen numerical-method gate only.** This is an old-FIT `color_05` red/blue pair across four fixed synthetic-voice/template contexts. It is not a reusable lens, a held-out result, a tool-action result, a security finding, or approval to open a fresh corpus.

## Evidence and checks

I reviewed the original reserved attempts `20261118T182308460007Z_00` through `_03` in index order. I independently recalculated the SHA-256 hashes below. The [Piper raw evidence manifest](VOXTRAL_FP32_NONBEAVER_PIPER_EVIDENCE_MANIFEST_V1.json) is SHA-256 `4f52336d46faef0909206c9df2f4f9d630242183b54912c0db51b5103ceec83a`, with sorted 27-file raw manifest digest `bdd3f1d3161c60201b361a1014a4fddbaf4e28e939d936bd170d2f486173c645`. The [V1 saved-data auditor](voxtral_fp32_nonbeaver_linearity_audit_v1.py), SHA-256 `70a1ac3c4e4da92bb1fa1034bd866195024dcd423ea35a2386e233bce5dca8e2`, was rerun read-only against APFS clones of all four raw attempt directories. Its output was byte-identical to each original `audit.json`. It rehashed each content-addressed tensor sidecar and recalculated the saved numeric data; no model inference was run. The frozen continuation gate's `check_prior(3)` and direct `_check_new_completed(3, ...)` both passed, validating the final slot itself rather than relying on a successor gate.

| Slot / context | Result SHA-256 | Audit SHA-256 | Tensor sidecar SHA-256 |
| --- | --- | --- | --- |
| `_00` Piper silver | `761e7540baf870876125db36cfa62740b62366076898adc4a615750a84fb7edf` | `2000cb46e0ad025ba6fc07df7c5a741debd06f09e07858eaad6b08363a7d2edb` | `363f2dbe24ba98582951793f4df2b26614c6ebfdfff76a941c94b633c4b0309f` |
| `_01` Piper teal | `e0a852540943f2c7bbbb50ff369450283cd550d9f3e8670f2441be5b6848eb40` | `692b5d2bc458e0b10a9223637c8b9d08ab5cb64c4b8f81835f99e16f27992372` | `89dafb044a4249a9858ed4251b6693244c3a7133dfe2ddfcba4fa8edc64595ad` |
| `_02` Kokoro silver | `f24e19592eec74cebafa4dbecbfb186b0d67c67e0d695e139b11f0984bffd77e` | `89462fc34f710c0dff01ae164a6765b3e9058e63199e892e44c4144aef0d3303` | `cc294722d36233e38532fe18c9e0123a74a5a7fa555411beb2c9f20e9780a3cb` |
| `_03` Kokoro teal | `2c3985e19013aacf80fb9af2dda4963a075ad5e5f6421bc7ab4d9c69afdfeb81` | `36585ae9b2fe9a61892a1d349182c0ceb5451a2e36e64aef49031604550ca293` | `00d512d6dcb5de7b2c3c331791e64f08008a0a967b145da99c0cc775c3fd5bad` |

For the two continuation slots, the independently hashed launch-start / launch-finish / worker-claim SHA-256 triples were:

- `_02`: `033f03c37dd2a0d8ccfa4c6561a6297a0e78016f096eaa1f43d3a0c427db9cac` / `48161da536e2da965697270517b42541fddf6f1f7f41839176f3802109f45b68` / `9dc4fdc453ecceac48ddae00ded6623fb13fcdb75ab2c00d30cc0574f1083e8c`.
- `_03`: `bd70d53eee56989d4e4bdbffe136ec2cc7c1afc8eda4bb3afb737284b60ff88f` / `c3c18fcdac8c293f9858082d5d45aa2dd8f33f94369e0681ff916f203b38e6f4` / `f2065e01c65091294c4a769ab210f4437e0401267fe92c11b4986a06327c34aa`.

The `_02` pre/post compressed host-power logs hash to `4fdccf8a4c260bc4a7b4c980612dd8119113af29e084e2036f112f8864d9a45d` / `0bccd3513de68ec9110cd7f78a4a89d30befc86afa4f19733500c4433c3da8da`; the `_03` logs hash to `732c54d709d492cb85c9e167f42c9bf84f62afd66f6a510599a1c855426d7499` / `9912d3fab9decf149afbcb44901cbfa58e9d9acc96e7ca41cf418220f19bbdf9`. The continuation gate verified exact pre/post raw log hashes and coverage, external power, open clamshell, no Sleep/DarkWake/thermal event in either new launch interval, and the ten-minute prelaunch stability checks. Signed UTC-minus-monotonic launch/worker gaps were `_02` +0.00438/+0.00307 seconds and `_03` +0.00677/+0.00301 seconds, each below five seconds. `_00` also met the original five-second rule. `_01` has the previously documented **+25.124338/+13.807611-second** gaps and passes only the [prospective continuation protocol's](VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_CONTINUATION_PROSPECTIVE_V1.md) exact-byte, exact-gap narrow exception; the overlapping power-log evidence does not establish the gap's cause.

All four result and audit statuses are complete, both worker and auditor return codes are zero, and all four saved invalidity comparisons agree. Each context has exactly 96 patched forwards, eight VJPs, two G1 clean graphs, two N0 clean references, one donor and one passive recipient capture, and two natural generations: **384 patched forwards and 32 VJPs in total**. Model-work times are 138.854, 151.869, 151.489, and 148.627 seconds (590.839 seconds total, or 959.322 seconds including the fixed 368.482-second prior baseline). Each is below the 300-second worker cap; the cumulative total is below 4,500 seconds. Peak MPS driver memory is at most 26.817 GiB, below 30 GiB; combined output is about 4.885 GiB, below the 8 GiB campaign cap.

## Frozen decision

The [frozen auditor's `site_gate`](voxtral_fp32_nonbeaver_linearity_audit_v1.py) requires all 24 cells valid per site, at least 18 resolved, median resolved maximum two-cycle relative error at most 0.25, and at most four cells over 0.50. Its four-context aggregation gives:

| Site | Valid / planned | Resolved / planned | Median maximum-cycle relative error | Cells over 0.50 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| encoder31 | 24/24 | 24/24 | 0.0244318 | 0/24 | pass |
| projector | 24/24 | 24/24 | 0.0412463 | 0/24 | pass |

The 24 primary max-cycle errors span **0.00305–0.31046** at encoder31 and **0.00753–0.25229** at the projector; the respective middle 50% ranges are 0.01630–0.08217 and 0.01508–0.08035. Thus the gate passes on its predeclared median and high-error-count rules, not on every cell being below 0.25. For the predeclared G1-versus-N0 comparison, the matched paired-output `||D_G−D_N||₂` values across 48 cycles per site have median/max **2.19e−5 / 3.83e−5** at encoder31 and **2.12e−5 / 4.81e−5** at the projector. Relative to each cycle's measured G1 norm, the corresponding median/max ratios are **0.00332 / 0.04354** and **0.00151 / 0.01517**. These are descriptive mode-interaction measurements, not a separately thresholded transfer claim. The first-answer top token changed in 4/48 encoder31 cycles and 2/48 projector cycles in each mode. That first-logit observation is not a complete generated-answer or tool-action endpoint; the two natural generations per context are baseline controls and do not turn these local edits into demonstrated complete-output effects.

The gate's `advance_to_fresh_corpus_feasibility_gate` flag is true. This establishes that the predeclared FP32 local-linearity measurement is numerically stable for this **single already-inspected FIT lexical pair** at these two sites and four synthetic contexts. It does not establish generalization, a reusable interpretation method, or any downstream security effect. A separate prospective fresh-corpus feasibility protocol and its own controls remain necessary.
