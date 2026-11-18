# Independent source review: Whisper FIT word-window causal runner V1

18 November 2026. Reviewed committed runner source through `fc1a88d`, the fake tests in `e11c59a`, and the frozen V1/V2/V3 prospective plans. **Decision: HOLD before model inference.** No checkpoint was loaded and no model output was inspected in this review.

## Verified source-only checks

- The pinned manifest and four already-open FIT WAVs are checked by file and PCM SHA-256. The runner independently verifies same-length, identical outside-slot PCM and differing inside-slot PCM. Word and prefix windows match the frozen plan.
- The intervention sites match the V2 correction: pre-hook at `encoder.layers[0]` input, post-hook after block 1, and post-hook after `encoder.layer_norm`. The pre-hook retains the remaining positional arguments. Saved baseline captures include all encoder blocks and final norm. Hook counts, shape, dtype and finite values are checked.
- The schedule contains four baselines, 36 site/window/self arms, four final-state full-donor arms and eight random arms: 52 complete decode attempts if no gate fails. The final-state all-row donor and self-patch exact-token checks are present. Random edits are restricted to the frozen word window and checked against the donor-minus-recipient whole-window Frobenius norm with the two frozen seeds.
- Source through `fc1a88d` records the runner's path-specific Git commit and SHA-256, the full cached snapshot file inventory, checkpoint hash, model/config identity, versions and selected installed implementation hashes in a preflight receipt. The run compares the complete receipt before weight loading and refuses an existing output directory.
- The V3 metric clarification is **GO** as a prospective description: `generation.scores` are labeled generation-processed step scores, candidate single-token alias score difference is optional, and no such score is used to determine complete-transcript success. The primary outcome requires an uncensored complete transcript. Eight fake/provenance tests passed under the pinned local Python environment; `py_compile` passed.

## Blocking finding

The V1 plan requires saving every decoded baseline and intervention output, including a failed or censored control. The runner checks baseline target-word/EOS eligibility **before** writing the baseline record. It checks self-patch and full-donor exact-token parity **before** writing the corresponding intervention record. If any of those gates fail after a successful decode, the append-only log contains only a generic failure record and loses the actual transcript/token IDs that explain the failure. This would make a failed run less auditable and could hide a consequential model behavior. Save the raw decoded result first, label its eligibility/control status, then raise and append the failure marker. Preserve the original arm count and hashes. A source rereview is required before the first live pass.

The source's time and RSS checks occur between decodes, including after the final arm; they do not preempt a single stalled decode. The 128-token generation cap bounds that arm's output length but not its wall time. This operational limitation is not the HOLD reason.
