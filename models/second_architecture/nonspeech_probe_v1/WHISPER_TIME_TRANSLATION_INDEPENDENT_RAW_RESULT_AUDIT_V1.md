# Independent raw-result audit: Whisper time translation

19 November 2026. **The saved evidence supports 1/4 strict union crossover cells and 0/2 isolated source-tail cells.** I independently reran the three fixed saved audits and `audit_joint()` from raw attempts with the admitted `models/second_architecture/.venv` interpreter. I loaded no model weights and did not edit any attempt. The generated joint object exactly matches `WHISPER_TIME_TRANSLATION_EMPIRICAL_RECEIPT_V1.json` at `37ff826`. Its recorded admission, GO, parent-result, progress, and records SHA-256 digests and fixed paths all match the current files for every stage.

| Stage | Saved audit | Calls / cap | Skipped arms | Whole time | Peak tree RSS | Evidence before result |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | PASS | 28 / 28 | 0 | 12.87 s | 844 MB | 190 MB |
| 8,000-sample offset | PASS | 62 / 62 | 0 | 18.80 s | 868 MB | 400 MB |
| 16,000-sample offset | PASS | 62 / 62 | 0 | 19.19 s | 874 MB | 400 MB |

All three parent receipts say `completed_pending_saved_audit`, worker exit code 0, and no incomplete progress entry. The saved auditor independently validated the parent power, host, RSS, and progress records, full WAV-to-mel/mask binding, sidecar hashes, complete token paths, detection/prefix stability, and edited hook-state arithmetic. Both translated attempts have complete, distinct zero/noise baselines in prepend and append. Their measured interior mel shifts have maximum absolute difference 0.0 in the prespecified windows; masks agree within each offset, with 1,050 and 1,100 active frames respectively. These statements mean the **saved evidence passed consistency checks**, not that an untrusted local owner could not fabricate it.

## Direct check of the decision rule

I separately indexed the raw `records.jsonl` token paths by layout, site, window, location, and arm. This comparison did not use the auditor's `summarize()` or joint counts. It reproduced the control pattern below; the saved auditor additionally checks full-donor first-step vectors, patch norms, EOS, and mel transport.

| Cell | Original window viable | Prepend: correct / opposite fails / relocated null | Append: same three checks | Strict result |
| --- | --- | --- | --- | --- |
| 8,000, post-convolution union | yes | yes / yes / yes | yes / yes / yes | **qualifies** |
| 8,000, after block 0 union | yes | **no** / yes / yes | yes / yes / yes | fails |
| 16,000, post-convolution union | yes | yes / yes / **no** | yes / yes / yes | fails |
| 16,000, after block 0 union | yes | yes / yes / **no** | yes / yes / yes | fails |
| Either offset, post-convolution source-tail | **no** | direct transfer fails | direct transfer fails | fails both |

The successful 8,000-sample post-convolution union cell moves with the waveform: the prepend donor transfer succeeds at the shifted rows, and append succeeds at the original rows. In both layouts the opposite direct transfer does not copy the donor's complete answer, and the equal-norm relocation at that opposite site preserves the recipient's complete answer. For the 8,000-sample after-block-0 union cell, the prepend shifted direct transfer fails to reach the donor's full token path. For both 16,000-sample union cells, the prepend competing-window relocation **changes** the recipient's full token path. It therefore fails the required null even though it does not copy the donor. Calling those cells successful would overstate localization. The original source-tail window failed its own viability gate at both sites, so neither translated source-tail result can establish isolated tracking.

One control has limited force: the unrelated donor-row (`source_random`) edit at the post-convolution union has zero measured state delta in these attempts. The nonzero, norm-matched displaced control remains null in the one qualified cell, and the separate relocation control is essential to the interpretation. The result concerns the **union** of two five-row windows around nominal rows 400 and 500; it does not identify which boundary drives the effect. After block 0, attention can mix information across rows. Leading silence may change context as well as timing. This is one engineered non-speech pair, with no human-perceptual or attacker-reachability test and no demonstrated security violation. The 1/4 count is a narrow timing-linked causal observation, not a general property of audio models or a discovered circuit.
