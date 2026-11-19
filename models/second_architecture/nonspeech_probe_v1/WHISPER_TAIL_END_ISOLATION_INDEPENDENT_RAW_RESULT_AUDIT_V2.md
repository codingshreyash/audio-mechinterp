# Independent raw-result audit: isolated Whisper tail/end V2

19 November 2026. **The three fixed saved attempts pass evidence audit; the prespecified strict result is 0/2 offsets.** I reran each V2 saved audit and `audit_joint()` from the fixed raw attempts under the admitted `models/second_architecture/.venv` interpreter. No model weights were loaded, and I changed no raw evidence or frozen source. The fresh joint object exactly equals [the committed empirical receipt](WHISPER_TAIL_END_ISOLATION_EMPIRICAL_RECEIPT_V2.json) at `ee2cf76`. For every stage, I independently matched the receipt's admission, GO, parent-result, progress, and records SHA-256 hashes and fixed path to the current files.

| Stage | Saved audit | Calls / cap | Skips | Whole time | Peak tree RSS | Evidence before result |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | PASS | 7 / 7 | 0 | 11.18 s | 838 MB | 63 MB |
| 8,000-sample offset | PASS | 24 / 24 | 0 | 13.68 s | 864 MB | 170 MB |
| 16,000-sample offset | PASS | 24 / 24 | 0 | 13.51 s | 867 MB | 170 MB |

Each parent receipt says `completed_pending_saved_audit`, exit code 0, and no incomplete progress entry. The attempts share Mac boot identity `1792085007295485` and are ordered original → +8,000 → +16,000 in UTC and monotonic time; UTC gaps after the prior attempt ended are 31.20 and 44.15 seconds. The saved auditor checked resource and power/host records, frozen WAV-to-full-mel/mask reconstruction, sidecar hashes, complete EOS paths, first-step vectors, detection/prefix, and exact edited/untouched hook rows. Both translated attempts retain complete, distinct zero/noise baselines in both layouts, with equal masks. Predeclared interior mel transport differs by **0.0** in the checked windows; active masks contain 1,050 and 1,100 frames.

## Raw token-path cross-check

I separately indexed `records.jsonl` by layout, window location, and arm, then compared complete generated-token paths directly to each layout's zero donor and noise recipient. This did not use `summarize()` or the receipt's derived counts. The original isolated `498..502` direct patch changes the noise recipient `[286,50257]` to zero donor `[509,50257]`; identity, unrelated-row, and nonzero displaced equal-norm controls preserve the recipient; full donor matches donor. Original isolated viability is true.

| Offset | Aligned direct in prepend / append | Append controls | Prepend decisive control | Strict offset |
| --- | --- | --- | --- | --- |
| +8,000 | Both reproduce donor through EOS | Qualified | **Aligned displaced equal-norm edit changes recipient** | **fails** |
| +16,000 | Both reproduce donor through EOS | Qualified | **Relocated aligned delta changes recipient** | **fails** |

At +8,000, prepend's unpatched noise path is `[286,478,406,257,283,13684,260,13,50257]`. Its aligned direct patch reaches donor `[509,50257]`, but the aligned displaced equal-norm control produces `[286,478,406,799,312,257,283,13684,260,13,50257]`. This is a complete different answer, so the prespecified control fails even though relocation to the competing fixed window preserves the recipient.

At +16,000, prepend's unpatched noise path is `[286,478,406,799,312,257,1578,2146,13,50257]`. The aligned direct patch again reaches donor `[509,50257]`. The competing fixed-window direct patch gives a **third complete answer**, `[286,50257]`, instead of donor or recipient. That meets only the primary failed-donor comparison, not the stronger recipient-preservation comparison. Relocating the nonzero aligned delta to that fixed window gives another different complete answer, `[286,478,406,257,283,13684,260,13,50257]`; the required relocation null fails. Both append layouts qualify, but both layouts must qualify for an offset.

I recomputed native `post_conv_pos` state norms from saved baseline arrays. The aligned old-tail/end delta is about **8.1954** in both layouts at both offsets. Append's nominal new-end native donor-minus-recipient delta is exactly **0.0** at both offsets. Its native direct patch there cannot show the model ignores the new end; the meaningful comparison is the **nonzero** relocated aligned delta, which preserves the append recipient. The unrelated-row donor edit also has zero measured delta at this site, so it is weak by itself. The nonzero displaced and relocation controls determine the failures above.

One post-run source test is stale: `test_direct_worker_without_fixed_attempt_stops_before_weights` expects a missing-attempt error. The fixed attempt now exists, so direct invocation instead reaches the live-pipe guard and raises `inherited parent pipe required before model import`; the test fails its expected-error regex. The observed direct launch still refuses **before weight loading**. This test artifact does not change the 0/2 raw-audit result; frozen post-GO source was left untouched.

**Interpretation:** 0/2 is a failed confirmatory within-pair isolation test under the frozen rule, despite donor transfer at the aligned five-row window in all four translated layouts. It must not be reported as isolated tail/end tracking or a security finding. The observations use one V1-informed engineered non-speech pair in Whisper Tiny. Internal state edits are not attacker-supplied waveform effects; neither a unique circuit nor behavior on human speech follows. A saved-audit `PASS` establishes internal consistency under the trusted local-owner assumption, not cryptographic provenance against that owner.
