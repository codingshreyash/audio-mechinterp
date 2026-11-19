# Amendment V3: capture before intervention

19 November 2026. **Prospective source-only order correction. No candidate-model inference.** This amendment resolves the dependency identified in the [independent V2 rereview](WHISPER_PADDING_CHANNEL_INDEPENDENT_DESIGN_REREVIEW_V2.md). It supersedes V2's schedule hash and per-case execution order. All other V1/V2 experimental definitions, output comparisons, source pins, 436-call denominator, and numeric resource ceilings remain in force.

The two silence no-hook/read-only capture pairs still run first. For each non-silence recipient at each site, the frozen 12-arm order is now:

`RR`, `read_only`, `RD`, `DR`, `DD`, `self`, `random_active_1`, `random_active_2`, `random_padding_1`, `random_padding_2`, `padding_detect_only`, `padding_transcribe_only`.

`RR` installs no hook and supplies the recipient's baseline output. The immediately following `read_only` call captures the recipient's native `log_mel` and `pre_block0` tensors in both detection and transcription phases, without changing its output. Only **after** that capture may a donor/recipient splice, self replacement, random edit, or phase-specific edit run. The `read_only` call in the later site block recaptures both sites and must match the earlier same-case native captures by exact shape, dtype and content hash. No implicit preliminary model forward is allowed.

The amended [planner](padding_channel_plan_v1.py) validates this donor-baseline → donor-capture → recipient-`RR` → recipient-capture → edit dependency during schedule construction and fails closed if an edit is moved before its capture. The [no-weight tests](test_padding_channel_plan_v1.py) assert the complete per-case arm order and deliberately swap `read_only` with `RD` to verify rejection. The canonical sorted-key schedule SHA-256 is now `d939618689e989a2dfe249cc6d4f0d3b32a532c184991680b77f11b4af6a0ace`. The total remains **436 calls**; the conservative core evidence estimate remains **1,375,203,488 bytes**. Neither a candidate checkpoint nor its outputs were used to make this correction.

This amendment is submitted for **design-only** rereview. The future worker, fake-model hook tests, admission receipt, independent executable GO and host gate specified in V2 are still required before any candidate inference.
