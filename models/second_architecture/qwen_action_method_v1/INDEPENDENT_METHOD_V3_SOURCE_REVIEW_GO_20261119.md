# Independent source review: Qwen action method V3 gate

19 November 2026. **GO for the V3 source-only development-gate correction; candidate inference remains on HOLD.** I reviewed commit `bb5ae31` against the [V2 HOLD](INDEPENDENT_METHOD_V2_REVIEW_HOLD_20261119.md). This review loaded no Qwen weights, ran no candidate inference, and opened no locked output.

The V3 commit adds only `gate_v3.py`, its seven no-weight tests, and a protocol amendment. The V1 wording and schedule, V2 tokenizer/readout freeze, scorer and numerical method are unchanged. The gate verifies SHA-256 `a6ddd9b007649173d23e38996883920467ef861f15b4491acde3cdd096e4ef18` for the 30-script V1 spec and `eb4b4f5d12827d2d2055d6e97d44b01e3a32d66b20514bbfdc41a16b07178d91` for the V2 readout freeze. It requires all 18 FIT score records and six typed plus six audio development records in the frozen manifest order.

## Independent checks

- All **23/23** V1, V2 and V3 no-weight endpoint/readout/gate tests pass. The exact V2 false-promotion fixture now yields `typed_vjp=0`, `audio_vjp=0`, `pass=False`, while a consistent synthetic score and token fixture passes. The earlier V2 gate accepted the same wrong-score fixture as `6/6`, `6/6`, `pass=True`.
- I independently replayed an empty continuation and a repeated-EOS continuation. V3 derives `censored` and `malformed`, respectively, and the six-row gate fails. A shuffled development row is rejected for manifest-order mismatch. A caller assertion of `predicted_label=LIST` when that row's VJP score argmax is `ASK` is rejected.
- V3 runs the V2 endpoint on each raw token path. Complete label and canonical-path flags come from those results; optional caller assertions cannot override them. It derives VJP and constant top classes from score matrices, recomputes the constant matrix from all 18 FIT score rows, and then applies the frozen V2 numerical thresholds. No outside class list, subset, or supplied canonical flag can recreate the V2 false pass through this interface.
- V3 validates finite three-class score rows and exact six-script IDs/order for both modalities. The V1/V2 120-arm directional schedule and 60 independent random-control seeds were already checked in the [V2 source review](INDEPENDENT_METHOD_V2_REVIEW_HOLD_20261119.md) and are unchanged by this commit.

## Boundary and remaining gates

The gate accepts `actual_scores` and `vjp_scores` as records; it cannot itself prove that those numbers came from the pinned model, correct class order, exact teacher-forced `label + EOS` logits, or the intended hidden-state site. It likewise cannot verify that raw continuation IDs came from the same candidate forward. Those bindings belong in a separately reviewed executable runner and saved-result audit. V3's source-only GO does **not** authorize development or locked candidate inference.

Before a model run, the 18 speech WAVs still need a byte-level source audit and blinded human wording review, and a runner must prove tokenizer/processor identity, teacher-forced alignment, hook call counts, complete-token stock/instrumented parity, full-donor controls, and measured memory/time gates. The old Qwen rank-16 Stage A failure remains failed; V3 does not claim a full Jacobian Lens, a transferable audio readout, an action error, or a security finding.
