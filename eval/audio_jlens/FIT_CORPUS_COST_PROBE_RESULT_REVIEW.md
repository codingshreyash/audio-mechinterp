# Independent saved-result review: fit-only FP32 corpus cost probe

Reviewed 2026-11-18 from `fit_corpus_cost_probe_v1_results.json` (SHA-256 `d4edc0f8c6d558c82bd6b7f9ef0e03fb92cdda269de44e943c1bec9f9373e42d`) and its memory JSONL. I did not rerun MPS.

## Finding

The bounded probe completed its stated **cost and gradient-connectivity** check. A shared forward exposed both Qwen audio sources and the prompt-last terminal residual; 16 backward VJPs produced finite, nonzero six-bin gradient sums at both sites. The measured **elapsed span** is **2.419 seconds for this one fit prompt** and the saved 48× projection is **116.108 seconds at identical per-example cost**. The elapsed span includes Python bin aggregation and logging, so its `one_example_model_work_seconds` field is broader than pure model kernel time. The projection is approximate wall time, not a measured corpus runtime or a fitted-lens result.

## Independent checks

- The recorded script, proposal, fit manifest, Piper red WAV, fit helper, Qwen adapter and time-map SHA-256 values match their files. The run is labeled `piper/red`, fit only; the model/Transformers revisions match the pinned Qwen Thinker. The report names target `text.residual.35`, the pre-final-norm prompt-last block output.
- Hook counts are exactly one at encoder31, connector, and target. Source shapes are `[156,1280]` and `[78,2048]`; target shape is `[1,102,2048]`. Coordinates are exactly `0..15`, with one per-coordinate result and one timing entry per VJP. These are fixed basis-coordinate **cost proxies**, not the proposed fit-only SVD vectors.
- The six intervals cover preword, four word quarters and postword from 0.996609 to 1.797703 seconds. Encoder31 has `[7,7,6,7,7,6]` rows per bin; connector has `[4,3,3,4,3,3]`. Indices are disjoint, nonempty, ordered, and within their source tensor bounds. For every one of 16 coordinates, each site's six summed-gradient norms is finite and strictly positive (192 values total).
- The synchronized forward took **0.2324 s**. Sixteen timed backwards sum to **1.9986 s** (median **0.1235 s** each); capture/report overhead within the one-example measurement is **0.1879 s**. The saved `48×` projection is arithmetically exact. Initial model load is separately recorded and excluded from that projection.
- The memory sidecar has 18 expected phases: post-load, post-forward, and each of 16 VJPs. Maximum MPS driver allocation was **18.842 GiB** and current allocated memory **18.514 GiB**, below the 30 GiB limit, with no `over_limit` record.

## Scope of the estimate

The measured clip is an original short Piper fit example (102 prompt tokens, 156 encoder rows). The new 48-example map-fitting corpus may have longer audio and prompts; compare their frozen shape distribution before using this projection as a six-hour decision. It also excludes fit-only basis collection, feature preprocessing, learned baselines, and evaluation. This result shows that two source-site VJPs can share a graph on this example; it does not establish transport accuracy, temporal localization, or a security finding.
