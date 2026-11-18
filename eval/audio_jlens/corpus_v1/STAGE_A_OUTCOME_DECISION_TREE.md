# Stage A outcome decision tree

**Prepared without development model outputs.** Apply the frozen [`PRE_DEV_EVAL_PROTOCOL_V1.md`](PRE_DEV_EVAL_PROTOCOL_V1.md) and independent auditor to the complete 96-direction result. Keep all invalid rows in planned denominators. Preserve the raw result, summary, hashes, and the exact failed conditions. A failed v1 gate remains a negative v1 result; do not relabel it after diagnostics. Keep the locked test sealed unless both v1 Stage A sites pass and the later Stage B conditions are frozen and completed.

```text
Completed 96-direction Stage A, independently audited?
  no  → record incomplete/invalid; fix only reproducibility or code defects under a new version.
  yes → clean semantic one-word+EOS ≥72/96 AND V2 prefix classified ≥72/96
         with ≥75% complete-answer purity among classified?
           no  → stop v1; inspect task/audio/score coverage on FIT only.
           yes → each site has ≥48/288 above-floor small-step effects and
                   ≥75% matching signs among those effects?
                     no  → stop v1; inspect local derivative and direction on FIT only.
                     yes → map MAE ≤0.75× EACH of zero, global ridge, six-bin ridge
                             over valid 288/288 small-step conditions per site?
                               no  → stop v1; inspect map fit/calibration on FIT only.
                               yes → absolute category-anchor score improves ≥68/96
                                       planned directions per site?
                                         no  → stop v1; inspect absolute anchor on FIT only.
                                         yes → Stage A pass only if BOTH sites pass all gates;
                                                 freeze and run Stage B, then decide separately.
```

| Failed condition | FIT-only diagnosis that does not use development item outcomes | Separately versioned improvement to test on fresh data |
| --- | --- | --- |
| Missing/invalid case, hash, source shape, row bin, hook, α=0, or memory | Reproduce the corresponding FIT integration smoke; check pinned processor/model, row-map, one-hook behavior, FP32 norm/head parity, and zero repeats. | Repair implementation or support rule in v2, rerun FIT smoke/calibration. Preserve the v1 failure and its raw artifacts. |
| Clean answer <72/96 | Recount FIT strict versus semantic one-word+EOS by word, voice, template and ASR transcription; listen to representative FIT WAVs. Separate speech recognition, prompt following, and surface spelling errors. | Freeze a new task prompt, TTS/rendering, or answer policy **using FIT only**, then test on a new development set. Do not remove wrong v1 development cases. |
| Prefix coverage <72/96 or purity <75% | On FIT, tabulate first-token class versus complete answer+EOS, prefix mass, ambiguous shared prefixes, and canonical full-sequence score. | Freeze revised candidate surfaces or a complete-sequence readout in v2 before fresh development; retain v1 V2-prefix results unchanged. |
| Too few above-floor effects or sign <75% | On FIT, plot measured versus predicted V2 change across the frozen α grid and both sites; inspect direction orientation, effective edit norms, projected terminal residual, and the known `deer`/`dear` homophone misses without excluding them. | Fit a new local/corpus map, target basis or source bin policy on FIT only; predeclare rank and regularization. Measure it on fresh pairs. |
| Map MAE fails against zero | On FIT, compare absolute score movement, numerical floor, FP32 local finite differences, projection capture, and curvature by α. | Improve the mechanistic estimator or choose a better predeclared small-step regime in v2; do not tune to v1 development errors. |
| Map MAE fails against global or six-bin ridge | On FIT, reproduce pair-held-out CV, training-only RMS, saved ridge X/Y, donor-row weights, and score-head predictions. Check whether the extra-capacity six-bin baseline wins only on FIT or consistently across FIT folds. | Try a corpus-conditioned Jacobian or an explicitly regularized map on FIT only. Keep ridge comparators fair and fixed in v2. |
| Absolute anchor <68/96 | On FIT, compare category-centered affine readout with its same-category constant anchor, inspect category leakage, source means, target basis, and score calibration. | Fit better category anchors or a non-oracle absolute readout on FIT only. Declare any changed category information as a new task variant. |

**If Stage A passes at both sites:** preserve the pass as a prefix-score result, then implement the predeclared Stage B time controls and canonical complete-sequence scoring using FIT integration tests first. Freeze the exact Stage B supplement, code, scores, thresholds, and resource bound before reading development Stage B outputs. Run all 96 directions at both sites; report time-control, natural-donor, full-row-patch, and full-answer outcomes separately. A Stage A pass alone establishes neither complete-answer transport nor a security finding. Advance to the locked test only under the frozen Stage A/B decision rule.

**If any Stage A gate fails:** stop v1 and publish the negative result with per-pair and per-voice counts. A v2 repair may use FIT-only diagnostics and new predeclared model/code, while the already-inspected v1 development split becomes exploratory for v2. To make a new prospective claim, create a fresh development set before any model tuning on its outcomes; keep the current locked test sealed until a versioned, reviewable protocol earns a one-shot test. Do not claim that changing a threshold or dropping an item rescues v1.
