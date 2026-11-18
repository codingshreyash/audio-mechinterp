# Independent result review: amended BF16 old-FIT numerical arm

18 November 2026. **Receipt and arithmetic verdict: PASS for this one old-FIT numerical diagnostic. Local linear prediction does not describe the measured BF16 changes at the four selected LM coordinates.** This review did not run the model or open the fresh v2 corpus. It is not an audio lens or a security finding.

## Independent checks

I audited the committed `runs/voxtral_v1_fit_backward_numerical_launch_v2/bf16/20261118T122332119062Z/` attempt (`a5ef248`). All ten raw files, including `runner/details.pt`, match their committed `HEAD` blobs byte for byte. The runner result SHA-256 is `904619af29e62ba716f3e9f924dcfdcc684ef449dc6cbf45f26265962d352a64`; the tensor sidecar SHA-256 is `a97f68178118c01469e0ea2d48c3f3a3a1293202af772a17337a62704350fe5a`. Launch start and finish bind the external freeze receipt SHA-256 `7c8029a9204b55ea2cf010e7e516a7ae34d125b1dff012f4b396b1244589f862`, reviewed runner `93d2c607…`, and auditor `d0902608…`. The pinned-venv `--check-freeze` still passes.

I reran the frozen CPU auditor on the raw runner directory. Its JSON equals the saved `audit.json` object and reports 36 verified zero/edit pairs and 72 patched forwards. I separately loaded the saved tensors: at each of encoder31 and projector, all 18 zero prepatch row tensors and all 18 edited prepatch row tensors equal the saved clean shared-graph selected rows bit for bit; all 18 zero terminal tensors equal the saved clean terminal tensor. Three repeated zero baselines per signed edit have zero four-coordinate and full-vector variation. The loaded layout receipt lists 762 named MPS parameters: 761 BF16 and one FP32 `model.audio_tower.embed_positions.weight` of shape `[1500,1280]`, exactly the amended expectation. The runner reports 29.97 s model work and 12.94 billion bytes observed peak MPS driver allocation, below its 30-GiB cap. The natural and passive suffix IDs are both `[5934,14844,2]`, as in the independently reviewed FP32 arm.

## Numerical comparison

These numbers use the saved raw tensor differences and the auditor's recomputed four-coordinate contractions. Relative L2 error is `||predicted−measured||₂ / ||measured||₂` when the selected measured change is nonzero. It is undefined as a useful ratio for the BF16 encoder α=−0.1 cell, where the selected four coordinates do not move at all.

| Arm and site | Nonzero measured four-coordinate cells | Median relative error across those cells | Full 3,072-coordinate terminal L2 range across six signed α values |
|---|---:|---:|---:|
| BF16 encoder31 | 5/6 | 0.9993 | 1.188–1.319 |
| BF16 projector | 6/6 | 0.9997 | 1.250–1.464 |
| FP32 encoder31 | 6/6 | 0.0722 | 0.0059–0.0958 |
| FP32 projector | 6/6 | 0.1655 | 0.0059–0.0982 |

The FP32 comparison is the separately frozen v1 arm and its [independent review](VOXTRAL_V1_FIT_BACKWARD_FP32_RESULT_INDEPENDENT_REVIEW.md), on the same old-FIT word/bin but with the whole model in FP32. Its terminal L2 grows with |α|: approximately 0.006 at .00625, 0.024 at .025, and 0.094–0.098 at .1. In BF16, the full terminal difference stays roughly 1.2–1.5 while the effective edit norm spans 0.054–1.426 across the two sites. Each BF16 edit changes 2,715–2,799 of 3,072 terminal coordinates. At encoder α=−0.1, the four selected coordinates remain exactly unchanged even though the full terminal delta L2 is 1.254. This is a descriptive, scale-insensitive and quantized-looking response pattern, not proof of its cause.

BF16 source-row rounding is material at small edits: for α=.025, the realized-mean deviation and row nonuniformity ratios are about 26.0%/28.6% at encoder31 and 15.0%/16.1% at projector, above the original 10% diagnostic threshold. At α=.1, those ratios fall to about 6.5%/7.4% and 3.6%/4.1%, yet the selected-coordinate prediction errors remain about 99% to 105% on nonzero signals. The four selected BF16 target coordinates have local ULP sizes 0.015625–0.0625; the α=.1 local predictions are smaller than one ULP per coordinate, while measured changes sometimes jump several ULPs. Better realized-source fidelity at α=.1 does not resolve the mismatch. A full-model arithmetic difference, source or downstream rounding, nonlinear response, and other numerical effects remain possible explanations; these data do not isolate them.

## Scope

The CPU auditor verifies saved edits, source parity, contraction arithmetic, receipt hashes, shapes, and finite values. It does **not** independently rerun model forwards, attest process memory continuously, or content-bind the saved donor/no-grad terminal vectors; source hashes for clean/donor whole states remain report-only. The FP32 and BF16 arms change the entire model arithmetic, not just the source edit cast. This one old-FIT recording, two source sites, six signed α values, and four raw LM coordinates cannot establish a reusable Jacobian, held-out transfer, a generated-speech behavior, or a security property. The natural next test is a separately frozen diagnostic that distinguishes source-rounding, target-rounding, and intervening BF16 computation on more than one FIT input before any development claim.
