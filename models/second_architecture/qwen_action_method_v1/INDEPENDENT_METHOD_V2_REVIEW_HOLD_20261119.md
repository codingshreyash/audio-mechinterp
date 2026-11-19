# Independent source review: Qwen action method V2

19 November 2026. **HOLD for using the V2 development gate or running candidate inference.** I reviewed committed amendment `9f7de75` against the V1 [source HOLD](INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md). I loaded no Qwen weights, read no candidate or locked outputs, and did not start a GPU.

## Checks that passed

- The 16 V1/V2 no-weight endpoint and readout tests pass. The V2 source audit passes against the local pinned modeling, processing, config and tokenizer hashes. The tokenizer encodes canonical `LIST`, `ASK`, and `CANCEL` as IDs `22852`, `7384`, and `88493`, with EOS `151645`. Commit `9f7de75` adds only V2 files; the V1 wording spec, plan, and original scorer from `ba95c8e` remain unchanged.
- `endpoint_v2.score_complete_answer` now records an empty continuation as censored; an early/repeated EOS or unexpected special token as malformed; an ordinary missing EOS as censored; and an exact label followed by terminal EOS as complete. The raw IDs are retained and a canonical-path flag is separate from semantic label acceptance. These changes address the two concrete V1 endpoint defects.
- The V2 readout is explicitly a three-target, text-fitted mean-VJP approximation to the complete `label + EOS` log score. It is **not** presented as a full-width Jacobian Lens. The frozen FIT has 18 typed contexts, two preselected shared-state sites, a fixed primary site, constant and centered dual-ridge comparators, a dose, and a development-only numerical rule. The arithmetic unit tests cover linear Taylor prediction, teacher-forced EOS contribution, ridge degeneracy and random-vector orthogonality.
- The directional schedule has 30 source/modal cells and 120 arms: 48 development and 72 locked. Its 60 random-control seeds are unique, and the target rotation is fixed independently of candidate outputs. V1 audio-junction controls and the earlier failed Qwen rank-16 Stage A gate remain separate. The source protocol requires a reviewed runner, human WAV QA, runtime tokenizer/shape/parity and memory checks before any candidate inference.

## Reproducible promotion-gate failure

`readout_v2.development_gate` accepts caller-supplied `typed_readout`, `audio_readout`, and `constant_readout` lists **without checking that they are the argmax of the corresponding submitted score matrices**. This permits a passing gate while the numerical VJP predicts the wrong class on all six items. I used balanced truth `LIST, LIST, ASK, ASK, CANCEL, CANCEL`, all six exact complete/canonical flags, and actual class scores with a narrow `0.51` versus `0.49` margin. I swapped those two values in each VJP prediction, making its argmax `ASK, ASK, CANCEL, CANCEL, LIST, LIST`. Passing the intended labels as the separate `typed_readout` and `audio_readout` arguments nevertheless returned `typed_vjp=6`, `audio_vjp=6`, a VJP/constant MAE ratio of `0.04` in both modalities, and `pass=True`. The frozen rule requires the readout's *actual* top class, so this is a false promotion.

The exact no-weight reproduction from repository root is:

```sh
models/qwen_omni/.venv/bin/python - <<'PY'
import numpy as np
from models.second_architecture.qwen_action_method_v1 import readout_v2 as r
truth = ['LIST','LIST','ASK','ASK','CANCEL','CANCEL']
actual = np.array([[.51,.49,0],[.51,.49,0],[0,.51,.49],
                   [0,.51,.49],[.49,0,.51],[.49,0,.51]])
wrong = np.array([[.49,.51,0],[.49,.51,0],[0,.49,.51],
                  [0,.49,.51],[.51,0,.49],[.51,0,.49]])
zero = np.zeros((6,3))
print(r.predicted_labels(wrong))
print(r.development_gate(truth,truth,truth,truth,truth,['LIST']*6,
                         [True]*6,[True]*6,actual,wrong,zero,
                         actual,wrong,zero))
PY
```

The printed predicted labels are all wrong, while the gate prints `pass: True`. The function also takes complete labels and canonical flags independently of the raw `endpoint_v2` records. A future caller could therefore supply a `canonical=True` flag for a malformed or censored continuation; the gate has no record-level consistency check. This is a second integrity gap in the promotion input, though no such candidate record exists yet.

## Clearance conditions

Version the development gate so it derives typed/audio/constant top classes from the score matrices itself, then derives complete labels and canonical flags from six V2 endpoint records per modality, with exact six-script manifest IDs and order checked. Reject a canonical flag on a non-complete path and score/label mismatches. Add a no-weight regression using the false-promotion example above and malformed/empty continuation records. An alternative minimal repair can assert every caller-supplied class equals `predicted_labels` for its matrix, while the future reviewed runner binds endpoint records and source IDs; this must be explicit before promotion.

The V2 numerical method can be studied as a bounded target-restricted approximation after this repair. It still needs the separately reviewed candidate runner, WAV byte and human review, tokenizer and hook alignment, stock/instrumented and donor parity, resource admissions, and a sealed locked-test rule. This review does **not** claim a Qwen audio result, a general reusable Jacobian map, or a security finding.
