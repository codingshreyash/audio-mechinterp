# Stage-B source and receipt schema for independent review

This is a **source scaffold**, not a live launch authorization. The reviewed
Stage-B replay primitives remain pinned by SHA-256. The new driver cannot enter
`--run`: it requires a later independent Stage-B CPU auditor, launcher review,
external committed freeze, and both audited Stage-A arm manifests. No model
was loaded while developing or testing this scaffold.

## Required upstream evidence

The driver checks a separately committed Stage-A freeze and one separately
committed attempt manifest for each Stage-A arm. It requires BF16 then FP32
commit chronology, one attempt per arm, full start/finish/result/audit hashes,
content-addressed probe and detail sidecars, matching recorded byte counts,
fixed Stage-A forward/VJP counts, source/protocol/auditor pins, parity success,
and cumulative model-process times. It then uses the reviewed replay loader to
check the BF16 exact cycle-0 source and LM-input tensor pairs. The result is
unavailable until actual Stage-A attempts and external manifests exist.

## Fixed Stage-B model work

| Fresh process | Ordered model work | Maximum |
| --- | --- | ---: |
| BF16 direct LM | 12 zero/edit LM-input pairs; bitwise parity against BF16 native states and first-answer logits | 24 direct LM forwards, 0 VJPs, 600 s |
| FP32 source replay | Encoder31 then projector; two clean plus one self parity forward per site; each site's single clean graph supplies its fixed derivatives; 12 zero/edit source pairs | 6 parity + 24 pair forwards, 36 + 32 VJPs |
| FP32 direct LM | 12 zero/edit LM-input pairs cast from archived BF16 tensors; first pair at each site checked against a full-model LM-input override | 24 direct LM + 4 override forwards, 0 further VJPs, 1,800 s total FP32 |

Every output is an exclusive receipt. Large details are serialized with a hard
1 GiB per-process ceiling and moved into an exclusive `blobs/<sha256>.pt` path.
The Stage-B detail sidecar saves each exact replay input, its tensor hash,
paired output, source VJPs and parity/validation outputs. The preflight
estimates tensor bytes before loading a model and refuses inadequate headroom.
The driver checks elapsed time, cumulative model-work time, and observed MPS
driver allocation at each checkpoint. The 30 GiB MPS cap and the 4,500 s
cumulative cap include the prior Stage-A work; no grid reduction is allowed.

## Remaining review gates

An independent Stage-B CPU auditor must verify saved tensor hashes, pair
membership, exact source edits, direct-LM parity, replay predictions,
chronology and all counts. A separate launcher must pin the reviewed driver
and auditor, start each phase in a fresh process, apply an external watchdog,
reserve one attempt per phase, and preserve failed receipts. The Stage-B BF16
external manifest must precede the FP32 launch. Those pieces are absent; the
driver's CLI remains closed. A later report must state that the raw sidecars
are local and a Git checkout alone cannot reproduce the numerical result.
