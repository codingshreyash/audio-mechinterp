# Old-FIT Stage B: external one-shot launch and audit protocol

18 November 2026. This protocol implements the Stage B phases in
`VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md`; it does not change the
one-case estimand, source sites, cycle-zero tensors, probes, signed steps,
planned counts, or interpretation thresholds. The original Stage A BF16
launcher remains `launch_failed`. Its later v2 audit establishes arithmetic
over saved tensors only. The separately frozen Stage A FP32 continuation is a
different, successful arm. Stage B uses both exact saved sources and preserves
that history.

## Source and freeze order

The driver, corrected CPU auditor, this launch gate, worker, audit adapter,
tests, and this protocol require independent source reviews. The driver CLI
stays closed. The worker is the only model entry point and checks an externally
committed freeze and matching reservation/start receipt before calling the
driver's `run_phase` function. The adapter loads no model; it calls the
separately reviewed CPU tensor auditor against the exact saved Stage A and B
files. A source review is not a freeze.

For BF16 direct-LM replay, create a separate first-add-immutable external
`VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_EXTERNAL_FREEZE_V1.json` only after the
launcher source review. The freeze binds every file listed in the launch gate's
`BOUND_FILES`, the launcher review commit, the exact Stage A model-work time,
the single permitted phase, and the 4,500-second and 5-GiB aggregate caps.
An independent reviewer then writes
`VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_EXTERNAL_FREEZE_INDEPENDENT_REVIEW_V1.md`.
The gate rejects missing, dirty, symlinked, edited-after-first-add, unreviewed,
or stale freeze/evidence bytes. It verifies all 24 Stage A attempt files,
including four local-only tensor sidecars, and keeps the BF16 failure status.

The BF16 phase is one reserved attempt. Before the child process, create an
exclusive `bf16_direct_lm_reservation.json`, then a UTC `launch_start.json`
under `runs/voxtral_old_fit_handoff_stage_b_launch_v1/bf16_direct_lm/<id>/`.
The worker writes only to
`runs/voxtral_old_fit_handoff_stage_b_v1/bf16_direct_lm/<id>/` and never
inherits a previous model allocator. Its required work is 24 direct-LM
forwards and zero VJPs. The parent enforces a 600-second model-work ceiling,
120-second process grace, 1-GiB worker output ceiling, 8-MiB capture ceiling
per stream, 30-GiB observed MPS driver cap in the worker, cumulative
4,500-second model-work cap, and 5-GiB serialized-output cap. The model-work
clock includes the 63.245136791840196-second failed BF16 Stage A process and
155.3182365419343-second FP32 Stage A process. A failed attempt consumes its
reservation and all observed model-process time; there is no retry, partial
grid, or edited source under that attempt ID.

After the worker exits, the parent rechecks source/freeze hashes and runs the
CPU audit adapter with a 120-second watchdog. The adapter rehashes the Stage A
sidecars and recomputes the Stage B saved-tensor arithmetic. Its output is
saved as `audit.json` alongside the worker result. The parent writes a final
UTC `launch_finish.json` with child/auditor return codes, exact hashes, elapsed
time, resource totals and either success or a preserved failure reason.
Small raw files and wrapper receipts must be committed unchanged before an
external BF16 Stage B manifest. The large content-addressed sidecar remains
local and is required for an independent re-audit. The BF16 manifest has the
driver's fixed schema, including the launch start/finish, reservation, freeze,
result and audit hashes. An independent result reviewer must check those
bytes, all tensor arithmetic and the original Stage A failure, then commit a
separate GO/HOLD review. Saved files cannot independently attest that a model
forward or backward actually occurred.

## FP32 continuation

Only after the BF16 Stage B manifest and independent result review exist, a
second first-add-immutable external
`VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_FP32_CONTINUATION_FREEZE_V1.json` may be
committed. It binds the first freeze, exact BF16 Stage B manifest/review bytes
and first-add commits, completed BF16 parent model-work time, Stage A time, and
the same aggregate caps. A separate independent review of that second freeze
is required. The FP32 phase uses its own exclusive reservation and launch
receipts and a fresh process, with 1,800-second model-work ceiling. It runs
24 source-pair forwards, six source parity forwards, 68 source VJPs, 24 direct
LM forwards and four full-model LM-input override forwards. The first freeze
alone never authorizes FP32. If the BF16 attempt fails, no FP32 continuation is
permitted under this protocol.

The CPU audit checks selected-row equality, exact source and replay input
hashes, projector self derivative identity, stage projections and VJP
contractions, all paired-zero noise, odd/even response, null resolution,
counts, tensor hashes, progress, layout and resource receipts. The audit
scope remains one old FIT lexical contrast. A Stage B result may diagnose
whether the BF16/FP32 difference appears in direct LM replay or in FP32
downstream replay of identical archived BF16 sources. It cannot by itself
identify one BF16 kernel, establish transfer to new speech, or prove a
security vulnerability.

The exact source-review markers required by the gate are:

* `Decision: GO for Stage-B continuation driver source; live model HOLD until launcher review and freeze.`
* `Decision: GO for repinned Stage-B auditor source only; live Stage B remains HOLD pending its separate admission, chronology, freeze and review gates.`
* `Decision: GO for Stage-B launcher source; model run HOLD until external freeze.`

The separate BF16 result review and freeze reviews have explicit GO markers
in the launch gate source. All missing receipts and reviews are hard stops,
never implied approvals.
