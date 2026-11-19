# Isolated Whisper tail/end V2 executable handoff

19 November 2026. **Source-only. No V2 target-model inference, admission, or GO has been performed.** The committed prospective protocol and plan fix one engineered non-speech zero/noise pair, the original attempt and two offsets, the `post_conv_pos` site, five-row tail/end windows, controls, and the `1e-3` native-delta threshold. This handoff does not authorize execution. Independent executable review must precede three new stage-specific admissions and committed GOs.

## Fixed stages and interpretation

The original stage has two no-hook baselines followed by five arms, at most seven calls. Each of `offset_8000` and `offset_16000` has four baselines followed by ten arms per layout, at most 24 calls. All baselines run before patches within their stage. A clean censored or no-contrast pair is recorded and its layout's arms are skipped. A censored patch is retained and fails the complete-answer criterion. Original isolated-window failure does not select another window or suppress the two later fixed schedules; it blocks the joint claim. Every attempt has a deterministic terminal skipped-arm list. No stage may be rerun under the same GO.

For prepend, the aligned old tail/end window moves to rows `523..527` or `548..552`; for append it stays at `498..502`. Append's translated rows are the **new waveform end**, where V1's saved native donor-minus-recipient delta was zero. V2 records fresh native norms at both windows. The competing direct donor patch is reported even when it produces a third answer. The equal-norm relocation of the nonzero aligned delta into the competing window is the interpretable control when the new-end native delta is below `1e-3`. A strict result requires complete donor EOS transfer at aligned rows, complete recipient EOS paths for identity, unrelated, displaced and relocated controls, full-donor path and raw-vector parity, opposite direct failure to transfer, eligible contrasts, mel-shift qualification, original-window viability, and both offsets passing. A third answer from the opposite direct patch satisfies only the primary failure-to-transfer check; a separate secondary field reports whether it preserved the recipient. A 2/2 result is conditional on this single constructed pair and cannot establish a unique circuit or security problem.

## Source and execution gates

The V2 plan is `WHISPER_TAIL_END_ISOLATION_PLAN_V2.json`; it pins V1 plan SHA-256 `b1aa22f0b0c9a357a641dcc7d3ee5e1a789d2337d9b0bbb81ae131e6e3162f61` and the eight-WAV manifest SHA-256 `eec732fb24a4b238ad0666d52d168f5ba41ec38c9d315a057a39e0d53be1435b`. It reuses their bytes and creates no new audio. The V2 gate binds the committed source census and the inherited V4 asset/runtime admission. Use the admitted `models/second_architecture/.venv/bin/python` from the repository root. Parent, worker, and auditor use new `runs/whisper_tail_end_isolation_v2/{stage}_attempt_001` paths; V1 attempts, admissions and GOs stay untouched.

After independent executable review and one stage's committed admission and GO, the intended parent command is:

`models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.whisper_tail_end_isolation_parent_v2 --stage original --admission <fixed V2 path> --go <fixed V2 path>`

Substitute `offset_8000` or `offset_16000` for the later stages. The parent checks a fresh fixed path, an open lid, no thermal warning, battery at least 80% initially and above 70% while running, and power samples at most 15 seconds apart. It limits the full attempt to 1,800 seconds, process-tree RSS below 4 GiB, and retained evidence below 1 GiB. It creates the fixed attempt, records host/power/resource evidence, and sends a live heartbeat. The inherited pipe prevents accidental direct runs under a trusted local-owner assumption; it is not cryptographic parent authentication.

After the parent reports `completed_pending_saved_audit`, the read-only no-weight auditor command for each stage is:

`models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.whisper_tail_end_isolation_saved_audit_v2 --stage original --admission <fixed V2 path> --go <fixed V2 path>`

The saved auditor checks parent and host receipts, call order and skips, full EOS or censor paths, patch arithmetic and untouched encoder rows, full frozen-WAV rehash and pinned-processor mel/mask reconstruction for every baseline, source and applied relocation norms, and the predeclared decision. A stage `PASS` says the saved evidence is internally consistent. The joint no-weight command **reaudits all three fixed raw attempts** and does not trust editable summary JSON:

`models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.whisper_tail_end_isolation_saved_audit_v2 --join`

It reports 0/2, 1/2 or 2/2 strict offsets, original viability, the stronger competing-patch secondary condition, and the append new-end native-delta caveat.

## Source-only checks

From the repository root, run:

`models/second_architecture/.venv/bin/python -m unittest models.second_architecture.nonspeech_probe_v1.test_whisper_tail_end_isolation_plan_v2 models.second_architecture.nonspeech_probe_v1.test_whisper_tail_end_isolation_worker_v2 models.second_architecture.nonspeech_probe_v1.test_whisper_tail_end_isolation_saved_audit_v2 models.second_architecture.nonspeech_probe_v1.test_whisper_tail_end_isolation_gate_v2 models.second_architecture.nonspeech_probe_v1.test_whisper_tail_end_isolation_parent_v2 -v`

These are no-model tests of the schedule, branch rules, strict decision, live-parent and resource controls. They do not establish target behavior. Independent review should also inspect the V1 helpers reused by V2 and the V3 hook arithmetic before any GO.
