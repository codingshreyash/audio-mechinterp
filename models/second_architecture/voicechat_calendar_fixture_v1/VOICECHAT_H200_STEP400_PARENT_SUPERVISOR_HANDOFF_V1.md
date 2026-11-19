# VoiceChat step-400 parent supervisor: source-only handoff V1

19 November 2026. **HOLD GPU/model inference.** This is a bounded constituent for
the reviewed step-400 diagnostic, not an executable launch or a GO. It leaves
the pinned NeMo helper, previous H200 attempts, and the reviewed probe core
unchanged.

## Candidate code and verified behavior

[`voicechat_step400_supervisor_v1.py`](voicechat_step400_supervisor_v1.py)
starts one externally admitted command in a new POSIX session and monitors its
process group. It requires a nonexistent attempt directory, an existing staged
checkpoint directory, the output on the same filesystem, and at least 30 GiB
of free disk after staging. Its defaults are a 130,000 MiB sampled device and
process GPU cap, 1,800 s wall cap, 64 MiB retained evidence cap with a 64 KiB
terminal reserve, and bounded stdout/stderr. It passes only an inherited
heartbeat pipe descriptor to the child and kills the process group on missing
heartbeat, cap breach, NVML failure, observed sampling gap, evidence overflow,
or child failure. It never converts a pre-400 resource stop to success.

The default telemetry is direct `pynvml`, with one initialization held across
polls; it uses the staged V5 runtime's pinned NVML binding. Every completed
sample has begin/end monotonic timestamps and the gap since the preceding
completed sample, including launch-to-first-sample. The parent aims for 0.1 s
polls and rejects **any observed gap above 0.2 s**, including a slow NVML call.
The JSONL poll ledger, bounded logs, terminal receipt, and SHA-256 file census
are retained. After a killed child, an aggregate evidence overflow causes
bounded prefixes to be retained with each truncation declared in the receipt.
The receipt status is only `bounded_child_complete_pending_saved_audit`; it
does not attest to six checkpoints or a model response.

No-model tests in
[`test_voicechat_step400_supervisor_v1.py`](test_voicechat_step400_supervisor_v1.py)
passed on the local CPU: a fake child and poll ledger, slow telemetry gap,
GPU cap stop, missing heartbeat, child nonzero, log limit, evidence overflow,
disk floor, and consumed attempt rejection. Command:

`python3 -W error::ResourceWarning -m unittest models.second_architecture.voicechat_calendar_fixture_v1.test_voicechat_step400_supervisor_v1 -q`

The separately reviewed probe-core tests import PyTorch and were not runnable
in the Mac's system Python, which has no `torch`. The reviewed core's prior
fake-model test result is unchanged by this constituent.

## HOLD items before any run

1. Build the exact child worker around the pinned official random-number
   sample preparation and `run_pinned_pass1_only`; bind and independently review
   the exact NeMo callback, model, WAV, prompt, checkpoint, response fixture,
   child environment, ledger, and heartbeat behavior. This parent accepts an
   arbitrary command and therefore **does not authenticate the child**.
2. Build a fresh external freeze and independent review for this diagnostic.
   Bind source hashes, ignored files, input/runtime/host package inventory,
   actual output path, command, environment, caps, and the review identity.
   The V5 public-sample freeze is consumed and cannot be reused. On host,
   verify direct NVML process reporting and actual cadence before one-shot GO.
3. Independently review this parent source. Its fake tests do not prove that a
   live H200 can sustain 0.2 s coverage under model load. A failed or missing
   poll ledger is a coverage failure, never evidence of an in-cap trajectory.
   Review failure sealing for unexpected child-created links, special files,
   or more than 32 files; this candidate currently rejects them and may lack a
   terminal receipt in those adversarial cases. The admitted child must not
   create them.
4. After a permitted run, an independent saved auditor must rehash all files,
   reconstruct every poll interval, checkpoint order, baseline and step-400
   stop, evidence truncations, cap status, and terminal receipt before making
   any diagnostic interpretation. Neither this parent nor the core alone can
   establish a memory mechanism or security finding.

No GPU, model inference, instance start, or official helper edit occurred for
this handoff.
