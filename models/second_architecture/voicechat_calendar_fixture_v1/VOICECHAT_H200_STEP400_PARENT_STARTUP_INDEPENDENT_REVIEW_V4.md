# Independent review: step-400 parent startup receipt V4

19 November 2026. This supersedes the startup-failure portion of
[the V3 review](VOICECHAT_H200_STEP400_PARENT_SUPERVISOR_HARDENING_INDEPENDENT_REVIEW_V3.md).
Reviewed supervisor commit `f0f9094`. **PASS for the scoped startup repair;
HOLD executable GO.** No GPU or model inference ran.

The parent now creates the exclusive sibling failure-receipt inode *before*
it creates the attempt directory. Directory creation, three initial evidence
files, `os.pipe`, selector creation, poll-file opening, NVML initialization,
and child process spawn occur inside the exception handler. I injected four
early failures without changing the supervisor source:

| Injection | Attempt directory | Sibling receipt |
| --- | --- | --- |
| `os.pipe` raises | Exists | `invalid_or_partial`, `parent_exception` |
| `output_dir.mkdir` raises | Absent | `invalid_or_partial`, `parent_exception` |
| initial `stdout.log` creation raises | Exists | `invalid_or_partial`, `parent_exception` |
| selector creation raises | Exists | `invalid_or_partial`, `parent_exception` |

A preexisting sibling path is rejected before attempt creation; its contents
are left alone. A preexisting attempt directory is also rejected before the
parent reserves a sibling path. The new regression test covers `os.pipe`.
The 17-test parent suite initially had **one intermittent failure** in an
older test's assertion that child stdout contains `reservation held`; the
same test passed alone and the full suite passed **17/17** on rerun. That
stdout race does not change the receipt check, but the test should not be
treated as perfectly stable.

The scoped repair does not authenticate a writable sibling path against a
hostile same-UID process. A failure to reserve that path happens before the
attempt is claimed and returns an exception without a receipt. A child can
still race later receipt checks, and a synchronous NVML call that never
returns can defeat the wall-time watchdog. The parent still accepts an
arbitrary command and has no reviewed exact launcher, fresh host freeze,
live polling measurement, or one-shot GO. These are executable blockers.
