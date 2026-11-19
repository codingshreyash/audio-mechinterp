# Independent review: step-400 parent receipt hardening

19 November 2026. Reviewed commit `270fcae` against the previous V2 source
review HOLD. **PASS for the two scoped repairs; HOLD executable GO.** I did not
launch a GPU or load the model.

The parent now reserves the predictable sibling failure-receipt path with an
exclusive, no-follow open before it launches the child. Its terminal write
uses the held file descriptor and checks that the path still names the same
regular, single-link inode. If a same-UID child replaces the path or writes
into the empty reserved file, the parent reports an unverified receipt rather
than accepting child text as its own. The five added fake-process tests cover
exclusive-create collision, unlink/replacement, in-place write, NVML startup
failure, and process-spawn failure. All **16/16** parent tests passed in the
pinned local environment with warnings treated as errors:

`models/second_architecture/.venv/bin/python -W error -m unittest models.second_architecture.voicechat_calendar_fixture_v1.test_voicechat_step400_supervisor_v1 -q`

For a failure during NVML initialization or `Popen`, after the reservation is
made, the `except` path now emits a bounded no-go sibling receipt. It records
the exception *type* without its text and attempts to stop a spawned child
group. This fixes the specific previously identified silent consumed-attempt
paths at those two calls.

The guarantee does not cover every pre-spawn failure. Output directory
creation, the initial log/poll file opens, `os.pipe`, and selector setup occur
before the receipt reservation and its `try`. I injected an `os.pipe` failure:
the attempt directory was consumed and **no terminal receipt existed**; no
child was spawned. In-place same-UID writes are detected only at the sampled
checks. A child can still modify the receipt path after a check or otherwise
compete with the parent on a writable shared directory. This is a fail-closed
source boundary, not a parent-authentication proof against a hostile child.
The code still accepts an arbitrary child command, has no pinned executable
launcher or fresh host freeze, and has no live H200 NVML timing measurement.
Those gaps keep executable GO on HOLD. The saved-evidence auditor must treat
missing or replaced receipts as no-go.
