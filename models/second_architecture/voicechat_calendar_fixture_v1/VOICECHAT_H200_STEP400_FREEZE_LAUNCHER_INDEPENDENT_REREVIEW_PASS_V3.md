# Independent step-400 freeze/launcher rereview: source-only PASS

19 November 2026. Supersedes the [V2 executable HOLD](VOICECHAT_H200_STEP400_FREEZE_LAUNCHER_INDEPENDENT_REREVIEW_HOLD_V2.md)
for the build-mode credential issue, after fix `750d987`. Reviewed with the
worker, parent, saved auditor, V1/V2 protocol and V5 public-sample preflight.
**PASS for the launcher source constituent under a trusted same-UID host.**
This is not a signed host GO or a live GPU result. I did not invoke `build`,
`launch`, a model, or a GPU.

## Build credential fix

`build()` now checks that `VOICECHAT_STEP400_REVIEW_KEY_HEX` is **absent as its
first operation**, before `check_paths`, source hashing, V5 pip inventory, or
Nano offline resolution. The CLI build branch enters that same function;
the new direct-call and CLI no-model tests assert that preflight and
subprocesses are not reached when the key is inherited. The launch branch
still pops the key before any host preflight and excludes it from the model
child's frozen environment. This resolves the concrete V2 HOLD.

## Independent adversarial reruns

The three safe source-only reproductions from the earlier review still fail
closed at `750d987`:

| Mutation | Result |
| --- | --- |
| Add untracked executable `models/__init__.py` to a temporary Git project after recording its named source identity | `project_source()` rejects it. |
| Remove an attempt output directory after creating a sibling GO claim, then reuse the same claim path | `O_EXCL` rejects the second consumption. |
| Grow canonical JSON after its initial file stat but before bounded read | `read_json()` rejects the changed file. |

The child bootstrap's fake-project test still confirms that a regular
`models` package in the earlier NeMo path does not shadow the reviewed
project worker. Static source order confirms launch-mode key removal precedes
`check_paths()` and `expected()`. The reviewed HMAC remains freeze-specific;
wrong `PYTHONPATH`, missing review, and changed scope are denied in the
existing no-model tests and source checks.

The launcher suite passes **12/12 tests**. The combined probe, worker,
parent, auditor and launcher suite passes **58/58 tests** with `python -B`
and `ResourceWarning` treated as an error. These tests use fake processes and
telemetry. They establish source and receipt compatibility, not actual H200
poll cadence or memory behavior.

## Trust and execution boundary

This PASS assumes the same-UID environment running the launcher and storing
the source, review, claim and attempt tree is trusted. An actor able to edit
`models/__init__.py` before the launcher imports can execute code before
`project_source()` checks it; an actor with write access to the claim
directory can remove the sibling GO claim. The source freeze and claim
prevent drift and accidental replay under the stated trust boundary. They
are not a privilege separation mechanism against a hostile same-UID actor.

Execution still requires a fresh staged H200 host, exact external freeze and
separate signed one-attempt review, validation of package/NeMo/Nano/native
FFmpeg and disk identities, and review of the parent supervisor's remaining
live NVML and same-UID receipt limits. The H200 remains stopped. No memory
trajectory, model answer, or security finding follows from this PASS.
