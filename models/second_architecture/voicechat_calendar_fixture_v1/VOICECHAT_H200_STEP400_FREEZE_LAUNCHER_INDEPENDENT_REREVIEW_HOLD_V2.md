# Independent step-400 launcher rereview: executable HOLD

19 November 2026. Supersedes the [V1 source review](VOICECHAT_H200_STEP400_FREEZE_LAUNCHER_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md)
for `32715b4`/`28448fa` with the fixes in `f09ebdd`. Reviewed against the
worker, parent, saved-auditor, V1/V2 protocol, and V5 public-sample
preflight. **The four original launch-path defects were addressed, but the
executable remains on HOLD for the residual credential and entrypoint
boundary below.** I did not invoke `build`, `launch`, a model, or a GPU.

## Independent no-model results

| Original V1 case | Result at `f09ebdd` |
| --- | --- |
| Add untracked executable `models/__init__.py` without changing named hashes or HEAD | `project_source()` now rejects it. I repeated the temporary Git-repo mutation and got `unreviewed package initializer in project import path`. |
| Review key inherited by launch-time pip/Nano preflight subprocess | `launch()` removes `VOICECHAT_STEP400_REVIEW_KEY_HEX` before calling `expected()`; the model child environment still excludes it. |
| Reuse same signed GO after removing the attempt output | A durable sibling claim is created with `O_EXCL` before parent spawn. I called `consume_go()` twice with the attempt directory removed between calls; the second call failed with `FileExistsError`. |
| Grow canonical review/freeze JSON between stat and read | `read_json()` now opens with `O_NOFOLLOW`, reads at most cap+1, and verifies inode, size, mtime and ctime after reading. An independent growth mutation returned `review/freeze changed during bounded read`. |

The new fixed bootstrap removes the child CWD from `sys.path`, places the
pinned NeMo directory first, and pre-seeds the three project namespace
packages at exact paths. Its no-model child test places a malicious regular
`models` package in the earlier NeMo path and confirms only the intended
project worker runs. The 10 launcher tests pass, as do all **56 combined
no-model tests** for the probe, worker, parent, saved auditor, and launcher,
with `-B` and `ResourceWarning` treated as an error. The tests validate
source control flow and fake-process schemas; no runtime H200 behavior is
claimed.

## Residual HOLD conditions

1. **Build-mode review-key propagation.** `main()` removes the key only in
   `mode == "launch"`, and `launch()` removes it again. `build()` does not
   remove or reject it. If a user invokes build while
   `VOICECHAT_STEP400_REVIEW_KEY_HEX` remains in the shell environment,
   `sample_ready_preload()` runs the V5 `pip freeze` and Nano offline
   resolver with that inherited key; the resolver explicitly starts from
   `os.environ.copy()`. This retains the V1 credential leak on the build
   path. Remove/reject the key before either mode enters preflight, and
   include a no-model test for build mode with the key present.
2. **Checks occur after Python entrypoint imports.** The new
   `project_source()` guard rejects an added initializer before model spawn,
   but the launcher itself imports through the `models` namespace before
   `project_source()` runs. If a same-UID actor adds `models/__init__.py`
   before invoking the launcher as a module, Python executes that file
   during import and only then reaches the guard. The fixed bootstrap
   protects the *model child* import origin; it does not establish that the
   launcher entrypoint was immutable before its own first import. A trusted
   external entrypoint/source staging boundary is required if malicious
   same-UID modification is in scope.
3. **The consumed-GO claim is same-UID removable.** The sibling file is an
   effective accidental-replay guard and is atomic against concurrent
   creation. A process with write access to its directory can unlink the
   claim and the attempt directory, then reuse the same signed review. The
   one-attempt claim must be explicitly scoped to a trusted same-UID host,
   or stored in a protected directory/service outside that actor's control.

The build credential leak is independently actionable even on a trusted
host; it is the immediate reason this rereview remains HOLD. The entrypoint
and claim points define the same-UID threat limit, not evidence that such an
actor existed. No saved memory trajectory or model result was produced.
