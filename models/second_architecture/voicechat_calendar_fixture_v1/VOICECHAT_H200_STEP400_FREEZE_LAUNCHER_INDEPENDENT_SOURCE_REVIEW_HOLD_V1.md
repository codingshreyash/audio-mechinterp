# Independent step-400 freeze/launcher source review: HOLD

19 November 2026. Targets: initial candidate `32715b4` and its test-binding
follow-up `28448fa`. Reviewed against the V1/V2 prospective protocol, worker
`f621096`, parent `f0f9094`, saved-auditor rereview `abfd5b4`, and the V5
public-sample preflight. **Source-only executable HOLD.** I did not invoke
`build`, `launch`, a model, or a GPU.

## Checks that held

The six launcher no-model tests pass under `python -B` with resource warnings
as errors. The command is fixed to the diagnostic worker; the old V5 sample
review has a different schema/key and cannot directly authorize it. A missing
or changed signature is denied. `expected()` requires exactly the pinned NeMo
checkout as `PYTHONPATH` before expensive preflight, and
`sample.child_environment()` excludes `VOICECHAT_STEP400_REVIEW_KEY_HEX`
from the **model child**. The build path requires the exact successful
`sample_only_ready_preload_passed_no_model_work` status. The launch path
rechecks current source, checkpoint, packages, native FFmpeg, Nano cache,
offline metadata, H200 runtime and actual free disk against the signed
freeze; the build-time preload identity is compared with a fresh preload
check. These source calls do not prove a particular live host was checked.

## Findings that prevent an executable GO

### 1. Unfrozen importable project code can execute

`project_source()` hashes ten named local files and records `git rev-parse
HEAD`; it does not require a clean project worktree or census importable
extras. `models`, `models/second_architecture`, and the fixture directory are
currently namespace packages without `__init__.py`. In a temporary Git repo
containing the ten named files, I added untracked `models/__init__.py` with
executable code. `project_source()` returned **the exact same frozen source
identity and HEAD** before and after the addition. The signed child bootstrap
imports `models.second_architecture...voicechat_step400_worker_v1`; Python
would run that initializer before the worker's own checks. The same concern
applies to importable files shadowing project modules through the child CWD
or a package earlier on `sys.path`: the bootstrap appends the project root
after the pinned NeMo `PYTHONPATH` and installed packages, and it does not
verify the origin of the imported project modules.

The source freeze must reject unreviewed importable files/initializers and
bind the origin of every project module the child imports. This can be a
targeted package-tree census; a blanket clean-repository requirement may be
impractical given research artifacts. The actual NeMo checkout has its own
clean-worktree check, which does not cover this project tree.

### 2. The review HMAC key reaches preflight subprocesses

`launch()` leaves `VOICECHAT_STEP400_REVIEW_KEY_HEX` in parent
`os.environ`. It calls `expected()`, whose V5
`installed_environment_receipt()` starts `pip freeze` without an explicit
sanitized `env`. The Nano offline resolver constructs its subprocess
environment with `os.environ.copy()`. I mocked only `subprocess.run` and
confirmed `verify_nano_offline()` passes the review key to the resolver
child. Thus the key is excluded from the model child but **not parent-only**
as the handoff claims. Remove the key from all preflight subprocess
environments after parsing it, or pass an explicit stripped environment.
Make the review/authentication boundary explicit and test that no child
process receives the credential.

### 3. One-attempt review is replayable if output disappears

The HMAC binds the exact freeze and output path, and an existing output
directory prevents a normal repeat. The launcher never creates a separate
durable consumed-GO claim; `verify_review()` accepted the same signed record
twice in a no-model call. If the attempt directory is moved or deleted after
one run, the same signed review can pass `check_paths()` again at the same
path. A one-attempt claim requires a protected, durable external consumption
record with atomic creation. This is a source-level replay condition; no
second model run was attempted.

### 4. Review/freeze read cap is checked only before reading

`read_json()` stats the file against its cap and then calls unbounded
`Path.read_bytes()`. In a no-model test I supplied a valid canonical JSON
body larger than the cap after the initial stat; the function accepted it.
An attacker able to grow/replace the file during that interval could force
an oversized read, despite the advertised 256-KiB/16-KiB caps. Use a
single opened regular file descriptor, bounded `read(cap+1)`, and post-read
size/hash checks, with no symlink follow; test a changing file.

## Scope of the HOLD

The first two findings affect the authenticated executable/source boundary;
they are enough for HOLD. The replay and bounded-read issues strengthen it.
The earlier saved-auditor source HOLD was resolved at `abfd5b4`, but its
source-only PASS does not authenticate the launcher or lift the parent
supervisor's remaining live/executable limits. `python -B` is an operational
precondition: invoking this candidate without it can create the very
`__pycache__` entries `project_source()` rejects. No new memory trace,
VoiceChat answer, or security finding is established here.
