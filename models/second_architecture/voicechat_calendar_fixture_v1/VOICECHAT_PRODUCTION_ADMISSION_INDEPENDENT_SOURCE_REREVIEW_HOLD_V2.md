# VoiceChat fixture child-environment independent source rereview v2

Decision: **HOLD for live production worker source after `566bd37`.** The fix closes key retention in the intended supervised path, but a direct worker entry can retain arbitrary extra environment variables at the model boundary while passing the new public-environment projection check. No model weights, NeMo or GPU were used in this review.

| Exact changed role | SHA-256 |
| --- | --- |
| admission | `cd6c5fcdda4e22f666f5c9e1686f3827e2a6a12d08f94ad7694604c54064e837` |
| single-process supervisor | `62c941069bf843fd4829e4f2247f2bb51794fa630abd61c62a6a27181a8672e7` |
| worker | `61a0167128927c63e9e67c9e5864f7e3dc318e44ee820f71c50b314d87ede2ed` |
| admission tests | `d3766d2559ab8af061484db4087610ef7675ed04842e2b7a1262e8281b6965a6` |
| supervisor tests | `7ef0584c645e90ccca037810a328690aa9e232b95c88a88564d05b1d6ae05b42` |
| worker tests | `f3975d84d6eadae2fc4fb77f1a2b249111e66671d6a36e49253514e1ed33043c` |
| admission handoff | `82703486244020685dc517b8f944c4795f3306f82ff3310ee6240391c9848cd5` |

The full 18-role canonical source-map SHA-256 is `bd22139b1a2192f9685ee45c3368c4068d96d0905186bcbe42657d3e1a619582`; `verify_source_commit(566bd371014ab09a3132c749996db5ba480fdeee, source_hashes())` passed. All 20 no-model admission/supervisor/worker tests passed, as did `py_compile` and `git diff --check 566bd37^ 566bd37`.

## What the fix establishes

- The supervisor freezes an explicit public child environment and its hash, excludes an unrelated parent secret, then adds only the fixture freeze-review key and controller key needed for child admission. An independent fake supervisor probe observed the unrelated secret absent and both keys present at child start.
- The worker requires both keys during admission/preflight and removes both from `os.environ` in a `finally` block even when admission raises. An independent fake direct-worker probe confirmed both keys present during admission, both absent at the fake arm/model boundary, and absent after a synthetic admission exception.
- The complete source-plan rehash and prior tools-hash protection are unaffected. The default external anchor and official-sample verifier still fail closed. No stock arms or final trimmed-code/PCM parity comparison are implemented by this one-process slice.

## Remaining blocker: projection is not full environment isolation

In a fake direct worker call, the environment contained `UNRELATED_PARENT_SECRET=synthetic` plus the two required key variables. The fake admission returned the current `child_public_environment()` projection as its frozen value. After admission, the worker removed the two keys, and its equality check against that projection passed, but `UNRELATED_PARENT_SECRET` remained visible at the fake arm/model boundary. The projection ignores all names outside `CHILD_ENV_KEYS`; equality of projected dictionaries cannot detect extra variables. This is not a bypass of the current default-closed anchor/sample gates, and the reviewed supervisor does not forward that extra variable. It is a concrete latent direct-worker CLI gap if those gates are later connected.

Require the worker to scrub every variable outside the frozen public allowlist after admission or reject any extra key before model factory. Test both the normal fake model-boundary path and admission-exception path, including an arbitrary extra variable. Re-review that exact source before a live worker GO. The two required admission keys can remain present during pinned preflight, but must be absent before NeMo import/model creation.
