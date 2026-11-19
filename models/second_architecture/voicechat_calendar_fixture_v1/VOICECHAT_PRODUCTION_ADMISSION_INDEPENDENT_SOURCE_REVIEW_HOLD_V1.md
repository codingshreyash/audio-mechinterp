# VoiceChat fixture admission and adapter supervisor independent source review v1

Decision: **HOLD for live fixture source admission.** The bounded source slice in `38d54fb` correctly closes the prior fake tools-hash bypass and stays closed by default, but it sends controller/review secrets and arbitrary inherited environment variables into the eventual model process. No model weights, NeMo, GPU or calendar fixture were run in this review.

| Changed source/test role | SHA-256 |
| --- | --- |
| `live_parity_production_admission_v1.py` | `6aa6425f2c1dce3977af651e0015c1e6ac99dd5b3a8ac0d1c19939496b7bd4c4` |
| `live_parity_single_process_supervisor_v1.py` | `0c37507f3f8f8a9738512f9f2f5a8ed0a35cf5c80e6a8afc1c89037f443c587b` |
| `live_parity_fixture_worker_v1.py` | `e3ecee6bb93a8a186035167262147377c2ba95b702aec82cfb45c066152f5526` |
| `test_live_parity_production_admission_v1.py` | `aca065aaf17df879d00803d40cbb9c4b46c6c6ab4bbab5f351d10bb2226c5712` |
| `test_live_parity_single_process_supervisor_v1.py` | `54e376e35c2502eba4e1e995d5e4876b8a9a00a31166e63055270b5989c2a810` |
| `test_live_parity_fixture_worker_v1.py` | `7f38236153a11b5b223b7274d9bdfec28319b81eba05bad8d0184fcc10af4fc0` |
| `VOICECHAT_PRODUCTION_ADMISSION_SOURCE_HANDOFF_V1.md` | `6bc35f709dbb902d96a592e26e1ab1d423021a47606b9625069ff2dd1247aa5c` |

The complete `source_hashes()` map has 18 roles; its canonical JSON SHA-256 is `aa08f0ba68b0f2ddbf3dabced0bbae3a4403b6efbe8436234b4314ff0d3fe3b5`. `verify_source_commit(38d54fb6cc48e5815708f3d028096b2f6e3f0515, source_hashes())` passed for all roles.

## Confirmed gates and scope

- `admit_fixture` rejects synthetic plans, requires fixed B/Piper and arm order, verifies the reviewed plan's complete current `_input_hashes` map, runs the real 12-WAV QA checker and selected receipt verifier by default, requires an external anchor callback, binds exact source roles to a committed tree and source-review record, verifies an HMAC-covered executable freeze, reruns `ready` then `preload`, and rehashes after admission. The real QA checker validates signed C grants; the external human/anchor attestation callback remains deliberately closed. The changed-tools probe in the new tests proves `_input_hashes` rejects a modified `tools.json` before a model factory.
- Parent and child both default to closed official-sample receipt verifiers. The single-process supervisor reserves an exclusive root and launches one adapter child only after admission, uses the bounded process runner, checks PID/resource/status and nested capture, reloads its lossless capture, and derives stock response bytes from the **saved** adapter trace. A stock arm must later compare its response hash to this saved file; that launcher does not yet exist.
- The worker repeats admission and QA before its NeMo import and model factory. The official helper hash is checked at import. It records adapter passes and saves the raw capture. Source/tests have no production injection switch exposed by the worker CLI.
- Eighteen no-model tests passed across admission, supervisor and worker. `py_compile` and `git diff --check 38d54fb^ 38d54fb` passed.

## Blocking child-environment defect

`launch_adapter_once` passes `os.environ.copy()` directly to the fresh model child. The worker CLI reads `VOICECHAT_FIXTURE_FREEZE_REVIEW_KEY_HEX` and `VOICECHAT_CONTROLLER_KEY_HEX` from this environment to repeat admission, then retains the variables through the later NeMo import and VoiceChat model load. The executable freeze binds source, paths, seed and caps, but **does not bind or allowlist this child environment**. In a fake-only independent probe, a synthetic freeze key, controller key and unrelated parent secret all appeared in the executor's child environment while the adapter capture returned success. No secret values were printed. This is a concrete regression from the independently reviewed public-sample path, which excludes its review key before starting its model child.

Before live source GO, freeze an explicit minimal child environment, pass only credentials required for child preflight, and scrub them after child admission/QA and before NeMo import or model creation. Add a fake-process test that verifies the unrelated secret never enters the child and a fake model-factory test that verifies the needed keys are absent from `os.environ` at load time. Re-review the exact amended source.

## Limits outside this one-process slice

The official sample result verifier, external immutable anchor attestation, real human evidence, three stock processes and final trimmed-code/PCM parity comparison are not implemented here. Their defaults fail closed. This review does not claim they are satisfied or that an adapter capture is parity or a security finding. The process runner samples GPU and aggregate output use; its terminal cap semantics and partial-evidence seal are inherited from the separately reviewed public-sample implementation.
