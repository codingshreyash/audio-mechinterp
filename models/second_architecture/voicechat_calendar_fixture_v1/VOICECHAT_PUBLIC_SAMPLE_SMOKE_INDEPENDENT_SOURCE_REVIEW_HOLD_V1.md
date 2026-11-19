# VoiceChat public-sample smoke independent source review v1

Decision: **HOLD for executable source GO and for any model run.** Reviewed `cd69511` against the committed sample-only amendment (`91a6d83`) and conceptual review (`cf399b0`). This review used fake dependencies and no model weights, GPU, public sample, or calendar fixture.

| Reviewed file | SHA-256 |
| --- | --- |
| `public_sample_smoke_v1.py` | `043e7e27365a8e4af559785429b10d685b74365808a6b07bce2a142789b801a8` |
| `test_public_sample_smoke_v1.py` | `7b9f81e289853ea2a105859093ad62d5fefd937a345331e5e82334993db4a05d` |
| `VOICECHAT_PUBLIC_SAMPLE_SMOKE_SOURCE_HANDOFF_V1.md` | `96e2fa7cf118ae32fd820d4bb2353ff625c0751c039191cc3dcfe49df89bfc17` |

## Checks that passed

- The runner does not import NeMo. Its separate `ready` then `preload` path invokes the shared pinned source, checkpoint, Nano cache, package, runtime/GPU, and offline-environment checks and adds exact public WAV/response hashes and Nano `refs/main` bytes. It omits fixture QA/anchor/grants only for this public sample path. The existing fixture path is unchanged.
- The fixed subprocess command supplies exactly the four required official-script arguments and leaves template, tools, system message, sample ID, and device at pinned script defaults. Source hashes cover the official script/helper/template/model code; preflight checks the full VoiceChat weight and six Nano metadata files. The expected freeze includes runner/tests/handoff/preflight/amendment/conceptual-review hashes, command, environment, identities, paths, and caps.
- The output directory is created exclusively before preflight, consuming the run ID on failure. The production CLI has no fake-dependency switch. It checks an exact-source GO JSON record, the freeze hash, and an HMAC over the independent freeze review before making the directory. Independent authorship/key custody is a procedural external requirement; the HMAC verifies possession of the provided key, not reviewer independence on its own.
- The child is a fresh `Popen` process in a new session. The loop polls elapsed time, process-tree GPU memory via `nvidia-smi`, and aggregate output size, streams bounded stdout/stderr, and kills the process group on a detected breach. This is sampled monitoring, not a measured instantaneous GPU peak or a hard aggregate filesystem quota.
- All 11 no-model tests passed. `py_compile` and `git diff --check cd69511^ cd69511` passed.

## Blocking evidence defects

1. **A cap failure can lack a saved failure record.** In a fake-executor probe using an 8,192-byte cap, one 9,000-byte output caused `evidence_inventory` to raise `Total sample evidence cap exceeded`. `run_sample` attempted to write `failure.json`, but `_write_evidence` rejected the write and the exception was suppressed. Neither `failure.json` nor `result.json` existed afterward. This violates the amendment's requirement to record any failure/timeout and retain independently interpretable partial evidence.
2. **The aggregate output cap is detected after an overrun.** A fake subprocess wrote two 3,000-byte files before the next monitor poll under a 4,096-byte total cap. `execute_official` did detect and call `_kill_tree`, but 6,000 bytes remained. `RLIMIT_FSIZE` only limits each child file. State this as a sampled soft total cap or implement a hard aggregate quota before claiming the total evidence cap is enforced. At minimum, reserve a bounded failure/stop record and report the observed overrun explicitly; the executable freeze must use the reviewed semantics.
3. **The final evidence inventory is incomplete.** Success builds `artifact_sha256` before writing `result.json`, so the result does not account for that generated file. Failure records do not include a hash/size inventory of surviving partial files. The amendment requires all generated files and their SHA-256/size, including failure evidence. A separate final manifest or verifiable post-run inventory is needed without recursively hashing itself.

Re-review exact amended source and tests after those fixes. This HOLD document is not a `voicechat_public_sample_source_review_v1` GO JSON record and cannot satisfy the runner's source-review gate.
