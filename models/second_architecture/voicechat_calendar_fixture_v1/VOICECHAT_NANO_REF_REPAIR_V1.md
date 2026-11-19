# VoiceChat sample Nano cache ref repair

19 November 2026. **Source repair only. No new model inference. HOLD any rerun until a new independent six-role source review and a new host-specific executable freeze pass.** The first existing-A100 sample attempt sealed `failed_or_incomplete_environment_smoke` before model import. Its saved `failure.json` reports that `AutoConfig.from_pretrained` could not find the Nano repository config offline, despite the six pinned metadata files being present.

The cache had `refs/main` containing the 40-character commit **plus LF**. The pinned `huggingface-hub` 0.34.4 `try_to_load_from_cache` reads that ref file verbatim and uses the entire string as the snapshot directory name. A directory named by the 40-character commit exists; a directory named by the commit plus LF does not. On the prepared A100, `try_to_load_from_cache(..., revision="main")` returned `None` while `revision=<40-character commit>` returned the cached `config.json`. The old `check_nano_cache` used `.strip()` and therefore admitted the unusable ref; the sample runner additionally required LF. This was a setup gate defect, not an audio model observation.

The repaired shared preflight now requires the exact 40 ASCII bytes with no newline. Its self-test rejects a wrong commit and the formerly accepted newline. The sample runner's exact ref hash, length, frozen identity, and source preflight hash are updated together. Staging instructions write `ref.write_text(revision)`; the checkpoint and six Nano metadata pins, offline-only policy, official sample, resource caps, and calendar HOLD are unchanged. The original source GO and signed failed-attempt freeze remain historical evidence and must not be reused for a repaired run.

Verification: shared no-model preflight self-test PASS. Sample runner no-model suite **17/17 PASS** under an isolated environment with `huggingface-hub==0.34.4`; its new test confirms that the real library resolves `config.json` for a 40-byte `refs/main` and returns no cached file for the same ref with LF. No network call or model load occurs in that test. Running the suite without Hub installed yields 16 PASS and one explicit skip. `git diff --check` passes.

| Source role | Repaired SHA-256 |
| --- | --- |
| Shared preflight | `6233cc01b6bbd69b71aa978f3b2dabc0210bf6175faf4f34bf73eb44ead586a0` |
| Sample runner | `5a699e544f94767ef40d0e82be4fc1e08d905c9bf88a7d8946a0faab086c606e` |
| Sample tests | `9173afff61df3d4a187d31b983e738e0cc81a7f3a4ff218c96a3b426822da7ad` |

Before another sample attempt, stage only the repaired source on the stopped A100, change only the dedicated task-created Nano cache ref to the exact 40-byte commit, recheck the unchanged weight and package pins, and create a **new** output directory. A separate reviewer must bind the repaired six-role hash map and actual host state in a fresh freeze review. The failed run stays sealed.
