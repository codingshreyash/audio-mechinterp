# Independent audit: existing A100 VoiceChat public-sample attempt 001

19 November 2026. **PASS saved-failure integrity; HOLD model-result claim.** The official public sample did not reach the model child. The failure is a pinned Nano cache-reference format defect, not an observed VoiceChat response or a security finding.

## Saved attempt

I independently rehashed the local [attempt directory](runs/public_sample_existing_a100_attempt_001/). `freeze.json` is SHA-256 `abdddb67967f3fff32287446667f496e7022eda3fcd0d8d03068002e41be9c84`; `freeze-review.json` is `0da56403aca83a2568085e03de13e45a445267befdf08f2d50c88ec82230d6ce`. The saved scope and failure both bind the same freeze hash. The signed review claims the expected source-review hash, but the external HMAC key was intentionally not copied locally, so this saved-only audit does not independently recompute the HMAC. The runner created the scope only after its own signature validation.

`final_manifest.sha256` matches `final_manifest.json` (`f5e2aea03717affc828aacc38e60f0c6c1e19ed2533000a6599fb0eff7d4727f`). Its two entries, `scope.json` and `failure.json`, match their recorded byte counts and SHA-256 values. The terminal status is `failed_or_incomplete_environment_smoke`. `failure.json` records `ValueError` from the Nano offline metadata/class resolver at `2026-11-19T16:56:24.773124Z`, with `run: null`. There is no `admission.json` or `official_outputs/`. The failure text is truncated by the evidence cap, so the saved JSON alone does not show the entire Python traceback.

At the frozen runner source, `sample_ready_preload(...)` runs before it writes `admission.json` or calls `executor(...)`, which starts the official example. Its ready phase calls `verify_nano_offline(...)`. The observed `run: null`, absent admission, and absent official output agree with a failure at that ready check. Under this reviewed runner path, **no VoiceChat model child or inference began**. This audit does not make a claim about arbitrary unrelated processes on the host.

## Root cause, independently reproduced

The frozen preflight accepted `refs/main` by comparing `main_ref.read_text().strip()` to the pinned 40-character revision. The sample runner additionally **required** the 41-byte form: revision plus line feed. Its staging instructions also wrote that form. In pinned [`huggingface_hub` 0.34.4 source](https://github.com/huggingface/huggingface_hub/blob/v0.34.4/src/huggingface_hub/file_download.py#L1260-L1297), `try_to_load_from_cache` reads `refs/main` with `f.read()` and uses the complete result as the snapshot-directory name, without trimming it. The same untrimmed read occurs in its offline download path. Thus a 41-byte ref points to a nonexistent `<revision>\n` directory even when the six pinned files exist under `<revision>`.

I reproduced this without network or model weights in an isolated local environment with **huggingface_hub 0.34.4**: a synthetic cached `config.json` returned `None` from `try_to_load_from_cache(...)` when `refs/main` held `revision + LF`, and returned the exact cached path when the ref held the 40-byte revision. This distinguishes the cache-reference defect from missing downloaded files. It also explains why the old source and executable reviews passed static hash checks but the live offline resolver failed: those reviews did not exercise the branch-ref lookup using the actual pinned library.

Keep this attempt and its freeze immutable. A retry needs a new source version that pins a **40-byte ref**, a regression test of offline repo-ID lookup, updated source hashes and independent source review, a fresh dedicated cache/reference, a new output directory and executable freeze, and a new one-attempt run. Do not relabel this failure as a VoiceChat behavior result.
