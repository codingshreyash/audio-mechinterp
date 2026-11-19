# VoiceChat public-sample smoke independent source rereview v2

Decision: **GO for the exact public-sample-only runner source after `15c9745`; no model run is authorized by this review alone.** This supersedes the source HOLD recorded in `VOICECHAT_PUBLIC_SAMPLE_SMOKE_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md` for the four defects it identified. An external executable freeze and independent freeze review remain required before running the unchanged NVIDIA sample. Calendar fixture admission remains separate and closed.

| Exact source role | SHA-256 |
| --- | --- |
| runner, `public_sample_smoke_v1.py` | `86f827abc6eeacf95b0c48f1f9bc8026d13a86ff4896deaf9b59cbae68a9b78a` |
| tests, `test_public_sample_smoke_v1.py` | `57f8c1d04f42255c684ac998f90414ca82bd3b6bcc3e2e432fef7564a8c515b8` |
| handoff, `VOICECHAT_PUBLIC_SAMPLE_SMOKE_SOURCE_HANDOFF_V1.md` | `6c0740aea220861caaefc10732983ac880f37ce4a860e3d35c46861caef16ea6` |
| preflight, `voicechat_h100_local_preflight.py` | `874d0e7347546c9210815503a57cd0ed5a16f5e16b7862c7ba8f61951dd85757` |
| amendment, `VOICECHAT_PUBLIC_SAMPLE_SMOKE_AMENDMENT_V1.md` | `9e4fe95f0f3828c8bedfb6d36705a5824feffcb34854371ba6450e0ddb9b8be3` |
| conceptual review, `VOICECHAT_PUBLIC_SAMPLE_SMOKE_AMENDMENT_INDEPENDENT_CONCEPTUAL_REVIEW_GO_V1.md` | `798d2d50f04a76fcc8b7f26f0f3366121bb1d0138d0eecefc19f8a263d330aad` |

The six hashes above were independently recomputed with `expected_freeze_fields` and the local files. The command still supplies only the four required official script arguments; the two-phase sample-only preflight, exact NVIDIA WAV/response/source/checkpoint/Nano pins, freeze/source-GO HMAC binding, exclusive one-attempt output, and absence of calendar inputs are unchanged from the earlier review.

## Resolution of the HOLD defects

1. **Aggregate overrun:** an independent fake executor wrote two 40,000-byte artifacts under a 65,536-byte cap. The run detected overflow, wrote `failure.json` and `final_manifest.json`, recorded 85,472 observed bytes before preservation, retained 51,413 bytes, and recorded both original SHA-256 values and retained SHA-256 values. `final_manifest.sha256` matched the manifest. This is a sampled cap with bounded *retained* evidence, not a hard cap on transient bytes written between polls; the handoff now says so explicitly.
2. **Telemetry failure:** an independent fake executor raised `RuntimeError("telemetry unavailable")` after writing a partial log. The final failure manifest named both the log and `failure.json`. The separate fake-process unit test verifies an exception from the telemetry callback invokes `_kill_tree`.
3. **Review-key isolation:** with dummy review and controller keys present before the run, the frozen child environment passed to the fake executor contained neither key; the runner removes both from the parent before preflight. Freeze identity now hashes the explicit child allowlist rather than the key-bearing full parent environment. No key material was printed or saved during this review.
4. **Final inventory:** the successful manifest includes `result.json` with its exact retained SHA-256; the manifest digest file supplies the documented self-reference boundary. Failure manifests include surviving partial files and `failure.json`, with original and retained hashes or an explicit unavailable-original-hash status above the 1 GiB read ceiling.

All 16 no-model tests passed. `py_compile` and `git diff --check 15c9745^ 15c9745` passed. No model weights, public sample, GPU, or calendar fixture were run. The reviewer-authored machine binding for the exact six-role map is `VOICECHAT_PUBLIC_SAMPLE_SOURCE_REVIEW_GO_V1.json`; its future use still requires a separately authenticated external freeze reviewer and key custody. This GO does not turn sampled VRAM telemetry into a continuous peak guarantee or the sample smoke into a security finding.
