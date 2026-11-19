# Independent V4 native FFmpeg source review: HOLD

19 November 2026. Reviewed commits `f6cda0c`, `99d9190`, and `fca79a5`, the current seven-role source tree, and 28 local synthetic tests. No cloud action, model import, or model inference was performed in this review.

## Decision

**HOLD** on a V4 public-sample model run. The no-model Conda/TorchCodec repair may be sound, but the current runner does not establish the claimed equality between each explicit package URL and the installed Conda metadata. A source fix, focused tests, new seven-role source hashes, independent GO review record, and fresh executable freeze/review are needed. The prior V3 GO and freeze do not authorize this changed native runtime.

## Exact source map reviewed

| Role | File | SHA-256 |
| --- | --- | --- |
| `runner` | `public_sample_smoke_v1.py` | `e36bff2d95a4995ec8ccdd9966bdd7b364d813b85dc137457b7cc4e4938ec648` |
| `tests` | `test_public_sample_smoke_v1.py` | `8db86760fd693c71bcbfaad9eadf99d063fc591a60775c57127f71b083e79b1e` |
| `handoff` | `VOICECHAT_PUBLIC_SAMPLE_SMOKE_SOURCE_HANDOFF_V1.md` | `19c2de98606ecfa09bd8f1104722e3d37f92b7789597e18cd0a204e00b3b1c45` |
| `preflight` | `../voicechat_h100_local_preflight.py` | `6233cc01b6bbd69b71aa978f3b2dabc0210bf6175faf4f34bf73eb44ead586a0` |
| `amendment` | `VOICECHAT_PUBLIC_SAMPLE_SMOKE_AMENDMENT_V1.md` | `9e4fe95f0f3828c8bedfb6d36705a5824feffcb34854371ba6450e0ddb9b8be3` |
| `concept_review` | `VOICECHAT_PUBLIC_SAMPLE_SMOKE_AMENDMENT_INDEPENDENT_CONCEPTUAL_REVIEW_GO_V1.md` | `798d2d50f04a76fcc8b7f26f0f3366121bb1d0138d0eecefc19f8a263d330aad` |
| `native_amendment` | `VOICECHAT_PUBLIC_SAMPLE_NATIVE_FFMPEG_AMENDMENT_V4.md` | `9bb24dcd3a7988bb7e3c38d1e838698418631b99459431395474012e0c26ea65` |

The runner's `SOURCE_FILES` has exactly these seven roles; the handoff names seven. It hard-pins the original amendment and shared preflight hashes. `build_conda_native_manifest_v1.py` is a separate staging helper, not one of the seven executable source roles; its outputs still need independent inspection before a freeze.

## What passed source review

`native_ffmpeg_runtime` is included in `expected_freeze_fields`; `run_sample` recomputes the full expected receipt before reserving an output directory and after ready/preload, before the child process. `scope.json` retains the first receipt. The manifest requires exact top-level keys, schema `voicechat_native_source_manifest_v1`, kind `conda_explicit`, major 7, one named explicit spec, and one archive descriptor per explicit URL. Each preserved archive is SHA-256 checked against its manifest descriptor and the URL's MD5/SHA-256 fragment; percent-encoded URL basenames are decoded for matching. Metadata files, manifest, spec, archive receipt, first `LD_LIBRARY_PATH` component, required FFmpeg 7 aliases, resolved top-level `.so` targets, TorchCodec core, and public WAV enter the frozen receipt. A top-level `.so` link that resolves outside the isolated `lib` directory is rejected. The original command remains the pinned NVIDIA public sample with only checkpoint, WAV, response JSON, and output directory; caps remain at most 1,800 seconds, 75,000 MiB sampled VRAM, and 1 GiB retained evidence. The CLI has no test-injection switch. The local CPU suite passed: `28 tests`.

## Blocking gap and reproduction

`_conda_source_receipt` checks each installed `conda-meta/<archive stem>.json` only for `metadata_value.get("sha256") == archive_sha`. It does **not** compare `fn`, `url`, package name/version/build/channel, or the exact metadata package set with the explicit URLs. The explicit-line regex also accepts an arbitrary HTTPS origin. I changed a synthetic fixture's installed metadata to `name=unrelated`, `fn=unrelated-0-h0.conda`, and `url=https://other.invalid/noarch/unrelated-0-h0.conda`, while leaving its archive SHA-256 intact. `native_ffmpeg_runtime` returned a valid receipt. Thus a freeze can attest a self-consistent archive digest while failing to establish that the installed package record corresponds to the explicit Conda source the reviewer intended. The staging helper checks source URLs when creating the manifest, but it is not an executable gate and its seven-role omission means the runner must verify the resulting artifacts itself.

Require every explicit URL to be parsed as an official `https://conda.anaconda.org/conda-forge/{linux-64,noarch}/...` package URL; compare its decoded basename and full URL against the matching `conda-meta` `fn` and `url`; validate name/version/build/channel as appropriate from the installed record; and require a one-to-one set of package JSON records and explicit lines. Keep archive SHA-256 equality and URL-fragment digest checks. Add negative tests for wrong metadata URL/filename/identity, extra metadata JSON, and non-conda-forge HTTPS URL. Rehash all seven roles after any edit.

## Native-byte boundary

The amendment says it enumerates shared-library filenames in the isolated `lib` directory. The code does this for **top-level names containing `.so`**; it does not hash every regular file recursively under `<prefix>/lib`, as the earlier [Conda fallback review](VOICECHAT_CONDA_FFMPEG7_FALLBACK_INDEPENDENT_SOURCE_REVIEW_20261119.md) requested. Nested or non-`.so` native assets can therefore change without changing this receipt. Other absolute `LD_LIBRARY_PATH` components are string bound but their contents are not byte bound. The V4 amendment acknowledges that the receipt does not prove the dynamic linker's actual selected files. Before GO, either extend the binding to the promised full library tree and actual resolved runtime dependencies, or narrow the claim and independently inspect the complete prefix and startup link map in the host freeze review. A passing no-model decode establishes only that this WAV decode path works.
