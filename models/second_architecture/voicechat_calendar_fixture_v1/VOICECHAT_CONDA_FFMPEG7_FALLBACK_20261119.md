# VoiceChat FFmpeg 7 task-local fallback (source only)

19 November 2026. **HOLD on another model run.** This amends the [isolated FFmpeg repair plan](VOICECHAT_ISOLATED_FFMPEG_REPAIR_PLAN_20261119.md) after the Ubuntu package route proved too broad. The A100 was stopped without installing Ubuntu packages globally. The extraction starter set exposed more than 40 missing transitive SONAMEs, and a simulated `apt-get install` proposed 93 new packages plus 15 upgrades, including `libstdc++6`. Do not continue that route for this sample. This document does not mutate the host or run VoiceChat.

## Why this candidate

TorchCodec 0.10.0 supports FFmpeg majors 4–8 and needs shared FFmpeg libraries for audio decoding ([tagged TorchCodec instructions](https://github.com/meta-pytorch/torchcodec/blob/v0.10.0/README.md#installing-torchcodec)). The saved attempt-003 [stderr](runs/public_sample_existing_a100_attempt_003/voicechat-sample-output-003/stderr.log) tried `libtorchcodec_core7.so` and lacked `libavutil.so.59`. [Conda-forge publishes FFmpeg 7.1.1 for linux-64](https://anaconda.org/conda-forge/ffmpeg/files?version=7.1.1). An isolated conda prefix lets its solver place FFmpeg and transitive libraries together without touching Ubuntu's package database or the existing Python venv. This is a candidate until the exact `torchaudio.load` path decodes the public WAV.

Use **micromamba 2.9.0-0**, official `linux-64` release binary, from [`mamba-org/micromamba-releases`](https://github.com/mamba-org/micromamba-releases/releases/tag/2.9.0-0). Its published SHA-256 is `366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3`. The [micromamba manual](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html) supports creating a separate root prefix and environment; its [explicit-spec guide](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html#explicit-spec-files) describes checksum-bearing package locks. Do not use `/latest`, `curl | sh`, shell initialization, or an existing conda base environment.

## Task-local bootstrap and solve

The only target environment is `/home/ubuntu/voicechat-sample-ffmpeg-conda7`. Keep micromamba itself and its package cache under `/home/ubuntu/voicechat-sample-micromamba-2.9.0-0`. This creates a new process environment for the official Python venv; it does not replace that venv. Verify Ubuntu `x86_64` and the preexisting disk-space gates first.

```bash
set -euo pipefail
TASK_MM_ROOT=/home/ubuntu/voicechat-sample-micromamba-2.9.0-0
TASK_MM_BIN="$TASK_MM_ROOT/micromamba-linux-64"
TASK_MM_SHA=366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3
TASK_FFMPEG_PREFIX=/home/ubuntu/voicechat-sample-ffmpeg-conda7
test "$(uname -s)" = Linux && test "$(uname -m)" = x86_64
test ! -e "$TASK_FFMPEG_PREFIX"
mkdir -p "$TASK_MM_ROOT"
curl -fL --retry 2 --output "$TASK_MM_BIN" \
  https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64
printf '%s  %s\n' "$TASK_MM_SHA" "$TASK_MM_BIN" | sha256sum -c -
chmod 755 "$TASK_MM_BIN"
"$TASK_MM_BIN" --version
export MAMBA_ROOT_PREFIX="$TASK_MM_ROOT/root"
"$TASK_MM_BIN" create --prefix "$TASK_FFMPEG_PREFIX" \
  --override-channels --channel conda-forge 'ffmpeg=7.1.1' --yes
```

Treat the solver result as a **staging result**, not a reusable pin. Inspect that all package URLs resolve to the official conda-forge channel, that the FFmpeg record is `7.1.1` on `linux-64`, and that the prefix contains `libavutil.so.59`, `libavcodec.so.61`, and `libavformat.so.61` (inspect actual SONAMEs rather than assuming their presence). Do not run `micromamba install` against the Python venv, use `sudo`, or change global loader configuration. The conda package may include other C++ runtime libraries; loading the whole prefix can affect process-local resolution. Capture `ldd`/loader evidence for TorchCodec, Torch, and CUDA dependencies, and stop if a core library resolves to an unexpected or incompatible file. Avoid silently substituting FFmpeg 6 or another 7 build.

## Lock and native-byte receipt

Before model work, create an explicit package lock and `/home/ubuntu/voicechat-sample-ffmpeg-conda7/native-source-manifest.json` from the actual prefix. The manifest is a task-local source receipt; it must be hashed by, and validated independently from, the reviewed runner. Include:

1. Schema/version, creation UTC, absolute prefix and micromamba root, `linux-64` architecture, micromamba release URL and binary SHA-256.
2. Every `conda-meta/*.json` package record: name, version, build string, channel, source URL, package filename, recorded SHA-256 (or SHA-256 of the matching downloaded archive if metadata lacks it), and MD5 if supplied. Reject records without an independently verified archive SHA-256. Preserve the downloaded package archives or cache with those hashes.
3. A deterministic `@EXPLICIT` conda package lock with full URLs and checksums. Hash the lock file and the sorted package inventory. Keep the solver JSON/log and micromamba version. The explicit lock is for a **fresh-prefix replay**, not an invitation to resolve later against changing repodata. The [conda explicit-spec documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) describes exact package URLs and checksums.
4. Relative path, file type, symlink target, byte length, and SHA-256 for every regular file under `<prefix>/lib` and the resolved target of every `<prefix>/lib/libav*.so*` symlink. Reject symlinks escaping the prefix. Include the checksum of `libtorchcodec_core7.so`, the public WAV, and the preexisting Python inventory receipt. Exclude the manifest itself from its content hash to avoid a circular digest.
5. Exact `LD_LIBRARY_PATH` with `<prefix>/lib` as its **first** component, preserving any prior suffix; `PATH` only if needed for a documented child tool. Record actual ELF resolution (`readelf -d`, `ldd`, and if needed `LD_DEBUG=libs`) and all no-model smoke outputs.

Hash the canonical JSON manifest file itself and put its path and SHA-256 in the new reviewed freeze. The runner must independently recompute the hashes of the package archives and prefix library tree, compare against the manifest and freeze at launch, and reject changed/missing/unexpected files or environment values. Merely hashing the manifest file is insufficient. The V3 runner already freezes the `LD_LIBRARY_PATH` string and sorted `pip freeze --all`, but does **not** freeze native libraries; add a `native_ffmpeg_runtime` field to `expected_freeze_fields`, its freeze validation, and final evidence. Bind the exact prefix and `<prefix>/native-source-manifest.json` path, not an ambient `CONDA_PREFIX`. Recompute after the no-model check and immediately before the official sample child. No reviewed freeze from attempt 003 applies to this altered runtime.

## No-model acceptance gate

Run in a new process with the isolated library path set **before** Python starts. The `torchaudio.load` call is the same decode path that failed inside the official child. The `AudioDecoder` call checks TorchCodec directly. Neither loads VoiceChat.

```bash
set -euo pipefail
TASK_FFMPEG_PREFIX=/home/ubuntu/voicechat-sample-ffmpeg-conda7
TASK_SAMPLE=/home/ubuntu/voicechat-sample-Speech/examples/speechlm2/sample_audio/sample_fc.wav
export LD_LIBRARY_PATH="$TASK_FFMPEG_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
sha256sum "$TASK_SAMPLE"
ldd /home/ubuntu/voicechat-sample-py312/lib/python3.12/site-packages/torchcodec/libtorchcodec_core7.so
/home/ubuntu/voicechat-sample-py312/bin/python -B -c 'import sys, torchaudio; from torchcodec.decoders import AudioDecoder; p=sys.argv[1]; wave,sr=torchaudio.load(p); assert wave.ndim == 2 and wave.numel() > 0 and sr > 0; decoded=AudioDecoder(p).get_all_samples(); assert decoded.data.numel() > 0; print({"torchaudio_shape":tuple(wave.shape),"sample_rate":sr,"torchcodec_shape":tuple(decoded.data.shape)})' "$TASK_SAMPLE"
```

Retain stdout, stderr, exit status, library-resolution evidence, package lock, archive hashes, prefix-library hashes, and input SHA-256 under a new bounded no-model output. Failure in either decoder, an unresolved SONAME, unexpected core-library resolution, or package/archive mismatch is a stop. A successful decode only establishes that this native audio dependency works; it does not establish that VoiceChat inference or the calendar experiment works.

After a clean no-model gate, obtain fresh independent six-role source review for the runner change, create and verify a new host freeze/review for the exact manifest and environment, and use `voicechat-sample-output-004` or a later fresh output directory for one official **public sample only**. Retain the prior time, VRAM, and evidence caps. Stop the A100 promptly after the run or any gate failure.
