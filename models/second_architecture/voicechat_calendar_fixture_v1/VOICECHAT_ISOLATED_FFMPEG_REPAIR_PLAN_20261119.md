# VoiceChat public-sample FFmpeg repair plan (source only)

19 November 2026. **HOLD on another model run.** This document proposes a narrowly isolated native-library repair for the official public sample. It does not change the A100, install software, or authorize calendar-audio work.

## Evidence and diagnosis

Attempt 003 entered the official VoiceChat child, built the model, and reached `load_wav_16k_mono` for `examples/speechlm2/sample_audio/sample_fc.wav`. Its saved [stderr](runs/public_sample_existing_a100_attempt_003/voicechat-sample-output-003/stderr.log) then shows `torchaudio.load` importing TorchCodec and failing to load every candidate `libtorchcodec_core{8,7,6,5,4}.so`: each reports a missing `libavutil.so.{60,59,58,57,56}`. No generated audio was retained. The host had `torch==2.10.0+cu128`, `torchaudio==2.10.0`, and `torchcodec==0.10.0`; the saved `apt-cache policy` showed no installed FFmpeg libraries and an Ubuntu 22.04 candidate at `7:4.4.2-0ubuntu0.22.04.1`. The missing FFmpeg shared libraries are the first demonstrated blocker. A later ABI or codec error remains possible.

The [TorchCodec 0.10.0 installation instructions](https://github.com/meta-pytorch/torchcodec/blob/v0.10.0/README.md#installing-torchcodec) support FFmpeg major versions 4–8 and require separate shared libraries for media decoding. [Ubuntu's Jammy package index](https://packages.ubuntu.com/jammy-updates/libavutil56) identifies `libavutil56` version `7:4.4.2-0ubuntu0.22.04.1` as FFmpeg 4.4.2. The `pydub` executable warning in stderr is separate: TorchCodec needs the libraries, and the FFmpeg command-line program is not required for the failing `torchaudio.load` path.

## Preferred isolated route

Stage Ubuntu Jammy `amd64` FFmpeg 4.4.2 **runtime library packages only** under `/home/ubuntu/voicechat-sample-ffmpeg4-jammy`, leaving system packages and the Python venv unchanged. Use the host's configured Ubuntu apt indexes, verify `apt-cache policy`, then download exact package versions without `sudo` and extract them with `dpkg-deb -x`. Start with `libavutil56`, `libavcodec58`, `libavformat58`, `libswresample3`, and `libswscale5`; inspect `readelf -d` on `libtorchcodec_core4.so` and `ldd` on extracted libraries for additional needed Ubuntu packages. Download any additional packages at recorded exact versions, and stop if dependencies would require replacing system glibc, libstdc++, CUDA, or PyTorch libraries. No global `apt install` is in this plan. The package set is a **candidate**, not a claim that five archives suffice.

Example staging commands, to run only after the host and source gates are reviewed:

```bash
set -euo pipefail
TASK_FFMPEG_ROOT=/home/ubuntu/voicechat-sample-ffmpeg4-jammy
TASK_FFMPEG_DEBS="$TASK_FFMPEG_ROOT/debs"
TASK_FFMPEG_EXTRACT="$TASK_FFMPEG_ROOT/extracted"
TASK_FFMPEG_VER='7:4.4.2-0ubuntu0.22.04.1'
mkdir -p "$TASK_FFMPEG_DEBS" "$TASK_FFMPEG_EXTRACT"
cd "$TASK_FFMPEG_DEBS"
for pkg in libavutil56 libavcodec58 libavformat58 libswresample3 libswscale5; do
  apt download "$pkg=$TASK_FFMPEG_VER"
done
sha256sum ./*.deb > ../deb_sha256.txt
for archive in ./*.deb; do dpkg-deb -x "$archive" "$TASK_FFMPEG_EXTRACT"; done
TASK_FFMPEG_LIB="$TASK_FFMPEG_EXTRACT/usr/lib/x86_64-linux-gnu"
readelf -d /home/ubuntu/voicechat-sample-py312/lib/python3.12/site-packages/torchcodec/libtorchcodec_core4.so > ../torchcodec_core4_dynamic.txt
LD_LIBRARY_PATH="$TASK_FFMPEG_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd /home/ubuntu/voicechat-sample-py312/lib/python3.12/site-packages/torchcodec/libtorchcodec_core4.so > ../torchcodec_core4_ldd.txt
```

`apt download` is read/write only in this task directory; it does not install the packages. Inspect `.deb` package metadata and the package hashes before extraction. Never trust a package just because its filename matches. Reject mixed distributions or architecture and unresolved `not found` lines. If needed, extend the archive set with exact-version Ubuntu runtime dependencies, document why each was needed, and rerun linkage checks. Launch a *new process* with `LD_LIBRARY_PATH` set; changing it inside an already running Python process is insufficient for a reliable loader test. Keep the preexisting `LD_LIBRARY_PATH` suffix for CUDA libraries, and freeze the resulting full string.

An isolated conda prefix is a fallback if Ubuntu archive extraction cannot satisfy the linkage cleanly. TorchCodec's [installation instructions](https://github.com/meta-pytorch/torchcodec/blob/v0.10.0/README.md#installing-torchcodec) suggest conda FFmpeg, and the project's [Linux CUDA workflow](https://github.com/meta-pytorch/torchcodec/blob/main/.github/workflows/linux_cuda_wheel.yaml) exposes a conda library directory through `LD_LIBRARY_PATH`. A fallback requires a separately pinned package manifest, archive hashes, prefix library hashes, and a new review. Do not silently switch versions or install into the existing venv.

## No-model acceptance gate

Before any model load, run the **exact failing decode path** in a fresh Python process against the pinned public sample, plus a direct TorchCodec decode. Use `-B` and no network; retain stdout, stderr, exit code, sample SHA-256, `readelf`/`ldd`, package manifest, and resolved-library hashes. For example:

```bash
TASK_FFMPEG_LIB=/home/ubuntu/voicechat-sample-ffmpeg4-jammy/extracted/usr/lib/x86_64-linux-gnu
TASK_SAMPLE=/home/ubuntu/voicechat-sample-Speech/examples/speechlm2/sample_audio/sample_fc.wav
export LD_LIBRARY_PATH="$TASK_FFMPEG_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
sha256sum "$TASK_SAMPLE"
/home/ubuntu/voicechat-sample-py312/bin/python -B -c 'import sys, torchaudio; from torchcodec.decoders import AudioDecoder; p=sys.argv[1]; wave,sr=torchaudio.load(p); assert wave.ndim == 2 and wave.numel() > 0 and sr > 0; decoder=AudioDecoder(p); samples=decoder.get_all_samples(); assert samples.data.numel() > 0; print({"torchaudio_shape":tuple(wave.shape),"sample_rate":sr,"torchcodec_shape":tuple(samples.data.shape)})' "$TASK_SAMPLE"
```

The direct `AudioDecoder` check is an additional smoke test; `torchaudio.load` is the decisive check because the official child calls it. If either fails, record the new traceback and stop. Do not interpret a successful decode as model success or as a security finding. An `ldd` success alone is insufficient: symbol, codec, and ABI issues can appear at import/decode time.

## Freeze and review needed before another model attempt

The reviewed V3 runner currently binds the *string* value of `LD_LIBRARY_PATH` in `child_environment` and binds the sorted `pip freeze --all` digest. Neither binds the bytes of native `.so` files outside the Python environment. Amend `expected_freeze_fields` to include a `native_ffmpeg_runtime` receipt with: OS/architecture, exact Ubuntu package names and versions, SHA-256 of every downloaded `.deb`, SHA-256 of each resolved `libav*.so.*` target loaded from the isolated prefix, actual absolute `LD_LIBRARY_PATH`, `libtorchcodec_core4.so` SHA-256, and the public sample's SHA-256. Reject missing, symlink-escaped, changed, or unexpected files; recompute this receipt at launch and after the no-model decode. The receipt must enter the freeze comparison and final output evidence, not just an unvalidated side note. Keep the full Python inventory check.

This changes the runner source and freeze. Obtain a new independent source review under the existing six-role gate, generate a new host freeze/review for the staged native runtime, and use a fresh output directory (`voicechat-sample-output-004` or later). The earlier V3 GO and attempt 003 evidence are historical; they do not authorize an altered environment. Keep the existing 1,800-second, 75,000-MiB VRAM, and 1-GiB evidence caps. Stop the instance promptly after the authorized sample or a gate failure.

Success for this repair is limited to a clean **no-model public WAV decode** with frozen native-library bytes. Only then is a fourth official public-sample attempt justified. A successful sample would establish that VoiceChat can run on this host; it would still not establish any calendar behavior or mechanistic finding.
