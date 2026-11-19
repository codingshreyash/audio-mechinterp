# H200 VoiceChat staging: source-only disk and environment review

19 November 2026. **Source-only, no model-run GO.** Brev reports `voicechat-public-sample-h200-v1` build **COMPLETED**. Its Ubuntu 24.04.5 guest has one H200 with 143,771 MiB and one `/dev/vda1` root filesystem: 123,690,532,864 bytes total, 100,431,302,656 bytes available before setup, no second volume. A requested `--min-disk 200` did not deliver a 200 GB filesystem; see the earlier [Brev disk investigation](VOICECHAT_PUBLIC_SAMPLE_H100_DISK_FAILURE_AMENDMENT_20261119.md). The prospective [H200 V5 sample amendment](VOICECHAT_PUBLIC_SAMPLE_H200_AMENDMENT_V5.md) now uses an H200-only **30 GiB post-stage** floor. The shared H100/A100 60 GiB gate stays unchanged. Staging is feasible only with the measured single-copy budget below and per-step disk checks. The amended runner still requires fresh independent source and executable-freeze GO before inference. This review changes no cloud host and runs no model.

## Hard storage gate

The pinned `model.safetensors` is **44,382,749,892 bytes** (41.33 GiB), SHA-256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`, at VoiceChat revision `443794ea956ef0065f001967ffd00e77f519cb39`. The *new H200 sample runner only* requires **30 GiB = 32,212,254,720 bytes available after staging**. This makes the strict minimum before the one weight blob 76,595,004,612 bytes, plus all environment/source/native/cache/output costs. With the measured 100,431,302,656 bytes available, at most **23,836,298,044 bytes** may be spent on those other items before the floor is breached.

The exact A100 `du -sb` reference sizes are venv **12,439,787,625**, NeMo tree **273,116,132**, six-file Nano cache **34,717,302**, isolated FFmpeg prefix **747,788,803**, and HF checkpoint cache **44,382,810,676** bytes. Their sum is **57,878,220,538 bytes**; the H200 projection is therefore **42,553,082,118 bytes free**, **10,340,827,398 bytes** over the H200 floor. These are measured A100 sizes and a *projection*, not H200 measurements. The [V5 amendment](VOICECHAT_PUBLIC_SAMPLE_H200_AMENDMENT_V5.md) instead counts the raw weight blob, 44,382,749,892 bytes, yielding its valid raw-blob projection of 42,553,142,902 bytes free; the measured HF cache adds 60,784 bytes of metadata. Keep the one root disk, one checkpoint blob, a cache snapshot symlink, no `local_dir` copy, no persistent pip cache/wheelhouse, and only bounded output. The H100/A100 single-root 160 GiB and 60 GiB gates in the older [sample packet](VOICECHAT_PUBLIC_SAMPLE_H100_EXECUTION_PACKET_20261119.md) are historical policies for those GPU paths; they do not apply to the new H200-only sample gate.

Measure `df -B1 /` and `du -sb` after **each** Python, package, NeMo, FFmpeg, Nano, and checkpoint stage. Before any planned allocation of `n` bytes, require `available - n >= 32,212,254,720` with an explicit allowance for temporary downloads. The strict runtime admission tests the actual post-stage available bytes. If it falls below 30 GiB, stop; do not delete unrelated data or bypass the runner. The 10.34 GB projected margin must cover all extra downloads, scratch, log output and up to 1 GiB of retained run evidence.

Read-only capacity commands on the completed host:

```bash
brev ls --json
brev exec voicechat-public-sample-h200-v1 'date -u; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits; lsblk -b -o NAME,SIZE,FSTYPE,MOUNTPOINTS; findmnt -T /home/ubuntu; df -B1 /; df -B1 /data 2>/dev/null || true'
```

Record the exact `df -B1` available bytes, `findmnt` device and filesystem for the *actual checkpoint parent*, not just the provider's advertised disk. The delivered `/dev/vda1` presently qualifies for tightly bounded staging under the H200-only amendment. It does not qualify under the older H100/A100 pre-stage rule.

A read-only H200 single-root calculation is:

```bash
python3 - <<'PY'
import os, shutil
free = shutil.disk_usage('/').free
weight = 44_382_749_892
floor = 30 * 1024**3
other_a100 = 12_439_787_625 + 273_116_132 + 34_717_302 + 747_788_803
print({'root_free_bytes': free, 'minimum_preweight_bytes': weight + floor,
       'maximum_other_bytes': free - weight - floor,
       'projected_free_after_a100_sized_stage': free - 44_382_810_676 - other_a100,
       'h200_poststage_floor_bytes': floor})
PY
```

## Task-local Python 3.12 bootstrap, proposed commands only

The A100's saved `stderr.log` and pip dry-run environment identify uv-managed CPython **3.12.16**. A clean Ubuntu 24.04 guest can have `python3.12` while lacking `pip` or `python3.12-venv`, so do not rely on system `ensurepip` or modify the system interpreter. Use the standalone uv **0.13.9** Linux x86-64 archive whose [upstream release](https://github.com/astral-sh/uv/releases/tag/0.13.9) reports SHA-256 `a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5` and size 17,068,176 bytes. [uv's Python guide](https://docs.astral.sh/uv/guides/install-python/) and [CLI reference](https://docs.astral.sh/uv/reference/cli/) document managed Python, `--install-dir`, and `uv venv --seed`. These are reviewable *candidate commands*, not evidence that the H200 installation has run:

```bash
set -euo pipefail
TASK=/home/ubuntu/voicechat-sample-stage
UV_ARCHIVE="$TASK/uv-x86_64-unknown-linux-gnu-0.13.9.tar.gz"
UV_BIN="$TASK/bin/uv"
VENV=/home/ubuntu/voicechat-sample-py312
mkdir -p "$TASK/bin" "$TASK/tmp" "$TASK/python" "$TASK/uv-cache" "$TASK/receipts"
curl --fail --location --retry 3 \
  https://github.com/astral-sh/uv/releases/download/0.13.9/uv-x86_64-unknown-linux-gnu.tar.gz \
  --output "$UV_ARCHIVE"
printf '%s  %s\n' a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5 "$UV_ARCHIVE" | sha256sum --check
tar -xzf "$UV_ARCHIVE" -C "$TASK/bin" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
"$UV_BIN" --version | tee "$TASK/receipts/uv-version.txt"
export UV_PYTHON_INSTALL_DIR="$TASK/python" UV_CACHE_DIR="$TASK/uv-cache" TMPDIR="$TASK/tmp"
"$UV_BIN" python install 3.12.16 --install-dir "$UV_PYTHON_INSTALL_DIR"
"$UV_BIN" venv --seed --managed-python --python 3.12.16 "$VENV"
"$VENV/bin/python" --version | tee "$TASK/receipts/python-version.txt"
"$VENV/bin/python" -m pip install --no-cache-dir --report "$TASK/receipts/seed-pins.json" \
  pip==26.3 setuptools==84.2.1 wheel==0.48.0
"$VENV/bin/python" -m pip --version | tee "$TASK/receipts/pip-version.txt"
readlink -f "$VENV/bin/python" | tee "$TASK/receipts/python-resolved-path.txt"
sha256sum "$UV_BIN" "$(readlink -f "$VENV/bin/python")" > "$TASK/receipts/bootstrap-executable-hashes.txt"
df -B1 / | tee "$TASK/receipts/df-after-python.txt"
rm -f -- "$UV_ARCHIVE"
```

The standalone archive was independently downloaded and checked locally: 17,068,176 bytes, matching SHA-256, with members `uv-x86_64-unknown-linux-gnu/{uv,uvx}`. The uv-managed distribution lives in the task directory and is pinned by exact CPython version; preserve its resolved path and bytes in the H200 environment receipt. `uv venv --seed` is used only to provide pip inside the new venv; all later installs use that venv's Python. The seed pins above match the saved A100 inventory. A 430-line `pip freeze` is **not** a complete wheel lock; H200 must record each resolved distribution URL/hash in `pip --report` and compare every installed version to the A100 receipt before a freeze.

For the CUDA and source stages, use the venv Python, task-local temp, no persistent pip cache, an install report per transaction, and a disk check after each transaction. The CUDA index and exact versions are from [PyTorch's published 2.10 instructions](https://pytorch.org/get-started/previous-versions/):

```bash
"$VENV/bin/python" -m pip install --no-cache-dir --report "$TASK/receipts/pytorch-cu128-install.json" \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
df -B1 / | tee "$TASK/receipts/df-after-torch.txt"
```

For a full version-pinned replay, first transfer the saved 430-line A100 inventory into `"$TASK/a100-freeze-after-mamba.txt"` and verify its **file** SHA-256 `0cf5115ff69b9b5a39efed85049bb19a015016660d5e39b3241f17401d76bc7d`. That is separate from the runner's normalized inventory digest. Make an install list from it by replacing only the machine-local PyArrow URL with `pyarrow==25.0.1` and deferring the two CUDA source builds and local NeMo package. The existing PyArrow wheel had SHA-256 `5389cdf79447ed1515c9e31620e6e1e2302249564d603f2ad727d4f6d313e4c3`; prefer that exact `cp312`/manylinux wheel when its official URL and hash can be verified, otherwise record a new wheel hash and review the difference. The [Mamba closure plan](VOICECHAT_MAMBA_DEPENDENCY_CLOSURE_SOURCE_PLAN_20261119.md) explains why all eight declared dependencies, already present in the A100 inventory, must remain. Proposed package replay:

```bash
A100_INV="$TASK/a100-freeze-after-mamba.txt"
printf '%s  %s\n' 0cf5115ff69b9b5a39efed85049bb19a015016660d5e39b3241f17401d76bc7d "$A100_INV" | sha256sum --check
"$VENV/bin/python" - "$A100_INV" "$TASK/requirements-from-a100.txt" <<'PY'
import pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:])
lines = src.read_text().splitlines()
assert len(lines) == 430
out = []
for line in lines:
    name = line.split("==", 1)[0].lower().replace("_", "-")
    if line.startswith("pyarrow @ file:///home/ubuntu/voicechat-sample-wheelhouse/"):
        out.append("pyarrow==25.0.1")
    elif name in {"nemo-toolkit", "mamba-ssm", "causal-conv1d"}:
        continue
    else:
        assert "==" in line and " @ " not in line, line
        out.append(line)
dst.write_text("\n".join(out) + "\n")
PY
"$VENV/bin/python" -m pip install --dry-run --no-cache-dir \
  --report "$TASK/receipts/a100-pins-dry-run.json" -r "$TASK/requirements-from-a100.txt"
# Inspect the dry-run report and current df. Stop if any core pin or wheel tag changes.
"$VENV/bin/python" -m pip install --no-cache-dir \
  --report "$TASK/receipts/a100-pins-install.json" -r "$TASK/requirements-from-a100.txt"
df -B1 / | tee "$TASK/receipts/df-after-base-pins.txt"
git clone https://github.com/NVIDIA-NeMo/Speech.git /home/ubuntu/voicechat-sample-Speech
git -C /home/ubuntu/voicechat-sample-Speech checkout --detach 097dfe9e2f55baf653b83035868bdc89849f1b47
"$VENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps \
  --report "$TASK/receipts/nemo-local-source-install.json" \
  -e '/home/ubuntu/voicechat-sample-Speech[all]'
"$VENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps \
  --report "$TASK/receipts/mamba-source-install.json" \
  causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1
"$VENV/bin/python" -m pip check | tee "$TASK/receipts/pip-check.txt"
"$VENV/bin/python" -m pip freeze --all | tee "$TASK/receipts/h200-freeze.txt"
df -B1 / | tee "$TASK/receipts/df-after-python-packages.txt"
```

This replay is still a **candidate** because `pip freeze` pins versions, not all artifacts or build options. In particular, verify PyTorch remains `2.10.0+cu128`, TorchCodec's FFmpeg linkage, NeMo's installed source commit, compiled Mamba/Causal extension imports, NumPy/PyArrow/Datasets imports, and all 11 VoiceChat card pins. Compare the H200 `pip freeze` package names/versions to the A100 inventory, allowing the expected local NeMo source form. A resolver error, incompatible Ubuntu 24 wheel, changed CUDA artifact, package drift, failed `pip check`, or a projected/actual disk breach is a **HOLD**, not a reason to loosen a pin. Save every report before deleting its task-owned temporary download files.

There is one task-owned uv cache and one task-owned temp directory. After every package/source install is complete and its install report and environment validation are saved, remove **only verified transient contents** under those two task-owned paths; do not prune unrelated machine caches. Do not retain a copied wheelhouse or a second model checkout. The managed interpreter and venv must remain. Use `df -B1 /`, not `du` estimates alone, as the admission fact. `pip --no-cache-dir` prevents the usual persistent wheel cache, but downloads/builds still occupy temporary space while installing; stop before a step whose worst-case temporary allocation could cross the floor. The final package receipt, `pip check`, `pip freeze --all`, native loader test, and full checkpoint hash must be reviewed before any sample child.

## Fastest reproducible staging after a disk pass

1. **Recreate, do not transplant, the Python 3.12 environment.** The A100 used uv-managed **CPython 3.12.16**, as its saved traceback and pip dry-run record show. Its 430-line `pip freeze --all` inventory has sorted digest `f6993856420dd48ddc370b8c7b7c0fe26fe76bf5cf0843fcdb190dc4568ed5df`, but it is an installed-version receipt, not a wheel lock. It includes a `pyarrow @ file:///home/ubuntu/voicechat-sample-wheelhouse/...` entry whose local path may not exist on H200. Python's [venv documentation](https://docs.python.org/3/library/venv.html) says virtual environments are not movable/copyable; the A100 was Ubuntu 22.04 and H200 reports 24.04. Copying its venv may preserve absolute shebang/base-interpreter paths and native-library assumptions. The task-local uv/CPython bootstrap above avoids any system package change and records the new H200 wheel inventory. Do not retain a 12 GB wheelhouse on this disk. A direct venv copy can only be an experimental diagnostic after independent review, never the staging baseline.
2. **Install the card and repaired runtime pins.** The A100 receipt pinned `torch==2.10.0+cu128`, `torchaudio==2.10.0`, `torchvision==0.25.0`, `torchcodec==0.10.0`, `numpy==1.26.4`, `pyarrow==25.0.1`, `datasets==5.0.1`, `mamba-ssm==2.3.2.post1`, `causal-conv1d==1.6.2.post1`, and the other [11 card pins](../voicechat_h100_local_preflight.py). [PyTorch publishes the 2.10 CUDA 12.8 wheel index](https://pytorch.org/get-started/previous-versions/); select it explicitly, and keep the whole 430-line A100 inventory as a comparison target rather than blindly using it as an install command. Use `pip --no-cache-dir` for every package transaction; place `TMPDIR`, uv data and HF/Xet caches under task-owned paths on the measured root. Record package provenance, `pip check`, exact versions, `pip freeze --all`, import-only Torch/NeMo/Mamba checks, and CPU TorchCodec WAV decode. Any rebuilt CUDA extension or other package drift needs its own receipt and source review.
3. **Recreate the small native FFmpeg prefix from its explicit lock.** The saved A100 Conda FFmpeg 7.1.1 source manifest has SHA-256 `2bb4a9030bdf0110658acf7217736c909f14be99d1729ff0df75e54691ed1855`; the `@EXPLICIT` spec SHA-256 is `aa72bfe89d57c8139196fa17ec482b838c891be96e592c71416022c3cade2dd6` for 129 archives totaling 143,626,181 bytes. [Micromamba's explicit-spec guidance](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html#explicit-spec-files) supports exact URL/checksum replay. Install into a new task-local prefix at `/home/ubuntu/voicechat-sample-ffmpeg-conda7`, verify all archive hashes and installed metadata, bind the full recursive native tree and actual `LD_LIBRARY_PATH`, then run fresh-process `torchaudio.load` and `AudioDecoder` on the pinned public WAV. The A100's `[1,190278]`/16 kHz decode is evidence that its runtime worked there, not proof on Ubuntu 24.04.
4. **Stage source and checkpoint once.** Check out NVIDIA-NeMo/Speech at `097dfe9e2f55baf653b83035868bdc89849f1b47` and rehash the five pinned source files. Stage only VoiceChat `config.json` and `model.safetensors` in one Hugging Face hub cache on the qualifying checkpoint filesystem, using the revision above and the returned snapshot path; avoid `local_dir` or a second weight copy. [Hugging Face cache docs](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) explain that snapshot paths link to a single blob. Put Xet/temp caches on the filesystem with space, check both files' exact hashes, and recheck `df -B1` after each major step. Stage only the six pinned Nano metadata files at revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`; write `refs/main` as exactly 40 ASCII bytes, no newline, in a dedicated cache; no Nano weights. Keep NeMo worktree clean.
5. **Stop at a new H200 review gate.** Freeze H200-specific absolute paths, OS/GPU/driver, wheel/package inventory, native prefix bytes and loader map, checkpoint and sample hashes, Nano cache, exact command/environment, and a fresh output ID. The A100 V4 freeze and source GO do not apply. The prospective H200 V5 amendment is source only and needs independent review plus a host executable freeze/review before one official public-sample child. The A100 attempt 004 reached the 75,000 MiB cap at 75,044 MiB, which motivates H200 VRAM, but it produced no completed sample and does not justify skipping the disk or environment gates.

The core artifact checks after a storage pass are `sha256sum` of the one resolved weight blob (`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`), `config.json` (`2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5`), and public `sample_fc.wav` (`265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136`). Verify the single-blob layout with `readlink -f <snapshot>/model.safetensors`, `du -sb <hub-cache>`, and `df -B1 <checkpoint-parent>`; hash the resolved file, not just the symlink text. Record `git rev-parse HEAD` in NeMo and require the pinned commit above. These checks are read-only and must precede any freeze.

The single-copy checkpoint operation itself can use `hf_hub_download` with **one** cache and no `local_dir`:

```bash
export HF_HOME="$TASK/voicechat-hf-home"
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
"$VENV/bin/python" - <<'PY'
import pathlib, shutil
from huggingface_hub import hf_hub_download
repo = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
rev = "443794ea956ef0065f001967ffd00e77f519cb39"
floor = 30 * 1024**3
weight_bytes = 44_382_749_892
before = shutil.disk_usage("/").free
# Reserve 4 GiB of the projected 10.34 GB margin for transient transfer files.
assert before >= floor + weight_bytes + 4 * 1024**3, before
for name in ("config.json", "model.safetensors"):
    p = pathlib.Path(hf_hub_download(repo_id=repo, filename=name, revision=rev))
    print(name, "snapshot", p, "blob", p.resolve(), "bytes", p.resolve().stat().st_size)
assert shutil.disk_usage("/").free >= floor
PY
df -B1 / | tee "$TASK/receipts/df-after-checkpoint.txt"
du -sb "$HF_HUB_CACHE" | tee "$TASK/receipts/checkpoint-cache-du.txt"
```

That predownload reserve is an operational margin, not a guarantee about Xet's peak temporary allocation. During the 44 GB transfer, independently watch `df -B1 /` and interrupt it if the 30 GiB floor is endangered; preserve an incomplete download for review only if enough space remains, otherwise remove only its identified task-owned incomplete file after recording the failure. Do not create another weight copy to hash it: run `sha256sum` on the resolved blob in place. Verify `config.json` separately, then check there is exactly one regular weight blob in the HF cache and that the snapshot entry is a symlink to it. The Nano metadata cache must be separate and contain no Nano weights.

The shortest safe path is therefore: **use the completed H200 and measured root → rebuild pinned Python and package environment without persistent download caches → stage one checkpoint blob → check actual free bytes ≥30 GiB → no-model audio/import checks → new source and host freeze review → one separately authorized public sample**. A staging projection alone cannot authorize inference.
