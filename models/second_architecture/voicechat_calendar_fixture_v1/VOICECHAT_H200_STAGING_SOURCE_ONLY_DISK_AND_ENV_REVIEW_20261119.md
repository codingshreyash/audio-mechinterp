# H200 VoiceChat staging: source-only disk and environment review

19 November 2026. **HOLD on staging while `voicechat-public-sample-h200-v1` is BUILDING.** A transient shell reported Ubuntu 24.04.5, one H200 with 143,771 MiB, a single 120 GiB root, about 104.48 GB free, and no Python/torch/NeMo/HF staging. A requested `--min-disk 200` is not proof of a delivered 200 GB filesystem; the earlier [Brev disk investigation](VOICECHAT_PUBLIC_SAMPLE_H100_DISK_FAILURE_AMENDMENT_20261119.md) found that this filter was bypassed with an explicit `--type`. Wait for terminal build completion, then decide from actual mounts and free bytes. This review changes no cloud host and runs no model.

## Hard storage gate

The pinned `model.safetensors` is **44,382,749,892 bytes** (41.33 GiB), SHA-256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`, at VoiceChat revision `443794ea956ef0065f001967ffd00e77f519cb39`. Shared preflight and the proposed H200 sample policy require **60 GiB = 64,424,509,440 bytes free on the checkpoint filesystem after staging**. Therefore that filesystem needs **at least 108,807,259,332 free bytes before the one weight blob**, plus room for any cache/temp/output placed there. At ~104.48 **decimal** GB free, the current root would miss the mathematical minimum by ~4.33 GB before any Python, Conda, NeMo or download cost. Even if the reported figure were GiB, it would leave only ~3.15 GiB for all those costs. Do not begin staging on that root unless capacity changes.

For a **single filesystem**, use the existing conservative 160 GiB free-before-setup gate (`171,798,691,840` bytes) from the [sample packet](VOICECHAT_PUBLIC_SAMPLE_H100_EXECUTION_PACKET_20261119.md). For a **split layout**, require a distinct, writable checkpoint volume with at least 110 GiB free (`118,111,600,640` bytes) and root at least 80 GiB free (`85,899,345,920` bytes) before setup; maintain a 20 GiB root floor and the exact 60 GiB postweight checkpoint floor throughout. These are operational margins beyond the mathematical minimum. Do not resize, format, mount, prune, or transfer a 44 GB blob as a way to guess whether the gate passes.

Read-only gate commands after Brev shows terminal ready state:

```bash
brev ls --json
brev exec voicechat-public-sample-h200-v1 'date -u; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits; lsblk -b -o NAME,SIZE,FSTYPE,MOUNTPOINTS; findmnt -T /home/ubuntu; df -B1 /; df -B1 /data 2>/dev/null || true'
```

Record the exact `df -B1` available bytes, `findmnt` device and filesystem for the *actual checkpoint parent*, not just the provider's advertised disk. If no qualifying filesystem exists when the build finishes, **HOLD** and stop the task-created H200 through the established instance-lifecycle process; choose a genuinely larger guest or attach a separately reviewed writable volume. The H200's ample VRAM cannot repair a disk gate.

A read-only gate calculation for a proposed `/data` mount is:

```bash
python3 - <<'PY'
import os, shutil
root = shutil.disk_usage('/').free
data_is_distinct = os.path.isdir('/data') and os.stat('/data').st_dev != os.stat('/').st_dev
data = shutil.disk_usage('/data').free if data_is_distinct else None
weight = 44382749892
postweight_floor = 60 * 1024**3
print({'root_free_bytes': root, 'distinct_data_free_bytes': data,
       'minimum_checkpoint_prestage_bytes': weight + postweight_floor,
       'single_root_gate': root >= 160 * 1024**3,
       'split_gate': data is not None and data >= 110 * 1024**3 and root >= 80 * 1024**3})
PY
```

## Fastest reproducible staging after a disk pass

1. **Recreate, do not transplant, the Python 3.12 environment.** The A100's 430-line `pip freeze --all` inventory has sorted digest `f6993856420dd48ddc370b8c7b7c0fe26fe76bf5cf0843fcdb190dc4568ed5df`, but it is an installed-version receipt, not a wheel lock. It includes a `pyarrow @ file:///home/ubuntu/voicechat-sample-wheelhouse/...` entry whose local path may not exist on H200. Python's [venv documentation](https://docs.python.org/3/library/venv.html) says virtual environments are not movable/copyable; the A100 was Ubuntu 22.04 and H200 reports 24.04. Copying its venv may preserve absolute shebang/base-interpreter paths and native-library assumptions, and matching `pip freeze` would not prove matching wheel bytes. Use a new H200 Python 3.12 venv at the same task path, exact version pins and a task-local wheelhouse with recorded wheel SHA-256. Reuse preserved A100 wheel archives **if their bytes and tags are verified**; otherwise fetch from pinned upstream indexes, hash them, and make a new H200 package inventory. A direct venv copy can only be an experimental diagnostic after independent review, never the staging baseline.
2. **Install the card and repaired runtime pins.** The A100 receipt pinned `torch==2.10.0+cu128`, `torchaudio==2.10.0`, `torchvision==0.25.0`, `torchcodec==0.10.0`, `numpy==1.26.4`, `pyarrow==25.0.1`, `datasets==5.0.1`, `mamba-ssm==2.3.2.post1`, `causal-conv1d==1.6.2.post1`, and the other [11 card pins](../voicechat_h100_local_preflight.py). [PyTorch publishes the 2.10 CUDA 12.8 wheel index](https://pytorch.org/get-started/previous-versions/); select it explicitly, and keep the whole 430-line A100 inventory as a comparison target rather than blindly using it as an install command. Record every wheel/source archive hash, `pip check`, exact versions, `pip freeze --all`, import-only Torch/NeMo/Mamba checks, and CPU TorchCodec WAV decode. Any rebuilt CUDA extension or other package drift needs its own receipt and source review.
3. **Recreate the small native FFmpeg prefix from its explicit lock.** The saved A100 Conda FFmpeg 7.1.1 source manifest has SHA-256 `2bb4a9030bdf0110658acf7217736c909f14be99d1729ff0df75e54691ed1855`; the `@EXPLICIT` spec SHA-256 is `aa72bfe89d57c8139196fa17ec482b838c891be96e592c71416022c3cade2dd6` for 129 archives totaling 143,626,181 bytes. [Micromamba's explicit-spec guidance](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html#explicit-spec-files) supports exact URL/checksum replay. Install into a new task-local prefix at `/home/ubuntu/voicechat-sample-ffmpeg-conda7`, verify all archive hashes and installed metadata, bind the full recursive native tree and actual `LD_LIBRARY_PATH`, then run fresh-process `torchaudio.load` and `AudioDecoder` on the pinned public WAV. The A100's `[1,190278]`/16 kHz decode is evidence that its runtime worked there, not proof on Ubuntu 24.04.
4. **Stage source and checkpoint once.** Check out NVIDIA-NeMo/Speech at `097dfe9e2f55baf653b83035868bdc89849f1b47` and rehash the five pinned source files. Stage only VoiceChat `config.json` and `model.safetensors` in one Hugging Face hub cache on the qualifying checkpoint filesystem, using the revision above and the returned snapshot path; avoid `local_dir` or a second weight copy. [Hugging Face cache docs](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) explain that snapshot paths link to a single blob. Put Xet/temp caches on the filesystem with space, check both files' exact hashes, and recheck `df -B1` after each major step. Stage only the six pinned Nano metadata files at revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`; write `refs/main` as exactly 40 ASCII bytes, no newline, in a dedicated cache; no Nano weights. Keep NeMo worktree clean.
5. **Stop at a new H200 review gate.** Freeze H200-specific absolute paths, OS/GPU/driver, wheel/package inventory, native prefix bytes and loader map, checkpoint and sample hashes, Nano cache, exact command/environment, and a fresh output ID. The A100 V4 freeze and source GO do not apply. The prospective H200 V5 amendment is source only and needs independent review plus a host executable freeze/review before one official public-sample child. The A100 attempt 004 reached the 75,000 MiB cap at 75,044 MiB, which motivates H200 VRAM, but it produced no completed sample and does not justify skipping the disk or environment gates.

The core artifact checks after a storage pass are `sha256sum` of the one resolved weight blob (`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`), `config.json` (`2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5`), and public `sample_fc.wav` (`265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136`). Verify the single-blob layout with `readlink -f <snapshot>/model.safetensors`, `du -sb <hub-cache>`, and `df -B1 <checkpoint-parent>`; hash the resolved file, not just the symlink text. Record `git rev-parse HEAD` in NeMo and require the pinned commit above. These checks are read-only and must precede any freeze.

The shortest safe path is therefore: **finish H200 build → remeasure mounts → secure adequate disk → rebuild pinned environment and one checkpoint blob → no-model audio/import checks → new source/host freeze review**. If the disk remains 120 GiB root only, staging should end at the first gate.
