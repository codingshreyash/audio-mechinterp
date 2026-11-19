# H200 VoiceChat Torch distribution identity repair

19 November 2026. **Source-only recommendation, no H200 mutation or model run.** The H200 V5 unsigned builder stops because `importlib.metadata.version("torch")` reports `2.10.0+cu128`, while the reviewed sample preflight requires exact `2.10.0`. The saved A100 `pip freeze --all` and admission receipt reported `torch==2.10.0`, `torchvision==0.25.0`, and `torchaudio==2.10.0`. The H200's cu128-index installation was a staging choice; it does not meet the source-reviewed distribution identity gate. Python packaging [version rules](https://packaging.python.org/en/latest/specifications/version-specifiers/) explain why a bare `pip install torch==2.10.0` can leave `2.10.0+cu128` installed: a public-version equality specifier ignores a candidate's local suffix.

**Recommendation: replace all three Torch-family distributions with the exact PyPI CPython 3.12 Linux x86-64 wheels and keep the V5 source GO unchanged.** This follows the VoiceChat card's unsuffixed Torch package versions and avoids expanding the H200 pin rule. The current H200 has 41,690,677,248 bytes free, or 9,478,422,528 bytes above the H200 30 GiB floor; the three compressed wheels total **925,590,366 bytes**. This is enough planning headroom for a measured transaction, not a guarantee about peak download/unpack/build space. Check actual `df -B1` before, during, and after; stop if the floor is threatened.

The official [PyPI Torch 2.10.0 metadata](https://pypi.org/pypi/torch/2.10.0/json) lists CUDA 12.8 runtime libraries and Triton 3.6.0; I compared all **16** Linux x86-64 CUDA/Triton requirement pins to the saved A100 freeze and every one matches. The chosen `torch-2.10.0-3` wheel was uploaded 2026-03-11 and is the newer build tag. The precise PyPI wheel identities are:

| Distribution | PyPI CPython 3.12 Linux wheel | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| torch 2.10.0 | `torch-2.10.0-3-cp312-cp312-manylinux_2_28_x86_64.whl` | 915,622,781 | `98c01b8bb5e3240426dcde1446eed6f40c778091c8544767ef1168fc663a05a6` |
| torchvision 0.25.0 | `torchvision-0.25.0-cp312-cp312-manylinux_2_28_x86_64.whl` | 8,072,351 | `f25aa9e380865b11ea6e9d99d84df86b9cc959f1a007cd966fc6f1ab2ed0e248` |
| torchaudio 2.10.0 | `torchaudio-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl` | 1,895,234 | `0e77b2956448d63790a99beed0b74ac8b8cd3a94dcdd9ad01974411078f46278` |

Candidate H200 transaction, using the existing task venv and receipts directory. First inspect the dry-run report and existing NVIDIA CUDA dependency pins. The `#sha256=` URL fragments bind download bytes; `--force-reinstall` makes the local-version replacement explicit. `--no-deps` prevents a resolver from changing the already reviewed CUDA dependency closure. These commands are not run by this reviewer:

```bash
set -euo pipefail
TASK=/home/ubuntu/voicechat-sample-stage
PY=/home/ubuntu/voicechat-sample-py312/bin/python
export TMPDIR="$TASK/tmp"
mkdir -p "$TMPDIR" "$TASK/receipts"
df -B1 / | tee "$TASK/receipts/df-before-torch-pypi-repair.txt"
"$PY" -m pip freeze --all > "$TASK/receipts/pip-freeze-before-torch-pypi-repair.txt"
TORCH_WHEEL='torch @ https://files.pythonhosted.org/packages/b3/7a/abada41517ce0011775f0f4eacc79659bc9bc6c361e6bfe6f7052a6b9363/torch-2.10.0-3-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=98c01b8bb5e3240426dcde1446eed6f40c778091c8544767ef1168fc663a05a6'
VISION_WHEEL='torchvision @ https://files.pythonhosted.org/packages/68/2f/f24b039169db474e8688f649377de082a965fbf85daf4e46c44412f1d15a/torchvision-0.25.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=f25aa9e380865b11ea6e9d99d84df86b9cc959f1a007cd966fc6f1ab2ed0e248'
AUDIO_WHEEL='torchaudio @ https://files.pythonhosted.org/packages/98/25/e55a30d7138f8fe56ed006df25b0a3c27681f0ec7bc9989e1778e6d559c3/torchaudio-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=0e77b2956448d63790a99beed0b74ac8b8cd3a94dcdd9ad01974411078f46278'
"$PY" -m pip install --dry-run --no-cache-dir --no-deps --force-reinstall \
  --report "$TASK/receipts/torch-pypi-dry-run.json" \
  "$TORCH_WHEEL" "$VISION_WHEEL" "$AUDIO_WHEEL"
# Review wheel URLs/hashes and disk before the real install.
"$PY" -m pip install --no-cache-dir --no-deps --force-reinstall \
  --report "$TASK/receipts/torch-pypi-install.json" \
  "$TORCH_WHEEL" "$VISION_WHEEL" "$AUDIO_WHEEL"
df -B1 / | tee "$TASK/receipts/df-after-torch-pypi-repair.txt"
"$PY" - <<'PY'
import importlib.metadata as m
assert {n:m.version(n) for n in ('torch','torchvision','torchaudio')} == {
    'torch':'2.10.0','torchvision':'0.25.0','torchaudio':'2.10.0'}
PY
```

**Native-extension gate after replacement.** The existing `causal-conv1d` and `mamba-ssm` extensions were compiled against the former Torch wheel. Rebuild those exact pinned versions from source distributions under the replacement Torch, with the same build mode as the H200 replay:

```bash
"$PY" -m pip install --no-cache-dir --no-build-isolation --no-deps \
  --no-binary=causal-conv1d,mamba-ssm --force-reinstall \
  --report "$TASK/receipts/mamba-causal-after-torch-rebuild.json" \
  causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1
df -B1 / | tee "$TASK/receipts/df-after-torch-extension-rebuild.txt"
"$PY" -m pip freeze --all > "$TASK/receipts/pip-freeze-after-torch-repair.txt"
"$PY" -m pip check > "$TASK/receipts/pip-check-after-torch-repair.txt" 2>&1 || true
```

Save the built `.so` hashes, and review `pip check` rather than treating `|| true` as a pass. Recheck all eleven VoiceChat card pins, Mamba3/causal imports, `torch.version.cuda`, `torchvision`/`torchaudio` imports, the exact FFmpeg 7 loader path, `torchaudio.load` and TorchCodec `AudioDecoder` on the pinned public WAV, and the official `offline_voicechat` import in fresh processes. The TorchCodec wheel may need no reinstall if these checks pass; a failure is a HOLD for a separate repair. None of these checks loads VoiceChat weights or proves CUDA kernel behavior. Record exact package URLs/wheel hashes outside the runner because the V5 executable freeze binds installed version lines and the native FFmpeg tree, not every Torch/Mamba binary byte.

After this package change, any earlier unsigned H200 freeze is stale. Regenerate it, rehash the single checkpoint blob, recheck actual free disk ≥32,212,254,720 bytes, and obtain a fresh independent host freeze signature before one public sample. If the exact PyPI replacement or extension rebuild fails, consider a *separate*, explicitly reviewed H200-only source amendment for the exact `+cu128` distributions and their wheel hashes; never silently broaden `PINNED_PACKAGES` or reuse the existing source GO. The previous A100 binary wheel build tag is not preserved in `pip freeze`, so this is parity at distribution version and CUDA dependency identity, not proof of byte-for-byte identical A100 Torch wheels.
