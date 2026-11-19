# H200 VoiceChat NeMo dependency conflict: source-only review

19 November 2026. No H200 package or model operation was performed in this review.

## Finding

The pinned [NeMo `setup.py`](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/setup.py) includes its `lightning` group in `[all]`. That group's [requirements file](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/requirements/requirements_lightning.txt) requires `transformers>=4.51.0,<=4.52.0`. NVIDIA's [VoiceChat model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) itself says `pip install -e ".[all]"` **then** install `transformers==4.56.0`; the local sample runner's pinned preflight also requires 4.56.0. Thus a *single constrained resolver transaction* for `.[all]` and 4.56.0 cannot succeed. This is a packaging constraint conflict, not evidence that pinned NeMo source code fails with 4.56.0. The A100 inventory had NeMo 2.4.0rc0 and Transformers 4.56.0, and its public sample reached model construction before the FFmpeg failure.

**Recommendation:** preserve `transformers==4.56.0`. Install the pinned NeMo checkout in editable **base** form, without `[all]`, after replaying the saved A100 dependency versions. `--no-deps` is narrow and acceptable only with that separate dependency installation and the no-model checks below. Do not use a bare `--no-deps` NeMo install as a complete environment, or relax the sample preflight to 4.52.0.

## Candidate H200 commands

The earlier [staging memo](VOICECHAT_H200_STAGING_SOURCE_ONLY_DISK_AND_ENV_REVIEW_20261119.md) defines and hashes `"$TASK/requirements-from-a100.txt"`: it contains the A100's version pins, substitutes the verified official PyArrow 25.0.1 wheel URL for the machine-local `file://` URL, and defers NeMo plus the two source-built CUDA extensions. With the H200's already installed CUDA 12.8 Torch 2.10.0 stack, use that file and retain a pip report for each transaction:

```bash
set -euo pipefail
TASK=/home/ubuntu/voicechat-sample-stage
PY=/home/ubuntu/voicechat-sample-py312/bin/python
SPEECH=/home/ubuntu/voicechat-sample-Speech
test "$(git -C "$SPEECH" rev-parse HEAD)" = 097dfe9e2f55baf653b83035868bdc89849f1b47
test -f "$TASK/requirements-from-a100.txt"
"$PY" -m pip install --dry-run --no-cache-dir \
  --report "$TASK/receipts/a100-pins-dry-run.json" -r "$TASK/requirements-from-a100.txt"
# Inspect the dry-run resolution, package URLs/tags and df before installing.
"$PY" -m pip install --no-cache-dir \
  --report "$TASK/receipts/a100-pins-install.json" -r "$TASK/requirements-from-a100.txt"
"$PY" -m pip install --no-cache-dir --no-build-isolation --no-deps \
  --report "$TASK/receipts/nemo-base-install.json" -e "$SPEECH"
"$PY" -m pip install --no-cache-dir --no-build-isolation --no-deps \
  --report "$TASK/receipts/mamba-source-install.json" \
  causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1
"$PY" -m pip freeze --all > "$TASK/receipts/h200-freeze-after-nemo.txt"
"$PY" -m pip check > "$TASK/receipts/h200-pip-check.txt" 2>&1 || true
df -B1 / > "$TASK/receipts/df-after-nemo.txt"
```

Review `pip check` rather than treating `|| true` as a pass. The A100 had a documented `decord==0.6.0` wheel/platform warning, accepted only for the bounded audio sample; every new warning needs a separate decision. In a **fresh process**, verify all 11 runner card pins, NumPy 1.26.4, PyArrow 25.0.1, Datasets 5.0.1, Mamba's eight dependency pins, and imports of `nemo.collections.speechlm2.inference.utils.offline_voicechat`, `Mamba3`, TorchCodec, `torchaudio` and `lightning`. The official WAV decode requires the separately locked FFmpeg 7 prefix and its `LD_LIBRARY_PATH`; do not interpret a missing FFmpeg library as a NeMo package conflict. These checks import modules and decode the public WAV only; they do not load VoiceChat weights.

The full A100 inventory is a known import-capable **version set**, not an exact wheel lock. Ubuntu 24.04 wheel selection, source builds and CUDA ABI may differ; save URLs/hashes and stop on unexpected changes. Monitor the actual single-root disk after every transaction and retain at least **30 GiB** after staging under the reviewed H200-only sample gate. The current H200 source/freeze review remains separate; none of these install commands authorizes model inference.
