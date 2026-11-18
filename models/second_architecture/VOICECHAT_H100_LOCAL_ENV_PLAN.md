# Pinned VoiceChat-11B H100 environment plan

**Status, 18 November 2026:** local-only preparation. No instance was started or created, no weights were downloaded, and no model or ASR inference was run. Do not use cloud compute while the frozen Voxtral development run is active. The assay and its evidence gates are in [`VOICECHAT_DUAL_HEAD_ASSAY_PROTOCOL.md`](VOICECHAT_DUAL_HEAD_ASSAY_PROTOCOL.md), reviewed in [`VOICECHAT_DUAL_HEAD_ASSAY_INDEPENDENT_REVIEW.md`](VOICECHAT_DUAL_HEAD_ASSAY_INDEPENDENT_REVIEW.md).

## Exact inputs and resource envelope

| Item | Pinned value or requirement |
| --- | --- |
| NeMo Speech source | `NVIDIA-NeMo/Speech@097dfe9e2f55baf653b83035868bdc89849f1b47`, matching the [published VoiceChat branch code](https://github.com/NVIDIA-NeMo/Speech/tree/097dfe9e2f55baf653b83035868bdc89849f1b47) |
| VoiceChat checkpoint | `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B@443794ea956ef0065f001967ffd00e77f519cb39`, matching the [model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) |
| Optional text comparator | `nvidia/NVIDIA-Nemotron-Nano-9B-v2@6533e8de2c68e4536bf7c411d7a3ce5734111476`; separate from the VoiceChat task, so stage only if needed |
| Main weight file | `model.safetensors`, **44,382,749,892 bytes** (41.34 GiB); expected SHA-256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`, from Hugging Face's pinned revision metadata/`x-linked-etag`. Verify actual bytes after any later download. |
| Host | Linux, one H100 with at least 75,000 MiB reported VRAM, batch size 1. NVIDIA lists H100 test hardware in its [offline instructions](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B#offline-inference). This is a planning gate, not proof of sufficient headroom for hooks. |
| Disk planning | Allow at least **150 GiB free before staging**: 41.34 GiB weights, possible duplicate HF cache, source/build packages, fixtures, and bounded logs. Require at least **60 GiB free after staging** in the local preflight. These are conservative operational estimates, not measured consumption. |

The [official model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B#offline-inference) specifies Python 3.12, PyTorch 2.10.0, torchvision/torchaudio 0.25.0/2.10.0, transformers 4.56.0, tokenizers 0.22.0, lhotse 1.32.2, huggingface-hub 0.34.4, hf-xet 1.1.9, torchcodec 0.10.0, causal-conv1d 1.6.2.post1, and mamba-ssm 2.3.2.post1. Install `.[all]` from the pinned NeMo source and record `pip freeze` as a run artifact because transitives are not fully pinned by the card. The checkpoint's `config.json` must hash to `2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5`, enable `use_function_head`, set `duplex_function_channel_weight=2.0`, and name the Nano 9B v2 backbone.

The read-only `brev ls --json` check on 18 November 2026 showed `voicechat-public-11b` **STOPPED**, type `gpu-h100-sxm.1gpu-16vcpu-200gb`, with `shell_status=NOT READY` and `health_status=UNHEALTHY`. It is a possible later reuse candidate, but this listing does not establish a working shell, usable disk, cached checkpoint, or correct CUDA environment. It was not started, entered, or modified.

## Commands for a later, separately scheduled setup

These are reference commands, **not commands executed in this preparation**. Use a single prepared H100 host after the Voxtral cloud run is over. Avoid a second copy of the 44.4 GB weights if the host already contains a verified pinned checkpoint.

```bash
git clone https://github.com/NVIDIA-NeMo/Speech.git Speech
cd Speech
git checkout --detach 097dfe9e2f55baf653b83035868bdc89849f1b47
conda create -y -n voicechat-pinned python=3.12
conda activate voicechat-pinned
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install -e '.[all]'
pip uninstall -y nvidia-resiliency-ext
pip install transformers==4.56.0 tokenizers==0.22.0 lhotse==1.32.2 huggingface-hub==0.34.4 hf-xet==1.1.9 torchcodec==0.10.0 torch_audiomentations jinja2
pip install ninja packaging wheel einops
pip install --no-build-isolation --no-deps causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1
python -m pip freeze > voicechat-pinned-pip-freeze.txt
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --revision 443794ea956ef0065f001967ffd00e77f519cb39 --local-dir /path/to/voicechat-checkpoint
```

The [`voicechat_h100_local_preflight.py`](voicechat_h100_local_preflight.py) script performs **only local reads** and returns a nonzero status on drift. It pins SHA-256 for the five relevant NeMo source files, checkpoint config, and weight; checks the exact git commit, function-head config, key package versions, GPU name/VRAM, disk headroom, and frozen fixture hashes. It never imports NeMo, accesses the network, downloads a model, or runs inference.

```bash
python voicechat_h100_local_preflight.py --self-test
python voicechat_h100_local_preflight.py --phase source --nemo-dir /path/to/Speech --checkpoint-dir /path/to/voicechat-checkpoint
python voicechat_h100_local_preflight.py --phase ready --nemo-dir /path/to/Speech --checkpoint-dir /path/to/voicechat-checkpoint --fixture-manifest /path/to/fixtures/FROZEN_FIXTURE.json
```

The fixture manifest must have `status=frozen_voicechat_dual_head_fixture`; `system_prompt`, `tool_schema`, and `stub_calendar` entries with relative `path` and SHA-256; and exactly 18 distinct WAV entries with relative path, SHA-256, and completed `listened` and `independent_asr_checked` flags. Freeze the exact tool JSON, prompt, stub responses, original WAVs, dates, voices, and transcripts before the assay. The preflight verifies bytes and audit flags; it does not itself judge speech intelligibility or confirm that a tool schema is semantically correct.

## Fail-closed smoke sequence after staging

1. Pass local `--phase source`, then `--phase ready`. Verify the source revision and file hashes before any hook code is patched. Record actual package versions, checkpoint SHA, GPU memory, free disk, fixture SHA, and run command in a receipt.
2. On one official sample with a very short prompt, compare the unmodified official helper against the instrumented helper and no-hook versus passive-hook runs at fixed seeds/settings. Preserve **both** pass results; require identical text/function IDs, parsed call positions/JSON, and locally captured trimmed `gen_codes`, plus waveform agreement within a specified numerical tolerance. The pinned VoiceChat return currently sets `tokens_audio=None`, so codec IDs require a local capture before `_post_inference`. If identity fails, stop before fixtures.
3. Verify that each hook is reached on the intended generated frame: `stt_model.llm` return before both heads, `function_head` before greedy selection, and `lm_head` before text sampling. The target must not be a prompt position or an injected call/response token. Log both pre-boost head output and actual returned/sampled logits. Confirm fixture-specific stub response routing and independent confirmation denial on list-only or ask-first requests.
4. Only then run a tiny dry fixture pilot, measure peak VRAM and per-pass time, and decide whether the planned 18-WAV/two-pass matrix fits. Do not count a text/codec/audio flip, model discrepancy, or security finding from this setup plan alone.

Do not treat the separate Nano text checkpoint as a same-model typed-user control. The [official function-call example](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/offline_voicechat_fc_infer.py) supplies a text **system prompt** and audio user input. The [offline helper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/utils/offline_voicechat.py#L171-L289) does two passes and injects a prewritten response rather than executing a live tool.
