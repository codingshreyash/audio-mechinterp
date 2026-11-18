# Pinned VoiceChat-11B environment plan: H100 reference, A100 alternative

**Status, 18 November 2026:** local-only preparation. No instance was started or created, no weights were downloaded, and no model or ASR inference was run. Do not start a cloud instance while the frozen Voxtral development run is active on local MPS. The assay and its evidence gates are in [`VOICECHAT_DUAL_HEAD_ASSAY_PROTOCOL.md`](VOICECHAT_DUAL_HEAD_ASSAY_PROTOCOL.md), reviewed in [`VOICECHAT_DUAL_HEAD_ASSAY_INDEPENDENT_REVIEW.md`](VOICECHAT_DUAL_HEAD_ASSAY_INDEPENDENT_REVIEW.md).

## Exact inputs and resource envelope

| Item | Pinned value or requirement |
| --- | --- |
| NeMo Speech source | `NVIDIA-NeMo/Speech@097dfe9e2f55baf653b83035868bdc89849f1b47`, matching the [published VoiceChat branch code](https://github.com/NVIDIA-NeMo/Speech/tree/097dfe9e2f55baf653b83035868bdc89849f1b47) |
| VoiceChat checkpoint | `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B@443794ea956ef0065f001967ffd00e77f519cb39`, matching the [model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) |
| Optional text comparator | `nvidia/NVIDIA-Nemotron-Nano-9B-v2@6533e8de2c68e4536bf7c411d7a3ce5734111476`; separate from the VoiceChat task, so stage only if needed |
| Main weight file | `model.safetensors`, **44,382,749,892 bytes** (41.34 GiB); expected SHA-256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`, from Hugging Face's pinned revision metadata/`x-linked-etag`. Verify actual bytes after any later download. |
| Host | Linux, one H100 with at least 75,000 MiB reported VRAM, batch size 1. NVIDIA lists H100 test hardware and A100 support in its [model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B#software-integration). Neither card nor this plan measures headroom for these hooks. |
| Disk planning | Allow at least **150 GiB free before staging**: 41.34 GiB weights, possible duplicate HF cache, source/build packages, fixtures, and bounded logs. Require at least **60 GiB free after staging** in the local preflight. These are conservative operational estimates, not measured consumption. |
| Network for later setup | Allow outbound access to GitHub, Hugging Face checkpoint/Xet storage, and package indexes, or transfer the exact pinned files by another route and verify their local hashes. Plan for more than 44.4 GB of checkpoint traffic if there is no verified cache; package/build downloads add traffic. The preflight itself uses no network. |

The [official model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B#offline-inference) specifies Python 3.12, PyTorch 2.10.0, torchvision/torchaudio 0.25.0/2.10.0, transformers 4.56.0, tokenizers 0.22.0, lhotse 1.32.2, huggingface-hub 0.34.4, hf-xet 1.1.9, torchcodec 0.10.0, causal-conv1d 1.6.2.post1, and mamba-ssm 2.3.2.post1. Install `.[all]` from the pinned NeMo source and record `pip freeze` as a run artifact because transitives are not fully pinned by the card. The checkpoint's `config.json` must hash to `2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5`, enable `use_function_head`, set `duplex_function_channel_weight=2.0`, and name the Nano 9B v2 backbone.

The read-only `brev ls --json` check on 18 November 2026 showed `voicechat-public-11b` **STOPPED**, type `gpu-h100-sxm.1gpu-16vcpu-200gb`, with `shell_status=NOT READY` and `health_status=UNHEALTHY`. It is a possible later reuse candidate, but this listing does not establish a working shell, usable disk, cached checkpoint, or correct CUDA environment. It was not started, entered, or modified.

An **A100 80GB** is an alternative after the local MPS run completes. The read-only Brev inventory on 18 November lists a fresh `massedcompute_A100_80G` with 80GB VRAM and 256GB disk at **$1.62/hour**, versus **$5.40/hour** for the stopped H100 type. Prices and availability can change. NVIDIA lists A100 support, but VoiceChat load time, peak memory, and the two-pass hook runtime on this A100 are **unmeasured**. Run the same one-sample identity and memory pilot before any fixture matrix; the 80GB label alone is not a fit guarantee. The existing H100 remains the reference because NVIDIA reports it as test hardware.

## Commands for a later, separately scheduled setup

These are reference commands, **not commands executed in this preparation**. Use one prepared H100 or A100 80GB host after the local MPS Voxtral development run is over. Avoid a second copy of the 44.4 GB weights if the host already contains a verified pinned checkpoint.

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
python -m pip freeze > /path/to/run-artifacts/voicechat-pinned-pip-freeze.txt
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --revision 443794ea956ef0065f001967ffd00e77f519cb39 --local-dir /path/to/voicechat-checkpoint
```

The [`voicechat_h100_local_preflight.py`](voicechat_h100_local_preflight.py) script performs **only local reads** and returns a nonzero status on drift. It pins SHA-256 for the five relevant NeMo source files, checkpoint config, and weight; checks the exact git commit, clean worktree, function-head config, key package versions, GPU name/VRAM, disk headroom, and frozen fixture and QA-receipt hashes. It never imports NeMo, accesses the network, downloads a model, or runs inference. The worktree rule rejects all modified/staged tracked files and ordinary untracked files via `git status --porcelain=v1 --untracked-files=all -z`. It enumerates ignored files with `git ls-files --others --ignored --exclude-standard -z` and allows only named, non-symlink setuptools metadata files directly under a root `*.egg-info/` directory, which an editable install may generate. Ignored Python, compiled caches, config, templates, build code, and any other ignored file fail. Keep other build outputs, caches, and logs outside the NeMo checkout. The five hashes are extra checks, not substitutes for this source-tree gate; the receipt records the number of allowed generated metadata files.

```bash
python voicechat_h100_local_preflight.py --self-test
python voicechat_h100_local_preflight.py --phase source --nemo-dir /path/to/Speech --checkpoint-dir /path/to/voicechat-checkpoint
python voicechat_h100_local_preflight.py --phase ready --nemo-dir /path/to/Speech --checkpoint-dir /path/to/voicechat-checkpoint --fixture-manifest /path/to/fixtures/FROZEN_FIXTURE.json
# On an A100 80GB host only, add: --gpu-policy a100-80gb
```

The fixture manifest must have `status=frozen_voicechat_dual_head_fixture`; `system_prompt`, `tool_schema`, and `stub_calendar` entries with `path` and SHA-256; exactly two distinct `tts_voice_ids`; `human_voice_id` (or null); an explicit `human_script_ids` subset; and one frozen `trailing_silence_samples` value. Every artifact path must be normalized and strictly relative to the manifest directory: no absolute path, `.`, `..`, repeated separator, or backslash, and its resolved target must remain inside that directory. The mandatory matrix is **6 scripts** (`A`, `B`, `C`, `D`, `A_literal`, `B_literal`) × **2 TTS voices = 12 WAVs**. Up to six separately recorded human WAVs extend only the scripts named in `human_script_ids`. Every declared `(script_id, voice_id)` must appear exactly once with a unique resolved WAV path, unique final-WAV SHA-256, and frozen transcript. No absent human case is silently counted as tested.

Each WAV entry must also contain `listening_receipt` and `asr_receipt` descriptors with normalized relative `path` and SHA-256. Each referenced JSON receipt must have `schema="voicechat_wav_qa_receipt_v1"`, `kind` (`listening` or `independent_asr`), `verdict="pass"`, the case's `script_id`, `voice_id`, and final padded `wav_sha256`, plus `reference_transcript`, `observed_transcript`, and an ISO-8601 UTC `reviewed_utc`. The preflight compares both transcript fields to the manifest transcript after Unicode NFKC, case folding, and punctuation/whitespace normalization. Listening receipts require a nonempty `reviewer_id`. ASR receipts require a non-VoiceChat `model_id`, nonempty `model_revision`, and a `settings` object. The files and their hashes must be unique across all cases and types. Capture these two receipts separately from the fixture manifest after actual human listening and independent ASR; `listened: true` and `independent_asr_checked: true` flags are no longer accepted as evidence. The preflight verifies receipt structure and byte binding, but cannot prove that a reviewer listened, that ASR actually ran, or that a declared human voice is authentic. Keep source evidence for human recordings outside this preflight and review it before making a human-voice claim.

The [official offline instructions](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B#offline-inference) require sufficient trailing silence on a custom WAV but give no fixed duration. Choose and freeze one duration **before inference**; five seconds of exact digital silence (`80,000` samples at 16 kHz) is a conservative starting fixture design, to be checked in the identity pilot. The preflight enforces the manifest's exact trailing-zero count, at least one second, and hashes the **final padded WAV**. Perform human listening and independent ASR on that same padded WAV; never add padding dynamically after observing outcomes. If the pilot shows the frozen duration is inadequate, construct and review a new fixture before measuring outcomes. Freeze the tool JSON, prompt, stub responses, dates, voices, and transcripts as well. The preflight checks receipt structure and hashes; it does not itself judge speech intelligibility or semantic tool correctness.

## Fail-closed smoke sequence after staging

1. Pass local `--phase source`, then `--phase ready`. Verify the source revision and file hashes before any hook code is patched. Record actual package versions, checkpoint SHA, GPU memory, free disk, fixture SHA, and run command in a receipt.
2. On one official sample with a very short prompt, compare the unmodified official helper against the instrumented helper and no-hook versus passive-hook runs at fixed seeds/settings. Preserve **both** pass results; require identical text/function IDs, parsed call positions/JSON, and locally captured trimmed `gen_codes`, plus waveform agreement within a specified numerical tolerance. The pinned VoiceChat return currently sets `tokens_audio=None`, so codec IDs require a local capture before `_post_inference`. If identity fails, stop before fixtures.
3. Verify that each hook is reached on the intended generated frame: `stt_model.llm` return before both heads, `function_head` before greedy selection, and `lm_head` before text sampling. The target must not be a prompt position or an injected call/response token. Log both pre-boost head output and actual returned/sampled logits. Confirm fixture-specific stub response routing and independent confirmation denial on list-only or ask-first requests.
4. Only then run a tiny dry fixture pilot, measure peak VRAM and per-pass time, and decide whether the **12 mandatory plus 0–6 declared human WAVs** across two passes fit. Do not count a text/codec/audio flip, model discrepancy, or security finding from this setup plan alone.

Do not treat the separate Nano text checkpoint as a same-model typed-user control. The [official function-call example](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/offline_voicechat_fc_infer.py) supplies a text **system prompt** and audio user input. The [offline helper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/utils/offline_voicechat.py#L171-L289) does two passes and injects a prewritten response rather than executing a live tool.
