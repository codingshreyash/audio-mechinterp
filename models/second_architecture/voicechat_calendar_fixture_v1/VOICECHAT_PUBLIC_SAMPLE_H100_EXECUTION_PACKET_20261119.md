# VoiceChat public sample: bounded H100 execution packet

**Status: source and capacity plan; no new host, download, load, or sample run.** This packet is for the unchanged NVIDIA public random-number sample only. It does not admit the calendar WAVs. The separately reviewed [sample-only runner](public_sample_smoke_v1.py) is the first available bounded VoiceChat load path; there is no reviewed standalone load-only runner.

## Capacity and cost decision

The existing stopped Brev `qwen-audio-mechinterp-a100` is `a100-80gb.1x` ($1.98/hour, 128GB advertised disk). A read-only start showed an A100 80GB PCIe with 81,920 MiB VRAM and 92,438,929,408 bytes (86.09 GiB) free on the 124 GiB root filesystem. No VoiceChat checkpoint or NeMo checkout was found in `/home/ubuntu`. The frozen VoiceChat weight is 44,382,749,892 bytes (41.33 GiB), and [preflight](../voicechat_h100_local_preflight.py) requires 60 GiB free **after** checkpoint staging. Thus at least 101.33 GiB must be free before the weight is staged; this A100 is short by 15.24 GiB before packages or caches. The read-only check ended with `brev ls` showing the instance `STOPPED`. Do not erase the existing Qwen environment to force a pass.

Live Brev search on 2026-11-19 lists one stoppable 80GB H100 type, `gpu-h100-sxm.1gpu-16vcpu-200gb` (Nebius), at $5.40/hour, configurable to 256GB disk with `--min-disk 256`. Disk is listed at $0.12264/GB-month: 256GB is at most $31.40/month, about $0.043/hour on a 730-hour month, with billing details to verify at creation. Plan at most **four active hours** for this setup and one sample attempt: $21.60 GPU compute plus about $0.17 disk over those four hours. Stop at four active hours even if setup is unfinished; a stopped host can preserve the checkpoint, with storage billing continuing. The sample process itself has a separate **1,800-second** wall cap. These are operational spend limits, not a provider-enforced maximum; monitor the instance and stop it explicitly. Do not create a nonstoppable 2TB A100 for this workflow because its checkpoint would be lost on deletion and its bill would continue until deletion.

Before creation, rerun `brev search --gpu-name H100 --stoppable --min-disk 256 --json`; accept only one 80GB H100 at the expected type/rate. The exact candidate command is:

```sh
brev create voicechat-public-sample-h100 --type gpu-h100-sxm.1gpu-16vcpu-200gb --min-disk 256 --stoppable --count 1
```

`brev create ... --dry-run` returned only that type. After boot, inspect `brev ls`, `df -B1 /`, `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`, `python3.12 --version`, and `date -u` before transferring files. Stop if there is not one H100 with at least 75,000 MiB, a usable 256GB volume, Python 3.12 or a reviewed way to install it, and at least 160 GiB free before packages and weight download. That 160 GiB setup threshold is a conservative operational margin; the pinned hard preflight remains 60 GiB **after** staging.

## Immutable source and environment

The already independent [source GO](VOICECHAT_PUBLIC_SAMPLE_SMOKE_INDEPENDENT_SOURCE_REREVIEW_GO_V2.md) binds runner SHA-256 `86f827abc6eeacf95b0c48f1f9bc8026d13a86ff4896deaf9b59cbae68a9b78a`, tests `57f8c1d04f42255c684ac998f90414ca82bd3b6bcc3e2e432fef7564a8c515b8`, handoff `6c0740aea220861caaefc10732983ac880f37ce4a860e3d35c46861caef16ea6`, and preflight `874d0e7347546c9210815503a57cd0ed5a16f5e16b7862c7ba8f61951dd85757`. Those four local hashes were rechecked on 2026-11-19 and all 16 no-model tests passed. Transfer tracked repository files from a fixed Git tree or a hash-checked archive; do not use an edited NeMo checkout. Verify the full six-role source hash map against [the independent source-review record](VOICECHAT_PUBLIC_SAMPLE_SOURCE_REVIEW_GO_V1.json) on the H100 before preparing the freeze.

Stage exactly:

| Component | Revision or exact requirement |
| --- | --- |
| NVIDIA-NeMo/Speech | Git commit `097dfe9e2f55baf653b83035868bdc89849f1b47`, clean tree and five pinned source files |
| VoiceChat checkpoint | `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B@443794ea956ef0065f001967ffd00e77f519cb39`; `config.json` SHA-256 `2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5`; `model.safetensors` SHA-256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`, 44,382,749,892 bytes |
| Nano metadata | `nvidia/NVIDIA-Nemotron-Nano-9B-v2@6533e8de2c68e4536bf7c411d7a3ce5734111476`; exactly six pinned config/code/tokenizer files in a dedicated HF cache, no Nano weights |
| Runtime | Python 3.12; `torch 2.10.0`, `torchvision 0.25.0`, `torchaudio 2.10.0`, `transformers 4.56.0`, `tokenizers 0.22.0`, `lhotse 1.32.2`, `huggingface-hub 0.34.4`, `hf-xet 1.1.9`, `torchcodec 0.10.0`, `causal-conv1d 1.6.2.post1`, `mamba-ssm 2.3.2.post1` |
| Public input | NeMo `sample_fc.wav` SHA-256 `265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136` and `random_number_response.json` SHA-256 `e36ae164cdc14a0b9bbed7b4be2b37223dc3730b39de1edafb99e73638e2b60f` |

Use these fixed locations on the new host, or bind different absolute locations in the freeze before execution. The commands below stop before model import. Run them in a fresh shell with Python 3.12, from the staged repository root. Transfer the tracked local runner directory and preflight with `git archive`/`brev copy` (about 14 MB of current source), then verify its six reviewed hashes. Do not copy untracked local research output into the sample host.

```sh
git clone https://github.com/NVIDIA-NeMo/Speech.git /home/ubuntu/voicechat/Speech
git -C /home/ubuntu/voicechat/Speech checkout --detach 097dfe9e2f55baf653b83035868bdc89849f1b47
python3.12 -m venv /home/ubuntu/voicechat-py312
source /home/ubuntu/voicechat-py312/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install -e '/home/ubuntu/voicechat/Speech[all]'
python -m pip uninstall -y nvidia-resiliency-ext
python -m pip install transformers==4.56.0 tokenizers==0.22.0 lhotse==1.32.2 huggingface-hub==0.34.4 hf-xet==1.1.9 torchcodec==0.10.0 torch_audiomentations jinja2 ninja packaging wheel einops
python -m pip install --no-build-isolation --no-deps causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --revision 443794ea956ef0065f001967ffd00e77f519cb39 --local-dir /home/ubuntu/voicechat-checkpoint
```

For Nano, download **only** `config.json`, `configuration_nemotron_h.py`, `modeling_nemotron_h.py`, `tokenizer_config.json`, `tokenizer.json`, and `special_tokens_map.json` at `6533e8de2c68e4536bf7c411d7a3ce5734111476` into `/home/ubuntu/voicechat-nano-hf-home/hub` with `huggingface_hub.hf_hub_download`; then set its `refs/main` to that commit plus LF as in the [pinned plan](../VOICECHAT_H100_LOCAL_ENV_PLAN.md). Do not download Nano weights. Stage the VoiceChat weight once. Save `pip freeze`, source hashes, checkpoint hashes, `df -B1`, and GPU details outside the NeMo checkout. Do not put build outputs, `__pycache__`, or an extra HF repo in the dedicated Nano cache. Set `PYTHONPATH` to the pinned NeMo directory only and set `HF_HOME`, `HF_HUB_CACHE`, `HF_MODULES_CACHE`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1` for the sample admission and child. The shared preflight's full weight hash, package pins, clean NeMo worktree, offline Nano class resolution, generated module hashes, single GPU, and post-staging disk check must all pass. Do not weaken any gate if setup is difficult.

## Freeze and one-attempt run gate

Choose absolute resolved paths for the source checkout, checkpoint, dedicated Nano cache, and a **new** sample output directory. Use `public_sample_smoke_v1.expected_freeze_fields(nemo_dir, checkpoint_dir, nano_home, output_dir, "h100")` on the prepared host and its final environment. It generates the exact official command and source/child-environment hashes. The independent source-review JSON already exists as `VOICECHAT_PUBLIC_SAMPLE_SOURCE_REVIEW_GO_V1.json`; verify that its six-role hash map equals `expected["source_sha256"]`.

The candidate freeze must be canonical JSON with `schema="voicechat_public_sample_smoke_freeze_v1"`, `status="externally_frozen_sample_only"`, `run_id=output_dir.name`, the exact `expected` map, `caps={"wall_seconds":1800,"vram_mib":75000,"total_evidence_bytes":1073741824}`, and `source_review={"path": absolute_source_review_path,"sha256": source_review_file_sha256}`. The source-review record is a source GO, not a freeze signature. An **independent freeze reviewer** must inspect that exact candidate, the staged hashes and actual H100/disk readings, then produce `voicechat_public_sample_freeze_review_v1` with `status="independently_reviewed_executable_freeze"`, exact freeze/source-review SHA-256 values, reviewer ID, UTC review time, and HMAC-SHA256 over the canonical unsigned review object using a dedicated external key of at least 32 bytes. Preserve the key outside the repo; pass it only to the parent runner via `VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX`, never to the model child. The runner validates the HMAC and source map before any model import.

Only after that review, invoke:

```sh
python3.12 -m models.second_architecture.voicechat_calendar_fixture_v1.public_sample_smoke_v1 \
  --nemo-dir /absolute/pinned/Speech \
  --checkpoint-dir /absolute/voicechat-checkpoint \
  --nano-hf-home /absolute/dedicated-nano-hf-home \
  --output-dir /absolute/new/voicechat-sample-run-id \
  --freeze /absolute/reviewed/freeze.json \
  --freeze-review /absolute/reviewed/freeze-review.json \
  --gpu-policy h100
```

The runner executes only the unchanged `examples/speechlm2/offline_voicechat_fc_infer.py` with `--checkpoint`, `--wav`, `--api-response-json`, and `--output-dir`, using the pinned random-number WAV and response. It starts one fresh process, caps wall time at 1,800 seconds, sampled process-tree VRAM at 75,000 MiB, and retained evidence at 1 GiB, then seals a success or failure manifest. Capture and independently review the output and hashes before any later use. A successful result is `environment_smoke_only`: it proves neither stock/adapter parity nor a calendar action discrepancy. Keep all calendar inference on HOLD until authentic 12/12 human QA, adjudications, C grants, external anchor, fixture-specific admission, fresh-process stock/adapter parity, and a separate fixture executable freeze pass.
