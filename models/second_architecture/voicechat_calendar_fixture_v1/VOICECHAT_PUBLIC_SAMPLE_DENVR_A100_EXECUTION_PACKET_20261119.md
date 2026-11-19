# VoiceChat official public sample: single denvr A100 execution packet

**Status: source and dry-run only.** No denvr instance was created and no VoiceChat model was loaded or run for this packet. This route is for the unchanged NVIDIA random-number public sample in the [reviewed runner](public_sample_smoke_v1.py). It admits no calendar WAV or calendar result. The earlier [H100 attempts](VOICECHAT_PUBLIC_SAMPLE_H100_ATTEMPT_RECORD_20261119.md) ended before packages, weights, or inference; both task-created workspaces were deleted.

The original six-role source GO predates the [Nano `refs/main` repair](VOICECHAT_NANO_REF_REPAIR_V1.md); this packet is not a current executable GO. Re-review the repaired source and create a new host-specific freeze before any sample run.

## Exact candidate and cost boundary

On 2026-11-19, Brev CLI `search --gpu-name A100 --min-disk 2125 --json` returned `denvr_A100_sxm4_80G`: one A100 80GB, 24 vCPUs, 116 GiB RAM, fixed `target_disk_gb=2125`, $1.80/hour, `stoppable=false`. A read-only one-candidate JSON dry run printed `denvr_A100_sxm4_80G (2125GB disk)`. Search inventory and rate can change. Use the exact filter below immediately before creation and abort on zero/multiple candidates or any changed field. The JSON path matters: the installed Brev CLI ignores `--min-disk` as a capacity request when paired with literal `--type`, which produced the earlier 120 GiB H100 guest disk. The dry run proves the **client request**, not delivered guest capacity.

```sh
set -euo pipefail
brev ls --json | jq -e '[.workspaces[] | select(.name == "voicechat-public-sample-denvr-a100")] | length == 0' >/dev/null
brev search --gpu-name A100 --min-disk 2125 --json \
  | jq -e '[.[] | select(.type == "denvr_A100_sxm4_80G" and .cloud == "denvr" and .provider == "shadeform" and .gpu_count == 1 and .vram_per_gpu_gb == 80 and .target_disk_gb == 2125 and .disk_min_gb == 2125 and .disk_max_gb == 2125 and .stoppable == false and .price_per_hour <= 1.8)] | if length == 1 then . else error("expected exactly one denvr A100 candidate") end' \
  > /tmp/voicechat-denvr-a100-2125-candidate.json
jq -c '.[] | {type,cloud,provider,gpu_count,vram_per_gpu_gb,target_disk_gb,price_per_hour,stoppable}' /tmp/voicechat-denvr-a100-2125-candidate.json
brev create voicechat-public-sample-denvr-a100 --count 1 --dry-run \
  < /tmp/voicechat-denvr-a100-2125-candidate.json
# Creation, only after the one-line dry run still says denvr_A100_sxm4_80G (2125GB disk):
brev create voicechat-public-sample-denvr-a100 --count 1 \
  < /tmp/voicechat-denvr-a100-2125-candidate.json
```

Use **one** candidate and `--count 1`; no fallback type or second instance. Save the search JSON, dry-run output, create stdout/stderr, create request UTC time, and the task-created instance **ID** from `brev ls --json` immediately. Do not mistake a CLI exit code or `Success! Created` for a ready host: the second H100 attempt printed success and then failed in the provider. Check `brev ls --json` until the named ID is `RUNNING` and the shell is ready. If provisioning fails or stalls, delete this task-created ID and verify terminal removal; do not continue setup.

This offer is **not stoppable**. Start an operational deletion deadline at the create request, conservatively no later than four wall-clock hours after that request. At the listed rate, four hours of compute is $7.20; exact billing and any storage/tax charges are unverified. Set a separate alert at three hours 45 minutes to export evidence. At the deadline, issue `brev delete <task-created-ID>` even if setup or the sample is unfinished, then poll `brev ls --json` until that ID is absent. Preserve the command responses and status timeline. An accepted delete request alone does not prove billing ended. If deletion fails or the ID persists, check the Brev console/activity/billing and escalate with the ID and timestamps; do not create another host. Brev exposes no provider-enforced four-hour TTL here, so an unattended run cannot honestly claim a guaranteed spend cap. The runner's 1,800-second cap limits the sample process only, not the instance bill.

## Postboot gates before any download

Run these on the new host and save their raw output locally. The host paths below assume `/home/ubuntu`; inspect the actual login/home first and bind all final absolute paths in the freeze. The 2,125 GB search value might not appear as the root filesystem. Use only a real, writable filesystem with verified free bytes for the checkout, environment, checkpoint, Nano cache, and output.

```sh
date -u
id
python3.12 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
lsblk -b
df -B1
df -B1 /home/ubuntu
```

Require exactly one A100 with at least 75,000 MiB reported VRAM, Python 3.12 or a reviewed install path, and at least **160 GiB free before setup** on the intended staging filesystem. The 160 GiB gate is an operational margin. The pinned [preflight](../voicechat_h100_local_preflight.py) independently requires at least **60 GiB free after checkpoint staging** on the checkpoint filesystem. If the delivered host lacks capacity or the free filesystem cannot be used by the runner, preserve the readings and delete the instance. Do not relax the preflight or erase a pre-existing workspace.

## Stage pinned source and checkpoint

Follow the source hashes, exact revisions, package versions, Nano metadata list, and offline cache layout in the [pinned environment plan](../VOICECHAT_H100_LOCAL_ENV_PLAN.md) and [H100 source packet](VOICECHAT_PUBLIC_SAMPLE_H100_EXECUTION_PACKET_20261119.md). The H100 packet's *cloud creation and cost route* is superseded by this A100 route; its unchanged sample source and staging instructions still apply. In particular:

1. Transfer only the tracked reviewed local runner/preflight/source-review files from a fixed Git tree or hash-checked archive. Verify the six-role source map against [the independent source GO](VOICECHAT_PUBLIC_SAMPLE_SOURCE_REVIEW_GO_V1.json); runner SHA-256 is `86f827abc6eeacf95b0c48f1f9bc8026d13a86ff4896deaf9b59cbae68a9b78a` and preflight SHA-256 is `874d0e7347546c9210815503a57cd0ed5a16f5e16b7862c7ba8f61951dd85757`.
2. Check out NVIDIA-NeMo/Speech at clean commit `097dfe9e2f55baf653b83035868bdc89849f1b47`. Use Python 3.12 and the pinned runtime versions in the H100 packet; record `pip freeze` and any CUDA/build errors. Transitive dependencies are not fully frozen by the upstream card.
3. Stage `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B@443794ea956ef0065f001967ffd00e77f519cb39` exactly once. The weight is 44,382,749,892 bytes with SHA-256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`. Download only the six pinned Nano metadata files at `6533e8de2c68e4536bf7c411d7a3ce5734111476` into the dedicated cache; no Nano weights.
4. Save source, model, sample WAV/JSON, cache, package, GPU, and free-disk hash/identity receipts **outside** the NeMo checkout. Keep the checkout clean. Set only the pinned NeMo path in `PYTHONPATH`; use the specified dedicated `HF_HOME`, `HF_HUB_CACHE`, and `HF_MODULES_CACHE`, offline flags, and `PYTHONDONTWRITEBYTECODE=1` for admission and child execution.

If staging exceeds the deletion deadline, export partial evidence and delete. A successful package installation or weight download is not a model result.

## External freeze, one sample attempt, and evidence

Use `public_sample_smoke_v1.expected_freeze_fields(nemo_dir, checkpoint_dir, nano_home, output_dir, "a100-80gb")` with final resolved host paths. The candidate canonical JSON must have `schema="voicechat_public_sample_smoke_freeze_v1"`, `status="externally_frozen_sample_only"`, `run_id=output_dir.name`, that exact `expected` map, `caps={"wall_seconds":1800,"vram_mib":75000,"total_evidence_bytes":1073741824}`, and the absolute independent source-review path plus its SHA-256. Have an independent reviewer check the candidate against **actual** host GPU/disk readings, source hashes, staged model/cache, and package receipts. They must make the `voicechat_public_sample_freeze_review_v1` record with `status="independently_reviewed_executable_freeze"`, exact freeze and source-review hashes, reviewer ID and UTC time, and HMAC-SHA256 over the canonical unsigned review object with a dedicated external key of at least 32 bytes. Keep the key outside the repository and pass it only as `VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX` to the parent runner. The runner validates this review before model import. A source GO alone is not an executable freeze.

Only after that review, in the final pinned environment:

```sh
python3.12 -m models.second_architecture.voicechat_calendar_fixture_v1.public_sample_smoke_v1 \
  --nemo-dir /absolute/pinned/Speech \
  --checkpoint-dir /absolute/voicechat-checkpoint \
  --nano-hf-home /absolute/dedicated-nano-hf-home \
  --output-dir /absolute/new/voicechat-sample-run-id \
  --freeze /absolute/reviewed/freeze.json \
  --freeze-review /absolute/reviewed/freeze-review.json \
  --gpu-policy a100-80gb
```

The runner performs the unchanged official sample in one fresh process with a 1,800-second wall cap, sampled process-tree VRAM cap of 75,000 MiB, and retained evidence cap of 1 GiB. Copy the output directory, `final_manifest.json`, `final_manifest.sha256`, freeze/review records, preflight/staging receipts, and Brev lifecycle evidence to local durable storage **before deletion**; verify local hashes and manifest status. Keep partial/failure evidence too. Delete the task-created instance at the four-hour deadline regardless of result, and record its final absence in `brev ls --json`. Do not leave a nonstoppable host running to retain weights.

`environment_smoke_only` would show that the pinned public sample can run on this A100. It would not establish model security behavior, speech/tool mismatch, stock-versus-adapter parity, or any calendar result. Calendar inference remains on HOLD pending authentic all-12 human QA, adjudications, C grants, external anchor, fixture-specific admission, fresh-process parity, and a separate fixture executable freeze.

## Unresolved risks

- The provider may reject creation again, or deliver a disk size/filesystem different from the JSON request. The postboot gate decides.
- A100 support is listed by NVIDIA, but this model's load time, peak memory, CUDA extension builds, and sample completion on this exact denvr image are unmeasured. Stop at any pin, memory, or deadline failure.
- The nonstoppable instance loses its staged checkpoint on deletion. Evidence must be exported first; the 4-hour budget has no provider-enforced TTL.
- If independent freeze review cannot be completed within the budget, do not run the sample. Export staging evidence and delete.
