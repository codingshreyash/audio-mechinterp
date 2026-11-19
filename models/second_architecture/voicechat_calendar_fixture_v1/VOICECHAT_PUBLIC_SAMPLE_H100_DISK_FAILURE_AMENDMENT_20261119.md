# Independent amendment: H100 disk request failed

**Decision: HOLD the VoiceChat public-sample run.** This amends my earlier [source-only packet review](VOICECHAT_PUBLIC_SAMPLE_H100_PACKET_INDEPENDENT_REVIEW_20261119.md). Its statement that the proposed `--type ... --min-disk 256` command would provision a 256 GB disk was wrong. The source review of the frozen runner remains separate; no sample model run occurred on this host.

## Observed failure

On 2026-11-19, Brev CLI v0.6.328 created `voicechat-public-sample-h100` (`4l2urj2zc`) using `--type gpu-h100-sxm.1gpu-16vcpu-200gb --min-disk 256 --stoppable`. The preflight read `lsblk`/`df`: the only guest root disk was **128,849,018,880 bytes = 120 GiB**, with **104,487,313,408 bytes = 97.31 GiB free** and no separate usable volume. That fails the packet's **160 GiB free before setup** gate. It also cannot hold the 44,382,749,892-byte (41.33 GiB) checkpoint and satisfy the pinned **60 GiB free afterward** gate, even before package and cache usage. No weights or packages were staged. A stop by name returned `Internal Server Error` while the host was `BUILDING`; a stop by instance ID was accepted. At this review's latest read-only `brev ls --json`, the host was still `STOPPING`, with shell `NOT READY`. Acceptance of the stop request is **not** evidence that compute billing has ended.

## Root cause in the installed CLI

The [v0.6.328 create source](https://github.com/brevdev/brev-cli/blob/v0.6.328/pkg/cmd/gpucreate/gpucreate.go) parses a literal `--type` as an `InstanceSpec` with `DiskGB=0`. `resolveInstanceTypes` returns that spec directly, bypassing the search/filter branch where `--min-disk 256` becomes `DiskGB=256`. `createWorkspace` writes `DiskStorage` only if `DiskGB>0`. Otherwise, [the v0.6.328 store source](https://github.com/brevdev/brev-cli/blob/v0.6.328/pkg/store/workspace.go) supplies `DefaultDiskStorage="120Gi"`, exactly matching the guest disk. Thus `--min-disk` is a search filter that was **bypassed entirely** when paired with explicit `--type`. The original dry run printed only the type name, concealing that `DiskGB=0`. The installed CLI rejects `--disk-size` as an unknown flag; [v0.6.337 create source](https://github.com/brevdev/brev-cli/blob/v0.6.337/pkg/cmd/gpucreate/gpucreate.go) still shows no `--disk-size` flag, so upgrading to that version is not a proven remedy.

Read-only dry-run comparison on v0.6.328:

```text
brev create ... --type gpu-h100-sxm.1gpu-16vcpu-200gb --stoppable --min-disk 256 --dry-run
→ gpu-h100-sxm.1gpu-16vcpu-200gb

brev create ... --gpu-name H100 --stoppable --min-disk 256 --dry-run
→ gpu-h100-sxm.1gpu-16vcpu-200gb (256GB disk), a3-megagpu-8g:nvidia-h100-mega-80gb:8 (256GB disk)
```

The second command would retain a much more expensive eight-GPU fallback. For an exact single candidate, use a validated search-result JSON file, then pipe it to `brev create` **without `--type`**. The current CLI parses `target_disk_gb` from JSON stdin into `DiskGB` and sends `DiskStorage="256Gi"` for this path. A read-only dry run of this exact-type JSON path printed `gpu-h100-sxm.1gpu-16vcpu-200gb (256GB disk)`.

```sh
set -euo pipefail
brev search --gpu-name H100 --stoppable --min-disk 256 --json \
  | jq -e '[.[] | select(.type == "gpu-h100-sxm.1gpu-16vcpu-200gb" and .gpu_count == 1 and .stoppable == true and .disk_min_gb <= 256 and .disk_max_gb >= 256 and .target_disk_gb == 256 and .price_per_hour <= 5.4)] | if length == 1 then . else error("expected exactly one candidate") end' \
  > /tmp/voicechat-h100-256-candidate.json
jq -e 'length == 1 and .[0].target_disk_gb == 256' /tmp/voicechat-h100-256-candidate.json >/dev/null
brev create voicechat-public-sample-h100-retry --count 1 --dry-run \
  < /tmp/voicechat-h100-256-candidate.json
# Only after reviewing the dry-run output and current price/status, omit --dry-run.
```

The search JSON may change; recheck rate, GPU count, disk bounds, and the exact candidate before any creation. Do not create the retry while the first instance remains `STOPPING` or while its billing state is unclear. The JSON/CLI path proves what the client **requests**, not what the provider delivers: after boot, inspect `lsblk -b`, `df -B1 /`, and `df -B1` for any other usable mount before downloads. Require at least 160 GiB free before setup and 60 GiB free after checkpoint staging. Retain the packet's four-active-hour operational cap and 1,800-second sample cap; they are not provider-enforced cost limits.

## Stop-state handling

[Brev's v0.6.328 stop source](https://github.com/brevdev/brev-cli/blob/v0.6.328/pkg/cmd/stop/stop.go) sends one stop request and explicitly tells the caller to check `brev ls`; it does not wait for `STOPPED`. Its [store call](https://github.com/brevdev/brev-cli/blob/v0.6.328/pkg/store/workspace.go) is a `PUT` to the workspace stop endpoint. Brev labels stop behavior provider-dependent, and its [instance lifecycle docs](https://docs.nvidia.com/brev/concepts/gpu-instances) distinguish `Running` from `Stopped` billing. I found no documented ten-minute `STOPPING` allowance, so the sustained state is **unresolved**, not safely assumed stopped. Read-only remediation is to check `brev ls --json`, the console instance status, Activity Logs, and Billing & Usage. If it remains `STOPPING`, give Brev support the instance ID, failed name-stop error, accepted ID-stop time, status history, and billing concern. Do not restart or retry provisioning the same host as a disk fix. It contains no staged checkpoint.

The original [execution packet](VOICECHAT_PUBLIC_SAMPLE_H100_EXECUTION_PACKET_20261119.md) must replace its exact `--type ... --min-disk 256` candidate command and its claim that this combination configures a 256 GB disk before another execution attempt. The source-only code review can remain valid, but its cloud execution route is **HOLD** until this amendment is incorporated and a new host passes the observed disk gate.
