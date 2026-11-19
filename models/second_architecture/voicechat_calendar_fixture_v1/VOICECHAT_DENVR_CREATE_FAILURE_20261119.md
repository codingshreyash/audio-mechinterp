# VoiceChat sample: Denver A100 create failed before provisioning

**19 November 2026. No Denver host was created. No packages, checkpoint, or model were staged or run.**

The independently reviewed single-candidate packet at `928438c` passed a fresh live inventory check and `brev create --dry-run` at 16:26 UTC. The candidate was exactly `denvr_A100_sxm4_80G`, one 80 GB A100 with 2,125 GB advertised disk, nonstoppable, at a listed $1.80/hour. The dry run printed `denvr_A100_sxm4_80G (2125GB disk)`.

At `2026-11-19T16:26:03Z`, the exact one-candidate JSON was passed to `brev create voicechat-public-sample-denvr-a100 --count 1`. The installed Brev CLI returned: `Skipping: instance type "denvr_A100_sxm4_80G" is not a recognized type; run 'brev search' to see available types`, then `Only created 0/1 instances`. It exited 1. A subsequent `brev ls --json` contained only the pre-existing stopped Qwen A100 and garak L40S. The searchable candidate and client dry run therefore did **not** establish an executable create route. There is no Denver instance to delete or bill in this attempt.

As a separate capacity check, the pre-existing `qwen-audio-mechinterp-a100` was started and read without altering stored files. The guest reported one `NVIDIA A100 80GB PCIe` with 81,920 MiB VRAM. `/` had 92,438,323,200 bytes free; `/data` was a mounted ext4 filesystem with 119,767,867,392 bytes free. A separate 960,197,124,096-byte NVMe block device had no detected filesystem or mount and was **not** touched. Python 3.12.16 exists in `/home/ubuntu/qwen-py312/bin/python`, although `/usr/bin/python3` is 3.10.12. This shows a possible split-filesystem route, but it is not yet an independently reviewed staging or inference plan. A stop command was accepted; terminal STOPPED state must be verified before treating billing as ended.

The stop reached `STOPPED` and `shell_status=NOT READY` in `brev ls --json` by 16:29 UTC. Both pre-existing workspaces are STOPPED, and no task-created workspace exists.

The next cloud action needs a new, source-reviewed existing-host plan that accounts for the 44,382,749,892-byte VoiceChat weight without duplicating it, keeps at least 60 GiB free on the checkpoint filesystem after staging, leaves the existing Qwen environment intact, and binds final paths in the external executable freeze. The official sample remains on HOLD; calendar inference has separate human QA and adapter gates.
