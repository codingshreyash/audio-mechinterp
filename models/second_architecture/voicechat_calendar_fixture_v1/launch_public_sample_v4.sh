#!/usr/bin/env bash
# One-shot public-sample launcher. The reviewed Python runner owns admission,
# process limits, evidence sealing, and the exact official child command.
set -euo pipefail
umask 077
source /home/ubuntu/voicechat-sample-env-20261119.sh
export LD_LIBRARY_PATH="/home/ubuntu/voicechat-sample-ffmpeg-conda7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
key_file=/home/ubuntu/voicechat-sample-freeze-v4/freeze-review.key
test "$(stat -c %a "$key_file")" = 600
export VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX="$(/usr/bin/python3 -c 'from pathlib import Path; p=Path("/home/ubuntu/voicechat-sample-freeze-v4/freeze-review.key"); b=p.read_bytes(); assert len(b)==32; print(b.hex())')"
cd /home/ubuntu/voicechat-sample-repo-v4-cap
set +e
/home/ubuntu/voicechat-sample-py312/bin/python -B -m models.second_architecture.voicechat_calendar_fixture_v1.public_sample_smoke_v1 \
  --nemo-dir /home/ubuntu/voicechat-sample-Speech \
  --checkpoint-dir /data/voicechat-sample-hub/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39 \
  --nano-hf-home /home/ubuntu/voicechat-sample-nano-hf-home \
  --output-dir /home/ubuntu/voicechat-sample-output-004 \
  --freeze /home/ubuntu/voicechat-sample-freeze-v4/freeze.json \
  --freeze-review /home/ubuntu/voicechat-sample-freeze-v4/freeze-review.json \
  --gpu-policy a100-80gb \
  > /home/ubuntu/voicechat-sample-freeze-v4/launcher.stdout \
  2> /home/ubuntu/voicechat-sample-freeze-v4/launcher.stderr
runner_rc=$?
set -e
unset VOICECHAT_SAMPLE_FREEZE_REVIEW_KEY_HEX
printf '%s\n' "$runner_rc" > /home/ubuntu/voicechat-sample-freeze-v4/launcher.exit
printf 'voicechat_public_sample_runner_exit=%s\n' "$runner_rc"
# Brev may retry a nonzero exec. The runner's one-shot output dir and durable
# exit receipt are authoritative; return zero here to avoid a second launch.
exit 0
