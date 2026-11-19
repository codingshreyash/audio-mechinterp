# Whisper padding channel V3 executable handoff

Status: **HOLD for independent executable rereview; no candidate weights loaded.**

The V3 worker, parent, and auditor preserve the reviewed 436-call schedule and V5 feature evidence. They use `runs/padding_channel_v1/attempt_003`, fixed source-only admission `PADDING_CHANNEL_ADMISSION_V3.json`, and a future independently committed `PADDING_CHANNEL_EXECUTABLE_GO_V3.json`. The saved audit path is `runs/padding_channel_v1/attempt_003_audit.json`. The two remaining V2 HOLD fixes are specified in [prospective amendment V6](WHISPER_PADDING_CHANNEL_PROSPECTIVE_V6.md): exact audit destination and a post-completion terminal receipt with deadline checks.

After an independent GO, and only then, the one-shot parent command is:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.padding_channel_parent_v3 --admission /Users/shrranjan/audio-mechinterp/models/second_architecture/nonspeech_probe_v1/PADDING_CHANNEL_ADMISSION_V3.json --go /Users/shrranjan/audio-mechinterp/models/second_architecture/nonspeech_probe_v1/PADDING_CHANNEL_EXECUTABLE_GO_V3.json
```

The saved auditor must run separately after parent completion. It performs no candidate inference and must reject missing or late terminal proof. Neither command is authorized by this handoff alone.
