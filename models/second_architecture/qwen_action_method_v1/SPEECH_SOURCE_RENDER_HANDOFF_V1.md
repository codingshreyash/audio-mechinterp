# Qwen action-label speech sources V1

19 November 2026. This is a **source-only** 18-WAV render for the frozen action-label method rehearsal. It does not load Qwen weights, inspect old locked outputs, or authorize candidate inference. The separate [Qwen V1 review](INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md) holds candidate inference until endpoint and lens-method corrections, plus human and runtime gates, are complete.

`render_sources.py` uses the previously pinned [VoiceChat TTS implementation](../voicechat_acoustic_action_v1/corpus.py) for Piper Amy, macOS Samantha, and Kokoro Bella. The source spec and implementation SHA-256 values, TTS asset hashes, OS build and `say` hash, per-utterance raw, converted, and final WAV hashes, conversion metadata, and pre/post Mac AC/lid/thermal evidence are saved. It uses the existing uniform level rule, resamples to 16-kHz mono PCM16, rejects a clip that would be clipped or reach 16 seconds, then appends exact zero frames to 256,000 samples. The complete source is rendered independently; no speech is trimmed or sped up to fit.

The render is one-shot to a new absolute directory, capped at 900 seconds and 200 MiB, with immutable start and failure receipts. `audit_sources.py` independently reconstructs every converted and final PCM array from its raw audio, rehashes all 54 audio files, checks the frozen 18-case plan, source/host identities and time/evidence caps. It writes a separate technical audit. A technical byte pass says nothing about whether people hear the intended words.

`listening_bundle.py` creates two separate reviewer directories with different, independently shuffled `q01.wav`–`q18.wav` sequences and browser forms. Each reviewer sees audio and ordinal only. A technical mapping outside the reviewer directories links ordinals to frozen cases and WAV hashes; do not send that mapping with the review forms. The form records a verbatim transcript, meaning, date, event, decisive-phrase timing, and unclear words. It exports a JSON review, but creates **no** human response itself. Two English-fluent people must each complete all 18 clips without scripts or expected labels. A third person resolves critical disagreement. The source cannot pass the human gate from synthetic audio or ASR alone.

Source-only commands from repository root after code review:

```sh
eval/tts/.venv/bin/python -m unittest models.second_architecture.qwen_action_method_v1.test_render_sources -q
eval/tts/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.render_sources --output /absolute/new/attempt_path
eval/tts/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.audit_sources --folder /absolute/new/attempt_path --output /absolute/new/audit.json
eval/tts/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.listening_bundle --source /absolute/new/attempt_path --output /absolute/new/review_bundle
```

Every renderer failure remains a separate attempt. Do not replace a failed clip after viewing candidate model output. Do not release the locked Qwen answers until the separately reviewed method, 18/18 human wording gate, live tokenizer/shape/parity checks, and resource pilot pass.
