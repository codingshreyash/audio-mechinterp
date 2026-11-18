# VoiceChat 12-WAV ASR QA source handoff and freeze proposal

**State:** source and tests prepared; no ASR inference run. An independent source reviewer must inspect the runner/protocol and the audited render before creating the live freeze receipt. Do not treat this proposal as a freeze receipt.

The prospective runner is `asr_qa_v1.py`, SHA-256 `29da8e97fa40a7f54965215c48a325a21262abf33423eddd26fa4ea2c5c1b3b1`. The protocol SHA-256 is `575d94a642ec061eb1c81766bd516ddbd0a1a54501a7d018d7bdb6665d77f2d5`. The inherited v2 ASR source SHA-256 is `6cc4e6eaa3cf93c3cdf1617589fb1d2ad5eb23890f376aafc5124029474ddcb5`; the inherited model pins SHA-256 is `1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b`. Manifest and script hashes are `bfafef903eaff14a444c015d0c38886e3bedde928ef8c5ca9100aac00370ca7c` and `c3416890395c89c03bdbf2bad9e62e0215bf46d6d9a749f8cb043af1cbe2f99d`.

`--validate-only` passed in `reproductions/audio_prior/.venv` with all 12 pinned WAVs, both cached checkpoint revisions, dependency versions, and source file hashes. Five source/synthetic tests and `py_compile` passed. Validation does not load either ASR model or create transcripts.

After independent review, the only accepted `ASR_QA_FREEZE_V1.json` must contain these exact fields and values (additional reviewer/provenance fields are allowed):

```json
{
  "status": "independently_reviewed_voicechat_asr_qa_frozen",
  "runner_sha256": "29da8e97fa40a7f54965215c48a325a21262abf33423eddd26fa4ea2c5c1b3b1",
  "protocol_sha256": "575d94a642ec061eb1c81766bd516ddbd0a1a54501a7d018d7bdb6665d77f2d5",
  "manifest_sha256": "bfafef903eaff14a444c015d0c38886e3bedde928ef8c5ca9100aac00370ca7c",
  "scripts_sha256": "c3416890395c89c03bdbf2bad9e62e0215bf46d6d9a749f8cb043af1cbe2f99d",
  "pins_sha256": "1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b",
  "v2_asr_source_sha256": "6cc4e6eaa3cf93c3cdf1617589fb1d2ad5eb23890f376aafc5124029474ddcb5"
}
```

Once frozen, run Whisper first, then Wav2Vec2, each with a new output directory. Preserve and anchor both raw transcript directories before `--compare-only`. The comparison is descriptive and retains all 12 cases per recognizer; independent listening QA remains required before VoiceChat inference. Any source change requires a new review and freeze receipt.
