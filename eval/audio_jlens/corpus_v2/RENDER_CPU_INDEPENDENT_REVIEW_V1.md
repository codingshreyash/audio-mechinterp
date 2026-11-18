# Independent CPU review of prospective Qwen v2 corpus renderer

**Verdict: blocked before freeze or rendering.** This review covers committed implementation `fd2330a` and no v2 WAV, ASR, Qwen, or Voxtral output. The current technical auditor does not prove that a named word fragment is the audio inside its assembled clip.

## Sources and checks

| Source | SHA-256 |
|---|---|
| `render_v2.py` | `5109c97f92e199ea41ead8658701febaf3eaf3182bd99b319a978c0af274ad69` |
| `audit_wavs_v2.py` | `dbefff84c7b051b0328251eb4e0d80a125bba47522a962a31d23576189ab5c71` |
| `RENDER_PROTOCOL_PROSPECTIVE_V1.md` | `9e2085e6a1a056b53d99e001dbc4be36c80c7d742c232b075335e0a4718aa436` |
| `test_render_and_audit_v2.py` | `2ff7cefd5a3431bedceb4c079a4432f332c3b98383bc35769beb03a1e770ace8` |

The renderer's CPU preflight passed in the pinned TTS environment (`eval/tts/.venv/bin/python render_v2.py --validate-only`): 240 matched records, 480 planned WAVs, Piper `1.8.0` and Kokoro ONNX `0.6.1`, local pinned asset checks. The three synthetic tests passed. I also checked the selection expansion: 60 primary pairs yield 36/12/12 FIT/development/locked pairs, 144/48/48 matched records respectively, two words per record, and 480 WAVs total. The 12 screened reserves are absent. The render command checks an exact-hash freeze receipt, creates a fresh directory exclusively, and writes the manifest after completing all records. It does not load an answer or ASR model.

## Blocking gap

`audit_fragment()` reopens and hashes each processed fragment. `audit_pair()` separately reopens and hashes each assembled clip, confirms its geometry, and checks that matched clips have identical PCM outside the word slot. It never compares the assembled clip's word-slot samples with the corresponding processed word fragment; nor does it independently reconstruct the prefix, zero gaps, right padding, and suffix. A clip could therefore contain changed or swapped audio inside the word slot while retaining the expected fragment files. If its clip file/PCM hashes and peak in the manifest were refreshed, the current readback checks would accept it. This breaks the claimed connection between the selected word label and the actual audio reaching the model.

Before creating `RENDER_FREEZE_V1.json`, make the CPU auditor independently assemble each of the 480 expected PCM arrays from the already audited processed fragment files: `prefix + 1920 zero samples + right-padded word fragment + 1920 zero samples + suffix`. Require exact integer-sample equality to the reopened clip, in addition to the current matched-pair outside-slot invariant. Add a synthetic regression that changes only the word-slot samples, refreshes the clip hashes and peak, and expects the audit to fail because the clip no longer equals the named word fragment. Keep the failed first review and source hash in the record; re-review the revised auditor before rendering.

Passing the revised technical audit would still establish waveform and provenance integrity only. The protocol separately requires ASR and listening evidence for intelligibility before Qwen inference. No v2 audio has been rendered by this review.
