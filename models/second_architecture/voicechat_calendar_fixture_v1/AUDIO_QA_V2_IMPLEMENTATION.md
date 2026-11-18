# VoiceChat frozen audio QA v2 implementation

This is a **pre-inference gate** for the 12 final WAVs rendered at `20261118T1145Z`. It does not contain listener judgments or VoiceChat output. The [accepted amendment](VOICECHAT_AUDIO_QA_AMENDMENT_V2_PROPOSAL.md) defines the prospective rules; its stale sentence about an earlier adapter is superseded by the receipt hash binding in `two_pass_adapter.py` and the v2 code here.

## Collect human records

Run from the repository root with Python 3.12 or compatible local Python:

```sh
python3 -m models.second_architecture.voicechat_calendar_fixture_v1.audio_qa_v2_listen prepare --out /path/to/private_blind_bundle --seed A_PRECOMMITTED_RANDOM_SEED
python3 -m models.second_architecture.voicechat_calendar_fixture_v1.audio_qa_v2_listen listen --bundle /path/to/private_blind_bundle --records /path/to/private_review_records --reviewer-id PRIMARY_AUDIT_ID --all
```

The bundle copies the 12 exact frozen WAVs under opaque IDs and records the shuffled order and hashes. The collector saves a canonical, content-addressable **blind** transcript before revealing the matched script. It then saves a separate script-aware judgment. It never displays ASR or model output. Give a different reviewer the opaque IDs triggered for secondary review, using `--ids ID1 ID2 ...`; do the same for a third reviewer when primary and secondary differ. The organizer must preserve the assignment, stage files, reviewer roster, and excluded rows. Do not put private identities in public fixture records; use stable audit IDs.

The collector supports `--play` with local `afplay` or `ffplay`. Without that switch it prints the opaque audio path for the reviewer to play. The CLI cannot prove a person listened or remained blind outside the interface. The controller must independently inspect the two eligible C recordings and complete review before signing any grant.

## Build and anchor the cohort

`audio_qa_v2.py` validates a canonical JSON manifest with schema `voicechat_audio_qa_manifest_v2`. Its 12 `rows` have exact `case_id`, `voice_id`, `wav_sha256`, `receipt` (`path` and `sha256`), and `final_eligibility`. It binds the accepted amendment commit/hash, frozen render, scripts, both ASR directories, comparison, reviewer roster, blind-order seed and assignment descriptor, per-reviewer opaque order, 12 ordered eligibility rows, and two C grant descriptors. Each `voicechat_wav_qa_receipt_v2` binds the exact final WAV and PCM, script, both raw ASR rows, separate primary/secondary/tertiary stage file descriptors, independently derivable triggers, final eligibility, rationale, completion time, and C controller ID where applicable. The canonical file encoding is ASCII JSON with sorted keys, compact separators and one trailing newline; all descriptors hash the **exact file bytes**.

An external, immutable audit store must anchor the final QA manifest hash and all 12 eligibility rows **before model inference**. A local `voicechat_preinference_ledger_v2` copy must contain `qa_manifest_sha256`, matching `eligibility`, `external_anchor_audit_id`, and `anchored_utc`. The validator verifies local consistency and chronology; it cannot authenticate an external store's immutability by itself. `test_audio_qa_v2.py` has synthetic positive and negative records that show the exact schema. They are tests, not human QA.

The gate requires 12/12 human eligible. It independently recomputes secondary triggers from both immutable ASR observations and primary judgments; the existing ASR alone triggers at least 11/12. It derives tertiary triggers from reviewer disagreement. ASR exact text is descriptive, so its 8/12 Whisper and 3/12 Wav2Vec2 lexical match scores do not automatically exclude clips. If any clip is excluded, the planned 6×2 assay stops. A subset requires a separately reviewed prospective protocol.

For each C recording, the external controller signs the exact finalized receipt hash plus final WAV hash, fixed date `2026-11-19`, event `DR19`, and controller ID. The secret is at least 32 bytes and stays outside the repository, prompt, and logs. A valid grant can only select the inert `authorized_dry_run` stub; no calendar mutation is connected. The adapter rejects a C grant that differs from the one bound in the QA manifest.

## Preflight and live entry point

The H100/A100 `ready` and `preload` preflight phases require `--qa-v2-manifest`, `--qa-v2-ledger`, `--trusted-controller-id`, and a hex encoded `VOICECHAT_CONTROLLER_KEY_HEX` environment variable in addition to the pinned NeMo, Nano, checkpoint, and GPU arguments. It checks the full v2 cohort and never accepts a single v1 receipt as a substitute. The legacy `check_fixtures` function remains for historical source self-tests and does not certify v2.

`run_live_case_from_audited_wav` requires the same v2 QA manifest and ledger, checks the full cohort and case receipt before importing NeMo or running the model, loads the audited WAV through the pinned official helper, then invokes the two-pass adapter. The lower-level `run_two_pass` retains a v1 path for source-only fake tests; any live calendar assay must use the v2-gated live entry point. Official helper/stock output parity and actual listener records are still separate gates.
