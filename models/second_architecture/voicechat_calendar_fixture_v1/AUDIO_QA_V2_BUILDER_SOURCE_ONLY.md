# VoiceChat QA v2 evidence builder

This builder is source-only. It does not listen, infer with VoiceChat, sign a
controller grant, create an external hash anchor, or create a run ledger.
The listener CLI must first save blind and script-aware records for every
required human review.

Run from the repository root after both listeners finish. Supply the actual
reviewer IDs in their saved stage files and an actual UTC time later than all
required reviews:

    python3 -m models.second_architecture.voicechat_calendar_fixture_v1.audio_qa_v2_build receipts --bundle /path/to/private_blind_bundle --records-root /path/to/private_review_records --out /path/to/new_qa_evidence_directory --primary-id primary01 --secondary-id secondary01 --completed-at-utc ACTUAL_UTC_AFTER_LISTENING --controller-id controller_audit_id

Add --tertiary-id tertiary01 when primary and secondary records disagree.
The command derives secondary and tertiary triggers from pinned ASR rows and
human stages. It rejects missing, duplicate, extra, conflicting, and
noncanonical stage files. It copies all original stage bytes unchanged,
creates twelve canonical receipts, and reports the 12-cell eligibility count.
A nominated controller ID in a C receipt does not grant authorization. An
excluded clip stays recorded and prevents the planned full-matrix manifest.

After an independent controller has reviewed the two C recordings and signed
both exact C receipt hashes in a separate grant directory, set the existing
VOICECHAT_CONTROLLER_KEY_HEX environment variable and run:

    python3 -m models.second_architecture.voicechat_calendar_fixture_v1.audio_qa_v2_build manifest --prepared-dir /path/to/new_qa_evidence_directory --grant-dir /path/to/external_signed_c_grants --trusted-controller-id controller_audit_id

The manifest command checks every copied stage and receipt again, verifies
the two external signatures, and copies their exact grant bytes. It requires
12/12 human-eligible WAVs. The final manifest hash must then be anchored in
an external immutable audit store. A separate pre-inference ledger must bind
that anchor and all twelve eligibility rows before the existing v2 live gate
may run. Local code cannot prove that listeners remained blind or that an
external store is immutable.
