# VoiceChat 12-WAV human listening workflow

The local reviewer page is [reviewer A's blind form](/Users/shrranjan/VoiceChat-Blind-Listening-20261118/reviewer_A/BLIND.html). Open it in a browser on this Mac. It contains 12 audio clips under opaque IDs and no reference scripts or ASR output. Listen to every clip, write what you hear, answer every question, and click **Download completed JSON** once. Send the downloaded `reviewer_A__blind_export.json` file to the study coordinator. The page does not save progress between reloads.

The coordinator imports that file before revealing the reference scripts. Only then does the coordinator create `reviewer_A/AWARE.html`. Reviewer A opens that second page, listens to every clip again, compares the audio with its displayed script, answers the questions about what is *audible*, and sends `reviewer_A__script_aware_export.json`.

The coordinator then creates reviewer B's `BLIND.html` with exactly the clips requiring a second review under the existing QA rules. Eleven clips are already flagged by the frozen ASR comparisons; reviewer A's observations may flag the twelfth. Reviewer B must independently complete the blind form before the coordinator creates B's script-aware form. B should not see A's responses. A third reviewer gets only clips where A and B disagree under the existing QA rules. A third reviewer is unnecessary if they agree.

The HTML form exports are intake files. They are not listening receipts or evidence that a person listened. The coordinator imports them into the pinned `voicechat_listener_stage_v2` records, then runs the unchanged receipt builder. The study controller still needs to audit reviewer identity and independence, source access, and the later admission grant.

Run the commands below from `/Users/shrranjan/audio-mechinterp`. The export path normally points to the browser's Downloads folder. Do not edit the downloaded JSON. Do not put the downloaded JSON or any other files inside the records directory.

```sh
BASE=/Users/shrranjan/VoiceChat-Blind-Listening-20261118
RECORDS=/Users/shrranjan/VoiceChat-Listening-Records-20261119
TOOL=models.second_architecture.voicechat_calendar_fixture_v1.audio_qa_v2_web

# After reviewer A sends the blind export:
python3 -m "$TOOL" import-blind --bundle "$BASE" --records-root "$RECORDS" --role primary --primary-id reviewer_A --secondary-id reviewer_B --export /path/to/reviewer_A__blind_export.json
python3 -m "$TOOL" make-aware --bundle "$BASE" --records-root "$RECORDS" --role primary --primary-id reviewer_A --secondary-id reviewer_B --out-dir "$BASE/reviewer_A"

# After reviewer A sends the script-aware export:
python3 -m "$TOOL" import-aware --bundle "$BASE" --records-root "$RECORDS" --role primary --primary-id reviewer_A --secondary-id reviewer_B --export /path/to/reviewer_A__script_aware_export.json
python3 -m "$TOOL" plan --bundle "$BASE" --records-root "$RECORDS" --role secondary --primary-id reviewer_A --secondary-id reviewer_B
python3 -m "$TOOL" make-blind --bundle "$BASE" --records-root "$RECORDS" --role secondary --primary-id reviewer_A --secondary-id reviewer_B --out-dir "$BASE/reviewer_B"

# After reviewer B sends the blind export:
python3 -m "$TOOL" import-blind --bundle "$BASE" --records-root "$RECORDS" --role secondary --primary-id reviewer_A --secondary-id reviewer_B --export /path/to/reviewer_B__blind_export.json
python3 -m "$TOOL" make-aware --bundle "$BASE" --records-root "$RECORDS" --role secondary --primary-id reviewer_A --secondary-id reviewer_B --out-dir "$BASE/reviewer_B"

# After reviewer B sends the script-aware export:
python3 -m "$TOOL" import-aware --bundle "$BASE" --records-root "$RECORDS" --role secondary --primary-id reviewer_A --secondary-id reviewer_B --export /path/to/reviewer_B__script_aware_export.json
python3 -m "$TOOL" plan --bundle "$BASE" --records-root "$RECORDS" --role tertiary --primary-id reviewer_A --secondary-id reviewer_B --tertiary-id reviewer_C
```

If the tertiary plan reports zero clips, stop. Otherwise, run `make-blind`, `import-blind`, `make-aware`, and `import-aware` for `--role tertiary --tertiary-id reviewer_C` in the same order. The reviewer C directory must be distinct. A and B pages should not be shared with C.

The existing builder is `models.second_architecture.voicechat_calendar_fixture_v1.audio_qa_v2_build`. It consumes the per-clip records after all triggered reviews are complete. Its output and the controller's grant remain separate from this intake workflow.
