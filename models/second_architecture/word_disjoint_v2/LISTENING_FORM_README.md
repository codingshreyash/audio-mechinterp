# Blind listening form for the 32 new clips

The source builder reads only `rendered_v2/listening/QUEUE.json` and the 32 anonymous WAV files beside it. It rejects any extra or missing WAV, duplicate ID, unexpected public queue field, or symlink. For each WAV it recomputes the queue builder's opaque ID from the queue's manifest SHA-256 and actual WAV SHA-256; swapped anonymous filenames are rejected before the page is written. It embeds each verified anonymous WAV filename and SHA-256 in one offline HTML page. It never reads `QA_PRIVATE_CASES.json`, the reference scripts, or model outputs.

Build the page after the technical audio audit and listening queue are complete:

```sh
python3 -m models.second_architecture.word_disjoint_v2.listening_form
python3 -m unittest models.second_architecture.word_disjoint_v2.test_listening_form -v
node --check models/second_architecture/word_disjoint_v2/listening_form.js
```

Open `rendered_v2/listening/LISTENING_FORM.html` in a browser. Give each independent reviewer a separate browser profile and access only to the `listening` directory. Each reviewer uses a unique ID, listens to all 32 clips, enters a full verbatim transcript and middle word, marks clear/unclear, and selects confidence 1–5. Use `[inaudible]` if a word cannot be understood. After each submit, the page locks that first impression in the browser's local storage and advances to the next clip. On reload, entering the same ID resumes the queue. After 32 submissions, the reviewer downloads one JSON export and sends it to the study coordinator. The form refuses to advance if local storage fails.

Export schema `word_disjoint_blind_review_v1` has top-level `reviewer_id`, `queue_sha256`, `planned_clips`, `opaque_ids_in_order`, `started_at_utc`, `records`, and `exported_at_utc`. The 32 records are in queue order. Each record contains `opaque_id`, `reviewer_id`, `wav_sha256`, `submitted_at_utc`, `verbatim_transcript`, `middle_word`, `clear_or_unclear`, and integer `confidence`. Do not edit or merge the reviewers' initial exports. The independent QA auditor must compare their complete records to the private reference map later. The page does not prove that a human listened or that two reviewer IDs belong to two people; the coordinator must establish reviewer identity and independence.

The private case map stays outside the listener-facing directory. A completed form is **not** the 32/32 QA receipt and does not authorize candidate-model inference.
