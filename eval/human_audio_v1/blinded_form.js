"use strict";

const DATA = JSON.parse(document.getElementById("embedded-data").textContent);
const CLIPS = DATA.clips;
const $ = (id) => document.getElementById(id);
let REVIEW = null;
let STORAGE_KEY = null;

function need(ok, message) { if (!ok) throw new Error(message); }
function message(text) { $("error").textContent = text; }
function validRecord(record, clip, reviewer) {
  need(record && record.opaque_id === clip.opaque_id && record.wav_sha256 === clip.wav_sha256
    && record.reviewer_id === reviewer && typeof record.submitted_at_utc === "string"
    && typeof record.verbatim_transcript === "string" && record.verbatim_transcript.trim()
    && ["clear", "unclear"].includes(record.clarity)
    && ["yes", "no", "unsure"].includes(record.tail_audible)
    && typeof record.notes === "string", "Stored listening record is incomplete.");
}
function validReview(review, reviewer) {
  need(review && review.schema === "human_audio_blind_review_v1"
    && review.queue_sha256 === DATA.queue_sha256
    && review.bundle_binding === DATA.bundle_binding
    && review.reviewer_id === reviewer
    && review.planned_clips === CLIPS.length
    && Array.isArray(review.records) && review.records.length <= CLIPS.length,
    "Stored review does not match this queue.");
  review.records.forEach((record, i) => validRecord(record, CLIPS[i], reviewer));
}
function show() {
  const index = REVIEW.records.length;
  $("start-card").classList.add("hidden");
  $("review-card").classList.toggle("hidden", index === CLIPS.length);
  $("done-card").classList.toggle("hidden", index !== CLIPS.length);
  if (index === CLIPS.length) return;
  $("progress").max = CLIPS.length;
  $("progress").value = index;
  $("progress-text").textContent = `Clip ${index + 1} of ${CLIPS.length}`;
  $("audio").src = CLIPS[index].file;
  $("transcript").value = "";
  $("clarity").value = "";
  $("tail-audible").value = "";
  $("notes").value = "";
}
function start() {
  try {
    message("");
    need(DATA.schema === "human_audio_blind_queue_v1" && CLIPS.length >= 1,
      "This form has no valid clips.");
    const reviewer = $("reviewer").value.trim();
    need(reviewer.length >= 2 && reviewer.length <= 80, "Enter a reviewer ID of 2–80 characters.");
    STORAGE_KEY = `human-audio-blind-v1|${DATA.queue_sha256}|${encodeURIComponent(reviewer)}`;
    const stored = window.localStorage.getItem(STORAGE_KEY);
    REVIEW = stored ? JSON.parse(stored) : {
      schema: "human_audio_blind_review_v1", reviewer_id: reviewer,
      queue_sha256: DATA.queue_sha256, bundle_binding: DATA.bundle_binding,
      planned_clips: CLIPS.length, started_at_utc: new Date().toISOString(), records: []
    };
    validReview(REVIEW, reviewer);
    if (!stored) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(REVIEW));
    $("reviewer").disabled = true;
    show();
  } catch (error) { message(`Cannot start: ${error.message}`); }
}
function submit() {
  try {
    message("");
    need(REVIEW !== null && REVIEW.records.length < CLIPS.length, "No current clip.");
    const transcript = $("transcript").value.trim();
    const clarity = $("clarity").value;
    const tailAudible = $("tail-audible").value;
    need(transcript && ["clear", "unclear"].includes(clarity)
      && ["yes", "no", "unsure"].includes(tailAudible),
      "Enter words heard, clarity and after-speech sound judgment.");
    const clip = CLIPS[REVIEW.records.length];
    const record = {opaque_id: clip.opaque_id, wav_sha256: clip.wav_sha256,
      reviewer_id: REVIEW.reviewer_id, submitted_at_utc: new Date().toISOString(),
      verbatim_transcript: transcript, clarity,
      tail_audible: tailAudible, notes: $("notes").value.trim()};
    const next = {...REVIEW, records: [...REVIEW.records, record]};
    validReview(next, REVIEW.reviewer_id);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    REVIEW = next;
    show();
  } catch (error) { message(`Cannot lock clip: ${error.message}`); }
}
function exportReview() {
  try {
    message("");
    need(REVIEW !== null && REVIEW.records.length === CLIPS.length,
      "Complete all clips before export.");
    validReview(REVIEW, REVIEW.reviewer_id);
    const complete = {...REVIEW, exported_at_utc: new Date().toISOString()};
    const blob = new Blob([JSON.stringify(complete, null, 2) + "\n"], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `human_review_${REVIEW.reviewer_id.replace(/[^a-z0-9_-]/gi, "_")}_${DATA.queue_sha256.slice(0, 12)}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    message("Completed review exported.");
  } catch (error) { message(`Cannot export: ${error.message}`); }
}
$("start").addEventListener("click", start);
$("submit").addEventListener("click", submit);
$("export").addEventListener("click", exportReview);
