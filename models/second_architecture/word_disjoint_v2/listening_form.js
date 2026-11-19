"use strict";

const DATA = JSON.parse(document.getElementById("embedded-data").textContent);
const $ = (id) => document.getElementById(id);
const CLIPS = DATA.clips;
let review = null;
let storageKey = null;

function error(message) { $("error").textContent = message; }
function require(ok, message) { if (!ok) throw new Error(message); }
function validReview(saved, reviewerId) {
  require(saved && saved.schema === "word_disjoint_blind_review_v1"
    && saved.reviewer_id === reviewerId
    && saved.queue_sha256 === DATA.queue_sha256
    && saved.planned_clips === 32
    && JSON.stringify(saved.opaque_ids_in_order) === JSON.stringify(CLIPS.map((c) => c.opaque_id))
    && Array.isArray(saved.records) && saved.records.length <= 32,
    "Saved review does not match this queue and reviewer.");
  saved.records.forEach((record, index) => {
    const clip = CLIPS[index];
    require(record.reviewer_id === reviewerId && record.opaque_id === clip.opaque_id
      && record.wav_sha256 === clip.wav_sha256
      && typeof record.submitted_at_utc === "string"
      && typeof record.verbatim_transcript === "string" && record.verbatim_transcript.trim()
      && typeof record.middle_word === "string" && record.middle_word.trim()
      && ["clear", "unclear"].includes(record.clear_or_unclear)
      && Number.isInteger(record.confidence) && record.confidence >= 1
      && record.confidence <= 5, "Saved first-impression record is incomplete or out of order.");
  });
}
function showCurrent() {
  const index = review.records.length;
  $("start-card").classList.add("hidden");
  $("review-card").classList.toggle("hidden", index === 32);
  $("done-card").classList.toggle("hidden", index !== 32);
  $("progress").value = index;
  $("progress-text").textContent = `${index} of 32 first impressions locked`;
  if (index === 32) return;
  const clip = CLIPS[index];
  $("clip-number").textContent = `${index + 1} of 32`;
  $("clip-id").textContent = clip.opaque_id;
  $("audio").src = clip.file;
  $("transcript").value = "";
  $("middle-word").value = "";
  $("clarity").value = "";
  $("confidence").value = "";
}
function start() {
  try {
    error("");
    require(DATA.schema === "word_disjoint_blind_form_v1"
      && DATA.planned_clips === 32 && CLIPS.length === 32,
      "Form has an invalid clip population.");
    const reviewerId = $("reviewer").value.trim();
    require(reviewerId.length >= 2 && reviewerId.length <= 80,
      "Enter a unique reviewer ID with 2–80 characters.");
    storageKey = `word-disjoint-blind-v1|${DATA.queue_sha256}|${encodeURIComponent(reviewerId)}`;
    const stored = window.localStorage.getItem(storageKey);
    review = stored ? JSON.parse(stored) : {
      schema: "word_disjoint_blind_review_v1",
      reviewer_id: reviewerId,
      queue_sha256: DATA.queue_sha256,
      planned_clips: 32,
      opaque_ids_in_order: CLIPS.map((c) => c.opaque_id),
      started_at_utc: new Date().toISOString(),
      records: []
    };
    validReview(review, reviewerId);
    if (!stored) window.localStorage.setItem(storageKey, JSON.stringify(review));
    $("reviewer").disabled = true;
    showCurrent();
  } catch (exc) { error(`Could not start or resume: ${exc.message}`); }
}
function submit() {
  try {
    error("");
    require(review !== null && review.records.length < 32, "No current clip.");
    const transcript = $("transcript").value.trim();
    const middle = $("middle-word").value.trim();
    const clarity = $("clarity").value;
    const confidence = Number($("confidence").value);
    require(transcript && middle && ["clear", "unclear"].includes(clarity)
      && Number.isInteger(confidence) && confidence >= 1 && confidence <= 5,
      "Enter a full transcript, middle word, clarity, and confidence. Use [inaudible] when needed.");
    const clip = CLIPS[review.records.length];
    const entry = {
      opaque_id: clip.opaque_id, reviewer_id: review.reviewer_id,
      wav_sha256: clip.wav_sha256, submitted_at_utc: new Date().toISOString(),
      verbatim_transcript: transcript, middle_word: middle,
      clear_or_unclear: clarity, confidence
    };
    const next = {...review, records: [...review.records, entry]};
    validReview(next, review.reviewer_id);
    window.localStorage.setItem(storageKey, JSON.stringify(next));
    review = next;
    showCurrent();
  } catch (exc) { error(`Submission was not stored: ${exc.message}`); }
}
function exportReview() {
  try {
    error("");
    require(review !== null && review.records.length === 32,
      "Complete all 32 first impressions before exporting.");
    validReview(review, review.reviewer_id);
    const documentData = {...review, exported_at_utc: new Date().toISOString()};
    const blob = new Blob([JSON.stringify(documentData, null, 2) + "\n"],
      {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const safeId = review.reviewer_id.replace(/[^a-z0-9_-]+/gi, "_").slice(0, 60);
    anchor.href = url;
    anchor.download = `blind_review_${safeId}_${DATA.queue_sha256.slice(0, 12)}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (exc) { error(`Export failed: ${exc.message}`); }
}

$("start").addEventListener("click", start);
$("submit").addEventListener("click", submit);
$("export").addEventListener("click", exportReview);
