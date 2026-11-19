"use strict";
const DATA = JSON.parse(document.getElementById("embedded-data").textContent);
const QUEUE = DATA.queue;
const CLIPS = QUEUE.clips;
const $ = (id) => document.getElementById(id);
let review = null, storageKey = null;
function require(ok, message) { if (!ok) throw new Error(message); }
function error(message) { $("error").textContent = message; }
function check(saved, reviewer) {
  require(saved && saved.schema === "voicechat_acoustic_qa48_first_v1"
    && saved.reviewer_id === reviewer && saved.queue_sha256 === DATA.queue_sha256
    && saved.role === QUEUE.role && Array.isArray(saved.records)
    && saved.records.length <= CLIPS.length, "Saved review differs from queue.");
  saved.records.forEach((rec, index) => {
    require(rec.opaque_id === CLIPS[index].opaque_id
      && rec.wav_sha256 === CLIPS[index].wav_sha256 && rec.reviewer_id === reviewer
      && typeof rec.verbatim_transcript === "string" && rec.verbatim_transcript.trim()
      && typeof rec.uncertain_spans === "string"
      && ["date_heard", "event_heard", "authorization_heard"].every(
        (key) => typeof rec[key] === "string" && rec[key].trim())
      && ["clear", "unclear"].includes(rec.clarity)
      && Number.isInteger(rec.confidence) && rec.confidence >= 1 && rec.confidence <= 5,
      "Stored first impression incomplete or out of order.");
  });
}
function show() {
  const index = review.records.length;
  $("start-card").classList.add("hidden");
  $("review-card").classList.toggle("hidden", index === CLIPS.length);
  $("done-card").classList.toggle("hidden", index !== CLIPS.length);
  if (index === CLIPS.length) return;
  const clip = CLIPS[index];
  $("progress").textContent = `${index + 1} of ${CLIPS.length} • ${clip.opaque_id}`;
  $("audio").src = clip.file;
  for (const id of ["transcript", "uncertain", "date", "event", "authorization",
                     "clarity", "confidence"]) $(id).value = "";
}
function start() {
  try {
    error("");
    require(QUEUE.schema === "voicechat_acoustic_qa48_blind_queue_v1"
      && CLIPS.length === QUEUE.planned_clips && CLIPS.length > 0,
      "Invalid blind queue.");
    const reviewer = $("reviewer").value.trim();
    require(/^[A-Za-z0-9_.-]{3,64}$/.test(reviewer), "Enter a unique 3–64 character reviewer ID.");
    storageKey = `voicechat-qa48|${DATA.queue_sha256}|${encodeURIComponent(reviewer)}`;
    const saved = localStorage.getItem(storageKey);
    review = saved ? JSON.parse(saved) : {
      schema: "voicechat_acoustic_qa48_first_v1", reviewer_id: reviewer,
      queue_sha256: DATA.queue_sha256, role: QUEUE.role,
      started_at_utc: new Date().toISOString(), records: []
    };
    check(review, reviewer);
    if (!saved) localStorage.setItem(storageKey, JSON.stringify(review));
    $("reviewer").disabled = true;
    show();
  } catch (exc) { error(`Could not start: ${exc.message}`); }
}
function submit() {
  try {
    error("");
    require(review && review.records.length < CLIPS.length, "No current clip.");
    const value = (id) => $(id).value.trim();
    const confidence = Number(value("confidence"));
    require(value("transcript") && value("uncertain") && value("date")
      && value("event") && value("authorization")
      && ["clear", "unclear"].includes(value("clarity"))
      && Number.isInteger(confidence) && confidence >= 1 && confidence <= 5,
      "Complete every field; use “none” or “unclear” when needed.");
    const clip = CLIPS[review.records.length];
    const rec = {
      opaque_id: clip.opaque_id, wav_sha256: clip.wav_sha256,
      reviewer_id: review.reviewer_id, submitted_at_utc: new Date().toISOString(),
      verbatim_transcript: value("transcript"), uncertain_spans: value("uncertain"),
      date_heard: value("date"), event_heard: value("event"),
      authorization_heard: value("authorization"), clarity: value("clarity"),
      confidence
    };
    const next = {...review, records: [...review.records, rec]};
    check(next, review.reviewer_id);
    localStorage.setItem(storageKey, JSON.stringify(next));
    review = next;
    show();
  } catch (exc) { error(`Submission was not stored: ${exc.message}`); }
}
function exportReview() {
  try {
    error("");
    require(review && review.records.length === CLIPS.length, "Review every clip first.");
    const exported = {...review, exported_at_utc: new Date().toISOString()};
    const blob = new Blob([JSON.stringify(exported, null, 2) + "\n"],
      {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `qa48_first_${QUEUE.role}_${review.reviewer_id}.json`;
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (exc) { error(`Export failed: ${exc.message}`); }
}
$("start").addEventListener("click", start);
$("submit").addEventListener("click", submit);
$("export").addEventListener("click", exportReview);
