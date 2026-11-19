"use strict";
const EMBEDDED = JSON.parse(document.getElementById("embedded-data").textContent);
const TEMPLATE = EMBEDDED.template;
const $ = (id) => document.getElementById(id);
const FIELDS = ["date_match", "event_match", "authorization_match",
                "same_critical_meaning_as_clean"];
const IDS = ["date", "event", "authorization", "same"];
const KEY = `voicechat-qa48-compare|${EMBEDDED.template_sha256}|${TEMPLATE.first_export_sha256}|${TEMPLATE.reviewer_id}`;
let saved = [];
function require(ok, message) { if (!ok) throw new Error(message); }
function error(message) { $("error").textContent = message; }
function check() {
  require(TEMPLATE.schema === "voicechat_acoustic_qa48_compare_v1"
    && /^[0-9a-f]{64}$/.test(EMBEDDED.template_sha256)
    && Array.isArray(TEMPLATE.records) && TEMPLATE.records.length > 0,
    "Invalid comparison template.");
  require(Array.isArray(saved) && saved.length <= TEMPLATE.records.length,
    "Saved comparison population differs.");
  saved.forEach((row, index) => {
    const original = TEMPLATE.records[index];
    const immutable = ["opaque_id", "reference_script", "source_id", "arm",
      "initial_verbatim", "clean_initial_verbatim", "clean_comparison_basis"];
    require(Object.keys(row).sort().join("|") === Object.keys(original).sort().join("|")
      && immutable.every((name) => row[name] === original[name])
      && FIELDS.every((name) => ["yes", "no", "unclear"].includes(row[name]))
      && typeof row.compared_at_utc === "string",
      "Saved comparison differs from template.");
  });
}
function show() {
  const index = saved.length;
  $("review-card").classList.toggle("hidden", index === TEMPLATE.records.length);
  $("done-card").classList.toggle("hidden", index !== TEMPLATE.records.length);
  if (index === TEMPLATE.records.length) return;
  const row = TEMPLATE.records[index];
  $("progress").textContent = `${index + 1} of ${TEMPLATE.records.length} • ${row.opaque_id}`;
  $("reference").textContent = row.reference_script;
  $("initial").textContent = row.initial_verbatim;
  $("clean-basis").textContent = row.clean_comparison_basis === "first_clean_verbatim"
    ? "Your first impression of the clean version"
    : "Reference script (sparse third review)";
  $("clean-initial").textContent = row.clean_initial_verbatim;
  IDS.forEach((id) => $(id).value = "");
}
function submit() {
  try {
    error("");
    require(saved.length < TEMPLATE.records.length, "No current comparison.");
    const values = IDS.map((id) => $(id).value);
    require(values.every((v) => ["yes", "no", "unclear"].includes(v)),
      "Record all four comparisons.");
    const row = {...TEMPLATE.records[saved.length], compared_at_utc:new Date().toISOString()};
    FIELDS.forEach((field, index) => row[field] = values[index]);
    const next = [...saved, row];
    saved = next;
    check();
    localStorage.setItem(KEY, JSON.stringify(saved));
    show();
  } catch (exc) { error(`Submission was not stored: ${exc.message}`); }
}
function exportReview() {
  try {
    error("");
    require(saved.length === TEMPLATE.records.length, "Complete every comparison first.");
    const documentData = {...TEMPLATE, records:saved,
      exported_at_utc:new Date().toISOString()};
    const blob = new Blob([JSON.stringify(documentData, null, 2) + "\n"],
      {type:"application/json"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `qa48_compare_${TEMPLATE.role}_${TEMPLATE.reviewer_id}.json`;
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (exc) { error(`Export failed: ${exc.message}`); }
}
try {
  const stored = localStorage.getItem(KEY);
  saved = stored ? JSON.parse(stored) : [];
  check(); show();
} catch (exc) { error(`Could not open comparison: ${exc.message}`); }
$("submit").addEventListener("click", submit);
$("export").addEventListener("click", exportReview);
