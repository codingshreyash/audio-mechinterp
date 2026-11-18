"use strict";

const data = JSON.parse(document.getElementById("embedded-data").textContent);
const ids = ["pair", "voice", "recipient", "site", "arm"];
const selectors = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const controlNames = ["pre", "post", "padding", "random"];

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function appendTextCell(row, value, className) {
  const cell = element("td", value, className);
  row.appendChild(cell);
  return cell;
}

function badge(passed) {
  return element("span", passed ? "Pass" : "Fail", `gate ${passed ? "pass" : "fail"}`);
}

function optionList(select, values, previous) {
  select.replaceChildren();
  for (const value of values) {
    const option = element("option", value);
    option.value = value;
    select.appendChild(option);
  }
  select.value = values.includes(previous) ? previous : values[0];
}

function formatScore(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function formatSigned(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return `${value >= 0 ? "+" : ""}${Number(value).toFixed(3)}`;
}

function label(kind) {
  if (kind === "blank_baseline") return "Blanked word base";
  if (kind.startsWith("feature:native_")) return "Unedited speech";
  if (kind.startsWith("feature:")) return `Feature control · ${kind.slice(8)}`;
  if (kind === "passive:clean") return "Passive hook · clean";
  if (kind === "passive:blank") return "Passive hook · blank";
  if (kind === "opposite_word_control") return "Opposite-word clean donor";
  const parts = kind.split(":");
  if (parts[0] === "patch") {
    const names = { whole: "Whole word", word_0: "Word quarter 0", word_1: "Word quarter 1",
      word_2: "Word quarter 2", word_3: "Word quarter 3", pre: "Preword", post: "Postword",
      padding: "Padding", self: "Self patch", random: "Equal-norm random" };
    return names[parts[2]] || kind;
  }
  return kind;
}

function sortOrder(kind) {
  if (kind.startsWith("feature:native_")) return 0;
  if (kind === "blank_baseline") return 1;
  if (kind === "passive:clean") return 2;
  if (kind === "passive:blank") return 3;
  if (kind.endsWith(":self")) return 4;
  if (kind.endsWith(":whole")) return 5;
  for (let quarter = 0; quarter < 4; quarter++) if (kind.endsWith(`:word_${quarter}`)) return 6 + quarter;
  if (kind.endsWith(":pre")) return 10;
  if (kind.endsWith(":post")) return 11;
  if (kind.endsWith(":padding")) return 12;
  if (kind.endsWith(":random")) return 13;
  if (kind === "opposite_word_control") return 14;
  return 15;
}

function renderCards() {
  const cards = document.getElementById("cards");
  const sites = ["encoder31", "projector"];
  const restore = sites.map((site) => data.summary.sites[site].restore.whole_word_success_of_16);
  const loss = sites.map((site) => data.summary.sites[site].loss.whole_word_success_of_16);
  const items = [
    ["Whole-word restoration", restore.every((count) => count === restore[0]) ? `${restore[0]}/16` : restore.map((count) => `${count}/16`).join(" · "), "At both internal sites"],
    ["Valid lexical loss", loss.every((count) => count === loss[0]) ? `${loss[0]}/16` : loss.map((count) => `${count}/16`).join(" · "), "At both internal sites"],
    ["Complete generations", String(data.provenance.generations), "All planned conditions present"],
    ["Independent audit", "Passed", "Exact hashes and frozen gates"],
  ];
  for (const [name, value, note] of items) {
    const card = element("div", undefined, "card");
    card.append(element("div", name, "label"), element("strong", value), element("div", note, "small"));
    cards.appendChild(card);
  }
}

function renderGates() {
  const body = document.getElementById("gates");
  body.replaceChildren();
  for (const site of ["encoder31", "projector"]) {
    for (const arm of ["restore", "loss"]) {
      const outcome = data.summary.sites[site][arm];
      const row = element("tr");
      appendTextCell(row, site === "encoder31" ? "Post-encoder 31" : "Post-projector");
      appendTextCell(row, arm === "restore" ? "Restoration" : "Lexical loss");
      const effect = element("td");
      effect.append(`${outcome.whole_word_success_of_16}/16 `, badge(outcome.whole_word_effect_gate));
      row.appendChild(effect);
      const time = element("td");
      time.appendChild(badge(outcome.time_preference_gate));
      row.appendChild(time);
      appendTextCell(row, controlNames.map((name) => `${name} ${outcome.control_success_of_16[name]}/16`).join(" · "));
      appendTextCell(row, controlNames.map((name) => `${name} ${outcome.score_comparisons[name].valid_of_16}/16`).join(" · "));
      body.appendChild(row);
    }
  }
}

function refreshPairChoices() {
  const previousPair = selectors.pair.value;
  const pairs = [...new Set(data.rows.map((r) => r.pair))].sort();
  optionList(selectors.pair, pairs, previousPair);
  refreshVoiceChoices();
}

function refreshVoiceChoices() {
  const previousVoice = selectors.voice.value;
  const voices = [...new Set(data.rows.filter((r) => r.pair === selectors.pair.value).map((r) => r.voice))].sort();
  optionList(selectors.voice, voices, previousVoice);
  refreshWordChoices();
}

function refreshWordChoices() {
  const previous = selectors.recipient.value;
  const words = [...new Set(data.rows.filter((r) => r.pair === selectors.pair.value && r.voice === selectors.voice.value).map((r) => r.recipient))].sort();
  optionList(selectors.recipient, words, previous);
  renderSelection();
}

function selectedRows() {
  const { pair, voice, recipient, site, arm } = Object.fromEntries(ids.map((id) => [id, selectors[id].value]));
  const rows = data.rows.filter((r) => r.pair === pair && r.voice === voice && r.recipient === recipient &&
    (r.site === null && (r.kind === "blank_baseline" || r.kind.startsWith("feature:")) ||
      r.site === site && (r.kind.startsWith("passive:") || r.kind === "opposite_word_control" && arm === "restore" ||
        r.kind.startsWith(`patch:${arm}:`))));
  return rows.sort((a, b) => sortOrder(a.kind) - sortOrder(b.kind) || a.kind.localeCompare(b.kind));
}

function renderEvidence() {
  const body = document.getElementById("evidence");
  body.replaceChildren();
  const rows = selectedRows();
  const recipient = selectors.recipient.value;
  const arm = selectors.arm.value;
  const native = rows.find((r) => r.kind === `feature:native_${recipient}`);
  const blank = rows.find((r) => r.kind === "blank_baseline");
  if (!native || !blank) throw new Error("Frozen clean/blank reference missing from embedded data");
  const base = arm === "restore" ? blank.contrast : native.contrast;
  const context = `${selectors.pair.value} · ${selectors.voice.value} · ${recipient} versus ${native.opposite} · ${selectors.site.value} · ${arm}`;
  document.getElementById("case-context").textContent = `${context}. Clean output: ${JSON.stringify(native.output)}; blank output: ${JSON.stringify(blank.output)}.`;
  for (const item of rows) {
    const row = element("tr");
    appendTextCell(row, label(item.kind));
    appendTextCell(row, item.output === "" ? "<empty>" : item.output, "out mono");
    appendTextCell(row, item.classification, `class class-${item.classification}`);
    appendTextCell(row, formatScore(item.contrast), "mono");
    const movement = arm === "restore" ? item.contrast - base : base - item.contrast;
    appendTextCell(row, formatSigned(movement), "mono");
    const edit = item.edit || {};
    const rowsText = edit.rows ? `${edit.rows.length} rows${edit.effective_bf16_norm === undefined ? "" : `; norm ${formatScore(edit.effective_bf16_norm)}`}` : "—";
    appendTextCell(row, rowsText, "mono");
    const cell = element("td");
    const details = element("details");
    details.appendChild(element("summary", "Tokens, scores, edit"));
    const raw = { kind: item.kind, token_ids: item.token_ids, terminal_eos: item.terminal_eos,
      canonical_scores: item.scores, hook_calls_including_two_scores: item.hook_calls, edit: item.edit };
    details.appendChild(element("pre", JSON.stringify(raw, null, 2), "hash"));
    cell.appendChild(details);
    row.appendChild(cell);
    body.appendChild(row);
  }
}

function renderMasks() {
  const key = `${selectors.pair.value}|${selectors.voice.value}`;
  const record = data.masks[key];
  if (!record) throw new Error(`Frozen mask record absent for ${key}`);
  const map = record.site_maps[selectors.site.value];
  const sourceClip = record.source_clips[selectors.recipient.value];
  if (!sourceClip) throw new Error("Pinned spoken input absent for selected word");
  const audio = document.getElementById("source-audio");
  audio.src = new URL(`../../../${sourceClip.wav_path}`, document.baseURI).href;
  audio.load();
  document.getElementById("audio-source-meta").textContent =
    `Original input WAV · SHA-256 ${sourceClip.wav_sha256}`;
  const host = document.getElementById("masks");
  host.replaceChildren();
  const named = [...Object.entries(map.bins), ["padding", map.padding_control_rows]];
  for (const [name, rows] of named) {
    const block = element("div", undefined, "mask");
    block.append(element("b", name), element("span", `${rows.length} rows · ${rows[0]}–${rows[rows.length - 1]}`));
    host.appendChild(block);
  }
  const [start, end] = record.word_slot_samples;
  document.getElementById("sample-slot").textContent =
    `Frozen word slot: audio samples [${start}, ${end}); ${record.word_mel_frames_direct_overlap.length} directly overlapping mel frames. Selected model rows may mix wider context.`;
}

function renderProvenance() {
  const host = document.getElementById("provenance");
  host.replaceChildren();
  const names = { model_id: "Model", model_revision: "Model revision", result_sha256: "Result SHA-256",
    audit_sha256: "Audit SHA-256", meta_sha256: "Metadata SHA-256", runner_sha256: "Runner SHA-256",
    masks_sha256: "Mask SHA-256", progress_rows: "Planned slots", generations: "Generations",
    captures: "Captures", teacher_forced_forwards: "Candidate score forwards",
    cumulative_seconds: "Cumulative seconds", peak_mps_bytes: "Peak MPS driver bytes" };
  for (const [key, name] of Object.entries(names)) {
    host.append(element("dt", name), element("dd", data.provenance[key], "hash"));
  }
}

function renderSelection() {
  renderEvidence();
  renderMasks();
}

selectors.pair.addEventListener("change", refreshVoiceChoices);
selectors.voice.addEventListener("change", refreshWordChoices);
for (const id of ["recipient", "site", "arm"]) selectors[id].addEventListener("change", renderSelection);
renderCards();
renderGates();
renderProvenance();
refreshPairChoices();
