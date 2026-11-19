"use strict";

const DATA = JSON.parse(document.getElementById("embedded-data").textContent);
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const ARM_LABEL = {
  W: "W · donor word rows", P: "P · preceding rows", Q: "Q · following rows",
  self_W: "Self W", full_donor: "Full donor", R1: "Random R1", R2: "Random R2"
};
const ARM_ORDER = ["W", "P", "Q", "self_W", "full_donor", "R1", "R2"];
const BASE = new Map(DATA.baselines.map((r) => [`${r.mode}|${r.case_key}`, r]));

function option(value, label) {
  const e = document.createElement("option");
  e.value = value; e.textContent = label; return e;
}
function svg(tag, attrs = {}, text = null) {
  const e = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) e.setAttribute(key, String(value));
  if (text !== null) e.textContent = text;
  return e;
}
function fixed(value, digits = 3) {
  return Number(value).toFixed(digits);
}
function badge(value, css) {
  const e = document.createElement("span");
  e.className = `badge ${css}`; e.textContent = value; return e;
}
function selectValues(id, values, oldValue) {
  const node = $(id); node.replaceChildren();
  values.forEach(([value, label]) => node.append(option(value, label)));
  if (values.some(([value]) => value === oldValue)) node.value = oldValue;
}
function choices() {
  const pair = $("pair").value;
  const engine = $("engine").value;
  const old = $("recipient").value;
  const words = [...new Set(DATA.patches.filter((r) => r.pair_id === pair && r.engine === engine)
    .map((r) => r.recipient_key.split("_").at(-1)))];
  selectValues("recipient", words.map((w) => [w, w]), old);
}

function populate() {
  const pairs = [...new Set(DATA.patches.map((r) => r.pair_id))];
  const engines = [...new Set(DATA.patches.map((r) => r.engine))];
  selectValues("pair", pairs.map((p) => [p, p.replace("_", " ")]), pairs[0]);
  selectValues("engine", engines.map((e) => [e, e === "piper" ? "Piper" : "Kokoro"]), "piper");
  choices();
  $("pair").addEventListener("change", () => { choices(); render(); });
  $("engine").addEventListener("change", () => { choices(); render(); });
  ["recipient", "mode", "site", "stage"].forEach((id) => $(id).addEventListener("change", render));
}

function summary() {
  $("scope").textContent = DATA.scope;
  const patches = DATA.patches;
  const primaryW = patches.filter((r) => r.arm === "W" && r.primary_eligible);
  const primaryBySite = [...new Set(primaryW.map((r) => r.site))].map((site) => {
    const rows = primaryW.filter((r) => r.site === site);
    return {site, total: rows.length, transfers: rows.filter((r) => r.target_only_transfer).length};
  });
  if (primaryBySite.length !== 2 || primaryBySite.some((s) => s.total !== 4 || s.transfers !== 4)) {
    throw new Error("Embedded primary-site counts differ from independent review");
  }
  const random = patches.filter((r) => r.arm === "R1" || r.arm === "R2");
  const randomPrimary = random.filter((r) => r.primary_eligible);
  const donorText = (r) => BASE.get(`${r.mode}|${r.donor_key}`).transcript;
  const recipientText = (r) => BASE.get(`${r.mode}|${r.recipient_key}`).transcript;
  const cards = [
    ["Source recordings", String(Object.keys(DATA.clips).length), "Two paired words × two synthetic voices × two pairs"],
    ["Audited patch outputs", String(patches.length + DATA.complement_patches.length),
      "224 original word-window + 192 factorial; same FIT corpus and two sites"],
    ["Primary W transfer", "4/4 per site", "Same four direction/mode cells at two sites; one word pair and voice; S/C correlated"],
    ["Random transcript changes", `${random.filter((r) => r.transcript !== recipientText(r)).length}/${random.length}`, `${randomPrimary.filter((r) => r.target_only_transfer).length}/${randomPrimary.length} primary random edits made the donor-word swap`],
  ];
  const panel = $("cards"); panel.replaceChildren();
  for (const [label, number, detail] of cards) {
    const card = document.createElement("div"); card.className = "card";
    const l = document.createElement("span"); l.className = "label"; l.textContent = label;
    const n = document.createElement("strong"); n.textContent = number;
    const d = document.createElement("span"); d.className = "small"; d.textContent = detail;
    card.append(l, n, d); panel.append(card);
  }
  if (patches.filter((r) => r.arm === "W" && r.transcript === donorText(r)).length !== 32) {
    throw new Error("Embedded donor-transcript summary differs");
  }
  const dl = $("provenance"); dl.replaceChildren();
  for (const [key, value] of Object.entries({model: DATA.model, revision: DATA.revision, ...DATA.provenance})) {
    const dt = document.createElement("dt"); dt.textContent = key.replaceAll("_", " ");
    const dd = document.createElement("dd"); dd.className = "mono"; dd.textContent = String(value);
    dl.append(dt, dd);
  }
}

function chart(pair, engine, stage) {
  const s = DATA.traces[`${pair}|${engine}|S`][stage];
  const c = DATA.traces[`${pair}|${engine}|C`][stage];
  if (s.length !== c.length || s.points.length !== c.points.length) throw new Error("Trace axes differ");
  const root = $("trace-plot"); root.replaceChildren();
  const x0 = 52, x1 = 878, y0 = 20, y1 = 220;
  const max = Math.max(s.full_maxabs, c.full_maxabs, 1e-12);
  const window = s.nominal_window;
  const wx = x0 + (x1 - x0) * window[0] / s.length;
  const ww = (x1 - x0) * (window[1] - window[0]) / s.length;
  root.append(svg("rect", {x:wx, y:y0, width:ww, height:y1-y0, fill:"#506a78", opacity:"0.37"}));
  for (const frac of [0, .5, 1]) {
    const y = y1 - frac * (y1 - y0);
    root.append(svg("line", {x1:x0, x2:x1, y1:y, y2:y, stroke:"#3a5361", "stroke-width":1}));
    root.append(svg("text", {x:5, y:y+4, fill:"#a9c0ca", "font-size":11}, fixed(frac * max, 2)));
  }
  for (const [trace, color] of [[s, "#f5b76f"], [c, "#68d8d6"]]) {
    const points = trace.points.map((v, i) => {
      const x = x0 + (x1-x0) * (i+.5) / trace.points.length;
      const y = y1 - Math.min(v/max, 1) * (y1-y0);
      return `${fixed(x,2)},${fixed(y,2)}`;
    }).join(" ");
    root.append(svg("polyline", {points, fill:"none", stroke:color,
      "stroke-width":2, "stroke-linejoin":"round", "stroke-linecap":"round"}));
  }
  root.append(svg("text", {x:x0, y:247, fill:"#a9c0ca", "font-size":12}, "Start"));
  root.append(svg("text", {x:x1-53, y:247, fill:"#a9c0ca", "font-size":12}, "End"));
  const meta = $("trace-meta"); meta.replaceChildren();
  for (const [mode, trace] of [["S · stock", s], ["C · shared statistics", c]]) {
    const card = document.createElement("div");
    const title = document.createElement("strong"); title.textContent = mode;
    const p = document.createElement("p"); p.className = "small";
    p.textContent = `First nonzero row: ${trace.first_nonzero_index}; exact maximum outside word: ${fixed(trace.offslot_maxabs, 6)}; full maximum: ${fixed(trace.full_maxabs, 6)}.`;
    card.append(title, p); meta.append(card);
  }
}

function table(rows, recipientText, donorText) {
  const body = $("patch-rows"); body.replaceChildren();
  for (const arm of ARM_ORDER) {
    const row = rows.find((r) => r.arm === arm);
    if (!row) throw new Error(`Missing arm ${arm}`);
    const tr = document.createElement("tr");
    function cell(value, className = "") {
      const td = document.createElement("td"); td.textContent = String(value);
      if (className) td.className = className;
      tr.append(td); return td;
    }
    cell(ARM_LABEL[arm]); cell(row.transcript, "output");
    const relation = row.transcript === donorText ? "Matches donor transcript"
      : row.transcript === recipientText ? "Matches recipient transcript" : "Different transcript";
    cell(relation);
    const intended = document.createElement("td");
    intended.append(row.primary_eligible
      ? badge(row.target_only_transfer ? "Yes" : "No", row.target_only_transfer ? "good" : "neutral")
      : badge("Not eligible", "warn")); tr.append(intended);
    const control = document.createElement("td");
    control.append(row.control_pass === null ? document.createTextNode("—")
      : badge(row.control_pass ? "PASS" : "FAIL", row.control_pass ? "good" : "warn"));
    tr.append(control);
    const windowKey = arm === "self_W" || arm.startsWith("R") ? "W" : arm;
    const support = arm === "full_donor" ? "all rows" : `${windowKey} [${row.windows[windowKey].join(", ")})`;
    cell(`${support}; L2 ${fixed(row.delta_l2, 2)}`, "mono");
    cell(row.changed_frames_vs_recipient, "mono");
    body.append(tr);
  }
}

function factorial(rows, recipientText, donorText) {
  const body = $("factorial-rows"); body.replaceChildren();
  const arms = [
    ["self_C", "Recipient", "Recipient"],
    ["W", "Donor", "Recipient"],
    ["C", "Recipient", "Donor"],
    ["full_donor", "Donor", "Donor"]
  ];
  for (const [arm, word, outside] of arms) {
    const row = rows.find((r) => r.arm === arm);
    if (!row) throw new Error(`Missing factorial arm ${arm}`);
    const tr = document.createElement("tr");
    const relation = row.transcript === donorText ? "Matches donor transcript"
      : row.transcript === recipientText ? "Matches recipient transcript" : "Different transcript";
    for (const value of [word, outside, row.transcript, relation, fixed(row.delta_l2, 2)]) {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    }
    body.append(tr);
  }
  const outside = rows.find((r) => r.arm === "C");
  const word = rows.find((r) => r.arm === "W");
  const random = rows.filter((r) => r.arm === "R1_C" || r.arm === "R2_C");
  if (random.length !== 2 || random.some((r) => r.transcript !== recipientText)) {
    throw new Error("Factorial random-outside controls differ from reviewed result");
  }
  $("factorial-note").textContent = `The two random outside-region edits kept the recipient transcript in this selected cell. Natural W edit L2 ${fixed(word.natural_w_l2, 2)}; natural O edit L2 ${fixed(outside.natural_c_l2, 2)}. The W and O edits have different sizes. Outside-only failure shows lack of sufficiency under this patch, not absence of word information outside W.`;
}

function render() {
  const pair = $("pair").value, engine = $("engine").value,
    recipient = $("recipient").value, mode = $("mode").value,
    site = $("site").value, stage = $("stage").value;
  const key = `${pair}_${engine}_${recipient}`;
  const rows = DATA.patches.filter((r) => r.pair_id === pair && r.engine === engine
    && r.mode === mode && r.recipient_key === key && r.site === site);
  if (rows.length !== 7) throw new Error("Selected patch matrix incomplete");
  const donorKey = rows[0].donor_key;
  const rc = DATA.clips[key], dc = DATA.clips[donorKey];
  const rb = BASE.get(`${mode}|${key}`), db = BASE.get(`${mode}|${donorKey}`);
  if (!rc || !dc || !rb || !db) throw new Error("Selected clip/baseline absent");
  $("recipient-audio").src = new URL(rc.path, document.baseURI).href;
  $("donor-audio").src = new URL(dc.path, document.baseURI).href;
  $("recipient-word").textContent = `Reference word: ${recipient}; clean ASR ${rb.eligible ? "matched" : "misheard"}`;
  $("donor-word").textContent = `Reference word: ${dc.word}; clean ASR ${db.eligible ? "matched" : "misheard"}`;
  $("recipient-text").textContent = rb.transcript;
  $("donor-text").textContent = db.transcript;
  const slot = rc.slot_samples.map((v) => fixed(v/16000, 2));
  $("selection-summary").textContent = `Only the word slot differs in the paired raw PCM. Slot: ${slot[0]}–${slot[1]} seconds. Primary intended-word eligible: ${rows[0].primary_eligible ? "yes" : "no"}. Selected model input mode: ${mode}.`;
  $("patch-context").textContent = `Recipient clean transcription: “${rb.transcript}” Paired donor clean transcription: “${db.transcript}” The patch changes an internal state, while the recipient waveform stays fixed.`;
  chart(pair, engine, stage);
  table(rows, rb.transcript, db.transcript);
  const complementRows = DATA.complement_patches.filter((r) => r.pair_id === pair
    && r.engine === engine && r.mode === mode && r.recipient_key === key && r.site === site);
  if (complementRows.length !== 6) throw new Error("Selected factorial matrix incomplete");
  factorial(complementRows, rb.transcript, db.transcript);
}

populate(); summary(); render();
