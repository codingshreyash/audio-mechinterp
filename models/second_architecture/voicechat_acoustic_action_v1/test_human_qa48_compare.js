"use strict";
// Fake DOM only: verifies stale localStorage cannot relabel a changed reference.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const script = fs.readFileSync(path.join(__dirname, "human_qa48_compare.js"), "utf8");
const baseRow = {
  opaque_id: "a".repeat(24), reference_script: "Original script",
  source_id: "A1__fake", arm: "clean", initial_verbatim: "first words",
  clean_initial_verbatim: "first words",
  clean_comparison_basis: "first_clean_verbatim",
  date_match: null, event_match: null, authorization_match: null,
  same_critical_meaning_as_clean: null, compared_at_utc: null
};
const template = {
  schema: "voicechat_acoustic_qa48_compare_v1", role: "primary",
  reviewer_id: "person_one", first_export_sha256: "b".repeat(64),
  records: [baseRow]
};
const oldDigest = "c".repeat(64);
const oldKey = `voicechat-qa48-compare|${oldDigest}|${template.first_export_sha256}|${template.reviewer_id}`;
const stored = [{
  ...baseRow, date_match:"yes", event_match:"yes",
  authorization_match:"yes", same_critical_meaning_as_clean:"yes",
  compared_at_utc:"2026-11-19T09:00:00Z"
}];

function run(page, storage) {
  const nodes = new Map();
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {
      textContent:"", value:"", classList:{toggle() {}}, addEventListener() {}
    });
    return nodes.get(id);
  }
  node("embedded-data").textContent = JSON.stringify(page);
  const reads = [];
  const context = {
    document:{getElementById:node},
    localStorage:{getItem(key) { reads.push(key); return storage[key] || null; }},
  };
  vm.runInNewContext(script, context, {filename:"human_qa48_compare.js"});
  return {error:node("error").textContent, reads};
}

const current = run({template, template_sha256:oldDigest},
                    {[oldKey]:JSON.stringify(stored)});
assert.equal(current.error, "");
assert.deepEqual(current.reads, [oldKey]);

const altered = JSON.parse(JSON.stringify(template));
altered.records[0].reference_script = "Changed script";
const sameDigestAttack = run({template:altered, template_sha256:oldDigest},
                             {[oldKey]:JSON.stringify(stored)});
assert.match(sameDigestAttack.error, /Saved comparison differs from template/);

const newDigest = "d".repeat(64);
const newTemplateRun = run({template:altered, template_sha256:newDigest},
                           {[oldKey]:JSON.stringify(stored)});
assert.equal(newTemplateRun.error, "");
assert.notEqual(newTemplateRun.reads[0], oldKey);
console.log("3 comparison resume checks passed");
