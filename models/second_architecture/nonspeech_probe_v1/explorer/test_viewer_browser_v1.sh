#!/usr/bin/env bash
# Browser-level state regression. Requires the agent-browser CLI and local Chrome.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
command -v agent-browser >/dev/null
SESSION="$(agent-browser session id --scope worktree --prefix nonspeech-viewer-browser-test)"
trap 'agent-browser --session "$SESSION" close >/dev/null 2>&1 || true' EXIT

agent-browser --session "$SESSION" open "file://$HERE/descriptive_v1.html" >/dev/null
agent-browser --session "$SESSION" select '#duration' 8 >/dev/null
agent-browser --session "$SESSION" select '#family' beep_train >/dev/null
agent-browser --session "$SESSION" check '#changed' >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
  const rows = [...document.querySelectorAll('#rows tr')];
  const idsNow = rows.map(r => r.querySelector('button')?.textContent).filter(Boolean);
  const heading = document.querySelector('#detail h2')?.textContent;
  const marked = rows.filter(r => r.classList.contains('selected')).map(r => r.querySelector('button')?.textContent);
  if (JSON.stringify(idsNow) !== JSON.stringify(['d08_beep_train_high', 'd08_beep_train_low'])) throw Error('wrong filtered rows: ' + idsNow);
  if (heading !== idsNow[0] || JSON.stringify(marked) !== JSON.stringify([heading])) throw Error('hidden selection after filter: ' + heading + ' / ' + marked);
  return true;
})()
JS

agent-browser --session "$SESSION" click '#rows tr:nth-child(2) button' >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
  const heading = document.querySelector('#detail h2')?.textContent;
  const marked = [...document.querySelectorAll('#rows tr.selected button')].map(x => x.textContent);
  if (heading !== 'd08_beep_train_low' || JSON.stringify(marked) !== JSON.stringify([heading])) throw Error('row click did not select detail');
  return true;
})()
JS

agent-browser --session "$SESSION" select '#duration' 2 >/dev/null
agent-browser --session "$SESSION" select '#family' band_noise >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
  const rows = [...document.querySelectorAll('#rows tr')];
  const heading = document.querySelector('#detail h2')?.textContent;
  const detail = document.querySelector('#detail')?.textContent;
  if (rows.length !== 1 || rows[0].querySelector('button') || !rows[0].textContent.includes('No cases match')) throw Error('empty table state failed');
  if (heading !== 'Case detail' || !detail.includes('No cases match')) throw Error('stale detail in empty state');
  return true;
})()
JS

agent-browser --session "$SESSION" select '#family' all >/dev/null
agent-browser --session "$SESSION" uncheck '#changed' >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
  const visible = [...document.querySelectorAll('#rows tr button')].map(x => x.textContent);
  const heading = document.querySelector('#detail h2')?.textContent;
  const marked = [...document.querySelectorAll('#rows tr.selected button')].map(x => x.textContent);
  if (visible.length !== 10 || !visible.includes(heading) || JSON.stringify(marked) !== JSON.stringify([heading])) throw Error('selection did not recover after empty filter');
  return true;
})()
JS

echo 'PASS: browser filter, row click, empty state, and recovery'
