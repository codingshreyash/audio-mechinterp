'use strict';

const byId = (name) => document.getElementById(name);
const state = { ids: [], index: 0, plays: 0, completed: new Set() };

function currentId() { return state.ids[state.index]; }

function showClip() {
  const id = currentId();
  if (!id) return;
  state.plays = 0;
  byId('audio').pause();
  byId('audio').src = '/audio/' + encodeURIComponent(id);
  byId('audio').load();
  byId('position').textContent = `Clip ${state.index + 1} of ${state.ids.length}` +
    (state.completed.has(id) ? ' · initial review already saved for this stage' : '');
  byId('opaque').textContent = 'Blind ID: ' + id;
  byId('plays').textContent = 'Plays recorded: 0';
  byId('word').value = '';
  byId('note').value = '';
  byId('clear').checked = false;
  byId('confidence').value = '1';
  byId('status').textContent = '';
}

async function loadProgress() {
  const reviewer = byId('reviewer').value.trim();
  const stage = byId('stage').value;
  if (!/^[A-Za-z0-9_.-]{3,64}$/.test(reviewer)) {
    state.completed = new Set();
    showClip();
    return;
  }
  const url = '/api/progress?stage=' + encodeURIComponent(stage) +
    '&reviewer_id=' + encodeURIComponent(reviewer);
  const response = await fetch(url, {cache: 'no-store'});
  if (!response.ok) throw new Error('Could not load saved progress.');
  state.completed = new Set((await response.json()).completed_ids);
  showClip();
}

function move(offset) {
  state.index = Math.max(0, Math.min(state.ids.length - 1, state.index + offset));
  showClip();
}

function nextUnreviewed() {
  const found = state.ids.findIndex((id, i) => i >= state.index && !state.completed.has(id));
  const fallback = state.ids.findIndex((id) => !state.completed.has(id));
  const next = found >= 0 ? found : fallback;
  byId('status').textContent = next >= 0 ? '' : 'Every clip has an initial response for this stage.';
  if (next >= 0) { state.index = next; showClip(); }
}

async function submitResponse(event) {
  event.preventDefault();
  const kind = byId('kind').value;
  const payload = {
    kind,
    stage: byId('stage').value,
    reviewer_id: byId('reviewer').value.trim(),
    opaque_id: currentId(),
    middle_word: byId('word').value.trim(),
    clear: byId('clear').checked,
    confidence: Number(byId('confidence').value),
    play_count: state.plays,
    note: byId('note').value
  };
  if (kind !== 'initial') {
    payload.prior_sequence = Number(byId('prior').value);
    payload.reason = byId('reason').value.trim();
  }
  const response = await fetch('/api/response', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload), cache: 'no-store'
  });
  if (!response.ok) {
    byId('status').textContent = 'Response rejected. Check the listener, review stage, play count, prior record, and duplicate status.';
    return;
  }
  const receipt = await response.json();
  if (kind === 'initial') state.completed.add(currentId());
  byId('status').textContent = 'Saved append-only record ' + receipt.sequence + '.';
  if (kind === 'initial' && state.index + 1 < state.ids.length) {
    setTimeout(() => move(1), 400);
  }
}

async function init() {
  const response = await fetch('/api/queue', {cache: 'no-store'});
  if (!response.ok) throw new Error('Could not load the frozen queue.');
  state.ids = (await response.json()).opaque_ids_in_order;
  byId('audio').addEventListener('play', () => {
    state.plays += 1;
    byId('plays').textContent = 'Plays recorded: ' + state.plays;
  });
  byId('previous').addEventListener('click', () => move(-1));
  byId('next').addEventListener('click', () => move(1));
  byId('next-unreviewed').addEventListener('click', nextUnreviewed);
  byId('reviewer').addEventListener('change', () => loadProgress().catch(showError));
  byId('stage').addEventListener('change', () => loadProgress().catch(showError));
  byId('response').addEventListener('submit', (event) => {
    submitResponse(event).catch(showError);
  });
  showClip();
}

function showError(error) { byId('status').textContent = error.message; }
init().catch(showError);
