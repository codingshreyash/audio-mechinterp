#!/usr/bin/env bash
# Browser state and evidence rendering regression; local file only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v agent-browser >/dev/null
SESSION="$(agent-browser session id --scope worktree --prefix nonspeech-causal-viewer-test)"
trap 'agent-browser --session "$SESSION" close >/dev/null 2>&1 || true' EXIT
agent-browser --session "$SESSION" open "file://$HERE/causal_v1.html" >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
 const data=JSON.parse(document.querySelector('#causal-data').textContent);
 const rows=[...document.querySelectorAll('#case-rows tr')];
 if(rows.length!==18 || Object.keys(data.cases).length!==18) throw Error('missing case census');
 const donorRows=[...document.querySelectorAll('#donor-rows tr')];
 if(donorRows.length!==4) throw Error('missing donor control census');
 for(const [i,id,arm] of [[0,'d02_silence','no_hook'],[1,'d02_silence','read_only'],[2,'d08_silence','no_hook'],[3,'d08_silence','read_only']]) {
   const record=data.donors[id][arm],cells=[...donorRows[i].cells].map(x=>x.textContent);
   if(!cells[0].includes(id) || !cells[1].includes(record.call_name)) throw Error('wrong donor call '+i);
   if(!cells[2].includes(JSON.stringify(record.transcript)) || !cells[3].includes(JSON.stringify(record.generated_ids)) || !cells[4].includes(JSON.stringify(record.sequence_ids))) throw Error('donor output not visible '+i);
   if(!cells[5].includes('EOS '+record.eos_seen) || !cells[5].includes('censored '+record.censored) || !cells[6].includes(JSON.stringify(record.parity))) throw Error('donor termination or comparison flags hidden '+i);
 }
 if(rows.length*18+donorRows.length!==328) throw Error('328 call arithmetic failed');
 if(!document.querySelector('#detail h2')?.textContent.startsWith('d02_')) throw Error('initial detail absent');
 if(document.querySelectorAll('#detail audio').length!==2) throw Error('missing paired audio');
 if(document.querySelectorAll('#detail h3').length!==2) throw Error('missing stages');
 if(document.querySelectorAll('#detail table tbody tr').length!==18) throw Error('missing arms');
 if(!document.querySelector('#detail').textContent.includes('EOS')) throw Error('termination hidden');
 if(!document.querySelector('#hashes').textContent.includes(data.hashes.records_sha256)) throw Error('hash missing');
 return true;
})()
JS
agent-browser --session "$SESSION" select '#duration' d08 >/dev/null
agent-browser --session "$SESSION" select '#family' beep_train >/dev/null
agent-browser --session "$SESSION" select '#contrast' informative >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
 const ids=[...document.querySelectorAll('#case-rows tr button')].map(x=>x.textContent);
 if(JSON.stringify(ids)!==JSON.stringify(['d08_beep_train_high','d08_beep_train_low'])) throw Error('incorrect filter '+ids);
 if(document.querySelector('#detail h2').textContent!==ids[0]) throw Error('filtered selection stale');
 return true;
})()
JS
agent-browser --session "$SESSION" click '#case-rows tr:nth-child(2) button' >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
 if(document.querySelector('#detail h2').textContent!=='d08_beep_train_low') throw Error('row click failed');
 const data=JSON.parse(document.querySelector('#causal-data').textContent);
 const c=data.cases.d08_beep_train_low;
 if(!c.stages.A.random_1_both.censored || c.stages.A.random_1_both.eos_seen) throw Error('censor lost');
 if(c.mel_padding.padded_frames_with_difference_gt_1e_6!==2200) throw Error('padding caveat lost');
 return true;
})()
JS
agent-browser --session "$SESSION" select '#contrast' same >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
 if(document.querySelectorAll('#case-rows tr button').length) throw Error('empty filter failed');
 if(document.querySelector('#detail h2').textContent!=='Case detail') throw Error('stale hidden detail');
 return true;
})()
JS
agent-browser --session "$SESSION" select '#contrast' all >/dev/null
cat <<'JS' | agent-browser --session "$SESSION" eval --stdin >/dev/null
(() => {
 if(document.querySelectorAll('#case-rows tr button').length!==2) throw Error('filter recovery failed');
 if(document.querySelectorAll('#case-rows tr.selected').length!==1) throw Error('selection recovery failed');
 if(document.querySelectorAll('#donor-rows tr').length!==4) throw Error('donor controls lost under filter');
 return true;
})()
JS
echo 'PASS: causal viewer 328-call census, donor controls, stages, audio, hashes, filters, censor, and selection'
