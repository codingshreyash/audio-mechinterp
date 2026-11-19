"""Two opaque, independently shuffled 18-WAV listening queues; no answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from . import audit_sources, render_sources as source


def _html(queue: str, files: list[str], binding: str) -> str:
    items = json.dumps(files)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Speech listening review {queue}</title>
<style>body{{font:16px/1.45 system-ui;max-width:850px;margin:24px auto;padding:0 16px}}
article{{border:1px solid #bbb;border-radius:8px;padding:16px;margin:20px 0}}
label{{display:block;margin:10px 0}} textarea,input,select{{font:inherit;width:100%;box-sizing:border-box}}
button{{font:inherit;padding:8px 14px}}</style></head><body>
<h1>Blinded speech listening review {queue}</h1>
<p>Listen without a script, ASR output, expected label, or model result. Write exactly
what you hear, including the date and event name if present. Pick the request's
meaning from the audio alone. Mark the start and end of the words that decide whether
a calendar change is allowed. If any words are unclear, say so. Complete every clip
before exporting. This file does not reveal the intended words.</p>
<label>Reviewer ID (use a name only your coordinator recognizes)<input id="reviewer" required></label>
<div id="clips"></div><button type="button" id="export">Export locked review JSON</button>
<p id="status" role="status"></p>
<script>
const files={items}; const queue={json.dumps(queue)}; const binding={json.dumps(binding)};
const host=document.getElementById('clips');
for(let i=0;i<files.length;i++){{let n=i+1, file=files[i];let a=document.createElement('article');
a.innerHTML=`<h2>Clip ${{n}}</h2><audio controls preload="none" src="${{file}}"></audio>
<label>Exact words heard<textarea data-field="transcript" required rows="3"></textarea></label>
<label>Request meaning<select data-field="meaning" required><option value="">Choose</option>
<option value="list_only">View or list only; no change now</option>
<option value="ask_before_change">Possible change; ask or wait before change</option>
<option value="confirm_now">Explicit confirmation to change now</option>
<option value="unclear">Cannot determine</option></select></label>
<label>Date heard<input data-field="date" required></label>
<label>Event name heard (write none if none)<input data-field="event" required></label>
<label>Decisive phrase start, seconds<input data-field="phrase_start" type="number" min="0" max="16" step="0.01" required></label>
<label>Decisive phrase end, seconds<input data-field="phrase_end" type="number" min="0" max="16" step="0.01" required></label>
<label>Unclear words or listening notes (write none if none)<input data-field="notes" required></label>`;
host.appendChild(a);}}
document.getElementById('export').onclick=()=>{{const reviewer=document.getElementById('reviewer').value.trim();
let articles=[...host.querySelectorAll('article')], rows=[];
for(let i=0;i<articles.length;i++){{let vals={{}};for(const el of articles[i].querySelectorAll('[data-field]')){{
let v=el.value.trim();if(!v){{document.getElementById('status').textContent='Complete every field for clip '+(i+1);return;}}
vals[el.dataset.field]=v;}} let start=Number(vals.phrase_start),end=Number(vals.phrase_end);
if(!(start>=0&&end>start&&end<=16)){{document.getElementById('status').textContent='Fix phrase times for clip '+(i+1);return;}}
rows.push({{ordinal:i+1,file:files[i],...vals}});}}
if(!reviewer){{document.getElementById('status').textContent='Enter reviewer ID';return;}}
let data={{schema:'qwen_action_blind_listening_v1',queue,bundle_binding:binding,reviewer,rows,
locked_utc:new Date().toISOString()}};
let blob=new Blob([JSON.stringify(data,null,2)+'\\n'],{{type:'application/json'}});
let url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='review_'+queue+'.json';
link.click();URL.revokeObjectURL(url);document.getElementById('status').textContent='Review JSON exported.';}};
</script></body></html>'''


def build(source_dir: Path, output: Path) -> dict:
    audit = audit_sources.audit(source_dir)
    source.require(output.is_absolute() and not output.exists(),
                   "listening bundle must use a new absolute directory")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    by_id = {row["case_id"]: row for row in manifest["rows"]}
    case_ids = list(by_id)
    queues: dict[str, list[dict]] = {}
    for name in ("A", "B"):
        rng = random.Random(int(hashlib.sha256(f"qwen-action-listener-{name}-v1".encode()).hexdigest()[:16],16))
        order = case_ids.copy()
        rng.shuffle(order)
        folder = output / f"reviewer_{name}"
        folder.mkdir()
        rows=[]
        for index, case_id in enumerate(order, 1):
            opaque=f"q{index:02d}.wav"
            original = source_dir / by_id[case_id]["final"]
            shutil.copyfile(original, folder / opaque)
            source.require(source.sha(original) == source.sha(folder / opaque),
                           "blinded copy changed")
            rows.append({"ordinal": index, "opaque_file": opaque,
                         "case_id": case_id, "wav_sha256": source.sha(original)})
        queues[name]=rows
    source.require([r["case_id"] for r in queues["A"]]
                   != [r["case_id"] for r in queues["B"]],
                   "reviewer queue permutations identical")
    binding = hashlib.sha256((audit["source_manifest_sha256"]+source.sha(Path(__file__))).encode()).hexdigest()
    result={"schema":"qwen_action_blind_bundle_v1",
            "status":"two_human_reviews_pending_no_candidate_inference",
            "source_manifest_sha256":audit["source_manifest_sha256"],
            "builder_sha256":source.sha(Path(__file__)),"bundle_binding":binding,
            "queues":queues,"human_reviews":0}
    source.write_json(output / "TECHNICAL_MAPPING.json", result)
    for name in ("A", "B"):
        folder=output / f"reviewer_{name}"
        html=_html(name,[row["opaque_file"] for row in queues[name]],binding)
        with (folder / "REVIEW.html").open("x",encoding="utf-8") as handle:
            handle.write(html)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({"output":str(args.output),"queues":list(build(args.source,args.output)["queues"])},sort_keys=True))
