"""Independent read-only CPU audit of a *completed* eight-record Voxtral FIT run.

Do not invoke until result.json exists. Does not load model weights or access
development/locked-test WAVs or candidate outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = HERE / "voxtral_fit_v2_causal_full.py"
FIRST_RUNNER = HERE / "voxtral_fit_v2_causal_first_record.py"
MASKS = HERE / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json"
SELECTION = HERE / "VOXTRAL_FIT_V2_SELECTION.json"
INDEX = HERE / "voxtral_corpus_v1_index.json"
PILOT = HERE / "runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/result.json"
FIRST_RESULT = HERE / "runs/voxtral_fit_v2_causal_first_record/20261118T091057373532Z/result.json"
DESIGN = HERE / "VOXTRAL_FIT_V2_CAUSAL_DESIGN.md"
GATE = HERE / "runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/v2_gate.json"
EXPECTED = {
    "runner": "4e0a17f80d70ca1498f2487d8cca81b20e36381bfeb35945807b2587b89373cd",
    "first_runner": "64d799bbb40b58b128b049877d18bde7774c2dc16717af97e692aa4531e907b5",
    "first_result": "7803ff057c309b64b603735ddf229a29ef9fc076594314c32758d1f5e2d176f4",
    "masks": "a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e",
    "selection": "f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b",
    "index": "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971",
    "pilot": "6cc84dc1a4ced9ce17380eab3836079c462c9974f31337f146416730b5536ccf",
    "design": "cd2cbd762af3568f85d43c2dec47824666a95d689a3dc89d9cbdf6471ef54ab0",
    "gate": "79622d883e954a5076afac2e97b3809833e7fff5ea89b4ae5ccbd137b819429f",
}
FILES = {"runner":RUNNER,"first_runner":FIRST_RUNNER,"first_result":FIRST_RESULT,
         "masks":MASKS,"selection":SELECTION,"index":INDEX,"pilot":PILOT,
         "design":DESIGN,"gate":GATE}
SITES = ("encoder31", "projector")
ARMS = ("restore", "loss")
PATCHES = ("whole","word_0","word_1","word_2","word_3","pre","post","padding","self","random")
CONTROLS = ("pre","post","padding","random")
CAP_BYTES = 30 * 1024**3
CAP_SECONDS = 45 * 60


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def near(a, b, tol=1e-5) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a,b=float(a),float(b)
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol*max(1.0,abs(a),abs(b))


def classify(ids: list[int], visible: str, special: set[int], recipient: str, opposite: str) -> str:
    if any(t in special and not (i==len(ids)-1 and t==2) for i,t in enumerate(ids)):
        return "malformed_special"
    if not ids or ids[-1]!=2:
        return "missing_eos"
    s=visible.strip()
    if not s or not s.isascii() or not s.isalpha():
        return "malformed_surface"
    s=s.casefold()
    return "recipient" if s==recipient.casefold() else "opposite" if s==opposite.casefold() else "other"


def slot(row: dict) -> tuple:
    if row["type"]=="capture":
        return ("capture",row["site"],row["word"],row["kind"])
    name=row["kind"]
    if name.startswith("patch:"):
        _,arm,patch=name.split(":",2)
        return ("patch",row["site"],row["recipient"],arm,patch)
    return ("generation",row["site"],row["recipient"],name)


def planned_slots(words: list[str]) -> set[tuple]:
    a,b=words
    keys=set()
    for site in SITES:
        for word in words:
            for kind in ("clean","blank"):
                keys.add(("capture",site,word,kind))
                keys.add(("generation",site,word,f"passive:{kind}"))
            keys.add(("generation",site,word,"opposite_word_control"))
            for arm in ARMS:
                for patch in PATCHES:
                    keys.add(("patch",site,word,arm,patch))
    for word in words:
        keys.add(("generation",None,word,"blank_baseline"))
    for name,w in ((f"native_{a}",a),(f"native_{b}",b),
                   (f"word_{a}_outside_{b}",a),(f"{a}_with_{b}_padding",a),
                   (f"word_{b}_outside_{a}",b),(f"{b}_with_{a}_padding",b)):
        keys.add(("generation",None,w,f"feature:{name}"))
    require(len(keys)==108,"internal 108-slot grammar error")
    return keys


def recompute_summary(rows: list[dict], case_ids: list[str]) -> dict:
    """Independent all-planned gate arithmetic, without runner.summarize()."""
    require(len(case_ids)==16 and len(set(case_ids))==16,"wrong directional case set")
    directions={k:{} for k in case_ids}
    for row in rows:
        if row["type"]=="capture":
            continue
        k=f"{row['pair_id']}|{row['engine']}|fit_chosen|{row['recipient']}"
        require(k in directions,"row outside frozen case set")
        s=(row["site"] or "none",row["kind"])
        require(s not in directions[k],"duplicate directional condition")
        directions[k][s]=row
    eligible={k:directions[k][("none","blank_baseline")]["classification"]!="recipient" for k in case_ids}
    n_eligible=sum(eligible.values())
    result={"planned_directions":16,"restoration_eligible":n_eligible,
            "restoration_eligibility_gate":n_eligible>=12,"sites":{}}
    valid={"recipient","opposite","other"}
    for site in SITES:
        result["sites"][site]={}
        for arm in ARMS:
            def success(k,name):
                label=directions[k][(site,f"patch:{arm}:{name}")]["classification"]
                return eligible[k] and label=="recipient" if arm=="restore" else label in ("opposite","other")
            whole=sum(bool(success(k,"whole")) for k in case_ids)
            quarters={f"word_{q}":sum(bool(success(k,f"word_{q}")) for k in case_ids) for q in range(4)}
            controls={name:sum(bool(success(k,name)) for k in case_ids) for name in CONTROLS}
            breakdown={}
            for pair_voice in dict.fromkeys("|".join(k.split("|")[:2]) for k in case_ids):
                subset=[k for k in case_ids if k.startswith(pair_voice+"|")]
                require(len(subset)==2,"pair/voice directional balance")
                breakdown[pair_voice]={"planned":2,"whole_success":sum(bool(success(k,"whole")) for k in subset)}
            invalid=sum(directions[k][(site,f"patch:{arm}:{name}")]["classification"]=="invalid"
                        for k in case_ids for name in ("whole",*CONTROLS))
            comparisons={}
            for control in CONTROLS:
                nvalid=wins=0
                for k in case_ids:
                    a=directions[k][(site,f"patch:{arm}:whole")]
                    b=directions[k][(site,f"patch:{arm}:{control}")]
                    if a["classification"] not in valid or b["classification"] not in valid:
                        continue
                    ca,cb=a["recipient_minus_opposite"],b["recipient_minus_opposite"]
                    if ca is None or cb is None or not math.isfinite(ca) or not math.isfinite(cb):
                        continue
                    nvalid+=1
                    wins+=(ca>cb if arm=="restore" else ca<cb)
                comparisons[control]={"valid_of_16":nvalid,"strict_wins_of_16":wins,
                                      "passes":nvalid>=14 and wins>=12}
            rate=whole/n_eligible if n_eligible else None
            effect=(whole>=12 and n_eligible>=12 and rate>=.75) if arm=="restore" else whole>=12
            time_gate=(invalid==0 and whole>max(controls.values()) and
                       all(v["passes"] for v in comparisons.values()))
            result["sites"][site][arm]={"whole_word_success_of_16":whole,
                "eligible_success_rate":rate if arm=="restore" else None,
                "whole_word_effect_gate":effect,"word_quarter_success_of_16":quarters,
                "pair_voice_breakdown":breakdown,"control_success_of_16":controls,
                "invalid_main_or_control_conditions":invalid,"score_comparisons":comparisons,
                "time_preference_gate":time_gate}
        counter=Counter(directions[k][(site,"opposite_word_control")]["classification"] for k in case_ids)
        result["sites"][site]["opposite_word_control_classes_of_16"]=dict(counter)
    return result


def same(a,b) -> bool:
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b,strict=True))
    if isinstance(a,float) or isinstance(b,float):
        return near(a,b)
    return a==b


def audit(run_dir: Path) -> dict:
    import torch
    import voxtral_fit_pilot_capability_v1 as base
    from transformers import AutoProcessor
    require((run_dir/"result.json").is_file(),"run incomplete; do not inspect partial outputs")
    for k,p in FILES.items():
        require(sha(p)==EXPECTED[k],f"frozen {k} SHA changed")
    meta=json.loads((run_dir/"meta.json").read_text())
    final=json.loads((run_dir/"result.json").read_text())
    masks=json.loads(MASKS.read_text())
    pilot=json.loads(PILOT.read_text())
    selection=json.loads(SELECTION.read_text())
    index=json.loads(INDEX.read_text())
    require(meta["script_sha256"]==EXPECTED["runner"] and
            meta["first_runner_sha256"]==EXPECTED["first_runner"] and
            meta["first_run_sha256"]==EXPECTED["first_result"] and
            meta["masks_sha256"]==EXPECTED["masks"] and
            meta["selection_sha256"]==EXPECTED["selection"] and
            meta["index_sha256"]==EXPECTED["index"] and
            meta["pilot_sha256"]==EXPECTED["pilot"] and
            meta["design_sha256"]==EXPECTED["design"] and
            meta["gate_sha256"]==EXPECTED["gate"],"run metadata SHA bindings mismatch")
    require(meta["snapshot"]==base.validate_snapshot(index) and
            meta["installed_sources"]==base.validate_transformers_source(),
            "checkpoint/installed source provenance mismatch")
    require(meta["planned"]=={"records":8,"directions":16,"generations":800,
                              "captures":64,"teacher_forced_forwards":1600},"planned workload changed")
    require(meta["memory_cap_bytes"]==CAP_BYTES and meta["wall_cap_seconds"]==CAP_SECONDS,
            "resource policy changed")
    require(final["status"]=="eight_record_fit_complete" and final["directions"]==16 and
            final["meta_sha256"]==sha(run_dir/"meta.json"),"terminal status/metadata mismatch")
    records=masks["records"]
    require(len(records)==8,"wrong frozen mask population")
    case_ids=[f"{r['pair_id']}|{r['engine']}|fit_chosen|{w}" for r in records for w in r["words"]]
    require(meta["case_ids"]==case_ids and len(set(case_ids))==16 and
            meta["record_keys"]==[f"{r['pair_id']}|{r['engine']}|{r['phrasing_id']}" for r in records],
            "directional FIT population mismatch")
    selected={(c["pair_id"],c["engine"],c["phrasing_id"],c["word"]):c for c in selection["clips"]}
    index_rows={(r["pair_id"],r["engine"],r["phrasing_id"]):r for r in index["rows"]}
    processor=AutoProcessor.from_pretrained(str(base.SNAPSHOT),local_files_only=True)
    require(processor.tokenizer.eos_token_id==2 and int(processor.audio_token_id)==24,
            "processor special IDs changed")
    all_rows=[]
    checkpoint_budgets=[]
    stamps=set()
    special=set(processor.tokenizer.all_special_ids)
    for ordinal,record in enumerate(records,1):
        pair,engine,template=record["pair_id"],record["engine"],record["phrasing_id"]
        require(template=="fit_chosen" and index_rows[pair,engine,template]["split"]=="fit_pool",
                "non-FIT record")
        cp_path=run_dir/f"record_{ordinal:02d}.checkpoint.json"
        cp=json.loads(cp_path.read_text())
        require(final["checkpoint_sha256"][str(ordinal)]==sha(cp_path),"final checkpoint hash mismatch")
        require(cp["ordinal"]==ordinal and cp["record_key"]==f"{pair}|{engine}|fit_chosen" and
                cp["run_meta_sha256"]==sha(run_dir/"meta.json") and
                cp["progress_records"]==108 and cp["capture_records"]==8 and
                cp["generation_records"]+cp["invalid_conditions"]==100 and
                cp["teacher_forced_forwards"]==2*cp["generation_records"],
                f"record {ordinal} checkpoint counts/provenance")
        require(cp["mps_driver_observed_max_bytes"]<CAP_BYTES and
                0<cp["cumulative_budget_seconds"]<=CAP_SECONDS and
                (not checkpoint_budgets or cp["cumulative_budget_seconds"]>checkpoint_budgets[-1]),
                f"record {ordinal} resource checkpoint")
        checkpoint_budgets.append(cp["cumulative_budget_seconds"])
        stamps.add(cp["invocation_stamp"])
        log=run_dir/cp["attempt_file"]
        require(sha(log)==cp["attempt_sha256"],f"record {ordinal} attempt hash")
        rows=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        require(len(rows)==108,f"record {ordinal} log length")
        keys=[slot(r) for r in rows]
        require(set(keys)==planned_slots(record["words"]) and len(set(keys))==108,
                f"record {ordinal} exact planned slots")
        captures={slot(r):r for r in rows if r["type"]=="capture"}
        gens=[r for r in rows if r["type"]=="generation"]
        invalid=[r for r in rows if r["type"]=="invalid_condition"]
        require(len(captures)==8 and len(gens)==cp["generation_records"] and
                len(invalid)==cp["invalid_conditions"],f"record {ordinal} row type counts")
        clip_by_word={c["word"]:c for c in record["clips"]}
        batches={}
        for word in record["words"]:
            clip=clip_by_word[word]
            require(clip["wav_sha256"]==selected[pair,engine,template,word]["wav_sha256"] and
                    sha(ROOT/clip["wav_path"])==clip["wav_sha256"],"FIT WAV bytes/selection mismatch")
            conv=[{"role":"user","content":[{"type":"audio","path":str((ROOT/clip["wav_path"]).resolve())},
                                             {"type":"text","text":base.PROMPT}]}]
            batches[word]=processor.apply_chat_template(conv,tokenize=True,return_dict=True,return_tensors="pt")
            saved=[p for p in pilot["records"] if (p["pair_id"],p["engine"],p["phrasing_id"],p["expected_word"])==
                   (pair,engine,template,word)]
            require(len(saved)==1 and saved[0]["classification"]["exact_expected_word_and_eos"] and
                    saved[0]["prompt_input_ids"]==batches[word]["input_ids"][0].tolist(),
                    "natural FIT endpoint or prompt mismatch")
        def tensor_sha(t):
            return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        mh=meta["feature_hashes"][f"{pair}|{engine}|fit_chosen"]
        feature={w:batches[w]["input_features"] for w in record["words"]}
        for w,native in feature.items():
            require(tensor_sha(native)==mh["native"][w],"native feature hash")
            blank=native.clone();blank[:,:,record["word_mel_frames_direct_overlap"]]=native[:,:,2999:3000]
            require(tensor_sha(blank)==mh["blank"][w],"blank feature hash")
        a,b=record["words"]
        fa,fb=feature[a],feature[b]
        mask=torch.zeros(3000,dtype=torch.bool);mask[record["word_mel_frames_direct_overlap"]]=True
        first_pad=min(i for i in range(3000) if 160*i-200>=clip_by_word[a]["frames"])
        conditions={f"native_{a}":fa,f"native_{b}":fb}
        for donor,outside,fd,fo in ((a,b,fa,fb),(b,a,fb,fa)):
            hybrid=fo.clone();hybrid[:,:,mask]=fd[:,:,mask]
            padding=fd.clone();padding[:,:,first_pad:]=fo[:,:,first_pad:]
            conditions[f"word_{donor}_outside_{outside}"]=hybrid
            conditions[f"{donor}_with_{outside}_padding"]=padding
        require({k:tensor_sha(v) for k,v in conditions.items()}==mh["six_conditions"],
                "six feature-control hashes")
        bykey={slot(r):r for r in rows}
        natural={w:next(p["generated_token_ids"] for p in pilot["records"] if
                        (p["pair_id"],p["engine"],p["phrasing_id"],p["expected_word"])==
                        (pair,engine,template,w)) for w in record["words"]}
        for row in rows:
            if row["type"]=="capture":
                site=row["site"]
                raw=[1,1500,1280] if site=="encoder31" else [375,3072]
                shape=raw[1:] if site=="encoder31" else raw
                require(row["hook_calls"]==1 and row["raw_hook_shape"]==raw and
                        row["stored_state_shape"]==shape and
                        row["raw_hook_dtype"]==row["stored_state_dtype"]=="torch.bfloat16" and
                        len(row["sha256"])==64,"capture hook/source state metadata")
                continue
            if row["type"]=="invalid_condition":
                require(row["classification"]=="invalid" and row["recipient_minus_opposite"] is None and
                        not row["edit"]["valid"],"invalid row semantics")
                continue
            ids=row["generated_token_ids"]
            visible=processor.batch_decode([ids],skip_special_tokens=True)[0]
            with_special=processor.batch_decode([ids],skip_special_tokens=False)[0]
            require(row["raw_decoded"]==visible and row["decoded_with_special_tokens"]==with_special and
                    row["classification"]==classify(ids,visible,special,row["recipient"],row["opposite"]) and
                    row["terminal_eos"]==bool(ids and ids[-1]==2),"suffix decode/classification")
            score=row["canonical_scores"]
            require(set(score)==set(record["words"]) and all(math.isfinite(float(v)) for v in score.values()) and
                    near(row["recipient_minus_opposite"],score[row["recipient"]]-score[row["opposite"]]),
                    "canonical candidate-score arithmetic")
            require(row["mps_driver_bytes_after"]<CAP_BYTES and row["seconds_including_scores"]>=0,
                    "generation resource fields")
            site=row["site"]
            if site is None:
                require(row["hook_calls_including_two_scores"]==0 and not row["hook_events"],
                        "unhooked generation hook events")
            else:
                raw=[1,1500,1280] if site=="encoder31" else [375,3072]
                require(row["hook_calls_including_two_scores"]==3 and len(row["hook_events"])==3 and
                        all(h=={"shape":raw,"dtype":"torch.bfloat16"} for h in row["hook_events"]),
                        "hook count/shape/dtype")
            word=row["recipient"]
            if row["kind"]==f"feature:native_{word}":
                require(ids==natural[word],"native replay parity")
            if row["kind"].startswith("passive:") or row["kind"].endswith(":self"):
                blank_base=(row["kind"]=="passive:blank" or row["kind"]=="patch:restore:self")
                target=(bykey[("generation",None,word,"blank_baseline")]["generated_token_ids"]
                        if blank_base else natural[word])
                require(ids==target,"passive/self baseline parity")
            if row["kind"].startswith("patch:"):
                _,arm,name=row["kind"].split(":",2)
                edit=row["edit"]
                smap=record["site_maps"][site]
                expected_rows=(smap["whole_word_rows"] if name in ("whole","self","random") else
                               smap["padding_control_rows"] if name=="padding" else smap["bins"][name])
                require(edit["mode"]==name and edit["rows"]==expected_rows and edit["valid"],
                        "patch row mask/mode")
                if name in CONTROLS:
                    target=float(edit["target_bf16_norm"])
                    require(target>0 and abs(edit["effective_bf16_norm"]-target)<=max(.05*target,1e-6)+1e-5 and
                            near(edit["alpha"],target/edit["unscaled_edit_norm"]),
                            "equal-norm BF16 control math")
                if name=="random":
                    label=f"voxtral-v2-causal-v1|{pair}|{engine}|fit_chosen|{word}|{site}|{arm}|random_0"
                    seed=int.from_bytes(hashlib.sha256(label.encode()).digest(),"big")&((1<<63)-1)
                    require(edit["random_seed_label"]==label and edit["random_seed"]==seed,
                            "random control seed")
                if name=="self":
                    require(edit["alpha"]==0 and edit["effective_bf16_norm"]==0,"self edit not zero")
                require(len(edit["edited_bf16_sha256"])==64,"edited rows SHA missing")
            if row["kind"]=="opposite_word_control":
                donor=bykey[("capture",site,row["opposite"],"clean")]
                require(row["edit"]["donor_sha256"]==donor["sha256"],"opposite donor hash")
        all_rows.extend(rows)
    require(final["progress_rows"]==len(all_rows)==864 and
            final["generations"]==sum(r["type"]=="generation" for r in all_rows) and
            final["captures"]==sum(r["type"]=="capture" for r in all_rows)==64 and
            final["invalid_conditions"]==sum(r["type"]=="invalid_condition" for r in all_rows) and
            final["generations"]+final["invalid_conditions"]==800,
            "full FIT workload/count arithmetic")
    actual_summary=recompute_summary(all_rows,case_ids)
    require(same(actual_summary,final["summary"]),"16-direction frozen gate arithmetic mismatch")
    memory_logs=list(run_dir.glob("invocation_*.memory.jsonl"))
    error_paths=list(run_dir.glob("invocation_*.error.json"))
    error_stamps={p.name.removeprefix("invocation_").removesuffix(".error.json") for p in error_paths}
    memory_stamps={p.name.removeprefix("invocation_").removesuffix(".memory.jsonl") for p in memory_logs}
    require(stamps.issubset(memory_stamps) and memory_stamps.issubset(stamps|error_stamps),
            "unaccounted invocation memory log")
    peak=max(int(r["mps_driver_observed_max_bytes"]) for r in
             (json.loads((run_dir/f"record_{i:02d}.checkpoint.json").read_text()) for i in range(1,9)))
    for path in memory_logs:
        for line in path.read_text().splitlines():
            if not line.strip():continue
            entry=json.loads(line)
            require(entry["driver_max_bytes"]<CAP_BYTES,"memory-log cap reached")
    debts=[]
    documented_attempts=set()
    for path in sorted(error_paths):
        error=json.loads(path.read_text())
        stamp=path.name.removeprefix("invocation_").removesuffix(".error.json")
        require(error["status"]=="incomplete_invocation" and
                error["memory_log_sha256"]==sha(run_dir/f"invocation_{stamp}.memory.jsonl") and
                math.isfinite(error["cumulative_budget_seconds"]),"error/debt receipt invalid")
        debts.append(error["cumulative_budget_seconds"])
        if error.get("attempt_file"):
            documented_attempts.add(error["attempt_file"])
    completed_attempts={json.loads((run_dir/f"record_{i:02d}.checkpoint.json").read_text())["attempt_file"]
                        for i in range(1,9)}
    all_attempts={p.name for p in run_dir.glob("record_*_attempt_*.jsonl")}
    require((all_attempts-completed_attempts).issubset(documented_attempts),
            "unaccounted partial attempt log")
    require(all(x<=CAP_SECONDS for x in checkpoint_budgets) and
            all(x<=y for x,y in zip(debts,debts[1:])),"cumulative time budget drift")
    return {"status":"passed_read_only_full_fit_audit","run_dir":str(run_dir),
            "meta_sha256":sha(run_dir/"meta.json"),"result_sha256":sha(run_dir/"result.json"),
            "records":8,"directions":16,"captures":final["captures"],
            "generations":final["generations"],"invalid_conditions":final["invalid_conditions"],
            "teacher_forced_forwards":2*final["generations"],
            "last_checkpoint_cumulative_seconds":checkpoint_budgets[-1],
            "peak_checkpoint_mps_driver_bytes":peak,"summary":actual_summary,
            "limits":"scalar model scores checked by saved arithmetic and frozen code, not recomputed by a second model forward"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.run_dir),indent=2))


if __name__=="__main__":
    main()
