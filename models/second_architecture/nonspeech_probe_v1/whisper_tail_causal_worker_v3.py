"""Gated CPU FP32 Whisper encoder-state pilot; importing loads no weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Any

from . import natural_tail_runner_v4 as v4
from . import whisper_tail_causal_plan_v3 as plan_source

HERE=Path(__file__).resolve().parent
ROOT=v4.ROOT
RUNS=HERE/"runs/whisper_tail_causal_v3"
ATTEMPTS={"whisper_multilingual":RUNS/"whisper_multilingual_attempt_001"}
ADMISSIONS={"whisper_multilingual":HERE/"WHISPER_TAIL_CAUSAL_MULTILINGUAL_ADMISSION_V3.json"}
GOS={"whisper_multilingual":HERE/"WHISPER_TAIL_CAUSAL_MULTILINGUAL_GO_V3.json"}
REVIEW=HERE/"WHISPER_TAIL_CAUSAL_EXECUTABLE_REVIEW_V3.md"
SOURCES=("whisper_tail_causal_worker_v3.py","whisper_tail_causal_parent_v3.py",
         "whisper_tail_causal_saved_audit_v3.py","whisper_tail_causal_host_v1.py",
         "whisper_tail_causal_plan_v3.py","test_whisper_tail_causal_plan_v3.py",
         "test_whisper_tail_causal_v3.py","test_whisper_tail_causal_host_independent_v1.py",
         "test_whisper_tail_causal_parent_monitor_independent_v1.py",
         "test_whisper_tail_causal_saved_fixture_v3.py",
         "test_whisper_tail_causal_go_history_v3.py",
         "WHISPER_TAIL_CAUSAL_CROSS_FAMILY_PROSPECTIVE_V3.md",
         "WHISPER_TAIL_CAUSAL_PLAN_V3.json","natural_tail_runner_v4.py",
         "descriptive_worker_v1.py",
         "whisper_tail_causal_worker_v1.py","whisper_tail_causal_parent_v1.py",
         "whisper_tail_causal_saved_audit_v1.py","whisper_tail_causal_plan_v1.py",
         "WHISPER_TAIL_CAUSAL_MULTILINGUAL_ADMISSION_V1.json",
         "WHISPER_TAIL_CAUSAL_MULTILINGUAL_GO_V1.json",
         "runs/whisper_tail_causal_v1/whisper_multilingual_attempt_001_audit.json",
         "WHISPER_TAIL_CAUSAL_INTERPRETATION_INDEPENDENT_V1.md",
         "WHISPER_TAIL_CAUSAL_V1_INDEPENDENT_RESULT_REVIEW.md",
         "WHISPER_TAIL_CAUSAL_V1_RAW_FILE_INVENTORY.json",
         "whisper_tail_causal_worker_v2.py","whisper_tail_causal_parent_v2.py",
         "whisper_tail_causal_saved_audit_v2.py","whisper_tail_causal_plan_v2.py",
         "WHISPER_TAIL_CAUSAL_MULTILINGUAL_ADMISSION_V2.json",
         "WHISPER_TAIL_CAUSAL_MULTILINGUAL_GO_V2.json",
         "runs/whisper_tail_causal_v2/whisper_multilingual_attempt_001_audit.json",
         "runs/whisper_tail_causal_v2/whisper_multilingual_attempt_001_historical_audit.json",
         "WHISPER_TAIL_CAUSAL_V2_INDEPENDENT_RESULT_REVIEW.md",
         "WHISPER_TAIL_CAUSAL_V2_RAW_FILE_INVENTORY.json")
WALL_SECONDS=1800
RSS_BYTES=4*1024**3
EVIDENCE_BYTES=1024**3
POST_RESERVE_SECONDS=180


def require(ok:bool,message:str)->None:
    if not ok:raise ValueError(message)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_clean(path:Path)->None:
    require(path.is_file() and not path.is_symlink(),f"missing committed file: {path.name}")
    rel=str(path.relative_to(ROOT))
    require(subprocess.run(("git","ls-files","--error-unmatch","--",rel),
                           cwd=ROOT,capture_output=True).returncode==0
            and all(subprocess.run(("git","diff",*flags,"--quiet","--",rel),
                                   cwd=ROOT,capture_output=True).returncode==0
                    for flags in ((),("--cached",))),f"file not tracked/clean: {rel}")


def _commit(path:Path)->str:
    return subprocess.run(("git","log","-1","--format=%H","--",str(path)),
                          cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()


def _original_go_commit(path:Path)->str:
    """Use the first committed GO receipt, even after a byte-identical restore."""
    rel=str(path.relative_to(ROOT))
    result=subprocess.run(("git","log","--diff-filter=A","--reverse","--format=%H",
                           "--",rel),cwd=ROOT,text=True,capture_output=True,check=True)
    commits=result.stdout.splitlines()
    require(len(commits)==1,"GO receipt must have one creation commit")
    return commits[0]


def _commit_at(path:Path,revision:str)->str:
    rel=str(path.relative_to(ROOT))
    return subprocess.run(("git","log","-1","--format=%H",revision,"--",rel),
                          cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()


def _sha_at(path:Path,revision:str)->str:
    rel=str(path.relative_to(ROOT))
    blob=subprocess.run(("git","show",f"{revision}:{rel}"),cwd=ROOT,
                        capture_output=True,check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def _ancestor(before:str,after:str)->bool:
    return subprocess.run(("git","merge-base","--is-ancestor",before,after),
                          cwd=ROOT,check=False).returncode==0


def source_identity()->dict[str,str]:
    result={}
    for name in SOURCES:
        path=HERE/name
        _tracked_clean(path)
        result[name]=sha(path)
    return result


def admission(model:str)->dict:
    require(model in ATTEMPTS,"unknown checkpoint")
    frozen=json.loads(plan_source.PLAN_PATH.read_text())
    require(frozen==plan_source.build(),"source-only causal plan changed")
    return {"schema":"whisper_tail_causal_admission_v3","status":"source_only_no_weights",
            "model":model,"attempt":str(ATTEMPTS[model]),
            "plan_sha256":sha(plan_source.PLAN_PATH),
            "source_identity":source_identity(),"v4_admission":v4.admission(model),
            "limits":{"whole_wall_seconds":WALL_SECONDS,"tree_rss_bytes":RSS_BYTES,
                      "evidence_bytes":EVIDENCE_BYTES,"battery_start_percent":80,
                      "battery_abort_percent":70,"max_power_probe_gap_seconds":15}}


def validate_go(model:str,admission_path:Path,go_path:Path)->dict:
    require(model in ATTEMPTS and admission_path==ADMISSIONS[model]
            and go_path==GOS[model],"fixed causal admission/GO paths required")
    for path in (admission_path,go_path,REVIEW,plan_source.PLAN_PATH):_tracked_clean(path)
    expected=admission(model)
    require(json.loads(admission_path.read_text())==expected,"causal admission changed")
    require(source_identity()==expected["source_identity"],
            "current source bytes differ from admission")
    go=json.loads(go_path.read_text())
    require(go.get("schema")=="whisper_tail_causal_go_v3"
            and go.get("status")=="independent_executable_go"
            and go.get("model")==model and go.get("attempt")==str(ATTEMPTS[model])
            and go.get("admission_sha256")==sha(admission_path)
            and go.get("plan_sha256")==sha(plan_source.PLAN_PATH)
            and go.get("source_identity")==expected["source_identity"]
            and go.get("review_sha256")==sha(REVIEW),"causal GO binding invalid")
    go_commit=_original_go_commit(go_path)
    require(_sha_at(go_path,go_commit)==sha(go_path)
            and _sha_at(admission_path,go_commit)==sha(admission_path)
            and _sha_at(REVIEW,go_commit)==sha(REVIEW),
            "current GO/admission/review bytes differ from original GO tree")
    for name,digest in expected["source_identity"].items():
        require(_sha_at(HERE/name,go_commit)==digest,
                f"source bytes differ from original GO tree: {name}")
    source_commits=[_commit_at(HERE/name,go_commit) for name in SOURCES]
    candidate_commit=_commit_at(admission_path,go_commit)
    review_commit=_commit_at(REVIEW,go_commit)
    require(all(c and c!=review_commit and _ancestor(c,review_commit)
                for c in [*source_commits,candidate_commit])
            and review_commit!=go_commit and _ancestor(review_commit,go_commit),
            "independent causal source/admission/review/GO order invalid")
    return expected


def _save_array(path:Path,array:Any)->dict:
    import numpy as np
    value=np.asarray(array,dtype=np.float32)
    require(value.shape==(1500,384) and np.isfinite(value).all(),"encoder state invalid")
    with path.open("xb") as out:
        np.save(out,value,allow_pickle=False);out.flush();os.fsync(out.fileno())
    return {"file":path.name,"sha256":sha(path),"shape":[1500,384],"dtype":"float32"}


def _pcm(item:dict)->Any:
    import numpy as np
    path=v4.RENDER/item["wav"]
    require(path.is_file() and not path.is_symlink() and sha(path)==item["wav_sha256"],
            "frozen causal WAV changed")
    with wave.open(str(path),"rb") as audio:
        require((audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),
                 audio.getnframes())==(1,2,16000,item["frames"]),"WAV geometry changed")
        raw=audio.readframes(audio.getnframes())
    return np.frombuffer(raw,dtype="<i2")


def _site_module(model:Any,site:str)->tuple[Any,bool]:
    encoder=model.model.encoder
    return {"post_conv_pos":(encoder.layers[0],True),
            "post_block0":(encoder.layers[0],False),
            "post_final_norm":(encoder.layer_norm,False)}[site]


def run_one(model_name:str,processor:Any,model:Any,pcm:Any,sidecars:Path,
            stem:str,*,arm:dict|None,states:dict)->tuple[dict,dict,dict]:
    """Run a fresh generate call; patch only generation-phase encoder rows."""
    import numpy as np
    import torch
    phase=["generate"]
    original_detect=model.detect_language if model_name=="whisper_multilingual" else None
    if original_detect is not None:
        def detect(*args:Any,**kwargs:Any)->Any:
            phase[0]="detect"
            try:return original_detect(*args,**kwargs)
            finally:phase[0]="generate"
        model.detect_language=detect
    handles=[];observed={};patch_receipt={"hook_calls":0}
    sites=plan_source.SITES if arm is None else (arm["site"],)
    for site in sites:
        module,pre=_site_module(model,site)
        def intervene(tensor:Any,*,key:str=site)->Any:
            if phase[0]!="generate":return tensor
            require(tuple(tensor.shape)==(1,1500,384)
                    and tensor.dtype==torch.float32,"encoder hook geometry/dtype changed")
            original=tensor[0].detach().cpu().numpy().copy()
            if arm is None:
                require(key not in observed,"baseline encoder hook called twice")
                observed[key]=original
                return tensor
            patch_receipt["hook_calls"]+=1
            require(patch_receipt["hook_calls"]==1,"patch hook called twice")
            recipient=states[arm["recipient"]][key]
            donor=states[arm["donor"]][key]
            require(np.array_equal(original,recipient),"recipient state drifted from no-hook baseline")
            selected=arm["target_rows"] if arm["kind"] in ("target_transfer","identity_sham","full_donor") else arm["source_rows"]
            changed=tensor.clone()
            if arm["kind"] in ("target_transfer","full_donor"):
                values=donor[selected]
            elif arm["kind"]=="identity_sham":
                values=recipient[selected]
            elif arm["kind"]=="source_random":
                values=donor[selected]
            else:
                require(arm["kind"]=="norm_matched_shifted_delta", "unknown patch kind")
                delta=donor[arm["target_rows"]]-recipient[arm["target_rows"]]
                values=recipient[selected]+delta
            changed[0,selected]=torch.from_numpy(np.ascontiguousarray(values)).to(changed.device)
            untouched=np.ones(1500,dtype=bool);untouched[selected]=False
            require(torch.equal(changed[0,untouched],tensor[0,untouched]),
                    "unselected hook rows changed")
            patch_receipt["native_ref"]=_save_array(sidecars/f"{stem}_native.npy",original)
            patch_receipt["edited_ref"]=_save_array(
                sidecars/f"{stem}_edited.npy",changed[0].detach().cpu().numpy())
            actual=changed[0,selected].detach().cpu().numpy()-original[selected]
            patch_receipt.update({"inside_delta_l2":float(np.linalg.norm(actual.astype(np.float64))),
                "outside_maxabs":0.0,"selected_rows":selected,
                "donor_state_sha256":hashlib.sha256(donor.tobytes()).hexdigest(),
                "recipient_state_sha256":hashlib.sha256(recipient.tobytes()).hexdigest()})
            if arm["kind"]=="norm_matched_shifted_delta":
                intended=donor[arm["target_rows"]]-recipient[arm["target_rows"]]
                patch_receipt["target_delta_l2"]=float(np.linalg.norm(intended.astype(np.float64)))
                require(abs(patch_receipt["inside_delta_l2"]-patch_receipt["target_delta_l2"])<=1e-4,
                        "norm-matched control delta changed")
            return changed
        if pre:
            def hook_pre(_module:Any,args:tuple,*,key:str=site,operation=intervene)->Any:
                new=operation(args[0],key=key)
                return (new,*args[1:]) if new is not args[0] else None
            handles.append(module.register_forward_pre_hook(hook_pre))
        else:
            def hook_post(_module:Any,_args:tuple,out:Any,*,key:str=site,operation=intervene)->Any:
                value=out[0] if isinstance(out,(tuple,list)) else out
                new=operation(value,key=key)
                if new is value:return None
                if isinstance(out,tuple):return (new,*out[1:])
                if isinstance(out,list):return [new,*out[1:]]
                return new
            handles.append(module.register_forward_hook(hook_post))
    try:
        outcome=v4.infer_one(model_name,processor,model,pcm,sidecars,stem)
        require(all(key in observed for key in sites) if arm is None
                else patch_receipt["hook_calls"]==1,"encoder hook count invalid")
        return outcome,observed,patch_receipt
    finally:
        for handle in reversed(handles):handle.remove()
        if original_detect is not None:model.detect_language=original_detect


def run(model_name:str,attempt:Path,admission_path:Path,go_path:Path,fd:int)->None:
    require(attempt==ATTEMPTS[model_name] and attempt.is_dir() and not attempt.is_symlink(),
            "fixed parent-created causal attempt required")
    guard=v4.start_parent_guard(attempt,fd)
    guard.check()
    receipt=validate_go(model_name,admission_path,go_path)
    guard.check()
    start=json.loads((attempt/"launch_start.json").read_text())
    require(start.get("schema")=="whisper_tail_causal_parent_v3"
            and start.get("parent_pid")==guard.parent_pid
            and start.get("deadline_monotonic")==guard.deadline
            and start.get("admission_sha256")==sha(admission_path)
            and start.get("go_sha256")==sha(go_path),"bounded causal parent mismatch")
    import numpy as np
    import torch
    torch.set_num_threads(4)
    processor,model=v4.old._snapshot_load(model_name,receipt["v4_admission"]["assets"])
    guard.check()
    frozen=json.loads(plan_source.PLAN_PATH.read_text())
    unit=next(u for u in frozen["units"] if u["model"]==model_name)
    sidecars=attempt/"sidecars";sidecars.mkdir()
    progress=attempt/"progress.jsonl";records=attempt/"records.jsonl"
    states={};baseline={}
    calls=[{"type":"baseline","variant":v} for v in ("zero","noise")]+[
        {"type":"arm",**a} for a in frozen["arms"] if a["model"]==model_name]
    require(len(calls)==56,"causal call census changed")
    for index,call in enumerate(calls,1):
        guard.check()
        variant=call["variant"] if call["type"]=="baseline" else call["recipient"]
        pcm=_pcm(unit[variant])
        with progress.open("a") as out:
            out.write(json.dumps({"type":"attempt","index":index})+"\n");out.flush();os.fsync(out.fileno())
        outcome,captured,patch=run_one(model_name,processor,model,pcm,sidecars,
                                       f"{index:03d}",arm=None if call["type"]=="baseline" else call,
                                       states=states)
        if call["type"]=="baseline":
            expected=unit[variant]
            require(outcome["output"]["sequence_ids"]==expected["sequence_ids"]
                    and outcome["output"]["prefix_ids"]==expected["prefix_ids"]
                    and outcome["output"]["termination"]==expected["termination"],
                    "fresh baseline differs from V4")
            for key in ("mel","mask","first_raw_logits","first_processed_scores"):
                live=np.load(sidecars/outcome["sidecars"][key]["file"],allow_pickle=False)
                reference=v4.ATTEMPTS[model_name]/"sidecars"/expected[key]["file"]
                require(reference.is_file() and not reference.is_symlink()
                        and sha(reference)==expected[key]["sha256"],
                        "frozen V4 first-step sidecar changed")
                old=np.load(reference,allow_pickle=False)
                require(old.shape==live.shape and old.dtype==live.dtype,
                        "V4 first-step sidecar geometry changed")
                require(np.array_equal(live,old),"fresh first-step vector differs from V4")
            states[variant]=captured
            baseline[variant]=outcome
            state_refs={site:_save_array(sidecars/f"{variant}_{site}.npy",array)
                        for site,array in captured.items()}
        else:
            state_refs=None
            require(outcome["output"]["prefix_ids"]==baseline[variant]["output"]["prefix_ids"],
                    "generation-only patch changed detected-language prefix")
            require(outcome["output"]["language_detection"]
                    ==baseline[variant]["output"]["language_detection"],
                    "generation-only patch changed language detection")
        with records.open("a") as out:
            out.write(json.dumps({"type":"call","index":index,"plan":call,
                "output":outcome,"patch":patch,"state_refs":state_refs},sort_keys=True)+"\n")
            out.flush();os.fsync(out.fileno())
        with progress.open("a") as out:
            out.write(json.dumps({"type":"complete","index":index})+"\n");out.flush();os.fsync(out.fileno())
        guard.check()
    with records.open("a") as out:
        out.write(json.dumps({"type":"complete","calls":56})+"\n");out.flush();os.fsync(out.fileno())


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("preflight","run"))
    parser.add_argument("--model",choices=tuple(ATTEMPTS),required=True)
    parser.add_argument("--admission",type=Path,required=True)
    parser.add_argument("--go",type=Path)
    parser.add_argument("--attempt",type=Path)
    parser.add_argument("--heartbeat-fd",type=int)
    args=parser.parse_args()
    if args.command=="preflight":
        require(args.go is None and args.attempt is None and args.heartbeat_fd is None
                and args.admission==ADMISSIONS[args.model] and not args.admission.exists(),
                "fresh fixed source-only admission required")
        v4.write_once(args.admission,admission(args.model))
    else:
        require(args.go is not None and args.attempt is not None
                and args.heartbeat_fd is not None,"live parent required")
        run(args.model,args.attempt,args.admission,args.go,args.heartbeat_fd)


if __name__=="__main__":main()
