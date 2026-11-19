"""Independent saved-data auditor for the exploratory Wav2Vec2 dev matrix.

No model weights are loaded. It reconstructs every patch from saved clean
states, rehashes every sidecar and re-decodes every saved complete CTC output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import wav2vec2_dev_word_disjoint_v1 as runner

HEX=re.compile(r"[0-9a-f]{64}\Z")


def require(ok:bool,message:str)->None:
    if not ok:raise ValueError(message)


def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def token_list(value:str)->list[str]:
    # Deliberately do not call the runner's transcript parser.
    return re.sub(r"[\W_]+"," ",unicodedata.normalize("NFKC",value).casefold()).split()


def unique(pairs:list[tuple[str,Any]])->dict[str,Any]:
    out={}
    for k,v in pairs:
        require(k not in out,"duplicate JSON key")
        out[k]=v
    return out


def parse(raw:str)->Any:
    return json.loads(raw,object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def array(output:Path,relative:str,digest:str,shape:tuple[int,...])->np.ndarray:
    require(isinstance(relative,str) and relative.startswith("captures/")
            and ".." not in Path(relative).parts and isinstance(digest,str)
            and HEX.fullmatch(digest) is not None,"invalid sidecar reference")
    path=output/relative
    require(path.resolve().is_relative_to(output.resolve()) and path.is_file()
            and not path.is_symlink() and sha(path)==digest,"sidecar hash/path invalid")
    x=np.load(path,allow_pickle=False)
    require(x.dtype==np.float32 and tuple(x.shape)==shape and np.isfinite(x).all(),
            "sidecar shape/dtype/finite invalid")
    return x


def patch_independent(recipient:np.ndarray,donor:np.ndarray,
                      window:tuple[int,int],arm:str,seed:int|None)->np.ndarray:
    require(recipient.shape==donor.shape and recipient.dtype==donor.dtype==np.float32,
            "clean state mismatch")
    a,b=window
    x=recipient.copy()
    if arm=="self":return x
    if arm=="full":return donor.copy()
    if arm=="W":x[:,a:b]=donor[:,a:b]
    elif arm=="O":
        x[:,:a]=donor[:,:a];x[:,b:]=donor[:,b:]
    elif arm in ("random_W","random_O"):
        require(type(seed) is int,"seed missing")
        if arm=="random_W":
            r=recipient[:,a:b];d=donor[:,a:b]
        else:
            r=np.concatenate((recipient[:,:a],recipient[:,b:]),axis=1)
            d=np.concatenate((donor[:,:a],donor[:,b:]),axis=1)
        radius=float(np.linalg.norm(d.astype(np.float64)-r.astype(np.float64)))
        z=np.random.Generator(np.random.PCG64(seed)).standard_normal(r.shape)
        z_norm=float(np.linalg.norm(z))
        require(radius>0 and z_norm>0 and math.isfinite(radius) and math.isfinite(z_norm),
                "random radius invalid")
        edited=r+(z*(radius/z_norm)).astype(np.float32)
        if arm=="random_W":x[:,a:b]=edited
        else:
            x[:,:a]=edited[:,:a];x[:,b:]=edited[:,a:]
    else:raise ValueError("arm invalid")
    return x


def tensor_sha(array:np.ndarray)->str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def audit(output:Path,preflight_path:Path,freeze_path:Path)->dict[str,Any]:
    from transformers import Wav2Vec2Processor
    current=runner.preflight()
    require(runner.canonical(parse(preflight_path.read_text()))==runner.canonical(current),
            "preflight drift")
    freeze_hash=runner.verify_freeze(freeze_path,preflight_path,current)
    records=output/"records.jsonl"
    require(records.is_file() and not records.is_symlink(),"records absent")
    raw=records.read_bytes()
    require(raw.endswith(b"\n"),"truncated records")
    rows=[parse(line) for line in raw.decode().splitlines()]
    require(len(rows)==674 and rows[0].get("type")=="preflight"
            and rows[-1].get("type")=="complete", "matrix row count incomplete")
    require(rows[0].get("receipt")==current
            and rows[0].get("preflight_sha256")==sha(preflight_path)
            and rows[0].get("freeze_sha256")==freeze_hash,"run binding differs")
    require(rows[-1].get("clean_rows")==96 and rows[-1].get("patch_rows")==576
            and not any(x.get("type")=="failure" for x in rows),"run incomplete")
    cases,stats,refs=runner.trusted_dev_cases()
    expected_clean=[(c.key,m) for c in cases for m in runner.base.MODES]
    clean_rows=rows[1:97]
    require([(x.get("case_key"),x.get("mode")) for x in clean_rows]==expected_clean,
            "clean row order/coverage differs")
    processor=Wav2Vec2Processor.from_pretrained(runner.base.SNAPSHOT,local_files_only=True)
    expected_files={"records.jsonl"}
    clean={}
    for row in clean_rows:
        require(row.get("type")=="clean" and row.get("schema")==runner.SCHEMA
                and row.get("hook_count")==1,"clean row/hook invalid")
        case=next(c for c in cases if c.key==row["case_key"])
        mode=row["mode"]
        require(row.get("pair_id")==case.pair_id and row.get("engine")==case.engine
                and row.get("word")==case.word,"clean identity differs")
        t=(case.frames-400)//320+1
        shape={"input":(1,case.frames),"state":(1,t,768),"logits":(1,t,32)}
        captures={}
        for stage,expected_shape in shape.items():
            rel=f"captures/{mode}_{case.key}_{stage}.npy"
            require(row["paths"].get(stage)==rel
                    and row["shape"].get(stage)==list(expected_shape),"clean sidecar receipt differs")
            captures[stage]=array(output,rel,row["sha256"][stage],expected_shape)
            expected_files.add(rel)
        pcm=runner.base.load_pcm(case)
        reconstructed=runner.base.normalized_input(pcm,mode=mode,
                         stats=stats[(case.pair_id,case.engine)],processor=processor)
        require(np.array_equal(captures["input"][0],reconstructed),"processor input differs")
        ids=captures["logits"].argmax(axis=-1)[0].astype(np.int64).tolist()
        transcript=processor.batch_decode([ids])[0]
        words=token_list(transcript)
        require(row.get("argmax_ids")==ids and row.get("transcript")==transcript
                and row.get("tokens")==words and row.get("reference_tokens")==refs[case.key]
                and row.get("eligible") is (words==refs[case.key]),
                "clean CTC/full-template result differs")
        clean[(case.key,mode)]=(row,captures)
    grouped:dict[tuple[str,str],list[runner.base.Case]]={}
    for case in cases:
        grouped.setdefault((case.pair_id,case.engine),[]).append(case)
    for pair in grouped.values():
        require(len(pair)==2,"paired case coverage invalid")
        first=clean[(pair[0].key,"C")][1]["input"][0]
        second=clean[(pair[1].key,"C")][1]["input"][0]
        s,e=pair[0].slot
        require(np.array_equal(first[:s],second[:s])
                and np.array_equal(first[e:],second[e:]),
                "common-normalized inputs differ outside word slot")
    patch_rows=rows[97:-1]
    expected_patch=[(d["index"],arm) for d in current["directions"] for arm in runner.ARMS]
    require([(r.get("direction"),r.get("arm")) for r in patch_rows]==expected_patch,
            "patch row order/coverage differs")
    failures=[];counts=Counter()
    for row in patch_rows:
        require(row.get("type")=="patch" and row.get("schema")==runner.SCHEMA
                and row.get("site")==runner.SITE and row.get("hook_count")==1,
                "patch identity/hook invalid")
        d=current["directions"][row["direction"]]
        arm=row["arm"];mode=d["mode"]
        require(row.get("recipient_key")==d["recipient_key"]
                and row.get("donor_key")==d["donor_key"]
                and row.get("mode")==mode,"direction identity differs")
        r,rc=clean[(d["recipient_key"],mode)]
        donor,dc=clean[(d["donor_key"],mode)]
        s,e=d["slot"];t=rc["state"].shape[1]
        a,b=math.ceil((s-399)/320),(e-1)//320+1
        require(0<a<b<t and row.get("window")==[a,b],"window differs")
        seed=(runner.SEED_BASE+2*d["index"]+(0 if arm=="random_W" else 1)
              if arm in ("random_W","random_O") else None)
        require(row.get("seed")==seed,"seed differs")
        expected=patch_independent(rc["state"],dc["state"],(a,b),arm,seed)
        require(row.get("patch_tensor_sha256")==tensor_sha(expected),
                "patch hash differs from independent reconstruction")
        rw=rc["state"][:,a:b];dw=dc["state"][:,a:b]
        ro=np.concatenate((rc["state"][:,:a],rc["state"][:,b:]),axis=1)
        do=np.concatenate((dc["state"][:,:a],dc["state"][:,b:]),axis=1)
        norms={"natural_w_l2":float(np.linalg.norm(dw.astype(np.float64)-rw.astype(np.float64))),
               "natural_o_l2":float(np.linalg.norm(do.astype(np.float64)-ro.astype(np.float64))),
               "realized_l2":float(np.linalg.norm(expected.astype(np.float64)-rc["state"].astype(np.float64)))}
        require(all(math.isclose(row["norms"].get(k),v,rel_tol=1e-10,abs_tol=1e-10)
                    for k,v in norms.items()),"edit norm differs")
        rel=f"captures/d{d['index']:02d}_{arm}_logits.npy"
        require(row.get("logits_path")==rel
                and row.get("logits_shape")==list(rc["logits"].shape),
                "patched-logit receipt differs")
        logits=array(output,rel,row.get("logits_sha256"),tuple(rc["logits"].shape))
        expected_files.add(rel)
        ids=logits.argmax(axis=-1)[0].astype(np.int64).tolist()
        transcript=processor.batch_decode([ids])[0]
        words=token_list(transcript)
        strict=r["eligible"] and donor["eligible"]
        control=(bool(np.array_equal(logits,rc["logits"])) if arm=="self" else
                 bool(np.array_equal(logits,dc["logits"])) if arm=="full" else None)
        require(row.get("argmax_ids")==ids and row.get("transcript")==transcript
                and row.get("tokens")==words and row.get("strict_eligible") is strict
                and row.get("matches_donor_transcript") is (transcript==donor["transcript"])
                and row.get("matches_recipient_transcript") is (transcript==r["transcript"])
                and row.get("intended_word_only") is (strict and words==donor["reference_tokens"])
                and row.get("control_pass")==control,"patched CTC/control outcome differs")
        if control is False:failures.append((d["index"],arm))
        counts[arm]+=1
    require(set(counts)==set(runner.ARMS) and all(counts[arm]==96 for arm in runner.ARMS),
            "arm population differs")
    actual={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    require(actual==expected_files,"extra/missing output files before audit")
    bytes_used=sum((output/p).stat().st_size for p in actual)
    require(bytes_used<=runner.MAX_OUTPUT_BYTES,"actual output cap exceeded")
    return {"schema":runner.SCHEMA+"_audit","status":"independent_saved_data_audit_pass",
            "records_sha256":sha(records),"clean_rows":96,"patch_rows":576,
            "arm_counts":dict(counts),"control_failures":failures,
            "mechanism_gate_pass":not failures,"output_bytes":bytes_used,
            "byte_ledger_bound":current["byte_ledger"]["total_bound"]}


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--freeze",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.output,a.preflight,a.freeze)
    path=a.output/"audit.json"
    require(not path.exists(),"audit output exists")
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
