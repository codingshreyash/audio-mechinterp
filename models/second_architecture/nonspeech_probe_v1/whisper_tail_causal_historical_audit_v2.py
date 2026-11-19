"""Read-only GO-time provenance repair for one completed Whisper causal attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from . import whisper_tail_causal_worker_v2 as worker
from . import whisper_tail_causal_saved_audit_v2 as saved

HERE=worker.HERE
ROOT=worker.ROOT
MODEL="whisper_multilingual"
ORIGINAL_GO_COMMIT="abf736a97462734af7bf4f38cf3c205db37c36d0"
ORIGINAL_GO_SHA="005b5aa403066ea26b57b4b3251fec47533b2d8b1b3db6c0af685faf58a3ab52"
ORIGINAL_MEMO_SHA="673dfb52ab7dc5616a9f2afe70676e3f634e4c424fafd782c1d705e53cff95a5"
ORIGINAL_SAVED_AUDIT_SHA="961f1c47ed283dec8ba5394ea93ddfefb90f29affe7de7f79925ee4a2cdf9ccc"
RAW_INVENTORY_SHA="13ae62219d5805b28e0df93b58f901835f7ee7d91733792ee38294d2c165f9f3"
SOURCE_NAMES=("whisper_tail_causal_historical_audit_v2.py",
              "test_whisper_tail_causal_historical_audit_v2.py",
              "WHISPER_TAIL_CAUSAL_HISTORICAL_AUDIT_AMENDMENT_V2.md",
              "WHISPER_TAIL_CAUSAL_V2_RAW_FILE_INVENTORY.json",
              "runs/whisper_tail_causal_v2/whisper_multilingual_attempt_001_audit.json")
ADMISSION=HERE/"WHISPER_TAIL_CAUSAL_HISTORICAL_ADMISSION_V2.json"
REVIEW=HERE/"WHISPER_TAIL_CAUSAL_HISTORICAL_EXECUTABLE_REVIEW_V2.md"
GO=HERE/"WHISPER_TAIL_CAUSAL_HISTORICAL_GO_V2.json"
OUTPUT=worker.RUNS/"whisper_multilingual_attempt_001_historical_audit.json"
SAVED=worker.RUNS/"whisper_multilingual_attempt_001_audit.json"
RAW_INVENTORY=HERE/"WHISPER_TAIL_CAUSAL_V2_RAW_FILE_INVENTORY.json"


def require(ok:bool,message:str)->None:
    if not ok:raise ValueError(message)


def sha_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def sha(path:Path)->str:
    return sha_bytes(path.read_bytes())


def rel(path:Path)->str:
    require(path.is_absolute() and path.is_relative_to(ROOT),"artifact outside repo")
    return str(path.relative_to(ROOT))


def git(*args:str)->bytes:
    result=subprocess.run(("git",*args),cwd=ROOT,capture_output=True)
    require(result.returncode==0,f"historical git query failed: {args[:2]}")
    return result.stdout


def historical_bytes(path:Path)->bytes:
    return git("show",f"{ORIGINAL_GO_COMMIT}:{rel(path)}")


def historical_commit(path:Path)->str:
    value=git("log","-1",ORIGINAL_GO_COMMIT,"--format=%H","--",rel(path)).decode().strip()
    require(len(value)==40,"historical file commit missing")
    return value


def current_bytes(path:Path)->bytes:
    return path.read_bytes()


def check_artifact(path:Path,expected_sha:str)->str:
    """Bind current clean bytes and the immutable GO tree to one admitted hash."""
    worker._tracked_clean(path)
    historical=historical_bytes(path)
    require(sha_bytes(historical)==expected_sha
            and current_bytes(path)==historical,
            f"current or GO-time artifact differs: {rel(path)}")
    return historical_commit(path)


def historical_receipt()->dict:
    """The original rejection must be precisely the later-edit ancestry false alarm."""
    original_admission=worker.ADMISSIONS[MODEL]
    original_go=worker.GOS[MODEL]
    try:
        worker.validate_go(MODEL,original_admission,original_go)
    except ValueError as error:
        require(str(error)=="independent causal source/admission/review/GO order invalid",
                f"original GO failed for a different reason: {error}")
    else:
        raise ValueError("historical exception not required; original GO already valid")
    require(worker._ancestor(ORIGINAL_GO_COMMIT,"HEAD"),"original GO not in current history")
    expected=worker.admission(MODEL)
    require(json.loads(original_admission.read_text())==expected,
            "original source-only admission changed")
    old_go=json.loads(original_go.read_text())
    require(sha(original_go)==ORIGINAL_GO_SHA
            and old_go.get("schema")=="whisper_tail_causal_go_v2"
            and old_go.get("status")=="independent_executable_go"
            and old_go.get("model")==MODEL
            and old_go.get("attempt")==str(worker.ATTEMPTS[MODEL])
            and old_go.get("admission_sha256")==sha(original_admission)
            and old_go.get("plan_sha256")==sha(worker.plan_source.PLAN_PATH)
            and old_go.get("source_identity")==expected["source_identity"]
            and old_go.get("review_sha256")==sha(worker.REVIEW),
            "original GO fields changed")
    source_commits=[]
    for name,pinned in expected["source_identity"].items():
        source_commits.append(check_artifact(HERE/name,pinned))
    candidate_commit=check_artifact(original_admission,sha(original_admission))
    review_commit=check_artifact(worker.REVIEW,sha(worker.REVIEW))
    go_commit=check_artifact(original_go,ORIGINAL_GO_SHA)
    require(go_commit==ORIGINAL_GO_COMMIT
            and candidate_commit!=review_commit!=go_commit
            and all(commit!=review_commit and worker._ancestor(commit,review_commit)
                    for commit in [*source_commits,candidate_commit])
            and worker._ancestor(review_commit,go_commit),
            "GO-time source/admission/review lineage invalid")
    memo=HERE/"WHISPER_TAIL_CAUSAL_INTERPRETATION_INDEPENDENT_V1.md"
    require(sha(memo)==ORIGINAL_MEMO_SHA
            and expected["source_identity"][memo.name]==ORIGINAL_MEMO_SHA,
            "restored V1 interpretation bytes differ from original pin")
    return {"schema":"whisper_tail_causal_historical_receipt_v2",
            "original_go_commit":ORIGINAL_GO_COMMIT,
            "original_go_sha256":ORIGINAL_GO_SHA,
            "original_admission_sha256":sha(original_admission),
            "original_review_sha256":sha(worker.REVIEW),
            "v1_interpretation_sha256":ORIGINAL_MEMO_SHA,
            "historical_source_commits":dict(zip(worker.SOURCES,source_commits,strict=True)),
            "candidate_commit":candidate_commit,"review_commit":review_commit,
            "source_identity":expected["source_identity"],
            "original_admission":expected}


def evidence_inventory()->list[dict]:
    attempt=worker.ATTEMPTS[MODEL]
    require(attempt.is_dir() and not attempt.is_symlink(),"fixed completed attempt missing")
    entries=[]
    for path in sorted(attempt.rglob("*")):
        require(not path.is_symlink(),"attempt symlink forbidden")
        if path.is_file():
            entries.append({"path":str(path.relative_to(attempt)),
                            "bytes":path.stat().st_size,"sha256":sha(path)})
    require(len(entries)==403 and sum(x["bytes"] for x in entries)<worker.EVIDENCE_BYTES,
            "completed attempt inventory/cap changed")
    return entries


def verify_inventory(pinned:list[dict],current:list[dict])->None:
    require(pinned==current,"completed raw attempt file/hash inventory changed")


def source_identity()->dict[str,str]:
    result={}
    for name in SOURCE_NAMES:
        path=HERE/name;worker._tracked_clean(path);result[name]=sha(path)
    return result


def admission()->dict:
    receipt=historical_receipt()
    worker._tracked_clean(SAVED)
    require(sha(SAVED)==ORIGINAL_SAVED_AUDIT_SHA,"original compact V2 audit changed")
    prior=json.loads(SAVED.read_text())
    require(prior.get("status")=="PASS" and prior.get("calls")==56
            and prior.get("attempt")==str(worker.ATTEMPTS[MODEL]),
            "saved V2 audit not a complete PASS")
    worker._tracked_clean(RAW_INVENTORY)
    require(sha(RAW_INVENTORY)==RAW_INVENTORY_SHA,"committed V2 raw inventory changed")
    raw=json.loads(RAW_INVENTORY.read_text())
    live=evidence_inventory()
    require(raw.get("raw_file_count")==403 and raw.get("audit_sha256")==ORIGINAL_SAVED_AUDIT_SHA
            and raw.get("plan_sha256")==sha(worker.plan_source.PLAN_PATH)
            and raw.get("model")==MODEL
            and raw.get("files")==[{**row,"path":str((worker.ATTEMPTS[MODEL]/row["path"])
                                              .relative_to(ROOT))} for row in live],
            "committed V2 raw inventory differs from attempt")
    return {"schema":"whisper_tail_causal_historical_admission_v2",
            "status":"source_only_saved_evidence_no_weights",
            "model":MODEL,"attempt":str(worker.ATTEMPTS[MODEL]),
            "source_identity":source_identity(),"historical_receipt":receipt,
            "raw_inventory":live,"committed_raw_inventory_sha256":RAW_INVENTORY_SHA,
            "saved_v2_audit_sha256":ORIGINAL_SAVED_AUDIT_SHA}


def validate_posthoc_go()->dict:
    for path in (ADMISSION,REVIEW,GO):worker._tracked_clean(path)
    candidate=json.loads(ADMISSION.read_text())
    current=admission()
    verify_inventory(candidate.get("raw_inventory"),current["raw_inventory"])
    require(candidate==current,"historical admission/evidence changed")
    approval=json.loads(GO.read_text())
    require(approval.get("schema")=="whisper_tail_causal_historical_go_v2"
            and approval.get("status")=="independent_saved_only_go"
            and approval.get("model")==MODEL
            and approval.get("attempt")==str(worker.ATTEMPTS[MODEL])
            and approval.get("admission_sha256")==sha(ADMISSION)
            and approval.get("review_sha256")==sha(REVIEW)
            and approval.get("source_identity")==candidate["source_identity"]
            and approval.get("original_go_commit")==ORIGINAL_GO_COMMIT,
            "historical saved-audit GO invalid")
    sources=[worker._commit(HERE/name) for name in SOURCE_NAMES]
    candidate_commit,review_commit,go_commit=map(worker._commit,(ADMISSION,REVIEW,GO))
    require(all(c and c!=review_commit and worker._ancestor(c,review_commit)
                for c in [*sources,candidate_commit])
            and review_commit!=go_commit and worker._ancestor(review_commit,go_commit),
            "historical source/admission/review/GO order invalid")
    return candidate


def audit_saved()->dict:
    candidate=validate_posthoc_go()
    receipt=candidate["historical_receipt"]["original_admission"]
    calls=0
    def exact_original_call(model:str,admission_path:Path,go_path:Path)->dict:
        nonlocal calls
        calls+=1
        require((model,admission_path,go_path)==
                (MODEL,worker.ADMISSIONS[MODEL],worker.GOS[MODEL]),
                "historical GO substitution escaped original saved-audit call")
        require(calls==1,"historical GO substitution reused")
        return receipt
    with patch.object(worker,"validate_go",side_effect=exact_original_call):
        computed=saved.audit(MODEL,worker.ADMISSIONS[MODEL],worker.GOS[MODEL])
    require(calls==1,"unchanged V2 saved auditor did not request original GO once")
    require(computed==json.loads(SAVED.read_text()),
            "unchanged V2 saved audit differs from original complete result")
    return {"schema":"whisper_tail_causal_historical_audit_v2","status":"PASS",
            "model":MODEL,"attempt":str(worker.ATTEMPTS[MODEL]),
            "original_go_commit":ORIGINAL_GO_COMMIT,
            "historical_admission_sha256":sha(ADMISSION),
            "saved_v2_audit_sha256":sha(SAVED),
            "calls":computed["calls"],"primary_replication_count":computed["primary_replication_count"],
            "full_replication":computed["full_replication"]}


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("preflight","audit"))
    args=parser.parse_args()
    if args.command=="preflight":
        require(not ADMISSION.exists(),"historical admission already exists")
        worker.v4.write_once(ADMISSION,admission())
        print(json.dumps({"status":"source_only_saved_evidence_no_weights",
                          "admission":str(ADMISSION),"sha256":sha(ADMISSION)}))
    else:
        require(not OUTPUT.exists(),"historical audit output already exists")
        result=audit_saved()
        worker.v4.write_once(OUTPUT,result)
        print(json.dumps({"status":result["status"],"output":str(OUTPUT),
                          "sha256":sha(OUTPUT)}))


if __name__=="__main__":main()
