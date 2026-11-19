"""No-weight complete saved-evidence fixture for both Whisper causal checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest import mock

import numpy as np

from . import whisper_tail_causal_saved_audit_v1 as auditor
from . import whisper_tail_causal_worker_v1 as worker


def write(path:Path,value:object)->None:
    path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(base:Path,model:str)->tuple[Path,Path,Path,dict]:
    attempt=base/"attempt";attempt.mkdir();sidecars=attempt/"sidecars";sidecars.mkdir()
    admission=base/"admission.json";admission.write_text("{}\n")
    go=base/"go.json";go.write_text("{}\n")
    frozen=json.loads(worker.plan_source.PLAN_PATH.read_text())
    old_attempt=worker.v4.ATTEMPTS[model]
    saved=[json.loads(line) for line in (old_attempt/"records.jsonl").read_text().splitlines()[:-1]]
    by={(r["plan"]["case_id"],r["plan"]["condition"]):r for r in saved}
    source={"zero":by[(worker.plan_source.CASE,"z32000")],
            "noise":by[(worker.plan_source.CASE,"n32000_s0")]}
    zeros=np.zeros((1500,384),dtype=np.float32)
    zero_path=base/"zero.npy";np.save(zero_path,zeros,allow_pickle=False)
    state_sha=sha(zero_path);raw_sha=hashlib.sha256(zeros.tobytes()).hexdigest()
    def state_ref(name:str)->dict:
        os.link(zero_path,sidecars/name)
        return {"file":name,"sha256":state_sha,"shape":[1500,384],"dtype":"float32"}
    def output_ref(original:dict,index:int,key:str)->dict:
        ref=copy.deepcopy(original["sidecars"][key]);name=f"{index:03d}_{key}.npy"
        os.link(old_attempt/"sidecars"/ref["file"],sidecars/name)
        ref["file"]=name
        return ref
    calls=[{"type":"baseline","variant":v} for v in ("zero","noise")]+[
        {"type":"arm",**a} for a in frozen["arms"] if a["model"]==model]
    records=[];progress=[]
    for index,call in enumerate(calls,1):
        recipient=call["variant"] if call["type"]=="baseline" else call["recipient"]
        donor=recipient if call["type"]=="baseline" else call["donor"]
        # A full-donor fake uses recipient preprocessing and donor token/logit outputs.
        chosen=donor if call["type"]=="arm" and call["kind"]=="full_donor" else recipient
        outcome={key:copy.deepcopy(value) for key,value in source[recipient].items()
                 if key not in ("type","plan","sidecars","output")}
        outcome["output"]=copy.deepcopy(source[chosen]["output"])
        outcome["output"]["language_detection"]=copy.deepcopy(
            source[recipient]["output"]["language_detection"])
        outcome["sidecars"]={}
        for key in ("mel","mask","first_raw_logits","first_processed_scores","first_decoder_state"):
            original=source[recipient] if key in ("mel","mask") else source[chosen]
            outcome["sidecars"][key]=output_ref(original,index,key)
        row={"type":"call","index":index,"plan":call,"output":outcome}
        if call["type"]=="baseline":
            row["patch"]={"hook_calls":0}
            row["state_refs"]={site:state_ref(f"{index:03d}_{site}.npy")
                               for site in worker.plan_source.SITES}
        else:
            selected=call["target_rows"] if call["kind"] in (
                "target_transfer","identity_sham","full_donor") else call["source_rows"]
            row["patch"]={"hook_calls":1,"selected_rows":selected,
                "native_ref":state_ref(f"{index:03d}_native.npy"),
                "edited_ref":state_ref(f"{index:03d}_edited.npy"),
                "donor_state_sha256":raw_sha,"recipient_state_sha256":raw_sha,
                "inside_delta_l2":0.0,"outside_maxabs":0.0}
            if call["kind"]=="norm_matched_shifted_delta":
                row["patch"]["target_delta_l2"]=0.0
            row["state_refs"]=None
        records.append(row)
        progress.extend([{"type":"attempt","index":index},{"type":"complete","index":index}])
    write(attempt/"records.jsonl",records[0])
    (attempt/"records.jsonl").write_text("\n".join(json.dumps(r) for r in
        [*records,{"type":"complete","calls":56}])+"\n")
    (attempt/"progress.jsonl").write_text("\n".join(json.dumps(p) for p in progress)+"\n")
    start=datetime(2026,11,19,12,0,tzinfo=timezone.utc)
    host_samples=[{"monotonic_seconds":100.1,"battery_percent":99,"pmset_percent":99,
                   "external_connected":False,"power_text":"Now drawing from 'Battery Power' 99%;",
                   "thermal_text":"No thermal warning level has been recorded\nNo performance warning level has been recorded",
                   "lid_open":True,"thermal_ok":True,
                   "collected_utc":(start+timedelta(milliseconds=100)).isoformat()},
                  {"monotonic_seconds":101.9,"battery_percent":98,"pmset_percent":98,
                   "external_connected":False,"power_text":"Now drawing from 'Battery Power' 98%;",
                   "thermal_text":"No thermal warning level has been recorded\nNo performance warning level has been recorded",
                   "lid_open":True,"thermal_ok":True,
                   "collected_utc":(start+timedelta(seconds=1.9)).isoformat()}]
    (attempt/"host_samples.jsonl").write_text("\n".join(json.dumps(x) for x in host_samples)+"\n")
    rss={"sample_index":1,"monotonic_seconds":100.5,"tree_rss_bytes":123456,
         "evidence_bytes":1000}
    write(attempt/"worker_rss_samples.jsonl",rss)
    (attempt/"post_power_log.txt").write_text("fake power log\n")
    prior_admission=worker.HERE/("NATURAL_TAIL_MULTILINGUAL_ADMISSION_V4_2.json"
                                  if model=="whisper_multilingual" else
                                  "NATURAL_TAIL_ENGLISH_ADMISSION_V4_2.json")
    receipt={"source_identity":{},"v4_admission":{"assets":{
        "snapshot":json.loads(prior_admission.read_text())["assets"]["snapshot"]}}}
    launch={"schema":"whisper_tail_causal_parent_v1","status":"started","model":model,
        "attempt":str(attempt),"parent_pid":os.getpid(),"start_utc":start.isoformat(),
        "start_monotonic":100.0,"deadline_monotonic":1900.0,"nonce_sha256":"a"*64,
        "admission_sha256":sha(admission),"go_sha256":sha(go),"source_identity":{}}
    write(attempt/"launch_start.json",launch)
    result={**launch,"status":"completed_pending_saved_audit","whole_elapsed_seconds":3.0,
        "end_utc":(start+timedelta(seconds=3)).isoformat(),
        "worker_exit_utc":(start+timedelta(seconds=1)).isoformat(),
        "host_monitor_end_monotonic":102.0,"host_samples_sha256":sha(attempt/"host_samples.jsonl"),
        "host_samples_count":2,"post_power_log_sha256":sha(attempt/"post_power_log.txt"),
        "post_power_log_collected_utc":(start+timedelta(seconds=2.5)).isoformat(),
        "evidence_bytes_before_result":1000,"monitor":{
            "worker_returncode":0,"worker_stop_reason":"normal_exit",
            "worker_monitor_error":None,"parent_heartbeat_count":2,
            "worker_peak_tree_rss_bytes":123456,"worker_rss_samples":1,
            "worker_rss_sample_log_sha256":sha(attempt/"worker_rss_samples.jsonl")}}
    write(attempt/"parent_result.json",result)
    return attempt,admission,go,receipt


class SyntheticSavedEvidenceTest(unittest.TestCase):
    def test_complete_saved_audit_without_weights(self):
        for model in worker.v4.MODELS:
            with self.subTest(model=model),tempfile.TemporaryDirectory() as tmp:
                attempt,admission,go,receipt=make_fixture(Path(tmp),model)
                self.assertTrue(__import__("models.second_architecture.nonspeech_probe_v1.whisper_tail_causal_parent_v1",
                                           fromlist=["complete_worker_output"]).complete_worker_output(attempt,model))
                with mock.patch.dict(worker.ATTEMPTS,{model:attempt}),\
                     mock.patch.object(worker,"validate_go",return_value=receipt),\
                     mock.patch.object(auditor.old_host,"validate_log",return_value=None):
                    result=auditor.audit(model,admission,go)
                self.assertEqual(result["status"],"PASS")
                self.assertEqual(result["calls"],56)
                self.assertEqual(len(result["target_qualifications"]),12)

    def test_tampered_complete_answer_rejected(self):
        model="whisper_english"
        with tempfile.TemporaryDirectory() as tmp:
            attempt,admission,go,receipt=make_fixture(Path(tmp),model)
            path=attempt/"records.jsonl"
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[2]["output"]["output"]["sequence_ids"][-2]=123
            path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            with mock.patch.dict(worker.ATTEMPTS,{model:attempt}),\
                 mock.patch.object(worker,"validate_go",return_value=receipt),\
                 mock.patch.object(auditor.old_host,"validate_log",return_value=None):
                with self.assertRaises(ValueError):
                    auditor.audit(model,admission,go)


if __name__=="__main__":unittest.main()
