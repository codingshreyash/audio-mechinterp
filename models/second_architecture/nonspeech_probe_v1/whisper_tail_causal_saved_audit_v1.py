"""Read-only, no-weight audit of one frozen Whisper causal attempt."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as old_host
from . import natural_tail_saved_audit_v4 as v4_audit
from . import whisper_tail_causal_host_v1 as host
from . import whisper_tail_causal_parent_v1 as parent
from . import whisper_tail_causal_worker_v1 as worker


def require(ok:bool,message:str)->None:
    if not ok:raise ValueError(message)


def state(attempt:Path,ref:dict)->np.ndarray:
    name=ref.get("file")
    require(isinstance(name,str) and name==Path(name).name and name.endswith(".npy")
            and ref.get("shape")==[1500,384] and ref.get("dtype")=="float32",
            "hook state reference invalid")
    path=attempt/"sidecars"/name
    require(path.is_file() and not path.is_symlink() and worker.sha(path)==ref["sha256"],
            "hook state hash changed")
    value=np.load(path,allow_pickle=False)
    require(value.shape==(1500,384) and value.dtype==np.float32
            and np.isfinite(value).all(),"hook state geometry or values invalid")
    return value


def verify_patch(native:np.ndarray,edited:np.ndarray,recipient:np.ndarray,
                 donor:np.ndarray,call:dict,receipt:dict)->None:
    require(np.array_equal(native,recipient),"native hook differs from recipient baseline")
    kind=call["kind"]
    selected=call["target_rows"] if kind in ("target_transfer","identity_sham","full_donor") else call["source_rows"]
    require(receipt.get("hook_calls")==1 and receipt.get("selected_rows")==selected
            and receipt.get("donor_state_sha256")==worker.hashlib.sha256(donor.tobytes()).hexdigest()
            and receipt.get("recipient_state_sha256")==worker.hashlib.sha256(recipient.tobytes()).hexdigest(),
            "patch receipt identity invalid")
    expected=recipient.copy()
    if kind in ("target_transfer","full_donor"):expected[selected]=donor[selected]
    elif kind=="identity_sham":expected[selected]=recipient[selected]
    elif kind=="source_random":expected[selected]=donor[selected]
    else:
        require(kind=="norm_matched_shifted_delta","unknown patch kind")
        expected[selected]=recipient[selected]+(
            donor[call["target_rows"]]-recipient[call["target_rows"]])
    require(np.array_equal(edited,expected),"edited hook rows differ from frozen patch")
    untouched=np.ones(1500,dtype=bool);untouched[selected]=False
    require(np.array_equal(edited[untouched],native[untouched]),
            "unselected hook rows changed")
    magnitude=float(np.linalg.norm((edited[selected]-native[selected]).astype(np.float64)))
    require(receipt.get("outside_maxabs")==0.0
            and math.isclose(magnitude,receipt["inside_delta_l2"],rel_tol=1e-7,abs_tol=1e-6),
            "patch norm receipt changed")
    if kind=="norm_matched_shifted_delta":
        target=float(np.linalg.norm((donor[call["target_rows"]]
                                     -recipient[call["target_rows"]]).astype(np.float64)))
        require(math.isclose(target,receipt["target_delta_l2"],rel_tol=1e-7,abs_tol=1e-6)
                and abs(target-magnitude)<=1e-4,"shifted control norm changed")


def _output_complete(output:dict)->bool:
    return output.get("termination")=="eos" and output.get("eos_seen") is True \
        and output.get("censored") is False and isinstance(output.get("sequence_ids"),list)


def validate_whisper_record(model:str,outcome:dict,raw:np.ndarray,scores:np.ndarray,
                            mask:np.ndarray,generation_config:dict)->None:
    output=outcome["output"]
    ids,prefix,generated=(output[k] for k in ("sequence_ids","prefix_ids","generated_ids"))
    rows=output["score_rows"]
    require(ids==prefix+generated and 1<=len(generated)<=128
            and len(generated)==output["actual_generation_steps"]==len(rows),
            "Whisper output sequence/score rows invalid")
    eos=50257 if model=="whisper_multilingual" else 50256
    require((generated[-1]==eos)==output["eos_seen"]
            and (output["termination"]=="eos")==output["eos_seen"]
            and (output["termination"]=="length_censored")==output["censored"]
            and (output["eos_seen"] or len(generated)==128)
            and eos not in generated[:-1],"EOS or censor proof invalid")
    require(all(row["token_id"]==generated[i]
                and isinstance(row["raw_logits_sha256"],str)
                and len(row["raw_logits_sha256"])==64
                and isinstance(row["processed_logits_sha256"],str)
                and len(row["processed_logits_sha256"])==64
                and math.isfinite(row["raw_selected_logit"])
                and math.isfinite(row["selected_logit"])
                for i,row in enumerate(rows)),"selected score proof invalid")
    first=rows[0];token=generated[0]
    require(first["raw_logits_sha256"]==worker.hashlib.sha256(raw.tobytes()).hexdigest()
            and first["processed_logits_sha256"]==worker.hashlib.sha256(scores.tobytes()).hexdigest()
            and first["raw_top_id"]==int(raw.argmax())
            and first["top_id"]==int(scores.argmax())
            and math.isclose(first["raw_selected_logit"],float(raw[token]),abs_tol=1e-5)
            and math.isclose(first["selected_logit"],float(scores[token]),abs_tol=1e-5),
            "first-step vector/score proof invalid")
    counts=outcome["capture_counts"]
    require(counts.get("generate")=={name:1 for name in
            ("mel","post_conv_pos","post_block0","final_encoder")},
            "transcription encoder hook census invalid")
    kwargs=outcome["generation_kwargs"]
    multilingual=model=="whisper_multilingual"
    require((kwargs.get("task")=="transcribe" and "language" in kwargs
             and kwargs["language"] is None) if multilingual else
            ("task" not in kwargs and "language" not in kwargs),
            "generation signature changed")
    detection=output["language_detection"]
    if multilingual:
        require(counts.get("detect")=={name:1 for name in
                ("post_conv_pos","post_block0","final_encoder")},
                "detection encoder hook census invalid")
        candidate=detection.get("language_logits",{}) if isinstance(detection,dict) else {}
        lang_ids=generation_config["lang_to_id"]
        chosen=min(candidate,key=lambda name:(-candidate[name],lang_ids[name])) if candidate else None
        require(set(candidate)==set(lang_ids) and detection["candidate_count"]==len(candidate)
                and detection["chosen_language_ids"]==[lang_ids[chosen]]
                and prefix==[generation_config["decoder_start_token_id"],lang_ids[chosen],
                    generation_config["task_to_id"]["transcribe"],
                    generation_config["no_timestamps_token_id"]],
                "detected language or prefix invalid")
    else:
        require("detect" not in counts and detection is None
                and prefix==[generation_config["decoder_start_token_id"],
                             generation_config["no_timestamps_token_id"]],
                "English-only language branch changed")
    active=1000
    require(outcome["geometry"]["active_mel_frames"]==active
            and outcome["geometry"]["mask_sha256"]
            ==worker.hashlib.sha256(bytes(mask.astype(np.uint8))).hexdigest(),
            "mask summary invalid")


def audit(model:str,admission:Path,go:Path)->dict:
    receipt=worker.validate_go(model,admission,go)
    attempt=worker.ATTEMPTS[model]
    require(attempt.is_dir() and not attempt.is_symlink(),"fixed causal attempt missing")
    launch=json.loads((attempt/"launch_start.json").read_text())
    result=json.loads((attempt/"parent_result.json").read_text())
    require(result.get("status")=="completed_pending_saved_audit"
            and launch.get("status")=="started"
            and launch.get("schema")==result.get("schema")=="whisper_tail_causal_parent_v1"
            and all(launch.get(k)==result.get(k) for k in (
                "model","attempt","parent_pid","start_utc","start_monotonic",
                "deadline_monotonic","nonce_sha256","admission_sha256","go_sha256",
                "source_identity"))
            and result.get("model")==model and result.get("attempt")==str(attempt)
            and result.get("admission_sha256")==worker.sha(admission)
            and result.get("go_sha256")==worker.sha(go)
            and result.get("source_identity")==receipt["source_identity"],
            "parent/GO identity invalid")
    duration=result["whole_elapsed_seconds"]
    require(0<duration<worker.WALL_SECONDS
            and result["deadline_monotonic"]-result["start_monotonic"]==worker.WALL_SECONDS
            and result["evidence_bytes_before_result"]<worker.EVIDENCE_BYTES
            and parent.bytes_used(attempt)<worker.EVIDENCE_BYTES,
            "whole deadline/evidence cap invalid")
    monitor=result["monitor"]
    require(monitor["worker_returncode"]==0 and monitor["worker_stop_reason"]=="normal_exit"
            and monitor["worker_monitor_error"] is None
            and monitor["parent_heartbeat_count"]>0
            and 0<monitor["worker_peak_tree_rss_bytes"]<worker.RSS_BYTES
            and monitor["worker_rss_samples"]>0,"resource monitor invalid")
    rss_path=attempt/"worker_rss_samples.jsonl"
    require(worker.sha(rss_path)==monitor["worker_rss_sample_log_sha256"],"RSS log hash invalid")
    rss=[json.loads(line) for line in rss_path.read_text().splitlines()]
    require(len(rss)==monitor["worker_rss_samples"]
            and [row["sample_index"] for row in rss]==list(range(1,len(rss)+1))
            and max(row["tree_rss_bytes"] for row in rss)==monitor["worker_peak_tree_rss_bytes"]
            and all(0<row["tree_rss_bytes"]<worker.RSS_BYTES
                    and row["evidence_bytes"]<worker.EVIDENCE_BYTES
                    and result["start_monotonic"]<=row["monotonic_seconds"]
                    <=result["deadline_monotonic"] for row in rss),"RSS rows invalid")
    host_path=attempt/"host_samples.jsonl"
    require(worker.sha(host_path)==result["host_samples_sha256"],"host log hash invalid")
    samples=[json.loads(line) for line in host_path.read_text().splitlines()]
    require(len(samples)==result["host_samples_count"] and len(samples)>=2,
            "host sample count invalid")
    require(samples[-1]["monotonic_seconds"]<=result["host_monitor_end_monotonic"]
            <=result["deadline_monotonic"],"host monitor end invalid")
    host.validate_cadence(samples,start=result["start_monotonic"],
                          end=result["host_monitor_end_monotonic"])
    start_utc=datetime.fromisoformat(result["start_utc"])
    worker_exit=datetime.fromisoformat(result["worker_exit_utc"])
    end_utc=datetime.fromisoformat(result["end_utc"])
    require(start_utc<=worker_exit<=end_utc
            and abs((end_utc-start_utc).total_seconds()-duration)<=0.25,
            "whole clock proof invalid")
    post_log=attempt/"post_power_log.txt"
    require(worker.sha(post_log)==result["post_power_log_sha256"],"post power log hash invalid")
    old_host.validate_log(post_log.read_text(),start_utc,worker_exit,
                          datetime.fromisoformat(result["post_power_log_collected_utc"]))
    require(parent.progress_census(attempt/"progress.jsonl")
            =={"complete":56,"attempted_incomplete":None}
            and parent.complete_worker_output(attempt,model),"incomplete worker output")
    frozen=json.loads(worker.plan_source.PLAN_PATH.read_text())
    unit=next(u for u in frozen["units"] if u["model"]==model)
    calls=[{"type":"baseline","variant":v} for v in ("zero","noise")]+[
        {"type":"arm",**a} for a in frozen["arms"] if a["model"]==model]
    records=[json.loads(line) for line in (attempt/"records.jsonl").read_text().splitlines()]
    baselines={};states={};valid_sites={};outcomes=[]
    vocab=51865 if model=="whisper_multilingual" else 51864
    generation_config=json.loads((Path(receipt["v4_admission"]["assets"]["snapshot"])
                                  /"generation_config.json").read_text())
    for index,(call,row) in enumerate(zip(calls,records[:-1],strict=True),1):
        require(row["index"]==index and row["plan"]==call,"call order changed")
        outcome=row["output"];output=outcome["output"]
        require(outcome["generation_kwargs"].get("past_key_values") is None
                and outcome["generation_kwargs"].get("encoder_outputs") is None,
                "stale decoder/encoder cache supplied")
        refs=outcome["sidecars"]
        require(set(refs)=={"mel","mask","first_raw_logits","first_processed_scores",
                            "first_decoder_state"},"Whisper sidecar census invalid")
        mel=v4_audit.load_sidecar(attempt,refs["mel"],(80,3000),"float32")
        mask=v4_audit.load_sidecar(attempt,refs["mask"],(3000,),"int64")
        raw=v4_audit.load_sidecar(attempt,refs["first_raw_logits"],(vocab,),"float32")
        scores=v4_audit.load_sidecar(attempt,refs["first_processed_scores"],(vocab,),"float32")
        v4_audit.load_sidecar(attempt,refs["first_decoder_state"],
                              (1,4 if model=="whisper_multilingual" else 2,384),"float32")
        require(np.isfinite(mel).all() and np.isfinite(raw).all()
                and np.array_equal(mask[:1000],np.ones(1000,dtype=np.int64))
                and np.array_equal(mask[1000:],np.zeros(2000,dtype=np.int64)),
                "processor mel/mask invalid")
        validate_whisper_record(model,outcome,raw,scores,mask,generation_config)
        variant=call["variant"] if call["type"]=="baseline" else call["recipient"]
        if call["type"]=="baseline":
            expected=unit[variant]
            require(output["sequence_ids"]==expected["sequence_ids"]
                    and output["prefix_ids"]==expected["prefix_ids"]
                    and output["termination"]==expected["termination"],
                    "baseline differs from V4")
            for key in ("mel","mask","first_raw_logits","first_processed_scores"):
                old_ref=expected[key]
                old_path=worker.v4.ATTEMPTS[model]/"sidecars"/old_ref["file"]
                require(worker.sha(old_path)==old_ref["sha256"],"V4 vector changed")
                shape,dtype=((80,3000),"float32") if key=="mel" else \
                    ((3000,),"int64") if key=="mask" else ((vocab,),"float32")
                actual=v4_audit.load_sidecar(attempt,refs[key],shape,dtype)
                require(np.array_equal(actual,np.load(old_path,allow_pickle=False)),
                        "fresh baseline vector differs from V4")
            states[variant]={site:state(attempt,ref) for site,ref in row["state_refs"].items()}
            require(set(states[variant])==set(worker.plan_source.SITES)
                    and row["patch"]=={"hook_calls":0},"baseline hook census invalid")
            baselines[variant]={"output":output,"raw":raw,"mask":mask,"mel":mel}
            if len(baselines)==2:
                require(np.array_equal(baselines["zero"]["mask"],baselines["noise"]["mask"]),
                        "same-length fresh masks differ")
        else:
            site=call["site"]
            require(np.array_equal(mel,baselines[variant]["mel"])
                    and np.array_equal(mask,baselines[variant]["mask"]),
                    "patch arm preprocessing differs from recipient")
            native=state(attempt,row["patch"]["native_ref"])
            edited=state(attempt,row["patch"]["edited_ref"])
            verify_patch(native,edited,states[variant][site],states[call["donor"]][site],
                         call,row["patch"])
            require(output["prefix_ids"]==baselines[variant]["output"]["prefix_ids"]
                    and output["language_detection"]
                    ==baselines[variant]["output"]["language_detection"],
                    "generation-only patch changed language branch")
            if call["kind"] in ("identity_sham","full_donor"):
                reference=baselines[variant if call["kind"]=="identity_sham" else call["donor"]]
                ok=(_output_complete(output) and output["sequence_ids"]==reference["output"]["sequence_ids"]
                    and np.array_equal(raw,reference["raw"]))
                valid_sites.setdefault((site,call["recipient"]),[]).append(ok)
            outcomes.append({"index":index,"site":site,"region":call["region"],
                "kind":call["kind"],"recipient":variant,"donor":call["donor"],
                "eos_complete":_output_complete(output),
                "sequence_ids":output["sequence_ids"],"transcript":output["transcript"],
                "delta_l2":row["patch"]["inside_delta_l2"],
                "matches_recipient":_output_complete(output) and
                    output["sequence_ids"]==baselines[variant]["output"]["sequence_ids"],
                "matches_donor":_output_complete(output) and
                    output["sequence_ids"]==baselines[call["donor"]]["output"]["sequence_ids"]})
    require(len(valid_sites)==6 and all(len(values)==3 for values in valid_sites.values()),
            "hook sham/positive-control census invalid")
    keyed={(o["site"],o["region"],o["recipient"],o["kind"]):o for o in outcomes}
    qualifications=[]
    for target in (o for o in outcomes if o["kind"]=="target_transfer"):
        site,region,recipient=target["site"],target["region"],target["recipient"]
        identity=keyed[(site,region,recipient,"identity_sham")]
        full=keyed[(site,"all_encoder_rows",recipient,"full_donor")]
        random=keyed[(site,region,recipient,"source_random")]
        shifted=keyed[(site,region,recipient,"norm_matched_shifted_delta")]
        qualifications.append({"target_index":target["index"],
            "control_qualified":all((identity["matches_recipient"],full["matches_donor"],
                random["matches_recipient"],shifted["matches_recipient"])),
            "target_eos_complete":target["eos_complete"],
            "target_matches_donor":target["matches_donor"],
            "source_random_delta_l2":random["delta_l2"],
            "shifted_delta_l2":shifted["delta_l2"]})
    return {"schema":"whisper_tail_causal_saved_audit_v1","status":"PASS",
            "model":model,"attempt":str(attempt),"calls":56,"outcomes":outcomes,
            "target_qualifications":qualifications,
            "whole_elapsed_seconds":duration,"peak_tree_rss_bytes":monitor["worker_peak_tree_rss_bytes"],
            "host_samples":len(samples)}


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=worker.v4.MODELS,required=True)
    parser.add_argument("--admission",type=Path,required=True)
    parser.add_argument("--go",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.model,args.admission,args.go),sort_keys=True))


if __name__=="__main__":main()
