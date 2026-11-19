"""One-shot bounded parent for the frozen Whisper Tiny causal pilot."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import psutil

from . import causal_parent_v2 as bounded
from . import whisper_tail_causal_host_v1 as host
from . import whisper_tail_causal_worker_v2 as worker

ROOT=worker.ROOT
LOCK=worker.RUNS/"active_model_lock"
POLL_SECONDS=0.1


def canonical(value:object)->str:
    return json.dumps(value,sort_keys=True,separators=(",",":"))


def write_once(path:Path,value:object)->None:
    with path.open("x") as out:
        out.write(canonical(value)+"\n");out.flush();os.fsync(out.fileno())


def bytes_used(path:Path)->int:
    return bounded.output_bytes(path)


def progress_census(path:Path)->dict:
    if not path.is_file():return {"complete":0,"attempted_incomplete":None}
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    worker.require(len(rows)<=112,"progress exceeds schedule")
    complete=0
    for index in range(1,57):
        offset=2*(index-1)
        if offset>=len(rows):break
        worker.require(rows[offset]=={"type":"attempt","index":index},"progress attempt order")
        if offset+1>=len(rows):break
        worker.require(rows[offset+1]=={"type":"complete","index":index},"progress complete order")
        complete+=1
    worker.require(len(rows) in (2*complete,2*complete+1),"progress extras")
    return {"complete":complete,"attempted_incomplete":complete+1 if len(rows)%2 else None}


def complete_worker_output(attempt:Path,model:str)->bool:
    try:
        if progress_census(attempt/"progress.jsonl")!={"complete":56,"attempted_incomplete":None}:
            return False
        frozen=json.loads(worker.plan_source.PLAN_PATH.read_text())
        calls=[{"type":"baseline","variant":v} for v in ("zero","noise")]+[
            {"type":"arm",**a} for a in frozen["arms"] if a["model"]==model]
        rows=[json.loads(line) for line in (attempt/"records.jsonl").read_text().splitlines()]
        if len(rows)!=57 or rows[-1]!={"type":"complete","calls":56}:return False
        names=set();folder=attempt/"sidecars"
        for index,(call,row) in enumerate(zip(calls,rows[:-1],strict=True),1):
            if row.get("type")!="call" or row.get("index")!=index or row.get("plan")!=call:return False
            refs=list(row["output"]["sidecars"].values())
            if call["type"]=="baseline":refs+=list(row["state_refs"].values())
            else:refs += [row["patch"]["native_ref"],row["patch"]["edited_ref"]]
            for ref in refs:
                name=ref["file"];path=folder/name
                if name!=Path(name).name or not path.is_file() or path.is_symlink():return False
                names.add(name)
        return len(names)==394 and names=={p.name for p in folder.iterdir()}
    except (OSError,ValueError,TypeError,KeyError):return False


def _host_sample(samples:list[dict],out,*,initial:bool=False)->None:
    sample=host.quick_probe(initial=initial)
    if samples:
        worker.require(0<sample["monotonic_seconds"]-samples[-1]["monotonic_seconds"]<=15,
                       "host probe cadence lost")
    samples.append(sample)
    out.write(canonical(sample)+"\n");out.flush();os.fsync(out.fileno())


def run_bounded(command:list[str],attempt:Path,model:str,*,nonce:str,
                start:float,deadline:float,samples:list[dict],host_out)->dict:
    read_fd,write_fd=os.pipe();process=None
    peak=beats=count=0;reason="normal_exit";error=None
    log=attempt/"worker_rss_samples.jsonl"
    try:
        with ThreadPoolExecutor(max_workers=1) as probes,\
             (attempt/"worker_stdout.txt").open("xb") as stdout,\
             (attempt/"worker_stderr.txt").open("xb") as stderr,\
             log.open("x") as rss_out:
            process=subprocess.Popen(command+["--heartbeat-fd",str(read_fd)],
                cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,
                pass_fds=(read_fd,),env={**os.environ,"HF_HUB_OFFLINE":"1",
                                        "TRANSFORMERS_OFFLINE":"1","OMP_NUM_THREADS":"4"})
            os.close(read_fd);read_fd=-1
            os.write(write_fd,b"HELLO "+nonce.encode()+b"\n")
            next_beat=time.monotonic();next_host=samples[-1]["monotonic_seconds"]+2
            pending=None;probe_started=None
            while process.poll() is None:
                now=time.monotonic()
                if now>=deadline-worker.POST_RESERVE_SECONDS:
                    reason="wall_limit";bounded.kill_group(process);break
                if pending is None and now>=next_host:
                    probe_started=now;pending=probes.submit(host.quick_probe,initial=False)
                if pending is not None and pending.done():
                    try:
                        sample=pending.result()
                        worker.require(0<sample["monotonic_seconds"]-samples[-1]["monotonic_seconds"]<=15,
                                       "host probe cadence lost")
                        samples.append(sample);host_out.write(canonical(sample)+"\n")
                        host_out.flush();os.fsync(host_out.fileno())
                    except BaseException as exc:
                        reason,error="host_probe_failed",repr(exc);bounded.kill_group(process);break
                    next_host=sample["monotonic_seconds"]+2;pending=None
                if pending is not None and now-probe_started>=15:
                    reason,error="host_probe_failed","in-run host probe timed out"
                    bounded.kill_group(process);break
                try:
                    rss=bounded.process_tree_rss_bytes(process.pid)
                    size=bytes_used(attempt)
                except BaseException as exc:
                    reason,error="monitor_error",repr(exc);bounded.kill_group(process);break
                if rss is not None and rss>0:
                    count+=1;peak=max(peak,rss)
                    rss_out.write(canonical({"sample_index":count,"monotonic_seconds":now,
                        "tree_rss_bytes":rss,"evidence_bytes":size})+"\n");rss_out.flush()
                    if rss>=worker.RSS_BYTES:
                        reason="rss_limit";bounded.kill_group(process);break
                if size>=worker.EVIDENCE_BYTES-32*1024**2:
                    reason="evidence_limit";bounded.kill_group(process);break
                if now>=next_beat:
                    try:os.write(write_fd,b"H\n");beats+=1
                    except BrokenPipeError:
                        grace_end=min(time.monotonic()+1,deadline-worker.POST_RESERVE_SECONDS)
                        while process.poll() is None and time.monotonic()<grace_end:
                            grace_now=time.monotonic()
                            if pending is not None and pending.done():
                                try:
                                    sample=pending.result()
                                    worker.require(0<sample["monotonic_seconds"]-samples[-1]["monotonic_seconds"]<=15,
                                                   "host probe cadence lost")
                                    samples.append(sample);host_out.write(canonical(sample)+"\n")
                                    host_out.flush();os.fsync(host_out.fileno())
                                except BaseException as exc:
                                    reason,error="host_probe_failed",repr(exc);bounded.kill_group(process);break
                                next_host=sample["monotonic_seconds"]+2;pending=None
                            if pending is None and grace_now>=next_host:
                                probe_started=grace_now;pending=probes.submit(host.quick_probe,initial=False)
                            if pending is not None and grace_now-probe_started>=15:
                                reason,error="host_probe_failed","in-run host probe timed out"
                                bounded.kill_group(process);break
                            try:
                                rss=bounded.process_tree_rss_bytes(process.pid)
                                size=bytes_used(attempt)
                            except BaseException as exc:
                                reason,error="monitor_error",repr(exc);bounded.kill_group(process);break
                            if rss is not None and rss>0:
                                count+=1;peak=max(peak,rss)
                                rss_out.write(canonical({"sample_index":count,"monotonic_seconds":grace_now,
                                    "tree_rss_bytes":rss,"evidence_bytes":size})+"\n");rss_out.flush()
                                if rss>=worker.RSS_BYTES:
                                    reason="rss_limit";bounded.kill_group(process);break
                            if size>=worker.EVIDENCE_BYTES-32*1024**2:
                                reason="evidence_limit";bounded.kill_group(process);break
                            time.sleep(min(POLL_SECONDS,max(0,grace_end-time.monotonic())))
                        if reason=="normal_exit" and not (process.poll()==0 and complete_worker_output(attempt,model)):
                            reason="heartbeat_pipe_lost"
                            if process.poll() is None:bounded.kill_group(process)
                        break
                    next_beat=now+0.25
                time.sleep(POLL_SECONDS)
            process.wait(timeout=10)
            if pending is not None:
                try:
                    sample=pending.result(timeout=max(0,15-(time.monotonic()-probe_started)))
                    worker.require(0<sample["monotonic_seconds"]-samples[-1]["monotonic_seconds"]<=15,
                                   "final in-run host probe cadence lost")
                    samples.append(sample);host_out.write(canonical(sample)+"\n")
                    host_out.flush();os.fsync(host_out.fileno())
                except BaseException as exc:
                    reason,error="host_probe_failed",repr(exc)
            rss_out.flush();os.fsync(rss_out.fileno())
    except BaseException:
        if process is not None and process.poll() is None:bounded.kill_group(process)
        raise
    finally:
        if read_fd>=0:os.close(read_fd)
        os.close(write_fd)
    return {"worker_returncode":process.returncode,"worker_stop_reason":reason,
            "worker_peak_tree_rss_bytes":peak,"worker_rss_samples":count,
            "worker_rss_sample_log_sha256":worker.sha(log),
            "worker_monitor_error":error,"parent_heartbeat_count":beats}


def launch(model:str,admission:Path,go:Path)->dict:
    receipt=worker.validate_go(model,admission,go)
    attempt=worker.ATTEMPTS[model]
    worker.require(attempt.is_absolute() and attempt.parent.is_dir() and not attempt.exists(),
                   "fresh fixed causal attempt required")
    worker.require(shutil.disk_usage(attempt.parent).free>=2*worker.EVIDENCE_BYTES,
                   "disk headroom below 2 GiB")
    worker.require(psutil.virtual_memory().available>=5*1024**3,
                   "memory headroom below 5 GiB")
    worker.require(not LOCK.exists(),"another causal checkpoint active")
    start=time.monotonic();deadline=start+worker.WALL_SECONDS
    start_utc=datetime.now(timezone.utc)
    initial=host.quick_probe(initial=True)
    worker.require(initial["monotonic_seconds"]-start<=15
                   and time.monotonic()<deadline-worker.POST_RESERVE_SECONDS,
                   "initial host probe exhausted budget")
    nonce=secrets.token_hex(16);LOCK.mkdir()
    try:attempt.mkdir()
    except BaseException:
        LOCK.rmdir()
        raise
    result={"schema":"whisper_tail_causal_parent_v2","status":"invalid_or_partial",
            "model":model,"attempt":str(attempt),"parent_pid":os.getpid(),
            "start_utc":start_utc.isoformat(),"start_monotonic":start,
            "deadline_monotonic":deadline,"nonce_sha256":worker.hashlib.sha256(nonce.encode()).hexdigest(),
            "admission_sha256":worker.sha(admission),"go_sha256":worker.sha(go),
            "source_identity":receipt["source_identity"]}
    try:
        write_once(attempt/"launch_start.json",{**result,"status":"started"})
        samples=[]
        with (attempt/"host_samples.jsonl").open("x") as host_out:
            samples.append(initial);host_out.write(canonical(initial)+"\n");host_out.flush();os.fsync(host_out.fileno())
            command=[sys.executable,"-m",
                     "models.second_architecture.nonspeech_probe_v1.whisper_tail_causal_worker_v2",
                     "run","--model",model,"--attempt",str(attempt),
                     "--admission",str(admission),"--go",str(go)]
            monitor=run_bounded(command,attempt,model,nonce=nonce,start=start,
                                deadline=deadline,samples=samples,host_out=host_out)
            worker_exit=datetime.now(timezone.utc)
            _host_sample(samples,host_out)
        monitor_end=time.monotonic()
        host.validate_cadence(samples,start=start,end=monitor_end)
        result["host_monitor_end_monotonic"]=monitor_end
        result["monitor"]=monitor
        result["worker_exit_utc"]=worker_exit.isoformat()
        result["host_samples_sha256"]=worker.sha(attempt/"host_samples.jsonl")
        result["host_samples_count"]=len(samples)
        result["progress"]=progress_census(attempt/"progress.jsonl")
        worker.require(time.monotonic()<deadline,"whole deadline before post log")
        power=host.power_log(start_utc,datetime.now(timezone.utc))
        with (attempt/"post_power_log.txt").open("x") as out:
            out.write(power);out.flush();os.fsync(out.fileno())
        result["post_power_log_sha256"]=worker.sha(attempt/"post_power_log.txt")
        result["post_power_log_collected_utc"]=datetime.now(timezone.utc).isoformat()
        if (monitor["worker_returncode"]==0 and monitor["worker_stop_reason"]=="normal_exit"
                and 0<monitor["worker_peak_tree_rss_bytes"]<worker.RSS_BYTES
                and result["progress"]=={"complete":56,"attempted_incomplete":None}
                and complete_worker_output(attempt,model)
                and bytes_used(attempt)<worker.EVIDENCE_BYTES and time.monotonic()<deadline):
            result["status"]="completed_pending_saved_audit"
    except BaseException as exc:
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
    finally:
        result["end_utc"]=datetime.now(timezone.utc).isoformat()
        result["whole_elapsed_seconds"]=time.monotonic()-start
        result["evidence_bytes_before_result"]=bytes_used(attempt)
        if result["whole_elapsed_seconds"]>=worker.WALL_SECONDS or bytes_used(attempt)>=worker.EVIDENCE_BYTES:
            result["status"]="invalid_or_partial"
        try:write_once(attempt/"parent_result.json",result)
        finally:LOCK.rmdir()
    return result


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=tuple(worker.ATTEMPTS),required=True)
    parser.add_argument("--admission",type=Path,required=True)
    parser.add_argument("--go",type=Path,required=True)
    args=parser.parse_args()
    result=launch(args.model,args.admission,args.go)
    print(canonical({"status":result["status"],"attempt":result["attempt"]}))
    if result["status"]!="completed_pending_saved_audit":raise SystemExit(2)


if __name__=="__main__":main()
