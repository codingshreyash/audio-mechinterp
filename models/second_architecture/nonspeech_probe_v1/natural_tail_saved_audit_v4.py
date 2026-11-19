"""Saved-evidence audit; reads no model weights and changes no attempt files."""
from __future__ import annotations

import argparse
import json
import math
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from . import natural_tail_runner_v4 as worker
from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate

CONDITIONS = ("base", "z8000", "n8000_s0", "n8000_s1",
              "z32000", "n32000_s0", "n32000_s1")


def must(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_sidecar(attempt: Path, ref: dict, shape: tuple, dtype: str) -> np.ndarray:
    name = ref.get("file")
    must(isinstance(name, str) and name == Path(name).name and name.endswith(".npy"),
         "sidecar name invalid")
    path = attempt/"sidecars"/name
    must(path.is_file() and not path.is_symlink() and worker.old.sha_file(path) == ref["sha256"],
         f"sidecar bytes changed: {name}")
    array = np.load(path, allow_pickle=False)
    must(array.shape == shape and array.dtype == np.dtype(dtype)
         and ref["shape"] == list(shape) and ref["dtype"] == dtype
         and worker.old.sha_bytes(np.ascontiguousarray(array).tobytes())
         == ref["values_sha256"], f"sidecar shape/hash changed: {name}")
    return array


def _expected_plan(model: str) -> list[dict]:
    # Separate reconstruction from the manifest, without calling runner.schedule.
    manifest = worker.RENDER/"manifest.json"
    must(worker.old.sha_file(manifest) == worker.RENDER_SHA256, "render changed")
    rows = json.loads(manifest.read_text())["rows"]
    must(len(rows) == 140, "render row count")
    cases = {}
    for row in rows:
        cases.setdefault(row["case_id"], []).append(row)
    must(len(cases) == 20 and all([x["condition"] for x in group] == list(CONDITIONS)
                                  for group in cases.values()), "render order")
    selected = rows+[group[0] for group in cases.values()]
    return [{"index":i+1, "model":model, "case_id":r["case_id"],
             "condition":r["condition"], "replay":i >= 140,
             "wav":r["wav"], "wav_sha256":r["wav_sha256"],
             "pcm_sha256":r["pcm_sha256"], "frames":r["frames"]}
            for i,r in enumerate(selected)]


def _pcm(row: dict) -> np.ndarray:
    path = worker.RENDER/row["wav"]
    must(path.is_file() and not path.is_symlink()
         and worker.old.sha_file(path) == row["wav_sha256"], "WAV changed")
    with wave.open(str(path), "rb") as audio:
        must((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
              audio.getnframes()) == (1,2,16000,row["frames"]), "WAV format changed")
        raw = audio.readframes(audio.getnframes())
    must(worker.old.sha_bytes(raw) == row["pcm_sha256"], "PCM changed")
    return np.frombuffer(raw, dtype="<i2")


def _regions(a: np.ndarray, b: np.ndarray, source_end: int,
             a_end: int, b_end: int) -> dict:
    must(a.shape == b.shape == (80,3000), "mel pair shape")
    boundary = set()
    for point in (source_end, a_end, b_end):
        boundary.update(range(max(0,point-5), min(3000,point+6)))
    zones = {"source_interior":range(0,source_end),
             "appended_or_duration_interior":range(source_end,max(a_end,b_end)),
             "nominal_padding_interior":range(max(a_end,b_end),3000),
             "boundary":sorted(boundary)}
    result = {}
    for name, frames in zones.items():
        selected = [x for x in frames if x not in boundary or name == "boundary"]
        delta = a[:,selected].astype(np.float64)-b[:,selected].astype(np.float64)
        result[name] = {"frames":len(selected),
                        "changed_frames":int(np.count_nonzero(np.any(delta != 0,axis=0)))
                        if selected else 0,
                        "max_abs":float(np.max(np.abs(delta))) if selected else 0.0,
                        "l2":float(np.linalg.norm(delta))}
    return result


def audit(attempt: Path, admission: Path, go: Path, model: str) -> dict:
    must(model in worker.MODELS and attempt == worker.ATTEMPTS[model]
         and attempt.is_dir() and not attempt.is_symlink(),
         "fixed V4 attempt unavailable")
    receipt = worker.validate_go(admission, go, model)
    plan = _expected_plan(model)
    must(receipt["schedule"] == plan, "frozen plan differs")
    parent = json.loads((attempt/"parent_result.json").read_text())
    must(parent["status"] == "completed_pending_saved_audit"
         and parent["model"] == model and parent["attempt"] == str(attempt)
         and parent["source_identity"] == receipt["source_identity"]
         and parent["admission_sha256"] == worker.old.sha_file(admission)
         and parent["go_sha256"] == worker.old.sha_file(go)
         and parent["progress"]["complete"] == 160
         and parent["whole_elapsed_seconds"] < worker.WALL_SECONDS
         and 0 < parent["worker_seconds_granted"]
         < worker.WALL_SECONDS-480
         and parent["evidence_bytes_before_result"] < worker.EVIDENCE_BYTES,
         "parent attempt not eligible")
    launch=json.loads((attempt/"launch_start.json").read_text())
    must(launch["status"] == "started"
         and {k:launch[k] for k in ("schema","model","attempt","start_utc",
             "parent_pid","deadline_monotonic","nonce_sha256",
             "admission_sha256","go_sha256","source_identity")}
         == {k:parent[k] for k in ("schema","model","attempt","start_utc",
             "parent_pid","deadline_monotonic","nonce_sha256",
             "admission_sha256","go_sha256","source_identity")},
         "live parent launch receipt differs")
    monitor=parent["monitor"]
    must(monitor["worker_returncode"] == 0
         and monitor["worker_stop_reason"] == "normal_exit"
         and not monitor["worker_timed_out"]
         and 0 < monitor["worker_peak_tree_rss_bytes"] < worker.RSS_BYTES
         and monitor["worker_rss_samples"] > 0,
         "parent worker resource verdict invalid")
    must(monitor["parent_heartbeat_count"] > 0,
         "bounded parent heartbeat missing")
    samples_path=attempt/"worker_rss_samples.jsonl"
    must(samples_path.is_file() and not samples_path.is_symlink()
         and worker.old.sha_file(samples_path) == monitor["worker_rss_sample_log_sha256"],
         "resource sample log changed")
    samples=[json.loads(line) for line in samples_path.read_text().splitlines()]
    must(len(samples) == monitor["worker_rss_samples"]
         and [x["sample_index"] for x in samples] == list(range(1,len(samples)+1))
         and max(x["tree_rss_bytes"] for x in samples)
         == monitor["worker_peak_tree_rss_bytes"]
         and all(0 < x["tree_rss_bytes"] < worker.RSS_BYTES
                 and x["evidence_bytes"] < worker.EVIDENCE_BYTES
                 for x in samples), "resource samples differ")
    pre=json.loads((attempt/"pre_host.json").read_text())
    post=json.loads((attempt/"post_host.json").read_text())
    host_gate.validate_power(pre)
    host_gate.validate_power(post)
    host_gate.load_archive(attempt,pre,"pre")
    post_log=host_gate.load_archive(attempt,post,"post")
    started=datetime.fromisoformat(parent["start_utc"])
    collected=datetime.fromisoformat(post["collected_utc"])
    host_gate.validate_log(post_log,started,collected,collected)
    lag=(collected-datetime.fromisoformat(parent["worker_exit_utc"])).total_seconds()
    must(0 <= lag <= 120 and math.isclose(lag,parent["post_probe_lag_seconds"],abs_tol=1e-3),
         "post-host lag differs")
    must(sum(x.stat().st_size for x in attempt.rglob("*") if x.is_file())
         < worker.EVIDENCE_BYTES, "whole evidence cap")
    progress = [json.loads(line) for line in (attempt/"progress.jsonl").read_text().splitlines()]
    must(progress == [event for r in plan for event in
                      ({"type":"attempt","index":r["index"]},
                       {"type":"complete","index":r["index"]})], "call progress differs")
    records = [json.loads(line) for line in (attempt/"records.jsonl").read_text().splitlines()]
    must(len(records) == 161 and records[-1] == {"type":"complete","calls":160},
         "record count or completion differs")
    by_key = {}
    pcm_by_key = {}
    generation_config=json.loads((Path(receipt["assets"]["snapshot"])/
                                  "generation_config.json").read_text())
    for row, record in zip(plan, records[:-1], strict=True):
        must(record["type"] == "call" and record["plan"] == row,
             "call order/identity differs")
        refs = record["sidecars"]
        must(set(refs) == {"mel","mask","first_raw_logits",
                           "first_processed_scores","first_decoder_state"},
             "sidecar inventory differs")
        mel = load_sidecar(attempt,refs["mel"],(80,3000),"float32")
        mask = load_sidecar(attempt,refs["mask"],(3000,),"int64")
        vocab = 51865 if model == "whisper_multilingual" else 51864
        raw = load_sidecar(attempt,refs["first_raw_logits"],(vocab,),"float32")
        scores = load_sidecar(attempt,refs["first_processed_scores"],(vocab,),"float32")
        load_sidecar(attempt,refs["first_decoder_state"],
                     (1,4 if model == "whisper_multilingual" else 2,384),"float32")
        must(np.isfinite(mel).all() and np.isfinite(raw).all()
             and np.isfinite(scores).any(), "nonfinite processor/raw/processed evidence")
        active = (row["frames"]+159)//160
        must(np.array_equal(mask[:active],np.ones(active,dtype=np.int64))
             and np.array_equal(mask[active:],np.zeros(3000-active,dtype=np.int64)),
             "attention mask geometry differs")
        must(record["geometry"]["active_mel_frames"] == active
             and record["geometry"]["mask_sha256"]
             == worker.old.sha_bytes(bytes(mask.astype(np.uint8))),
             "attention mask summary differs")
        out = record["output"]
        ids,prefix,generated = (out[k] for k in
                                 ("sequence_ids","prefix_ids","generated_ids"))
        must(ids == prefix+generated and len(generated) == out["actual_generation_steps"]
             == len(out["score_rows"]) and 1 <= len(generated) <= 128,
             "complete output sequence differs")
        eos = 50257 if model == "whisper_multilingual" else 50256
        must((generated[-1] == eos) == out["eos_seen"] and
             (out["termination"] == "eos") == out["eos_seen"] and
             (out["termination"] == "length_censored") == out["censored"] and
             (out["eos_seen"] or len(generated) == 128) and
             eos not in generated[:-1], "EOS/censor state differs")
        must(all(isinstance(step["token_id"],int)
                 and step["token_id"] == generated[i]
                 and isinstance(step["raw_logits_sha256"],str)
                 and len(step["raw_logits_sha256"]) == 64
                 and isinstance(step["processed_logits_sha256"],str)
                 and len(step["processed_logits_sha256"]) == 64
                 and math.isfinite(step["raw_selected_logit"])
                 and math.isfinite(step["selected_logit"])
                 for i,step in enumerate(out["score_rows"])),
             "per-step selected score evidence differs")
        first = out["score_rows"][0]
        token = generated[0]
        must(first["token_id"] == token
             and first["raw_logits_sha256"] == worker.old.sha_bytes(raw.tobytes())
             and first["processed_logits_sha256"] == worker.old.sha_bytes(scores.tobytes())
             and first["raw_top_id"] == int(raw.argmax())
             and first["top_id"] == int(scores.argmax())
             and math.isclose(first["raw_selected_logit"],float(raw[token]),abs_tol=1e-5)
             and math.isclose(first["selected_logit"],float(scores[token]),abs_tol=1e-5),
             "first-step vector/score mismatch")
        kwargs=record["generation_kwargs"]
        must(("task" in kwargs and kwargs["task"] == "transcribe"
              and "language" in kwargs and kwargs["language"] is None)
             if model == "whisper_multilingual" else
             ("task" not in kwargs and "language" not in kwargs),
             "generation call signature differs")
        detection=out["language_detection"]
        if model == "whisper_multilingual":
            candidate=detection["language_logits"] if isinstance(detection,dict) else {}
            language_ids=generation_config["lang_to_id"]
            chosen_name=min(candidate,key=lambda name:(-candidate[name],language_ids[name])) if candidate else None
            must(isinstance(detection,dict)
                 and len(detection["chosen_language_ids"]) == 1
                 and prefix[1] == detection["chosen_language_ids"][0]
                 and set(candidate) == set(language_ids)
                 and len(candidate) == detection["candidate_count"]
                 and language_ids[chosen_name] == prefix[1]
                 and prefix == [generation_config["decoder_start_token_id"],
                                prefix[1],generation_config["task_to_id"]["transcribe"],
                                generation_config["no_timestamps_token_id"]],
                 "multilingual detection/prefix differs")
        else:
            must(detection is None and "detect" not in record["capture_counts"]
                 and prefix == [generation_config["decoder_start_token_id"],
                                generation_config["no_timestamps_token_id"]],
                 "English-only detection appeared")
        must(math.isclose(record["derived_preclamp_log_spec_max"],
                          float(mel.max())*4-4,abs_tol=1e-5)
             and math.isclose(record["derived_preclamp_floor"],
                              float(mel.max())*4-12,abs_tol=1e-5),
             "log-spectrogram max/floor differs")
        key=(row["case_id"],row["condition"],row["replay"])
        by_key[key]={"mel":mel,"mask":mask,"output":out,
                     "raw":raw,"scores":scores,"row":row}
        pcm_by_key[key]=_pcm(row)
    parity_failures=[]
    for case in {r["case_id"] for r in plan}:
        base=by_key[(case,"base",False)]
        replay=by_key[(case,"base",True)]
        if (base["output"]["sequence_ids"] != replay["output"]["sequence_ids"]
            or not np.allclose(base["raw"],replay["raw"],atol=1e-5,rtol=1e-5)
            or not np.allclose(base["scores"],replay["scores"],atol=1e-5,rtol=1e-5)):
            parity_failures.append(case)
    comparisons=[]
    for case in {r["case_id"] for r in plan}:
        base=by_key[(case,"base",False)]
        source_end=(base["row"]["frames"]+159)//160
        for length in (8000,32000):
            zero=by_key[(case,f"z{length}",False)]
            zero_end=(zero["row"]["frames"]+159)//160
            a=pcm_by_key[(case,"base",False)]
            b=pcm_by_key[(case,f"z{length}",False)]
            padded_identical=np.array_equal(np.pad(a,(0,480000-len(a))),
                                            np.pad(b,(0,480000-len(b))))
            must(padded_identical, "zero tail padded PCM unexpectedly changed")
            comparisons.append({"case_id":case,"kind":"zero_vs_base",
                                "length":length,"padded_pcm_identical":True,
                                "mask_identical":bool(np.array_equal(base["mask"],zero["mask"])),
                                "mel_regions":_regions(base["mel"],zero["mel"],
                                                       source_end,source_end,zero_end),
                                "sequence_equal":base["output"]["sequence_ids"]
                                ==zero["output"]["sequence_ids"]})
            for seed in (0,1):
                noise=by_key[(case,f"n{length}_s{seed}",False)]
                comparisons.append({"case_id":case,"kind":"noise_vs_zero",
                                    "length":length,"seed":seed,
                                    "mask_identical":bool(np.array_equal(noise["mask"],zero["mask"])),
                                    "mel_regions":_regions(noise["mel"],zero["mel"],
                                                           source_end,zero_end,zero_end),
                                    "sequence_equal":noise["output"]["sequence_ids"]
                                    ==zero["output"]["sequence_ids"]})
    primary_zero=sum(x["kind"]=="zero_vs_base" and not x["case_id"].endswith("_silence")
                     for x in comparisons)
    primary_noise=sum(x["kind"]=="noise_vs_zero" and not x["case_id"].endswith("_silence")
                      for x in comparisons)
    must((len(comparisons),primary_zero,primary_noise) == (120,36,72),
         "comparison denominator changed")
    return {"schema":"whisper_natural_tail_saved_audit_v4",
            "status":"PASS" if not parity_failures else "HOLD_REPLAY_PARITY",
            "model":model,"attempt":str(attempt),"calls":160,
            "base_replay_failures":parity_failures,
            "all_zero_vs_base":40,"all_noise_vs_zero":80,
            "primary_zero_vs_base":primary_zero,"primary_noise_vs_zero":primary_noise,
            "comparisons":comparisons}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--attempt",type=Path,required=True)
    parser.add_argument("--admission",type=Path,required=True)
    parser.add_argument("--go",type=Path,required=True)
    parser.add_argument("--model",choices=worker.MODELS,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    must(not args.output.exists() and args.attempt not in args.output.parents,
         "audit output must be new and outside immutable attempt")
    result=audit(args.attempt,args.admission,args.go,args.model)
    worker.write_once(args.output,result)
    print(worker.canonical({k:result[k] for k in ("status","model","calls")}))


if __name__ == "__main__":
    main()
