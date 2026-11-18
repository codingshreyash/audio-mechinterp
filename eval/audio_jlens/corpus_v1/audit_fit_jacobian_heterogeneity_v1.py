"""Read-only CPU audit of the completed 48-context FIT Jacobian diagnostic.

Do not run against an active/partial job. This loads saved FIT tensors only;
it does not load the Qwen model or open development/locked-test WAVs/answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PREFIX = HERE / "fit_jacobian_heterogeneity_v1"
SCRIPT = HERE / "fit_jacobian_heterogeneity_v1.py"
PROTOCOL = HERE / "FIT_JACOBIAN_HETEROGENEITY_PROTOCOL_V1.md"
STAGE1 = HERE / "fit_rank16_transport_v1.stage1.pt"
LENS = HERE / "fit_rank16_transport_v1.lens.pt"
CALIBRATION = HERE / "fit_uniform_calibration_v1.raw.pt"
EXPECTED_SCRIPT_SHA = "b1b9e023394a517aabe4e56404ed83ba215c0ad8e8665a771aee3915f07df3e4"
EXPECTED_PROTOCOL_SHA = "69ea797ee956d16f78b014df18621f206d1279ebe978c96a005e041675e51ba9"
EXPECTED_LENS_SHA = "fef84a20f437c37ca665e4026bb3645ccd558e549c7ffe2adff20e6a867ba12c"
EXPECTED_STAGE1_SHA = "8276aca0e24fcaa1397693e0353f9c7ee59190036f3516c0c13ebcbc6208b641"
EXPECTED_CALIBRATION_SHA = "76085f0dab8f37b6052024b582ff34533cac3d1c579b27b39a631c536d9cd110"
SITES = ("encoder31", "connector")
N = 48
RANK = 16
ALPHA = .025
MEMORY_CAP = 30 * 1024**3
WALL_CAP = 1200


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def near(a, b, *, tol=1e-5):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol * max(1.0, abs(a), abs(b))


def tensor_sha(value: torch.Tensor) -> str:
    x = value.detach().cpu().contiguous()
    if x.dtype == torch.bfloat16:
        x = x.view(torch.uint16)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def terminal_metrics(predicted: torch.Tensor, actual: torch.Tensor, basis: torch.Tensor) -> dict:
    projected = basis @ (basis.T @ actual)
    observed_norm = float(actual.norm())
    projected_norm = float(projected.norm())
    predicted_norm = float(predicted.norm())
    return {
        "full_terminal_l2_error": float((predicted-actual).norm()),
        "projected_terminal_l2_error": float((predicted-projected).norm()),
        "rank16_capture_fraction": projected_norm**2/observed_norm**2 if observed_norm > 0 else None,
        "projected_cosine": float(torch.dot(predicted,projected)/(predicted_norm*projected_norm))
                            if predicted_norm > 0 and projected_norm > 0 else None,
    }


def describe(values: list[float]) -> dict:
    x = torch.tensor(values, dtype=torch.float64)
    return {"n": len(values), "mean": float(x.mean()),
            "median": float(torch.quantile(x,.5)), "p10": float(torch.quantile(x,.1)),
            "p90": float(torch.quantile(x,.9))}


def similarity(local: torch.Tensor, cases: list[dict]) -> dict:
    flat = local.flatten(1).double()
    flat = flat / flat.norm(dim=1)[:,None]
    groups = defaultdict(list)
    for i in range(N):
        for j in range(i+1,N):
            a,b = cases[i],cases[j]
            value = float(torch.dot(flat[i],flat[j]))
            if a["pair_id"] == b["pair_id"]:
                key = "within_pair_same_voice" if a["engine"] == b["engine"] else "within_pair_cross_voice"
            elif a["category"] == b["category"]:
                key = "between_pair_same_category"
            else:
                key = "across_categories"
            groups[key].append(value)
            groups["same_voice_all" if a["engine"] == b["engine"] else "cross_voice_all"].append(value)
    counts = {"within_pair_same_voice":24,"within_pair_cross_voice":48,
              "between_pair_same_category":192,"across_categories":864,
              "same_voice_all":552,"cross_voice_all":576}
    check(sum(counts[k] for k in list(counts)[:4]) == 1128, "similarity partition math")
    check({k:len(groups[k]) for k in counts} == counts, "similarity strata counts")
    return {k:describe(groups[k]) for k in counts}


def audit(prefix: Path) -> dict:
    result_path = prefix.with_suffix(".result.pt")
    checkpoint_path = prefix.with_suffix(".checkpoint.pt")
    report_path = prefix.with_suffix(".report.json")
    memory_path = prefix.with_suffix(".memory.jsonl")
    for path in (result_path, checkpoint_path, report_path, memory_path):
        check(path.is_file(), f"completed artifact missing: {path}")
    check(sha(SCRIPT)==EXPECTED_SCRIPT_SHA and sha(PROTOCOL)==EXPECTED_PROTOCOL_SHA,
          "frozen runner/protocol SHA changed")
    check(sha(LENS)==EXPECTED_LENS_SHA and sha(STAGE1)==EXPECTED_STAGE1_SHA and
          sha(CALIBRATION)==EXPECTED_CALIBRATION_SHA,"frozen FIT source SHA changed")
    report=json.loads(report_path.read_text())
    result=torch.load(result_path,map_location="cpu",weights_only=True)
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    stage1=torch.load(STAGE1,map_location="cpu",weights_only=True)
    lens=torch.load(LENS,map_location="cpu",weights_only=True)
    calibration=torch.load(CALIBRATION,map_location="cpu",weights_only=True)
    check(report["result_sha256"]==sha(result_path) and
          report["checkpoint_sha256"]==sha(checkpoint_path) and
          report["memory_sha256"]==sha(memory_path),"terminal artifact hash mismatch")
    meta=report["metadata"]
    check(result["metadata"]==checkpoint["metadata"]==meta,"checkpoint/result/report metadata mismatch")
    check(meta["script_sha256"]==EXPECTED_SCRIPT_SHA and
          meta["protocol_sha256"]==EXPECTED_PROTOCOL_SHA and
          meta["lens_sha256"]==EXPECTED_LENS_SHA and
          meta["stage1_sha256"]==EXPECTED_STAGE1_SHA and
          meta["calibration_sha256"]==EXPECTED_CALIBRATION_SHA,
          "metadata source hash mismatch")
    check(meta["planned_fit_contexts"]==N and meta["rank"]==RANK and
          meta["alpha"]==ALPHA and meta["max_mps_driver_bytes"]==MEMORY_CAP and
          meta["max_wall_seconds_per_invocation"]==WALL_CAP,
          "metadata policy changed")
    ids=meta["case_ids"]
    cases=[]
    for case_id in ids:
        parts=case_id.split("/")
        check(len(parts)==4 and parts[1]=="fit_chosen" and parts[2] in ("piper","kokoro"),
              "non-FIT case identifier")
        cases.append({"case_id":case_id,"pair_id":parts[0],"category":parts[0].split("_")[0],
                      "engine":parts[2],"word":parts[3]})
    check(len(ids)==N and len(set(ids))==N and len({c["pair_id"] for c in cases})==12,
          "wrong 48-case FIT set")
    check(all(sum(c["pair_id"]==p for c in cases)==4 for p in {c["pair_id"] for c in cases}),
          "wrong pair context balance")
    check(ids==[r["case_id"] for r in stage1["records"]]==
          [r["case_id"] for r in calibration["records"]],"FIT case order mismatch")
    records=result["records"]
    check(len(records)==N and len(checkpoint["records"])==N and
          [r["case_id"] for r in records]==ids,"result/checkpoint case count/order")
    for i,(rec,ck,src,cal) in enumerate(zip(records,checkpoint["records"],stage1["records"],calibration["records"],strict=True)):
        check(rec["case_id"]==ck["case_id"]==src["case_id"]==cal["case_id"],f"case {i} ID mismatch")
        check(rec["audio_sha256"]==ck["audio_sha256"]==src["audio_sha256"]==cal["audio_sha256"],
              f"case {i} audio hash mismatch")
        check(0 <= rec["grad_vs_no_grad_terminal_max_abs"] <= 1e-3,
              f"case {i} grad/no-grad parity failed")
        times=rec["vjp_seconds"]
        check(len(times)==16 and all(math.isfinite(t) and t>=0 for t in times),
              f"case {i} VJP timing invalid")
        for site,width in (("encoder31",1280),("connector",2048)):
            matrix=rec["jacobians"][site]
            check(torch.equal(matrix,ck["jacobians"][site]),f"case {i}/{site} checkpoint matrix drift")
            check(tuple(matrix.shape)==(6,16,width) and matrix.dtype==torch.float32 and
                  bool(torch.isfinite(matrix).all()) and float(matrix.norm())>0,
                  f"case {i}/{site} Jacobian invalid")
            check(all(float(matrix[b].norm())>0 for b in range(6)),f"case {i}/{site} zero bin")
            check(rec["source_rows_per_bin"][site]==list(map(len,src["row_indices"][site])) and
                  rec["source_rows_per_bin"][site]==ck["source_rows_per_bin"][site],
                  f"case {i}/{site} source row mismatch")
            check(cal["sites"][site]["source_rows"]==src["row_indices"][site] and
                  cal["sites"][site]["source_shape"]==src["source_shapes"][site],
                  f"case {i}/{site} calibration source geometry drift")
    basis=lens["basis"].float()
    check(tuple(basis.shape)==(2048,16) and bool(torch.isfinite(basis).all()),"basis invalid")
    summaries={}
    for site in SITES:
        saved=result["site_analysis"][site]
        local=torch.stack([r["jacobians"][site] for r in records])
        check(saved["local_matrix_shape"]==list(local.shape),f"{site} local shape report")
        check(saved["matrix_hashes"]==[tensor_sha(x) for x in local],f"{site} matrix hash mismatch")
        mean=local.double().mean(0).float()
        frozen=lens["jacobians"][site].float()
        difference=float((mean-frozen).abs().max())
        check(difference<=1e-3 and near(saved["recomputed_mean_vs_frozen_max_abs"],difference, tol=1e-6) and
              near(report["mean_vs_frozen_max_abs"][site],difference,tol=1e-6),
              f"{site} corpus average reconstruction mismatch")
        ids_by_pair={p:[j for j,c in enumerate(cases) if c["pair_id"]==p] for p in {c["pair_id"] for c in cases}}
        total=local.double().sum(0)
        lopo={p:((total-local[indices].double().sum(0))/44).float() for p,indices in ids_by_pair.items()}
        metrics_by_name=defaultdict(lambda:defaultdict(list))
        pair_by_name=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,(case,row) in enumerate(zip(cases,saved["rows"],strict=True)):
            cal=calibration["records"][i]["sites"][site]
            check(row["case_id"]==case["case_id"] and row["pair_id"]==case["pair_id"] and
                  row["engine"]==case["engine"] and row["category"]==case["category"] and
                  row["audio_sha256"]==records[i]["audio_sha256"],f"{site}/{i} analysis row identity")
            check(near(row["measured_v2_delta_score"],cal["measured_v2_delta_score"]),
                  f"{site}/{i} measured FIT score drift")
            h=stage1["records"][i]["target_state"].float()
            check(h.shape==(2048,),f"{site}/{i} target shape")
            actual=cal["terminal_delta_per_alpha"].float()*ALPHA
            delta=cal["delta_bins"].float()
            models={"local":local[i],"frozen_average":frozen,
                    "pair_heldout_average":lopo[case["pair_id"]]}
            for name,matrix in models.items():
                coords=torch.einsum("bkd,bd->k",matrix,delta)*ALPHA
                predicted=basis@coords
                expected=terminal_metrics(predicted,actual,basis)
                pred=row["predictions"][name]
                for metric,value in expected.items():
                    check(near(pred[metric],value,tol=2e-5),f"{site}/{i}/{name}/{metric} arithmetic")
                check(near(pred["score_absolute_error"],abs(pred["score_delta"]-cal["measured_v2_delta_score"])),
                      f"{site}/{i}/{name} score error arithmetic")
                if name=="frozen_average":
                    check(near(pred["score_delta"],cal["map_predicted_v2_delta_score"],tol=1e-3),
                          f"{site}/{i} frozen score cross-check")
                for metric in ("score_absolute_error","full_terminal_l2_error","projected_terminal_l2_error"):
                    metrics_by_name[name][metric].append(pred[metric])
                    pair_by_name[case["pair_id"]][name][metric].append(pred[metric])
        for name,metrics in metrics_by_name.items():
            for metric,values in metrics.items():
                check(len(values)==N and near(saved["metrics"][name][{"score_absolute_error":"score_mae",
                     "full_terminal_l2_error":"full_terminal_l2_mean",
                     "projected_terminal_l2_error":"projected_terminal_l2_mean"}[metric]],sum(values)/N),
                     f"{site}/{name}/{metric} summary")
            for pair in ids_by_pair:
                pm=pair_by_name[pair][name]
                for metric,key in (("score_absolute_error","score_mae"),
                                   ("projected_terminal_l2_error","projected_terminal_l2_mean")):
                    check(len(pm[metric])==4 and near(saved["pair_metrics"][pair][name][key],sum(pm[metric])/4),
                          f"{site}/{pair}/{name}/{metric} pair summary")
        counts=torch.tensor([r["source_rows_per_bin"][site] for r in records],dtype=torch.float32)
        for key,tensor in (("similarity_uniform_edit_derivative",local),
                           ("similarity_per_row_mean_gradient",local/counts[:,:,None,None])):
            expected=similarity(tensor,cases)
            for group,stats in expected.items():
                for metric,value in stats.items():
                    check(near(saved[key][group][metric],value) and
                          near(report[key][site][group][metric],value),
                          f"{site}/{key}/{group}/{metric} similarity")
        check(saved["metrics"]==report["metrics"][site] and
              saved["pair_metrics"]==report["pair_metrics"][site],f"{site} report/result summary mismatch")
        summaries[site]={name:saved["metrics"][name] for name in ("local","frozen_average","pair_heldout_average")}
    memory=[json.loads(line) for line in memory_path.read_text().splitlines() if line.strip()]
    phases=[m["phase"] for m in memory]
    check(phases.count("after_analysis")==1,"final analysis memory phase missing")
    check(sum(p.startswith("after_model_load") for p in phases)>=1,"model-load memory phase missing")
    for i,case in enumerate(cases):
        for k in range(RANK):
            check(phases.count(f"local_{case['case_id']}_coordinate_{k}")==1,
                  f"case {i} coordinate {k} memory phase missing/duplicate")
        check(phases.count(f"after_case_{i+1}_checkpoint")==1,f"case {i} checkpoint phase missing/duplicate")
    peak=max(int(m["driver_allocated_bytes"]) for m in memory)
    check(all(m["limit_driver_bytes"]==MEMORY_CAP and not m["over_limit"] and
              m["driver_allocated_bytes"]<MEMORY_CAP for m in memory),"MPS memory cap/log inconsistency")
    # The clock is checked in runner code at operation boundaries. The log timestamps
    # can check post-load segments, not preload time or a single uninterruptible kernel.
    starts=[i for i,p in enumerate(phases) if p=="after_model_load"]
    segments=[memory[a:b] for a,b in zip(starts,starts[1:]+[len(memory)],strict=True)]
    max_logged_segment=max(seg[-1]["unix_time"]-seg[0]["unix_time"] for seg in segments)
    check(max_logged_segment<WALL_CAP,"logged post-load invocation exceeded 20 minutes")
    return {"status":"passed_read_only_fit_audit","result_sha256":sha(result_path),
            "checkpoint_sha256":sha(checkpoint_path),"memory_sha256":sha(memory_path),
            "contexts":len(records),"vjps":sum(len(r["vjp_seconds"]) for r in records),
            "matrix_shapes":{s:list(records[0]["jacobians"][s].shape) for s in SITES},
            "peak_mps_driver_bytes":peak,"max_logged_postload_segment_seconds":max_logged_segment,
            "metrics":summaries,
            "limits":"saved scalar score deltas cannot be independently recomputed from artifacts without pinned final norm/head weights; score error and frozen calibration cross-checks audited"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix",type=Path,default=PREFIX)
    args=parser.parse_args()
    print(json.dumps(audit(args.prefix),indent=2))


if __name__=="__main__":
    main()
