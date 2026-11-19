"""Source-only frozen arm plan for one V4 Whisper Tiny causal pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from . import natural_tail_runner_v4 as v4

HERE = Path(__file__).resolve().parent
ROOT = v4.ROOT
V4_RUNS = HERE / "runs/natural_tail_waveform_v1"
CASE = "d08_beep_train_low"
LENGTH = 32000
SEED = 0
SITES = ("post_conv_pos", "post_block0", "post_final_norm")
REGIONS = {"boundary": list(range(398,403))+list(range(498,503)),
           "tail_interior": list(range(403,498))}
KINDS = ("target_transfer", "identity_sham", "source_random",
         "norm_matched_shifted_delta")
PLAN_PATH = HERE / "WHISPER_TAIL_CAUSAL_PLAN_V1.json"
REVIEW = HERE / "WHISPER_NATURAL_TAIL_V4_INDEPENDENT_RESULT_REVIEW.md"
INVENTORY = HERE / "WHISPER_NATURAL_TAIL_V4_RAW_FILE_INVENTORY.json"
PROTOCOL = HERE / "WHISPER_TAIL_CAUSAL_PROSPECTIVE_V1.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def build() -> dict:
    inventory = json.loads(INVENTORY.read_text())
    require(inventory.get("status") == "read_only_inventory_no_inference",
            "independent inventory status changed")
    review_text = REVIEW.read_text()
    require("**PASS for both complete V4 attempts" in review_text,
            "independent V4 result review not passed")
    require(len(REGIONS["boundary"]) == 10 and len(REGIONS["tail_interior"]) == 95
            and not (set(REGIONS["boundary"]) & set(REGIONS["tail_interior"])),
            "preselected encoder row regions changed")
    random_rows = {}
    for name, rows in REGIONS.items():
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"whisper-tail-causal-v1:{CASE}:{LENGTH}:{SEED}:{name}".encode()).digest(),"big"))
        random_rows[name] = sorted(rng.sample(range(8,392),len(rows)))
    units=[]
    arms=[]
    inputs={"review_sha256":sha(REVIEW),"inventory_sha256":sha(INVENTORY),
            "protocol_sha256":sha(PROTOCOL)}
    for model in v4.MODELS:
        audit_path = V4_RUNS / f"{model}_attempt_v4_audit.json"
        attempt = v4.ATTEMPTS[model]
        records_path = attempt / "records.jsonl"
        admission_path = HERE / ("NATURAL_TAIL_MULTILINGUAL_ADMISSION_V4_2.json"
                                  if model == "whisper_multilingual" else
                                  "NATURAL_TAIL_ENGLISH_ADMISSION_V4_2.json")
        audit = json.loads(audit_path.read_text())
        require(audit.get("status") == "PASS" and audit.get("calls") == 160
                and audit.get("base_replay_failures") == [], "V4 audit not passed")
        files = {item["path"]:item for item in inventory["models"][model]["files"]}
        for path in (audit_path,records_path):
            rel=str(path.relative_to(ROOT))
            require(rel in files and files[rel]["sha256"] == sha(path)
                    and files[rel]["bytes"] == path.stat().st_size,
                    f"V4 saved input differs: {rel}")
        rows=[json.loads(line) for line in records_path.read_text().splitlines()]
        require(len(rows)==161 and rows[-1]=={"type":"complete","calls":160},
                "V4 record census differs")
        by={(r["plan"]["case_id"],r["plan"]["condition"],r["plan"]["replay"]):r
            for r in rows[:-1]}
        zero=by[(CASE,f"z{LENGTH}",False)]
        noise=by[(CASE,f"n{LENGTH}_s{SEED}",False)]
        base=by[(CASE,"base",False)]
        comparison=[c for c in audit["comparisons"] if
                    c["case_id"]==CASE and c["kind"]=="noise_vs_zero"
                    and c["length"]==LENGTH and c["seed"]==SEED]
        require(len(comparison)==1 and comparison[0]["mask_identical"]
                and base["plan"]["frames"]==128000
                and zero["plan"]["frames"]==noise["plan"]["frames"]==160000,
                "same-length pair or mask changed")
        changed = not comparison[0]["sequence_equal"]
        require(changed is (model=="whisper_multilingual")
                and all(r["output"]["termination"]=="eos" for r in (zero,noise))
                and (model!="whisper_multilingual" or
                     zero["output"]["prefix_ids"]==noise["output"]["prefix_ids"]),
                "frozen V4 contrast changed")
        unit={"model":model,"case_id":CASE,"tail_samples":LENGTH,"seed":SEED,
              "v4_changed_sequence":changed,"source_frames":128000,
              "zero":_input(zero),"noise":_input(noise),
              "random_source_rows":random_rows}
        units.append(unit)
        inputs[f"{model}_audit_sha256"]=sha(audit_path)
        inputs[f"{model}_records_sha256"]=sha(records_path)
        inputs[f"{model}_admission_sha256"]=sha(admission_path)
        for site in SITES:
            for region, target in REGIONS.items():
                for recipient,donor in (("noise","zero"),("zero","noise")):
                    for kind in KINDS:
                        arms.append({"model":model,"site":site,"region":region,
                            "recipient":recipient,"donor":donor,"kind":kind,
                            "target_rows":target,"source_rows":random_rows[region],
                            "phase":"generate_only"})
            for recipient,donor in (("noise","zero"),("zero","noise")):
                arms.append({"model":model,"site":site,"region":"all_encoder_rows",
                    "recipient":recipient,"donor":donor,"kind":"full_donor",
                    "target_rows":list(range(1500)),"source_rows":[],
                    "phase":"generate_only"})
    require(len(units)==2 and len(arms)==108
            and all(sum(a["model"]==m for a in arms)==54 for m in v4.MODELS),
            "frozen arm census changed")
    return {"schema":"whisper_tail_causal_plan_v1","status":"source_only_no_inference",
            "inputs":inputs,"sites":list(SITES),"regions":REGIONS,
            "units":units,"arms":arms,"generation_calls_per_model":56,
            "generation_calls_total":112}


def _input(record: dict) -> dict:
    return {"wav":record["plan"]["wav"],"wav_sha256":record["plan"]["wav_sha256"],
            "frames":record["plan"]["frames"],
            "sequence_ids":record["output"]["sequence_ids"],
            "prefix_ids":record["output"]["prefix_ids"],
            "termination":record["output"]["termination"],
            "mel":record["sidecars"]["mel"],
            "mask":record["sidecars"]["mask"],
            "first_raw_logits":record["sidecars"]["first_raw_logits"],
            "first_processed_scores":record["sidecars"]["first_processed_scores"]}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=PLAN_PATH)
    args=parser.parse_args()
    require(not args.output.exists(), "source-only plan already exists")
    args.output.write_text(json.dumps(build(),sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({"status":"source_only_no_inference","path":str(args.output),
                      "sha256":sha(args.output)}))


if __name__=="__main__":
    main()
