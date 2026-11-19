"""Source-only design for the exploratory corpus-v1 development Wav2Vec2 assay.

No inference is allowed until the V2 plan, this runner and its auditor receive
independent GO reviews and a hash-bound freeze receipt. Locked WAVs are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

import wav2vec2_fit_baseline_stage_capture_v1 as base
import wav2vec2_fit_complement_patch_v1 as fit

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = ROOT / "eval/audio_jlens/corpus_v1/SPEC_V1.json"
SPEC_SHA = "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265"
PLAN = HERE / "WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md"
AUDITOR = HERE / "wav2vec2_dev_word_disjoint_audit_v1.py"
SCHEMA = "wav2vec2_dev_word_disjoint_v1"
SITE = "post_block11"
ARMS = ("self", "W", "O", "full", "random_W", "random_O")
SEED_BASE = 16048191
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 512 * 1024**2
RSS_RESERVE = 1024**3
OUTPUT_RESERVE = 32 * 1024**2


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return base.sha(path)


def canonical(value: Any) -> str:
    return base.canonical(value)


def tokens(value: str) -> list[str]:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", value).casefold()).split()


def source_commit(path: Path) -> str:
    relative = str(path.resolve().relative_to(ROOT))
    for args in (("git", "diff", "--quiet", "--", relative),
                 ("git", "diff", "--cached", "--quiet", "--", relative)):
        require(subprocess.run(args, cwd=ROOT, check=False).returncode == 0,
                f"uncommitted source changes: {relative}")
    revision = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True
    ).strip()
    require(len(revision) == 40, f"source not committed: {relative}")
    return revision


def trusted_dev_cases() -> tuple[list[base.Case], dict[tuple[str, str], dict[str, float]], dict[str, list[str]]]:
    """Read only development manifest rows and their two WAVs per pair."""
    require(sha(SPEC) == SPEC_SHA and sha(base.MANIFEST) == base.MANIFEST_SHA,
            "corpus specification or manifest changed")
    spec = json.loads(SPEC.read_text())
    template = next(x for x in spec["templates"]["development"] if x["id"] == "dev_listen")
    manifest = json.loads(base.MANIFEST.read_text())
    pairs = [x for x in manifest["pairs"] if x["split"] == "development"
             and x["phrasing_id"] == "dev_listen"]
    require(len(pairs) == 24, "development pair/voice coverage differs")
    require({x["engine"] for x in pairs} == {"piper", "kokoro"}, "engine set differs")
    require(len({x["pair_id"] for x in pairs}) == 12, "development pair count differs")
    seen: set[tuple[str, str]] = set()
    cases: list[base.Case] = []
    stats: dict[tuple[str, str], dict[str, float]] = {}
    references: dict[str, list[str]] = {}
    for pair in sorted(pairs, key=lambda p: (p["pair_id"], p["engine"])):
        pair_id, engine = pair["pair_id"], pair["engine"]
        require((pair_id, engine) not in seen and pair.get("map_fit") is False,
                "duplicate or FIT development row")
        seen.add((pair_id, engine))
        words = pair["words"]
        require(len(words) == 2 and len(set(words)) == 2, "word pair invalid")
        slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
        require(pair["outside_slot_pcm_identical"] is True
                and pair["inside_slot_pcm_differs"] is True,
                "PCM contrast claims absent")
        pair_cases = []
        for word in words:
            item = pair["outputs"][word]
            path = ROOT / item["path"]
            require(path.resolve().is_relative_to(ROOT / "eval/audio_jlens/corpus_v1/rendered/clips"),
                    "WAV path escaped render directory")
            case = base.Case(pair_id, engine, word, path, item["file_sha256"],
                             item["pcm_int16_sha256"], int(item["frames"]), slot)
            pair_cases.append(case)
            cases.append(case)
            references[case.key] = tokens(" ".join((template["prefix"], word, template["suffix"])))
            require(references[case.key].count(word) == 1, "reference word count invalid")
        a, b = [base.load_pcm(case) for case in pair_cases]
        s, e = slot
        require(0 < s < e < len(a) == len(b)
                and np.array_equal(a[:s], b[:s]) and np.array_equal(a[e:], b[e:])
                and not np.array_equal(a[s:e], b[s:e]), "actual paired PCM contrast invalid")
        outside = np.concatenate((a[:s], a[e:])).astype(np.float32) / np.float32(32768)
        mean, var = np.float32(outside.mean()), np.float32(outside.var())
        require(bool(np.isfinite(mean) and np.isfinite(var) and var > 0),
                "outside-stat normalization invalid")
        stats[(pair_id, engine)] = {"mean": float(mean), "variance": float(var)}
    require(len(cases) == 48 and len(stats) == 24 and len(references) == 48,
            "development case matrix incomplete")
    return cases, stats, references


def directions(cases: list[base.Case]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[base.Case]] = {}
    for case in cases:
        grouped.setdefault((case.pair_id, case.engine), []).append(case)
    out = []
    for (pair_id, engine), pair in sorted(grouped.items()):
        require(len(pair) == 2 and pair[0].frames == pair[1].frames
                and pair[0].slot == pair[1].slot, "paired time/slot mismatch")
        for recipient, donor in ((pair[0], pair[1]), (pair[1], pair[0])):
            for mode in base.MODES:
                out.append({"index": len(out), "pair_id": pair_id, "engine": engine,
                            "mode": mode, "recipient_key": recipient.key,
                            "donor_key": donor.key, "recipient_word": recipient.word,
                            "donor_word": donor.word, "slot": list(recipient.slot),
                            "frames": recipient.frames})
    require(len(out) == 96, "directed mode cell count differs")
    return out


def frame_window(slot: tuple[int, int], t: int) -> tuple[int, int]:
    s, e = slot
    a, b = math.ceil((s - 399) / 320), (e - 1) // 320 + 1
    require(0 < a < b < t, "invalid word/outside state regions")
    return a, b


def edit_state(recipient: np.ndarray, donor: np.ndarray, window: tuple[int, int],
               arm: str, seed: int | None) -> tuple[np.ndarray, dict[str, float]]:
    require(recipient.shape == donor.shape and recipient.ndim == 3
            and recipient.shape[0] == 1 and recipient.shape[-1] == 768
            and recipient.dtype == donor.dtype == np.float32
            and np.isfinite(recipient).all() and np.isfinite(donor).all(),
            "clean state mismatch")
    a, b = window
    require(0 < a < b < recipient.shape[1], "window outside state")
    rw, dw = recipient[:, a:b], donor[:, a:b]
    ro = np.concatenate((recipient[:, :a], recipient[:, b:]), axis=1)
    do = np.concatenate((donor[:, :a], donor[:, b:]), axis=1)
    nw = float(np.linalg.norm(dw.astype(np.float64) - rw.astype(np.float64)))
    no = float(np.linalg.norm(do.astype(np.float64) - ro.astype(np.float64)))
    require(min(nw, no) > 0 and math.isfinite(nw) and math.isfinite(no),
            "zero/nonfinite natural region edit")
    result = recipient.copy()
    if arm == "self":
        pass
    elif arm == "W":
        result[:, a:b] = dw
    elif arm == "O":
        result[:, :a], result[:, b:] = donor[:, :a], donor[:, b:]
    elif arm == "full":
        result[:] = donor
    elif arm in ("random_W", "random_O"):
        require(type(seed) is int, "random seed absent")
        original = rw if arm == "random_W" else ro
        radius = nw if arm == "random_W" else no
        z = np.random.Generator(np.random.PCG64(seed)).standard_normal(original.shape)
        z_norm = float(np.linalg.norm(z))
        require(math.isfinite(z_norm) and z_norm > 0, "invalid random vector")
        edited = original + (z * (radius / z_norm)).astype(np.float32)
        if arm == "random_W":
            result[:, a:b] = edited
        else:
            result[:, :a] = edited[:, :a]
            result[:, b:] = edited[:, a:]
    else:
        raise ValueError("unknown arm")
    require(bool(np.isfinite(result).all()), "nonfinite patch")
    realized = float(np.linalg.norm(result.astype(np.float64)-recipient.astype(np.float64)))
    return result, {"natural_w_l2": nw, "natural_o_l2": no, "realized_l2": realized}


def tensor_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def estimate_bytes(cases: list[base.Case]) -> dict[str, int]:
    # Fixed shape from the pinned 7-convolution checkpoint; one site only.
    max_frames = max(c.frames for c in cases)
    t = (max_frames - 400) // 320 + 1
    npy_headers = (96 * 3 + 576) * 256
    clean = 96 * (max_frames*4 + t*768*4 + t*32*4)
    patched = 576 * t*32*4
    total = clean + patched + npy_headers + 32*1024**2 + 64*1024**2
    require(total <= MAX_OUTPUT_BYTES, "preflight upper bound exceeds 512 MiB")
    return {"max_frames": max_frames, "max_t": t, "clean_payload_bound": clean,
            "patch_logits_payload_bound": patched, "npy_header_bound": npy_headers,
            "metadata_bound": 32*1024**2, "contingency": 64*1024**2,
            "total_bound": total, "cap": MAX_OUTPUT_BYTES}


def preflight() -> dict[str, Any]:
    import torch
    import transformers
    require(PLAN.is_file() and AUDITOR.is_file(), "V2 plan/auditor absent")
    base_receipt = base.preflight()  # pinned checkpoint/hash/dependency check, no weights loaded
    cases, stats, references = trusted_dev_cases()
    rows = directions(cases)
    bound = estimate_bytes(cases)
    return {"schema": SCHEMA + "_preflight", "plan_sha256": sha(PLAN),
            "runner_sha256": sha(Path(__file__)), "runner_commit": source_commit(Path(__file__)),
            "auditor_sha256": sha(AUDITOR), "auditor_commit": source_commit(AUDITOR),
            "fit_hook_helper_sha256": sha(Path(fit.__file__)),
            "fit_hook_helper_commit": source_commit(Path(fit.__file__)),
            "spec_sha256": SPEC_SHA, "manifest_sha256": base.MANIFEST_SHA,
            "base_preflight": base_receipt, "python_version": sys.version.split()[0],
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "cases": [{"key": c.key, "frames": c.frames, "slot": list(c.slot),
                       "wav_sha256": c.wav_sha256, "pcm_sha256": c.pcm_sha256,
                       "reference_tokens": references[c.key]} for c in cases],
            "stats": {f"{p}|{e}": v for (p,e),v in stats.items()},
            "directions": rows, "site": SITE, "arms": list(ARMS),
            "seed_rule": "16048191 + 2*direction_index + (0=random_W,1=random_O)",
            "expected_clean": 96, "expected_patches": 576, "byte_ledger": bound,
            "limits": {"seconds": MAX_SECONDS, "rss": MAX_RSS_BYTES,
                       "output": MAX_OUTPUT_BYTES, "rss_reserve": RSS_RESERVE,
                       "output_reserve": OUTPUT_RESERVE}}


def verify_freeze(freeze: Path, receipt_path: Path, receipt: dict[str, Any]) -> str:
    frozen = json.loads(freeze.read_text())
    require(frozen.get("schema") == SCHEMA + "_freeze"
            and frozen.get("status") == "independently_reviewed_frozen"
            and frozen.get("preflight_sha256") == sha(receipt_path)
            and frozen.get("runner_sha256") == receipt["runner_sha256"]
            and frozen.get("auditor_sha256") == receipt["auditor_sha256"]
            and frozen.get("fit_hook_helper_sha256") == receipt["fit_hook_helper_sha256"]
            and frozen.get("plan_sha256") == receipt["plan_sha256"],
            "freeze binding invalid")
    for key in ("plan_review", "runner_review", "auditor_review"):
        item = frozen.get(key)
        require(isinstance(item, dict) and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str), f"missing {key}")
        path = ROOT / item["path"]
        require(path.resolve().is_relative_to(ROOT) and path.is_file()
                and not path.is_symlink() and sha(path) == item["sha256"]
                and "Decision: GO" in path.read_text(), f"invalid {key}")
    return sha(freeze)


def check_budget(start: float, output: Path, *, before_forward: bool = False) -> None:
    require(time.monotonic()-start <= MAX_SECONDS, "30-minute limit reached")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    written = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    require(rss <= MAX_RSS_BYTES and written <= MAX_OUTPUT_BYTES, "RSS/output cap reached")
    if before_forward:
        require(rss + RSS_RESERVE <= MAX_RSS_BYTES
                and written + OUTPUT_RESERVE <= MAX_OUTPUT_BYTES,
                "insufficient forward headroom")


def execute(receipt_path: Path, freeze: Path, output: Path) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    current = preflight()
    require(canonical(json.loads(receipt_path.read_text())) == canonical(current),
            "preflight/source/input drift")
    freeze_hash = verify_freeze(freeze, receipt_path, current)
    require(not output.exists(), "output already exists")
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    base.append_record(records, {"type":"preflight","schema":SCHEMA,
                                 "receipt":current,"preflight_sha256":sha(receipt_path),
                                 "freeze_sha256":freeze_hash})
    start = time.monotonic()
    in_progress: dict[str,Any] | None = None
    try:
        torch.set_num_threads(4)
        check_budget(start, output, before_forward=True)
        processor = Wav2Vec2Processor.from_pretrained(base.SNAPSHOT, local_files_only=True)
        model, load_info = Wav2Vec2ForCTC.from_pretrained(
            base.SNAPSHOT, local_files_only=True, output_loading_info=True)
        require(set(load_info["missing_keys"]) == {"wav2vec2.masked_spec_embed"}
                and not load_info["unexpected_keys"] and not load_info["mismatched_keys"]
                and not load_info["error_msgs"], "checkpoint loading report changed")
        model = model.to("cpu",dtype=torch.float32).eval()
        require(not model.training and all(p.device.type == "cpu" and p.dtype == torch.float32
                                           for p in model.parameters()), "model not eval CPU float32")
        cases, stats, refs = trusted_dev_cases()
        clean: dict[tuple[str,str],dict[str,Any]] = {}
        for case in cases:
            pcm = base.load_pcm(case)
            for mode in base.MODES:
                in_progress={"phase":"clean","case":case.key,"mode":mode}
                check_budget(start,output,before_forward=True)
                values=base.normalized_input(pcm,mode=mode,stats=stats[(case.pair_id,case.engine)],processor=processor)
                tensor=torch.from_numpy(values.copy())[None]
                captures:dict[str,np.ndarray]={}
                count=[0]
                def hook(_m:Any,_a:Any,result:Any)->None:
                    count[0]+=1
                    require(count[0]==1 and isinstance(result,tuple) and len(result)>=1,
                            "clean hook topology changed")
                    captures[SITE]=base.validate_tensor(SITE,result[0],case.frames)
                handle=model.wav2vec2.encoder.layers[11].register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        response=model(input_values=tensor,attention_mask=None,
                                       output_hidden_states=False,return_dict=True)
                finally:handle.remove()
                require(count==[1] and SITE in captures,"clean hook failed")
                logits=base.validate_tensor("logits",response.logits,case.frames)
                ids=logits.argmax(axis=-1)[0].astype(np.int64).tolist()
                transcript=processor.batch_decode([ids])[0]
                normalized=tokens(transcript)
                eligible=normalized==refs[case.key]
                paths={}; hashes={}
                for name,array in (("input",values[None]),("state",captures[SITE]),("logits",logits)):
                    rel=f"captures/{mode}_{case.key}_{name}.npy"
                    paths[name]=rel
                    hashes[name]=fit.save_array(output/rel,array,output,start)
                row={"type":"clean","schema":SCHEMA,"case_key":case.key,
                     "pair_id":case.pair_id,"engine":case.engine,"word":case.word,"mode":mode,
                     "paths":paths,"sha256":hashes,"shape":{n:list(a.shape) for n,a in
                     (("input",values[None]),("state",captures[SITE]),("logits",logits))},
                     "argmax_ids":ids,"transcript":transcript,"tokens":normalized,
                     "reference_tokens":refs[case.key],"eligible":eligible,"hook_count":count[0]}
                base.append_record(records,row)
                clean[(case.key,mode)]=row
                in_progress=None
                check_budget(start,output)
        for direction in current["directions"]:
            d=direction["index"]; mode=direction["mode"]
            recipient=clean[(direction["recipient_key"],mode)]
            donor=clean[(direction["donor_key"],mode)]
            input_values=np.load(output/recipient["paths"]["input"],allow_pickle=False)
            rstate=np.load(output/recipient["paths"]["state"],allow_pickle=False)
            dstate=np.load(output/donor["paths"]["state"],allow_pickle=False)
            rlogits=np.load(output/recipient["paths"]["logits"],allow_pickle=False)
            dlogits=np.load(output/donor["paths"]["logits"],allow_pickle=False)
            window=frame_window(tuple(direction["slot"]),rstate.shape[1])
            strict=recipient["eligible"] and donor["eligible"]
            for arm in ARMS:
                in_progress={"phase":"patch","direction":d,"arm":arm}
                check_budget(start,output,before_forward=True)
                seed=(SEED_BASE+2*d+(0 if arm=="random_W" else 1)
                      if arm in ("random_W","random_O") else None)
                patch,norms=edit_state(rstate,dstate,window,arm,seed)
                logits,count=fit.patched_forward(model,torch.from_numpy(input_values.copy()),
                                                 site=SITE,clean_state=rstate,patched_state=patch)
                ids=logits.argmax(axis=-1)[0].astype(np.int64).tolist()
                transcript=processor.batch_decode([ids])[0]
                rel=f"captures/d{d:02d}_{arm}_logits.npy"
                digest=fit.save_array(output/rel,logits,output,start)
                control=(bool(np.array_equal(logits,rlogits)) if arm=="self" else
                         bool(np.array_equal(logits,dlogits)) if arm=="full" else None)
                row={"type":"patch","schema":SCHEMA,"direction":d,"arm":arm,
                     "recipient_key":direction["recipient_key"],"donor_key":direction["donor_key"],
                     "mode":mode,"site":SITE,"window":list(window),"seed":seed,
                     "patch_tensor_sha256":tensor_sha(patch),"norms":norms,
                     "logits_path":rel,"logits_sha256":digest,"logits_shape":list(logits.shape),
                     "argmax_ids":ids,"transcript":transcript,"tokens":tokens(transcript),
                     "strict_eligible":strict,"matches_donor_transcript":transcript==donor["transcript"],
                     "matches_recipient_transcript":transcript==recipient["transcript"],
                     "intended_word_only":strict and tokens(transcript)==donor["reference_tokens"],
                     "control_pass":control,"hook_count":count}
                base.append_record(records,row)
                in_progress=None
                check_budget(start,output)
        base.append_record(records,{"type":"complete","schema":SCHEMA,"clean_rows":96,
                                    "patch_rows":576,"elapsed_seconds":time.monotonic()-start,
                                    "rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                    "output_bytes":fit.output_bytes(output)})
    except BaseException as exc:
        base.append_record(records,{"type":"failure","schema":SCHEMA,
                                    "error_type":type(exc).__name__,"error":str(exc),
                                    "in_progress":in_progress,"elapsed_seconds":time.monotonic()-start})
        raise


def main() -> None:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight")
    run=sub.add_parser("run")
    for flag in ("preflight","freeze","output"):
        run.add_argument("--"+flag,required=True,type=Path)
    args=parser.parse_args()
    if args.command=="preflight":print(json.dumps(preflight(),sort_keys=True,indent=2))
    else:execute(args.preflight,args.freeze,args.output)

if __name__=="__main__":main()
