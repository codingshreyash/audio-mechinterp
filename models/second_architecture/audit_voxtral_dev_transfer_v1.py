"""Independent CPU-only audit of a completed Voxtral development transfer.

Never opens partial development outcomes. No model forward is made: saved scores
are checked by arithmetic and frozen source provenance, not recomputed logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = {
    "protocol": (HERE / "PROSPECTIVE_VOXTRAL_DEV_TRANSFER_V1.md",
                 "779fb190e93ab233f386cb1c33bd01d69fd9dc7e8a5d8c4276549bba55ec64b9"),
    "runner": (HERE / "voxtral_dev_transfer_v1.py",
               "6f428f9840e2e9defd6eb6ba3be8c67d0ca624c1e25fee8122eb2e356944a6ed"),
    "first_helper": (HERE / "voxtral_fit_v2_causal_first_record.py",
                     "64d799bbb40b58b128b049877d18bde7774c2dc16717af97e692aa4531e907b5"),
    "preprocess_helper": (HERE / "voxtral_preprocessing_decomposition.py",
                          "026445da1ac2c4c2f4b8f45ab2f8d7a1f4679924774714951b16d5cd29de0788"),
    "mask_builder": (HERE / "voxtral_dev_transfer_masks.py",
                     "f62b7f83832257b99d01a57c9b4374a8b8a087fac83cd9bc50054f7d87fcf49a"),
    "masks": (HERE / "VOXTRAL_DEV_TRANSFER_MASKS_V1.json",
              "eeed0dc96aeda651be1034381a770fd0ed2022706d3f33c08f967391638c1181"),
    "index": (HERE / "voxtral_corpus_v1_index.json",
              "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971"),
    "fit_runner": (HERE / "voxtral_fit_v2_causal_full.py",
                   "4e0a17f80d70ca1498f2487d8cca81b20e36381bfeb35945807b2587b89373cd"),
    "fit_run": (HERE / "runs/voxtral_fit_v2_causal_full/20261118T0928Z/result.json",
                "e29a686e30505a69a3e4b26bddde606a02db753a3a5410c38432cdd14ac3baaf"),
}
SITES = ("encoder31", "projector")
ARMS = ("restore", "loss")
PATCHES = ("whole", "word_0", "word_1", "word_2", "word_3",
           "pre", "post", "padding", "self", "random")
CONTROLS = ("pre", "post", "padding", "random")
VALID = {"recipient", "opposite", "other"}
CAP_SECONDS = 7200
CAP_BYTES = 30 * 1024**3
CLIP_ROOT = (ROOT / "eval/audio_jlens/corpus_v1/rendered/clips").resolve()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(test: bool, reason: str) -> None:
    if not test:
        raise AssertionError(reason)


def near(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not isinstance(a, (float, int)) or not isinstance(b, (float, int)):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol * max(1, abs(a), abs(b))


def same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return near(a, b)
    return a == b


def classify(ids: list[int], visible: str, special: set[int], recipient: str, opposite: str) -> str:
    if any(token in special and not (i == len(ids)-1 and token == 2)
           for i, token in enumerate(ids)):
        return "malformed_special"
    if not ids or ids[-1] != 2:
        return "missing_eos"
    surface = visible.strip()
    if not re.fullmatch(r"[A-Za-z]+", surface, flags=re.ASCII):
        return "malformed_surface"
    folded = surface.casefold()
    return "recipient" if folded == recipient.casefold() else (
        "opposite" if folded == opposite.casefold() else "other")


def slot(row: dict) -> tuple:
    if row["type"] == "capture":
        return ("capture", row["site"], row["word"], row["kind"])
    kind = row["kind"]
    if kind.startswith("patch:"):
        _, arm, patch = kind.split(":")
        return ("patch", row["site"], row["recipient"], arm, patch)
    return ("generation", row["site"], row["recipient"], kind)


def planned_slots(words: list[str]) -> set[tuple]:
    a, b = words
    planned: set[tuple] = set()
    for word in words:
        planned.add(("generation", None, word, "blank_baseline"))
    for name, word in ((f"native_{a}", a), (f"native_{b}", b),
                       (f"word_{a}_outside_{b}", a), (f"{a}_with_{b}_padding", a),
                       (f"word_{b}_outside_{a}", b), (f"{b}_with_{a}_padding", b)):
        planned.add(("generation", None, word, f"feature:{name}"))
    for site in SITES:
        for word in words:
            for kind in ("clean", "blank"):
                planned.add(("capture", site, word, kind))
                planned.add(("generation", site, word, f"passive:{kind}"))
            planned.add(("generation", site, word, "opposite_word_control"))
            for arm in ARMS:
                for patch in PATCHES:
                    planned.add(("patch", site, word, arm, patch))
    require(len(planned) == 108, "auditor planned grammar is wrong")
    return planned


def verify_frozen() -> tuple[list[dict], dict]:
    for name, (path, expected) in P.items():
        require(path.is_file() and sha(path) == expected, f"frozen {name} hash changed")
    masks = json.loads(P["masks"][0].read_text())
    index = json.loads(P["index"][0].read_text())
    require(masks["source_index_sha256"] == P["index"][1] and
            masks["protocol_sha256"] == P["protocol"][1] and
            masks["builder_sha256"] == P["mask_builder"][1], "mask provenance mismatch")
    records = masks["records"]
    require(len(records) == 48 and len({(r["pair_id"], r["engine"], r["phrasing_id"])
                                        for r in records}) == 48, "development records missing or duplicated")
    pairs = sorted({r["pair_id"] for r in records})
    require(len(pairs) == 12 and Counter(p.split("_")[0] for p in pairs) ==
            {"animal": 3, "color": 3, "nature": 3, "object": 3}, "pair/category balance")
    require({(r["pair_id"], r["engine"], r["phrasing_id"]) for r in records} ==
            {(p, e, t) for p in pairs for e in ("piper", "kokoro")
             for t in ("dev_listen", "dev_hear")}, "voice/template balance")
    indexed = {(r["pair_id"], r["engine"], r["phrasing_id"]): r for r in index["rows"]}
    words_by_split={split:set() for split in ("fit_pool","development","locked_test")}
    for source in index["rows"]:
        words_by_split[source["split"]].update(source["words"])
    require(len(words_by_split["development"])==24 and
            not words_by_split["development"] & words_by_split["fit_pool"] and
            not words_by_split["development"] & words_by_split["locked_test"],
            "development word disjointness")
    for record in records:
        pair, engine, template = (record[k] for k in ("pair_id", "engine", "phrasing_id"))
        source = indexed[pair, engine, template]
        require(source["split"] == "development" and source["words"] == record["words"],
                "nondevelopment or word mismatch")
        require(len(record["words"]) == 2 and len(set(record["words"])) == 2, "word pair invalid")
        require({clip["word"] for clip in record["clips"]}==set(record["words"])
                and len(record["clips"])==2,"record clip/word balance")
        pcm = []
        for clip in record["clips"]:
            word = clip["word"]
            original = source["clips"][word]
            path = (ROOT / clip["wav_path"]).resolve()
            require(path.is_relative_to(CLIP_ROOT) and clip["wav_sha256"] == original["wav_sha256"]
                    and clip["pcm_int16_sha256"] == original["pcm_int16_sha256"]
                    and sha(path) == clip["wav_sha256"], "development WAV provenance")
            with wave.open(str(path), "rb") as stream:
                require((stream.getnchannels(), stream.getframerate(), stream.getsampwidth(),
                         stream.getcomptype(), stream.getnframes()) ==
                        (1, 16000, 2, "NONE", clip["frames"]), "development WAV format")
                raw = stream.readframes(stream.getnframes())
            require(hashlib.sha256(raw).hexdigest() == clip["pcm_int16_sha256"], "development PCM")
            pcm.append(raw)
        start, end = record["word_slot_start_sample"], record["word_slot_end_sample_exclusive"]
        width = end-start
        require(width>0 and record["blank_fill_mel_frame"]==2999 and
                record["clips"][0]["frames"]==record["clips"][1]["frames"] and
                2999*160-200>=record["clips"][0]["frames"],
                "word interval, clip length, or blank-fill frame")
        edges=[start+(width*k)//4 for k in range(5)]
        expected_intervals={"pre":[start-width,start],
                            **{f"word_{q}":[edges[q],edges[q+1]] for q in range(4)},
                            "post":[end,end+width]}
        require(record["intervals_half_open_samples"]==expected_intervals and
                expected_intervals["pre"][0]>=0 and
                expected_intervals["post"][1]<=record["clips"][0]["frames"],
                "word-quarter/pre/post intervals")
        mel=[i for i in range(3000) if i*160-200<end and i*160+200>start]
        require(mel and 2999 not in mel and
                record["word_mel_frames_direct_overlap"]==mel,
                "direct-overlap mel mask")
        require(pcm[0][:2*start] == pcm[1][:2*start] and
                pcm[0][2*end:] == pcm[1][2*end:] and
                pcm[0][2*start:2*end] != pcm[1][2*start:2*end], "outside-word PCM mismatch")
        for site in SITES:
            sm = record["site_maps"][site]
            bins = sm["bins"]
            centers=([320*j for j in range(1500)] if site=="encoder31" else
                     [160*(8*r+3) for r in range(375)])
            expected_bins={name:[j for j,center in enumerate(centers) if lo<=center<hi]
                           for name,(lo,hi) in expected_intervals.items()}
            require(bins==expected_bins,"site center-row construction")
            all_rows = [j for name in ("pre", "word_0", "word_1", "word_2", "word_3", "post")
                        for j in bins[name]]
            whole = sorted(j for q in range(4) for j in bins[f"word_{q}"])
            expected_padding=[j for j,center in enumerate(centers)
                              if center>=record["clips"][0]["frames"]+200][:len(whole)]
            require(all(bins[name] for name in bins) and len(all_rows) == len(set(all_rows))
                    and sm["whole_word_rows"] == whole and
                    sm["padding_control_rows"] == expected_padding and
                    len(expected_padding)==len(whole) and
                    not set(sm["padding_control_rows"]) & set(all_rows), "development mask geometry")
    return records, index


def independent_summary(rows: list[dict], cases: list[str]) -> dict:
    require(len(cases) == 96 and len(set(cases)) == 96, "96-case population")
    d = {case: {} for case in cases}
    for row in rows:
        if row["type"] == "capture":
            continue
        case = "|".join((row["pair_id"], row["engine"], row["phrasing_id"], row["recipient"]))
        require(case in d, "unknown directional row")
        key = (row["site"] or "none", row["kind"])
        require(key not in d[case], "duplicate directional row")
        d[case][key] = row
    require(all(len(d[k]) == 50 for k in cases), "directional slot count")
    clean = {}
    eligible = {}
    blanks = Counter()
    for case in cases:
        word = case.rsplit("|", 1)[-1]
        condition = d[case]
        clean[case] = condition["none", f"feature:native_{word}"]["classification"] == "recipient"
        blank = condition["none", "blank_baseline"]["classification"]
        eligible[case] = clean[case] and blank != "recipient"
        blanks[blank] += 1
    nclean, neligible = sum(clean.values()), sum(eligible.values())
    result = {"planned_directions": 96, "clean_correct": nclean,
              "restoration_eligible": neligible, "blank_classes": dict(blanks),
              "task_validity_gate": nclean >= 80 and neligible >= 72, "sites": {}}
    for site in SITES:
        result["sites"][site] = {}
        for arm in ARMS:
            def success(case: str, patch: str) -> bool:
                label = d[case][site, f"patch:{arm}:{patch}"]["classification"]
                return eligible[case] and label == "recipient" if arm == "restore" else (
                    clean[case] and label in {"opposite", "other"})
            whole = sum(success(c, "whole") for c in cases)
            quarters = {f"word_{q}": sum(success(c, f"word_{q}") for c in cases) for q in range(4)}
            controls = {name: sum(success(c, name) for c in cases) for name in CONTROLS}
            groups = {}
            for grouping, count in (("category", 24), ("voice", 48), ("template", 48)):
                buckets = defaultdict(list)
                for case in cases:
                    pair, engine, template, _ = case.split("|")
                    label = (pair.split("_", 1)[0] if grouping == "category" else
                             engine if grouping == "voice" else template)
                    buckets[label].append(case)
                require(all(len(v) == count for v in buckets.values()), f"{grouping} imbalance")
                groups[grouping] = {k: {"planned": count, "whole_success": sum(success(c, "whole") for c in v)}
                                    for k, v in sorted(buckets.items())}
            invalid = sum(d[c][site, f"patch:{arm}:{p}"]["classification"] == "invalid"
                          for c in cases for p in ("whole", *CONTROLS))
            comparisons = {}
            for name in CONTROLS:
                valid = wins = finite = finite_wins = 0
                for case in cases:
                    a = d[case][site, f"patch:{arm}:whole"]
                    b = d[case][site, f"patch:{arm}:{name}"]
                    x, y = a["recipient_minus_opposite"], b["recipient_minus_opposite"]
                    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (x, y)):
                        continue
                    finite += 1
                    finite_wins += x > y if arm == "restore" else x < y
                    if a["classification"] in VALID and b["classification"] in VALID:
                        valid += 1
                        wins += x > y if arm == "restore" else x < y
                comparisons[name] = {"valid_of_96": valid, "strict_wins_of_96": wins,
                                     "finite_score_only_pairs_of_96": finite,
                                     "finite_score_only_wins_of_96": finite_wins,
                                     "passes": valid >= 84 and wins >= 72}
            denom = neligible if arm == "restore" else nclean
            rate = whole / denom if denom else None
            subgroup_pass = (all(x["whole_success"] >= 12 for x in groups["category"].values())
                             and all(x["whole_success"] >= 30 for x in groups["voice"].values())
                             and all(x["whole_success"] >= 30 for x in groups["template"].values()))
            effect = (result["task_validity_gate"] and invalid == 0 and whole >= 72
                      and rate is not None and rate >= .75 and subgroup_pass
                      and all(whole > c and c <= 18 for c in controls.values()))
            time = invalid == 0 and whole > max(controls.values()) and all(
                v["passes"] for v in comparisons.values())
            result["sites"][site][arm] = {
                "whole_word_success_of_96": whole, "eligible_success_rate": rate,
                "whole_word_effect_gate": effect, "word_quarter_success_of_96": quarters,
                "subgroup_breakdown": groups, "control_success_of_96": controls,
                "invalid_main_or_control_conditions": invalid,
                "score_comparisons": comparisons, "time_preference_gate": time}
        result["sites"][site]["opposite_word_control_classes_of_96"] = dict(Counter(
            d[c][site, "opposite_word_control"]["classification"] for c in cases))
    result["development_advance_gate"] = (result["task_validity_gate"]
        and all(result["sites"][s][a]["whole_word_effect_gate"] for s in SITES for a in ARMS)
        and all(result["sites"][s]["loss"]["time_preference_gate"] for s in SITES))
    return result



def tensor_sha(tensor) -> str:
    import torch
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def verify_feature_hashes(processor, record: dict, saved: dict) -> dict:
    import torch
    import voxtral_fit_pilot_capability_v1 as base
    features = {}
    prompt_ids = []
    for clip in record["clips"]:
        word = clip["word"]
        path = (ROOT / clip["wav_path"]).resolve()
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": str(path)}, {"type": "text", "text": base.PROMPT}]}]
        batch = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt")
        require(tuple(batch["input_features"].shape) == (1, 128, 3000)
                and batch["input_ids"][0].tolist().count(24) == 375,
                "processor feature or placeholder geometry")
        prompt_ids.append(batch["input_ids"])
        features[word] = batch["input_features"]
        require(tensor_sha(features[word]) == saved["native"][word], "native mel hash")
        blank = features[word].clone()
        blank[:, :, record["word_mel_frames_direct_overlap"]] = features[word][:, :, 2999:3000]
        require(tensor_sha(blank) == saved["blank"][word], "blank mel hash")
    require(torch.equal(*prompt_ids), "matched prompt tokens")
    a, b = record["words"]
    fa, fb = features[a], features[b]
    mark = torch.zeros(3000, dtype=torch.bool)
    mark[record["word_mel_frames_direct_overlap"]] = True
    padded = min(i for i in range(3000) if 160*i - 200 >= record["clips"][0]["frames"])
    made = {f"native_{a}": fa, f"native_{b}": fb}
    for donor, outside, fd, fo in ((a,b,fa,fb), (b,a,fb,fa)):
        hybrid = fo.clone()
        hybrid[:, :, mark] = fd[:, :, mark]
        pad = fd.clone()
        pad[:, :, padded:] = fo[:, :, padded:]
        made[f"word_{donor}_outside_{outside}"] = hybrid
        made[f"{donor}_with_{outside}_padding"] = pad
    require({k: tensor_sha(v) for k,v in made.items()} == saved["six_conditions"],
            "feature-control mel hashes")
    return features


def verify_row(row: dict, record: dict, byslot: dict, processor, special: set[int]) -> None:
    typ = row["type"]
    site = row["site"]
    words = record["words"]
    if typ == "capture":
        expected_shape = [1,1500,1280] if site == "encoder31" else [375,3072]
        stored_shape = expected_shape[1:] if site == "encoder31" else expected_shape
        require(site in SITES and row["word"] in words and row["kind"] in ("clean", "blank")
                and row["hook_calls"] == 1 and row["raw_hook_shape"] == expected_shape
                and row["stored_state_shape"] == stored_shape
                and row["raw_hook_dtype"] == row["stored_state_dtype"] == "torch.bfloat16"
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
                "capture provenance/shape/hash")
        return
    word = row["recipient"]
    opposite = row["opposite"]
    require(word in words and opposite in words and word != opposite and
            row["pair_id"] == record["pair_id"] and row["engine"] == record["engine"] and
            row["phrasing_id"] == record["phrasing_id"], "row recipient/record identity")
    if typ == "invalid_condition":
        require(row["kind"].startswith("patch:") and row["classification"] == "invalid"
                and row["recipient_minus_opposite"] is None
                and row["edit"]["valid"] is False, "invalid patch fields")
        require(not row["kind"].endswith(":self"), "self patch invalid; parity not measured")
    else:
        require(typ == "generation", "unknown record type")
        ids = row["generated_token_ids"]
        require(isinstance(ids, list) and all(isinstance(i,int) and i>=0 for i in ids),
                "suffix token IDs malformed")
        visible = processor.batch_decode([ids], skip_special_tokens=True)[0]
        with_special = processor.batch_decode([ids], skip_special_tokens=False)[0]
        require(visible == row["raw_decoded"] and with_special == row["decoded_with_special_tokens"]
                and row["terminal_eos"] == bool(ids and ids[-1] == 2)
                and row["nonterminal_special_token_ids"] == [v for v in
                     (ids[:-1] if ids and ids[-1] == 2 else ids) if v in special]
                and row["classification"] == classify(ids, visible, special, word, opposite),
                "saved suffix decode/classification/terminal EOS mismatch")
        scores = row["canonical_scores"]
        require(set(scores) == set(words) and all(isinstance(v, (float,int)) and math.isfinite(v)
                for v in scores.values()) and
                near(row["recipient_minus_opposite"], scores[word]-scores[opposite], 1e-8),
                "candidate score arithmetic")
        require(isinstance(row["seconds_including_scores"], (float,int)) and
                0 <= row["seconds_including_scores"] <= CAP_SECONDS and
                0 <= row["mps_driver_bytes_after"] < CAP_BYTES,
                "generation resource values")
        expected_hook = [1,1500,1280] if site == "encoder31" else [375,3072]
        if site is None:
            require(row["hook_calls_including_two_scores"] == 0 and
                    row["hook_events"] == [] and row["edited_rows_shape"] is None
                    and row["edited_rows_dtype"] is None, "unhooked slot had hooks/edits")
        else:
            require(site in SITES and row["hook_calls_including_two_scores"] == 3 and
                    row["hook_events"] == [{"shape": expected_hook, "dtype": "torch.bfloat16"}]*3,
                    "hook count/shape/dtype")
        if row["kind"].startswith("passive:") or row["kind"].endswith(":self"):
            blank = row["kind"] in ("passive:blank", "patch:restore:self")
            reference = byslot[("generation", None, word, "blank_baseline") if blank else
                               ("generation", None, word, f"feature:native_{word}")]
            require(ids == reference["generated_token_ids"], "passive/self suffix-token parity")
    if row["kind"].startswith("patch:"):
        _, arm, name = row["kind"].split(":")
        require(arm in ARMS and name in PATCHES and site in SITES, "patch arm/site")
        edit = row["edit"]
        smap = record["site_maps"][site]
        mask = (smap["whole_word_rows"] if name in ("whole", "self", "random") else
                smap["padding_control_rows"] if name == "padding" else smap["bins"][name])
        require(edit["mode"] == name and edit["rows"] == mask and
                edit["valid"] == (typ == "generation"), "patch support/validity")
        if typ == "generation":
            require(row["edited_rows_shape"] == [len(mask), 1280 if site=="encoder31" else 3072]
                    and row["edited_rows_dtype"] == "torch.bfloat16"
                    and re.fullmatch(r"[0-9a-f]{64}", edit["edited_bf16_sha256"]) is not None,
                    "patched tensor shape/type/hash")
            if name in CONTROLS:
                target, effective = edit["target_bf16_norm"], edit["effective_bf16_norm"]
                require(target > 0 and near(edit["alpha"],
                    target/edit["unscaled_edit_norm"], 1e-5)
                    and abs(effective-target) <= max(.05*target, 1e-6)+1e-5,
                    "single-scale BF16 norm rule")
            if name == "self":
                require(edit["alpha"] == 0 and edit["effective_bf16_norm"] == 0,
                        "self edit nonzero")
        if name == "random":
            case = "|".join((record["pair_id"],record["engine"],record["phrasing_id"],word))
            label = f"voxtral-dev-transfer-v1|{case}|{site}|{arm}|random_0"
            seed = int.from_bytes(hashlib.sha256(label.encode()).digest(), "big") & ((1<<63)-1)
            require(edit["random_seed_label"] == label and edit["random_seed"] == seed,
                    "random seed/namespace")
    elif row["kind"] == "opposite_word_control":
        donor = byslot["capture", site, opposite, "clean"]
        mask = record["site_maps"][site]["whole_word_rows"]
        require(row["edit"]["mode"] == "opposite_word" and
                row["edit"]["rows"] == mask and
                row["edit"]["donor_sha256"] == donor["sha256"] and
                row["edited_rows_shape"] == [len(mask),1280 if site=="encoder31" else 3072],
                "opposite word donor/source support")


def verify_receipts(run_dir: Path, checkpoints: list[dict]) -> dict:
    starts = sorted(run_dir.glob("invocation_*.start.json"))
    errors = sorted(run_dir.glob("invocation_*.error.json"))
    completes = sorted(run_dir.glob("invocation_*.complete.json"))
    require(starts and len(completes) == 1, "missing or multiple final invocation receipts")
    start_by_stamp = {p.name[11:-11]: json.loads(p.read_text()) for p in starts}
    error_by_stamp = {p.name[11:-11]: json.loads(p.read_text()) for p in errors}
    complete_by_stamp = {p.name[11:-14]: json.loads(p.read_text()) for p in completes}
    checkpoint_by_ordinal={cp["ordinal"]:cp for cp in checkpoints}
    require(len(start_by_stamp)==len(starts) and
            set(start_by_stamp)==set(error_by_stamp)|set(complete_by_stamp) and
            not set(error_by_stamp)&set(complete_by_stamp), "unreceipted/duplicate invocation")
    require(sorted(complete_by_stamp) == [max(start_by_stamp)], "final completion is not last")
    prior_debt = 0.0
    documented = set()
    for stamp in sorted(start_by_stamp):
        start = start_by_stamp[stamp]
        terminal = error_by_stamp.get(stamp, complete_by_stamp.get(stamp))
        require(start["stamp"] == terminal["stamp"] == stamp and
                start["status"] == "started" and
                start["runner_sha256"] == P["runner"][1] and
                start["protocol_sha256"] == P["protocol"][1] and
                start["masks_sha256"] == P["masks"][1] and
                start["wall_cap_seconds"] == CAP_SECONDS and
                near(start["prior_budget_seconds"], prior_debt), "invocation start/debt")
        debt = terminal["cumulative_budget_seconds"]
        require(isinstance(debt,(float,int)) and math.isfinite(debt) and
                prior_debt <= debt <= CAP_SECONDS, "cumulative wall budget")
        for cp in checkpoints:
            if cp["invocation_stamp"]==stamp:
                require(start["prior_budget_seconds"]<cp["cumulative_budget_seconds"]<=debt,
                        "checkpoint outside invocation elapsed budget")
        prior_debt = debt
        memory = run_dir/f"invocation_{stamp}.memory.jsonl"
        if memory.exists():
            for line in memory.read_text().splitlines():
                event=json.loads(line)
                require(0 <= event["driver_max_bytes"] < CAP_BYTES and
                        event["checkpoint"] in checkpoint_by_ordinal and
                        checkpoint_by_ordinal[event["checkpoint"]]["invocation_stamp"]==stamp and
                        event["driver_max_bytes"]==checkpoint_by_ordinal[event["checkpoint"]]["mps_driver_observed_max_bytes"],
                        "memory event/cap/checkpoint identity")
            require(terminal.get("memory_log_sha256") == sha(memory),
                    "invocation memory-log hash")
        else:
            require(terminal.get("memory_log_sha256") is None,
                    "missing memory log with hash receipt")
        if stamp in error_by_stamp:
            require(terminal["status"] == "incomplete_invocation", "error receipt status")
            if terminal.get("attempt_file"):
                documented.add(terminal["attempt_file"])
        else:
            require(terminal["status"] == "complete" and
                    terminal["result_sha256"] == sha(run_dir/"result.json"),
                    "completion receipt result hash")
    committed = {cp["attempt_file"] for cp in checkpoints}
    attempts = {p.name for p in run_dir.glob("record_*_attempt_*.jsonl")}
    require(committed.issubset(attempts) and (attempts-committed).issubset(documented),
            "unaccounted attempt")
    require(all(cp["invocation_stamp"] in start_by_stamp for cp in checkpoints) and
            checkpoints[-1]["cumulative_budget_seconds"] <= prior_debt,
            "checkpoint/invocation debt mismatch")
    return {"invocations": len(starts), "cumulative_seconds": prior_debt}


def audit(run_dir: Path) -> dict:
    require((run_dir/"result.json").is_file(), "run incomplete: do not inspect partial results")
    records, index = verify_frozen()
    import torch
    import transformers
    from transformers import AutoProcessor
    import voxtral_fit_pilot_capability_v1 as base
    require(torch.__version__=="2.15.0" and transformers.__version__=="5.20.2",
            "auditor runtime differs from pinned run")
    meta=json.loads((run_dir/"meta.json").read_text())
    final=json.loads((run_dir/"result.json").read_text())
    require(meta["script_sha256"]==P["runner"][1] and
            meta["protocol_sha256"]==P["protocol"][1] and
            meta["mask_builder_sha256"]==P["mask_builder"][1] and
            meta["masks_sha256"]==P["masks"][1] and
            meta["index_sha256"]==P["index"][1] and
            meta["first_record_helper_sha256"]==P["first_helper"][1] and
            meta["preprocessing_helper_sha256"]==P["preprocess_helper"][1] and
            meta["fit_runner_sha256"]==P["fit_runner"][1] and
            meta["fit_run_sha256"]==P["fit_run"][1] and
            meta["fit_design_sha256"]=="cd2cbd762af3568f85d43c2dec47824666a95d689a3dc89d9cbdf6471ef54ab0" and
            meta["fit_masks_sha256"]=="a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e" and
            meta["fit_selection_sha256"]=="f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b",
            "run source/model provenance")
    require(meta["snapshot"]==base.validate_snapshot(index) and
            meta["installed_sources"]==base.validate_transformers_source() and
            meta["prompt"]==base.PROMPT and
            meta["planned"]=={"records":48,"directions":96,"generations_or_invalid":4800,
                              "captures":384,"teacher_forced_forwards_max":9600,
                              "progress_rows":5184} and
            meta["memory_cap_bytes"]==CAP_BYTES and meta["wall_cap_seconds"]==CAP_SECONDS and
            meta["resource_projection"]["passes"] is True and
            meta["resource_projection"]["projected_total_seconds"]<CAP_SECONDS and
            meta["resource_projection"]["fit_peak_mps_driver_bytes"]<CAP_BYTES,
            "resource/model/workload metadata")
    cases=["|".join((r["pair_id"],r["engine"],r["phrasing_id"],w)) for r in records for w in r["words"]]
    keys=["|".join((r["pair_id"],r["engine"],r["phrasing_id"])) for r in records]
    require(meta["case_ids"]==cases and meta["record_keys"]==keys and
            final["status"]=="development_transfer_complete" and
            final["meta_sha256"]==sha(run_dir/"meta.json") and final["directions"]==96,
            "development case/terminal metadata")
    processor=AutoProcessor.from_pretrained(str(base.SNAPSHOT),local_files_only=True)
    require(processor.tokenizer.eos_token_id==2 and int(processor.audio_token_id)==24,
            "processor IDs changed")
    special=set(processor.tokenizer.all_special_ids)
    all_rows=[]
    cps=[]
    previous=0.0
    for ordinal, record in enumerate(records,1):
        key=keys[ordinal-1]
        cp_path=run_dir/f"record_{ordinal:02d}.checkpoint.json"
        require(cp_path.is_file() and final["checkpoint_sha256"][str(ordinal)]==sha(cp_path),
                "checkpoint hash/missing")
        cp=json.loads(cp_path.read_text())
        require(cp["status"]=="complete" and cp["ordinal"]==ordinal and cp["record_key"]==key and
                cp["run_meta_sha256"]==sha(run_dir/"meta.json") and
                cp["progress_records"]==108 and cp["capture_records"]==8 and
                cp["generation_records"]+cp["invalid_conditions"]==100 and
                cp["teacher_forced_forwards"]==2*cp["generation_records"] and
                previous<cp["cumulative_budget_seconds"]<=CAP_SECONDS and
                0<=cp["mps_driver_observed_max_bytes"]<CAP_BYTES and
                cp["seconds"]>0, "checkpoint counts/resources/provenance")
        previous=cp["cumulative_budget_seconds"]
        cps.append(cp)
        attempt=run_dir/cp["attempt_file"]
        require(attempt.parent==run_dir and attempt.name==cp["attempt_file"] and
                re.fullmatch(fr"record_{ordinal:02d}_attempt_[0-9]+[.]jsonl",attempt.name) and
                sha(attempt)==cp["attempt_sha256"], "attempt name/hash")
        rows=[json.loads(line) for line in attempt.read_text().splitlines()]
        names=[slot(row) for row in rows]
        require(len(rows)==108 and len(set(names))==108 and
                set(names)==planned_slots(record["words"]) and
                sum(r["type"]=="capture" for r in rows)==8 and
                sum(r["type"]=="generation" for r in rows)==cp["generation_records"] and
                sum(r["type"]=="invalid_condition" for r in rows)==cp["invalid_conditions"],
                "record exact planned slots/counts")
        saved=meta["feature_hashes"][key]
        verify_feature_hashes(processor,record,saved)
        frontend=meta["frontend_checks"][key]
        require(all(v<=1e-6 for v in frontend["independent_reproduction_error"].values())
                and frontend["shared_max_off_word_max_abs_error"]<=1e-6,
                "frontend reproduction recorded error")
        byslot={slot(row):row for row in rows}
        for row in rows:
            verify_row(row,record,byslot,processor,special)
        all_rows.extend(rows)
    require(len(all_rows)==5184 and final["progress_rows"]==5184 and
            final["captures"]==384 and
            final["generations"]==sum(r["type"]=="generation" for r in all_rows) and
            final["invalid_conditions"]==sum(r["type"]=="invalid_condition" for r in all_rows) and
            final["generations"]+final["invalid_conditions"]==4800,
            "full development workload")
    summary=independent_summary(all_rows,cases)
    require(same(summary,final["summary"]),"independent development gate arithmetic")
    receipts=verify_receipts(run_dir,cps)
    return {"status":"passed_independent_cpu_only_development_audit",
            "run_dir":str(run_dir),"result_sha256":sha(run_dir/"result.json"),
            "meta_sha256":sha(run_dir/"meta.json"),"records":48,"directions":96,
            "captures":384,"generations":final["generations"],
            "invalid_conditions":final["invalid_conditions"],
            "teacher_forced_score_forwards":2*final["generations"],
            "peak_checkpoint_mps_driver_bytes":max(cp["mps_driver_observed_max_bytes"] for cp in cps),
            "resource_receipts":receipts,"summary":summary,
            "limits":"Saved scalar scores and hidden-state hashes are audited without a second model forward"}

def validate_only() -> dict:
    records, _ = verify_frozen()
    return {"status": "validated_frozen_inputs_only_no_model_or_outcomes",
            "records": len(records), "directions": 2*len(records),
            "planned_progress_rows": 108*len(records),
            "runner_sha256": P["runner"][1], "mask_sha256": P["masks"][1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--audit", type=Path, metavar="RUN_DIR")
    args = parser.parse_args()
    outcome = validate_only() if args.validate_only else audit(args.audit)
    print(json.dumps(outcome, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
