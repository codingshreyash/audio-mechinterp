"""Bounded saved-logit CTC reference-margin diagnostic; no model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "models/second_architecture/runs"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h"
            / "snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
SPEC = ROOT / "eval/audio_jlens/corpus_v1/SPEC_V1.json"
FIT_BASE = RUNS / "wav2vec2_fit_baseline_stage_capture_v1/20261118T2352Z"
FIT_PATCH = RUNS / "wav2vec2_fit_complement_patch_v1/20261119T0035Z"
DEV = RUNS / "wav2vec2_dev_word_disjoint_v1/20261119T0103Z"
PINS = {
    "spec": (SPEC, "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265"),
    "fit_base_records": (FIT_BASE / "records.jsonl", "c9865f4ec306e57baf7dbfc7bd05037ec769034d95db70440972904fcab2507c"),
    "fit_base_audit": (FIT_BASE / "audit.json", "c84c17cdefbebe1007f106cb955f910e7a4ce0431745e59a4528fc2aae41bc49"),
    "fit_patch_records": (FIT_PATCH / "records.jsonl", "3eff05935744803e5eb3022a7417a4c3304e2465c148ad84613df53ba0392766"),
    "fit_patch_audit": (FIT_PATCH / "audit.json", "606ef0aae4af76b7903dcac72a8d51b8be7bd58e99bdf7b0481d22dcd91a4dbe"),
    "dev_records": (DEV / "records.jsonl", "66496f541c8a0c2f8bee1b96b58cdc50721cc415020071b5b517ed702f3df8e1"),
    "dev_audit": (DEV / "audit.json", "34ce592c9a92d7683dcd0719c36f237054a478fee0e893b0be1e797edd28465a"),
    "vocab": (SNAPSHOT / "vocab.json", "19727f8944fe6459fc3f240ae2c198395b740f6a029bd23e06656266b83bcf64"),
    "config": (SNAPSHOT / "config.json", "d3ec255c063d9f95057b553b19c20135b259875834a4fe9deb218a6be25b4cf3"),
}
FIT_SITES = ("feature_projection", "post_block11")
FIT_ARMS = ("self_C", "W", "C", "R1_C", "R2_C")
DEV_ARMS = ("self", "W", "O", "random_O")
MAX_SECONDS = 300
MAX_RESULT_BYTES = 2 * 1024**2


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(value: str) -> list[str]:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", value).casefold()).split()


def checked_rows(path: Path, expected_count: int) -> list[dict]:
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"truncated records: {path}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    require(len(rows) == expected_count and rows[0].get("type") == "preflight"
            and rows[-1].get("type") == "complete"
            and not any(row.get("type") == "failure" for row in rows),
            f"incomplete saved matrix: {path}")
    return rows


def checked_array(directory: Path, relative: str, digest: str,
                  shape: tuple[int, ...]) -> np.ndarray:
    require(relative.startswith("captures/") and ".." not in Path(relative).parts,
            "unsafe capture reference")
    path = directory / relative
    require(path.resolve().is_relative_to(directory.resolve()) and path.is_file()
            and not path.is_symlink() and sha(path) == digest,
            f"capture identity changed: {relative}")
    array = np.load(path, allow_pickle=False)
    require(array.dtype == np.float32 and tuple(array.shape) == shape
            and bool(np.isfinite(array).all()), f"capture invalid: {relative}")
    return array


def target_ids(reference: list[str], vocab: dict[str, int]) -> tuple[str, list[int]]:
    require(reference and all(isinstance(w, str) and w for w in reference),
            "empty reference")
    text = " ".join(reference).upper()
    symbols = ["|" if ch == " " else ch for ch in text]
    require(all(len(ch) == 1 and ch in vocab for ch in symbols),
            f"reference contains unknown CTC symbol: {text}")
    ids = [vocab[ch] for ch in symbols]
    require(all(4 <= value < 32 for value in ids), "special CTC target")
    return text, ids


def ctc_logp(logits: np.ndarray, target: list[int], blank: int = 0) -> float:
    """Exact blank-interleaved CTC forward sum in float64 log space."""
    require(logits.ndim == 2 and logits.shape[0] > 0 and len(target) > 0
            and len(target) <= logits.shape[0] and blank not in target,
            "CTC target/logit geometry invalid")
    x = logits.astype(np.float64)
    peak = x.max(axis=1, keepdims=True)
    logp = x - peak - np.log(np.exp(x - peak).sum(axis=1, keepdims=True))
    extended = np.full(2 * len(target) + 1, blank, dtype=np.int64)
    extended[1::2] = target
    skip = (extended[2:] != blank) & (extended[2:] != extended[:-2])
    state = np.full(len(extended), -np.inf)
    state[0], state[1] = logp[0, blank], logp[0, target[0]]
    for frame in logp[1:]:
        updated = state.copy()
        updated[1:] = np.logaddexp(updated[1:], state[:-1])
        updated[2:] = np.logaddexp(updated[2:], np.where(skip, state[:-2], -np.inf))
        state = updated + frame[extended]
    result = float(np.logaddexp(state[-1], state[-2]))
    require(math.isfinite(result), "nonfinite complete-reference CTC score")
    return result


def score(logits: np.ndarray, rec_ids: list[int], donor_ids: list[int]) -> dict[str, float]:
    require(logits.shape[0] == 1 and logits.shape[2] == 32, "CTC logits shape invalid")
    r = ctc_logp(logits[0], rec_ids)
    d = ctc_logp(logits[0], donor_ids)
    margin = d - r
    require(math.isfinite(margin), "nonfinite CTC reference margin")
    return {"logp_recipient": r, "logp_donor": d, "margin": margin}


def natural_norms(recipient: np.ndarray, donor: np.ndarray,
                  window: tuple[int, int]) -> tuple[float, float]:
    require(recipient.shape == donor.shape and recipient.shape[0] == 1
            and recipient.shape[2] == 768, "paired state shape differs")
    a, b = window
    require(0 < a < b < recipient.shape[1], "invalid word window")
    w = donor[:, a:b].astype(np.float64) - recipient[:, a:b].astype(np.float64)
    o = np.concatenate((donor[:, :a].astype(np.float64) - recipient[:, :a].astype(np.float64),
                        donor[:, b:].astype(np.float64) - recipient[:, b:].astype(np.float64)), axis=1)
    nw, no = float(np.linalg.norm(w)), float(np.linalg.norm(o))
    require(math.isfinite(nw) and math.isfinite(no) and nw > 0 and no > 0,
            "zero/nonfinite natural edit norm")
    return nw, no


def checked_audits() -> dict[str, str]:
    hashes = {}
    for label, (path, expected) in PINS.items():
        # Hugging Face snapshots store pinned assets as links into their own
        # content-addressed blobs. Run artifacts and the corpus spec are files.
        storage_ok = (path.resolve().is_relative_to(SNAPSHOT.parents[1] / "blobs")
                      if label in ("vocab", "config") else not path.is_symlink())
        require(path.is_file() and storage_ok and sha(path) == expected,
                f"pinned input changed: {label}")
        hashes[label] = expected
    for prefix in ("fit_base", "fit_patch", "dev"):
        audit = json.loads(PINS[prefix + "_audit"][0].read_text())
        require(audit.get("status") == "independent_saved_data_audit_pass"
                and audit.get("records_sha256") == hashes[prefix + "_records"],
                f"prior independent audit invalid: {prefix}")
    fit_audit = json.loads(PINS["fit_patch_audit"][0].read_text())
    dev_audit = json.loads(PINS["dev_audit"][0].read_text())
    require(fit_audit.get("control_gate_pass") is True
            and dev_audit.get("mechanism_gate_pass") is True,
            "prior control gate failed")
    return hashes


def observed_ids(logits: np.ndarray, row: dict) -> None:
    ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
    require(row.get("argmax_ids") == ids and isinstance(row.get("transcript"), str),
            "saved CTC output IDs differ from logits")


def fit_cells(vocab: dict[str, int], spec: dict) -> list[dict]:
    base_rows = checked_rows(FIT_BASE / "records.jsonl", 26)
    patch_rows = checked_rows(FIT_PATCH / "records.jsonl", 194)
    bases = {(r["mode"], r["case_key"]): r for r in base_rows if r.get("type") == "baseline"}
    patches = {(r["direction_index"], r["site"], r["arm"]): r
               for r in patch_rows if r.get("type") == "patch"}
    directions = patch_rows[0]["receipt"]["directions"]
    require(len(bases) == 16 and len(patches) == 192 and len(directions) == 16
            and set(patches) == {(d, s, a) for d in range(16)
                                 for s in FIT_SITES for a in ("W", "C", "self_C", "full_donor", "R1_C", "R2_C")},
            "FIT matrix incomplete/duplicate")
    require(patch_rows[-1].get("control_gate_pass") is True, "FIT controls failed")
    template = next(x for x in spec["templates"]["fit"] if x["id"] == "fit_chosen")
    case_info = {c["key"]: c for c in patch_rows[0]["receipt"]["base_preflight"]["cases"]}
    require(len(case_info) == 8, "FIT case count changed")
    result = []
    for d, direction in enumerate(directions):
        require(direction["direction_index"] == d, "FIT direction order changed")
        mode, rec_key, donor_key = (direction[k] for k in ("mode", "recipient_key", "donor_key"))
        rec, donor = bases[(mode, rec_key)], bases[(mode, donor_key)]
        require((rec["pair_id"], rec["engine"], donor["pair_id"], donor["engine"]) ==
                (direction["pair_id"], direction["engine"])*2,
                "FIT direction identity differs")
        ref_rec = tokens(" ".join((template["prefix"], case_info[rec_key]["word"], template["suffix"])))
        ref_donor = tokens(" ".join((template["prefix"], case_info[donor_key]["word"], template["suffix"])))
        rec_text, rec_ids = target_ids(ref_rec, vocab)
        donor_text, donor_ids = target_ids(ref_donor, vocab)
        strict = tokens(rec["transcript"]) == ref_rec and tokens(donor["transcript"]) == ref_donor
        for site in FIT_SITES:
            clean_states = []
            for baseline_row in (rec, donor):
                key = baseline_row["case_key"]
                t = (case_info[key]["frames"] - 400)//320 + 1
                rel = f"captures/{mode}_{key}_{site}.npy"
                clean_states.append(checked_array(FIT_BASE, rel,
                    baseline_row["capture_sha256"][site], (1, t, 768)))
            row_w = patches[(d, site, "W")]
            a, b = row_w["windows"]["W"]
            nw, no = natural_norms(*clean_states, (a, b))
            arms = {}
            for arm in FIT_ARMS:
                row = patches[(d, site, arm)]
                require(row["mode"] == mode and row["recipient_key"] == rec_key
                        and row["donor_key"] == donor_key and row["windows"]["W"] == [a,b]
                        and row["hook_count"] == 1 and math.isclose(row["natural_w_l2"], nw, rel_tol=1e-8)
                        and math.isclose(row["natural_c_l2"], no, rel_tol=1e-8),
                        "FIT patch identity/norm differs")
                logits = checked_array(FIT_PATCH, row["logits_path"],
                                       row["logits_sha256"], tuple(row["logits_shape"]))
                observed_ids(logits, row)
                if arm == "self_C":
                    rel = f"captures/{mode}_{rec_key}_logits.npy"
                    baseline_logits = checked_array(FIT_BASE, rel,
                        rec["capture_sha256"]["logits"], tuple(rec["logits_shape"]))
                    require(row["control_pass"] is True and np.array_equal(logits, baseline_logits),
                            "FIT self control differs from clean logits")
                arms[arm] = score(logits, rec_ids, donor_ids)
            self_margin = arms["self_C"]["margin"]
            for arm in FIT_ARMS:
                row = patches[(d, site, arm)]
                delta = arms[arm]["margin"] - self_margin
                norm = nw if arm == "W" else no if arm != "self_C" else None
                result.append({"set": "fit", "direction": d, "site": site, "mode": mode,
                               "pair_id": direction["pair_id"], "engine": direction["engine"],
                               "recipient_key": rec_key, "donor_key": donor_key,
                               "recipient_reference": rec_text, "donor_reference": donor_text,
                               "strict_clean_eligible": strict, "arm": arm,
                               "observed_transcript": row["transcript"],
                               "greedy_matches_recipient": row["transcript"] == rec["transcript"],
                               "natural_w_l2": nw, "natural_o_l2": no,
                               "edit_l2": norm, "delta_per_edit_l2": delta/norm if norm else None,
                               **arms[arm], "delta_margin": delta,
                               "at_least_double": delta >= math.log(2)})
    return result


def dev_cells(vocab: dict[str, int], spec: dict) -> list[dict]:
    rows = checked_rows(DEV / "records.jsonl", 674)
    clean = {(r["case_key"], r["mode"]): r for r in rows if r.get("type") == "clean"}
    patches = {(r["direction"], r["arm"]): r for r in rows if r.get("type") == "patch"}
    directions = rows[0]["receipt"]["directions"]
    require(len(clean) == 96 and len(patches) == 576 and len(directions) == 96
            and set(patches) == {(d,a) for d in range(96)
                                 for a in ("self", "W", "O", "full", "random_W", "random_O")},
            "development matrix incomplete/duplicate")
    require(rows[-1].get("clean_rows") == 96 and rows[-1].get("patch_rows") == 576,
            "development completion counts invalid")
    template = next(x for x in spec["templates"]["development"] if x["id"] == "dev_listen")
    result = []
    for d, direction in enumerate(directions):
        require(direction["index"] == d, "development direction order changed")
        mode, rec_key, donor_key = (direction[k] for k in ("mode", "recipient_key", "donor_key"))
        rec, donor = clean[(rec_key, mode)], clean[(donor_key, mode)]
        ref_rec = tokens(" ".join((template["prefix"], rec["word"], template["suffix"])))
        ref_donor = tokens(" ".join((template["prefix"], donor["word"], template["suffix"])))
        require(rec["reference_tokens"] == ref_rec and donor["reference_tokens"] == ref_donor,
                "development scripted reference differs")
        rec_text, rec_ids = target_ids(ref_rec, vocab)
        donor_text, donor_ids = target_ids(ref_donor, vocab)
        strict = bool(rec["eligible"] and donor["eligible"])
        require(strict == (tokens(rec["transcript"]) == ref_rec
                           and tokens(donor["transcript"]) == ref_donor),
                "development strict eligibility differs")
        a, b = patches[(d,"W")]["window"]
        states = []
        for clean_row in (rec, donor):
            states.append(checked_array(DEV, clean_row["paths"]["state"],
                                        clean_row["sha256"]["state"],
                                        tuple(clean_row["shape"]["state"])))
        nw, no = natural_norms(*states, (a,b))
        arms = {}
        for arm in DEV_ARMS:
            row = patches[(d,arm)]
            require(row["mode"] == mode and row["site"] == "post_block11"
                    and row["recipient_key"] == rec_key and row["donor_key"] == donor_key
                    and row["window"] == [a,b] and row["hook_count"] == 1
                    and math.isclose(row["norms"]["natural_w_l2"], nw, rel_tol=1e-8)
                    and math.isclose(row["norms"]["natural_o_l2"], no, rel_tol=1e-8),
                    "development patch identity/norm differs")
            logits = checked_array(DEV, row["logits_path"],
                                   row["logits_sha256"], tuple(row["logits_shape"]))
            observed_ids(logits, row)
            if arm == "self":
                clean_logits = checked_array(DEV, rec["paths"]["logits"],
                        rec["sha256"]["logits"], tuple(rec["shape"]["logits"]))
                require(row["control_pass"] is True and np.array_equal(logits, clean_logits),
                        "development self control differs from clean logits")
            arms[arm] = score(logits, rec_ids, donor_ids)
        self_margin = arms["self"]["margin"]
        for arm in DEV_ARMS:
            row = patches[(d,arm)]
            delta = arms[arm]["margin"] - self_margin
            norm = nw if arm == "W" else no if arm != "self" else None
            result.append({"set": "development", "direction": d,
                           "site": "post_block11", "mode": mode,
                           "pair_id": direction["pair_id"], "engine": direction["engine"],
                           "recipient_key": rec_key, "donor_key": donor_key,
                           "recipient_reference": rec_text, "donor_reference": donor_text,
                           "strict_clean_eligible": strict, "arm": arm,
                           "observed_transcript": row["transcript"],
                           "greedy_matches_recipient": row["transcript"] == rec["transcript"],
                           "natural_w_l2": nw, "natural_o_l2": no,
                           "edit_l2": norm, "delta_per_edit_l2": delta/norm if norm else None,
                           **arms[arm], "delta_margin": delta,
                           "at_least_double": delta >= math.log(2)})
    return result


def summaries(cells: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    by_cell = {(r["set"],r["direction"],r["site"],r["arm"]):r for r in cells}
    require(len(by_cell) == len(cells), "duplicate score cell")
    for row in cells:
        for level, detail in (("site", ()), ("mode", (row["mode"],)),
                              ("pair_voice", (row["mode"],row["pair_id"],row["engine"]))):
            groups[(row["set"],row["site"],level,*detail,row["arm"])].append(row)
    out = []
    for key, selected in sorted(groups.items()):
        for eligibility, subset in (("all",selected),
                                    ("strict",[r for r in selected if r["strict_clean_eligible"]])):
            deltas = [r["delta_margin"] for r in subset]
            out.append({"group": list(key), "eligibility": eligibility, "n": len(deltas),
                        "positive": sum(x > 0 for x in deltas),
                        "at_least_double": sum(x >= math.log(2) for x in deltas),
                        "median_delta": float(np.median(deltas)) if deltas else None,
                        "min_delta": min(deltas) if deltas else None,
                        "max_delta": max(deltas) if deltas else None})
    paired = []
    for row in cells:
        if row["arm"] not in ("R1_C", "R2_C", "random_O"):
            continue
        outside_arm = "C" if row["set"] == "fit" else "O"
        outside = by_cell[(row["set"],row["direction"],row["site"],outside_arm)]
        paired.append({"set":row["set"], "direction":row["direction"], "site":row["site"],
                       "mode":row["mode"], "pair_id":row["pair_id"], "engine":row["engine"],
                       "strict_clean_eligible":row["strict_clean_eligible"],
                       "random_arm":row["arm"],
                       "outside_minus_random_delta":outside["delta_margin"]-row["delta_margin"]})
    return out, paired


def run() -> dict:
    started = time.monotonic()
    hashes = checked_audits()
    vocab = json.loads(PINS["vocab"][0].read_text())
    config = json.loads(PINS["config"][0].read_text())
    require(config["pad_token_id"] == vocab["<pad>"] == 0 and config["vocab_size"] == 32,
            "CTC blank/vocabulary pin differs")
    spec = json.loads(SPEC.read_text())
    cells = fit_cells(vocab, spec) + dev_cells(vocab, spec)
    require(len(cells) == 32*5 + 96*4, "score population differs")
    summary, paired = summaries(cells)
    require(len(paired) == 32*2 + 96, "paired outside-control population differs")
    require(time.monotonic()-started <= MAX_SECONDS, "analysis wall-time cap exceeded")
    return {"schema":"wav2vec2_saved_ctc_reference_margin_v1",
            "status":"saved_logit_analysis_complete",
            "protocol_sha256":sha(HERE/"PROTOCOL.md"),
            "analyzer_sha256":sha(Path(__file__)),
            "pinned_inputs":hashes,
            "definition":"M=logP(complete donor script)-logP(complete recipient script); delta=M_arm-M_self",
            "counts":{"fit_cells":160,"development_cells":384,"paired_outside_random":160},
            "cells":cells,"summaries":summary,"paired_outside_random":paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"RESULT.json")
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite result")
    result = run()
    payload = (json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False,
                          separators=(",",":"))+"\n").encode()
    require(len(payload) <= MAX_RESULT_BYTES, "result exceeds 2 MiB cap")
    args.output.write_bytes(payload)
    print(json.dumps({"result":str(args.output),"bytes":len(payload),
                      "counts":result["counts"],"sha256":sha(args.output)},sort_keys=True))


if __name__ == "__main__":
    main()
