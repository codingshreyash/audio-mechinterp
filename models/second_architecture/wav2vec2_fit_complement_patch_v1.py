"""Prospective FIT-only Wav2Vec2 outside-word complement patch assay.

The ``preflight`` command reads source, assets and saved baseline, but loads no
model weights. ``run`` requires a byte-identical preflight and a separately
reviewed freeze receipt. See the complement V1/V2 prospective plans. This is
an internal ASR assay, not an audio attack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import wav2vec2_fit_baseline_stage_capture_v1 as baseline

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE / "runs/wav2vec2_fit_baseline_stage_capture_v1/20261118T2352Z"
BASELINE_RECORDS_SHA = "c9865f4ec306e57baf7dbfc7bd05037ec769034d95db70440972904fcab2507c"
BASELINE_AUDIT_SHA = "c84c17cdefbebe1007f106cb955f910e7a4ce0431745e59a4528fc2aae41bc49"
BASELINE_MANIFEST = HERE / "WAV2VEC2_FIT_BASELINE_V2_SIDECAR_MANIFEST_V1.json"
BASELINE_MANIFEST_SHA = "2f0484205d21b596795fbfbf30c1e65a63451dafc8740e5860c95be20ace2b53"
PRIOR_DIR = HERE / "runs/wav2vec2_fit_causal_patch_v1/20261119T0013Z"
PRIOR_RECORDS_SHA = "bacfc2bcdfe7239b960200735758c708389a7762a65761eb185c5caaa6e60285"
PRIOR_AUDIT_SHA = "fcfa865d56e0ebb1b719e8a890edd77fdb5aa39f2ae08aef782ff1b0d6eeb052"
PRIOR_MANIFEST = HERE / "WAV2VEC2_FIT_CAUSAL_SIDECAR_MANIFEST_V1.json"
PRIOR_MANIFEST_SHA = "1785a336a1016e7ec689c9a670532238b29f117e074e54bd45d0078ef8c570b6"
PLAN_V1 = HERE / "WAV2VEC2_FIT_COMPLEMENT_CAUSAL_PROSPECTIVE_V1.md"
PLAN_V1_SHA = "63724fd4237aa0176582e796eb70380ea0bd5120808b90565b0db01600a1c65d"
PLAN_V2 = HERE / "WAV2VEC2_FIT_COMPLEMENT_CAUSAL_PROSPECTIVE_V2.md"
PLAN_V2_SHA = "4937d53d061a1e90984037e7d011f39de1108de2c45ba1c1e544089a97746dae"
SCHEMA = "wav2vec2_fit_complement_patch_v1"
SITES = ("feature_projection", "post_block11")
ARMS = ("W", "C", "self_C", "full_donor", "R1_C", "R2_C")
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 512 * 1024**2
RSS_RESERVE_BYTES = 1024**3
OUTPUT_RESERVE_BYTES = 32 * 1024**2
RANDOM_SEED_BASE = 18470111


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return baseline.sha(path)


def canonical(value: Any) -> str:
    return baseline.canonical(value)


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


def read_baseline() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Bind every baseline row and needed capture before any new model load."""
    require(sha(BASELINE_DIR / "records.jsonl") == BASELINE_RECORDS_SHA,
            "reviewed baseline records changed")
    require(sha(BASELINE_DIR / "audit.json") == BASELINE_AUDIT_SHA,
            "reviewed baseline audit changed")
    require(sha(BASELINE_MANIFEST) == BASELINE_MANIFEST_SHA,
            "reviewed baseline capture manifest changed")
    audit = json.loads((BASELINE_DIR / "audit.json").read_text())
    require(audit.get("status") == "independent_saved_data_audit_pass"
            and audit.get("records_sha256") == BASELINE_RECORDS_SHA,
            "baseline independent audit is not a pass")
    manifest = json.loads(BASELINE_MANIFEST.read_text())
    inventory = {entry["path"]: entry for entry in manifest["files"]}
    require(len(inventory) == 320, "baseline manifest coverage changed")
    rows = [json.loads(line) for line in (BASELINE_DIR / "records.jsonl").read_text().splitlines()]
    require(len(rows) == 26 and rows[0].get("type") == "preflight"
            and rows[-1].get("type") == "complete", "baseline run incomplete")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows[1:-1]:
        if row.get("type") != "baseline":
            continue
        key = (row["mode"], row["case_key"])
        require(key not in index, "duplicate baseline row")
        index[key] = row
        for stage in ("input_values", *SITES, "logits"):
            relative = f"captures/{row['mode']}_{row['case_key']}_{stage}.npy"
            path = BASELINE_DIR / relative
            require(path.is_file() and not path.is_symlink()
                    and inventory[relative]["sha256"] == row["capture_sha256"][stage]
                    and path.stat().st_size == inventory[relative]["size_bytes"]
                    and sha(path) == inventory[relative]["sha256"],
                    f"reviewed capture bytes changed: {relative}")
    cases, _ = baseline.trusted_cases()
    require(set(index) == {(mode, case.key) for mode in baseline.MODES for case in cases},
            "baseline case/mode matrix differs")
    return index, {relative: entry["sha256"] for relative, entry in inventory.items()}


def read_prior_w() -> dict[tuple[int, str], dict[str, Any]]:
    """Rehash every prior W-logit capture needed for exact replay before weights."""
    require(sha(PRIOR_DIR / "records.jsonl") == PRIOR_RECORDS_SHA,
            "reviewed prior causal records changed")
    require(sha(PRIOR_DIR / "audit.json") == PRIOR_AUDIT_SHA,
            "reviewed prior causal audit changed")
    require(sha(PRIOR_MANIFEST) == PRIOR_MANIFEST_SHA,
            "reviewed prior causal sidecar manifest changed")
    audit = json.loads((PRIOR_DIR / "audit.json").read_text())
    require(audit.get("status") == "independent_saved_data_audit_pass"
            and audit.get("records_sha256") == PRIOR_RECORDS_SHA
            and audit.get("patch_rows") == 224, "prior causal audit is not a pass")
    inventory = {item["path"]: item for item in json.loads(PRIOR_MANIFEST.read_text())["files"]}
    require(len(inventory) == 448, "prior sidecar manifest population changed")
    rows = [json.loads(line) for line in (PRIOR_DIR / "records.jsonl").read_text().splitlines()]
    require(len(rows) == 226 and rows[0].get("type") == "preflight"
            and rows[-1].get("type") == "complete"
            and rows[-1].get("control_gate_pass") is True,
            "prior causal run incomplete or controls failed")
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows[1:-1]:
        if row.get("type") != "patch" or row.get("arm") != "W":
            continue
        key = (row["direction_index"], row["site"])
        require(key not in index and row.get("hook_count") == 1,
                "duplicate/invalid prior W row")
        rel = row["logits_path"]
        path = PRIOR_DIR / rel
        item = inventory[rel]
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == item["size_bytes"]
                and row["logits_sha256"] == item["sha256"] == sha(path),
                f"prior W logits changed: {rel}")
        index[key] = row
    require(set(index) == {(d, site) for d in range(16) for site in SITES},
            "prior W replay matrix differs")
    return index


def prior_w_logits(row: dict[str, Any]) -> np.ndarray:
    path = PRIOR_DIR / row["logits_path"]
    require(sha(path) == row["logits_sha256"], "prior W logits changed during run")
    array = np.load(path, allow_pickle=False)
    require(array.dtype == np.float32 and tuple(array.shape) == tuple(row["logits_shape"])
            and bool(np.isfinite(array).all()), "prior W logits invalid")
    return array


def directions(cases: list[baseline.Case]) -> list[dict[str, Any]]:
    by_key = {case.key: case for case in cases}
    result = []
    for pair_id, words in baseline.PAIR_WORDS.items():
        for engine in baseline.ENGINES:
            for recipient_word in words:
                donor_word = next(word for word in words if word != recipient_word)
                recipient_key = f"{pair_id}_{engine}_{recipient_word}"
                donor_key = f"{pair_id}_{engine}_{donor_word}"
                for mode in baseline.MODES:
                    result.append({"direction_index": len(result), "pair_id": pair_id,
                                   "engine": engine, "mode": mode,
                                   "recipient_key": recipient_key,
                                   "donor_key": donor_key,
                                   "slot_samples": list(by_key[recipient_key].slot),
                                   "frames": by_key[recipient_key].frames})
    require(len(result) == 16, "direction matrix differs")
    return result


def windows(slot: tuple[int, int], t: int) -> dict[str, tuple[int, int]]:
    s, e = slot
    start = math.ceil((s - 399) / 320)
    end = (e - 1) // 320 + 1
    result = {"W": (start, end), "C_left": (0, start), "C_right": (end, t)}
    require(0 < start < end < t, "word/complement window outside captured frames")
    return result


def complement(array: np.ndarray, window: dict[str, tuple[int, int]]) -> np.ndarray:
    left_end = window["W"][0]
    right_start = window["W"][1]
    return np.concatenate((array[:, :left_end], array[:, right_start:]), axis=1)


def candidate_state(recipient: np.ndarray, donor: np.ndarray, *, arm: str,
                    window: dict[str, tuple[int, int]], seed: int | None
                    ) -> tuple[np.ndarray, float, float, float]:
    require(recipient.shape == donor.shape and recipient.ndim == 3
            and recipient.shape[0] == 1 and recipient.shape[-1] == 768
            and recipient.dtype == donor.dtype == np.float32,
            "patch state shape/dtype mismatch")
    require(arm in ARMS, "unknown arm")
    patched = recipient.copy()
    a, b = window["W"]
    recipient_c = complement(recipient, window)
    donor_c = complement(donor, window)
    c_difference = donor_c.astype(np.float64) - recipient_c.astype(np.float64)
    natural_c_norm = float(np.linalg.norm(c_difference))
    natural_w_norm = float(np.linalg.norm(donor[:, a:b].astype(np.float64)
                                          - recipient[:, a:b].astype(np.float64)))
    require(math.isfinite(natural_c_norm) and natural_c_norm > 0
            and math.isfinite(natural_w_norm) and natural_w_norm > 0,
            "zero/nonfinite natural W or complement delta")
    if arm == "full_donor":
        patched[:] = donor
    elif arm == "W":
        patched[:, a:b] = donor[:, a:b]
    elif arm == "C":
        patched[:, :a] = donor[:, :a]
        patched[:, b:] = donor[:, b:]
    elif arm in ("R1_C", "R2_C"):
        require(seed is not None, "random seed missing")
        random = np.random.Generator(np.random.PCG64(seed)).standard_normal(recipient_c.shape)
        random_norm = float(np.linalg.norm(random))
        require(math.isfinite(random_norm) and random_norm > 0, "invalid random draw")
        edited_c = recipient_c + (random * (natural_c_norm / random_norm)).astype(np.float32)
        patched[:, :a] = edited_c[:, :a]
        patched[:, b:] = edited_c[:, a:]
    elif arm != "self_C":
        raise ValueError("unhandled complement arm")
    require(bool(np.isfinite(patched).all()), "nonfinite patch state")
    if arm == "self_C":
        require(np.array_equal(patched, recipient), "self patch differs")
    elif arm == "full_donor":
        require(np.array_equal(patched, donor), "full donor state differs")
    elif arm == "W":
        require(np.array_equal(patched[:, :a], recipient[:, :a])
                and np.array_equal(patched[:, b:], recipient[:, b:]),
                "W patch differs outside declared rows")
    else:
        require(np.array_equal(patched[:, a:b], recipient[:, a:b]),
                "complement patch differs inside W")
    realized_norm = float(np.linalg.norm(patched.astype(np.float64)
                                          - recipient.astype(np.float64)))
    return patched, realized_norm, natural_w_norm, natural_c_norm


def target_only_transfer(recipient: dict[str, Any], donor: dict[str, Any],
                         actual_words: list[str]) -> bool:
    r = recipient["normalized_words"]
    d = donor["normalized_words"]
    rw, dw = recipient["word"], donor["word"]
    if not (r.count(rw) == d.count(dw) == 1
            and dw not in r and rw not in d and len(r) == len(d)):
        return False
    i = r.index(rw)
    return d == [*r[:i], dw, *r[i + 1:]] and actual_words == d


def no_weight_preflight() -> dict[str, Any]:
    import torch
    import transformers

    require(sha(PLAN_V1) == PLAN_V1_SHA, "complement plan V1 changed")
    require(PLAN_V2.is_file() and sha(PLAN_V2) == PLAN_V2_SHA,
            "random-order amendment V2 absent or changed")
    cases, _ = baseline.trusted_cases()
    baseline_index, _ = read_baseline()
    prior_index = read_prior_w()
    base_receipt = baseline.preflight()
    require(base_receipt["model_revision"] == baseline.MODEL_REVISION
            and base_receipt["checkpoint_sha256"] == baseline.WEIGHTS_SHA
            and base_receipt["device"] == "cpu", "base preflight identity changed")
    rows = directions(cases)
    eligibility = {}
    for row in rows:
        recipient = baseline_index[(row["mode"], row["recipient_key"])]
        donor = baseline_index[(row["mode"], row["donor_key"])]
        t = recipient["capture_shapes"][SITES[0]][1]
        require(t == donor["capture_shapes"][SITES[0]][1], "paired time axes differ")
        positions = windows(tuple(row["slot_samples"]), t)
        for site in SITES:
            source = load_capture(row["mode"], row["recipient_key"], site, baseline_index)
            donor_state = load_capture(row["mode"], row["donor_key"], site, baseline_index)
            require(source.shape == donor_state.shape, "paired site state differs")
            natural_c = float(np.linalg.norm(complement(donor_state, positions).astype(np.float64)
                                             - complement(source, positions).astype(np.float64)))
            require(math.isfinite(natural_c) and natural_c > 0,
                    "natural complement norm invalid before model load")
            prior = prior_index[(row["direction_index"], site)]
            require(prior["mode"] == row["mode"]
                    and prior["recipient_key"] == row["recipient_key"]
                    and prior["donor_key"] == row["donor_key"],
                    "prior W direction differs")
        eligibility[str(row["direction_index"])] = {
            "recipient_clean": recipient["eligible"], "donor_clean": donor["eligible"],
            "primary_target_word_eligible": target_only_transfer(recipient, donor,
                                                                   donor["normalized_words"]),
            "windows": {key: list(value) for key, value in positions.items()},
            "t": t,
        }
    require(sum(v["primary_target_word_eligible"] for v in eligibility.values()) == 4,
            "frozen primary denominator changed")
    auditor = HERE / "wav2vec2_fit_complement_patch_audit_v1.py"
    require(auditor.is_file(), "saved-data auditor absent")
    return {"schema": "wav2vec2_fit_complement_preflight_v1",
            "runner_sha256": sha(Path(__file__)),
            "runner_source_git_commit": source_commit(Path(__file__)),
            "auditor_sha256": sha(auditor),
            "auditor_source_git_commit": source_commit(auditor),
            "plan_v1_sha256": PLAN_V1_SHA, "plan_v2_sha256": PLAN_V2_SHA,
            "baseline_records_sha256": BASELINE_RECORDS_SHA,
            "baseline_audit_sha256": BASELINE_AUDIT_SHA,
            "baseline_manifest_sha256": BASELINE_MANIFEST_SHA,
            "prior_records_sha256": PRIOR_RECORDS_SHA,
            "prior_audit_sha256": PRIOR_AUDIT_SHA,
            "prior_manifest_sha256": PRIOR_MANIFEST_SHA,
            "base_preflight": base_receipt,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": sys.version.split()[0],
            "directions": rows, "eligibility": eligibility,
            "sites": list(SITES), "arms": list(ARMS),
            "seed_rule": "18470111 + 4*direction_index + 2*site_index + random_index",
            "random_complement_order": "single draw [1,T-|W|,768], left then right",
            "expected_patch_rows": 192,
            "limits": {"max_seconds": MAX_SECONDS, "max_rss_bytes": MAX_RSS_BYTES,
                       "max_output_bytes": MAX_OUTPUT_BYTES,
                       "rss_reserve_bytes": RSS_RESERVE_BYTES,
                       "output_reserve_bytes": OUTPUT_RESERVE_BYTES}}


def verify_freeze(freeze: Path, expected_preflight: Path,
                  receipt: dict[str, Any]) -> str:
    frozen = json.loads(freeze.read_text())
    require(frozen.get("schema") == "wav2vec2_fit_complement_freeze_v1"
            and frozen.get("status") == "independently_reviewed_complement_patch_frozen"
            and frozen.get("preflight_sha256") == sha(expected_preflight)
            and frozen.get("runner_sha256") == receipt["runner_sha256"]
            and frozen.get("auditor_sha256") == receipt["auditor_sha256"]
            and frozen.get("plan_v1_sha256") == receipt["plan_v1_sha256"]
            and frozen.get("plan_v2_sha256") == receipt["plan_v2_sha256"],
            "independently reviewed freeze receipt mismatched")
    for key in ("plan_review", "runner_source_review", "auditor_source_review"):
        item = frozen.get(key)
        require(isinstance(item, dict) and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str), f"freeze review absent: {key}")
        path = ROOT / item["path"]
        require(path.resolve().is_relative_to(ROOT) and path.is_file()
                and not path.is_symlink() and sha(path) == item["sha256"],
                f"freeze review changed: {key}")
        require("Decision: GO" in path.read_text(),
                f"freeze review lacks explicit GO decision: {key}")
    return sha(freeze)


def current_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def check_budget(start: float, output: Path, *, before_forward: bool = False) -> None:
    require(time.monotonic() - start <= MAX_SECONDS, "30-minute budget exhausted")
    rss = current_rss_bytes()
    disk = output_bytes(output)
    require(rss <= MAX_RSS_BYTES and disk <= MAX_OUTPUT_BYTES, "RSS/output cap exceeded")
    if before_forward:
        require(rss + RSS_RESERVE_BYTES <= MAX_RSS_BYTES
                and disk + OUTPUT_RESERVE_BYTES <= MAX_OUTPUT_BYTES,
                "insufficient reserved headroom before forward")


def save_array(path: Path, array: np.ndarray, output: Path, start: float) -> str:
    require(not path.exists() and array.dtype == np.float32
            and bool(np.isfinite(array).all()), "array output invalid")
    require(output_bytes(output) + array.nbytes + (1 << 20) <= MAX_OUTPUT_BYTES,
            "array would exceed 512-MiB cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npy.tmp")
    require(not temp.exists(), "partial temp capture already exists")
    try:
        with temp.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    check_budget(start, output)
    return sha(path)


def append_record(path: Path, row: dict[str, Any]) -> None:
    baseline.append_record(path, row)


def load_capture(mode: str, case_key: str, stage: str,
                 baseline_index: dict[tuple[str, str], dict[str, Any]]) -> np.ndarray:
    row = baseline_index[(mode, case_key)]
    path = BASELINE_DIR / "captures" / f"{mode}_{case_key}_{stage}.npy"
    require(sha(path) == row["capture_sha256"][stage], "capture changed during run")
    array = np.load(path, allow_pickle=False)
    require(array.dtype == np.float32 and tuple(array.shape) == tuple(row["capture_shapes"][stage])
            and bool(np.isfinite(array).all()), "capture array invalid")
    return array


def patched_forward(model: Any, input_tensor: Any, *, site: str,
                    clean_state: np.ndarray, patched_state: np.ndarray
                    ) -> tuple[np.ndarray, int]:
    import torch

    require(site in SITES, "unknown patch site")
    module = (model.wav2vec2.feature_projection if site == "feature_projection"
              else model.wav2vec2.encoder.layers[11])
    count = 0

    def hook(_module: Any, _args: tuple[Any, ...], result: Any) -> Any:
        nonlocal count
        count += 1
        require(count == 1 and isinstance(result, tuple) and len(result) >= 1,
                "patch site did not return the expected tuple once")
        first = result[0]
        require(torch.is_tensor(first) and first.device.type == "cpu"
                and first.dtype == torch.float32
                and tuple(first.shape) == clean_state.shape
                and bool(torch.isfinite(first).all()), "recipient hook tensor invalid")
        observed = first.detach().contiguous().numpy()
        require(np.array_equal(observed, clean_state), "live recipient state differs from baseline")
        replacement = torch.from_numpy(patched_state.copy())
        require(tuple(replacement.shape) == tuple(first.shape), "replacement shape changed")
        return (replacement, *result[1:])

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result = model(input_values=input_tensor, attention_mask=None,
                           output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    require(count == 1, "patch hook count differs")
    logits = result.logits.detach().cpu().contiguous().numpy().copy()
    require(logits.dtype == np.float32 and logits.ndim == 3
            and logits.shape[0] == 1 and logits.shape[-1] == 32
            and bool(np.isfinite(logits).all()), "patched logits invalid")
    return logits, count


def execute(expected_preflight: Path, freeze: Path, output: Path) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    current = no_weight_preflight()
    require(canonical(json.loads(expected_preflight.read_text())) == canonical(current),
            "frozen no-weight preflight differs")
    freeze_sha = verify_freeze(freeze, expected_preflight, current)
    require(not output.exists(), "output directory already exists")
    baseline_index, _ = read_baseline()
    prior_index = read_prior_w()
    cases, stats = baseline.trusted_cases()
    by_key = {case.key: case for case in cases}
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    append_record(records, {"type": "preflight", "schema": SCHEMA,
                            "receipt": current,
                            "expected_preflight_sha256": sha(expected_preflight),
                            "freeze_sha256": freeze_sha})
    start = time.monotonic()
    in_progress: dict[str, Any] | None = None
    control_failures: list[dict[str, Any]] = []
    try:
        check_budget(start, output, before_forward=True)
        torch.set_num_threads(4)
        processor = Wav2Vec2Processor.from_pretrained(baseline.SNAPSHOT, local_files_only=True)
        model, loading_info = Wav2Vec2ForCTC.from_pretrained(
            baseline.SNAPSHOT, local_files_only=True, output_loading_info=True)
        require(set(loading_info["missing_keys"]) == {"wav2vec2.masked_spec_embed"}
                and not loading_info["unexpected_keys"]
                and not loading_info["mismatched_keys"]
                and not loading_info["error_msgs"],
                "unanticipated checkpoint load report")
        model = model.to("cpu", dtype=torch.float32).eval()
        require(not model.training and not model.config.do_stable_layer_norm
                and model.config.mask_feature_prob == 0.0,
                "pinned eval/topology gate failed")
        require(all(p.device.type == "cpu" and p.dtype == torch.float32
                    for p in model.parameters()), "model not CPU float32")
        for direction in current["directions"]:
            d = direction["direction_index"]
            mode = direction["mode"]
            recipient_key = direction["recipient_key"]
            donor_key = direction["donor_key"]
            recipient_case = by_key[recipient_key]
            pcm = baseline.load_pcm(recipient_case)
            values = baseline.normalized_input(
                pcm, mode=mode, stats=stats[(recipient_case.pair_id, recipient_case.engine)],
                processor=processor)
            expected_input = load_capture(mode, recipient_key, "input_values", baseline_index)
            require(np.array_equal(values[None], expected_input),
                    "recipient input differs from reviewed baseline")
            input_tensor = torch.from_numpy(values.copy())[None]
            for site_index, site in enumerate(SITES):
                recipient_state = load_capture(mode, recipient_key, site, baseline_index)
                donor_state = load_capture(mode, donor_key, site, baseline_index)
                recipient_logits = load_capture(mode, recipient_key, "logits", baseline_index)
                donor_logits = load_capture(mode, donor_key, "logits", baseline_index)
                positions = windows(recipient_case.slot, recipient_state.shape[1])
                prior_w = prior_index[(d, site)]
                prior_w_output = prior_w_logits(prior_w)
                require(positions == {key: tuple(value) for key, value in
                                      current["eligibility"][str(d)]["windows"].items()},
                        "frozen window changed")
                for arm in ARMS:
                    in_progress = {"direction_index": d, "site": site, "arm": arm}
                    check_budget(start, output, before_forward=True)
                    random_index = 0 if arm == "R1_C" else 1 if arm == "R2_C" else None
                    seed = (RANDOM_SEED_BASE + 4*d + 2*site_index + random_index
                            if random_index is not None else None)
                    patch, delta_norm, natural_w_norm, natural_c_norm = candidate_state(
                        recipient_state, donor_state, arm=arm, window=positions, seed=seed)
                    forward_start = time.monotonic()
                    logits, hook_count = patched_forward(
                        model, input_tensor, site=site,
                        clean_state=recipient_state, patched_state=patch)
                    check_budget(start, output)
                    ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
                    transcript = processor.batch_decode([ids])[0]
                    words = baseline.normalized_words(transcript)
                    target = target_only_transfer(
                        baseline_index[(mode, recipient_key)],
                        baseline_index[(mode, donor_key)], words)
                    donor_transcript = baseline_index[(mode, donor_key)]["transcript"]
                    recipient_transcript = baseline_index[(mode, recipient_key)]["transcript"]
                    matches_donor = transcript == donor_transcript
                    matches_recipient = transcript == recipient_transcript
                    control_pass: bool | None = None
                    if arm == "W":
                        control_pass = (np.array_equal(logits, prior_w_output)
                                        and ids == prior_w["argmax_ids"]
                                        and transcript == prior_w["transcript"])
                    if arm == "self_C":
                        control_pass = (np.array_equal(logits, recipient_logits)
                                        and ids == baseline_index[(mode, recipient_key)]["argmax_ids"])
                    if arm == "full_donor":
                        control_pass = (np.array_equal(logits, donor_logits)
                                        and ids == baseline_index[(mode, donor_key)]["argmax_ids"])
                    label = f"d{d:02d}_{site}_{arm}"
                    patch_relative = f"captures/{label}_patch.npy"
                    logits_relative = f"captures/{label}_logits.npy"
                    patch_sha = save_array(output / patch_relative, patch, output, start)
                    logits_sha = save_array(output / logits_relative, logits, output, start)
                    changed_frames = int(np.count_nonzero(
                        np.any(logits[0] != recipient_logits[0], axis=-1)))
                    rdist = float(np.linalg.norm(logits.astype(np.float64)
                                                 - recipient_logits.astype(np.float64)))
                    ddist = float(np.linalg.norm(logits.astype(np.float64)
                                                 - donor_logits.astype(np.float64)))
                    append_record(records, {
                        "type": "patch", "schema": SCHEMA,
                        "direction_index": d, "pair_id": direction["pair_id"],
                        "engine": direction["engine"], "mode": mode,
                        "recipient_key": recipient_key, "donor_key": donor_key,
                        "site": site, "site_index": site_index, "arm": arm,
                        "seed": seed,
                        "windows": {k: list(v) for k, v in positions.items()},
                        "patch_path": patch_relative, "patch_sha256": patch_sha,
                        "patch_shape": list(patch.shape), "delta_l2": delta_norm,
                        "natural_w_l2": natural_w_norm,
                        "natural_c_l2": natural_c_norm,
                        "logits_path": logits_relative, "logits_sha256": logits_sha,
                        "logits_shape": list(logits.shape),
                        "argmax_ids": ids, "transcript": transcript,
                        "normalized_words": words,
                        "primary_eligible": current["eligibility"][str(d)]["primary_target_word_eligible"],
                        "target_only_transfer": target,
                        "matches_donor_transcript": matches_donor,
                        "matches_recipient_transcript": matches_recipient,
                        "control_pass": control_pass,
                        "changed_frames_vs_recipient": changed_frames,
                        "logits_l2_vs_recipient": rdist,
                        "logits_l2_vs_donor": ddist,
                        "hook_count": hook_count,
                        "elapsed_seconds": time.monotonic() - forward_start,
                        "rss_bytes": current_rss_bytes(),
                    })
                    if control_pass is False:
                        control_failures.append({"direction_index": d,
                                                 "site": site, "arm": arm})
                    in_progress = None
                    check_budget(start, output)
        append_record(records, {"type": "complete", "schema": SCHEMA,
                                "patch_rows": 192, "elapsed_seconds": time.monotonic() - start,
                                "rss_bytes": current_rss_bytes(),
                                "output_bytes": output_bytes(output),
                                "control_gate_pass": not control_failures,
                                "control_failures": control_failures,
                                "load_info": {
                                    "missing_keys": sorted(loading_info["missing_keys"]),
                                    "unexpected_keys": sorted(loading_info["unexpected_keys"]),
                                    "mismatched_keys": sorted(loading_info["mismatched_keys"]),
                                    "error_msgs": loading_info["error_msgs"]}})
    except BaseException as exc:
        append_record(records, {"type": "failure", "schema": SCHEMA,
                                "error_type": type(exc).__name__, "error": str(exc),
                                "in_progress": in_progress,
                                "elapsed_seconds": time.monotonic() - start,
                                "rss_bytes": current_rss_bytes(),
                                "output_bytes": output_bytes(output)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--expected-preflight", required=True, type=Path)
    run.add_argument("--reviewed-freeze", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        require(not args.output.exists(), "preflight output already exists")
        args.output.write_text(json.dumps(no_weight_preflight(), indent=2,
                                          sort_keys=True, allow_nan=False) + "\n")
        print(args.output)
    else:
        execute(args.expected_preflight, args.reviewed_freeze, args.output)


if __name__ == "__main__":
    main()
