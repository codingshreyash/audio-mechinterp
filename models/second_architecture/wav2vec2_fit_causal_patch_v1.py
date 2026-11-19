"""Prospective FIT-only Wav2Vec2 activation-patch assay.

The ``preflight`` command reads source, assets and saved baseline, but loads no
model weights. ``run`` requires a byte-identical preflight and a separately
reviewed freeze receipt. See WAV2VEC2_FIT_CAUSAL_PATCH_PROSPECTIVE_V1.md and
its V2 topology amendment. This is an internal ASR assay, not an audio attack.
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
PLAN_V1 = HERE / "WAV2VEC2_FIT_CAUSAL_PATCH_PROSPECTIVE_V1.md"
PLAN_V1_SHA = "23a4303d03cf9c6483985555ac2abc1e6768059bb024527dbc3dfa7d170c5dcd"
PLAN_V2 = HERE / "WAV2VEC2_FIT_CAUSAL_PATCH_PROSPECTIVE_V2.md"
PLAN_V2_SHA = "d936f8728ae89715982bcf9c9dc06e7efc6a876a6ace894306af4e84aa21c977"
SCHEMA = "wav2vec2_fit_causal_patch_v1"
SITES = ("feature_projection", "post_block11")
ARMS = ("W", "P", "Q", "self_W", "full_donor", "R1", "R2")
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 512 * 1024**2
RSS_RESERVE_BYTES = 1024**3
OUTPUT_RESERVE_BYTES = 32 * 1024**2
RANDOM_SEED_BASE = 971103


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
        for stage in (*SITES, "logits"):
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
    k = end - start
    result = {"W": (start, end), "P": (start - k, start),
              "Q": (end, end + k)}
    require(k > 0 and all(0 <= a < b <= t for a, b in result.values()),
            "word/control window outside captured frames")
    return result


def candidate_state(recipient: np.ndarray, donor: np.ndarray, *, arm: str,
                    window: dict[str, tuple[int, int]], seed: int | None
                    ) -> tuple[np.ndarray, float]:
    require(recipient.shape == donor.shape and recipient.ndim == 3
            and recipient.shape[0] == 1 and recipient.shape[-1] == 768
            and recipient.dtype == donor.dtype == np.float32,
            "patch state shape/dtype mismatch")
    require(arm in ARMS, "unknown arm")
    patched = recipient.copy()
    if arm == "full_donor":
        patched[:] = donor
        delta = donor.astype(np.float64) - recipient.astype(np.float64)
    else:
        a, b = window["W" if arm in ("self_W", "R1", "R2") else arm]
        if arm in ("R1", "R2"):
            require(seed is not None, "random seed missing")
            target = donor[:, a:b].astype(np.float64) - recipient[:, a:b].astype(np.float64)
            norm = float(np.linalg.norm(target))
            require(math.isfinite(norm) and norm > 0, "zero/nonfinite donor word delta")
            random = np.random.Generator(np.random.PCG64(seed)).standard_normal(target.shape)
            random_norm = float(np.linalg.norm(random))
            require(math.isfinite(random_norm) and random_norm > 0, "invalid random draw")
            edit = (random * (norm / random_norm)).astype(np.float32)
            patched[:, a:b] = recipient[:, a:b] + edit
        elif arm != "self_W":
            patched[:, a:b] = donor[:, a:b]
        delta = patched.astype(np.float64) - recipient.astype(np.float64)
    require(bool(np.isfinite(patched).all()), "nonfinite patch state")
    if arm == "self_W":
        require(np.array_equal(patched, recipient), "self patch differs")
    elif arm == "full_donor":
        require(np.array_equal(patched, donor), "full donor state differs")
    else:
        a, b = window["W" if arm.startswith("R") else arm]
        require(np.array_equal(patched[:, :a], recipient[:, :a])
                and np.array_equal(patched[:, b:], recipient[:, b:]),
                "patch differs outside declared rows")
    return patched, float(np.linalg.norm(delta))


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

    require(sha(PLAN_V1) == PLAN_V1_SHA, "causal plan V1 changed")
    require(PLAN_V2.is_file() and sha(PLAN_V2) == PLAN_V2_SHA,
            "topology amendment V2 absent or changed")
    cases, _ = baseline.trusted_cases()
    baseline_index, _ = read_baseline()
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
        eligibility[str(row["direction_index"])] = {
            "recipient_clean": recipient["eligible"], "donor_clean": donor["eligible"],
            "primary_target_word_eligible": target_only_transfer(recipient, donor,
                                                                   donor["normalized_words"]),
            "windows": {key: list(value) for key, value in positions.items()},
            "t": t,
        }
    require(sum(v["primary_target_word_eligible"] for v in eligibility.values()) == 4,
            "frozen primary denominator changed")
    auditor = HERE / "wav2vec2_fit_causal_patch_audit_v1.py"
    require(auditor.is_file(), "saved-data auditor absent")
    return {"schema": "wav2vec2_fit_causal_preflight_v1",
            "runner_sha256": sha(Path(__file__)),
            "runner_source_git_commit": source_commit(Path(__file__)),
            "auditor_sha256": sha(auditor),
            "auditor_source_git_commit": source_commit(auditor),
            "plan_v1_sha256": PLAN_V1_SHA, "plan_v2_sha256": PLAN_V2_SHA,
            "baseline_records_sha256": BASELINE_RECORDS_SHA,
            "baseline_audit_sha256": BASELINE_AUDIT_SHA,
            "baseline_manifest_sha256": BASELINE_MANIFEST_SHA,
            "base_preflight": base_receipt,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": sys.version.split()[0],
            "directions": rows, "eligibility": eligibility,
            "sites": list(SITES), "arms": list(ARMS),
            "seed_rule": "971103 + 4*direction_index + 2*site_index + random_index",
            "expected_patch_rows": 224,
            "limits": {"max_seconds": MAX_SECONDS, "max_rss_bytes": MAX_RSS_BYTES,
                       "max_output_bytes": MAX_OUTPUT_BYTES,
                       "rss_reserve_bytes": RSS_RESERVE_BYTES,
                       "output_reserve_bytes": OUTPUT_RESERVE_BYTES}}


def verify_freeze(freeze: Path, expected_preflight: Path,
                  receipt: dict[str, Any]) -> str:
    frozen = json.loads(freeze.read_text())
    require(frozen.get("schema") == "wav2vec2_fit_causal_freeze_v1"
            and frozen.get("status") == "independently_reviewed_causal_patch_frozen"
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
                require(positions == {key: tuple(value) for key, value in
                                      current["eligibility"][str(d)]["windows"].items()},
                        "frozen window changed")
                for arm in ARMS:
                    in_progress = {"direction_index": d, "site": site, "arm": arm}
                    check_budget(start, output, before_forward=True)
                    random_index = 0 if arm == "R1" else 1 if arm == "R2" else None
                    seed = (RANDOM_SEED_BASE + 4*d + 2*site_index + random_index
                            if random_index is not None else None)
                    patch, delta_norm = candidate_state(
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
                    control_pass: bool | None = None
                    if arm == "self_W":
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
                        "logits_path": logits_relative, "logits_sha256": logits_sha,
                        "logits_shape": list(logits.shape),
                        "argmax_ids": ids, "transcript": transcript,
                        "normalized_words": words,
                        "primary_eligible": current["eligibility"][str(d)]["primary_target_word_eligible"],
                        "target_only_transfer": target,
                        "control_pass": control_pass,
                        "changed_frames_vs_recipient": changed_frames,
                        "logits_l2_vs_recipient": rdist,
                        "logits_l2_vs_donor": ddist,
                        "hook_count": hook_count,
                        "elapsed_seconds": time.monotonic() - forward_start,
                        "rss_bytes": current_rss_bytes(),
                    })
                    if control_pass is False:
                        raise RuntimeError(f"saved {arm} control failed at direction {d}/{site}")
                    in_progress = None
                    check_budget(start, output)
        append_record(records, {"type": "complete", "schema": SCHEMA,
                                "patch_rows": 224, "elapsed_seconds": time.monotonic() - start,
                                "rss_bytes": current_rss_bytes(),
                                "output_bytes": output_bytes(output),
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
