"""Independent saved-data audit of Wav2Vec2 FIT activation patches.

Loads no model weights. Reconstructs the complete matrix, patch tensors,
greedy CTC transcripts, and output controls from the pinned baseline and
saved logits. A pass verifies saved evidence, not that a forward happened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import wav2vec2_fit_causal_patch_v1 as runner

HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"nonfinite JSON constant: {value}")))


def read_rows(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "records absent or symlink")
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "records JSONL truncated")
    rows = [parse_json(line) for line in raw.decode("utf-8").splitlines()]
    require(all(isinstance(row, dict) for row in rows), "non-object row")
    return rows


def checked_array(output: Path, relative: str, expected_sha: str,
                  expected_shape: tuple[int, ...]) -> np.ndarray:
    require(isinstance(relative, str) and relative.startswith("captures/")
            and ".." not in Path(relative).parts
            and isinstance(expected_sha, str) and HEX.fullmatch(expected_sha) is not None,
            "sidecar reference invalid")
    path = output / relative
    require(path.resolve().is_relative_to(output.resolve()) and path.is_file()
            and not path.is_symlink() and sha(path) == expected_sha,
            f"sidecar path/hash invalid: {relative}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(array.dtype == np.float32 and tuple(array.shape) == expected_shape
            and bool(np.isfinite(array).all()),
            f"sidecar shape/dtype/finite invalid: {relative}")
    return array


def independent_patch(recipient: np.ndarray, donor: np.ndarray,
                      arm: str, windows: dict[str, tuple[int, int]],
                      seed: int | None) -> np.ndarray:
    """Build patch without invoking the runner's patch function."""
    require(recipient.shape == donor.shape and recipient.dtype == donor.dtype == np.float32,
            "baseline state shape/dtype mismatch")
    result = recipient.copy()
    if arm == "full_donor":
        return donor.copy()
    if arm == "self_W":
        return result
    a, b = windows["W" if arm in ("R1", "R2") else arm]
    if arm in ("R1", "R2"):
        require(type(seed) is int, "random seed absent")
        delta64 = donor[:, a:b].astype(np.float64) - recipient[:, a:b].astype(np.float64)
        radius = float(np.linalg.norm(delta64))
        require(radius > 0 and math.isfinite(radius), "random target radius invalid")
        z = np.random.Generator(np.random.PCG64(seed)).standard_normal(delta64.shape)
        zrad = float(np.linalg.norm(z))
        require(zrad > 0 and math.isfinite(zrad), "random draw radius invalid")
        result[:, a:b] = recipient[:, a:b] + (z * (radius / zrad)).astype(np.float32)
    else:
        result[:, a:b] = donor[:, a:b]
    return result


def independent_target_only(recipient: dict[str, Any], donor: dict[str, Any],
                            actual: list[str]) -> bool:
    rwords = recipient["normalized_words"]
    dwords = donor["normalized_words"]
    rw, dw = recipient["word"], donor["word"]
    if rwords.count(rw) != 1 or dwords.count(dw) != 1:
        return False
    if rw in dwords or dw in rwords or len(rwords) != len(dwords):
        return False
    modified = list(rwords)
    modified[modified.index(rw)] = dw
    return modified == dwords == actual


def audit(output: Path, expected_preflight: Path, freeze: Path) -> dict[str, Any]:
    from transformers import Wav2Vec2Processor

    receipt = runner.no_weight_preflight()
    require(runner.canonical(parse_json(expected_preflight.read_text()))
            == runner.canonical(receipt), "frozen preflight differs")
    freeze_sha = runner.verify_freeze(freeze, expected_preflight, receipt)
    rows = read_rows(output / "records.jsonl")
    require(len(rows) == 226, "incomplete or extra causal matrix rows")
    require(rows[0].get("type") == "preflight"
            and rows[0].get("schema") == runner.SCHEMA
            and rows[0].get("receipt") == receipt
            and rows[0].get("expected_preflight_sha256") == sha(expected_preflight)
            and rows[0].get("freeze_sha256") == freeze_sha,
            "run preflight record differs")
    require(rows[-1].get("type") == "complete"
            and rows[-1].get("schema") == runner.SCHEMA
            and rows[-1].get("patch_rows") == 224,
            "completion record absent")
    require(not any(row.get("type") == "failure" for row in rows),
            "failure record disqualifies completed run")
    index, _ = runner.read_baseline()
    processor = Wav2Vec2Processor.from_pretrained(
        runner.baseline.SNAPSHOT, local_files_only=True)
    expected_keys = [(direction["direction_index"], site, arm)
                     for direction in receipt["directions"]
                     for site in runner.SITES for arm in runner.ARMS]
    actual_keys = [(row.get("direction_index"), row.get("site"), row.get("arm"))
                   for row in rows[1:-1]]
    require(actual_keys == expected_keys, "patch row order/coverage differs")
    counts: Counter[tuple[str, str, str]] = Counter()
    expected_files = {"records.jsonl"}
    for row in rows[1:-1]:
        require(row.get("type") == "patch" and row.get("schema") == runner.SCHEMA,
                "non-patch matrix row")
        d = row["direction_index"]
        direction = receipt["directions"][d]
        site = row["site"]
        arm = row["arm"]
        site_index = runner.SITES.index(site)
        mode = direction["mode"]
        recipient_key = direction["recipient_key"]
        donor_key = direction["donor_key"]
        require(row.get("pair_id") == direction["pair_id"]
                and row.get("engine") == direction["engine"]
                and row.get("mode") == mode
                and row.get("recipient_key") == recipient_key
                and row.get("donor_key") == donor_key
                and row.get("site_index") == site_index,
                "direction identity differs")
        recipient = index[(mode, recipient_key)]
        donor = index[(mode, donor_key)]
        source = runner.load_capture(mode, recipient_key, site, index)
        donor_state = runner.load_capture(mode, donor_key, site, index)
        rlogits = runner.load_capture(mode, recipient_key, "logits", index)
        dlogits = runner.load_capture(mode, donor_key, "logits", index)
        s, e = direction["slot_samples"]
        wstart = math.ceil((s - 399) / 320)
        wend = (e - 1) // 320 + 1
        width = wend - wstart
        windows = {"W": (wstart, wend), "P": (wstart - width, wstart),
                   "Q": (wend, wend + width)}
        require(all(0 <= a < b <= source.shape[1] for a, b in windows.values())
                and row.get("windows") == {key: list(value) for key, value in windows.items()},
                "patch windows differ")
        random_index = 0 if arm == "R1" else 1 if arm == "R2" else None
        seed = (runner.RANDOM_SEED_BASE + 4*d + 2*site_index + random_index
                if random_index is not None else None)
        require(row.get("seed") == seed, "random seed differs")
        expected_patch = independent_patch(source, donor_state, arm, windows, seed)
        label = f"d{d:02d}_{site}_{arm}"
        patch_relative = f"captures/{label}_patch.npy"
        logits_relative = f"captures/{label}_logits.npy"
        require(row.get("patch_path") == patch_relative
                and row.get("logits_path") == logits_relative
                and row.get("patch_shape") == list(source.shape)
                and row.get("logits_shape") == list(rlogits.shape),
                "sidecar identity/shape receipt differs")
        patch = checked_array(output, patch_relative, row.get("patch_sha256"),
                              tuple(source.shape))
        logits = checked_array(output, logits_relative, row.get("logits_sha256"),
                               tuple(rlogits.shape))
        expected_files.update((patch_relative, logits_relative))
        require(np.array_equal(patch, expected_patch), "saved patch differs from independent reconstruction")
        delta64 = patch.astype(np.float64) - source.astype(np.float64)
        delta_norm = float(np.linalg.norm(delta64))
        require(math.isclose(row.get("delta_l2"), delta_norm, rel_tol=1e-10, abs_tol=1e-10),
                "patch delta norm differs")
        ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
        transcript = processor.batch_decode([ids])[0]
        words = runner.baseline.normalized_words(transcript)
        require(row.get("argmax_ids") == ids and row.get("transcript") == transcript
                and row.get("normalized_words") == words,
                "complete CTC decode differs")
        eligible = independent_target_only(recipient, donor, donor["normalized_words"])
        target = independent_target_only(recipient, donor, words)
        require(row.get("primary_eligible") is eligible
                and row.get("target_only_transfer") is target,
                "target-word outcome/eligibility differs")
        changed = int(np.count_nonzero(np.any(logits[0] != rlogits[0], axis=-1)))
        rdist = float(np.linalg.norm(logits.astype(np.float64)
                                     - rlogits.astype(np.float64)))
        ddist = float(np.linalg.norm(logits.astype(np.float64)
                                     - dlogits.astype(np.float64)))
        require(row.get("changed_frames_vs_recipient") == changed
                and math.isclose(row.get("logits_l2_vs_recipient"), rdist,
                                 rel_tol=1e-10, abs_tol=1e-10)
                and math.isclose(row.get("logits_l2_vs_donor"), ddist,
                                 rel_tol=1e-10, abs_tol=1e-10),
                "logit distance/frame summary differs")
        control_pass: bool | None = None
        if arm == "self_W":
            control_pass = np.array_equal(logits, rlogits) and ids == recipient["argmax_ids"]
        elif arm == "full_donor":
            control_pass = np.array_equal(logits, dlogits) and ids == donor["argmax_ids"]
        require(row.get("control_pass") is control_pass
                and control_pass is not False,
                "self/full-donor control failed or was misreported")
        require(row.get("hook_count") == 1
                and type(row.get("rss_bytes")) is int
                and 0 < row["rss_bytes"] <= runner.MAX_RSS_BYTES
                and isinstance(row.get("elapsed_seconds"), (int, float))
                and 0 <= row["elapsed_seconds"] <= runner.MAX_SECONDS,
                "hook/resource receipt invalid")
        counts[(site, arm, "eligible" if eligible else "descriptive")] += int(target)
    inventory = {path.relative_to(output).as_posix(): path.stat().st_size
                 for path in output.rglob("*") if path.is_file()}
    require(set(inventory) == expected_files, "sidecar inventory differs")
    complete = rows[-1]
    load = complete.get("load_info", {})
    require(load.get("missing_keys") == ["wav2vec2.masked_spec_embed"]
            and load.get("unexpected_keys") == []
            and load.get("mismatched_keys") == []
            and load.get("error_msgs") == [], "checkpoint load report differs")
    require(isinstance(complete.get("elapsed_seconds"), (int, float))
            and 0 <= complete["elapsed_seconds"] <= runner.MAX_SECONDS
            and type(complete.get("rss_bytes")) is int
            and 0 < complete["rss_bytes"] <= runner.MAX_RSS_BYTES,
            "completion resource receipt invalid")
    last_line_bytes = len((output / "records.jsonl").read_bytes().splitlines(keepends=True)[-1])
    require(type(complete.get("output_bytes")) is int
            and complete["output_bytes"] == sum(inventory.values()) - last_line_bytes
            and sum(inventory.values()) <= runner.MAX_OUTPUT_BYTES,
            "output byte receipt/cap differs")
    return {"status": "independent_saved_data_audit_pass",
            "records_sha256": sha(output / "records.jsonl"),
            "preflight_sha256": sha(expected_preflight),
            "freeze_sha256": freeze_sha,
            "patch_rows": 224,
            "primary_directions": sum(
                bool(item["primary_target_word_eligible"])
                for item in receipt["eligibility"].values()),
            "target_only_counts": {f"{site}/{arm}/{group}": counts[(site, arm, group)]
                                   for site in runner.SITES for arm in runner.ARMS
                                   for group in ("eligible", "descriptive")},
            "sidecar_count": len(expected_files) - 1,
            "sidecar_bytes": sum(size for path, size in inventory.items()
                                 if path != "records.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-preflight", required=True, type=Path)
    parser.add_argument("--reviewed-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "audit output already exists")
    result = audit(args.run_dir, args.expected_preflight, args.reviewed_freeze)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
