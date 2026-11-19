"""Independent saved-output audit for the Track B Wav2Vec2 two-site assay.

It loads the frozen tokenizer/processor, not model weights. Every constructed
patch is independently rebuilt from clean state sidecars and the frozen seed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import wav2vec2_dev_word_disjoint_audit_v1 as dev_audit
import wav2vec2_fit_baseline_stage_capture_v1 as base

from . import candidate_common as common
from . import run_wav2vec2 as runner


def audit(output: Path, preflight_path: Path, freeze_path: Path) -> dict[str, Any]:
    from transformers import Wav2Vec2Processor
    current = runner.preflight(human_qa=True)
    common.require(base.canonical(json.loads(preflight_path.read_text())) == base.canonical(current),
                   "Wav2Vec2 preflight drift")
    freeze_sha = runner.verify_freeze(preflight_path, freeze_path, current)
    records = output / "records.jsonl"
    common.require(records.is_file() and not records.is_symlink(), "Wav2Vec2 records absent")
    raw = records.read_bytes()
    common.require(raw.endswith(b"\n"), "truncated Wav2Vec2 records")
    rows = [dev_audit.parse(line) for line in raw.decode("utf-8").splitlines()]
    common.require(len(rows) == 834 and rows[0].get("type") == "preflight"
                   and rows[-1].get("type") == "complete", "Wav2Vec2 matrix incomplete")
    common.require(rows[0].get("receipt") == current
                   and rows[0].get("preflight_sha256") == common.sha(preflight_path)
                   and rows[0].get("freeze_sha256") == freeze_sha
                   and rows[-1].get("clean_rows") == 64
                   and rows[-1].get("patch_rows") == 768
                   and not any(x.get("type") == "failure" for x in rows),
                   "Wav2Vec2 run/provenance incomplete")
    cases, _ = common.fixture_cases()
    stats = runner.shared_stats(cases)
    expected_clean = [(c.key, mode) for c in cases for mode in ("S", "C")]
    clean_rows = rows[1:65]
    common.require([(x.get("case_key"), x.get("mode")) for x in clean_rows] == expected_clean,
                   "Wav2Vec2 clean row coverage/order differs")
    processor = Wav2Vec2Processor.from_pretrained(base.SNAPSHOT, local_files_only=True)
    expected_files = {"records.jsonl"}
    clean: dict[tuple[str, str], tuple[dict, dict[str, np.ndarray]]] = {}
    for row in clean_rows:
        case = next(c for c in cases if c.key == row["case_key"])
        mode = row["mode"]
        t = (case.frames - 400) // 320 + 1
        shapes = {"input": (1, case.frames),
                  "feature_projection": (1, t, 768),
                  "post_block11": (1, t, 768), "logits": (1, t, 32)}
        common.require(row.get("type") == "clean" and row.get("schema") == runner.SCHEMA
                       and row.get("pair_id") == case.pair_id
                       and row.get("engine") == case.engine
                       and row.get("word") == case.word
                       and row.get("hook_counts") == {s: 1 for s in common.SITES_W2V},
                       "Wav2Vec2 clean identity/hook invalid")
        arrays = {}
        for name, shape in shapes.items():
            rel = f"captures/{mode}_{case.key}_{name}.npy"
            common.require(row["paths"].get(name) == rel
                           and row["shape"].get(name) == list(shape),
                           "Wav2Vec2 clean sidecar receipt invalid")
            arrays[name] = dev_audit.array(output, rel, row["sha256"][name], shape)
            expected_files.add(rel)
        pcm = common.load_pcm(case)
        reconstructed = base.normalized_input(
            pcm, mode=mode, stats=stats[(case.pair_id, case.engine)], processor=processor)
        common.require(np.array_equal(arrays["input"][0], reconstructed),
                       "Wav2Vec2 processor input differs")
        ids = arrays["logits"].argmax(axis=-1)[0].astype(np.int64).tolist()
        transcript = processor.batch_decode([ids])[0]
        words = dev_audit.token_list(transcript)
        ref = common.reference_tokens(case.word)
        common.require(row.get("argmax_ids") == ids and row.get("transcript") == transcript
                       and row.get("tokens") == words and row.get("reference_tokens") == ref
                       and row.get("eligible") is (words == ref),
                       "Wav2Vec2 clean full-output result differs")
        clean[(case.key, mode)] = (row, arrays)
    grouped: dict[tuple[str, str], list[common.Case]] = {}
    for case in cases:
        grouped.setdefault((case.pair_id, case.engine), []).append(case)
    for pair in grouped.values():
        first = clean[(pair[0].key, "C")][1]["input"][0]
        second = clean[(pair[1].key, "C")][1]["input"][0]
        s, e = pair[0].slot
        common.require(np.array_equal(first[:s], second[:s])
                       and np.array_equal(first[e:], second[e:]),
                       "shared-stat model inputs differ outside word")
    patch_rows = rows[65:-1]
    expected = [(d["index"], site, arm) for d in current["directions"]
                for site in common.SITES_W2V for arm in common.ARMS]
    common.require([(x.get("direction"), x.get("site"), x.get("arm"))
                    for x in patch_rows] == expected, "Wav2Vec2 patch coverage/order differs")
    counts: Counter[str] = Counter()
    control_failures = []
    for row in patch_rows:
        d = current["directions"][row["direction"]]
        site = row["site"]
        arm = row["arm"]
        mode = d["mode"]
        rec, ra = clean[(d["recipient_key"], mode)]
        donor, da = clean[(d["donor_key"], mode)]
        common.require(row.get("type") == "patch" and row.get("schema") == runner.SCHEMA
                       and row.get("recipient_key") == d["recipient_key"]
                       and row.get("donor_key") == d["donor_key"]
                       and row.get("mode") == mode and row.get("hook_count") == 1,
                       "Wav2Vec2 patch identity/hook differs")
        s, e = d["slot"]
        t = ra[site].shape[1]
        a, b = math.ceil((s - 399) / 320), (e - 1) // 320 + 1
        common.require(0 < a < b < t and row.get("window") == [a, b],
                       "Wav2Vec2 patch window differs")
        seed = (runner.SEED_BASE + 4*d["index"] +
                2*common.SITES_W2V.index(site) + (0 if arm == "random_W" else 1)
                if arm in ("random_W", "random_O") else None)
        common.require(row.get("seed") == seed, "Wav2Vec2 patch seed differs")
        patch = dev_audit.patch_independent(ra[site], da[site], (a, b), arm, seed)
        common.require(row.get("patch_tensor_sha256") == dev_audit.tensor_sha(patch),
                       "Wav2Vec2 patch hash differs")
        rw, dw = ra[site][:, a:b], da[site][:, a:b]
        ro = np.concatenate((ra[site][:, :a], ra[site][:, b:]), axis=1)
        do = np.concatenate((da[site][:, :a], da[site][:, b:]), axis=1)
        norms = {"natural_w_l2": float(np.linalg.norm(dw.astype(np.float64)-rw.astype(np.float64))),
                 "natural_o_l2": float(np.linalg.norm(do.astype(np.float64)-ro.astype(np.float64))),
                 "realized_l2": float(np.linalg.norm(patch.astype(np.float64)-ra[site].astype(np.float64)))}
        common.require(all(math.isclose(row["norms"].get(k), v, rel_tol=1e-10, abs_tol=1e-10)
                           for k, v in norms.items()), "Wav2Vec2 edit norm differs")
        rel = f"captures/d{d['index']:02d}_{site}_{arm}_logits.npy"
        common.require(row.get("logits_path") == rel
                       and row.get("logits_shape") == list(ra["logits"].shape),
                       "Wav2Vec2 patch output path/shape differs")
        logits = dev_audit.array(output, rel, row["logits_sha256"],
                                 tuple(ra["logits"].shape))
        expected_files.add(rel)
        ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
        transcript = processor.batch_decode([ids])[0]
        words = dev_audit.token_list(transcript)
        strict = rec["eligible"] and donor["eligible"]
        control = (bool(np.array_equal(logits, ra["logits"])) if arm == "self" else
                   bool(np.array_equal(logits, da["logits"])) if arm == "full" else None)
        common.require(row.get("argmax_ids") == ids
                       and row.get("transcript") == transcript
                       and row.get("tokens") == words
                       and row.get("strict_eligible") is strict
                       and row.get("matches_donor_transcript") is (transcript == donor["transcript"])
                       and row.get("matches_recipient_transcript") is (transcript == rec["transcript"])
                       and row.get("intended_word_only") is (strict and words == donor["reference_tokens"])
                       and row.get("control_pass") == control,
                       "Wav2Vec2 patched full-output/control result differs")
        if control is False:
            control_failures.append((d["index"], site, arm))
        counts[f"{site}/{arm}"] += 1
    common.require(all(counts[f"{site}/{arm}"] == 64
                       for site in common.SITES_W2V for arm in common.ARMS),
                   "Wav2Vec2 arm population differs")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    allowed = expected_files | ({"audit.json"} if (output / "audit.json").exists() else set())
    common.require(actual_files == allowed, "Wav2Vec2 output files extra/missing")
    bytes_used = sum((output / name).stat().st_size for name in actual_files)
    common.require(bytes_used <= runner.MAX_OUTPUT_BYTES, "Wav2Vec2 output cap exceeded")
    return {"schema": runner.SCHEMA + "_audit",
            "status": "independent_saved_data_audit_pass",
            "records_sha256": common.sha(records),
            "clean_rows": 64, "patch_rows": 768,
            "arm_counts": dict(counts), "control_failures": control_failures,
            "mechanism_gate_pass": not control_failures,
            "output_bytes": bytes_used,
            "byte_ledger_bound": current["byte_ledger"]["total_bound"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output, args.preflight, args.freeze)
    target = args.output / "audit.json"
    common.require(not target.exists(), "refusing to overwrite Wav2Vec2 audit")
    target.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
