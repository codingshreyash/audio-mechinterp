"""Prospective Track B Wav2Vec2 two-site state-swap assay.

``prepare`` checks geometry without authorizing inference. ``preflight`` and
``run`` require the independently reviewed 32/32 human listening receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import wav2vec2_fit_baseline_stage_capture_v1 as base
import wav2vec2_fit_complement_patch_v1 as fit
import wav2vec2_dev_word_disjoint_v1 as dev

from . import candidate_common as common

HERE = Path(__file__).resolve().parent
ROOT = common.ROOT
AUDITOR = HERE / "audit_wav2vec2.py"
SOURCE_REVIEW = HERE / "CANDIDATE_WAV2VEC2_SOURCE_INDEPENDENT_REVIEW.md"
SCHEMA = "word_disjoint_v2_wav2vec2_v1"
SEED_BASE = 26048191
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 512 * 1024**2


def shared_stats(cases: list[common.Case]) -> dict[tuple[str, str], dict[str, float]]:
    by_pair: dict[tuple[str, str], list[common.Case]] = {}
    for case in cases:
        by_pair.setdefault((case.pair_id, case.engine), []).append(case)
    stats = {}
    for key, pair in by_pair.items():
        common.require(len(pair) == 2 and pair[0].slot == pair[1].slot,
                       "paired source mismatch")
        a, b = (common.load_pcm(x) for x in pair)
        s, e = pair[0].slot
        common.require(len(a) == len(b) and 0 < s < e < len(a)
                       and np.array_equal(a[:s], b[:s])
                       and np.array_equal(a[e:], b[e:])
                       and not np.array_equal(a[s:e], b[s:e]),
                       "paired PCM invariant differs")
        outside = np.concatenate((a[:s], a[e:])).astype(np.float32) / np.float32(32768)
        mean, variance = np.float32(outside.mean()), np.float32(outside.var())
        common.require(bool(np.isfinite(mean) and np.isfinite(variance)
                            and variance > 0), "shared-stat input invalid")
        stats[key] = {"mean": float(mean), "variance": float(variance)}
    common.require(len(stats) == 16, "shared-stat pair count differs")
    return stats


def byte_ledger(cases: list[common.Case]) -> dict[str, int]:
    max_frames = max(c.frames for c in cases)
    t = (max_frames - 400) // 320 + 1
    clean = 64 * (max_frames*4 + 2*t*768*4 + t*32*4)
    patches = 768 * t*32*4
    headers = (64*4 + 768)*256
    metadata = 32*1024**2
    contingency = 64*1024**2
    total = clean + patches + headers + metadata + contingency
    common.require(total <= MAX_OUTPUT_BYTES, "Wav2Vec2 byte bound exceeds 512 MiB")
    return {"max_frames": max_frames, "max_t": t, "clean_payload": clean,
            "patch_logits_payload": patches, "npy_header_bound": headers,
            "metadata_bound": metadata, "contingency": contingency,
            "total_bound": total, "cap": MAX_OUTPUT_BYTES}


def preflight(*, human_qa: bool) -> dict[str, Any]:
    import torch
    import transformers
    cases, fixture = common.fixture_cases()
    stats = shared_stats(cases)
    directions = common.directions(cases, ("S", "C"))
    qa = common.qa_gate(cases) if human_qa else None
    receipt = {"schema": SCHEMA + "_preflight",
               "admitted_for_candidate_inference": bool(qa),
               "fixture": fixture, "qa": qa,
               "model_preflight": base.preflight(),
               "runner_sha256": common.sha(Path(__file__)),
               "runner_commit": dev.source_commit(Path(__file__)),
               "auditor_sha256": common.sha(AUDITOR),
               "auditor_commit": dev.source_commit(AUDITOR),
               "common_sha256": common.sha(Path(common.__file__)),
               "common_commit": dev.source_commit(Path(common.__file__)),
               "fit_hook_sha256": common.sha(Path(fit.__file__)),
               "fit_hook_commit": dev.source_commit(Path(fit.__file__)),
               "source_bindings": common.source_bindings({
                   "runner": Path(__file__), "auditor": AUDITOR,
                   "common": Path(common.__file__),
                   "helper": Path(fit.__file__),
                   "test": common.HERE / "test_candidate_source.py"}),
               "plan_sha256": common.sha(common.PLAN),
               "plan_review_sha256": common.sha(common.PLAN_REVIEW),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "transformers": transformers.__version__,
               "cases": [{"key": c.key, "wav_sha256": c.wav_sha256,
                          "pcm_sha256": c.pcm_sha256, "frames": c.frames,
                          "slot": list(c.slot), "opaque_id": c.opaque_id}
                         for c in cases],
               "directions": directions,
               "stats": {f"{p}|{e}": value for (p, e), value in stats.items()},
               "sites": list(common.SITES_W2V), "arms": list(common.ARMS),
               "seed_rule": "26048191+4*direction_index+2*site_index+(0=random_W,1=random_O)",
               "expected_clean": 64, "expected_patches": 768,
               "byte_ledger": byte_ledger(cases),
               "limits": {"seconds": MAX_SECONDS, "rss": MAX_RSS_BYTES,
                          "output": MAX_OUTPUT_BYTES, "threads": 4}}
    return receipt


def verify_freeze(preflight_path: Path, freeze_path: Path, current: dict) -> str:
    common.require(current["admitted_for_candidate_inference"] is True,
                   "32/32 human QA not admitted")
    frozen = json.loads(freeze_path.read_text())
    expected = {"preflight_sha256": common.sha(preflight_path),
                "runner_sha256": current["runner_sha256"],
                "auditor_sha256": current["auditor_sha256"],
                "common_sha256": current["common_sha256"],
                "fit_hook_sha256": current["fit_hook_sha256"],
                "source_bindings": current["source_bindings"],
                "plan_sha256": current["plan_sha256"],
                "plan_review_sha256": current["plan_review_sha256"],
                "qa_receipt_sha256": current["qa"]["qa_receipt_sha256"],
                "qa_review_sha256": current["qa"]["qa_review_sha256"]}
    common.require(frozen.get("schema") == SCHEMA + "_freeze"
                   and frozen.get("status") == "independently_reviewed_frozen"
                   and all(frozen.get(k) == v for k, v in expected.items()),
                   "Wav2Vec2 freeze does not bind reviewed input/source")
    common.require(SOURCE_REVIEW.is_file()
                   and "Decision: GO" in SOURCE_REVIEW.read_text()
                   and frozen.get("source_review_sha256") == common.sha(SOURCE_REVIEW),
                   "Wav2Vec2 source GO not bound")
    common.reviewed_freeze_paths(preflight_path, freeze_path, SOURCE_REVIEW,
                                 current["source_bindings"])
    return common.sha(freeze_path)


def check_budget(start: float, output: Path, before_forward: bool = False) -> None:
    elapsed = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    written = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    common.require(elapsed <= MAX_SECONDS and rss <= MAX_RSS_BYTES
                   and written <= MAX_OUTPUT_BYTES, "Wav2Vec2 resource cap reached")
    if before_forward:
        common.require(rss + 1024**3 <= MAX_RSS_BYTES
                       and written + 32*1024**2 <= MAX_OUTPUT_BYTES,
                       "Wav2Vec2 forward headroom exhausted")


def capture_clean(model: Any, tensor: Any, frames: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import torch
    captured: dict[str, np.ndarray] = {}
    counts = {site: 0 for site in common.SITES_W2V}
    handles = []
    for site, module in (("feature_projection", model.wav2vec2.feature_projection),
                         ("post_block11", model.wav2vec2.encoder.layers[11])):
        def hook(_module: Any, _args: tuple, result: Any, site: str = site) -> None:
            counts[site] += 1
            common.require(counts[site] == 1 and isinstance(result, tuple)
                           and len(result) >= 1, "clean Wav2Vec2 hook topology changed")
            captured[site] = base.validate_tensor(site, result[0], frames)
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            result = model(input_values=tensor, attention_mask=None,
                           output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    common.require(counts == {site: 1 for site in common.SITES_W2V},
                   "clean Wav2Vec2 capture incomplete")
    return captured, base.validate_tensor("logits", result.logits, frames)


def execute(preflight_path: Path, freeze_path: Path, output: Path) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    current = preflight(human_qa=True)
    common.require(dev.canonical(json.loads(preflight_path.read_text())) == dev.canonical(current),
                   "Wav2Vec2 preflight/source/input drift")
    freeze_sha = verify_freeze(preflight_path, freeze_path, current)
    common.require(not output.exists(), "Wav2Vec2 output already exists")
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    base.append_record(records, {"type": "preflight", "schema": SCHEMA,
                                 "receipt": current, "preflight_sha256": common.sha(preflight_path),
                                 "freeze_sha256": freeze_sha})
    started = time.monotonic()
    in_progress = None
    try:
        torch.set_num_threads(4)
        check_budget(started, output, before_forward=True)
        processor = Wav2Vec2Processor.from_pretrained(base.SNAPSHOT, local_files_only=True)
        model, info = Wav2Vec2ForCTC.from_pretrained(
            base.SNAPSHOT, local_files_only=True, output_loading_info=True)
        common.require(set(info["missing_keys"]) == {"wav2vec2.masked_spec_embed"}
                       and not info["unexpected_keys"] and not info["mismatched_keys"]
                       and not info["error_msgs"], "Wav2Vec2 checkpoint load drift")
        model = model.to("cpu", dtype=torch.float32).eval()
        common.require(not model.training and all(p.device.type == "cpu"
                       and p.dtype == torch.float32 for p in model.parameters()),
                       "Wav2Vec2 not CPU float32 eval")
        cases, _ = common.fixture_cases()
        stats = shared_stats(cases)
        clean: dict[tuple[str, str], dict] = {}
        for case in cases:
            pcm = common.load_pcm(case)
            for mode in ("S", "C"):
                in_progress = {"phase": "clean", "case": case.key, "mode": mode}
                check_budget(started, output, before_forward=True)
                values = base.normalized_input(
                    pcm, mode=mode, stats=stats[(case.pair_id, case.engine)], processor=processor)
                states, logits = capture_clean(model, torch.from_numpy(values.copy())[None], case.frames)
                ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
                transcript = processor.batch_decode([ids])[0]
                paths, hashes, shapes = {}, {}, {}
                arrays = {"input": values[None], "feature_projection": states["feature_projection"],
                          "post_block11": states["post_block11"], "logits": logits}
                for name, array in arrays.items():
                    rel = f"captures/{mode}_{case.key}_{name}.npy"
                    paths[name], shapes[name] = rel, list(array.shape)
                    hashes[name] = fit.save_array(output / rel, array, output, started)
                row = {"type": "clean", "schema": SCHEMA, "case_key": case.key,
                       "pair_id": case.pair_id, "engine": case.engine,
                       "word": case.word, "mode": mode, "paths": paths,
                       "sha256": hashes, "shape": shapes, "argmax_ids": ids,
                       "transcript": transcript, "tokens": common.tokens(transcript),
                       "reference_tokens": common.reference_tokens(case.word),
                       "eligible": common.tokens(transcript) == common.reference_tokens(case.word),
                       "hook_counts": {site: 1 for site in common.SITES_W2V}}
                base.append_record(records, row)
                clean[(case.key, mode)] = row
                in_progress = None
                check_budget(started, output)
        for direction in current["directions"]:
            d, mode = direction["index"], direction["mode"]
            recipient = clean[(direction["recipient_key"], mode)]
            donor = clean[(direction["donor_key"], mode)]
            input_values = np.load(output / recipient["paths"]["input"], allow_pickle=False)
            rlogits = np.load(output / recipient["paths"]["logits"], allow_pickle=False)
            dlogits = np.load(output / donor["paths"]["logits"], allow_pickle=False)
            strict = recipient["eligible"] and donor["eligible"]
            for site_index, site in enumerate(common.SITES_W2V):
                rstate = np.load(output / recipient["paths"][site], allow_pickle=False)
                dstate = np.load(output / donor["paths"][site], allow_pickle=False)
                window = dev.frame_window(tuple(direction["slot"]), rstate.shape[1])
                for arm in common.ARMS:
                    in_progress = {"phase": "patch", "direction": d, "site": site, "arm": arm}
                    check_budget(started, output, before_forward=True)
                    seed = (SEED_BASE + 4*d + 2*site_index
                            + (0 if arm == "random_W" else 1)
                            if arm in ("random_W", "random_O") else None)
                    patch, norms = dev.edit_state(rstate, dstate, window, arm, seed)
                    logits, count = fit.patched_forward(
                        model, torch.from_numpy(input_values.copy()), site=site,
                        clean_state=rstate, patched_state=patch)
                    ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
                    transcript = processor.batch_decode([ids])[0]
                    rel = f"captures/d{d:02d}_{site}_{arm}_logits.npy"
                    digest = fit.save_array(output / rel, logits, output, started)
                    control = (bool(np.array_equal(logits, rlogits)) if arm == "self" else
                               bool(np.array_equal(logits, dlogits)) if arm == "full" else None)
                    row = {"type": "patch", "schema": SCHEMA, "direction": d,
                           "site": site, "arm": arm, "recipient_key": direction["recipient_key"],
                           "donor_key": direction["donor_key"], "mode": mode,
                           "window": list(window), "seed": seed,
                           "patch_tensor_sha256": dev.tensor_sha(patch), "norms": norms,
                           "logits_path": rel, "logits_sha256": digest,
                           "logits_shape": list(logits.shape), "argmax_ids": ids,
                           "transcript": transcript, "tokens": common.tokens(transcript),
                           "strict_eligible": strict,
                           "matches_donor_transcript": transcript == donor["transcript"],
                           "matches_recipient_transcript": transcript == recipient["transcript"],
                           "intended_word_only": strict and
                           common.tokens(transcript) == donor["reference_tokens"],
                           "control_pass": control, "hook_count": count}
                    base.append_record(records, row)
                    in_progress = None
                    check_budget(started, output)
        base.append_record(records, {"type": "complete", "schema": SCHEMA,
                                     "clean_rows": 64, "patch_rows": 768,
                                     "elapsed_seconds": time.monotonic() - started,
                                     "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                     "output_bytes": fit.output_bytes(output)})
    except BaseException as exc:
        base.append_record(records, {"type": "failure", "schema": SCHEMA,
                                     "error_type": type(exc).__name__, "error": str(exc),
                                     "in_progress": in_progress,
                                     "elapsed_seconds": time.monotonic() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("preflight")
    run = sub.add_parser("run")
    for key in ("preflight", "freeze", "output"):
        run.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(preflight(human_qa=False), sort_keys=True, indent=2))
    elif args.command == "preflight":
        print(json.dumps(preflight(human_qa=True), sort_keys=True, indent=2))
    else:
        execute(args.preflight, args.freeze, args.output)


if __name__ == "__main__":
    main()
