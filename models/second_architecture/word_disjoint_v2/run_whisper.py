"""Prospective Track B Whisper Tiny two-site state-swap assay.

The no-weight ``prepare`` enumerates cases and storage. ``preflight`` and
``run`` stay closed until independently audited 32/32 blinded human QA.
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

import whisper_fit_word_window_causal_v1 as fit_whisper
import wav2vec2_fit_complement_patch_v1 as fit_save
import wav2vec2_dev_word_disjoint_v1 as dev

from . import candidate_common as common

HERE = Path(__file__).resolve().parent
AUDITOR = HERE / "audit_whisper.py"
SOURCE_REVIEW = HERE / "CANDIDATE_WHISPER_SOURCE_INDEPENDENT_REVIEW.md"
SCHEMA = "word_disjoint_v2_whisper_v1"
SEED_BASE = 36048191
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 512 * 1024**2
MAX_NEW_TOKENS = 128


def active_frames(case: common.Case) -> int:
    n = math.ceil(case.frames / 320)
    common.require(0 < n <= 1500, "Whisper active frame count invalid")
    return n


def window(case: common.Case) -> tuple[int, int]:
    s, e = case.slot
    n = active_frames(case)
    a, b = max(0, math.floor(s / 320) - 5), min(n, math.ceil(e / 320) + 5)
    common.require(0 < a < b < n, "Whisper word/outside support invalid")
    return a, b


def byte_ledger(cases: list[common.Case]) -> dict[str, int]:
    mel_bytes = 80 * 3000 * 4
    state_bytes = 1500 * 384 * 4
    clean = 32 * (mel_bytes + 2*state_bytes)
    headers = (32 * 3) * 256
    metadata = 32*1024**2
    contingency = 64*1024**2
    total = clean + headers + metadata + contingency
    common.require(total <= MAX_OUTPUT_BYTES, "Whisper byte bound exceeds 512 MiB")
    return {"mel_bytes_per_case": mel_bytes, "state_bytes_per_site": state_bytes,
            "clean_payload": clean, "npy_header_bound": headers,
            "metadata_bound": metadata, "contingency": contingency,
            "total_bound": total, "cap": MAX_OUTPUT_BYTES}


def preflight(*, human_qa: bool) -> dict[str, Any]:
    import torch
    import transformers
    cases, fixture = common.fixture_cases()
    for case in cases:
        window(case)
    directions = common.directions(cases, ("stock",))
    qa = common.qa_gate(cases) if human_qa else None
    return {"schema": SCHEMA + "_preflight",
            "admitted_for_candidate_inference": bool(qa),
            "fixture": fixture, "qa": qa,
            "model_preflight": fit_whisper.preflight(),
            "runner_sha256": common.sha(Path(__file__)),
            "runner_commit": dev.source_commit(Path(__file__)),
            "auditor_sha256": common.sha(AUDITOR),
            "auditor_commit": dev.source_commit(AUDITOR),
            "common_sha256": common.sha(Path(common.__file__)),
            "common_commit": dev.source_commit(Path(common.__file__)),
            "fit_helper_sha256": common.sha(Path(fit_whisper.__file__)),
            "fit_helper_commit": dev.source_commit(Path(fit_whisper.__file__)),
            "plan_sha256": common.sha(common.PLAN),
            "plan_review_sha256": common.sha(common.PLAN_REVIEW),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cases": [{"key": c.key, "wav_sha256": c.wav_sha256,
                       "pcm_sha256": c.pcm_sha256, "frames": c.frames,
                       "slot": list(c.slot), "window": list(window(c)),
                       "active_frames": active_frames(c), "opaque_id": c.opaque_id}
                      for c in cases],
            "directions": directions, "sites": list(common.SITES_WHISPER),
            "arms": list(common.ARMS),
            "seed_rule": "36048191+4*direction_index+2*site_index+(0=random_W,1=random_O)",
            "generation": {"do_sample": False, "num_beams": 1,
                           "max_new_tokens": MAX_NEW_TOKENS,
                           "output_scores": False,
                           "return_dict_in_generate": True,
                           "return_attention_mask": True},
            "expected_clean": 32, "expected_patches": 384,
            "byte_ledger": byte_ledger(cases),
            "limits": {"seconds": MAX_SECONDS, "rss": MAX_RSS_BYTES,
                       "output": MAX_OUTPUT_BYTES, "threads": 4}}


def verify_freeze(preflight_path: Path, freeze_path: Path, current: dict) -> str:
    common.require(current["admitted_for_candidate_inference"] is True,
                   "32/32 human QA not admitted")
    frozen = json.loads(freeze_path.read_text())
    expected = {"preflight_sha256": common.sha(preflight_path),
                "runner_sha256": current["runner_sha256"],
                "auditor_sha256": current["auditor_sha256"],
                "common_sha256": current["common_sha256"],
                "fit_helper_sha256": current["fit_helper_sha256"],
                "plan_sha256": current["plan_sha256"],
                "plan_review_sha256": current["plan_review_sha256"],
                "qa_receipt_sha256": current["qa"]["qa_receipt_sha256"],
                "qa_review_sha256": current["qa"]["qa_review_sha256"]}
    common.require(frozen.get("schema") == SCHEMA + "_freeze"
                   and frozen.get("status") == "independently_reviewed_frozen"
                   and all(frozen.get(k) == v for k, v in expected.items()),
                   "Whisper freeze does not bind reviewed input/source")
    common.require(SOURCE_REVIEW.is_file()
                   and "Decision: GO" in SOURCE_REVIEW.read_text()
                   and frozen.get("source_review_sha256") == common.sha(SOURCE_REVIEW),
                   "Whisper source GO not bound")
    common.reviewed_freeze_paths(preflight_path, freeze_path, SOURCE_REVIEW,
                                 current["runner_commit"])
    return common.sha(freeze_path)


def check_budget(start: float, output: Path, before_forward: bool = False) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    written = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    common.require(time.monotonic() - start <= MAX_SECONDS and rss <= MAX_RSS_BYTES
                   and written <= MAX_OUTPUT_BYTES, "Whisper resource cap reached")
    if before_forward:
        common.require(rss + 1024**3 <= MAX_RSS_BYTES
                       and written + 32*1024**2 <= MAX_OUTPUT_BYTES,
                       "Whisper forward headroom exhausted")


def patch_state(recipient: np.ndarray, donor: np.ndarray, n: int,
                word_window: tuple[int, int], arm: str, seed: int | None
                ) -> tuple[np.ndarray, dict[str, float]]:
    common.require(recipient.shape == donor.shape == (1, 1500, 384)
                   and recipient.dtype == donor.dtype == np.float32
                   and np.isfinite(recipient).all() and np.isfinite(donor).all(),
                   "Whisper clean state shape/dtype invalid")
    a, b = word_window
    common.require(0 < a < b < n <= 1500, "Whisper patch window invalid")
    rw, dw = recipient[:, a:b], donor[:, a:b]
    ro = np.concatenate((recipient[:, :a], recipient[:, b:n]), axis=1)
    do = np.concatenate((donor[:, :a], donor[:, b:n]), axis=1)
    nw = float(np.linalg.norm(dw.astype(np.float64) - rw.astype(np.float64)))
    no = float(np.linalg.norm(do.astype(np.float64) - ro.astype(np.float64)))
    common.require(min(nw, no) > 0 and math.isfinite(nw) and math.isfinite(no),
                   "Whisper natural region edit zero/nonfinite")
    result = recipient.copy()
    if arm == "self":
        pass
    elif arm == "W":
        result[:, a:b] = dw
    elif arm == "O":
        result[:, :a], result[:, b:n] = donor[:, :a], donor[:, b:n]
    elif arm == "full":
        result[:, :n] = donor[:, :n]
    elif arm in ("random_W", "random_O"):
        common.require(type(seed) is int, "Whisper random seed absent")
        source = rw if arm == "random_W" else ro
        radius = nw if arm == "random_W" else no
        z = np.random.Generator(np.random.PCG64(seed)).standard_normal(source.shape)
        z_norm = float(np.linalg.norm(z))
        common.require(z_norm > 0 and math.isfinite(z_norm), "Whisper random draw invalid")
        edited = source + (z * (radius / z_norm)).astype(np.float32)
        if arm == "random_W":
            result[:, a:b] = edited
        else:
            result[:, :a], result[:, b:n] = edited[:, :a], edited[:, a:]
    else:
        raise ValueError("Whisper arm invalid")
    common.require(np.array_equal(result[:, n:], recipient[:, n:])
                   and np.isfinite(result).all(), "Whisper padded/finite invariant failed")
    realized = float(np.linalg.norm(result.astype(np.float64)-recipient.astype(np.float64)))
    return result, {"natural_w_l2": nw, "natural_o_l2": no,
                    "realized_l2": realized}


def _hook_module(model: Any, site: str) -> Any:
    encoder = model.model.encoder
    if site == "post_block0":
        return encoder.layers[0]
    if site == "post_norm":
        return encoder.layer_norm
    raise ValueError("unknown Whisper site")


def generate(model: Any, processor: Any, batch: Any, *,
             captures: dict[str, np.ndarray] | None = None,
             patch: tuple[str, np.ndarray, np.ndarray] | None = None
             ) -> tuple[dict[str, Any], dict[str, int]]:
    import torch
    counts = {site: 0 for site in common.SITES_WHISPER}
    handles = []
    if captures is not None:
        for site in common.SITES_WHISPER:
            def hook(_module: Any, _args: tuple, output: Any, site: str = site) -> None:
                counts[site] += 1
                common.require(counts[site] == 1, "Whisper clean hook repeated")
                fit_whisper.validate_state(output)
                captures[site] = output.detach().cpu().contiguous().numpy().copy()
            handles.append(_hook_module(model, site).register_forward_hook(hook))
    if patch is not None:
        site, clean_state, patched_state = patch
        def patch_hook(_module: Any, _args: tuple, output: Any) -> Any:
            counts[site] += 1
            common.require(counts[site] == 1, "Whisper patch hook repeated")
            fit_whisper.validate_state(output)
            observed = output.detach().cpu().contiguous().numpy()
            common.require(np.array_equal(observed, clean_state),
                           "Whisper live recipient state changed from clean baseline")
            return torch.from_numpy(patched_state.copy())
        handles.append(_hook_module(model, site).register_forward_hook(patch_hook))
    try:
        with torch.inference_mode():
            generated = model.generate(**batch, do_sample=False, num_beams=1,
                                       max_new_tokens=MAX_NEW_TOKENS,
                                       output_scores=False, return_dict_in_generate=True)
    finally:
        for handle in handles:
            handle.remove()
    ids = generated.sequences[0].detach().cpu().tolist()
    transcript = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
    eos = model.generation_config.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    censored = not bool(ids and ids[-1] in eos_set)
    return {"ids": ids, "transcript": transcript,
            "tokens": common.tokens(transcript), "censored": censored}, counts


def execute(preflight_path: Path, freeze_path: Path, output: Path) -> None:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    current = preflight(human_qa=True)
    common.require(dev.canonical(json.loads(preflight_path.read_text())) == dev.canonical(current),
                   "Whisper preflight/source/input drift")
    freeze_sha = verify_freeze(preflight_path, freeze_path, current)
    common.require(not output.exists(), "Whisper output already exists")
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    dev.base.append_record(records, {"type": "preflight", "schema": SCHEMA,
                                     "receipt": current, "preflight_sha256": common.sha(preflight_path),
                                     "freeze_sha256": freeze_sha})
    started = time.monotonic()
    in_progress = None
    try:
        torch.set_num_threads(4)
        check_budget(started, output, before_forward=True)
        processor = WhisperProcessor.from_pretrained(
            fit_whisper.MODEL_ID, revision=fit_whisper.MODEL_REVISION,
            local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            fit_whisper.MODEL_ID, revision=fit_whisper.MODEL_REVISION,
            local_files_only=True).to("cpu", dtype=torch.float32).eval()
        common.require(not model.training and all(p.device.type == "cpu"
                       and p.dtype == torch.float32 for p in model.parameters()),
                       "Whisper not CPU float32 eval")
        cases, _ = common.fixture_cases()
        by_key = {c.key: c for c in cases}
        clean: dict[str, dict] = {}
        for case in cases:
            in_progress = {"phase": "clean", "case": case.key}
            check_budget(started, output, before_forward=True)
            pcm = common.load_pcm(case)
            samples = pcm.astype(np.float32) / np.float32(32768)
            batch = processor(samples, sampling_rate=16000,
                              return_tensors="pt", return_attention_mask=True)
            common.require(tuple(batch["input_features"].shape) == (1, 80, 3000)
                           and "attention_mask" in batch,
                           "Whisper processor output changed")
            captures: dict[str, np.ndarray] = {}
            result, counts = generate(model, processor, batch, captures=captures)
            common.require(counts == {site: 1 for site in common.SITES_WHISPER},
                           "Whisper clean hook incomplete")
            arrays = {"mel": batch["input_features"].detach().cpu().numpy().copy(),
                      **captures}
            paths, hashes, shapes = {}, {}, {}
            for name, array in arrays.items():
                rel = f"captures/{case.key}_{name}.npy"
                paths[name], shapes[name] = rel, list(array.shape)
                hashes[name] = fit_save.save_array(output / rel, array, output, started)
            mask = batch["attention_mask"].detach().cpu().contiguous().numpy()
            row = {"type": "clean", "schema": SCHEMA, "case_key": case.key,
                   "pair_id": case.pair_id, "engine": case.engine, "word": case.word,
                   "paths": paths, "sha256": hashes, "shape": shapes,
                   "attention_mask_shape": list(mask.shape),
                   "attention_mask_sha256": dev.tensor_sha(mask),
                   **result, "reference_tokens": common.reference_tokens(case.word),
                   "eligible": not result["censored"] and
                   result["tokens"] == common.reference_tokens(case.word),
                   "hook_counts": counts}
            dev.base.append_record(records, row)
            clean[case.key] = row
            in_progress = None
            check_budget(started, output)
        for direction in current["directions"]:
            d = direction["index"]
            recipient = clean[direction["recipient_key"]]
            donor = clean[direction["donor_key"]]
            case = by_key[direction["recipient_key"]]
            n = active_frames(case)
            w = window(case)
            strict = recipient["eligible"] and donor["eligible"]
            mel = np.load(output / recipient["paths"]["mel"], allow_pickle=False)
            # Rebuild the exact attention mask from the original recipient WAV.
            pcm = common.load_pcm(case)
            samples = pcm.astype(np.float32) / np.float32(32768)
            batch = processor(samples, sampling_rate=16000,
                              return_tensors="pt", return_attention_mask=True)
            common.require(np.array_equal(batch["input_features"].numpy(), mel),
                           "Whisper live mel differs from saved clean input")
            for site_index, site in enumerate(common.SITES_WHISPER):
                rstate = np.load(output / recipient["paths"][site], allow_pickle=False)
                dstate = np.load(output / donor["paths"][site], allow_pickle=False)
                for arm in common.ARMS:
                    in_progress = {"phase": "patch", "direction": d,
                                   "site": site, "arm": arm}
                    check_budget(started, output, before_forward=True)
                    seed = (SEED_BASE + 4*d + 2*site_index
                            + (0 if arm == "random_W" else 1)
                            if arm in ("random_W", "random_O") else None)
                    patched, norms = patch_state(rstate, dstate, n, w, arm, seed)
                    result, counts = generate(
                        model, processor, batch, patch=(site, rstate, patched))
                    control = (result["ids"] == recipient["ids"] if arm == "self" else
                               result["ids"] == donor["ids"] if arm == "full" else None)
                    row = {"type": "patch", "schema": SCHEMA, "direction": d,
                           "site": site, "arm": arm,
                           "recipient_key": direction["recipient_key"],
                           "donor_key": direction["donor_key"],
                           "window": list(w), "active_frames": n, "seed": seed,
                           "patch_tensor_sha256": dev.tensor_sha(patched),
                           "norms": norms, **result,
                           "strict_eligible": strict,
                           "matches_donor_transcript": result["transcript"] == donor["transcript"],
                           "matches_recipient_transcript": result["transcript"] == recipient["transcript"],
                           "intended_word_only": strict and not result["censored"] and
                           result["tokens"] == donor["reference_tokens"],
                           "control_pass": control, "hook_counts": counts}
                    dev.base.append_record(records, row)
                    in_progress = None
                    check_budget(started, output)
        dev.base.append_record(records, {"type": "complete", "schema": SCHEMA,
                                         "clean_rows": 32, "patch_rows": 384,
                                         "elapsed_seconds": time.monotonic() - started,
                                         "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                         "output_bytes": fit_save.output_bytes(output)})
    except BaseException as exc:
        dev.base.append_record(records, {"type": "failure", "schema": SCHEMA,
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
