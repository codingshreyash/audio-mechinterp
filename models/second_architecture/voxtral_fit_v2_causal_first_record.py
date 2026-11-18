"""Frozen-mask Voxtral v2 FIT causal benchmark, first pair/voice only.

CPU --validate-only never loads model weights. --run explicitly performs the
complete first-record workload, then projects eight-record cost; it does not
continue to the other seven records. This is a feasibility/implementation
benchmark, not a population-level causal result or a security finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import voxtral_fit_pilot_capability_v1 as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = HERE / "VOXTRAL_FIT_V2_CAUSAL_DESIGN.md"
MASKS = HERE / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json"
SELECTION = HERE / "VOXTRAL_FIT_V2_SELECTION.json"
INDEX = HERE / "voxtral_corpus_v1_index.json"
GATE = HERE / "runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/v2_gate.json"
PILOT = HERE / "runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/result.json"
RUNS = HERE / "runs/voxtral_fit_v2_causal_first_record"
EXPECTED = {
    "design": "cd2cbd762af3568f85d43c2dec47824666a95d689a3dc89d9cbdf6471ef54ab0",
    "masks": "a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e",
    "selection": "f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b",
    "index": "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971",
    "gate": "79622d883e954a5076afac2e97b3809833e7fff5ea89b4ae5ccbd137b819429f",
    "pilot": "6cc84dc1a4ced9ce17380eab3836079c462c9974f31337f146416730b5536ccf",
    "v1_runner": "1d4d65a528d558e5bdda3b733eaab6e7e849ab9456d6842d5ee178d3efdfb8d8",
}
SITES = ("encoder31", "projector")
ARMS = ("restore", "loss")
PATCHES = ("whole", "word_0", "word_1", "word_2", "word_3", "pre", "post", "padding", "self", "random")
MAX_SECONDS = 45 * 60
NORM_REL_TOL = 0.05
NORM_ABS_TOL = 1e-6


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha(tensor) -> str:
    import torch
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return sha_bytes(value.numpy().tobytes())


def classify(ids: list[int], decoded: str, special_ids: set[int], eos: int,
             recipient: str, opposite: str) -> str:
    if any(t in special_ids and not (i == len(ids) - 1 and t == eos)
           for i, t in enumerate(ids)):
        return "malformed_special"
    if not ids or ids[-1] != eos:
        return "missing_eos"
    surface = decoded.strip()
    if not surface or not re.fullmatch(r"[A-Za-z]+", surface, flags=re.ASCII):
        return "malformed_surface"
    if surface.casefold() == recipient.casefold():
        return "recipient"
    if surface.casefold() == opposite.casefold():
        return "opposite"
    return "other"


def random_seed(case_id: str, site: str, arm: str) -> tuple[str, int]:
    label = f"voxtral-v2-causal-v1|{case_id}|{site}|{arm}|random_0"
    return label, int.from_bytes(hashlib.sha256(label.encode()).digest(), "big") & ((1 << 63) - 1)


def row_indices(site_map: dict, name: str) -> list[int]:
    if name == "whole" or name == "random" or name == "self":
        return site_map["whole_word_rows"]
    if name == "padding":
        return site_map["padding_control_rows"]
    return site_map["bins"][name]


def prepare_edit(base_state, donor_state, rows: list[int], whole_rows: list[int],
                 mode: str, case_id: str, site: str, arm: str):
    """Return BF16 edited rows and frozen-control diagnostics; pure CPU/MPS tensor math."""
    import torch

    if base_state.shape != donor_state.shape or not rows or not whole_rows:
        raise ValueError("Source/donor shape or row mask invalid")
    if base_state.dtype != torch.bfloat16 or donor_state.dtype != torch.bfloat16:
        raise ValueError("Patch states must be BF16")
    b = base_state[rows].float()
    d = donor_state[rows].float()
    word_b = base_state[whole_rows].float()
    word_d = donor_state[whole_rows].float()
    word_effective = word_d.to(torch.bfloat16).float() - word_b
    target = float(torch.linalg.vector_norm(word_effective))
    raw = d - b
    raw_norm = float(torch.linalg.vector_norm(raw))
    info = {"mode": mode, "rows": rows, "target_bf16_norm": target,
            "unscaled_edit_norm": raw_norm, "alpha": None, "valid": True}
    if mode == "self":
        edited = base_state[rows].clone()
        info["alpha"] = 0.0
    elif mode == "random":
        label, seed = random_seed(case_id, site, arm)
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(tuple(b.shape), generator=g, dtype=torch.float32)
        noise -= noise.mean()
        noise_norm = float(torch.linalg.vector_norm(noise))
        info.update({"random_seed_label": label, "random_seed": seed,
                     "unscaled_random_sha256": tensor_sha(noise),
                     "unscaled_edit_norm": noise_norm})
        if target <= 0 or noise_norm <= 0:
            info.update(valid=False, invalid_reason="zero_target_or_random_norm")
            return None, info
        info["alpha"] = target / noise_norm
        edited = (b.cpu() + info["alpha"] * noise).to(torch.bfloat16).to(base_state.device)
    elif mode in ("pre", "post", "padding"):
        if target <= 0 or raw_norm <= 0:
            info.update(valid=False, invalid_reason="zero_target_or_control_norm")
            return None, info
        info["alpha"] = target / raw_norm
        edited = (b + info["alpha"] * raw).to(torch.bfloat16)
    else:
        edited = donor_state[rows].clone()
        info["alpha"] = 1.0
    effective = float(torch.linalg.vector_norm(edited.float() - b))
    info.update({"effective_bf16_norm": effective,
                 "edited_bf16_sha256": tensor_sha(edited)})
    if mode in ("pre", "post", "padding", "random") and abs(effective - target) > max(NORM_REL_TOL * target, NORM_ABS_TOL):
        info.update(valid=False, invalid_reason="bf16_effective_norm_out_of_tolerance")
        return None, info
    return edited, info


def verify_frozen() -> tuple[dict, dict, dict, dict]:
    files = {"design": DESIGN, "masks": MASKS, "selection": SELECTION, "index": INDEX,
             "gate": GATE, "pilot": PILOT, "v1_runner": Path(base.__file__)}
    for name, path in files.items():
        if base.sha(path) != EXPECTED[name]:
            raise ValueError(f"Frozen {name} hash changed: {path}")
    if base.sha(base.SOURCE_MANIFEST) != base.EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Frozen corpus source manifest changed")
    masks = json.loads(MASKS.read_text())
    selection = json.loads(SELECTION.read_text())
    index = json.loads(INDEX.read_text())
    pilot = json.loads(PILOT.read_text())
    gate = json.loads(GATE.read_text())
    if masks["v2_selection_sha256"] != EXPECTED["selection"] or masks["source_index_sha256"] != EXPECTED["index"]:
        raise ValueError("Mask provenance mismatch")
    if not gate["intervention_gate_pass"] or not gate["stage2_all_16_exact_pass"]:
        raise ValueError("V2 natural Stage-2 gate failed")
    if len(masks["records"]) != 8 or len(selection["clips"]) != 32:
        raise ValueError("V2 population wrong size")
    first = masks["records"][0]
    if (first["pair_id"], first["engine"], first["phrasing_id"]) != ("animal_08", "piper", "fit_chosen"):
        raise ValueError("First benchmark record changed")
    if len(first["clips"]) != 2 or len(first["words"]) != 2:
        raise ValueError("First pair missing an endpoint")
    indexed = {(r["pair_id"], r["engine"], r["phrasing_id"]): r for r in index["rows"]}
    row = indexed[(first["pair_id"], first["engine"], first["phrasing_id"])]
    if row["words"] != first["words"]:
        raise ValueError("V2 first pair words changed")
    for clip in first["clips"]:
        if clip["wav_sha256"] != row["clips"][clip["word"]]["wav_sha256"]:
            raise ValueError("Mask/index WAV mismatch")
        path = (ROOT / clip["wav_path"]).resolve()
        if not path.is_relative_to((ROOT / "eval/audio_jlens/corpus_v1/rendered/clips").resolve()) or base.sha(path) != clip["wav_sha256"]:
            raise ValueError("Selected FIT WAV changed")
        p_rows = [r for r in pilot["records"] if r["pair_id"] == first["pair_id"] and
                  r["engine"] == first["engine"] and r["phrasing_id"] == first["phrasing_id"] and
                  r["expected_word"] == clip["word"]]
        if len(p_rows) != 1 or not p_rows[0]["classification"]["exact_expected_word_and_eos"]:
            raise ValueError("First pair natural result not exact")
    return first, row, masks, pilot


def make_batches(processor, record: dict) -> dict:
    import torch
    batches = {}
    for clip in record["clips"]:
        path = (ROOT / clip["wav_path"]).resolve()
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": str(path)}, {"type": "text", "text": base.PROMPT},
        ]}]
        batch = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")
        if tuple(batch["input_features"].shape) != (1, 128, 3000) or batch["input_ids"][0].tolist().count(24) != 375:
            raise ValueError("Unexpected first-record processor output")
        batches[clip["word"]] = batch
    words = record["words"]
    if not torch.equal(batches[words[0]]["input_ids"], batches[words[1]]["input_ids"]):
        raise ValueError("Matched FIT words have different prompt token IDs")
    return batches


def blank_features(features, record: dict):
    import torch
    mask = record["word_mel_frames_direct_overlap"]
    blank = features.clone()
    blank[:, :, mask] = features[:, :, 2999:3000]
    outside = torch.ones(3000, dtype=torch.bool)
    outside[mask] = False
    if not torch.equal(blank[:, :, outside], features[:, :, outside]):
        raise AssertionError("Word blank altered off-word features")
    return blank


def feature_conditions(batches: dict, record: dict) -> list[tuple[str, str, object]]:
    import torch
    a, b = record["words"]
    fa, fb = batches[a]["input_features"], batches[b]["input_features"]
    word = record["word_mel_frames_direct_overlap"]
    is_word = torch.zeros(3000, dtype=torch.bool)
    is_word[word] = True
    first_padding = min(i for i in range(3000) if 160 * i - 200 >= record["clips"][0]["frames"])
    out = [(f"native_{a}", a, fa.clone()), (f"native_{b}", b, fb.clone())]
    for donor, outside, fd, fo in ((a, b, fa, fb), (b, a, fb, fa)):
        hybrid = fo.clone()
        hybrid[:, :, is_word] = fd[:, :, is_word]
        out.append((f"word_{donor}_outside_{outside}", donor, hybrid))
        padded = fd.clone()
        padded[:, :, first_padding:] = fo[:, :, first_padding:]
        out.append((f"{donor}_with_{outside}_padding", donor, padded))
    if len(out) != 6:
        raise AssertionError("Wrong feature matrix size")
    return out


def frontend_check(processor, batches: dict, record: dict) -> dict:
    """Independently reproduce pinned Whisper features and shared-max control."""
    import torch
    from voxtral_preprocessing_decomposition import independent_raw_log_mel, normalize_raw

    raw = {}
    maxima = {}
    errors = {}
    for clip in record["clips"]:
        word = clip["word"]
        wav = (ROOT / clip["wav_path"]).resolve()
        raw[word], maxima[word] = independent_raw_log_mel(wav, processor.feature_extractor.mel_filters)
        reconstructed = normalize_raw(raw[word], maxima[word])
        errors[word] = float((reconstructed - batches[word]["input_features"][0]).abs().max())
        if errors[word] > 1e-6:
            raise ValueError(f"Independent Whisper feature reproduction failed for {word}: {errors[word]}")
    a, b = record["words"]
    shared_max = max(maxima.values())
    shared_a = normalize_raw(raw[a], shared_max)
    shared_b = normalize_raw(raw[b], shared_max)
    outside = torch.ones(3000, dtype=torch.bool)
    outside[record["word_mel_frames_direct_overlap"]] = False
    shared_outside_error = float((shared_a[:, outside] - shared_b[:, outside]).abs().max())
    if shared_outside_error > 1e-6:
        raise ValueError(f"Shared-max off-word mel mismatch: {shared_outside_error}")
    return {"raw_log_mel_maximum": maxima, "independent_reproduction_error": errors,
            "shared_maximum": shared_max, "shared_max_off_word_max_abs_error": shared_outside_error}


def site_module(model, site: str):
    return model.model.audio_tower.layers[31] if site == "encoder31" else model.model.multi_modal_projector


def expected_hook_shape(site: str) -> tuple[int, ...]:
    if site == "encoder31":
        return (1, 1500, 1280)
    if site == "projector":
        return (375, 3072)
    raise ValueError(site)


def capture_state(model, batch, site: str):
    import torch
    hits = []

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Audio hook output is not a tensor")
        if tuple(output.shape) != expected_hook_shape(site) or output.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected raw {site} hook shape/dtype: {tuple(output.shape)}/{output.dtype}")
        hits.append(output.detach().to("cpu", dtype=torch.bfloat16).clone())

    handle = site_module(model, site).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**{k: v.to("mps") for k, v in batch.items()}, use_cache=False, logits_to_keep=1)
        torch.mps.synchronize()
    finally:
        handle.remove()
    shape = expected_hook_shape(site)
    if len(hits) != 1 or tuple(hits[0].shape) != shape:
        raise ValueError(f"Unexpected {site} capture count/shape")
    state = hits[0][0] if site == "encoder31" else hits[0]
    return state, {"hook_calls": len(hits), "raw_hook_shape": list(shape),
                   "raw_hook_dtype": str(hits[0].dtype),
                   "stored_state_shape": list(state.shape), "stored_state_dtype": str(state.dtype)}


def answer_and_scores(model, processor, batch, recipient: str, opposite: str,
                      canonical: dict[str, list[int]], site: str | None = None,
                      rows: list[int] | None = None, edited=None) -> dict:
    import torch
    hits = []
    handle = None
    if site is not None:
        if edited is not None:
            width = expected_hook_shape(site)[-1]
            nrows = expected_hook_shape(site)[-2]
            if rows is None or not rows or min(rows) < 0 or max(rows) >= nrows:
                raise ValueError("Edited row list outside hook tensor")
            if tuple(edited.shape) != (len(rows), width) or edited.dtype != torch.bfloat16:
                raise ValueError(f"Edited rows wrong shape/dtype for {site}: {tuple(edited.shape)}/{edited.dtype}")
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Audio hook output is not tensor")
            if tuple(output.shape) != expected_hook_shape(site) or output.dtype != torch.bfloat16:
                raise ValueError(f"Unexpected raw {site} hook shape/dtype: {tuple(output.shape)}/{output.dtype}")
            hits.append({"shape": list(output.shape), "dtype": str(output.dtype)})
            if edited is None:
                return output
            patched = output.clone()
            if site == "encoder31":
                patched[0, rows] = edited.to(device=output.device, dtype=output.dtype)
            else:
                patched[rows] = edited.to(device=output.device, dtype=output.dtype)
            return patched
        handle = site_module(model, site).register_forward_hook(hook)
    try:
        gpu_batch = {k: v.to("mps") for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**gpu_batch, do_sample=False, use_cache=True,
                                       max_new_tokens=base.MAX_NEW_TOKENS, eos_token_id=2,
                                       pad_token_id=int(model.generation_config.pad_token_id))
        if site is not None and len(hits) != 1:
            raise ValueError(f"Expected one {site} prompt hook in generation, got {len(hits)}")
        ids = generated[0, batch["input_ids"].shape[1]:].tolist()
        raw = processor.batch_decode([ids], skip_special_tokens=True)[0]
        with_special = processor.batch_decode([ids], skip_special_tokens=False)[0]
        label = classify(ids, raw, set(processor.tokenizer.all_special_ids), 2, recipient, opposite)
        scores = {}
        for word in (recipient, opposite):
            scores[word] = base.score_sequence(model, batch, canonical[word])["complete_reply_logprob"]
            if site is not None and len(hits) != 1 + len(scores):
                raise ValueError(f"Expected one additional {site} hook in score forward for {word}; got {len(hits)}")
        torch.mps.synchronize()
    finally:
        if handle is not None:
            handle.remove()
    if site is not None:
        if len(hits) != 3:
            raise ValueError(f"Unexpected {site} hook calls/shapes: {hits}")
    return {"generated_token_ids": ids, "raw_decoded": raw,
            "decoded_with_special_tokens": with_special, "classification": label,
            "terminal_eos": bool(ids and ids[-1] == 2),
            "canonical_scores": scores, "recipient_minus_opposite": scores[recipient] - scores[opposite],
            "hook_calls_including_two_scores": len(hits), "hook_events": hits,
            "edited_rows_shape": list(edited.shape) if edited is not None else None,
            "edited_rows_dtype": str(edited.dtype) if edited is not None else None}


def project_resource(fixed_setup_seconds: float, variable_seconds: float, peak_bytes: int) -> dict:
    projected = fixed_setup_seconds + 8 * variable_seconds
    return {"first_record_variable_seconds": variable_seconds, "one_time_fixed_setup_seconds": fixed_setup_seconds,
            "projected_eight_record_seconds": projected, "time_cap_seconds": MAX_SECONDS,
            "peak_mps_driver_bytes": peak_bytes, "memory_cap_bytes": base.MEMORY_LIMIT_BYTES,
            "passes": projected <= MAX_SECONDS and peak_bytes < base.MEMORY_LIMIT_BYTES}


def gate_projection_on_workload(resource: dict, generations: int, captures: int,
                                invalid: int) -> dict:
    resource = dict(resource)
    resource.update({"first_record_planned_generations": 100,
                     "first_record_planned_teacher_forced_forwards": 200,
                     "first_record_planned_captures": 8,
                     "first_record_actual_generations": generations,
                     "first_record_actual_teacher_forced_forwards": 2 * generations,
                     "first_record_actual_captures": captures,
                     "first_record_invalid_conditions": invalid})
    complete = generations == 100 and captures == 8 and invalid == 0
    resource["full_workload_observed"] = complete
    if not complete:
        resource["projected_eight_record_seconds"] = None
        resource["projection_status"] = "unavailable_incomplete_first_record_workload"
        resource["passes"] = False
    else:
        resource["projection_status"] = "full_first_record_workload_measured"
    return resource


def main() -> None:
    run_start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="CPU-only provenance/processor validation")
    mode.add_argument("--run", action="store_true", help="explicit bounded MPS first-record benchmark")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validate_only and args.output_dir:
        parser.error("--output-dir requires --run")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    record, row, masks, pilot = verify_frozen()
    index = json.loads(INDEX.read_text())
    snapshot = base.validate_snapshot(index)
    sources = base.validate_transformers_source()
    import torch
    import transformers
    from transformers import AutoProcessor
    if torch.__version__ != "2.15.0" or transformers.__version__ != "5.20.2":
        raise ValueError("Pinned runtime versions changed")
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    batches = make_batches(processor, record)
    frontend = frontend_check(processor, batches, record)
    words = record["words"]
    canonical = {w: base.canonical_ids(row["clips"][w], w, 2) for w in words}
    for w in words:
        if processor.tokenizer.encode(w.capitalize(), add_special_tokens=False) + [2] != canonical[w]:
            raise ValueError("Canonical sequence changed")
        saved = next(r for r in pilot["records"] if r["pair_id"] == record["pair_id"] and
                     r["engine"] == record["engine"] and r["phrasing_id"] == record["phrasing_id"] and
                     r["expected_word"] == w)
        if saved["prompt_input_ids"] != batches[w]["input_ids"][0].tolist():
            raise ValueError("Processor prompt token IDs differ from v2 natural pilot")
    blank_batches = {}
    for w in words:
        blank_batches[w] = {k: v.clone() for k, v in batches[w].items()}
        blank_batches[w]["input_features"] = blank_features(batches[w]["input_features"], record)
    features = feature_conditions(batches, record)
    preflight_seconds = time.monotonic() - run_start
    meta = {"status": "first_record_fit_only", "script_sha256": base.sha(Path(__file__)),
            "design_sha256": base.sha(DESIGN), "masks_sha256": EXPECTED["masks"],
            "selection_sha256": EXPECTED["selection"], "index_sha256": EXPECTED["index"],
            "pilot_sha256": EXPECTED["pilot"], "v2_gate_sha256": EXPECTED["gate"],
            "snapshot": snapshot, "installed_sources": sources,
            "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                         "torch": torch.__version__, "transformers": transformers.__version__},
            "record": {k: record[k] for k in ("pair_id", "engine", "phrasing_id", "words")},
            "prompt": base.PROMPT, "generation": {"do_sample": False, "max_new_tokens": 32, "eos": 2},
            "planned": {"generations": 100, "captures": 8, "teacher_forced_forwards": 200},
            "feature_hashes": {w: {"native": tensor_sha(batches[w]["input_features"]),
                                   "blank": tensor_sha(blank_batches[w]["input_features"])} for w in words},
            "feature_matrix_hashes": {name: tensor_sha(t) for name, _, t in features},
            "independent_frontend_check": frontend,
            "model_weights_loaded_during_preflight": False}
    if args.validate_only:
        print(json.dumps({"status": "validated_cpu_only", "meta": meta}, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    from transformers import VoxtralForConditionalGeneration
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("MPS recommendation missing")
    torch.mps.set_per_process_memory_fraction(min(1.0, base.MEMORY_LIMIT_BYTES / recommended))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or RUNS / stamp
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.jsonl"
    progress.touch(exist_ok=False)
    meta["model_load_requested"] = True
    base.exclusive_json(output / "meta.json", meta)
    count = 0
    capture_count = 0
    generation_count = 0
    invalid_count = 0
    try:
        with base.MemorySampler(torch) as memory:
            t0 = time.monotonic()
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(base.SNAPSHOT), local_files_only=True, dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
            load_seconds = time.monotonic() - t0
            if torch.mps.driver_allocated_memory() >= base.MEMORY_LIMIT_BYTES:
                raise MemoryError("MPS cap reached after model load")
            variable_start = time.monotonic()
            source = {}
            for site in SITES:
                for w in words:
                    for kind, batch in (("clean", batches[w]), ("blank", blank_batches[w])):
                        state, capture_meta = capture_state(model, batch, site)
                        source[(site, w, kind)] = state
                        base.append_line(progress, {"type": "capture", "site": site, "word": w,
                                                    "kind": kind, "shape": list(state.shape),
                                                    "dtype": str(state.dtype), "sha256": tensor_sha(state),
                                                    **capture_meta})
                        count += 1
                        capture_count += 1
                        if time.monotonic() - run_start >= MAX_SECONDS:
                            raise TimeoutError("45-minute stop reached during source captures")
                        if torch.mps.driver_allocated_memory() >= base.MEMORY_LIMIT_BYTES or memory.driver_max >= base.MEMORY_LIMIT_BYTES:
                            raise MemoryError("30 GiB MPS cap reached during source captures")
            baseline = {}

            def record_answer(kind: str, recipient: str, opposite: str, batch, site=None,
                              rows=None, edited=None, info=None):
                nonlocal count, generation_count
                if time.monotonic() - run_start >= MAX_SECONDS:
                    raise TimeoutError("45-minute first-record stop reached")
                t = time.monotonic()
                result = answer_and_scores(model, processor, batch, recipient, opposite,
                                           canonical, site, rows, edited)
                result.update({"type": "generation", "kind": kind, "site": site,
                               "recipient": recipient, "opposite": opposite,
                               "edit": info, "seconds_including_scores": time.monotonic() - t,
                               "mps_driver_bytes_after": int(torch.mps.driver_allocated_memory())})
                base.append_line(progress, result)
                count += 1
                generation_count += 1
                if torch.mps.driver_allocated_memory() >= base.MEMORY_LIMIT_BYTES or memory.driver_max >= base.MEMORY_LIMIT_BYTES:
                    raise MemoryError("30 GiB MPS cap reached")
                return result

            for w in words:
                other = next(x for x in words if x != w)
                saved = next(r for r in pilot["records"] if r["pair_id"] == record["pair_id"] and
                             r["engine"] == record["engine"] and r["phrasing_id"] == record["phrasing_id"] and
                             r["expected_word"] == w)
                baseline[(w, "clean")] = saved["generated_token_ids"]
                blank = record_answer("blank_baseline", w, other, blank_batches[w])
                baseline[(w, "blank")] = blank["generated_token_ids"]
            for name, expected, tensor in features:
                other = next(x for x in words if x != expected)
                batch = {k: v.clone() for k, v in batches[expected].items()}
                batch["input_features"] = tensor
                result = record_answer(f"feature:{name}", expected, other, batch)
                if name == f"native_{expected}" and result["generated_token_ids"] != baseline[(expected, "clean")]:
                    raise ValueError("Native first-record replay differs from v2 pilot")
            for site in SITES:
                smap = record["site_maps"][site]
                for w in words:
                    other = next(x for x in words if x != w)
                    for kind, batch in (("clean", batches[w]), ("blank", blank_batches[w])):
                        passive = record_answer(f"passive:{kind}", w, other, batch, site)
                        if passive["generated_token_ids"] != baseline[(w, kind)]:
                            raise ValueError(f"Passive {site}/{w}/{kind} parity failed")
                    for arm in ARMS:
                        source_kind = "blank" if arm == "restore" else "clean"
                        donor_kind = "clean" if arm == "restore" else "blank"
                        bstate = source[(site, w, source_kind)]
                        dstate = source[(site, w, donor_kind)]
                        batch = blank_batches[w] if arm == "restore" else batches[w]
                        case_id = f"{record['pair_id']}|{record['engine']}|fit_chosen|{w}"
                        for name in PATCHES:
                            rows = row_indices(smap, name)
                            donor = bstate if name == "self" else dstate
                            edited, info = prepare_edit(bstate, donor, rows, smap["whole_word_rows"],
                                                        name, case_id, site, arm)
                            if not info["valid"]:
                                base.append_line(progress, {"type": "invalid_condition", "kind": name,
                                                            "site": site, "recipient": w, "arm": arm,
                                                            "edit": info})
                                count += 1
                                invalid_count += 1
                                continue
                            result = record_answer(f"patch:{arm}:{name}", w, other, batch, site,
                                                   rows, edited, info)
                            if name == "self" and result["generated_token_ids"] != baseline[(w, source_kind)]:
                                raise ValueError(f"Self {site}/{w}/{arm} parity failed")
                        if arm == "restore":
                            rows = smap["whole_word_rows"]
                            opposite_state = source[(site, other, "clean")]
                            info = {"mode": "opposite_word", "rows": rows,
                                    "donor_sha256": tensor_sha(opposite_state)}
                            record_answer("opposite_word_control", w, other, batch, site, rows,
                                          opposite_state[rows], info)
            variable_seconds = time.monotonic() - variable_start
            resource = project_resource(preflight_seconds + load_seconds, variable_seconds,
                                        max(memory.driver_max, int(torch.mps.driver_allocated_memory())))
            resource.update({"preflight_seconds": preflight_seconds, "model_load_seconds": load_seconds})
            resource = gate_projection_on_workload(resource, generation_count, capture_count, invalid_count)
            if count != 108:
                raise ValueError(f"First-record planned condition count incomplete: {count} != 108")
        final = {"status": "first_record_complete", "meta_sha256": base.sha(output / "meta.json"),
                 "progress_sha256": base.sha(progress), "progress_records": count,
                 "capture_records": capture_count, "generation_records": generation_count,
                 "teacher_forced_forward_records": 2 * generation_count,
                 "invalid_condition_records": invalid_count,
                 "resource_projection": resource,
                 "interpretation": "FIT first-record feasibility only; no 16-direction causal threshold tested"}
        base.exclusive_json(output / "result.json", final)
        print(json.dumps(final, indent=2))
    except Exception as exc:
        base.exclusive_json(output / "error.json", {"status": "failed", "type": type(exc).__name__,
                                                        "error": str(exc), "progress_records": count,
                                                        "capture_records": capture_count,
                                                        "generation_records": generation_count,
                                                        "invalid_condition_records": invalid_count,
                                                        "progress_sha256": base.sha(progress)})
        raise


if __name__ == "__main__":
    main()
