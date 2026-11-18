"""Fit-only scalar audio-state VJP gate for Qwen2.5-Omni-3B Thinker.

Default: original Piper red recipient, blue donor; connector and encoder 15.
No held-out manifest is read. CPU toy tests cover scoring and hook orientation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL_DIR = ROOT / "models" / "qwen_omni"
CONTROL_DIR = ROOT / "eval" / "controlled_pairs"
sys.path[:0] = [str(MODEL_DIR), str(MODEL_DIR / "time_map"), str(CONTROL_DIR)]
from adapter import (  # noqa: E402
    MODEL_ID, MODEL_REVISION, TRANSFORMERS_REVISION,
    load_pinned_thinker, prepare_audio_batch, resolve_hook_points,
)
from index import build_row_index  # noqa: E402
from patch_encoder_v1 import row_windows  # noqa: E402
from patch_connector_v1 import projected_windows  # noqa: E402


FIT_MANIFEST = ROOT / "eval" / "tts" / "outputs" / "controlled_pair_v1" / "controlled_pair_manifest.json"
FIT_MANIFEST_SHA = "318807c92971babacad93b69b7d8478e29f397219da0cb9ed6e3134649aea98b"
SYSTEM_PROMPT = "Answer the spoken question with one color word."
ANSWER_IDS = {"red": [6033, 151645], "blue": [10331, 151645]}
ANSWER_SURFACES = {"red": "Red", "blue": "Blue"}
ALPHAS = (0.0, 0.05, 0.10, 0.25, 1.0)
MAX_MPS_DRIVER_BYTES = 30 * 1024**3
SITES = {"connector": ((), "audio.to_language"), "encoder15": ((15,), "audio.encoder.15")}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_mps_memory(log_path: Path, phase: str) -> dict:
    """Persist measured MPS memory before aborting on a 30 GiB driver limit."""
    current = int(torch.mps.current_allocated_memory())
    driver = int(torch.mps.driver_allocated_memory())
    record = {
        "unix_time": time.time(), "phase": phase,
        "current_allocated_bytes": current,
        "driver_allocated_bytes": driver,
        "limit_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "over_limit": driver > MAX_MPS_DRIVER_BYTES,
    }
    with log_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"MPS memory {phase}: current={current / 1024**3:.2f} GiB "
          f"driver={driver / 1024**3:.2f} GiB", flush=True)
    if record["over_limit"]:
        raise MemoryError(
            f"MPS driver allocation passed 30 GiB at {phase}; stopping this run"
        )
    return record


def extension_with_candidate(batch: dict, candidate_ids: list[int]) -> tuple[dict, int]:
    """Append all but the final candidate token for causal teacher forcing."""
    if not candidate_ids or any(not isinstance(x, int) or x < 0 for x in candidate_ids):
        raise ValueError("Candidate must contain nonnegative token IDs, including EOS")
    prompt_len = int(batch["input_ids"].shape[1])
    if batch["input_ids"].ndim != 2 or batch["input_ids"].shape[0] != 1:
        raise ValueError("Expected one unpadded prompt")
    extended = dict(batch)
    tail = torch.tensor(candidate_ids[:-1], dtype=batch["input_ids"].dtype,
                        device=batch["input_ids"].device)[None, :]
    extended["input_ids"] = torch.cat((batch["input_ids"], tail), dim=1)
    if "attention_mask" in batch:
        ones = torch.ones_like(tail, dtype=batch["attention_mask"].dtype)
        extended["attention_mask"] = torch.cat((batch["attention_mask"], ones), dim=1)
    return extended, prompt_len


def score_from_logits(logits: torch.Tensor, prompt_len: int,
                      candidate_ids: list[int]) -> torch.Tensor:
    """Sum log P of the entire candidate, including EOS, from causal logits."""
    n = len(candidate_ids)
    selected = logits[0, prompt_len - 1:prompt_len - 1 + n].float()
    if tuple(selected.shape) != (n, logits.shape[-1]):
        raise ValueError("Model did not return a logit for every candidate token")
    target = torch.tensor(candidate_ids, dtype=torch.long, device=logits.device)
    return selected.log_softmax(dim=-1).gather(-1, target[:, None]).sum()


def scored_forward(model, batch: dict, candidate_ids: list[int]) -> torch.Tensor:
    extended, prompt_len = extension_with_candidate(batch, candidate_ids)
    outputs = model(**extended, use_cache=False, return_dict=True)
    return score_from_logits(outputs.logits, prompt_len, candidate_ids)


def _output_tensor(output: torch.Tensor | tuple) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("Audio site must return a two-dimensional tensor")
    return value


def _replace_output(output: torch.Tensor | tuple, replacement: torch.Tensor):
    return (replacement, *output[1:]) if isinstance(output, tuple) else replacement


def capture_state(model, batch: dict, module, *, use_cache: bool = False) -> torch.Tensor:
    calls = []

    def hook(_module, _inputs, output):
        calls.append(_output_tensor(output).detach().clone())

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**batch, use_cache=use_cache, return_dict=True)
    finally:
        handle.remove()
    if len(calls) != 1:
        raise ValueError(f"Expected one capture hook call, got {len(calls)}")
    return calls[0]


def score_and_gradient(model, batch: dict, candidate_ids: list[int],
                       module) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Return score, source state, and d score / d source-state elements."""
    calls = []

    def hook(_module, _inputs, output):
        state = _output_tensor(output).detach().requires_grad_(True)
        calls.append(state)
        return _replace_output(output, state)

    handle = module.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            score = scored_forward(model, batch, candidate_ids)
            if len(calls) != 1:
                raise ValueError(f"Expected one gradient hook call, got {len(calls)}")
            grad, = torch.autograd.grad(score, calls[0], allow_unused=False)
    finally:
        handle.remove()
    if not bool(torch.isfinite(grad).all()):
        raise FloatingPointError("Nonfinite score gradient")
    return float(score.detach()), calls[0].detach(), grad.detach()


def patch_tensor(base: torch.Tensor, donor: torch.Tensor, rows: list[int],
                 alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if base.shape != donor.shape or base.ndim != 2:
        raise ValueError("Donor/recipient audio-site shapes differ")
    if not rows or len(set(rows)) != len(rows) or min(rows) < 0 or max(rows) >= base.shape[0]:
        raise ValueError("Invalid audio-site rows")
    if not 0 <= alpha <= 1:
        raise ValueError("Patch fraction outside [0,1]")
    changed = base.clone()
    target = base[rows].float() + alpha * (donor[rows].float() - base[rows].float())
    changed[rows] = target.to(base.dtype)
    effective = changed[rows].float() - base[rows].float()
    return changed, effective


def score_with_patch(model, batch: dict, candidate_ids: list[int], module,
                     replacement: torch.Tensor, rows: list[int]) -> float:
    calls = []

    def hook(_module, _inputs, output):
        state = _output_tensor(output)
        if state.shape != replacement.shape:
            raise ValueError("Patch/source tensor shape mismatch")
        changed = state.clone()
        changed[rows] = replacement[rows].to(state.device, dtype=state.dtype)
        calls.append(True)
        return _replace_output(output, changed)

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            score = float(scored_forward(model, batch, candidate_ids))
    finally:
        handle.remove()
    if len(calls) != 1:
        raise ValueError(f"Expected one score patch call, got {len(calls)}")
    return score


def generated_answer(model, processor, batch: dict, *, module=None,
                     replacement=None, rows=None) -> dict:
    calls = []
    handle = None
    if module is not None:
        if replacement is None or not rows:
            raise ValueError("A generation patch needs tensor and rows")

        def hook(_module, _inputs, output):
            state = _output_tensor(output)
            if state.shape != replacement.shape:
                raise ValueError("Generation patch/source shape mismatch")
            changed = state.clone()
            changed[rows] = replacement[rows].to(state.device, dtype=state.dtype)
            calls.append(True)
            return _replace_output(output, changed)

        handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result = model.generate(
                **batch, do_sample=False, use_cache=True, max_new_tokens=16,
                eos_token_id=int(processor.tokenizer.eos_token_id),
                pad_token_id=int(processor.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        if handle is not None:
            handle.remove()
    if module is not None and len(calls) != 1:
        raise ValueError(f"Expected one generation patch call, got {len(calls)}")
    prompt_len = int(batch["input_ids"].shape[1])
    new_ids = result.sequences[0, prompt_len:].tolist()
    return {
        "token_ids": new_ids,
        "answer": processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "stopped_on_eos": bool(new_ids and new_ids[-1] == processor.tokenizer.eos_token_id),
        "patch_hook_calls": len(calls),
    }


def source_rows(row_map: dict, site: str, start: float, end: float) -> list[int]:
    if site == "connector":
        return projected_windows(row_map, start, end)["word_local"]
    if site == "encoder15":
        return row_windows(row_map, start, end)["word_local"]
    raise ValueError(f"Unknown site: {site}")


def verify_fit_files(engines: list[str]) -> dict:
    if sha(FIT_MANIFEST) != FIT_MANIFEST_SHA:
        raise ValueError("Fit manifest SHA-256 changed")
    manifest = json.loads(FIT_MANIFEST.read_text())
    if manifest["system_prompt"] != SYSTEM_PROMPT:
        raise ValueError("Fit system prompt changed")
    for engine in engines:
        if engine not in manifest["engines"]:
            raise ValueError(f"Unknown fit engine: {engine}")
        group = manifest["engines"][engine]
        if not group["prefix_and_suffix_pcm_identical"]:
            raise ValueError("Fit pair lacks identical outside-slot PCM")
        for word in ANSWER_IDS:
            info = group["outputs"][word]
            path = ROOT / info["path"]
            if sha(path) != info["file_sha256"]:
                raise ValueError(f"Fit WAV changed: {path}")
    return manifest


def verify_tokenizer(processor) -> None:
    if processor.tokenizer.eos_token_id != 151645:
        raise ValueError("Pinned EOS ID mismatch")
    for word, surface in ANSWER_SURFACES.items():
        ids = processor.tokenizer.encode(surface, add_special_tokens=False)
        if ids + [151645] != ANSWER_IDS[word]:
            raise ValueError(f"Candidate token ID mismatch for {word}: {ids}")


def build_inputs(model, processor, manifest: dict, engine: str) -> tuple[dict, dict]:
    group = manifest["engines"][engine]
    inputs = {}
    for word in ANSWER_IDS:
        info = group["outputs"][word]
        path = ROOT / info["path"]
        batch, digest = prepare_audio_batch(processor, path, system_prompt=SYSTEM_PROMPT)
        if digest != info["file_sha256"]:
            raise ValueError("Processor WAV identity mismatch")
        mel_frames = int(batch["feature_attention_mask"].sum())
        positions = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
        row_map = build_row_index(
            mel_frames=mel_frames, audio_token_positions=positions,
            n_window=model.config.audio_config.n_window,
            hop_length=processor.feature_extractor.hop_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )
        inputs[word] = {
            "batch": batch.to(model.device, dtype=model.dtype),
            "sha256": digest, "row_map": row_map,
        }
    if inputs["red"]["row_map"] != inputs["blue"]["row_map"]:
        raise ValueError("Matched fit row maps differ")
    return inputs, group


def run_case(model, processor, inputs: dict, group: dict, engine: str,
             recipient: str, donor_word: str, site: str,
             memory_log: Path) -> dict:
    audio_layers, point_name = SITES[site]
    points = resolve_hook_points(model, audio_layers=audio_layers, text_layers=())
    point = next(p for p in points if p.name == point_name)
    batch = inputs[recipient]["batch"]
    donor_batch = inputs[donor_word]["batch"]
    rows = source_rows(inputs[recipient]["row_map"], site,
                       group["word_slot_start_seconds"], group["word_slot_end_seconds"])
    if rows != source_rows(inputs[donor_word]["row_map"], site,
                           group["word_slot_start_seconds"], group["word_slot_end_seconds"]):
        raise ValueError("Donor/recipient source-row mismatch")
    donor_state = capture_state(model, donor_batch, point.module)
    donor_score, base_state_a, donor_grad = score_and_gradient(
        model, batch, ANSWER_IDS[donor_word], point.module
    )
    record_mps_memory(memory_log, f"{engine}/{recipient}_to_{donor_word}/{site}/donor_vjp")
    recipient_score, base_state_b, recipient_grad = score_and_gradient(
        model, batch, ANSWER_IDS[recipient], point.module
    )
    record_mps_memory(memory_log, f"{engine}/{recipient}_to_{donor_word}/{site}/recipient_vjp")
    if not torch.equal(base_state_a, base_state_b):
        raise ValueError("Candidate continuation changed an audio-site anchor state")
    base = base_state_a
    if base.shape != donor_state.shape:
        raise ValueError("Donor/source site shape mismatch")
    cached_base = capture_state(model, batch, point.module, use_cache=True)
    record_mps_memory(memory_log, f"{engine}/{recipient}_to_{donor_word}/{site}/cached_prefill")
    cache_prefill_max_abs_difference = float(
        (cached_base.float() - base.float()).abs().max()
    )
    if cache_prefill_max_abs_difference > 0.03125:
        raise ValueError(
            f"Cached/uncached audio prefill differs at {site}: "
            f"max abs {cache_prefill_max_abs_difference}"
        )
    grad = donor_grad.float() - recipient_grad.float()
    if not torch.isfinite(grad).all() or float(grad[rows].abs().max()) == 0:
        raise ValueError("Audio-state score gradient is invalid or disconnected")
    clean = generated_answer(model, processor, batch)
    clean_eligible = (
        clean["answer"].casefold() == recipient
        and clean["stopped_on_eos"]
    )
    if not clean_eligible:
        raise ValueError(
            f"Fit positive control is not clean-eligible: {engine} {recipient} "
            f"returned {clean!r}"
        )
    # Compare finite differences against the same no-grad execution path used
    # for every patched score. MPS BF16 grad-enabled kernels can differ slightly
    # from inference kernels even when their source activations are identical.
    with torch.no_grad():
        clean_donor_score = float(scored_forward(model, batch, ANSWER_IDS[donor_word]))
        clean_recipient_score = float(scored_forward(model, batch, ANSWER_IDS[recipient]))
    baseline = clean_donor_score - clean_recipient_score
    vjp_baseline = donor_score - recipient_score
    outside = torch.ones(base.shape[0], dtype=torch.bool, device=base.device)
    outside[rows] = False
    outside_slot_difference = donor_state[outside].float() - base[outside].float()
    map_key = "projected_rows" if site == "connector" else "encoder_rows"
    chunk_ids = sorted({
        inputs[recipient]["row_map"][map_key][row]["chunk_index"] for row in rows
    })
    conditions = []
    for alpha in ALPHAS:
        replacement, effective = patch_tensor(base, donor_state, rows, alpha)
        predicted = float((grad[rows] * effective).sum())
        actual_donor = score_with_patch(
            model, batch, ANSWER_IDS[donor_word], point.module, replacement, rows
        )
        actual_recipient = score_with_patch(
            model, batch, ANSWER_IDS[recipient], point.module, replacement, rows
        )
        measured = actual_donor - actual_recipient - baseline
        generated = generated_answer(
            model, processor, batch, module=point.module,
            replacement=replacement, rows=rows,
        )
        record_mps_memory(memory_log, f"{engine}/{recipient}_to_{donor_word}/{site}/alpha={alpha}")
        if alpha == 0 and (float(effective.norm()) != 0 or abs(measured) > 1e-3 or generated["token_ids"] != clean["token_ids"]):
            raise ValueError(
                "Zero edit failed self/score invariance: "
                f"effective_norm={float(effective.norm())}, "
                f"measured_delta={measured}, "
                f"clean_ids={clean['token_ids']}, patched_ids={generated['token_ids']}, "
                f"clean={clean['answer']!r}, patched={generated['answer']!r}"
            )
        conditions.append({
            "alpha": alpha,
            "effective_bf16_delta_norm": float(effective.norm()),
            "predicted_delta_score": predicted,
            "measured_delta_score": measured,
            "score_prediction_error": measured - predicted,
            "per_row_predicted_contributions": [
                float((grad[row] * effective[i]).sum()) for i, row in enumerate(rows)
            ],
            "patched_donor_answer_logprob": actual_donor,
            "patched_recipient_answer_logprob": actual_recipient,
            "generated": generated,
        })
    return {
        "engine": engine, "recipient": recipient, "donor": donor_word,
        "site": site, "site_width": int(base.shape[-1]), "site_rows_total": int(base.shape[0]),
        "word_rows": rows,
        "word_row_attention_chunk_ids": chunk_ids,
        "source_audio_sha256": inputs[recipient]["sha256"],
        "donor_audio_sha256": inputs[donor_word]["sha256"],
        "candidate_token_ids": {
            "recipient": ANSWER_IDS[recipient], "donor": ANSWER_IDS[donor_word]
        },
        "donor_minus_source_word_delta_norm": float((donor_state[rows].float() - base[rows].float()).norm()),
        "outside_word_rows_delta_norm": float(outside_slot_difference.norm()),
        "cache_prefill_max_abs_difference": cache_prefill_max_abs_difference,
        "gradient_word_norm": float(grad[rows].norm()),
        "baseline_score": baseline,
        "vjp_baseline_score": vjp_baseline,
        "grad_vs_no_grad_baseline_difference": vjp_baseline - baseline,
        "clean_generated": clean,
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", choices=("piper", "kokoro"), default=("piper",))
    parser.add_argument("--directions", nargs="+", choices=("red_to_blue", "blue_to_red"), default=("red_to_blue",))
    parser.add_argument("--sites", nargs="+", choices=tuple(SITES), default=("connector", "encoder15"))
    parser.add_argument("--output", type=Path, default=HERE / "fit_gate_v1_results.json")
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    engines = list(dict.fromkeys(args.engines))
    manifest = verify_fit_files(engines)
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision differs from pin")
    if args.validate_inputs_only:
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        )
        verify_tokenizer(processor)
        print(json.dumps({"fit_manifest_sha256": FIT_MANIFEST_SHA, "engines": engines,
                          "audio_sha256": {e: {w: manifest["engines"][e]["outputs"][w]["file_sha256"] for w in ANSWER_IDS} for e in engines},
                          "candidate_token_ids": ANSWER_IDS,
                          "candidate_surfaces": ANSWER_SURFACES,
                          "alpha_grid": list(ALPHAS)}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite result: {args.output}")
    memory_log = args.output.with_suffix(".memory.jsonl")
    if memory_log.exists():
        raise FileExistsError(f"Refusing to overwrite memory log: {memory_log}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This Qwen gate expects MPS; CPU toy tests need no Qwen")
    memory_log.write_text("")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.bfloat16,
        device_map="mps", low_cpu_mem_usage=True,
    )
    record_mps_memory(memory_log, "after_model_load")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    verify_tokenizer(processor)
    report = {
        "status": "fit_only_running", "purpose": "scalar audio-state Jacobian prediction versus causal patch",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "split_sha256": sha(HERE / "SPLIT_V1.json"),
        "plan_sha256": sha(HERE / "PLAN.md"),
        "script_sha256": sha(Path(__file__)),
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "system_prompt": SYSTEM_PROMPT,
        "candidate_surfaces": ANSWER_SURFACES,
        "candidate_token_ids": ANSWER_IDS,
        "alpha_grid": list(ALPHAS),
        "memory_log_path": str(memory_log),
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "device": "mps", "dtype": "bfloat16", "torch_version": torch.__version__,
        "engines": engines, "directions": args.directions,
        "sites": args.sites, "results": [],
    }
    for engine in engines:
        inputs, group = build_inputs(model, processor, manifest, engine)
        for direction in dict.fromkeys(args.directions):
            recipient, donor_word = direction.split("_to_")
            for site in dict.fromkeys(args.sites):
                result = run_case(model, processor, inputs, group,
                                  engine, recipient, donor_word, site,
                                  memory_log)
                report["results"].append(result)
                print(engine, direction, site, [(x["alpha"], x["predicted_delta_score"], x["measured_delta_score"], x["generated"]["answer"]) for x in result["conditions"]], flush=True)
    report["status"] = "fit_only_complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
