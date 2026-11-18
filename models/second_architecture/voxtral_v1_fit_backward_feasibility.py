"""Bounded old-FIT Voxtral backward and BF16 edit diagnostic; no v2 corpus access.

--validate-only checks frozen CPU inputs and never loads model weights.
--run is a single, non-resumable MPS attempt on one preselected FIT direction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import voxtral_fit_v2_causal_first_record as prior
import voxtral_fit_pilot_capability_v1 as base


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOXTRAL_V1_FIT_BACKWARD_FEASIBILITY_PROTOCOL.md"
RUNS = HERE / "runs/voxtral_v1_fit_backward_feasibility"
PROTOCOL_SHA = "221d31776234ec99d07bb15ad74cae7431c9dde75de83b997a1b312c847dfc49"
PRIOR_SHA = "64d799bbb40b58b128b049877d18bde7774c2dc16717af97e692aa4531e907b5"
FRONTEND_SHA = "026445da1ac2c4c2f4b8f45ab2f8d7a1f4679924774714951b16d5cd29de0788"
RECIPIENT = "beaver"
DONOR = "monkey"
BIN = "word_0"
ALPHA = 0.025
TARGET_COORDS = (0, 1, 2, 3)
MAX_SECONDS = 15 * 60


def preflight() -> dict:
    if (base.sha(PROTOCOL) != PROTOCOL_SHA or base.sha(Path(prior.__file__)) != PRIOR_SHA or
            base.sha(HERE / "voxtral_preprocessing_decomposition.py") != FRONTEND_SHA):
        raise ValueError("Pinned pilot protocol or audited FIT helpers changed")
    record, row, _, pilot = prior.verify_frozen()
    if record["words"] != [RECIPIENT, DONOR]:
        raise ValueError("Preselected FIT word orientation changed")
    index = json.loads(prior.INDEX.read_text())
    snapshot = base.validate_snapshot(index)
    sources = base.validate_transformers_source()
    import torch
    import transformers
    from transformers import AutoProcessor

    if torch.__version__ != "2.15.0" or transformers.__version__ != "5.20.2":
        raise ValueError("Pinned PyTorch or Transformers version changed")
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    batches = prior.make_batches(processor, record)
    frontend = prior.frontend_check(processor, batches, record)
    saved = next(r for r in pilot["records"] if r["pair_id"] == "animal_08" and
                 r["engine"] == "piper" and r["phrasing_id"] == "fit_chosen" and
                 r["expected_word"] == RECIPIENT)
    if not saved["classification"]["exact_expected_word_and_eos"]:
        raise ValueError("Preselected saved natural answer was not exact")
    if saved["prompt_input_ids"] != batches[RECIPIENT]["input_ids"][0].tolist():
        raise ValueError("Recipient prompt differs from audited FIT pilot")
    for site in prior.SITES:
        rows = record["site_maps"][site]["bins"][BIN]
        if not rows or min(rows) < 0 or max(rows) >= prior.expected_hook_shape(site)[-2]:
            raise ValueError(f"Frozen {site} word_0 mask invalid")
    return {"record": record, "row": row, "batches": batches, "processor": processor,
            "saved": saved, "snapshot": snapshot, "sources": sources, "frontend": frontend,
            "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                         "torch": torch.__version__, "transformers": transformers.__version__}}


def uniform_edit(recipient, donor, rows: list[int], alpha: float) -> dict:
    """Compute one intended uniform edit and its actual BF16 rowwise rounding."""
    import torch

    if recipient.dtype != torch.bfloat16 or donor.dtype != torch.bfloat16:
        raise ValueError("Source states must be BF16")
    if recipient.shape != donor.shape or not rows:
        raise ValueError("Source shape or row mask invalid")
    original = recipient[rows].float()
    donor_rows = donor[rows].float()
    delta = donor_rows.mean(dim=0) - original.mean(dim=0)
    intended = alpha * delta
    edited = (original + intended).to(torch.bfloat16)
    effective = edited.float() - original
    mean = effective.mean(dim=0)
    intended_norm = float(torch.linalg.vector_norm(intended))
    realized_norm = float(torch.linalg.vector_norm(effective))
    mean_norm = float(torch.linalg.vector_norm(mean))
    mean_fidelity = float(torch.linalg.vector_norm(mean - intended)) / max(intended_norm, 1e-6)
    row_nonuniformity = float(torch.linalg.vector_norm(effective - mean)) / math.sqrt(len(rows)) / max(mean_norm, 1e-6)
    if not math.isfinite(mean_fidelity) or not math.isfinite(row_nonuniformity):
        raise ValueError("Nonfinite BF16 edit diagnostic")
    return {"edited": edited, "effective": effective, "mean": mean, "intended": intended,
            "delta": delta, "intended_norm": intended_norm, "realized_norm": realized_norm,
            "mean_fidelity_ratio": mean_fidelity, "row_nonuniformity_ratio": row_nonuniformity}


def contract_rows(grad_rows, edit: dict) -> dict[str, float]:
    """All inputs are selected FP32 rows; sum gradients for a uniform edit."""
    import torch

    g = grad_rows.float()
    e = edit["effective"].float()
    if g.shape != e.shape:
        raise ValueError("Gradient/edit row shapes differ")
    return {"uniform_mean": float(torch.sum(g.sum(dim=0) * edit["mean"].float())),
            "rowwise_oracle": float(torch.sum(g * e)),
            "ideal_uniform": float(torch.sum(g.sum(dim=0) * edit["intended"].float()))}


def vector_error(pred: list[float], measured: list[float]) -> dict:
    import torch

    p = torch.tensor(pred, dtype=torch.float64)
    m = torch.tensor(measured, dtype=torch.float64)
    if p.shape != (4,) or m.shape != (4,) or not torch.isfinite(p).all() or not torch.isfinite(m).all():
        raise ValueError("Four-coordinate error input invalid")
    absolute = float(torch.linalg.vector_norm(p - m))
    measured_norm = float(torch.linalg.vector_norm(m))
    return {"absolute_l2": absolute, "relative_l2": absolute / max(measured_norm, 1e-6),
            "measured_l2": measured_norm}


def check_resource(torch, memory, start: float) -> dict:
    torch.mps.synchronize()
    driver = int(torch.mps.driver_allocated_memory())
    current = int(torch.mps.current_allocated_memory())
    peak_driver = max(driver, memory.driver_max)
    if peak_driver >= base.MEMORY_LIMIT_BYTES:
        raise MemoryError("30 GiB MPS driver cap reached")
    elapsed = time.monotonic() - start
    if elapsed >= MAX_SECONDS:
        raise TimeoutError("15-minute FIT backward pilot cap reached")
    return {"elapsed_seconds": elapsed, "driver_bytes": driver,
            "current_bytes": current, "peak_driver_bytes": peak_driver}


def capture_prompt(model, batch, *, grad: bool, patch_site: str | None = None,
                   rows: list[int] | None = None, edited=None) -> dict:
    """One prompt forward with three exact hooks; preserve encoder→projector graph."""
    import torch

    hits: dict[str, list] = {"encoder31": [], "projector": [], "lm29": []}
    prompt_len = int(batch["input_ids"].shape[1])
    if patch_site is not None and (patch_site not in prior.SITES or not rows or edited is None):
        raise ValueError("Patch site/rows/edited tensor incomplete")

    def audio_hook(site: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != prior.expected_hook_shape(site):
                raise ValueError(f"Unexpected {site} forward output shape")
            if output.dtype != torch.bfloat16:
                raise ValueError(f"Unexpected {site} forward output dtype")
            value = output.detach().requires_grad_(True) if grad and site == "encoder31" else output
            hits[site].append(value)
            if patch_site == site:
                if tuple(edited.shape) != (len(rows), value.shape[-1]) or edited.dtype != torch.bfloat16:
                    raise ValueError("Patch row shape/dtype changed")
                patched = value.clone()
                if site == "encoder31":
                    patched[0, rows] = edited.to(value.device)
                else:
                    patched[rows] = edited.to(value.device)
                return patched
            return value
        return hook

    def lm_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, prompt_len, 3072):
            raise ValueError("Unexpected LM29 prompt-final output shape")
        if output.dtype != torch.bfloat16:
            raise ValueError("Unexpected LM29 output dtype")
        hits["lm29"].append(output)

    handles = [
        model.model.audio_tower.layers[31].register_forward_hook(audio_hook("encoder31")),
        model.model.multi_modal_projector.register_forward_hook(audio_hook("projector")),
        model.model.language_model.layers[29].register_forward_hook(lm_hook),
    ]
    try:
        gpu_batch = {k: v.to("mps") for k, v in batch.items()}
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            model(**gpu_batch, use_cache=False, logits_to_keep=1)
        torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if any(len(v) != 1 for v in hits.values()):
        raise ValueError(f"Expected one prompt-forward hit per site: { {k: len(v) for k,v in hits.items()} }")
    encoder = hits["encoder31"][0]
    projector = hits["projector"][0]
    terminal = hits["lm29"][0][0, -1]
    if grad and (not encoder.requires_grad or not projector.requires_grad or not terminal.requires_grad):
        raise ValueError("Shared backward graph disconnected")
    if any(not torch.isfinite(v).all() for v in (encoder, projector, terminal)):
        raise ValueError("Nonfinite prompt hook state")
    return {"encoder31": encoder, "projector": projector, "terminal": terminal,
            "hook_calls": {k: len(v) for k, v in hits.items()}, "prompt_len": prompt_len,
            "hook_events": {k: {"shape": list(v[0].shape), "dtype": str(v[0].dtype)}
                            for k, v in hits.items()}}


def generate_suffix(model, batch, *, passive: bool) -> dict:
    """Natural generation plus optional unmodified triple-hook parity replay."""
    import torch

    hits = {"encoder31": [], "projector": [], "lm29": []}
    handles = []
    if passive:
        def hook(site):
            def record(_module, _inputs, output):
                if not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16:
                    raise ValueError(f"Invalid passive {site} output")
                hits[site].append(list(output.shape))
                return output
            return record
        handles = [model.model.audio_tower.layers[31].register_forward_hook(hook("encoder31")),
                   model.model.multi_modal_projector.register_forward_hook(hook("projector")),
                   model.model.language_model.layers[29].register_forward_hook(hook("lm29"))]
    try:
        with torch.inference_mode():
            generated = model.generate(**{k: v.to("mps") for k, v in batch.items()},
                                       do_sample=False, use_cache=True,
                                       max_new_tokens=base.MAX_NEW_TOKENS, eos_token_id=2,
                                       pad_token_id=int(model.generation_config.pad_token_id))
        torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if passive:
        if len(hits["encoder31"]) != 1 or hits["encoder31"][0] != [1, 1500, 1280]:
            raise ValueError("Passive encoder hook shape/count changed")
        if len(hits["projector"]) != 1 or hits["projector"][0] != [375, 3072]:
            raise ValueError("Passive projector hook shape/count changed")
        if not hits["lm29"] or hits["lm29"][0] != [1, int(batch["input_ids"].shape[1]), 3072]:
            raise ValueError("Passive LM29 prompt hook changed")
    suffix = generated[0, int(batch["input_ids"].shape[1]):].tolist()
    return {"suffix_ids": suffix, "hook_shapes": hits}


def run(prepared: dict, output: Path) -> None:
    import torch
    from transformers import VoxtralForConditionalGeneration

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    recommended = int(torch.mps.recommended_max_memory())
    if recommended <= 0:
        raise RuntimeError("MPS recommended limit unavailable")
    fraction = min(1.0, base.MEMORY_LIMIT_BYTES / recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    meta = {"purpose": "old_v1_fit_backward_feasibility_only", "script_sha256": base.sha(Path(__file__)),
            "protocol_sha256": PROTOCOL_SHA, "first_runner_sha256": PRIOR_SHA,
            "first_v2_fit_pilot_sha256": prior.EXPECTED["pilot"],
            "frontend_helper_sha256": FRONTEND_SHA,
            "model_snapshot": prepared["snapshot"], "installed_sources": prepared["sources"],
            "versions": prepared["versions"], "record_key": "animal_08|piper|fit_chosen",
            "recipient": RECIPIENT, "donor": DONOR, "bin": BIN, "alpha": ALPHA,
            "target_coordinates": list(TARGET_COORDS),
            "wav_hashes": {w: next(c["wav_sha256"] for c in prepared["record"]["clips"] if c["word"] == w)
                           for w in (RECIPIENT, DONOR)},
            "masks_sha256": prior.EXPECTED["masks"],
            "source_rows": {s: prepared["record"]["site_maps"][s]["bins"][BIN] for s in prior.SITES},
            "saved_natural_suffix_ids": prepared["saved"]["generated_token_ids"],
            "fresh_v2_corpus_opened": False, "cap_seconds": MAX_SECONDS,
            "cap_mps_driver_bytes": base.MEMORY_LIMIT_BYTES,
            "mps_recommended_max_bytes": recommended, "mps_fraction_limit": fraction}
    base.exclusive_json(output / "meta.json", meta)
    progress = output / "progress.jsonl"
    progress.touch(exist_ok=False)

    def receipt(phase: str, memory) -> None:
        base.append_line(progress, {"phase": phase, **check_resource(torch, memory, started)})

    try:
        with base.MemorySampler(torch) as memory:
            t0 = time.monotonic()
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(base.SNAPSHOT), local_files_only=True, dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
            model.requires_grad_(False)
            model_load_seconds = time.monotonic() - t0
            receipt("model_loaded", memory)
            batch = prepared["batches"][RECIPIENT]
            natural = generate_suffix(model, batch, passive=False)
            receipt("natural_generation", memory)
            passive = generate_suffix(model, batch, passive=True)
            receipt("passive_generation", memory)
            if natural["suffix_ids"] != prepared["saved"]["generated_token_ids"] or passive["suffix_ids"] != natural["suffix_ids"]:
                raise ValueError("Natural or passive suffix parity failed")
            no_grad = capture_prompt(model, batch, grad=False)
            no_grad_z = no_grad["terminal"].detach().float().cpu()
            no_grad_hooks = no_grad["hook_events"]
            receipt("no_grad_prompt", memory)
            donor = capture_prompt(model, prepared["batches"][DONOR], grad=False)
            donor_states = {s: donor[s].detach().cpu()[0 if s == "encoder31" else slice(None)].clone()
                            for s in prior.SITES}
            del donor
            receipt("donor_no_grad_prompt", memory)
            clean = capture_prompt(model, batch, grad=True)
            z_clean = clean["terminal"].detach().float().cpu()
            clean_hooks = clean["hook_events"]
            rec_states = {s: clean[s].detach().cpu()[0 if s == "encoder31" else slice(None)].clone()
                          for s in prior.SITES}
            receipt("recipient_shared_graph", memory)
            edits = {s: uniform_edit(rec_states[s], donor_states[s], meta["source_rows"][s], ALPHA)
                     for s in prior.SITES}
            zero_edits = {s: uniform_edit(rec_states[s], donor_states[s], meta["source_rows"][s], 0.0)
                          for s in prior.SITES}
            if any(float(torch.linalg.vector_norm(zero_edits[s]["effective"])) != 0 for s in prior.SITES):
                raise ValueError("Zero BF16 edit was not exact")
            grads = {s: [] for s in prior.SITES}
            pred = {s: {name: [] for name in ("uniform_mean", "rowwise_oracle", "ideal_uniform")}
                    for s in prior.SITES}
            vjp_seconds = []
            for i, coord in enumerate(TARGET_COORDS):
                receipt(f"before_vjp_{coord}", memory)
                t1 = time.monotonic()
                ge, gp = torch.autograd.grad(clean["terminal"].float()[coord],
                                              (clean["encoder31"], clean["projector"]),
                                              retain_graph=i < len(TARGET_COORDS) - 1,
                                              allow_unused=False)
                torch.mps.synchronize()
                vjp_seconds.append(time.monotonic() - t1)
                for site, g in (("encoder31", ge[0]), ("projector", gp)):
                    if not torch.isfinite(g).all() or float(torch.linalg.vector_norm(g)) <= 0:
                        raise ValueError(f"Disconnected/nonfinite/zero {site} gradient at {coord}")
                    selected = g.detach().float().cpu()[meta["source_rows"][site]].contiguous()
                    grads[site].append(selected)
                    for name, value in contract_rows(selected, edits[site]).items():
                        pred[site][name].append(value)
                receipt(f"after_vjp_{coord}", memory)
                del ge, gp
            del clean
            measured = {}
            for site in prior.SITES:
                rows = meta["source_rows"][site]
                zero_capture = capture_prompt(model, batch, grad=True, patch_site=site,
                                              rows=rows, edited=zero_edits[site]["edited"])
                z0 = zero_capture["terminal"].detach().float().cpu()
                zero_hooks = zero_capture["hook_events"]
                del zero_capture
                receipt(f"zero_patch_{site}", memory)
                alpha_capture = capture_prompt(model, batch, grad=True, patch_site=site,
                                               rows=rows, edited=edits[site]["edited"])
                za = alpha_capture["terminal"].detach().float().cpu()
                alpha_hooks = alpha_capture["hook_events"]
                del alpha_capture
                receipt(f"alpha_patch_{site}", memory)
                delta = (za - z0)[list(TARGET_COORDS)].tolist()
                measured[site] = {"zero_hook_events": zero_hooks, "alpha_hook_events": alpha_hooks,
                                  "delta_four": delta, "zero_minus_clean_four":
                                  (z0-z_clean)[list(TARGET_COORDS)].tolist(),
                                  "uniform_error": vector_error(pred[site]["uniform_mean"], delta),
                                  "rowwise_error": vector_error(pred[site]["rowwise_oracle"], delta),
                                  "ideal_error": vector_error(pred[site]["ideal_uniform"], delta)}
            details = {"source_states": rec_states, "donor_states": donor_states,
                       "selected_gradients": {s: torch.stack(grads[s]) for s in prior.SITES},
                       "effective_edits": {s: edits[s]["effective"] for s in prior.SITES},
                       "uniform_means": {s: edits[s]["mean"] for s in prior.SITES},
                       "intended_edits": {s: edits[s]["intended"] for s in prior.SITES},
                       "clean_terminal": z_clean, "no_grad_terminal": no_grad_z}
            with (output / "details.pt").open("xb") as file:
                torch.save(details, file)
                file.flush()
                os.fsync(file.fileno())
            receipt("details_saved", memory)
            result = {"status": "completed_numerical_feasibility_only",
                      "meta_sha256": base.sha(output / "meta.json"),
                      "progress_sha256": base.sha(progress),
                      "details_sha256": base.sha(output / "details.pt"),
                      "model_load_seconds": model_load_seconds,
                      "vjp_seconds": vjp_seconds,
                      "natural_suffix_ids": natural["suffix_ids"],
                      "passive_suffix_ids": passive["suffix_ids"],
                      "passive_hook_shapes": passive["hook_shapes"],
                      "no_grad_hook_events": no_grad_hooks,
                      "shared_graph_hook_events": clean_hooks,
                      "shared_graph_vjp_connected_both_sources": True,
                      "source_shapes": {s: list(rec_states[s].shape) for s in prior.SITES},
                      "source_hashes": {s: prior.tensor_sha(rec_states[s]) for s in prior.SITES},
                      "donor_hashes": {s: prior.tensor_sha(donor_states[s]) for s in prior.SITES},
                      "terminal_shape": list(z_clean.shape),
                      "no_grad_minus_grad_four": (no_grad_z-z_clean)[list(TARGET_COORDS)].tolist(),
                      "edits": {s: {"intended_norm": edits[s]["intended_norm"],
                                    "realized_norm": edits[s]["realized_norm"],
                                    "mean_fidelity_ratio": edits[s]["mean_fidelity_ratio"],
                                    "row_nonuniformity_ratio": edits[s]["row_nonuniformity_ratio"],
                                    "mean_fidelity_over_10pct": edits[s]["mean_fidelity_ratio"] > 0.1,
                                    "row_nonuniformity_over_10pct": edits[s]["row_nonuniformity_ratio"] > 0.1,
                                    "effective_sha256": prior.tensor_sha(edits[s]["effective"])}
                                for s in prior.SITES},
                      "predictions": pred, "measured": measured,
                      "peak_mps_driver_bytes": max(memory.driver_max, int(torch.mps.driver_allocated_memory())),
                      "peak_mps_current_bytes": memory.current_max,
                      "elapsed_seconds": time.monotonic() - started,
                      "scope": "one old FIT pair/direction, one bin, four raw coordinates; no v2 or efficacy"}
            if result["elapsed_seconds"] >= MAX_SECONDS or result["peak_mps_driver_bytes"] >= base.MEMORY_LIMIT_BYTES:
                raise RuntimeError("Resource cap reached before final result")
            base.exclusive_json(output / "result.json", result)
    except Exception as exc:
        base.exclusive_json(output / "error.json", {"status": "failed", "error_type": type(exc).__name__,
                                                      "error": str(exc), "elapsed_seconds": time.monotonic()-started,
                                                      "progress_sha256": base.sha(progress)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validate_only and args.output_dir is not None:
        parser.error("--output-dir requires --run")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    prepared = preflight()
    if args.validate_only:
        print(json.dumps({"status": "validated_cpu_only", "record_key": "animal_08|piper|fit_chosen",
                          "recipient": RECIPIENT, "donor": DONOR, "bin": BIN,
                          "rows": {s: prepared["record"]["site_maps"][s]["bins"][BIN]
                                   for s in prior.SITES},
                          "snapshot_revision": prepared["snapshot"]["revision"]}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run(prepared, args.output_dir or RUNS / stamp)


if __name__ == "__main__":
    main()
