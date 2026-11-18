"""Prospective old-FIT native handoff Stage A; one arithmetic arm per process.

This is source for independent review. A live arm requires a separate external
source freeze and launch gate. No Stage-B replay or fresh-v2 data is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import voxtral_fit_pilot_capability_v1 as fixture
import voxtral_fit_v2_causal_first_record as first
import voxtral_v1_fit_backward_numerical_followup_v2 as prior
import voxtral_old_fit_handoff_native_stage_a_v1 as native


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs/voxtral_old_fit_handoff_stage_a_v1"
MAX_DRIVER_BYTES = 30 * 1024**3
MAX_SECONDS = {"bf16": 900, "fp32": 1200}
CONTROL_ALPHAS = {0.025, -0.025, 0.1, -0.1}
RESULT_STATUS = "completed_old_fit_handoff_stage_a_native_only"


def directory_bytes(root: Path) -> int:
    seen = set()
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key)
                total += stat.st_size
    return total


def freeze_content_addressed(alias: Path) -> tuple[str, str]:
    """Move an exclusive sidecar into a SHA-named blob path."""
    digest = fixture.sha(alias)
    blob_dir = alias.parent / "blobs"
    blob_dir.mkdir(exist_ok=True)
    blob = blob_dir / f"{digest}.pt"
    if blob.exists():
        raise FileExistsError("Stage-A blob hash path already exists")
    alias.rename(blob)
    return digest, str(blob.relative_to(alias.parent))


class LimitedWriter:
    """Fail a large sidecar write before crossing an arm's output ceiling."""

    def __init__(self, stream, max_bytes: int):
        self.stream = stream
        self.max_bytes = max_bytes
        self.hit_cap = False

    def write(self, data):
        if self.stream.tell() + len(data) > self.max_bytes:
            self.hit_cap = True
            raise OSError("Stage-A serialized output ceiling reached")
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.stream.tell()

    def fileno(self):
        return self.stream.fileno()


def planned_counts() -> dict[str, int]:
    return {"native_patch_forwards": 72, "native_pairs": 36,
            "parity_prompt_forwards": 6, "donor_prompt_forwards": 1,
            "clean_grad_prompt_forwards": 1, "vjps": 44,
            "greedy_generations": 6, "lm0_control_zero_forwards": 24,
            "lm0_control_edited_forwards": 24,
            "lm0_control_self_forwards": 1}


def pure_prompt_forward(model, batch, dtype) -> dict:
    """No registered hooks; capture the complete post-norm prompt state."""
    with torch.no_grad():
        gpu = {key: value.to("mps") for key, value in batch.items()}
        state = model.model(**gpu, use_cache=False).last_hidden_state
        logits = model.lm_head(state[:, -1:])
    torch.mps.synchronize()
    if state.shape != (1, 396, 3072) or state.dtype != dtype or logits.shape != (1, 1, 131072):
        raise ValueError("Pure prompt state/logits shape or dtype changed")
    return {"final_full": state.detach().cpu().clone(),
            "first_answer_logits": logits[0, 0].detach().cpu().clone(),
            "hook_events": {}, "hook_mode": "none"}


def generate_source_condition(model, batch, dtype, site: str, rows: list[int],
                              direction: torch.Tensor, *, self_patch: bool) -> list[int]:
    """One greedy generation with an exact self or exact-zero source edit."""
    calls = []

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.dtype != dtype or \
                tuple(output.shape) != first.expected_hook_shape(site):
            raise ValueError("Generation source hook shape/dtype changed")
        calls.append(1)
        patched = output.clone()
        if not self_patch:
            original = output[0, rows] if site == "encoder31" else output[rows]
            zero = prior.realized_edit(original, direction.to(output.device), 0.0, dtype)
            if not torch.equal(zero["edited"], original):
                raise ValueError("Generation zero patch is not exact")
            if site == "encoder31":
                patched[0, rows] = zero["edited"]
            else:
                patched[rows] = zero["edited"]
        return patched

    module = model.model.audio_tower.layers[31] if site == "encoder31" else model.model.multi_modal_projector
    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            generated = model.generate(**{key: value.to("mps") for key, value in batch.items()},
                                       do_sample=False, use_cache=True,
                                       max_new_tokens=fixture.MAX_NEW_TOKENS, eos_token_id=2,
                                       pad_token_id=int(model.generation_config.pad_token_id))
        torch.mps.synchronize()
    finally:
        handle.remove()
    if len(calls) != 1:
        raise ValueError("Generation source hook count changed")
    return generated[0, int(batch["input_ids"].shape[1]):].tolist()


def parity_checks(model, batch, dtype, prepared: dict, receipt) -> dict:
    """Six prompt conditions and six greedy suffixes before numerical work."""
    rows = prepared["rows"]
    directions = prepared["directions"]
    receipt("before_pure_prompt")
    pure = pure_prompt_forward(model, batch, dtype)
    receipt("after_pure_prompt")
    passive = native.trace_forward(model, batch, dtype, rows, capture_final_full=True)
    receipt("after_passive_prompt")
    if not torch.equal(pure["final_full"], passive["saved"]["final_full"]) or \
            not torch.equal(pure["first_answer_logits"], passive["saved"]["first_answer_logits"]):
        raise ValueError("Pure/passive prompt state or logit parity failed")
    saved = {"pure": pure, "passive": {"final_full": passive["saved"]["final_full"],
                                      "first_answer_logits": passive["saved"]["first_answer_logits"],
                                      "hook_events": passive["saved"]["hook_events"]}}
    for site in prior.SITES:
        for mode in ("self", "zero"):
            trace = native.trace_forward(model, batch, dtype, rows, patch_site=site,
                                         direction=None if mode == "self" else directions[site],
                                         alpha=None if mode == "self" else 0.0,
                                         self_patch=mode == "self", capture_final_full=True)
            if not torch.equal(trace["saved"]["final_full"], pure["final_full"]) or \
                    not torch.equal(trace["saved"]["first_answer_logits"], pure["first_answer_logits"]):
                raise ValueError(f"Pure/{site}/{mode} complete-prompt parity failed")
            saved[f"{site}_{mode}"] = {"final_full": trace["saved"]["final_full"],
                                      "first_answer_logits": trace["saved"]["first_answer_logits"],
                                      "hook_events": trace["saved"]["hook_events"]}
            receipt(f"after_parity_{site}_{mode}")
    natural = prior.generate(model, batch, dtype, passive=False)["suffix_ids"]
    receipt("after_natural_generation")
    passive_suffix = prior.generate(model, batch, dtype, passive=True)["suffix_ids"]
    receipt("after_passive_generation")
    suffixes = {"pure": natural, "passive": passive_suffix}
    for site in prior.SITES:
        for mode in ("self", "zero"):
            suffixes[f"{site}_{mode}"] = generate_source_condition(
                model, batch, dtype, site, rows[site], directions[site], self_patch=mode == "self")
            receipt(f"after_generation_{site}_{mode}")
    if any(value != natural for value in suffixes.values()) or \
            (dtype == torch.bfloat16 and natural != prepared["saved"]["generated_token_ids"]):
        raise ValueError("Six-condition greedy suffix parity failed")
    return {"prompt_states": saved, "suffix_ids": suffixes, "passed": True}


def _raw_control_cells(observations: dict[str, list[dict]]):
    for site in prior.SITES:
        for cycle, order in enumerate(prior.CYCLES):
            for alpha in order:
                if alpha in CONTROL_ALPHAS:
                    matched = [r for r in observations[site] if r["cycle"] == cycle and r["alpha"] == alpha]
                    if len(matched) != 1:
                        raise ValueError("LM0 control lacks its exact native cell")
                    yield site, cycle, alpha, matched[0]


def lm0_controls(model, batch, dtype, observations: dict[str, list[dict]], receipt) -> dict:
    """All 24 fixed native-cell matched controls, even when norm matching fails."""
    unit = native.fixed_lm0_control_unit()
    records = []
    for site, cycle, alpha, native_record in _raw_control_cells(observations):
        baseline = native_record["zero"]["lm_terminal"]["lm0"]
        edited = native_record["edited"]["lm_terminal"]["lm0"]
        target_norm = float(torch.linalg.vector_norm((edited.float() - baseline.float()).double()))
        choice = native.choose_lm0_control_scale(baseline, unit, 1 if alpha > 0 else -1, target_norm)
        zero = native.compact_lm0_control_forward(model, batch, dtype, scalar=0.0,
                                                   unit=unit, sign=1 if alpha > 0 else -1)
        receipt(f"lm0_control_{site}_{cycle}_{alpha}_zero")
        edit = native.compact_lm0_control_forward(model, batch, dtype,
                                                   scalar=choice["scalar"], unit=unit,
                                                   sign=1 if alpha > 0 else -1)
        receipt(f"lm0_control_{site}_{cycle}_{alpha}_edited")
        if not torch.equal(zero["prepatch_lm0"], baseline) or \
                not torch.equal(edit["prepatch_lm0"], baseline) or \
                float(zero["effective_edit_lm0"].abs().sum()) != 0:
            raise ValueError("LM0 non-audio control baseline/zero drift")
        realized_norm = float(torch.linalg.vector_norm(edit["effective_edit_lm0"].double()))
        if abs(realized_norm - choice["realized_norm"]) > 1e-7:
            raise ValueError("LM0 control CPU target norm was not realized on device")
        records.append({"site": site, "cycle": cycle, "alpha": alpha,
                        "choice": choice, "zero": zero, "edited": edit,
                        "observed_realized_norm": realized_norm})
    if len(records) != 24:
        raise ValueError("LM0 non-audio control count changed")
    self_trace = native.compact_lm0_control_forward(model, batch, dtype, self_patch=True)
    receipt("lm0_control_self")
    if not torch.equal(self_trace["prepatch_lm0"], self_trace["lm_terminal"]["lm0"]):
        raise ValueError("LM0 self patch changed its own state")
    return {"unit": unit, "records": records, "self": self_trace}


def run_arm(arm: str, output: Path) -> None:
    if arm not in MAX_SECONDS:
        raise ValueError("Unknown Stage-A arithmetic arm")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.jsonl"
    progress.touch(exist_ok=False)
    fixture.append_line(progress, {"phase": "attempt_started", "arm": arm,
                                   "wall_utc": datetime.now(timezone.utc).isoformat(),
                                   "script_sha256": fixture.sha(Path(__file__)),
                                   "protocol_sha256": native.PROTOCOL_SHA256})
    max_seconds = MAX_SECONDS[arm]
    cap = native.ARCHIVE_CAP_BYTES[arm]

    def wall_check():
        elapsed = time.monotonic() - started
        if elapsed >= max_seconds:
            raise TimeoutError("Stage-A arm wall cap reached")
        return elapsed

    try:
        prepared = native.preflight_cpu()
        estimate = native.estimate_native_archive_bytes(arm)
        if estimate["estimated_tensor_payload_bytes"] > cap * .9:
            raise ValueError("Projected Stage-A tensor payload has insufficient cap headroom")
        probes = native.fixed_probes()
        with (output / "probes.pt").open("xb") as stream:
            torch.save(probes, stream)
            stream.flush()
            os.fsync(stream.fileno())
        probe_sha, probe_blob_path = freeze_content_addressed(output / "probes.pt")
        probe_bytes = (output / probe_blob_path).stat().st_size
        fixture.append_line(progress, {"phase": "preflight_and_probes_frozen", "elapsed_seconds": wall_check(),
                                       "probes_sha256": probe_sha, "probes_bytes": probe_bytes,
                                       "tensor_payload_estimate": estimate})
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        recommended = int(torch.mps.recommended_max_memory())
        if recommended <= 0:
            raise RuntimeError("MPS recommended memory unavailable")
        fraction = min(1.0, MAX_DRIVER_BYTES / recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        dtype = torch.bfloat16 if arm == "bf16" else torch.float32
        meta = {"status": "started_old_fit_handoff_stage_a_native_only", "arm": arm,
                "protocol_sha256": native.PROTOCOL_SHA256,
                "script_sha256": fixture.sha(Path(__file__)),
                "native_source_sha256": fixture.sha(Path(native.__file__)),
                "prior_runner_sha256": fixture.sha(Path(prior.__file__)),
                "probes_sha256": probe_sha, "probes_path": probe_blob_path,
                "old_fit_snapshot": prepared["snapshot"], "installed_source_hashes": prepared["sources"],
                "wav_hashes": {clip["word"]: clip["wav_sha256"] for clip in prepared["record"]["clips"]},
                "rows": prepared["rows"],
                "directions_sha256": {site: first.tensor_sha(prepared["directions"][site])
                                      for site in prior.SITES},
                "planned_counts": planned_counts(), "archive_estimate": estimate,
                "max_seconds": max_seconds, "max_driver_bytes": MAX_DRIVER_BYTES,
                "max_output_bytes": cap, "mps_fraction": fraction,
                "fresh_v2_corpus_opened": False}
        fixture.exclusive_json(output / "meta.json", meta)
        from transformers import VoxtralForConditionalGeneration

        with fixture.MemorySampler(torch) as memory:
            def receipt(phase, **extra):
                resource = prior.check_limit(torch, memory, started, max_seconds)
                if directory_bytes(output) >= cap:
                    raise OSError("Stage-A serialized output ceiling reached")
                fixture.append_line(progress, {"phase": phase, **resource, **extra})

            receipt("before_model_load")
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(fixture.SNAPSHOT), local_files_only=True, dtype=dtype,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
            model.requires_grad_(False)
            layout = prior.parameter_layout(model, arm)
            fixture.exclusive_json(output / "model_layout.json", layout)
            if not layout["valid"]:
                raise ValueError("Stage-A full-pipeline parameter layout changed: " +
                                 "; ".join(layout["errors"][:5]))
            receipt("model_loaded", named_parameter_count=layout["named_parameter_count"])
            batch = prepared["batches"]["beaver"]
            parity = parity_checks(model, batch, dtype, prepared, receipt)
            donor = native.trace_forward(model, prepared["batches"]["monkey"], dtype, prepared["rows"])
            donor_trace = {"source_hashes": {site: first.tensor_sha(donor["saved"]["source_rows"][site])
                                              for site in prior.SITES},
                           "lm_terminal": donor["saved"]["lm_terminal"],
                           "first_answer_logits": donor["saved"]["first_answer_logits"],
                           "hook_events": donor["saved"]["hook_events"]}
            receipt("donor_capture")
            clean = native.trace_forward(model, batch, dtype, prepared["rows"], grad=True,
                                         archive_full_source=True)
            clean_saved = clean["saved"]
            receipt("clean_shared_graph")
            vjps = native.source_vjps(clean, probes, prepared["rows"])
            del clean
            receipt("after_44_vjps")
            observations: dict[str, list[dict]] = {site: [] for site in prior.SITES}
            patch_count = 0
            for site in prior.SITES:
                for cycle, order in enumerate(prior.CYCLES):
                    for alpha in order:
                        pair = {}
                        for label, edit_alpha in (("zero", 0.0), ("edited", alpha)):
                            receipt(f"before_{site}_{cycle}_{label}_{alpha}")
                            trace = native.trace_forward(
                                model, batch, dtype, prepared["rows"], patch_site=site,
                                direction=prepared["directions"][site], alpha=edit_alpha,
                                archive_full_source=cycle == 0)
                            saved = trace["saved"]
                            if not torch.equal(saved["source_rows"][site], clean_saved["source_rows"][site]):
                                raise ValueError("Paired source selected rows differ from clean VJP source")
                            if saved["source_full_pre_hashes"][site] != clean_saved["source_full_pre_hashes"][site]:
                                raise ValueError("Paired full prepatch source hash differs from clean VJP source")
                            if cycle == 0 and not torch.equal(saved["source_full_pre"][site],
                                                              clean_saved["source_full_pre"][site]):
                                raise ValueError("Archived full prepatch source differs from clean VJP source")
                            if edit_alpha == 0 and float(saved["source_effective_edit"].abs().sum()) != 0:
                                raise ValueError("Paired zero edit was not exact")
                            pair[label] = saved
                            patch_count += 1
                            receipt(f"after_{site}_{cycle}_{label}_{alpha}", patch_count=patch_count)
                        observations[site].append({"cycle": cycle, "alpha": alpha,
                                                   "zero": pair["zero"], "edited": pair["edited"]})
            if patch_count != 72 or any(len(records) != 18 for records in observations.values()):
                raise ValueError("Native paired matrix incomplete")
            analysis = native.analyze_native_pairs(observations, vjps, probes,
                                                   prepared["rows"], prepared["directions"])
            controls = lm0_controls(model, batch, dtype, observations, receipt)
            details = {"probes": probes, "clean_trace": clean_saved, "donor_trace": donor_trace,
                       "source_vjps": vjps, "observations": observations,
                       "lm0_controls": controls, "parity": parity,
                       "directions": prepared["directions"], "analysis": analysis}
            actual_counts = {
                "native_patch_forwards": patch_count,
                "native_pairs": sum(len(records) for records in observations.values()),
                "parity_prompt_forwards": len(parity["prompt_states"]),
                "donor_prompt_forwards": 1,
                "clean_grad_prompt_forwards": 1,
                "vjps": sum(values["encoder31"].shape[0] for values in vjps.values()),
                "greedy_generations": len(parity["suffix_ids"]),
                "lm0_control_zero_forwards": sum("zero" in record for record in controls["records"]),
                "lm0_control_edited_forwards": sum("edited" in record for record in controls["records"]),
                "lm0_control_self_forwards": int("self" in controls),
            }
            if actual_counts != planned_counts():
                raise ValueError("Stage-A realized counts differ from frozen plan")
            # Keep at least one MiB for the result/error and final progress receipt.
            remaining = cap - directory_bytes(output) - 1_048_576
            if remaining <= 0:
                raise OSError("Stage-A serialized output ceiling reached before details")
            with (output / "details.pt").open("xb") as raw:
                limited = LimitedWriter(raw, remaining)
                torch.save(details, limited)
                limited.flush()
                os.fsync(limited.fileno())
            details_sha, details_blob_path = freeze_content_addressed(output / "details.pt")
            receipt("details_saved", details_bytes=(output / details_blob_path).stat().st_size)
            result = {"status": RESULT_STATUS, "arm": arm,
                      "script_sha256": fixture.sha(Path(__file__)),
                      "protocol_sha256": native.PROTOCOL_SHA256,
                      "native_source_sha256": fixture.sha(Path(native.__file__)),
                      "meta_sha256": fixture.sha(output / "meta.json"),
                      "model_layout_sha256": fixture.sha(output / "model_layout.json"),
                      "probes_sha256": probe_sha, "details_sha256": details_sha,
                      "probes_path": probe_blob_path, "details_path": details_blob_path,
                      "progress_sha256": fixture.sha(progress),
                      "probes_bytes": probe_bytes, "details_bytes": (output / details_blob_path).stat().st_size,
                      "bytes_before_result": directory_bytes(output),
                      "max_output_bytes": cap, "planned_counts": planned_counts(),
                      "actual_counts": actual_counts,
                      "natural_suffix_ids": parity["suffix_ids"]["pure"],
                      "parity_passed": parity["passed"],
                      "source_hashes": {site: first.tensor_sha(clean_saved["source_rows"][site])
                                        for site in prior.SITES},
                      "native_stage_analysis": analysis,
                      "peak_mps_driver_bytes": max(memory.driver_max, int(torch.mps.driver_allocated_memory())),
                      "elapsed_seconds": time.monotonic() - started,
                      "scope": "posthoc_one_old_fit_case_native_stage_only_no_security_claim"}
            if result["elapsed_seconds"] >= max_seconds or result["peak_mps_driver_bytes"] >= MAX_DRIVER_BYTES:
                raise RuntimeError("Stage-A resource cap reached before result")
            result_payload = (json.dumps(result, indent=2, ensure_ascii=False,
                                         allow_nan=False) + "\n").encode("utf-8")
            if directory_bytes(output) + len(result_payload) > cap:
                raise OSError("Stage-A serialized output ceiling reached before result")
            fixture.exclusive_json(output / "result.json", result)
    except Exception as exc:
        resource_limited = bool(locals().get("limited") and limited.hit_cap) or \
            isinstance(exc, (MemoryError, TimeoutError, OSError)) or \
            "out of memory" in str(exc).lower()
        fixture.exclusive_json(output / "error.json", {
            "status": "resource_limited_stage_a_arm" if resource_limited else "invalid_stage_a_arm",
            "arm": arm, "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "progress_sha256": fixture.sha(progress)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--arm", choices=tuple(MAX_SECONDS), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validate_only and args.output_dir is not None:
        parser.error("--output-dir requires --run")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.validate_only:
        prepared = native.preflight_cpu()
        print(json.dumps({"status": "validated_cpu_only", "arm": args.arm,
                          "protocol_sha256": native.PROTOCOL_SHA256,
                          "rows": prepared["rows"],
                          "archive_estimate": native.estimate_native_archive_bytes(args.arm),
                          "planned_counts": planned_counts()}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_arm(args.arm, args.output_dir or RUNS / args.arm / stamp)


if __name__ == "__main__":
    main()
