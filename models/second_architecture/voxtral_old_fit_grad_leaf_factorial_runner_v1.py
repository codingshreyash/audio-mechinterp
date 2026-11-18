"""One-case FP32 grad-context x leaf-flag factorial; live CLI requires a freeze.

The old Stage-B FP32 replay and V1 parity diagnostic failed and remain closed.
This runner isolates two factors and saves every fixed forward before any
cross-forward equality check. No VJP or answer generation occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from contextlib import ExitStack
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

import voxtral_old_fit_handoff_replay_stage_b_v1 as replay
import voxtral_old_fit_handoff_stage_b_runner_v1 as driver
import voxtral_old_fit_handoff_stage_a_runner_v1 as stage_a_runner
import voxtral_old_fit_handoff_native_stage_a_v1 as native
import voxtral_fit_pilot_capability_v1 as fixture
import voxtral_v1_fit_backward_numerical_followup_v2 as prior


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_FACTORIAL_PROSPECTIVE_V1.md"
RUNS = HERE / "runs/voxtral_old_fit_grad_leaf_factorial_v1"
FAILED_MANIFEST = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_FP32_FAILED_ATTEMPT_MANIFEST_V1.json"
FAILED_REVIEW = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_FP32_FAILED_ATTEMPT_INDEPENDENT_REVIEW_V1.md"
V1_FAILED_MANIFEST = HERE / "VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_FAILED_ATTEMPT_MANIFEST_V1.json"
V1_FAILED_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_MODE_PARITY_FAILED_ATTEMPT_INDEPENDENT_REVIEW_V1.md"
STATUS = "completed_old_fit_fp32_grad_leaf_factorial_diagnostic_only"
ERROR_STATUS = "invalid_old_fit_fp32_grad_leaf_factorial_diagnostic"
SCOPE = "one_old_fit_case_grad_leaf_factorial_only_no_vjp_no_security_claim"
FULL_ORDERS = {"encoder31": ("N0", "G1", "G0", "N1", "N1", "G0", "G1", "N0"),
               "projector": ("G1", "N0", "N1", "G0", "G0", "N1", "N0", "G1")}
CONDITIONS = ("N0", "N1", "G0", "G1")
SITES = ("encoder31", "projector")
ORIGINS = ("N0",)
FULL_COUNT = 16
DIRECT_COUNT = 16
MODEL_SECONDS = 300
MAX_DRIVER_BYTES = 30 * 1024**3
MAX_OUTPUT_BYTES = 1024**3
# The launcher also retains two capped 8 MiB streams and final receipts.
MAX_WORKER_BYTES = MAX_OUTPUT_BYTES - 24 * 1024**2
LOGIT_WIDTH = replay.LOGIT_WIDTH
SOURCE_SHA256 = {
    "encoder31": "e0abb561ad91255d1945c81c1dead1a75266f09d0e7f3695a123040fef47b2b6",
    "projector": "a96b5e19ffefe48e537b30ddecb9de11c8c40becf4b4dd3f2813ab5583fce100",
}
BATCH_SHA256 = {
    "input_ids": "915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a",
    "attention_mask": "9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282",
    "input_features": "4f23ae2441eca7587b7acf19d89f717292a4f1efb3067631c7abb7e27236ee61",
}


def sha(path: Path) -> str:
    return driver.sha(path)


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


def _tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype,
            label: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
            value.dtype != dtype or not torch.isfinite(value).all()):
        raise ValueError(f"{label} shape/dtype/finiteness invalid")
    return value


def checked_sources(pairs: tuple[dict, ...] | list[dict]) -> dict[str, torch.Tensor]:
    """Freeze the first cycle-zero clean archive source, with no posthoc pick."""
    expected = [(site, alpha) for site in SITES for alpha in replay.ALPHAS]
    if [(row.get("site"), row.get("alpha")) for row in pairs] != expected:
        raise ValueError("Old-FIT cycle-zero source order differs")
    sources: dict[str, torch.Tensor] = {}
    for site in SITES:
        rows = [row for row in pairs if row["site"] == site]
        clean = _tensor(rows[0]["zero"]["source_full_post"][site],
                        replay.SOURCE_SHAPES[site], torch.bfloat16,
                        f"{site} archived clean source")
        if any(not torch.equal(clean, row["zero"]["source_full_post"][site])
               for row in rows):
            raise ValueError(f"{site} archived clean source varies by signed step")
        if native.first.tensor_sha(clean) != SOURCE_SHA256[site]:
            raise ValueError(f"{site} archived full clean source hash differs")
        sources[site] = _cpu(clean)
    return sources


def checked_batch(prepared: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize the actual preflight BatchFeature before model construction."""
    value = prepared.get("batches", {}).get("beaver")
    if not isinstance(value, Mapping) or set(value) != set(BATCH_SHA256):
        raise ValueError("Old-FIT batch mapping or field set differs")
    specs = {"input_ids": ((1, 396), torch.int64),
             "attention_mask": ((1, 396), torch.int64),
             "input_features": ((1, 128, 3000), torch.float32)}
    batch = {}
    for key, (shape, dtype) in specs.items():
        item = _tensor(value[key], shape, dtype, f"Old-FIT {key}")
        item = _cpu(item)
        if native.first.tensor_sha(item) != BATCH_SHA256[key]:
            raise ValueError(f"Old-FIT {key} tensor hash differs")
        batch[key] = item
    if not torch.equal(batch["attention_mask"], torch.ones_like(batch["attention_mask"])):
        raise ValueError("Old-FIT attention mask differs")
    if int((batch["input_ids"] == 24).sum()) != 375:
        raise ValueError("Old-FIT audio placeholder count differs")
    return batch


def full_schedule() -> tuple[tuple[str, str, int], ...]:
    result = []
    for site in SITES:
        counts = {key: 0 for key in CONDITIONS}
        for mode in FULL_ORDERS[site]:
            result.append((site, mode, counts[mode]))
            counts[mode] += 1
    return tuple(result)


def direct_schedule() -> tuple[tuple[str, str, str, int], ...]:
    result = []
    for site in SITES:
        counts = {key: 0 for key in CONDITIONS}
        for mode in FULL_ORDERS[site]:
            result.append((site, "N0", mode, counts[mode]))
            counts[mode] += 1
    return tuple(result)


def _lm0_pre_hook(captured: list[torch.Tensor],
                  expected_shape: tuple[int, ...], flags: list[dict[str, bool]]):
    def hook(_module, args):
        if not args:
            raise ValueError("LM0 pre-block input absent")
        captured.append(_cpu(_tensor(args[0], expected_shape, torch.float32,
                                     "LM0 pre-block input")))
        flags.append({"grad_enabled": bool(torch.is_grad_enabled()),
                      "requires_grad": bool(args[0].requires_grad)})
    return hook


def full_forward(model: Any, batch: dict[str, torch.Tensor], *, site: str,
                 full_source: torch.Tensor, mode: str, ordinal: int,
                 repeat: int, device: str = "mps") -> dict[str, Any]:
    """Save all boundary states before comparing any two forwards."""
    if site not in SITES or mode not in CONDITIONS:
        raise ValueError("Full forward site/mode invalid")
    _tensor(full_source, replay.SOURCE_SHAPES[site], torch.bfloat16,
            "archived BF16 clean source")
    source_seen: list[torch.Tensor] = []
    factor_seen: list[dict[str, bool]] = []
    projector_seen: list[torch.Tensor] = []
    lm_input_seen: list[torch.Tensor] = []
    lm_input_flags: list[dict[str, bool]] = []
    lm0_input_seen: list[torch.Tensor] = []
    lm0_input_flags: list[dict[str, bool]] = []
    lm_hits, handles = replay._language_stages(model, dtype=torch.float32,
                                                expected_width=replay.LM_INPUT_SHAPE[-1])
    source_leaf = full_source.to(device=device, dtype=torch.float32).detach().clone()
    source_leaf.requires_grad_(mode.endswith("1"))
    module = (model.model.audio_tower.layers[31] if site == "encoder31" else
              model.model.multi_modal_projector)

    def source_override(_module, _args, output):
        _tensor(output, replay.SOURCE_SHAPES[site], torch.float32,
                "unmodified source boundary")
        factor_seen.append({"grad_enabled": bool(torch.is_grad_enabled()),
                            "leaf_requires_grad": bool(source_leaf.requires_grad)})
        return source_leaf

    def source_observer(_module, _args, output):
        source_seen.append(_cpu(_tensor(output, replay.SOURCE_SHAPES[site],
                                        torch.float32, "returned source boundary")))

    def projector_hook(_module, _args, output):
        projector_seen.append(_cpu(_tensor(output, replay.SOURCE_SHAPES["projector"],
                                          torch.float32, "projector output")))

    def lm_input_hook(_module, _args, kwargs):
        value = _tensor(kwargs.get("inputs_embeds"), replay.LM_INPUT_SHAPE,
                        torch.float32, "full LM input")
        lm_input_flags.append({"grad_enabled": bool(torch.is_grad_enabled()),
                               "requires_grad": bool(value.requires_grad)})
        lm_input_seen.append(_cpu(value))

    handles.extend((module.register_forward_hook(source_override),
                    module.register_forward_hook(source_observer),
                    model.model.multi_modal_projector.register_forward_hook(projector_hook),
                    model.model.language_model.register_forward_pre_hook(
                        lm_input_hook, with_kwargs=True),
                    model.model.language_model.layers[0].register_forward_pre_hook(
                        _lm0_pre_hook(lm0_input_seen, replay.LM_INPUT_SHAPE,
                                      lm0_input_flags))))
    try:
        gpu_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.enable_grad() if mode.startswith("G") else torch.no_grad():
            output = model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == "mps":
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if (len(factor_seen) != 1 or
            factor_seen[0] != {"grad_enabled": mode.startswith("G"),
                               "leaf_requires_grad": mode.endswith("1")} or
            len(source_seen) != 1 or len(projector_seen) != 1 or
            len(lm_input_seen) != 1 or len(lm_input_flags) != 1 or
            len(lm0_input_seen) != 1 or len(lm0_input_flags) != 1 or
            lm_input_flags[0]["grad_enabled"] != mode.startswith("G") or
            lm0_input_flags[0]["grad_enabled"] != mode.startswith("G") or
            any(len(values) != 1 for values in lm_hits.values())):
        raise ValueError("Full-model factor state or boundary hook count changed")
    if not torch.equal(source_seen[0], _cpu(source_leaf)):
        raise ValueError("Returned source differs from requested override")
    projector_full = projector_seen[0]
    positions_tensor = torch.nonzero(
        batch["input_ids"][0] == int(model.model.config.audio_token_id),
        as_tuple=False).flatten()
    positions = positions_tensor.tolist()
    if (len(positions) != 375 or
            not torch.equal(lm_input_seen[0][0, positions], projector_full) or
            not torch.equal(lm0_input_seen[0], lm_input_seen[0])):
        raise ValueError("Projector/LM input or LM0 pre-block handoff differs")
    logits = _cpu(_tensor(output.logits[0, -1], (LOGIT_WIDTH,), torch.float32,
                          "full first-answer logits"))
    record = {
        "kind": "full", "ordinal": ordinal, "site": site, "mode": mode,
        "factor_observed": factor_seen[0],
        "downstream_grad_flags": {"lm_input": lm_input_flags[0],
                                  "lm0_pre": lm0_input_flags[0]},
        "repeat": repeat, "source_sha256": native.first.tensor_sha(full_source),
        "source_post_full": source_seen[0],
        "projector_full": projector_full, "lm_input_full": lm_input_seen[0],
        "lm0_input_full": lm0_input_seen[0],
        "lm_terminal": {name: _cpu(values[0][0, -1])
                        for name, values in lm_hits.items()},
        "first_answer_logits": logits,
        "audio_placeholder_positions": positions,
    }
    # The returned graph is intentionally discarded; this diagnostic makes
    # no backward call and never retains live graphs across iterations.
    del output, source_leaf, lm_hits
    return record


def direct_forward(model: Any, batch: dict[str, torch.Tensor], *, site: str,
                   origin_mode: str, origin_ordinal: int, lm_input: torch.Tensor,
                   mode: str, ordinal: int, repeat: int,
                   device: str = "mps") -> dict[str, Any]:
    """Replay one fixed captured LM input under each factorial condition."""
    if site not in SITES or origin_mode not in ORIGINS or mode not in CONDITIONS:
        raise ValueError("Direct forward descriptor invalid")
    _tensor(lm_input, replay.LM_INPUT_SHAPE, torch.float32,
            "captured direct-LM origin input")
    leaf = lm_input.to(device).detach().clone().requires_grad_(mode.endswith("1"))
    lm_hits, handles = replay._language_stages(model, dtype=torch.float32,
                                                expected_width=replay.LM_INPUT_SHAPE[-1])
    lm0_input_seen: list[torch.Tensor] = []
    lm0_input_flags: list[dict[str, bool]] = []
    factor_seen: list[dict[str, bool]] = []
    handles.append(model.model.language_model.layers[0].register_forward_pre_hook(
        _lm0_pre_hook(lm0_input_seen, replay.LM_INPUT_SHAPE, lm0_input_flags)))
    kwargs = {"inputs_embeds": leaf, "use_cache": False}
    for key in ("attention_mask", "position_ids"):
        if key in batch:
            kwargs[key] = batch[key].to(device)
    try:
        with torch.enable_grad() if mode.startswith("G") else torch.no_grad():
            factor_seen.append({"grad_enabled": bool(torch.is_grad_enabled()),
                                "leaf_requires_grad": bool(leaf.requires_grad)})
            output = model.model.language_model(**kwargs)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            logits = model.lm_head(hidden[:, -1:, :])[0, -1]
        if device == "mps":
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    _tensor(logits, (LOGIT_WIDTH,), torch.float32, "direct first logits")
    if (len(factor_seen) != 1 or
            factor_seen[0] != {"grad_enabled": mode.startswith("G"),
                               "leaf_requires_grad": mode.endswith("1")} or
            len(lm0_input_seen) != 1 or len(lm0_input_flags) != 1 or
            lm0_input_flags[0]["grad_enabled"] != mode.startswith("G") or
            any(len(values) != 1 for values in lm_hits.values()) or
            not torch.equal(lm0_input_seen[0], _cpu(leaf))):
        raise ValueError("Direct LM factor state or LM0 input/hook parity differs")
    return {
        "kind": "direct", "ordinal": ordinal, "site": site,
        "factor_observed": factor_seen[0],
        "downstream_grad_flags": {"lm_input": {
            "grad_enabled": factor_seen[0]["grad_enabled"],
            "requires_grad": factor_seen[0]["leaf_requires_grad"]},
                                  "lm0_pre": lm0_input_flags[0]},
        "origin_mode": origin_mode, "origin_full_ordinal": origin_ordinal,
        "mode": mode, "repeat": repeat,
        "input_sha256": native.first.tensor_sha(lm_input),
        "lm_input_full": _cpu(leaf), "lm0_input_full": lm0_input_seen[0],
        "lm_terminal": {name: _cpu(values[0][0, -1])
                        for name, values in lm_hits.items()},
        "first_answer_logits": _cpu(logits),
    }


def _receipt(path: Path, value: dict) -> None:
    fixture.exclusive_json(path, value)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def run_diagnostic(output: Path, *, upstream: dict,
                   model_factory=None, device: str = "mps") -> dict:
    """One fixed forward schedule; the worker supplies reviewed admission."""
    if output.exists():
        raise FileExistsError("Parity diagnostic output already exists; no retry")
    if (not isinstance(upstream, dict) or
            set(upstream) != {"stage_a", "prior_model_work_seconds", "failed_manifest_sha256",
                              "failed_v1_diagnostic_manifest_sha256"} or
            not math.isfinite(upstream["prior_model_work_seconds"])):
        raise ValueError("Reviewed upstream admission absent")
    started = time.monotonic()
    stage_a = upstream["stage_a"]
    details, pairs, estimate = driver.phase_preflight("fp32_replay", stage_a=stage_a)
    sources = checked_sources(pairs)
    prepared = native.preflight_cpu()
    batch = checked_batch(prepared)
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.jsonl"
    progress.touch(exist_ok=False)
    stack = ExitStack()
    memory = None

    def mark(label: str) -> None:
        elapsed = time.monotonic() - started
        peak = int(torch.mps.driver_allocated_memory()) if device == "mps" else 0
        if memory is not None:
            peak = max(peak, memory.driver_max)
        if elapsed >= MODEL_SECONDS or peak >= MAX_DRIVER_BYTES or \
                stage_a_runner.directory_bytes(output) >= MAX_WORKER_BYTES:
            raise ValueError("Parity diagnostic process, MPS, or output cap reached")
        fixture.append_line(progress, {"phase": label, "elapsed_seconds": elapsed,
                                       "mps_driver_bytes": peak})

    try:
        meta = {
            "status": "started_old_fit_fp32_grad_leaf_factorial_diagnostic_only",
            "scope": SCOPE, "protocol_sha256": sha(PROTOCOL),
            "runner_sha256": sha(Path(__file__)),
            "failed_stage_b_fp32_manifest_sha256": upstream["failed_manifest_sha256"],
            "failed_v1_diagnostic_manifest_sha256":
                upstream["failed_v1_diagnostic_manifest_sha256"],
            "stage_a_manifest_sha256": {arm: stage_a["arms"][arm]["manifest_sha256"]
                                         for arm in ("bf16", "fp32")},
            "source_sha256": {site: native.first.tensor_sha(sources[site]) for site in SITES},
            "full_schedule": full_schedule(), "direct_schedule": direct_schedule(),
            "planned_counts": {"full_forwards": FULL_COUNT, "direct_forwards": DIRECT_COUNT,
                               "vjps": 0, "generations": 0},
            "prior_model_work_seconds": upstream["prior_model_work_seconds"],
            "process_cap_seconds": MODEL_SECONDS, "max_mps_driver_bytes": MAX_DRIVER_BYTES,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "max_worker_bytes": MAX_WORKER_BYTES,
            "archive_estimate": estimate, "fresh_v2_corpus_opened": False,
        }
        _receipt(output / "meta.json", meta)
        mark("preflight_complete_before_model_load")
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable")
            recommended = int(torch.mps.recommended_max_memory())
            if recommended <= 0:
                raise RuntimeError("MPS recommended memory unavailable")
            torch.mps.set_per_process_memory_fraction(min(1.0, MAX_DRIVER_BYTES / recommended))
            memory = stack.enter_context(fixture.MemorySampler(torch))
        if model_factory is None:
            from transformers import VoxtralForConditionalGeneration
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(fixture.SNAPSHOT), local_files_only=True, dtype=torch.float32,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
        else:
            if device != "cpu":
                raise ValueError("Injected model factory is CPU-test-only")
            model = model_factory(torch.float32)
        model.requires_grad_(False)
        if model.training or model.model.language_model.training:
            raise ValueError("Parity diagnostic requires eval mode")
        layout = prior.parameter_layout(model, "fp32")
        _receipt(output / "model_layout.json", layout)
        if not layout["valid"]:
            raise ValueError("Parity diagnostic parameter layout invalid")
        mark("model_loaded")
        batch_cpu = {key: _cpu(value) for key, value in batch.items()}
        if (tuple(batch_cpu["input_ids"].shape) != (1, 396) or
                not torch.equal(batch_cpu["attention_mask"],
                                torch.ones_like(batch_cpu["input_ids"]))):
            raise ValueError("Old-FIT input IDs or attention mask changed")
        audio_token_id = int(model.model.config.audio_token_id)
        if audio_token_id != 24 or int((batch_cpu["input_ids"] == audio_token_id).sum()) != 375:
            raise ValueError("Old-FIT audio placeholder token count changed")
        calls: list[dict[str, Any]] = []
        for site, mode, repeat in full_schedule():
            record = full_forward(model, batch, site=site, full_source=sources[site],
                                  mode=mode, ordinal=len(calls), repeat=repeat,
                                  device=device)
            calls.append(record)
            mark(f"after_full_{site}_{mode}_{repeat}")
        origins = {(site, mode): next(record for record in calls
                                      if record["site"] == site and record["mode"] == mode)
                   for site in SITES for mode in ORIGINS}
        for site, origin_mode, mode, repeat in direct_schedule():
            origin = origins[(site, origin_mode)]
            record = direct_forward(
                model, batch, site=site, origin_mode=origin_mode,
                origin_ordinal=origin["ordinal"], lm_input=origin["lm_input_full"],
                mode=mode, ordinal=len(calls), repeat=repeat, device=device)
            calls.append(record)
            mark(f"after_direct_{site}_{origin_mode}_{mode}_{repeat}")
        if len(calls) != FULL_COUNT + DIRECT_COUNT:
            raise ValueError("Parity diagnostic forward count changed")
        mark("all_planned_forwards_complete")
        payload = {"schema": "voxtral_old_fit_grad_leaf_factorial_tensors_v1",
                   "sources_bf16": sources, "batch": batch_cpu,
                   "audio_token_id": audio_token_id, "calls": calls}
        remaining = MAX_WORKER_BYTES - stage_a_runner.directory_bytes(output) - 1_048_576
        if remaining <= 0:
            raise OSError("Parity diagnostic tensor output cap lacks headroom")
        with (output / "details.pt").open("xb") as raw:
            limited = stage_a_runner.LimitedWriter(raw, remaining)
            torch.save(payload, limited)
            limited.flush()
            os.fsync(limited.fileno())
        details_sha, details_path = stage_a_runner.freeze_content_addressed(output / "details.pt")
        mark("details_saved")
        elapsed = time.monotonic() - started
        driver_bytes = int(torch.mps.driver_allocated_memory()) if device == "mps" else 0
        if memory is not None:
            driver_bytes = max(driver_bytes, memory.driver_max)
        result = {
            "status": STATUS, "scope": SCOPE, "protocol_sha256": sha(PROTOCOL),
            "runner_sha256": sha(Path(__file__)),
            "meta_sha256": sha(output / "meta.json"),
            "model_layout_sha256": sha(output / "model_layout.json"),
            "progress_sha256": sha(progress), "details_sha256": details_sha,
            "details_path": details_path, "details_bytes": (output / details_path).stat().st_size,
            "source_sha256": meta["source_sha256"],
            "failed_stage_b_fp32_manifest_sha256": upstream["failed_manifest_sha256"],
            "failed_v1_diagnostic_manifest_sha256":
                upstream["failed_v1_diagnostic_manifest_sha256"],
            "planned_counts": meta["planned_counts"],
            "actual_counts": {"full_forwards": FULL_COUNT, "direct_forwards": DIRECT_COUNT,
                              "vjps": 0, "generations": 0},
            "elapsed_seconds": elapsed, "peak_mps_driver_bytes": driver_bytes,
            "prior_model_work_seconds": upstream["prior_model_work_seconds"],
            "audio_token_id": audio_token_id,
            "execution_metadata": {
                "torch_version": torch.__version__,
                "macos_version": platform.mac_ver()[0],
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
                "model_eval": not model.training,
                "language_model_eval": not model.model.language_model.training,
                "requested_attention_implementation": "sdpa",
                "config_attention_implementation": str(
                    getattr(model.config, "_attn_implementation", None)),
            },
        }
        if stage_a_runner.directory_bytes(output) + 8192 >= MAX_WORKER_BYTES:
            raise OSError("Parity diagnostic output cap reached before result")
        _receipt(output / "result.json", result)
        return result
    except Exception as exc:
        _receipt(output / "error.json", {
            "status": ERROR_STATUS, "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "progress_sha256": sha(progress),
        })
        raise
    finally:
        stack.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (HERE / ".venv").resolve():
        raise ValueError("Parity diagnostic requires pinned second_architecture environment")
    import voxtral_old_fit_grad_leaf_factorial_launch_gate_v1 as gate
    upstream = gate.verify_worker_start(args.attempt_id, args.freeze_sha256)
    output = RUNS / args.attempt_id
    result = run_diagnostic(output, upstream=upstream, model_factory=None, device="mps")
    print(json.dumps({"status": result["status"], "result_sha256": sha(output / "result.json")},
                     allow_nan=False))


if __name__ == "__main__":
    main()
