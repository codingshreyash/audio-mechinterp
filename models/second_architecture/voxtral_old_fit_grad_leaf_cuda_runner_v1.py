"""Two-case old-FIT FP32 CUDA gradient-leaf factorial; live use requires a gate.

This is a diagnostic capture, not a security finding. No generation or VJP.
The frozen MPS runner remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import voxtral_old_fit_grad_leaf_factorial_runner_v1 as mps
import voxtral_old_fit_handoff_replay_stage_b_v1 as replay
import voxtral_old_fit_handoff_native_stage_a_v1 as native
import voxtral_fit_pilot_capability_v1 as fixture


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs/voxtral_old_fit_grad_leaf_cuda_v1"
PROTOCOL_V1 = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_REPLICATION_PROSPECTIVE_V1.md"
PROTOCOL_V2 = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_REPLICATION_PROSPECTIVE_V2.md"
SCHEMA = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_RESULT_SCHEMA_V1.md"
CASE_IDS = ("animal_08_kokoro_fit_chosen_beaver", "nature_09_piper_fit_chosen_waterfall")
WATERFALL_WAV = ROOT / "eval/audio_jlens/corpus_v1/rendered/clips/piper_nature_09_fit_chosen_waterfall.wav"
WATERFALL_WAV_SHA = "05b367cfc538c4c6a7dfd5e9d113b37be42eee1be497dd9c48f7548131adeaef"
BEAVER_ARCHIVE_SHA = "0bc13e5831d3c4c325ad903bcbbdadf9f40d2e38eb67346ac68847bb1840dea2"
BEAVER_BF16_SHA = mps.SOURCE_SHA256
BEAVER_FP32_SHA = {
    "encoder31": "f0efa8b0692c7cd32f89bb9aff0e6dfcb7a799dc785b09f61ab571eddb2299fd",
    "projector": "36ca0e274eae938d23d32d9924d1ec6d854f2607a7eaee40b9a2b47babb5daee",
}
BATCH_SHA = {
    0: mps.BATCH_SHA256,
    1: {**mps.BATCH_SHA256,
        "input_features": "caed5dc4c8328c670c081401c7ddf236f8def0ab3550c9fa62d6aee3a1f8b045"},
}
FULL_ORDERS = mps.FULL_ORDERS
SITES = mps.SITES
CONDITIONS = mps.CONDITIONS
MAX_CUDA_BYTES = 42 * 1024**3
MAX_RSS_BYTES = 110 * 1024**3
MAX_SECONDS = 600
MAX_OUTPUT_BYTES = 1024**3
MAX_WORKER_BYTES = MAX_OUTPUT_BYTES - 24 * 1024**2
LOGIT_WIDTH = replay.LOGIT_WIDTH


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    return native.first.tensor_sha(tensor)


def raw_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.shape == b.shape and a.dtype == b.dtype and
            a.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() ==
            b.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().clone()


def checked(tensor: Any, shape: tuple[int, ...], dtype: torch.dtype,
            label: str) -> torch.Tensor:
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or
            tensor.dtype != dtype or not bool(torch.isfinite(tensor).all())):
        raise ValueError(f"{label} shape/dtype/finiteness invalid")
    return tensor


def full_schedule() -> tuple[tuple[str, str, int], ...]:
    return mps.full_schedule()


def direct_schedule() -> tuple[tuple[str, str, str, int], ...]:
    return mps.direct_schedule()


def checked_batch(batch: dict[str, torch.Tensor], case_index: int) -> dict[str, torch.Tensor]:
    if case_index not in (0, 1) or set(batch) != set(BATCH_SHA[case_index]):
        raise ValueError("Case/batch mapping differs")
    specs = {"input_ids": ((1, 396), torch.int64),
             "attention_mask": ((1, 396), torch.int64),
             "input_features": ((1, 128, 3000), torch.float32)}
    result = {}
    for key, (shape, dtype) in specs.items():
        value = cpu(checked(batch[key], shape, dtype, f"batch {key}"))
        if tensor_sha(value) != BATCH_SHA[case_index][key]:
            raise ValueError(f"batch {key} hash differs")
        result[key] = value
    if (not torch.equal(result["attention_mask"], torch.ones_like(result["attention_mask"])) or
            int((result["input_ids"] == 24).sum()) != 375):
        raise ValueError("Attention mask or audio placeholders differ")
    return result


def checked_beaver_archive(path: Path, expected_sha: str) -> tuple[dict, dict, dict]:
    """Read only the previously reviewed MPS sidecar, never select a fresh source."""
    if expected_sha != BEAVER_ARCHIVE_SHA or sha(path) != BEAVER_ARCHIVE_SHA:
        raise ValueError("Archived beaver MPS sidecar hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "voxtral_old_fit_grad_leaf_factorial_tensors_v1":
        raise ValueError("Archived sidecar schema differs")
    batch = checked_batch(payload["batch"], 0)
    bf16 = {}
    fp32 = {}
    for site in SITES:
        value = cpu(checked(payload["sources_bf16"][site], replay.SOURCE_SHAPES[site],
                            torch.bfloat16, f"beaver {site} BF16 source"))
        cast = cpu(value.float())
        if tensor_sha(value) != BEAVER_BF16_SHA[site] or tensor_sha(cast) != BEAVER_FP32_SHA[site]:
            raise ValueError(f"Archived {site} source hash differs")
        bf16[site], fp32[site] = value, cast
    return batch, bf16, fp32


def checked_waterfall_batch() -> dict[str, torch.Tensor]:
    if sha(WATERFALL_WAV) != WATERFALL_WAV_SHA:
        raise ValueError("Waterfall WAV hash differs")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(fixture.SNAPSHOT), local_files_only=True)
    conversation = [{"role": "user", "content": [
        {"type": "audio", "path": str(WATERFALL_WAV)},
        {"type": "text", "text": fixture.PROMPT},
    ]}]
    raw = processor.apply_chat_template(conversation, tokenize=True, return_dict=True,
                                        return_tensors="pt")
    return checked_batch({key: raw[key] for key in BATCH_SHA[1]}, 1)


def language_hooks(model: Any) -> tuple[dict[str, list[torch.Tensor]], list]:
    return replay._language_stages(model, dtype=torch.float32,
                                   expected_width=replay.LM_INPUT_SHAPE[-1])


def lm0_pre_hook(captured: list[torch.Tensor], flags: list[dict[str, bool]]):
    def hook(_module, args):
        value = checked(args[0], replay.LM_INPUT_SHAPE, torch.float32, "LM0 pre")
        captured.append(cpu(value))
        flags.append({"grad_enabled": bool(torch.is_grad_enabled()),
                      "requires_grad": bool(value.requires_grad)})
    return hook


def checked_lm_terminals(hits: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Validate every saved post-block vector, including finiteness."""
    expected = {"lm0", "lm7", "lm14", "lm22", "lm29"}
    if set(hits) != expected or any(len(values) != 1 for values in hits.values()):
        raise ValueError("LM terminal hook count differs")
    return {name: cpu(checked(values[0][0, -1], (replay.LM_INPUT_SHAPE[-1],),
                              torch.float32, f"LM {name} terminal finiteness"))
            for name, values in hits.items()}


def passive_forward(model: Any, batch: dict[str, torch.Tensor], ordinal: int,
                    device: str = "cuda") -> dict:
    """Observe the two CUDA source sites without overriding either source."""
    hits = {site: [] for site in SITES}
    handles = []
    for site, module in (("encoder31", model.model.audio_tower.layers[31]),
                         ("projector", model.model.multi_modal_projector)):
        def observe(_module, _args, output, *, name=site):
            hits[name].append(cpu(checked(output, replay.SOURCE_SHAPES[name],
                                          torch.float32, f"passive {name}")))
        handles.append(module.register_forward_hook(observe))
    try:
        gpu_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == "cuda":
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    del output
    if any(len(hits[site]) != 1 for site in SITES):
        raise ValueError("Passive source hook count differs")
    return {"ordinal": ordinal, "sources": {site: hits[site][0] for site in SITES},
            "source_sha256": {site: tensor_sha(hits[site][0]) for site in SITES},
            "hook_counts": {site: len(hits[site]) for site in SITES}}


def full_forward(model: Any, batch: dict[str, torch.Tensor], *, site: str,
                 full_source: torch.Tensor, mode: str, ordinal: int, repeat: int,
                 device: str = "cuda") -> dict:
    if site not in SITES or mode not in CONDITIONS:
        raise ValueError("Full forward site/mode invalid")
    checked(full_source, replay.SOURCE_SHAPES[site], torch.float32, "FP32 source")
    source_seen, factor_seen, projector_seen, lm_input_seen = [], [], [], []
    lm_input_flags, lm0_input_seen, lm0_input_flags = [], [], []
    lm_hits, handles = language_hooks(model)
    source_leaf = full_source.to(device).detach().clone().requires_grad_(mode.endswith("1"))
    module = (model.model.audio_tower.layers[31] if site == "encoder31" else
              model.model.multi_modal_projector)

    def override(_module, _args, output):
        checked(output, replay.SOURCE_SHAPES[site], torch.float32, "unmodified source")
        factor_seen.append({"grad_enabled": bool(torch.is_grad_enabled()),
                            "leaf_requires_grad": bool(source_leaf.requires_grad)})
        return source_leaf

    def observe(_module, _args, output):
        source_seen.append(cpu(checked(output, replay.SOURCE_SHAPES[site],
                                       torch.float32, "returned source")))

    def projector(_module, _args, output):
        projector_seen.append(cpu(checked(output, replay.SOURCE_SHAPES["projector"],
                                          torch.float32, "projector")))

    def lm_input(_module, _args, kwargs):
        value = checked(kwargs.get("inputs_embeds"), replay.LM_INPUT_SHAPE,
                        torch.float32, "LM input")
        lm_input_flags.append({"grad_enabled": bool(torch.is_grad_enabled()),
                               "requires_grad": bool(value.requires_grad)})
        lm_input_seen.append(cpu(value))

    handles.extend((module.register_forward_hook(override),
                    module.register_forward_hook(observe),
                    model.model.multi_modal_projector.register_forward_hook(projector),
                    model.model.language_model.register_forward_pre_hook(lm_input, with_kwargs=True),
                    model.model.language_model.layers[0].register_forward_pre_hook(
                        lm0_pre_hook(lm0_input_seen, lm0_input_flags))))
    try:
        gpu_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.enable_grad() if mode.startswith("G") else torch.no_grad():
            output = model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == "cuda":
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    counts = {"source_override": len(factor_seen), "source_observer": len(source_seen),
              "projector": len(projector_seen), "lm_input": len(lm_input_seen),
              "lm0_pre": len(lm0_input_seen),
              **{name: len(values) for name, values in lm_hits.items()}}
    if (any(value != 1 for value in counts.values()) or len(lm_input_flags) != 1 or
            len(lm0_input_flags) != 1 or
            factor_seen[0] != {"grad_enabled": mode.startswith("G"),
                               "leaf_requires_grad": mode.endswith("1")} or
            lm_input_flags[0]["grad_enabled"] != mode.startswith("G") or
            lm0_input_flags[0]["grad_enabled"] != mode.startswith("G")):
        raise ValueError("Full forward factors or hook counts differ")
    if not raw_equal(source_seen[0], cpu(source_leaf)):
        raise ValueError("Returned source differs from frozen override")
    positions = torch.nonzero(batch["input_ids"][0] ==
                              int(model.model.config.audio_token_id)).flatten().tolist()
    if (len(positions) != 375 or
            not raw_equal(lm_input_seen[0][0, positions], projector_seen[0]) or
            not raw_equal(lm0_input_seen[0], lm_input_seen[0])):
        raise ValueError("Projector/LM or LM0 pre handoff differs")
    logits = cpu(checked(output.logits[0, -1], (LOGIT_WIDTH,), torch.float32,
                         "full first logits"))
    result = {"kind": "full", "ordinal": ordinal, "site": site, "mode": mode,
              "repeat": repeat, "factor_observed": factor_seen[0],
              "downstream_grad_flags": {"lm_input": lm_input_flags[0],
                                        "lm0_pre": lm0_input_flags[0]},
              "source_sha256": tensor_sha(full_source), "source_post_full": source_seen[0],
              "projector_full": projector_seen[0], "lm_input_full": lm_input_seen[0],
              "lm0_input_full": lm0_input_seen[0],
              "lm_terminal": checked_lm_terminals(lm_hits),
              "first_answer_logits": logits, "audio_placeholder_positions": positions,
              "hook_counts": counts}
    del output, source_leaf, lm_hits
    return result


def direct_forward(model: Any, batch: dict[str, torch.Tensor], *, site: str,
                   origin_ordinal: int, lm_input: torch.Tensor, mode: str,
                   ordinal: int, repeat: int, device: str = "cuda") -> dict:
    if site not in SITES or mode not in CONDITIONS:
        raise ValueError("Direct forward site/mode invalid")
    checked(lm_input, replay.LM_INPUT_SHAPE, torch.float32, "direct origin LM input")
    leaf = lm_input.to(device).detach().clone().requires_grad_(mode.endswith("1"))
    lm_hits, handles = language_hooks(model)
    lm0_seen, lm0_flags, factor_seen = [], [], []
    handles.append(model.model.language_model.layers[0].register_forward_pre_hook(
        lm0_pre_hook(lm0_seen, lm0_flags)))
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
        if device == "cuda":
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    counts = {"lm0_pre": len(lm0_seen), **{name: len(values) for name, values in lm_hits.items()}}
    if (any(value != 1 for value in counts.values()) or len(lm0_flags) != 1 or
            len(factor_seen) != 1 or
            factor_seen[0] != {"grad_enabled": mode.startswith("G"),
                               "leaf_requires_grad": mode.endswith("1")} or
            lm0_flags[0]["grad_enabled"] != mode.startswith("G") or
            not raw_equal(lm0_seen[0], cpu(leaf))):
        raise ValueError("Direct forward factor or hook/input parity differs")
    result = {"kind": "direct", "ordinal": ordinal, "site": site, "mode": mode,
              "repeat": repeat, "origin_mode": "N0", "origin_full_ordinal": origin_ordinal,
              "factor_observed": factor_seen[0],
              "downstream_grad_flags": {"lm_input": {
                  "grad_enabled": factor_seen[0]["grad_enabled"],
                  "requires_grad": factor_seen[0]["leaf_requires_grad"]},
                  "lm0_pre": lm0_flags[0]},
              "input_sha256": tensor_sha(lm_input), "lm_input_full": cpu(leaf),
              "lm0_input_full": lm0_seen[0],
              "lm_terminal": checked_lm_terminals(lm_hits),
              "first_answer_logits": cpu(checked(logits, (LOGIT_WIDTH,), torch.float32,
                                                   "direct first logits")),
              "hook_counts": counts}
    del output, leaf, lm_hits
    return result


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def put_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError("Worker output contains symlink")
        if item.is_file():
            total += item.stat().st_size
    return total


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class LimitedWriter:
    def __init__(self, raw, limit: int):
        self.raw, self.limit, self.count = raw, limit, 0

    def write(self, chunk: bytes) -> int:
        if self.count + len(chunk) > self.limit:
            raise OSError("Tensor sidecar exceeds output cap")
        count = self.raw.write(chunk)
        self.count += count
        return count

    def flush(self) -> None:
        self.raw.flush()

    def fileno(self) -> int:
        return self.raw.fileno()


def checked_model_layout(model: Any) -> dict:
    parameters, errors = [], []
    for name, value in model.named_parameters():
        if value.dtype != torch.float32 or value.device.type != "cuda":
            errors.append(f"{name}: {value.dtype}/{value.device}")
        if value.requires_grad:
            errors.append(f"{name}: trainable")
        parameters.append({"name": name, "dtype": str(value.dtype),
                           "device": str(value.device), "shape": list(value.shape),
                           "numel": value.numel()})
    if len(parameters) != 762:
        errors.append(f"parameter count {len(parameters)} != 762")
    if model.training or model.model.language_model.training:
        errors.append("model or language model in training mode")
    return {"valid": not errors, "errors": errors,
            "named_parameter_count": len(parameters),
            "parameter_count": sum(row["numel"] for row in parameters),
            "dtype_device_counts": {"torch.float32/cuda": len(parameters)},
            "parameters": parameters}


def configure_cuda() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA init")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, MAX_CUDA_BYTES / total), 0)


def execution_metadata(model: Any) -> dict:
    from importlib.metadata import version
    props = torch.cuda.get_device_properties(0)
    return {
        "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "platform": platform.platform(), "transformers_version": version("transformers"),
        "accelerate_version": version("accelerate"),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "requested_attention_implementation": "sdpa",
        "config_attention_implementation": str(
            getattr(model.config, "_attn_implementation", None)),
        "model_eval": not model.training,
        "language_model_eval": not model.model.language_model.training,
    }


def run_case(output: Path, *, case_index: int, attempt_id: str,
             freeze_sha256: str, admission: dict) -> dict:
    """Execute one preclaimed case. The independent gate supplies admission."""
    if output.exists():
        raise ValueError("Attempt directory already exists; no retry")
    started = time.monotonic()
    # The gate has already made an exclusive no-retry claim. Establish a
    # receipt-capable directory before any input/preflight/disk check.
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.jsonl"
    progress.touch(exist_ok=False)

    def mark(phase: str) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
        else:
            allocated = reserved = 0
        elapsed, rss = time.monotonic() - started, rss_bytes()
        if (elapsed >= MAX_SECONDS or rss >= MAX_RSS_BYTES or
                allocated >= MAX_CUDA_BYTES or reserved >= MAX_CUDA_BYTES or
                directory_bytes(output) >= MAX_WORKER_BYTES):
            raise RuntimeError(f"Worker resource cap reached at {phase}")
        append_json(progress, {"phase": phase, "elapsed_seconds": elapsed,
                               "cuda_allocated_bytes": allocated,
                               "cuda_reserved_bytes": reserved,
                               "process_rss_bytes": rss, "utc": utc()})

    try:
        if case_index not in (0, 1):
            raise ValueError("Case index invalid")
        required = {"environment_receipt_sha256", "checkpoint_tree_sha256"}
        if not isinstance(admission, dict) or not required.issubset(admission):
            raise ValueError("Reviewed admission absent")
        if case_index == 0 and not {"input_bundle_path", "input_bundle_sha256"}.issubset(admission):
            raise ValueError("Reviewed beaver sidecar admission absent")
        # Input and source values are fixed before the first model load.
        if case_index == 0:
            batch, bf16, sources = checked_beaver_archive(
                Path(admission["input_bundle_path"]), admission["input_bundle_sha256"])
        else:
            batch, bf16, sources = checked_waterfall_batch(), {}, {}
        if os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize < 30 * 1024**3:
            raise OSError("Fewer than 30 GiB free before worker start")
        source_rule = ("archived_mps_bf16_cast_fp32" if case_index == 0 else
                       "two_passive_cuda_no_grad_first_source")
        planned = {"passive_forwards": 0 if case_index == 0 else 2,
                   "full_forwards": 16, "direct_forwards": 16, "vjps": 0, "generations": 0}
        meta = {
        "schema": "voxtral_old_fit_grad_leaf_cuda_meta_v1", "status": "started",
        "case_id": CASE_IDS[case_index], "case_index": case_index,
        "attempt_id": attempt_id, "freeze_sha256": freeze_sha256,
        "protocol_v1_sha256": sha(PROTOCOL_V1), "protocol_v2_sha256": sha(PROTOCOL_V2),
        "schema_sha256": sha(SCHEMA), "runner_sha256": sha(Path(__file__)),
        "environment_receipt_sha256": admission["environment_receipt_sha256"],
        "checkpoint_tree_sha256": admission["checkpoint_tree_sha256"],
        "source_rule": source_rule, "batch_sha256": BATCH_SHA[case_index],
        "source_sha256": {site: tensor_sha(sources[site]) for site in sources},
        "full_schedule": full_schedule(), "direct_schedule": direct_schedule(),
        "planned_counts": planned,
        "caps": {"cuda_bytes": MAX_CUDA_BYTES, "rss_bytes": MAX_RSS_BYTES,
                 "worker_seconds": MAX_SECONDS, "worker_output_bytes": MAX_WORKER_BYTES,
                 "total_output_bytes": MAX_OUTPUT_BYTES},
        "started_utc": utc(), "started_monotonic": started,
        }

        put_json(output / "meta.json", meta)
        mark("preflight_complete_before_model_load")
        configure_cuda()
        mark("cuda_configured_before_model_load")
        from transformers import VoxtralForConditionalGeneration
        model = VoxtralForConditionalGeneration.from_pretrained(
            str(fixture.SNAPSHOT), local_files_only=True, dtype=torch.float32,
            low_cpu_mem_usage=True, device_map="cuda", attn_implementation="sdpa").eval()
        model.requires_grad_(False)
        layout = checked_model_layout(model)
        put_json(output / "model_layout.json", layout)
        if not layout["valid"]:
            raise ValueError("CUDA model layout invalid")
        audio_token_id = int(model.model.config.audio_token_id)
        if audio_token_id != 24:
            raise ValueError("Audio token ID differs")
        mark("model_loaded")
        passive = []
        if case_index == 1:
            for ordinal in (0, 1):
                passive.append(passive_forward(model, batch, ordinal))
                mark(f"after_passive_{ordinal}")
            if any(not raw_equal(passive[0]["sources"][site], passive[1]["sources"][site])
                   for site in SITES):
                raise ValueError("Waterfall passive source repeats differ bitwise")
            sources = {site: passive[0]["sources"][site] for site in SITES}
        source_receipt = {
            "schema": "voxtral_old_fit_grad_leaf_cuda_source_receipt_v1",
            "case_id": CASE_IDS[case_index], "source_rule": source_rule,
            "archived_sidecar_sha256": BEAVER_ARCHIVE_SHA if case_index == 0 else None,
            "source_bf16_sha256": {site: tensor_sha(bf16[site]) for site in bf16},
            "source_fp32_sha256": {site: tensor_sha(sources[site]) for site in SITES},
            "passive_source_sha256": [row["source_sha256"] for row in passive],
            "passive_equal": {site: True for site in SITES} if passive else {},
        }
        put_json(output / "source_receipt.json", source_receipt)
        mark("source_frozen")
        calls = []
        for site, mode, repeat in full_schedule():
            calls.append(full_forward(model, batch, site=site, full_source=sources[site],
                                      mode=mode, ordinal=len(calls), repeat=repeat))
            mark(f"after_full_{site}_{mode}_{repeat}")
        origins = {site: next(row for row in calls if row["site"] == site and
                               row["mode"] == "N0") for site in SITES}
        for site, origin_mode, mode, repeat in direct_schedule():
            if origin_mode != "N0":
                raise ValueError("Unexpected direct origin")
            origin = origins[site]
            calls.append(direct_forward(model, batch, site=site,
                                        origin_ordinal=origin["ordinal"],
                                        lm_input=origin["lm_input_full"], mode=mode,
                                        ordinal=len(calls), repeat=repeat))
            mark(f"after_direct_{site}_{mode}_{repeat}")
        if len(calls) != 32:
            raise ValueError("Factorial call count differs")
        payload = {"schema": "voxtral_old_fit_grad_leaf_cuda_tensors_v1",
                   "case_id": CASE_IDS[case_index], "case_index": case_index,
                   "batch": batch, "sources_bf16": bf16, "sources_fp32": sources,
                   "passive": passive, "calls": calls}
        remaining = MAX_WORKER_BYTES - directory_bytes(output) - 1024**2
        if remaining <= 0:
            raise OSError("Output cap lacks sidecar headroom")
        sidecar = output / "details.pt"
        with sidecar.open("xb") as raw:
            writer = LimitedWriter(raw, remaining)
            torch.save(payload, writer)
            writer.flush()
            os.fsync(writer.fileno())
        details_sha = sha(sidecar)
        blobs = output / "blobs"
        blobs.mkdir(exist_ok=False)
        destination = blobs / f"{details_sha}.pt"
        sidecar.rename(destination)
        mark("details_saved")
        result = {
            "schema": "voxtral_old_fit_grad_leaf_cuda_result_v1", "status": "completed",
            "case_id": CASE_IDS[case_index], "case_index": case_index,
            "attempt_id": attempt_id, "freeze_sha256": freeze_sha256,
            "protocol_v1_sha256": meta["protocol_v1_sha256"],
            "protocol_v2_sha256": meta["protocol_v2_sha256"],
            "schema_sha256": meta["schema_sha256"], "runner_sha256": meta["runner_sha256"],
            "meta_sha256": sha(output / "meta.json"),
            "source_receipt_sha256": sha(output / "source_receipt.json"),
            "model_layout_sha256": sha(output / "model_layout.json"),
            "progress_sha256": sha(progress), "details_sha256": details_sha,
            "details_path": str(destination.relative_to(output)),
            "details_bytes": destination.stat().st_size,
            "actual_counts": planned, "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_process_rss_bytes": rss_bytes(), "audio_token_id": audio_token_id,
            "execution_metadata": execution_metadata(model),
        }
        if directory_bytes(output) + 8192 >= MAX_WORKER_BYTES:
            raise OSError("Output cap reached before result")
        put_json(output / "result.json", result)
        return result
    except Exception as exc:
        if not (output / "error.json").exists():
            put_json(output / "error.json", {
                "status": "invalid", "error_type": type(exc).__name__,
                "error": str(exc), "elapsed_seconds": time.monotonic() - started,
                "progress_sha256": sha(progress),
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    import voxtral_old_fit_grad_leaf_cuda_launch_gate_v1 as gate
    admission = gate.verify_worker_start(args.case_index, args.attempt_id,
                                         args.freeze_sha256)
    output = RUNS / args.attempt_id
    result = run_case(output, case_index=args.case_index, attempt_id=args.attempt_id,
                      freeze_sha256=args.freeze_sha256, admission=admission)
    print(json.dumps({"status": result["status"], "result_sha256": sha(output / "result.json")},
                     allow_nan=False))


if __name__ == "__main__":
    main()
