"""Source-only Stage-B replay primitives for the old-FIT Voxtral handoff study.

This module has no model-loading or --run entry point. The prospective
protocol requires a separate source review, auditor, launcher, and freeze
before any live Stage-B process. These functions make the replay boundary
explicit and allow CPU fake-model tests of its tensor plumbing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import voxtral_old_fit_handoff_native_stage_a_v1 as native
import voxtral_v1_fit_backward_numerical_followup_v2 as prior


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md"
# Update only through a new review/freeze if the committed protocol changes.
PROTOCOL_SHA256 = "61a67ac87a4e4e17cb0b1f7ad972f25fda4e0d90e0b70798e7144b3fd6f2484b"
NATIVE_SOURCE_SHA256 = "99c398948260525a9aa7074abb11d2e2390d4feee1d74662a5e4971f2517de11"
NATIVE_TEST_SHA256 = "8771997481b787e02cd077a9be50abd1f6463e3f61980db7cee84a3b31713aeb"
NATIVE_TEST = HERE / "test_voxtral_old_fit_handoff_native_stage_a_v1.py"
SOURCE_REPLAY_STAGES = {
    "encoder31": (("projector_word", 4), ("lm_input_word", 4),
                  ("lm0", 4), ("lm14", 4), ("lm29", 16), ("lm29_raw", 4)),
    "projector": (("lm_input_word", 4), ("lm0", 4), ("lm14", 4),
                  ("lm29", 16), ("lm29_raw", 4)),
}
SOURCE_REPLAY_VJPS = {"encoder31": 36, "projector": 32}
SOURCE_REPLAY_PAIRS = 12
SOURCE_REPLAY_FORWARDS = 24
SOURCE_REPLAY_PARITY_FORWARDS = 6
BF16_DIRECT_LM_FORWARDS = 24
FP32_DIRECT_LM_FORWARDS = 24
FP32_OVERRIDE_VALIDATION_FORWARDS = 4
STAGE_B_CAP_SECONDS = {"bf16_direct_lm": 600, "fp32_replay": 1800}
STAGE_B_OUTPUT_CAP_BYTES = 1024**3
MAX_MPS_DRIVER_BYTES = 30 * 1024**3
CUMULATIVE_MODEL_WORK_CAP_SECONDS = 4500
ALPHAS = (0.00625, -0.00625, 0.025, -0.025, 0.1, -0.1)
ROWS = {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]}
SOURCE_SHAPES = {"encoder31": (1, 1500, 1280), "projector": (375, 3072)}
LM_INPUT_SHAPE = (1, 396, 3072)
LOGIT_WIDTH = 131072


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_protocol() -> None:
    if sha_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Stage-B protocol bytes changed")
    if (sha_file(Path(native.__file__)) != NATIVE_SOURCE_SHA256 or
            sha_file(NATIVE_TEST) != NATIVE_TEST_SHA256):
        raise ValueError("Committed native Stage-A dependency bytes changed")


def load_audited_native_attempt(attempt: Path, audit_path: Path, *,
                                expected_audit_sha256: str, expected_arm: str = "bf16",
                                expected_audit_status: str =
                                "audited_completed_old_fit_handoff_stage_a_native_only"
                                ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind exact Stage-A bytes to an external audit receipt before replay.

    This loader does not assert that an auditor exists yet. Stage B must supply
    a separately reviewed and content-addressed audit file; a local result
    cannot self-authorize replay. It hashes large tensor files in chunks.
    """
    require_protocol()
    if expected_arm != "bf16":
        raise ValueError("Exact-source replay requires the BF16 Stage-A arm")
    if (not isinstance(expected_audit_sha256, str) or len(expected_audit_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in expected_audit_sha256)):
        raise ValueError("Full external audit SHA-256 required")
    if sha_file(audit_path) != expected_audit_sha256:
        raise ValueError("Stage-A audit file differs from frozen hash")
    files = {"meta_sha256": "meta.json", "model_layout_sha256": "model_layout.json",
             "progress_sha256": "progress.jsonl"}
    result_path = attempt / "result.json"
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if (not isinstance(result, dict) or
            result.get("status") != "completed_old_fit_handoff_stage_a_native_only" or
            result.get("arm") != expected_arm or result.get("protocol_sha256") != PROTOCOL_SHA256 or
            result.get("parity_passed") is not True):
        raise ValueError("Stage-A result status, arm, protocol, or parity invalid")
    for key, name in files.items():
        path = attempt / name
        if sha_file(path) != result.get(key):
            raise ValueError(f"Stage-A {name} hash differs")
    sidecars = {}
    for stem in ("probes", "details"):
        expected_sha = result.get(f"{stem}_sha256")
        relative = result.get(f"{stem}_path")
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or
                any(char not in "0123456789abcdef" for char in expected_sha) or
                relative != f"blobs/{expected_sha}.pt"):
            raise ValueError(f"Stage-A {stem} content-addressed path differs")
        path = attempt / relative
        if (not path.is_file() or path.resolve().parent != (attempt / "blobs").resolve() or
                sha_file(path) != expected_sha):
            raise ValueError(f"Stage-A {stem} sidecar hash differs")
        if (type(result.get(f"{stem}_bytes")) is not int or
                path.stat().st_size != result[f"{stem}_bytes"]):
            raise ValueError(f"Stage-A {stem} sidecar size differs")
        sidecars[stem] = path
    audit = json.loads(audit_path.read_bytes(),
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if (not isinstance(audit, dict) or audit.get("status") != expected_audit_status or
            audit.get("arm") != expected_arm or
            audit.get("protocol_sha256") != PROTOCOL_SHA256 or
            audit.get("result_sha256") != hashlib.sha256(result_raw).hexdigest() or
            audit.get("details_sha256") != result["details_sha256"]):
        raise ValueError("Independent Stage-A audit does not bind this completed result")
    probes = torch.load(sidecars["probes"], map_location="cpu", weights_only=True)
    details = torch.load(sidecars["details"], map_location="cpu", weights_only=True)
    if (not isinstance(probes, dict) or not isinstance(details, dict) or
            not isinstance(details.get("probes"), dict) or
            set(probes) != {"lm", "projector_word", "lm_input_word"}):
        raise ValueError("Stage-A probe or details schema differs")
    for name, shape in (("lm", (3072, 16)), ("projector_word", (9216, 4)),
                        ("lm_input_word", (9216, 4))):
        tensor = probes[name]
        if (not isinstance(tensor, torch.Tensor) or
                tuple(tensor.shape) != shape or tensor.dtype != torch.float64 or
                not torch.equal(tensor, details["probes"].get(name))):
            raise ValueError(f"Stage-A {name} probes differ from audited details")
    cycle_zero_pairs(details)
    return result, details


def planned_source_vjps(site: str) -> tuple[tuple[str, int], ...]:
    if site not in SOURCE_REPLAY_STAGES:
        raise ValueError("Unknown source replay site")
    plan = tuple((stage, i) for stage, count in SOURCE_REPLAY_STAGES[site]
                 for i in range(count))
    if len(plan) != SOURCE_REPLAY_VJPS[site]:
        raise AssertionError("Source replay VJP count changed")
    return plan


def _check_tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype,
                  label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(f"{label} shape or dtype differs")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} is nonfinite")
    return value


def _native_saved(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"Missing {field} trace")
    return value.get("saved", value)


def cycle_zero_pairs(details: dict[str, Any], *, dtype: torch.dtype = torch.bfloat16
                     ) -> tuple[dict[str, Any], ...]:
    """Require every frozen cycle-0 BF16 clean/edit source and LM input pair.

    A full-source replay must use the archived *postpatch* tensors. Nonword
    elements must be bitwise identical within each pair; otherwise a
    selected-row Jacobian contraction would omit part of the actual edit.
    """
    observations = details.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(SOURCE_REPLAY_STAGES):
        raise ValueError("Stage-A observations missing or changed")
    pairs = []
    for site in SOURCE_REPLAY_STAGES:
        rows = observations[site]
        if not isinstance(rows, list) or len(rows) != 18:
            raise ValueError("Stage-A must retain all 18 records per site")
        first_cycle = [row for row in rows if row.get("cycle") == 0]
        if len(first_cycle) != len(ALPHAS) or tuple(float(row.get("alpha")) for row in first_cycle) != ALPHAS:
            raise ValueError("Stage-A cycle-0 signed-alpha order changed")
        for row in first_cycle:
            zero, edited = _native_saved(row, "zero"), _native_saved(row, "edited")
            zp = _check_tensor(zero["source_full_post"][site], SOURCE_SHAPES[site],
                               dtype, "zero full source")
            ep = _check_tensor(edited["source_full_post"][site], SOURCE_SHAPES[site],
                               dtype, "edited full source")
            _check_tensor(zero["lm_input_full"], LM_INPUT_SHAPE, dtype, "zero LM input")
            _check_tensor(edited["lm_input_full"], LM_INPUT_SHAPE, dtype, "edited LM input")
            for trace in (zero, edited):
                _check_tensor(trace["first_answer_logits"], (LOGIT_WIDTH,), dtype, "answer logits")
                positions = trace.get("audio_placeholder_positions")
                if (not isinstance(positions, list) or len(positions) != 375 or
                        len(set(positions)) != 375 or
                        any(type(p) is not int or p < 0 or p >= 396 for p in positions)):
                    raise ValueError("Audio placeholder positions changed")
            if zero["audio_placeholder_positions"] != edited["audio_placeholder_positions"]:
                raise ValueError("Pair audio placeholder positions differ")
            changed = ep != zp
            allowed = torch.zeros_like(changed, dtype=torch.bool)
            if site == "encoder31":
                allowed[0, ROWS[site]] = True
            else:
                allowed[ROWS[site]] = True
            if bool((changed & ~allowed).any()):
                raise ValueError("Replay pair changed source outside frozen word rows")
            if not bool(changed.any()):
                raise ValueError("Nonzero replay pair has no realized source edit")
            pairs.append({"site": site, "cycle": 0, "alpha": float(row["alpha"]),
                          "zero": zero, "edited": edited})
    if len(pairs) != SOURCE_REPLAY_PAIRS:
        raise AssertionError("Replay pair count changed")
    return tuple(pairs)


def estimate_replay_payload_bytes(details: dict[str, Any]) -> dict[str, int]:
    """Preload bound for the two Stage-B processes; metadata has headroom."""
    pairs = cycle_zero_pairs(details)
    source_bytes = sum(
        int(trace["source_full_post"][pair["site"]].numel() *
            trace["source_full_post"][pair["site"]].element_size())
        for pair in pairs for trace in (pair["zero"], pair["edited"]))
    lm_inputs = sum(int(trace["lm_input_full"].numel() * trace["lm_input_full"].element_size())
                    for pair in pairs for trace in (pair["zero"], pair["edited"]))
    # The FP32 source VJPs are stored only on selected rows. No target row is omitted.
    selected_gradients = sum(
        SOURCE_REPLAY_VJPS[site] * len(ROWS[site]) *
        (1280 if site == "encoder31" else 3072) * 4 for site in SOURCE_REPLAY_STAGES)
    stage_vectors = (SOURCE_REPLAY_FORWARDS + SOURCE_REPLAY_PARITY_FORWARDS +
                     BF16_DIRECT_LM_FORWARDS + FP32_DIRECT_LM_FORWARDS +
                     FP32_OVERRIDE_VALIDATION_FORWARDS) * len(native.LM_LAYERS) * 3072 * 4
    total = source_bytes + lm_inputs + selected_gradients + stage_vectors
    if total > STAGE_B_OUTPUT_CAP_BYTES:
        raise ValueError("Prospective Stage-B tensor payload exceeds per-process output cap")
    return {"source_pair_bytes": source_bytes, "lm_input_pair_bytes": lm_inputs,
            "selected_vjp_bytes": selected_gradients,
            "stage_vector_upper_bytes": stage_vectors,
            "estimated_tensor_payload_bytes": total,
            "per_process_output_cap_bytes": STAGE_B_OUTPUT_CAP_BYTES}


def check_replay_limits(phase: str, *, elapsed_seconds: float,
                        previous_model_work_seconds: float,
                        mps_driver_bytes: int, output_bytes: int) -> None:
    """Fail at the frozen process, cumulative, observed MPS, and output caps."""
    if phase not in STAGE_B_CAP_SECONDS:
        raise ValueError("Unknown Stage-B process")
    values = (elapsed_seconds, previous_model_work_seconds)
    if any(not isinstance(value, (int, float)) or not 0 <= value < float("inf") for value in values):
        raise ValueError("Invalid elapsed work counter")
    if elapsed_seconds >= STAGE_B_CAP_SECONDS[phase]:
        raise TimeoutError("Stage-B process wall cap reached")
    if previous_model_work_seconds + elapsed_seconds >= CUMULATIVE_MODEL_WORK_CAP_SECONDS:
        raise TimeoutError("Cumulative four-process model-work cap reached")
    if type(mps_driver_bytes) is not int or mps_driver_bytes < 0 or \
            mps_driver_bytes >= MAX_MPS_DRIVER_BYTES:
        raise MemoryError("Stage-B observed MPS driver cap reached")
    if type(output_bytes) is not int or output_bytes < 0 or \
            output_bytes >= STAGE_B_OUTPUT_CAP_BYTES:
        raise OSError("Stage-B output byte cap reached")


def _capture(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _language_stages(model: Any, *, dtype: torch.dtype, expected_width: int):
    """Register native post-block hooks; returns mutable captures and handles."""
    hits: dict[str, list[torch.Tensor]] = {f"lm{i}": [] for i in native.LM_LAYERS}
    handles = []
    for i in native.LM_LAYERS:
        def hook(_module, _inputs, output, *, layer=i):
            if (not isinstance(output, torch.Tensor) or output.ndim != 3 or
                    output.shape[0] != 1 or output.shape[-1] != expected_width or
                    output.dtype != dtype):
                raise ValueError(f"Unexpected LM{layer} replay state")
            hits[f"lm{layer}"].append(output)
        handles.append(model.model.language_model.layers[i].register_forward_hook(hook))
    return hits, handles


def direct_lm_forward(model: Any, batch: dict[str, torch.Tensor], lm_input: torch.Tensor,
                      dtype: torch.dtype, *, device: str = "mps",
                      expected_shape: tuple[int, ...] = LM_INPUT_SHAPE
                      ) -> dict[str, Any]:
    """Replay exactly one archived LM input without invoking audio preprocessing."""
    _check_tensor(lm_input, expected_shape, dtype, "direct LM input")
    hits, handles = _language_stages(model, dtype=dtype, expected_width=expected_shape[-1])
    kwargs = {"inputs_embeds": lm_input.to(device), "use_cache": False}
    for name in ("attention_mask", "position_ids"):
        if name in batch:
            kwargs[name] = batch[name].to(device)
    try:
        with torch.no_grad():
            output = model.model.language_model(**kwargs)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            logits = model.lm_head(hidden[:, -1:, :])[0, -1]
        if device == "mps":
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != 1 for values in hits.values()):
        raise ValueError("Direct LM stage invocation count changed")
    if (tuple(logits.shape) != (LOGIT_WIDTH,) or logits.dtype != dtype or
            not torch.isfinite(logits).all() or
            any(not torch.isfinite(values[0]).all() for values in hits.values())):
        raise ValueError("Direct LM logits or states differ")
    return {"lm_terminal": {name: _capture(values[0][0, -1]) for name, values in hits.items()},
            "first_answer_logits": _capture(logits),
            "input_sha256": native.first.tensor_sha(lm_input),
            "mode": "direct_lm"}


def assert_direct_parity(reference: dict[str, Any], replay: dict[str, Any]) -> None:
    """Bitwise state and first-logit parity, with no tolerance or case selection."""
    ref = reference.get("saved", reference)
    if set(ref["lm_terminal"]) != set(replay["lm_terminal"]):
        raise ValueError("LM replay layer set changed")
    for layer in ref["lm_terminal"]:
        if not torch.equal(ref["lm_terminal"][layer], replay["lm_terminal"][layer]):
            raise ValueError(f"Direct LM replay mismatch at {layer}")
    if not torch.equal(ref["first_answer_logits"], replay["first_answer_logits"]):
        raise ValueError("Direct LM first-answer logits differ")


def full_model_lm_override(model: Any, batch: dict[str, torch.Tensor], lm_input: torch.Tensor,
                           dtype: torch.dtype, *, device: str = "mps",
                           expected_shape: tuple[int, ...] = LM_INPUT_SHAPE
                           ) -> dict[str, Any]:
    """One full-model check that the direct-LM call uses identical downstream math."""
    _check_tensor(lm_input, expected_shape, dtype, "full-model override input")
    hits, layer_handles = _language_stages(model, dtype=dtype, expected_width=expected_shape[-1])
    lm_hits = []
    def override(_module, args, kwargs):
        original = kwargs.get("inputs_embeds")
        if (not isinstance(original, torch.Tensor) or tuple(original.shape) != expected_shape or
                original.dtype != torch.float32):
            raise ValueError("Full-model LM input shape/dtype changed")
        lm_hits.append(original)
        kwargs = dict(kwargs)
        kwargs["inputs_embeds"] = lm_input.to(original.device)
        return args, kwargs
    handle = model.model.language_model.register_forward_pre_hook(override, with_kwargs=True)
    try:
        gpu_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == "mps":
            torch.mps.synchronize()
    finally:
        handle.remove()
        for h in layer_handles:
            h.remove()
    if len(lm_hits) != 1 or any(len(values) != 1 for values in hits.values()) or \
            any(not torch.isfinite(values[0]).all() for values in hits.values()):
        raise ValueError("Full-model override hook count changed")
    logits = output.logits[0, -1]
    if tuple(logits.shape) != (LOGIT_WIDTH,) or logits.dtype != dtype or not torch.isfinite(logits).all():
        raise ValueError("Full-model override logits differ")
    return {"lm_terminal": {name: _capture(values[0][0, -1]) for name, values in hits.items()},
            "first_answer_logits": _capture(logits),
            "input_sha256": native.first.tensor_sha(lm_input),
            "mode": "full_model_lm_override"}


def source_replay_forward(model: Any, batch: dict[str, torch.Tensor], *,
                          site: str, full_source: torch.Tensor, probes: dict[str, torch.Tensor],
                          grad: bool = False, self_clone: bool = False, device: str = "mps",
                          expected_shape: tuple[int, ...] | None = None,
                          lm_shape: tuple[int, ...] = LM_INPUT_SHAPE
                          ) -> dict[str, Any]:
    """Override one complete audio source tensor and capture downstream stages.

    Only a single selected site is replaced. The returned live graph is used
    for the protocol's 36/32 source-replay VJPs. The source tensor is a fresh
    differentiable leaf in that one graph.
    """
    if site not in SOURCE_REPLAY_STAGES:
        raise ValueError("Unknown source replay site")
    shape = expected_shape or SOURCE_SHAPES[site]
    _check_tensor(full_source, shape, torch.bfloat16, "archived BF16 source")
    source_leaf = full_source.to(device=device, dtype=torch.float32).detach().clone()
    source_leaf.requires_grad_(grad)
    if self_clone:
        source_leaf = source_leaf.clone()
    hits: dict[str, list[torch.Tensor]] = {"projector": [], "lm_input": []}
    lm_hits, handles = _language_stages(model, dtype=torch.float32, expected_width=lm_shape[-1])
    module = (model.model.audio_tower.layers[31] if site == "encoder31" else
              model.model.multi_modal_projector)
    source_hits = []
    def source_hook(_module, _inputs, output):
        if (not isinstance(output, torch.Tensor) or tuple(output.shape) != shape or
                output.dtype != torch.float32):
            raise ValueError("Replay source hook shape/dtype changed")
        source_hits.append(output)
        return source_leaf
    def projector_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.dtype != torch.float32:
            raise ValueError("Replay projector dtype changed")
        hits["projector"].append(output)
    def lm_input_hook(_module, _inputs, kwargs):
        value = kwargs.get("inputs_embeds")
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != lm_shape or value.dtype != torch.float32:
            raise ValueError("Replay LM input shape/dtype changed")
        hits["lm_input"].append(value)
    handles.extend((module.register_forward_hook(source_hook),
                    model.model.multi_modal_projector.register_forward_hook(projector_hook),
                    model.model.language_model.register_forward_pre_hook(lm_input_hook, with_kwargs=True)))
    try:
        gpu_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.enable_grad() if grad else torch.no_grad():
            output = model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == "mps":
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if (len(source_hits) != 1 or
            any(len(values) != 1 for values in hits.values()) or
            any(len(values) != 1 for values in lm_hits.values()) or
            any(not torch.isfinite(values[0]).all() for values in hits.values()) or
            any(not torch.isfinite(values[0]).all() for values in lm_hits.values())):
        raise ValueError("Source replay hook count changed")
    projector = hits["projector"][0]
    lm_input = hits["lm_input"][0]
    positions = torch.nonzero(batch["input_ids"][0] == int(model.model.config.audio_token_id),
                              as_tuple=False).flatten()
    if positions.numel() != 375:
        raise ValueError("Replay audio placeholder count changed")
    if not torch.equal(lm_input[0, positions.to(lm_input.device)], projector):
        raise ValueError("Replay projector/LM handoff differs")
    if (tuple(output.logits.shape[-1:]) != (LOGIT_WIDTH,) or
            output.logits.dtype != torch.float32 or not torch.isfinite(output.logits).all()):
        raise ValueError("Replay answer logit dtype/width changed")
    word_positions = positions[ROWS["projector"]].to(lm_input.device)
    targets = {"projector_word": projector[ROWS["projector"]].reshape(-1),
               "lm_input_word": lm_input[0, word_positions].reshape(-1),
               **{name: values[0][0, -1] for name, values in lm_hits.items()}}
    return {
        "saved": {"source_sha256": native.first.tensor_sha(full_source),
                  "projector_word": _capture(targets["projector_word"]),
                  "lm_input_word": _capture(targets["lm_input_word"]),
                  "lm_terminal": {name: _capture(values[0][0, -1]) for name, values in lm_hits.items()},
                  "first_answer_logits": _capture(output.logits[0, -1]),
                  "audio_placeholder_positions": positions.tolist()},
        "graph_source": source_leaf if grad else None,
        "graph_targets": targets if grad else None,
    }


def replay_source_vjps(trace: dict[str, Any], probes: dict[str, torch.Tensor],
                       site: str) -> dict[str, torch.Tensor]:
    """Compute the exact 36/32 scalar replay derivatives on one clean graph."""
    source = trace.get("graph_source")
    targets = trace.get("graph_targets")
    if not isinstance(source, torch.Tensor) or not isinstance(targets, dict):
        raise ValueError("Clean source-replay graph missing")
    rows: dict[str, list[torch.Tensor]] = {}
    plan = planned_source_vjps(site)
    for index, (stage, probe_index) in enumerate(plan):
        vector = targets["lm29"] if stage == "lm29_raw" else targets[stage]
        if stage == "lm29_raw":
            scalar = vector.float()[probe_index]
        else:
            bank = probes["lm"] if stage.startswith("lm") and stage != "lm_input_word" else probes[stage]
            scalar = torch.dot(vector.float(), bank[:, probe_index].to(vector.device, dtype=torch.float32))
        gradient, = torch.autograd.grad(scalar, source, retain_graph=index < len(plan) - 1,
                                        allow_unused=False)
        if gradient is None or not torch.isfinite(gradient).all() or gradient.dtype != source.dtype:
            raise ValueError("Replay source VJP missing, nonfinite, or wrong dtype")
        selected = gradient[0, ROWS[site]] if site == "encoder31" else gradient[ROWS[site]]
        rows.setdefault(stage, []).append(_capture(selected.float().contiguous()))
    return {stage: torch.stack(values) for stage, values in rows.items()}


def verify_source_replay_pair(zero: dict[str, Any], edited: dict[str, Any],
                              source_zero: torch.Tensor, source_edited: torch.Tensor,
                              site: str) -> dict[str, Any]:
    """Check exact BF16 source pair membership and FP32 replay edit support."""
    _check_tensor(source_zero, SOURCE_SHAPES[site], torch.bfloat16, "zero source")
    _check_tensor(source_edited, SOURCE_SHAPES[site], torch.bfloat16, "edited source")
    if zero["saved"]["source_sha256"] != native.first.tensor_sha(source_zero) or \
            edited["saved"]["source_sha256"] != native.first.tensor_sha(source_edited):
        raise ValueError("Source replay hashes differ")
    diff = source_edited.float() - source_zero.float()
    selected = diff[0, ROWS[site]] if site == "encoder31" else diff[ROWS[site]]
    allowed = torch.zeros_like(diff, dtype=torch.bool)
    if site == "encoder31":
        allowed[0, ROWS[site]] = True
    else:
        allowed[ROWS[site]] = True
    if bool(((diff != 0) & ~allowed).any()):
        raise ValueError("Replay source difference extends beyond word rows")
    return {"site": site, "source_effective_edit": selected.contiguous(),
            "source_edit_l2": float(selected.norm()),
            "terminal_delta_l2": float(
                (edited["saved"]["lm_terminal"]["lm29"].float() -
                 zero["saved"]["lm_terminal"]["lm29"].float()).norm())}
