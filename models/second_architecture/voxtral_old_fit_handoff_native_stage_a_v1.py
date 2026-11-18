"""CPU-verifiable native Stage-A Voxtral trace primitives (no live entry point).

The old-FIT handoff protocol is being independently reviewed. This module
provides fixed-probe construction and passive hooks for a prospective runner;
it does not load weights, launch either arm, choose a result, or authorize a
model run. The source-only preflight deliberately rejects changed protocol or
frozen old-FIT dependencies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

import voxtral_fit_pilot_capability_v1 as fixture
import voxtral_fit_v2_causal_first_record as first
import voxtral_v1_fit_backward_numerical_followup_v2 as prior


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md"
# Draft source pin. An independent protocol GO and source freeze are still required.
PROTOCOL_SHA256 = "61a67ac87a4e4e17cb0b1f7ad972f25fda4e0d90e0b70798e7144b3fd6f2484b"
PRIOR_SHA256 = "93d2c607f4c218b9e027ffb4c221347663efb0729cc6e419b9c940dffda89b48"
LM_LAYERS = (0, 7, 14, 22, 29)
OLD_COORDS = (0, 1, 2, 3)
STAGE_ORDER = ("projector_word", "lm_input_word", "lm0", "lm7", "lm14", "lm22", "lm29")
STAGE_WIDTHS = {"projector_word": 3 * 3072, "lm_input_word": 3 * 3072,
                **{f"lm{i}": 3072 for i in LM_LAYERS}}
VJP_COUNT = 4 + 4 + 4 * 4 + 16 + 4
NATIVE_PAIRS_PER_ARM = 2 * 3 * 6
NATIVE_PATCH_FORWARDS_PER_ARM = 2 * NATIVE_PAIRS_PER_ARM
REPLAY_CYCLE_FORWARDS_PER_ARM = 2 * 6 * 2
LM0_CONTROL_FORWARDS_PER_ARM = 24 + 24 + 1
ARCHIVE_CAP_BYTES = {"bf16": int(1.5 * 1024**3), "fp32": int(1.5 * 1024**3)}


def preflight_cpu() -> dict[str, Any]:
    """Pin the same old-FIT sources without loading a model or opening v2 data."""
    if fixture.sha(PROTOCOL) != PROTOCOL_SHA256 or fixture.sha(Path(prior.__file__)) != PRIOR_SHA256:
        raise ValueError("Draft handoff protocol or prior reviewed runner changed")
    prepared = prior.preflight()
    if prepared["rows"] != {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]}:
        raise ValueError("Old-FIT word rows changed")
    return prepared


def _seed(protocol_sha256: str, label: str) -> int:
    if len(protocol_sha256) != 64 or any(c not in "0123456789abcdef" for c in protocol_sha256):
        raise ValueError("A full lowercase protocol SHA-256 is required")
    digest = hashlib.sha256(bytes.fromhex(protocol_sha256) + label.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _orthonormal_rademacher(protocol_sha256: str, label: str, width: int, count: int) -> torch.Tensor:
    """Deterministic CPU float64 QR with positive R diagonal."""
    if count > width or count <= 0:
        raise ValueError("Invalid probe dimensions")
    generator = torch.Generator(device="cpu").manual_seed(_seed(protocol_sha256, label))
    signs = torch.randint(0, 2, (width, count), generator=generator, dtype=torch.int8)
    matrix = signs.to(torch.float64).mul_(2).sub_(1)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diag(r)
    if not torch.isfinite(q).all() or not torch.isfinite(diagonal).all() or bool((diagonal == 0).any()):
        raise ValueError("Degenerate fixed probe QR")
    q *= torch.sign(diagonal).unsqueeze(0)
    if not torch.allclose(q.T @ q, torch.eye(count, dtype=torch.float64), atol=1e-12, rtol=0):
        raise ValueError("Fixed probes are not orthonormal")
    return q.contiguous()


def fixed_probes(protocol_sha256: str = PROTOCOL_SHA256) -> dict[str, torch.Tensor]:
    """Freeze 16 language, four projector-word and four LM-input-word probes."""
    return {
        "lm": _orthonormal_rademacher(protocol_sha256, "lm_residual_16", 3072, 16),
        "projector_word": _orthonormal_rademacher(protocol_sha256, "projector_word_4", 9216, 4),
        "lm_input_word": _orthonormal_rademacher(protocol_sha256, "lm_input_word_4", 9216, 4),
    }


def planned_vjps() -> tuple[tuple[str, int | None], ...]:
    """Exact ordered list of scalar cotangents for one native clean graph."""
    plan: list[tuple[str, int | None]] = []
    for stage in STAGE_ORDER:
        count = 16 if stage == "lm29" else 4
        plan.extend((stage, i) for i in range(count))
    plan.extend(("lm29_raw", i) for i in OLD_COORDS)
    if len(plan) != VJP_COUNT:
        raise AssertionError("Native VJP count drift")
    return tuple(plan)


def estimate_native_archive_bytes(arm: str) -> dict[str, int]:
    """Conservative tensor-payload estimate before any Stage-A model launch.

    It counts duplicated full pre/post source captures on the first 24 patched
    forwards, all 72 projector and LM-input tensors, and every fixed VJP. It
    counts compact LM0-control and parity-state captures. The later control
    implementation must save only the five LM terminal states, not duplicate
    full projector or LM-input tensors on those 49 forwards. The hard cap
    leaves room for serialization metadata and receipts.
    """
    if arm not in ARCHIVE_CAP_BYTES:
        raise ValueError("Unknown arithmetic arm")
    native_bytes = 2 if arm == "bf16" else 4
    per_forward_elements = 375 * 3072 + 396 * 3072 + len(LM_LAYERS) * 3072 + 12 * 1280 + 3 * 3072
    full_source_per_forward_elements = 2 * (1500 * 1280 + 375 * 3072)
    traces = (NATIVE_PATCH_FORWARDS_PER_ARM + 1) * per_forward_elements * native_bytes
    full_sources = (REPLAY_CYCLE_FORWARDS_PER_ARM + 1) * full_source_per_forward_elements * native_bytes
    gradients = VJP_COUNT * (12 * 1280 + 3 * 3072) * 4
    # The probe bank is intentionally present both in probes.pt (frozen before
    # load) and in details.pt (self-contained raw analysis sidecar).
    probes = 2 * (16 * 3072 + 2 * 4 * 9216) * 8
    first_answer_logits = (NATIVE_PATCH_FORWARDS_PER_ARM + 1) * 131072 * native_bytes
    source_effective_edits = 2 * 3 * 6 * (12 * 1280 + 3 * 3072) * 4
    per_coordinate_ulp = (NATIVE_PATCH_FORWARDS_PER_ARM + 1) * sum(STAGE_WIDTHS.values()) * 4
    compact_control = LM0_CONTROL_FORWARDS_PER_ARM * len(LM_LAYERS) * 3072 * native_bytes
    # Six complete prompt-final states and six logit vectors are retained for
    # independent hook parity, not only the five sampled residual vectors.
    parity_states = 6 * 396 * 3072 * native_bytes
    parity_logits = 6 * 131072 * native_bytes
    control_prepatch_and_edits = LM0_CONTROL_FORWARDS_PER_ARM * 3072 * (native_bytes + 4)
    control_unit_and_directions = (3072 + 1280 + 3072) * 4
    donor_terminal_and_logits = (len(LM_LAYERS) * 3072 + 131072) * native_bytes
    total = (traces + full_sources + gradients + probes + first_answer_logits + per_coordinate_ulp +
             source_effective_edits + compact_control + parity_states + parity_logits +
             control_prepatch_and_edits + control_unit_and_directions + donor_terminal_and_logits)
    return {"native_trace_bytes": traces, "first_cycle_full_source_bytes": full_sources,
            "selected_vjp_fp32_bytes": gradients, "probe_float64_bytes": probes,
            "first_answer_logit_bytes": first_answer_logits,
            "per_coordinate_native_ulp_bytes": per_coordinate_ulp,
            "source_effective_edit_bytes": source_effective_edits,
            "compact_lm0_control_bytes": compact_control,
            "control_prepatch_and_edit_bytes": control_prepatch_and_edits,
            "control_unit_and_directions_bytes": control_unit_and_directions,
            "donor_terminal_and_logit_bytes": donor_terminal_and_logits,
            "parity_state_bytes": parity_states, "parity_logit_bytes": parity_logits,
            "estimated_tensor_payload_bytes": total,
            "draft_hard_archive_cap_bytes": ARCHIVE_CAP_BYTES[arm]}


def _source_rows(value: torch.Tensor, site: str, rows: list[int]) -> torch.Tensor:
    return value[0, rows] if site == "encoder31" else value[rows]


def _capture_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def trace_forward(
    model: Any,
    batch: dict[str, torch.Tensor],
    dtype: torch.dtype,
    rows: dict[str, list[int]],
    *,
    device: str = "mps",
    grad: bool = False,
    patch_site: str | None = None,
    direction: torch.Tensor | None = None,
    alpha: float | None = None,
    self_patch: bool = False,
    archive_full_source: bool = False,
    capture_final_full: bool = False,
) -> dict[str, Any]:
    """One no-cache forward with passive stage captures and optional source edit.

    `grad=True` returns live graph references for 44 VJPs; other fields are CPU
    copies. It does not call generate or score a result. `archive_full_source`
    retains the full source tensor needed by the later, separately frozen replay.
    """
    if patch_site not in (None, *prior.SITES) or (patch_site is None and (direction is not None or alpha is not None or self_patch)):
        raise ValueError("Invalid patch specification")
    if patch_site is not None and ((direction is None) == (not self_patch) or
                                   (not self_patch and alpha is None) or
                                   (self_patch and alpha is not None)):
        raise ValueError("Choose exactly one self or directed patch")
    if set(rows) != set(prior.SITES) or rows["encoder31"] != list(range(96, 108)) or rows["projector"] != [24, 25, 26]:
        raise ValueError("Old-FIT source rows changed")
    if tuple(batch["input_ids"].shape) != (1, 396):
        raise ValueError("Old-FIT prompt token shape changed")
    audio_token_id = int(model.model.config.audio_token_id)
    audio_positions = torch.nonzero(batch["input_ids"][0] == audio_token_id, as_tuple=False).flatten()
    if audio_positions.numel() != 375:
        raise ValueError("Old-FIT audio placeholder count changed")
    hits: dict[str, list[torch.Tensor]] = {"encoder31": [], "projector": [], "lm_input": [],
                                          **{f"lm{i}": [] for i in LM_LAYERS}}
    final_hits: list[torch.Tensor] = []
    graph_sources: dict[str, torch.Tensor] = {}
    saved: dict[str, Any] = {"source_rows": {}, "source_full_pre": {}, "source_full_post": {},
                             "source_full_pre_hashes": {}, "source_full_post_hashes": {},
                             "source_effective_edit": None, "source_edit_metrics": None}

    def source_hook(site: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.dtype != dtype or tuple(output.shape) != first.expected_hook_shape(site):
                raise ValueError(f"Unexpected {site} state shape/dtype")
            value = output.detach().requires_grad_(True) if grad and site == "encoder31" else output
            graph_sources[site] = value
            saved["source_rows"][site] = _capture_cpu(_source_rows(value, site, rows[site]))
            if archive_full_source:
                saved["source_full_pre"][site] = _capture_cpu(value)
            saved["source_full_pre_hashes"][site] = first.tensor_sha(
                saved["source_full_pre"][site] if archive_full_source else value)
            patched = value
            if patch_site == site:
                if self_patch:
                    patched = value.clone()
                    edited_rows = _source_rows(value, site, rows[site]).clone()
                    saved["source_effective_edit"] = torch.zeros_like(edited_rows, dtype=torch.float32).cpu()
                    saved["source_edit_metrics"] = {"kind": "self"}
                else:
                    edit = prior.realized_edit(_source_rows(value, site, rows[site]), direction.to(value.device),
                                               float(alpha), dtype)
                    patched = value.clone()
                    edited_rows = edit["edited"]
                    saved["source_effective_edit"] = _capture_cpu(edit["effective"])
                    saved["source_edit_metrics"] = {
                        "kind": "zero" if alpha == 0 else "alpha",
                        "effective_norm": edit["effective_norm"],
                        "mean_fidelity_ratio": edit["mean_fidelity_ratio"],
                        "row_nonuniformity_ratio": edit["row_nonuniformity_ratio"],
                        "unchanged_fraction": edit["unchanged_fraction"],
                    }
                if site == "encoder31":
                    patched[0, rows[site]] = edited_rows
                else:
                    patched[rows[site]] = edited_rows
            if archive_full_source:
                saved["source_full_post"][site] = _capture_cpu(patched)
            saved["source_full_post_hashes"][site] = first.tensor_sha(
                saved["source_full_post"][site] if archive_full_source else patched)
            hits[site].append(patched)
            return patched
        return hook

    def lm_input_hook(_module, _inputs, kwargs):
        value = kwargs.get("inputs_embeds")
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 396, 3072) or value.dtype != dtype:
            raise ValueError("Unexpected LM input tensor")
        hits["lm_input"].append(value)

    def lm_hook(layer: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 396, 3072) or output.dtype != dtype:
                raise ValueError(f"Unexpected LM{layer} state")
            hits[f"lm{layer}"].append(output)
        return hook

    def final_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 396, 3072) or output.dtype != dtype:
            raise ValueError("Unexpected LM final norm state")
        final_hits.append(output)

    handles = [model.model.audio_tower.layers[31].register_forward_hook(source_hook("encoder31")),
               model.model.multi_modal_projector.register_forward_hook(source_hook("projector")),
               model.model.language_model.register_forward_pre_hook(lm_input_hook, with_kwargs=True)]
    handles.extend(model.model.language_model.layers[i].register_forward_hook(lm_hook(i)) for i in LM_LAYERS)
    if capture_final_full:
        handles.append(model.model.language_model.norm.register_forward_hook(final_hook))
    try:
        gpu_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.enable_grad() if grad else torch.no_grad():
            output = model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == "mps":
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != 1 for values in hits.values()):
        raise ValueError("Stage hook invocation count changed")
    if capture_final_full and len(final_hits) != 1:
        raise ValueError("Final norm hook count changed")
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 1, 131072) or logits.dtype != dtype:
        raise ValueError("First-answer logit shape/dtype changed")
    if not all(torch.isfinite(values[0]).all() for values in hits.values()):
        raise ValueError("Nonfinite stage activation")
    projector = hits["projector"][0]
    lm_input = hits["lm_input"][0]
    if not torch.equal(lm_input[0, audio_positions.to(lm_input.device)], projector.to(lm_input.dtype)):
        raise ValueError("Audio projector-to-LM placeholder handoff differs")
    if grad and (not graph_sources["encoder31"].requires_grad or not graph_sources["projector"].requires_grad):
        raise ValueError("Disconnected source graph")
    word_positions = audio_positions[rows["projector"]].to(lm_input.device)
    graph_targets = {
        "projector_word": projector[rows["projector"]].reshape(-1),
        "lm_input_word": lm_input[0, word_positions].reshape(-1),
        **{f"lm{i}": hits[f"lm{i}"][0][0, -1] for i in LM_LAYERS},
    }
    saved["projector_full"] = _capture_cpu(projector)
    saved["lm_input_full"] = _capture_cpu(lm_input)
    saved["first_answer_logits"] = _capture_cpu(logits[0, 0])
    if capture_final_full:
        saved["final_full"] = _capture_cpu(final_hits[0])
    saved["lm_terminal"] = {f"lm{i}": _capture_cpu(hits[f"lm{i}"][0][0, -1]) for i in LM_LAYERS}
    saved["audio_placeholder_positions"] = audio_positions.tolist()
    word_positions_cpu = [saved["audio_placeholder_positions"][i] for i in rows["projector"]]
    native_stage_vectors = {
        "projector_word": saved["projector_full"][rows["projector"]].reshape(-1),
        "lm_input_word": saved["lm_input_full"][0, word_positions_cpu].reshape(-1),
        **saved["lm_terminal"],
    }
    saved["native_ulp_spacing"] = {
        stage: (torch.nextafter(vector, torch.full_like(vector, float("inf"))).float() -
                vector.float()).contiguous()
        for stage, vector in native_stage_vectors.items()
    }
    saved["stage_hashes"] = {
        "projector": first.tensor_sha(saved["projector_full"]),
        "lm_input": first.tensor_sha(saved["lm_input_full"]),
        "first_answer_logits": first.tensor_sha(saved["first_answer_logits"]),
        **{key: first.tensor_sha(value) for key, value in saved["lm_terminal"].items()},
    }
    events = {stage: {"calls": len(values), "shape": list(values[0].shape),
                      "dtype": str(values[0].dtype)} for stage, values in hits.items()}
    if capture_final_full:
        events["final_norm"] = {"calls": len(final_hits), "shape": list(final_hits[0].shape),
                                "dtype": str(final_hits[0].dtype)}
    saved["hook_events"] = events
    return {"saved": saved, "graph_sources": graph_sources if grad else None,
            "graph_targets": graph_targets if grad else None,
            "events": events}


def source_vjps(clean_trace: dict[str, Any], probes: dict[str, torch.Tensor],
                rows: dict[str, list[int]]) -> dict[str, dict[str, torch.Tensor]]:
    """Compute exactly 44 connected scalar VJPs on one clean shared graph."""
    sources = clean_trace.get("graph_sources")
    targets = clean_trace.get("graph_targets")
    if not isinstance(sources, dict) or not isinstance(targets, dict):
        raise ValueError("A grad-enabled clean trace is required")
    plan = planned_vjps()
    collected: dict[str, dict[str, list[torch.Tensor]]] = {stage: {site: [] for site in prior.SITES}
                                                          for stage in STAGE_ORDER}
    collected["lm29_raw"] = {site: [] for site in prior.SITES}
    for index, (stage, probe_index) in enumerate(plan):
        target = targets["lm29"] if stage == "lm29_raw" else targets[stage]
        if target.ndim != 1 or target.numel() != STAGE_WIDTHS["lm29" if stage == "lm29_raw" else stage]:
            raise ValueError("VJP target width differs")
        if stage == "lm29_raw":
            scalar = target.float()[probe_index]
        else:
            bank = probes["lm"] if stage.startswith("lm") and stage != "lm_input_word" else probes[stage]
            scalar = torch.dot(target.float(), bank[:, probe_index].to(target.device, dtype=torch.float32))
        ge, gp = torch.autograd.grad(scalar, (sources["encoder31"], sources["projector"]),
                                     retain_graph=index < len(plan) - 1, allow_unused=False)
        for site, gradient in (("encoder31", ge[0]), ("projector", gp)):
            if gradient is None or not torch.isfinite(gradient).all() or gradient.dtype != sources[site].dtype:
                raise ValueError(f"Nonfinite or wrong-dtype {site} VJP")
            selected = gradient.detach().float().cpu()[rows[site]]
            collected[stage][site].append(selected.contiguous())
    return {stage: {site: torch.stack(values) for site, values in sites.items()}
            for stage, sites in collected.items()}


def project_stage(saved: dict[str, Any], stage: str, probes: dict[str, torch.Tensor],
                  rows: dict[str, list[int]]) -> torch.Tensor:
    """Project one saved CPU stage onto its fixed probe bank."""
    if stage == "projector_word":
        vector = saved["projector_full"][rows["projector"]].reshape(-1).float()
        bank = probes[stage].float()
    elif stage == "lm_input_word":
        positions = saved["audio_placeholder_positions"]
        word_positions = [positions[i] for i in rows["projector"]]
        vector = saved["lm_input_full"][0, word_positions].reshape(-1).float()
        bank = probes[stage].float()
    elif stage in STAGE_WIDTHS and stage.startswith("lm"):
        vector = saved["lm_terminal"][stage].float()
        bank = probes["lm"].float()[:, :16 if stage == "lm29" else 4]
    else:
        raise ValueError("Unknown fixed stage")
    return vector @ bank


def contract_vjps(gradients: torch.Tensor, realized_edit: torch.Tensor) -> torch.Tensor:
    """Contract rowwise FP32 source gradients with the exact realized edit."""
    if gradients.ndim != 3 or realized_edit.ndim != 2 or gradients.shape[1:] != realized_edit.shape:
        raise ValueError("VJP and realized-edit row shapes differ")
    return torch.einsum("nrd,rd->n", gradients.float(), realized_edit.float())


def stage_vector(saved: dict[str, Any], stage: str, rows: dict[str, list[int]]) -> torch.Tensor:
    if stage == "projector_word":
        return saved["projector_full"][rows["projector"]].reshape(-1).float()
    if stage == "lm_input_word":
        positions = saved["audio_placeholder_positions"]
        return saved["lm_input_full"][0, [positions[i] for i in rows["projector"]]].reshape(-1).float()
    if stage in STAGE_WIDTHS and stage.startswith("lm"):
        return saved["lm_terminal"][stage].float()
    raise ValueError("Unknown stage")


def full_stage_vector(saved: dict[str, Any], stage: str, rows: dict[str, list[int]]) -> torch.Tensor:
    """Complete saved boundary state for descriptive change metrics."""
    if stage == "projector_word":
        return saved["projector_full"].reshape(-1).float()
    if stage == "lm_input_word":
        return saved["lm_input_full"].reshape(-1).float()
    return stage_vector(saved, stage, rows)


def cycle_odd_even(measured_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep each frozen signed pair; median is only an optional summary."""
    output = {}
    selected = {(row["cycle"], row["alpha"]): row for row in measured_rows}
    if len(selected) != len(measured_rows):
        raise ValueError("Duplicate frozen signed pair in odd/even diagnostic")
    for magnitude in (0.00625, 0.025, 0.1):
        cycles = {}
        odds, evens = [], []
        for cycle in range(3):
            positive = selected.get((cycle, magnitude))
            negative = selected.get((cycle, -magnitude))
            if positive is None or negative is None:
                raise ValueError("Missing frozen signed pair for odd/even diagnostic")
            p = torch.tensor(positive["measured"], dtype=torch.float64)
            n = torch.tensor(negative["measured"], dtype=torch.float64)
            if p.shape != n.shape:
                raise ValueError("Signed pair vector widths differ")
            odd, even = (p - n) / 2, (p + n) / 2
            odds.append(odd)
            evens.append(even)
            cycles[str(cycle)] = {"positive_alpha": magnitude, "negative_alpha": -magnitude,
                                  "measured_odd": odd.tolist(), "measured_even": even.tolist()}
        output[str(magnitude)] = {
            "cycles": cycles,
            "median_measured_odd": torch.stack(odds).median(0).values.tolist(),
            "median_measured_even": torch.stack(evens).median(0).values.tolist(),
        }
    return output


def analyze_native_pairs(observations: dict[str, list[dict]], vjps: dict,
                         probes: dict[str, torch.Tensor], rows: dict[str, list[int]],
                         directions: dict[str, torch.Tensor]) -> dict[str, Any]:
    """CPU-only resolved/unresolved stage diagnostics on every planned pair."""
    analysis: dict[str, Any] = {}
    for site in prior.SITES:
        records = observations[site]
        if len(records) != 18 or sorted((r["cycle"], r["alpha"]) for r in records) != sorted(
                (cycle, alpha) for cycle, order in enumerate(prior.CYCLES) for alpha in order):
            raise ValueError("Native planned pair matrix incomplete")
        site_analysis = {}
        for stage in (*STAGE_ORDER, "lm29_raw"):
            parent_stage = "lm29" if stage == "lm29_raw" else stage
            zero_projected = []
            for record in records:
                if stage == "lm29_raw":
                    zero_projected.append(stage_vector(record["zero"], "lm29", rows)[list(OLD_COORDS)])
                else:
                    zero_projected.append(project_stage(record["zero"], parent_stage, probes, rows))
            zeros = torch.stack(zero_projected).double()
            noise = float(torch.cdist(zeros, zeros).max())
            resolution = max(3 * noise, 1e-6)
            measured_rows = []
            for index, record in enumerate(records):
                zero = record["zero"]
                edited = record["edited"]
                if stage == "lm29_raw":
                    measured = (stage_vector(edited, "lm29", rows) -
                                stage_vector(zero, "lm29", rows))[list(OLD_COORDS)]
                else:
                    measured = project_stage(edited, parent_stage, probes, rows) - zeros[index].float()
                effective = edited["source_effective_edit"]
                if not isinstance(effective, torch.Tensor):
                    raise ValueError("Edited source lacks an actual edit")
                gradients = vjps[stage][site]
                predicted = contract_vjps(gradients, effective)
                uniform = (gradients.float().sum(1) * effective.float().mean(0)).sum(1)
                ideal = (gradients.float().sum(1) *
                         (float(record["alpha"]) * directions[site].float())).sum(1)
                n_measured = float(torch.linalg.vector_norm(measured.double()))
                n_prediction = float(torch.linalg.vector_norm(predicted.double()))
                err = float(torch.linalg.vector_norm((predicted - measured).double()))
                resolved = n_measured > resolution
                cosine = None
                if resolved and n_prediction > resolution:
                    cosine = float(torch.dot(predicted.double(), measured.double()) /
                                   (n_prediction * n_measured))
                full_delta = (full_stage_vector(edited, parent_stage, rows) -
                              full_stage_vector(zero, parent_stage, rows))
                measured_rows.append({"cycle": record["cycle"], "alpha": record["alpha"],
                                      "measured": measured.tolist(), "predicted_rowwise": predicted.tolist(),
                                      "predicted_uniform": uniform.tolist(), "predicted_ideal": ideal.tolist(),
                                      "measured_l2": n_measured, "absolute_l2_error": err,
                                      "relative_l2_error_if_resolved": err / n_measured if resolved else None,
                                      "cosine_if_resolved": cosine, "resolved": resolved,
                                      "changed_full_coordinates": int((full_delta != 0).sum()),
                                      "full_coordinate_count": full_delta.numel(),
                                      "full_delta_l2": float(torch.linalg.vector_norm(full_delta.double()))})
            decisive = [r for r in measured_rows if abs(r["alpha"]) == 0.1]
            if len(decisive) != 6:
                raise ValueError("Decisive ±0.1 grid incomplete")
            if all(r["resolved"] and r["relative_l2_error_if_resolved"] <= .25 for r in decisive):
                verdict = "local_diagnostic_pass"
            elif all(r["resolved"] and r["relative_l2_error_if_resolved"] > .5 for r in decisive):
                verdict = "diagnostic_mismatch"
            else:
                verdict = "indeterminate"
            odd_even = cycle_odd_even(measured_rows)
            site_analysis[stage] = {"zero_repeat_spread_l2": noise, "resolution_l2": resolution,
                                    "records": measured_rows, "odd_even": odd_even, "verdict": verdict}
        analysis[site] = site_analysis
    return analysis


def fixed_lm0_control_unit(protocol_sha256: str = PROTOCOL_SHA256) -> torch.Tensor:
    """One deterministic CPU FP32 unit vector; alpha sign is applied later."""
    generator = torch.Generator(device="cpu").manual_seed(_seed(protocol_sha256, "lm0_non_audio_control"))
    vector = torch.randint(0, 2, (3072,), generator=generator, dtype=torch.int8).to(torch.float64)
    vector.mul_(2).sub_(1)
    return (vector / torch.linalg.vector_norm(vector)).to(torch.float32).contiguous()


def choose_lm0_control_scale(clean_native: torch.Tensor, unit: torch.Tensor,
                             sign: int, target_norm: float) -> dict[str, Any]:
    """Fixed 0/t, up to eight doubles, then 24 bisections on CPU native math."""
    import math

    if clean_native.shape != (3072,) or clean_native.dtype not in (torch.bfloat16, torch.float32) or \
            unit.shape != (3072,) or unit.dtype != torch.float32 or sign not in (-1, 1):
        raise ValueError("LM0 control source/unit/sign invalid")
    if not torch.isfinite(clean_native).all() or not torch.isfinite(unit).all():
        raise ValueError("Nonfinite LM0 control source/unit")
    if not math.isfinite(target_norm) or target_norm <= 0:
        return {"valid": False, "reason": "zero_or_nonfinite_target", "scalar": 0.0,
                "target_norm": target_norm if math.isfinite(target_norm) else None,
                "realized_norm": 0.0, "visited_candidates": 1}

    def measured(s: float) -> float:
        edited = (clean_native.float() + sign * s * unit).to(clean_native.dtype)
        return float(torch.linalg.vector_norm((edited.float() - clean_native.float()).double()))

    visited = [(0.0, 0.0), (target_norm, measured(target_norm))]
    low, high = 0.0, target_norm
    if visited[-1][1] < target_norm:
        for _ in range(8):
            low, high = high, high * 2
            visited.append((high, measured(high)))
            if visited[-1][1] >= target_norm:
                break
    if visited[-1][1] >= target_norm:
        for _ in range(24):
            midpoint = (low + high) / 2
            actual = measured(midpoint)
            visited.append((midpoint, actual))
            if actual >= target_norm:
                high = midpoint
            else:
                low = midpoint
    finite = [(s, value) for s, value in visited if math.isfinite(value)]
    if not finite:
        return {"valid": False, "reason": "no_finite_candidate", "scalar": 0.0,
                "target_norm": target_norm, "realized_norm": 0.0,
                "visited_candidates": len(visited)}
    scalar, actual = min(finite, key=lambda row: (abs(row[1] - target_norm), row[0]))
    relative = abs(actual - target_norm) / target_norm
    return {"valid": relative <= .05, "reason": "matched" if relative <= .05 else "no_5pct_match",
            "scalar": scalar, "target_norm": target_norm, "realized_norm": actual,
            "relative_match_error": relative, "visited_candidates": len(visited)}


def compact_lm0_control_forward(model: Any, batch: dict[str, torch.Tensor], dtype: torch.dtype,
                                *, device: str = "mps", scalar: float = 0.0,
                                unit: torch.Tensor | None = None, sign: int = 1,
                                self_patch: bool = False) -> dict[str, Any]:
    """No-cache LM0 terminal-only edit; save only later LM terminal vectors."""
    if sign not in (-1, 1) or (unit is None and not self_patch):
        raise ValueError("LM0 control configuration invalid")
    hits = {f"lm{i}": [] for i in LM_LAYERS}
    prepatch = []
    effective = []

    def layer_hook(layer: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.shape != (1, 396, 3072) or output.dtype != dtype:
                raise ValueError("LM0 control hook shape/dtype changed")
            if layer == 0:
                original = output[0, -1]
                prepatch.append(_capture_cpu(original))
                new_row = original.clone() if self_patch else \
                    (original.float() + sign * scalar * unit.to(original.device)).to(dtype)
                effective.append(_capture_cpu(new_row.float() - original.float()))
                patched = output.clone()
                patched[0, -1] = new_row
                hits[f"lm{layer}"].append(patched)
                return patched
            hits[f"lm{layer}"].append(output)
        return hook

    handles = [model.model.language_model.layers[i].register_forward_hook(layer_hook(i)) for i in LM_LAYERS]
    try:
        gpu_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == "mps":
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if len(prepatch) != 1 or len(effective) != 1 or any(len(values) != 1 for values in hits.values()):
        raise ValueError("LM0 control hook count changed")
    if not all(torch.isfinite(values[0]).all() for values in hits.values()):
        raise ValueError("Nonfinite LM0 control state")
    return {"prepatch_lm0": prepatch[0], "effective_edit_lm0": effective[0],
            "lm_terminal": {f"lm{i}": _capture_cpu(hits[f"lm{i}"][0][0, -1]) for i in LM_LAYERS},
            "kind": "self" if self_patch else "zero" if scalar == 0 else "edited",
            "hook_events": {stage: {"calls": len(values), "shape": list(values[0].shape),
                                    "dtype": str(values[0].dtype)} for stage, values in hits.items()}}
