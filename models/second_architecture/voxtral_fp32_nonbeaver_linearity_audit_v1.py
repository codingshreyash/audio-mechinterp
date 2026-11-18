"""Independent CPU arithmetic for the prospective non-beaver FP32 pilot.

This module audits saved tensors only. It does not load model weights or attest
that a claimed forward pass occurred.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SELECTION_SHA256 = "f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b"
BASIS_LABEL = b"color05_fp32_four_probe_v2"
BASIS_SEED = 3693039655029897049
BASIS_SHA256 = "10c9132454a01e034886a3e14bf9b9670e75bb3befd58300b9b45f8d9ba7751f"
ALPHAS = (0.00625, -0.00625, 0.025, -0.025, 0.1, -0.1)
SITES = ("encoder31", "projector")
MODES = ("G1", "N0")
CONTEXTS = (("piper", "silver"), ("piper", "teal"),
            ("kokoro", "silver"), ("kokoro", "teal"))
AUDIT_STATUS = "audited_voxtral_fp32_nonbeaver_saved_data_only"
RESULT_STATUS = "completed_fp32_nonbeaver_local_linearity_raw_only"
SCOPE = "old_fit_color05_fp32_local_linearity_numerical_only"
SCHEMA = "voxtral_fp32_nonbeaver_linearity_tensors_v1"
COUNTS = {"patch_forwards": 96, "vjps": 8, "g1_clean_graphs": 2,
          "n0_clean_references": 2, "donor_captures": 1,
          "recipient_passive_captures": 1,
          "natural_generations": 2}
PROTOCOL_V2_SHA256 = "04f99648f5483fa7f32da80515c31e3f8472511c954cd95711d8eba09bb5b42d"
PROTOCOL_V3_SHA256 = "a222ce0f06a0e82b2c22bf6231e605983d726610b2f51dfbe7299de3ea8d79bd"
PROTOCOL_V4_SHA256 = "edff669ab0c611a50160896b5b4b75e7621dde743c4e769a7b10af08f92a9d94"
PROTOCOL_V5_SHA256 = "99de95003190de6b5fb55c5ba8bed446dfea7cdd8cf53c0569d304bcb942b49b"
MAX_WORKER_BYTES = 2 * 1024**3 - 24 * 1024**2
MAX_MPS_BYTES = 30 * 1024**3
MAX_SECONDS = 300
HERE = Path(__file__).resolve().parent
MASKS_PATH = HERE / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json"
SELECTION_PATH = HERE / "VOXTRAL_FIT_V2_SELECTION.json"
PREFLIGHT_PATH = HERE / "VOXTRAL_FP32_NONBEAVER_CPU_PREFLIGHT_V5.json"
BYTE_BUDGET_PATH = HERE / "VOXTRAL_FP32_NONBEAVER_BYTE_BUDGET_V1.json"
PROTOCOL_V2_PATH = HERE / "VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V2.md"
PROTOCOL_V3_PATH = HERE / "VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V3.md"
PROTOCOL_V4_PATH = HERE / "VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V4.md"
PROTOCOL_V5_PATH = HERE / "VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V5.md"
MASKS_SHA256 = "a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e"
PREFLIGHT_SHA256 = "5ef97e47e66c37b6c8ecb64b366daf4a713cfea1dfdee2f7f59ca75324b08401"
BYTE_BUDGET_SHA256 = "53d19e065669cce63256f39c3b1112bdbbdb52691f2fe5ae475f865ad6191a7f"
RUNNER_SHA256 = "91e622f7fd8f4d9dac00ca7a0baaf02d74faf9326109d10674ebf5673310bc04"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CONTEXT_LABELS = ("piper_silver", "piper_teal", "kokoro_silver", "kokoro_teal")
SOURCE_SHAPES = {"encoder31": (1, 1500, 1280), "projector": (375, 3072)}
LM_INPUT_SHAPE = (1, 396, 3072)
LOGIT_SHAPE = (131072,)
WORD_COUNTS = {"piper": {"encoder31": 60, "projector": 15},
               "kokoro": {"encoder31": 73, "projector": 18}}
BATCH_SHA256 = {
    "input_ids": "915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a",
    "attention_mask": "9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282",
    ("piper", "silver"): "36ec6b38ba8202821e6cabf8e4047ff493a58dce826279385eecb86f6b325d48",
    ("piper", "teal"): "6879a7f915bc01a7d9c117f96dffe119d259e5d947e00f309b9aafdf93b66c4c",
    ("kokoro", "silver"): "5b8fd75b50812f0f6866470fecddcb96d67819da26473dbd7ea78e93bc5f74bc",
    ("kokoro", "teal"): "8e3aaa1a3857682797cae8931c20093abd7fc1e2e9ac2e2647c3aa7df4888221",
}
WAV_SHA256 = {
    ("piper", "silver"): "fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9",
    ("piper", "teal"): "112ed36abf5fb2be1ce77dcfffa9d1bb6deba10b4ecb2d14fc5960b8106626e9",
    ("kokoro", "silver"): "bcad702fa8b55c453286cb0e83ee3336a4bb6ade5e5ac9d4a42dc82c1a97c26e",
    ("kokoro", "teal"): "5038dbe167ae2ba6c9317f74c17bd9bb8ef7d07bbdee9da49f6dc0731e119365",
}
SOURCE_DEPENDENCY_FILES = {
    "fixture_sha256": HERE / "voxtral_fit_pilot_capability_v1.py",
    "fit_first_record_sha256": HERE / "voxtral_fit_v2_causal_first_record.py",
    "mask_builder_sha256": HERE / "voxtral_fit_v2_causal_masks.py",
    "numerical_followup_sha256": HERE / "voxtral_v1_fit_backward_numerical_followup_v2.py",
    "sidecar_writer_sha256": HERE / "voxtral_old_fit_handoff_stage_a_runner_v1.py",
}
RUNNER_PATH = HERE / "voxtral_fp32_nonbeaver_linearity_runner_v1.py"


@dataclass(frozen=True)
class AuditSpec:
    """Production dimensions are fixed; small injected dimensions are test-only."""

    source_shapes: dict[str, tuple[int, ...]]
    lm_input_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    feature_shape: tuple[int, ...]
    rows: dict[str, list[int]]
    basis: torch.Tensor
    batch_hashes: dict[Any, str]
    audio_token_id: int = 24


def production_spec(engine: str, rows: dict[str, list[int]]) -> AuditSpec:
    need(engine in WORD_COUNTS and set(rows) == set(SITES), "Frozen mask rows absent")
    for site in SITES:
        indices = rows[site]
        need(isinstance(indices, list) and len(indices) == WORD_COUNTS[engine][site] and
             all(type(i) is int for i in indices) and
             indices == sorted(set(indices)) and
             all(0 <= i < SOURCE_SHAPES[site][-2] for i in indices),
             f"{site} frozen mask rows invalid")
    return AuditSpec(SOURCE_SHAPES, LM_INPUT_SHAPE, LOGIT_SHAPE,
                     (1, 128, 3000), rows, fixed_basis(), BATCH_SHA256)


def patched_schedule(engine: str) -> list[tuple[str, str, int, float, str]]:
    need(engine in WORD_COUNTS, "Unknown frozen voice")
    sites = SITES if engine == "piper" else tuple(reversed(SITES))
    rows = [(site, mode, cycle, alpha, label)
            for site in sites for mode in MODES
            for cycle, alphas in ((0, ALPHAS), (1, tuple(reversed(ALPHAS))))
            for alpha in alphas for label in ("zero", "edit")]
    need(len(rows) == 96, "Frozen patched schedule count differs")
    return rows


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
         value.is_contiguous(), "Tensor hash needs a contiguous CPU tensor")
    tensor = value.view(torch.uint16) if value.dtype == torch.bfloat16 else value
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def finite_tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype,
                  label: str) -> torch.Tensor:
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
         tuple(value.shape) == shape and value.dtype == dtype and
         value.is_contiguous() and bool(torch.isfinite(value).all()),
         f"{label}: shape/dtype/device/contiguity/finiteness invalid")
    return value


def integer_tensor(value: Any, shape: tuple[int, ...], label: str) -> torch.Tensor:
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
         tuple(value.shape) == shape and value.dtype == torch.int64 and
         value.is_contiguous(), f"{label}: integer tensor invalid")
    return value


def checked_batch(batch: Any, engine: str, word: str,
                  spec: AuditSpec) -> tuple[list[int], dict[str, str]]:
    need(isinstance(batch, dict) and set(batch) ==
         {"input_ids", "attention_mask", "input_features"},
         "Frozen batch field set invalid")
    seq = spec.lm_input_shape[1]
    ids = integer_tensor(batch["input_ids"], (1, seq), "input_ids")
    mask = integer_tensor(batch["attention_mask"], (1, seq), "attention_mask")
    features = finite_tensor(batch["input_features"], spec.feature_shape,
                             torch.float32, "input_features")
    need(bool((mask == 1).all()), "Frozen attention mask differs")
    positions = torch.nonzero(ids[0] == spec.audio_token_id).flatten().tolist()
    need(len(positions) == spec.source_shapes["projector"][0],
         "Audio-placeholder count differs")
    observed = {key: tensor_sha(value) for key, value in batch.items()}
    need(observed["input_ids"] == spec.batch_hashes["input_ids"] and
         observed["attention_mask"] == spec.batch_hashes["attention_mask"] and
         observed["input_features"] == spec.batch_hashes[(engine, word)],
         "Frozen batch tensor hash differs")
    del features
    return positions, observed


TRACE_BASE_KEYS = {
    "site", "mode", "alpha", "source_pre_rows", "source_post_rows",
    "effective_rows", "source_pre_sha256", "source_post_sha256",
    "projector_sha256", "lm_input_sha256", "lm0_pre_sha256",
    "lm_terminal", "first_answer_logits", "audio_positions",
    "runtime_flags", "runtime_checks", "edit_metrics",
}
TRACE_FULL_KEYS = TRACE_BASE_KEYS | {
    "source_pre_full", "source_post_full", "projector_full", "lm_input_full",
}
TRACE_FLAGS = {"grad_enabled", "leaf_requires_grad", "lm0_grad_enabled",
               "lm0_input_requires_grad", "lm_input_requires_grad"}
TRACE_METRICS = {"effective_norm", "mean_fidelity_ratio",
                 "row_nonuniformity_ratio", "unchanged_fraction"}


def checked_trace(row: Any, *, site: str, mode: str, alpha: float,
                  full: bool, positions: list[int], spec: AuditSpec) -> dict[str, Any]:
    """Check one saved trace without inferring that its claimed forward ran."""
    need(isinstance(row, dict) and set(row) ==
         (TRACE_FULL_KEYS if full else TRACE_BASE_KEYS),
         "Trace field set differs")
    need((row["site"], row["mode"], row["alpha"]) == (site, mode, alpha) and
         row["audio_positions"] == positions,
         "Trace descriptor or audio positions differ")
    flags = row["runtime_flags"]
    expected_flag = mode == "G1"
    need(isinstance(flags, dict) and set(flags) == TRACE_FLAGS and
         all(type(value) is bool for value in flags.values()) and
         flags == {key: expected_flag for key in TRACE_FLAGS},
         "Trace recorded runtime flags differ")
    width = spec.source_shapes[site][-1]
    shape = (len(spec.rows[site]), width)
    pre = finite_tensor(row["source_pre_rows"], shape, torch.float32,
                        "prepatch selected rows")
    post = finite_tensor(row["source_post_rows"], shape, torch.float32,
                         "postpatch selected rows")
    effective = finite_tensor(row["effective_rows"], shape, torch.float32,
                              "realized selected rows")
    selected_edit_bitwise = exact_equal(post - pre, effective)
    checks = row["runtime_checks"]
    need(isinstance(checks, dict) and set(checks) ==
         {"selected_edit_bitwise", "zero_source_bitwise",
          "projector_lm_input_bitwise", "lm_input_lm0_pre_bitwise"} and
         all(type(v) is bool for v in checks.values()) and
         checks["selected_edit_bitwise"] == selected_edit_bitwise,
         "Trace runtime check fields or selected edit claim differ")
    need(isinstance(row["lm_terminal"], dict) and
         set(row["lm_terminal"]) == {"lm0", "lm14", "lm29"},
         "LM terminal stage set differs")
    for name, value in row["lm_terminal"].items():
        finite_tensor(value, (spec.basis.shape[0],), torch.float32,
                      f"{name} terminal")
    finite_tensor(row["first_answer_logits"], spec.logits_shape,
                  torch.float32, "first-answer logits")
    for name in ("source_pre_sha256", "source_post_sha256", "projector_sha256",
                 "lm_input_sha256", "lm0_pre_sha256"):
        need(isinstance(row[name], str) and len(row[name]) == 64 and
             all(char in "0123456789abcdef" for char in row[name]),
             f"{name} digest syntax invalid")
    metrics = row["edit_metrics"]
    need(isinstance(metrics, dict) and set(metrics) == TRACE_METRICS and
         all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()),
         "Reported edit metrics invalid")
    if full:
        source_shape = spec.source_shapes[site]
        source_pre = finite_tensor(row["source_pre_full"], source_shape,
                                   torch.float32, "full prepatch source")
        source_post = finite_tensor(row["source_post_full"], source_shape,
                                    torch.float32, "full postpatch source")
        projector = finite_tensor(row["projector_full"],
                                  spec.source_shapes["projector"],
                                  torch.float32, "full projector")
        lm_input = finite_tensor(row["lm_input_full"], spec.lm_input_shape,
                                 torch.float32, "full LM input")
        index_pre = source_pre[0, spec.rows[site]] if site == "encoder31" else source_pre[spec.rows[site]]
        index_post = source_post[0, spec.rows[site]] if site == "encoder31" else source_post[spec.rows[site]]
        need(exact_equal(index_pre, pre) and exact_equal(index_post, post),
             "Saved selected rows differ from full source")
        need(row["source_pre_sha256"] == tensor_sha(source_pre) and
             row["source_post_sha256"] == tensor_sha(source_post) and
             row["projector_sha256"] == tensor_sha(projector) and
             row["lm_input_sha256"] == tensor_sha(lm_input) and
             row["lm0_pre_sha256"] == tensor_sha(lm_input),
             "Full retained source/projector/LM hash differs")
        need(checks["projector_lm_input_bitwise"] ==
             exact_equal(lm_input[0, positions], projector),
             "Projector/LM handoff runtime claim differs")
        if site == "projector":
            need(exact_equal(source_post, projector),
                 "Projector posthook differs from returned source")
        if alpha == 0:
            need(checks["zero_source_bitwise"] ==
                 exact_equal(source_pre, source_post),
                 "Full zero-source runtime claim differs")
    return {"pre": pre, "post": post, "effective": effective,
            "terminal": row["lm_terminal"], "logits": row["first_answer_logits"],
            "checks": checks}


def checked_passive(row: Any, role: str, positions: list[int],
                    spec: AuditSpec) -> dict[str, torch.Tensor]:
    keys = {"kind", "mode", "encoder31_full", "projector_full", "lm_input_full",
            "lm_terminal", "first_answer_logits", "audio_positions", "runtime_flags"}
    need(isinstance(row, dict) and set(row) == keys and
         row["kind"] == role and row["mode"] == "N0" and
         row["audio_positions"] == positions,
         f"Passive {role} descriptor differs")
    need(row["runtime_flags"] == {"grad_enabled": False,
                                  "lm0_grad_enabled": False,
                                  "lm_input_requires_grad": False},
         f"Passive {role} runtime flags differ")
    encoder = finite_tensor(row["encoder31_full"],
                            spec.source_shapes["encoder31"],
                            torch.float32, f"{role} encoder31")
    projector = finite_tensor(row["projector_full"],
                              spec.source_shapes["projector"],
                              torch.float32, f"{role} projector")
    lm_input = finite_tensor(row["lm_input_full"], spec.lm_input_shape,
                             torch.float32, f"{role} LM input")
    need(exact_equal(lm_input[0, positions], projector),
         f"Passive {role} projector/LM handoff differs")
    need(isinstance(row["lm_terminal"], dict) and
         set(row["lm_terminal"]) == {"lm0", "lm14", "lm29"},
         f"Passive {role} LM terminal set differs")
    for name, value in row["lm_terminal"].items():
        finite_tensor(value, (spec.basis.shape[0],), torch.float32,
                      f"{role} {name}")
    finite_tensor(row["first_answer_logits"], spec.logits_shape,
                  torch.float32, f"{role} first logits")
    return {"encoder31": encoder, "projector": projector,
            "lm_input": lm_input}


def checked_generations(value: Any, logit_width: int) -> dict[str, int]:
    need(isinstance(value, dict) and set(value) == {"recipient", "donor"},
         "Natural-generation roles differ")
    lengths: dict[str, int] = {}
    for role, record in value.items():
        need(isinstance(record, dict) and set(record) == {"suffix_ids"} and
             isinstance(record["suffix_ids"], list) and
             0 <= len(record["suffix_ids"]) <= 32 and
             all(type(i) is int and 0 <= i < logit_width for i in record["suffix_ids"]),
             f"{role} natural-generation suffix invalid")
        lengths[role] = len(record["suffix_ids"])
    return lengths


def _metric_claim_close(observed: float, claimed: Any) -> bool:
    return (type(claimed) in (float, int) and math.isfinite(claimed) and
            abs(observed - float(claimed)) <= 1e-4 * max(1.0, abs(observed)))


def zero_metrics_exact(trace: dict[str, Any]) -> bool:
    return trace["edit_metrics"] == {
        "effective_norm": 0.0,
        "mean_fidelity_ratio": 0.0,
        "row_nonuniformity_ratio": 0.0,
        "unchanged_fraction": 1.0,
    }


def clean_zero_identity(trace: dict[str, Any]) -> dict[str, bool]:
    return {
        "zero_source_bitwise": exact_equal(trace["source_pre_full"],
                                           trace["source_post_full"]),
        "zero_effective_all_zero": not bool(torch.count_nonzero(trace["effective_rows"])),
        "zero_metrics_exact": zero_metrics_exact(trace),
        "runtime_selected_edit": trace["runtime_checks"]["selected_edit_bitwise"],
    }


def _five_boundary_parity(clean: dict[str, Any], zero: dict[str, Any]) -> dict[str, bool]:
    names = ("source_post_full", "projector_full", "lm_input_full",
             "first_answer_logits")
    result = {name: exact_equal(clean[name], zero[name]) for name in names}
    result["lm29"] = exact_equal(clean["lm_terminal"]["lm29"],
                                  zero["lm_terminal"]["lm29"])
    result["source_pre_full"] = exact_equal(clean["source_pre_full"],
                                              zero["source_pre_full"])
    return result


def _cycle_metrics(basis: torch.Tensor, gradients: torch.Tensor,
                   delta: torch.Tensor, alpha: float, zero_g: dict,
                   edit_g: dict, zero_n: dict, edit_n: dict,
                   drift_g: dict, drift_n: dict) -> dict[str, Any]:
    realized = edit_g["effective_rows"]
    fidelity_g = edit_fidelity(realized, delta, alpha)
    fidelity_n = edit_fidelity(edit_n["effective_rows"], delta, alpha)
    predictions = derivative_predictions(gradients, realized, delta, alpha)
    effect_g = measured_effect(basis, zero_g["lm_terminal"]["lm29"],
                               edit_g["lm_terminal"]["lm29"])
    effect_n = measured_effect(basis, zero_n["lm_terminal"]["lm29"],
                               edit_n["lm_terminal"]["lm29"])
    observed = effect_g["projected"]
    p_uniform, p_row = predictions["uniform"], predictions["rowwise"]
    numerator_g = float(torch.linalg.vector_norm(observed).item())
    numerator_n = float(torch.linalg.vector_norm(effect_n["projected"]).item())
    floor_g, floor_n = drift_g["resolution_floor"], drift_n["resolution_floor"]
    cosine = None
    if numerator_g > floor_g and float(torch.linalg.vector_norm(p_uniform).item()) > floor_g:
        cosine = float(torch.dot(p_uniform, observed).item() /
                       (torch.linalg.vector_norm(p_uniform).item() * numerator_g))
    interaction = observed - effect_n["projected"]
    logit_g = edit_g["first_answer_logits"].to(torch.float64) - zero_g["first_answer_logits"].to(torch.float64)
    logit_n = edit_n["first_answer_logits"].to(torch.float64) - zero_n["first_answer_logits"].to(torch.float64)
    layer_movements = {
        mode: {stage: float(torch.linalg.vector_norm(
            edited["lm_terminal"][stage].to(torch.float64) -
            zero["lm_terminal"][stage].to(torch.float64)).item())
               for stage in ("lm0", "lm14")}
        for mode, zero, edited in (("G1", zero_g, edit_g), ("N0", zero_n, edit_n))
    }
    return {
        "valid_g1_edit_fidelity": fidelity_g["valid"],
        "valid_n0_edit_fidelity": fidelity_n["valid"],
        "fidelity_g1": fidelity_g, "fidelity_n0": fidelity_n,
        "measured_g1": observed.tolist(), "measured_n0": effect_n["projected"].tolist(),
        "predicted_uniform": p_uniform.tolist(), "predicted_rowwise": p_row.tolist(),
        "predicted_ideal": predictions["ideal"].tolist(),
        "g1_projected_l2": numerator_g, "n0_projected_l2": numerator_n,
        "g1_full_lm29_l2": effect_g["full_l2"],
        "n0_full_lm29_l2": effect_n["full_l2"],
        "lm0_lm14_l2": layer_movements,
        "g1_logit_l2": float(torch.linalg.vector_norm(logit_g).item()),
        "n0_logit_l2": float(torch.linalg.vector_norm(logit_n).item()),
        "g1_top_token_changed": int(torch.argmax(edit_g["first_answer_logits"]).item()) !=
                                int(torch.argmax(zero_g["first_answer_logits"]).item()),
        "n0_top_token_changed": int(torch.argmax(edit_n["first_answer_logits"]).item()) !=
                                int(torch.argmax(zero_n["first_answer_logits"]).item()),
        "mode_interaction": interaction.tolist(),
        "mode_interaction_l2": float(torch.linalg.vector_norm(interaction).item()),
        "g1_resolved": numerator_g > floor_g,
        "n0_resolved": numerator_n > floor_n,
        "uniform_relative_error": relative_error(p_uniform, observed,
                                                  drift_g["observed_max_l2"]),
        "rowwise_relative_error": relative_error(p_row, observed,
                                                  drift_g["observed_max_l2"]),
        "uniform_cosine": cosine,
    }


def audit_payload(payload: Any, spec: AuditSpec) -> dict[str, Any]:
    """Recompute one complete 96-forward context from retained CPU tensors."""
    expected_top = {"schema", "context", "basis", "batches", "rows", "delta",
                    "donor", "recipient_passive", "sites", "forwards", "generations"}
    need(isinstance(payload, dict) and set(payload) == expected_top and
         payload["schema"] == SCHEMA, "Tensor-sidecar top schema invalid")
    context = payload["context"]
    need(isinstance(context, dict) and set(context) ==
         {"index", "label", "engine", "recipient", "donor"},
         "Context descriptor invalid")
    index = context["index"]
    need(type(index) is int and 0 <= index < 4 and
         (context["engine"], context["recipient"]) == CONTEXTS[index] and
         context["label"] == CONTEXT_LABELS[index] and
         context["donor"] == ("teal" if context["recipient"] == "silver" else "silver"),
         "Context frozen order/word relation differs")
    engine, recipient, donor_word = context["engine"], context["recipient"], context["donor"]
    need(payload["rows"] == spec.rows and set(payload["rows"]) == set(SITES),
         "Frozen whole-word masks differ")
    basis = finite_tensor(payload["basis"], tuple(spec.basis.shape),
                          torch.float64, "frozen four-probe basis")
    need(exact_equal(basis, spec.basis), "Saved four-probe basis differs")
    batches = payload["batches"]
    need(isinstance(batches, dict) and set(batches) == {"recipient", "donor"},
         "Frozen batch roles differ")
    positions, recipient_hashes = checked_batch(batches["recipient"], engine,
                                                recipient, spec)
    donor_positions, donor_hashes = checked_batch(batches["donor"], engine,
                                                  donor_word, spec)
    need(positions == donor_positions, "Recipient/donor audio positions differ")
    donor = checked_passive(payload["donor"], "donor", positions, spec)
    recipient_passive = checked_passive(payload["recipient_passive"],
                                        "recipient", positions, spec)
    generation_lengths = checked_generations(payload["generations"],
                                             spec.logits_shape[0])
    sites = payload["sites"]
    need(isinstance(sites, dict) and set(sites) == set(SITES),
         "Two fixed site records required")
    deltas = payload["delta"]
    need(isinstance(deltas, dict) and set(deltas) == set(SITES),
         "Two fixed donor directions required")
    site_data: dict[str, dict[str, Any]] = {}
    computed_invalidities: list[dict[str, Any]] = []
    for site in SITES:
        site_record = sites[site]
        need(isinstance(site_record, dict) and set(site_record) ==
             {"g1_clean", "n0_clean", "gradients"},
             f"{site}: source graph bundle differs")
        g1_clean = site_record["g1_clean"]
        n0_clean = site_record["n0_clean"]
        checked_trace(g1_clean, site=site, mode="G1", alpha=0.0,
                      full=True, positions=positions, spec=spec)
        checked_trace(n0_clean, site=site, mode="N0", alpha=0.0,
                      full=True, positions=positions, spec=spec)
        clean_identity = {}
        for mode, clean in (("N0", n0_clean), ("G1", g1_clean)):
            checks = clean_zero_identity(clean)
            clean_identity[mode] = all(checks.values())
            failed = [key for key, value in checks.items() if not value]
            if failed:
                computed_invalidities.append({
                    "stage": "site_clean_zero_identity",
                    "descriptor": {"site": site, "mode": mode},
                    "failed_checks": failed})
        width = spec.source_shapes[site][-1]
        nrows = len(spec.rows[site])
        gradients = finite_tensor(site_record["gradients"], (4, nrows, width),
                                  torch.float32, f"{site} four row VJPs")
        direction = finite_tensor(deltas[site], (width,), torch.float32,
                                  f"{site} donor direction")
        need(float(torch.linalg.vector_norm(direction).item()) > 0,
             f"{site}: zero donor direction")
        donor_full = donor[site]
        donor_rows = donor_full[0, spec.rows[site]] if site == "encoder31" else donor_full[spec.rows[site]]
        recipient_full = recipient_passive[site]
        recipient_rows = (recipient_full[0, spec.rows[site]] if site == "encoder31"
                          else recipient_full[spec.rows[site]])
        recomputed = (donor_rows.mean(dim=0) - recipient_rows.mean(dim=0)).contiguous()
        need(exact_equal(direction, recomputed),
             f"{site}: donor-minus-passive-recipient direction differs")
        passive_parity = exact_equal(recipient_full, n0_clean["source_pre_full"])
        if not passive_parity:
            computed_invalidities.append({
                "stage": "passive_recipient_to_n0_clean",
                "descriptor": {"site": site},
                "failed_checks": ["full_prepatch_source_bitwise"]})
        site_data[site] = {"g1_clean": g1_clean, "n0_clean": n0_clean,
                           "gradients": gradients, "delta": direction,
                           "passive_parity": passive_parity,
                           "clean_identity": clean_identity,
                           "parity_failures": [], "zeros": {mode: [] for mode in MODES},
                           "paired": {mode: {} for mode in MODES}}
    forwards = payload["forwards"]
    schedule = patched_schedule(engine)
    need(isinstance(forwards, list) and len(forwards) == 96,
         "Exactly 96 patched forward records required")
    for ordinal, (site, mode, cycle, alpha, label) in enumerate(schedule):
        record = forwards[ordinal]
        descriptor = {"ordinal": ordinal, "site": site, "mode": mode,
                      "cycle": cycle, "alpha": alpha, "label": label}
        need(isinstance(record, dict) and set(record) == set(descriptor) | {"trace"} and
             all(record[key] == value for key, value in descriptor.items()),
             f"forward {ordinal}: fixed schedule differs")
        trace = record["trace"]
        reference = site_data[site]["g1_clean" if mode == "G1" else "n0_clean"]
        checked_trace(trace, site=site, mode=mode,
                      alpha=(0.0 if label == "zero" else alpha),
                      full=label == "zero", positions=positions, spec=spec)
        checks = {
            "prepatch_full_hash": trace["source_pre_sha256"] == reference["source_pre_sha256"],
            "prepatch_selected_rows": exact_equal(trace["source_pre_rows"],
                                                     reference["source_pre_rows"]),
            "runtime_selected_edit": trace["runtime_checks"]["selected_edit_bitwise"],
        }
        if label == "zero":
            parity = _five_boundary_parity(reference, trace)
            checks.update(parity)
            checks["zero_source_bitwise"] = trace["runtime_checks"]["zero_source_bitwise"]
            checks["zero_effective_all_zero"] = not bool(torch.count_nonzero(
                trace["effective_rows"]))
            checks["zero_metrics_exact"] = zero_metrics_exact(trace)
            site_data[site]["zeros"][mode].append(trace)
            site_data[site]["parity_failures"].extend(
                {"mode": mode, "cycle": cycle, "alpha": alpha, "boundary": name}
                for name, passed in parity.items() if not passed)
        else:
            paired_zero = site_data[site]["paired"][mode].get((cycle, alpha), {}).get("zero")
            need(paired_zero is not None,
                 f"forward {ordinal}: edited forward lacks preceding zero")
            checks["paired_prepatch_selected_rows_bitwise"] = exact_equal(
                trace["source_pre_rows"], paired_zero["trace"]["source_pre_rows"])
            fidelity = edit_fidelity(trace["effective_rows"],
                                     site_data[site]["delta"], alpha)
            claims = trace["edit_metrics"]
            for observed, key in ((fidelity["effective_norm"], "effective_norm"),
                                  (fidelity["mean_fidelity_ratio"], "mean_fidelity_ratio"),
                                  (fidelity["row_nonuniformity_ratio"], "row_nonuniformity_ratio")):
                need(_metric_claim_close(observed, claims[key]),
                     f"forward {ordinal}: {key} report differs from saved edit")
            unchanged = float((trace["effective_rows"] == 0).to(torch.float32).mean().item())
            need(_metric_claim_close(unchanged, claims["unchanged_fraction"]),
                 f"forward {ordinal}: unchanged fraction differs")
            checks.update({"realized_norm": fidelity["effective_norm"] > 0,
                           "mean_fidelity": fidelity["mean_fidelity_ratio"] <= 0.10,
                           "row_uniformity": fidelity["row_nonuniformity_ratio"] <= 0.10})
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            computed_invalidities.append({"stage": "paired_forward",
                                         "descriptor": descriptor,
                                         "failed_checks": failed})
        key = (cycle, alpha)
        pair = site_data[site]["paired"][mode].setdefault(key, {})
        need(label not in pair, f"forward {ordinal}: duplicate paired label")
        pair[label] = {"trace": trace, "valid": not failed,
                       "failed_checks": failed}
    site_reports: dict[str, Any] = {}
    for site in SITES:
        record = site_data[site]
        drifts = {mode: zero_drift(basis, [item["lm_terminal"]["lm29"]
                                           for item in record["zeros"][mode]])
                  for mode in MODES}
        rows: list[dict[str, Any]] = []
        for alpha in ALPHAS:
            cycles = []
            for cycle in (0, 1):
                g = record["paired"]["G1"][(cycle, alpha)]
                n = record["paired"]["N0"][(cycle, alpha)]
                for pair in (g, n):
                    need(set(pair) == {"zero", "edit"},
                         f"{site} cycle {cycle} {alpha}: incomplete pair")
                numeric = _cycle_metrics(
                    basis, record["gradients"], record["delta"], alpha,
                    g["zero"]["trace"], g["edit"]["trace"],
                    n["zero"]["trace"], n["edit"]["trace"],
                    drifts["G1"], drifts["N0"])
                valid_cycle = (all(item["valid"] for item in (*g.values(), *n.values())) and
                               numeric["valid_g1_edit_fidelity"] and
                               numeric["valid_n0_edit_fidelity"])
                cycles.append({"cycle": cycle, "valid": valid_cycle,
                               "numeric": numeric})
            valid = bool(record["passive_parity"] and
                         all(record["clean_identity"].values()) and
                         all(row["valid"] for row in cycles) and
                         not record["parity_failures"])
            max_error = (max(row["numeric"]["uniform_relative_error"]
                             for row in cycles) if valid else None)
            rows.append({"context_index": index, "context_label": context["label"],
                         "site": site, "alpha": alpha, "valid": valid,
                         "resolved": bool(valid and all(row["numeric"]["g1_resolved"]
                                                        for row in cycles)),
                         "primary_max_cycle_error": max_error,
                         "rowwise_max_cycle_error":
                             max(row["numeric"]["rowwise_relative_error"]
                                 for row in cycles) if valid else None,
                         "cycles": cycles})
        by_alpha = {row["alpha"]: row for row in rows}
        odd_even = []
        for h in (0.00625, 0.025, 0.1):
            for cycle in (0, 1):
                positive = by_alpha[h]["cycles"][cycle]["numeric"]
                negative = by_alpha[-h]["cycles"][cycle]["numeric"]
                contrasts = {}
                for key in ("measured_g1", "measured_n0", "predicted_uniform",
                            "predicted_rowwise"):
                    plus = torch.as_tensor(positive[key], dtype=torch.float64)
                    minus = torch.as_tensor(negative[key], dtype=torch.float64)
                    contrasts[key] = {"odd": ((plus - minus) / 2).tolist(),
                                      "even": ((plus + minus) / 2).tolist()}
                odd_even.append({"absolute_alpha": h, "cycle": cycle,
                                 "contrasts": contrasts})
        g1_clean, n0_clean = record["g1_clean"], record["n0_clean"]
        cross_mode_clean = {
            "prepatch_full_bitwise": exact_equal(
                g1_clean["source_pre_full"], n0_clean["source_pre_full"]),
            "prepatch_selected_rows_bitwise": exact_equal(
                g1_clean["source_pre_rows"], n0_clean["source_pre_rows"]),
            "postpatch_full_bitwise": exact_equal(
                g1_clean["source_post_full"], n0_clean["source_post_full"]),
            "projector_full_bitwise": exact_equal(
                g1_clean["projector_full"], n0_clean["projector_full"]),
            "lm_input_full_bitwise": exact_equal(
                g1_clean["lm_input_full"], n0_clean["lm_input_full"]),
            "lm29_l2": float(torch.linalg.vector_norm(
                g1_clean["lm_terminal"]["lm29"].to(torch.float64) -
                n0_clean["lm_terminal"]["lm29"].to(torch.float64)).item()),
            "first_logit_l2": float(torch.linalg.vector_norm(
                g1_clean["first_answer_logits"].to(torch.float64) -
                n0_clean["first_answer_logits"].to(torch.float64)).item()),
        }
        site_reports[site] = {"cells": rows, "zero_drift": drifts,
                              "passive_recipient_to_n0_clean_full_prepatch_bitwise":
                                  record["passive_parity"],
                              "site_clean_zero_identity_valid": record["clean_identity"],
                              "cross_mode_clean_descriptive": cross_mode_clean,
                              "clean_zero_parity_failures": record["parity_failures"],
                              "odd_even": odd_even}
    return {"status": AUDIT_STATUS, "scope": SCOPE, "context": context,
            "saved_counts": COUNTS, "basis_sha256": BASIS_SHA256,
            "batch_sha256": {"recipient": recipient_hashes, "donor": donor_hashes},
            "generation_suffix_lengths": generation_lengths,
            "computed_invalidities": computed_invalidities,
            "sites": site_reports,
            "interpretation_limit":
                "One already-inspected FIT pair, saved tensors only. Edited full-source, "
                "projector, LM-input and LM0-pre hashes are claims where full tensors "
                "were not retained. This audit does not attest model execution, a "
                "reusable lens, an audio security finding, or time-localized acoustic "
                "causality."}


def exact_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality, including signed zero; inputs are already validated."""
    return (a.dtype == b.dtype and a.shape == b.shape and
            a.contiguous().view(torch.uint8).equal(b.contiguous().view(torch.uint8)))


def fixed_basis() -> torch.Tensor:
    """Recreate the approved four-probe QR recipe without consulting runner code."""
    seed_material = bytes.fromhex(SELECTION_SHA256) + BASIS_LABEL
    digest = hashlib.sha256(seed_material).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**63)
    need(seed == BASIS_SEED, "Frozen basis seed differs")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (3072, 4), generator=generator, dtype=torch.int8)
    matrix = signs.to(torch.float64).mul_(2).sub_(1)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diag(r)
    need(bool(torch.isfinite(q).all()) and bool(torch.isfinite(diagonal).all()) and
         bool((diagonal != 0).all()), "Degenerate frozen basis")
    q = (q * torch.sign(diagonal).unsqueeze(0)).contiguous()
    need(bool(torch.allclose(q.T @ q, torch.eye(4, dtype=torch.float64),
                             atol=1e-12, rtol=0)), "Basis is not orthonormal")
    payload = q.numpy().astype("<f8", copy=False).tobytes(order="C")
    observed = hashlib.sha256(b"float64[3072,4]\0" + payload).hexdigest()
    need(observed == BASIS_SHA256, "Frozen basis digest differs")
    return q


def projection(basis: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
    width, rank = basis.shape
    finite_tensor(basis, (width, rank), torch.float64, "basis")
    finite_tensor(terminal, (width,), torch.float32, "terminal")
    return basis.T @ terminal.to(torch.float64)


def zero_drift(basis: torch.Tensor, zeros: list[torch.Tensor]) -> dict[str, Any]:
    """All 66 unordered projected comparisons of twelve zero forwards."""
    need(len(zeros) == 12, "Exactly twelve zero forwards required")
    projected = [projection(basis, item) for item in zeros]
    spreads = [float(torch.linalg.vector_norm(projected[i] - projected[j]).item())
               for i in range(12) for j in range(i + 1, 12)]
    need(len(spreads) == 66 and all(math.isfinite(x) for x in spreads),
         "Zero-repeat spread invalid")
    observed = max(spreads)
    return {"all_pair_l2": spreads, "pair_count": 66,
            "observed_max_l2": observed, "resolution_floor": max(3 * observed, 1e-6)}


def derivative_predictions(grads: torch.Tensor, effective: torch.Tensor,
                           delta: torch.Tensor, alpha: float) -> dict[str, Any]:
    """Project saved row gradients onto realized and ideal uniform source edits."""
    need(grads.ndim == 3 and grads.shape[0] == 4 and
         effective.ndim == 2 and tuple(grads.shape[1:]) == tuple(effective.shape),
         "Gradient/edit row shape differs")
    need(delta.ndim == 1 and delta.shape[0] == effective.shape[1] and
         alpha in ALPHAS and math.isfinite(alpha), "Ideal edit descriptor differs")
    finite_tensor(grads, tuple(grads.shape), torch.float32, "row gradients")
    finite_tensor(effective, tuple(effective.shape), torch.float32, "realized rows")
    finite_tensor(delta, tuple(delta.shape), torch.float32, "donor direction")
    g = grads.to(torch.float64)
    e = effective.to(torch.float64)
    d = delta.to(torch.float64)
    summed = g.sum(dim=1)
    rowwise = torch.einsum("krd,rd->k", g, e)
    uniform = summed @ e.mean(dim=0)
    ideal = summed @ (alpha * d)
    return {"rowwise": rowwise, "uniform": uniform, "ideal": ideal}


def edit_fidelity(effective: torch.Tensor, delta: torch.Tensor,
                  alpha: float) -> dict[str, Any]:
    """The predeclared 10% mean-fidelity and row-nonuniformity checks."""
    need(effective.ndim == 2 and delta.ndim == 1 and
         effective.shape[1] == delta.shape[0] and alpha in ALPHAS,
         "Realized/ideal edit dimensions invalid")
    finite_tensor(effective, tuple(effective.shape), torch.float32, "effective")
    finite_tensor(delta, tuple(delta.shape), torch.float32, "delta")
    e = effective.to(torch.float64)
    intended = alpha * delta.to(torch.float64)
    mean = e.mean(dim=0)
    intended_norm = float(torch.linalg.vector_norm(intended).item())
    effective_norm = float(torch.linalg.vector_norm(e).item())
    mean_norm = float(torch.linalg.vector_norm(mean).item())
    fidelity = float(torch.linalg.vector_norm(mean - intended).item()) / max(intended_norm, 1e-6)
    row_deviation = torch.sqrt(((e - mean).square().sum(dim=1)).mean())
    nonuniformity = float(row_deviation.item()) / max(mean_norm, 1e-6)
    valid = (intended_norm > 0 and effective_norm > 0 and mean_norm > 0 and
             fidelity <= 0.10 and nonuniformity <= 0.10)
    return {"intended_norm": intended_norm, "effective_norm": effective_norm,
            "mean_norm": mean_norm, "mean_fidelity_ratio": fidelity,
            "row_nonuniformity_ratio": nonuniformity, "valid": valid}


def measured_effect(basis: torch.Tensor, zero: torch.Tensor,
                    edited: torch.Tensor) -> dict[str, Any]:
    finite_tensor(zero, (basis.shape[0],), torch.float32, "zero target")
    finite_tensor(edited, (basis.shape[0],), torch.float32, "edited target")
    full = edited.to(torch.float64) - zero.to(torch.float64)
    return {"full_l2": float(torch.linalg.vector_norm(full).item()),
            "projected": basis.T @ full}


def relative_error(prediction: torch.Tensor, measured: torch.Tensor,
                   observed_drift: float) -> float:
    need(prediction.shape == measured.shape and prediction.ndim == 1 and
         math.isfinite(observed_drift) and observed_drift >= 0,
         "Relative-error operands invalid")
    need(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(measured).all()),
         "Relative-error operands nonfinite")
    numerator = float(torch.linalg.vector_norm(prediction - measured).item())
    denominator = max(float(torch.linalg.vector_norm(measured).item()), observed_drift, 1e-6)
    return numerator / denominator


def site_gate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """All 24 planned cells per site remain in the denominator."""
    need(len(cells) == 24, "Site must contain exactly 24 planned cells")
    for cell in cells:
        need(isinstance(cell, dict) and type(cell.get("valid")) is bool and
             type(cell.get("resolved")) is bool,
             "Cell validity/resolution fields invalid")
        if cell["valid"]:
            error = cell.get("primary_max_cycle_error")
            need(type(error) in (float, int) and math.isfinite(error) and error >= 0,
                 "Valid cell error invalid")
        else:
            need(not cell["resolved"] and cell.get("primary_max_cycle_error") is None,
                 "Invalid cell must not receive an error or resolved label")
    valid = [cell for cell in cells if cell.get("valid") is True]
    resolved = [cell for cell in valid if cell.get("resolved") is True]
    errors = [float(cell["primary_max_cycle_error"]) for cell in resolved]
    need(all(math.isfinite(x) and x >= 0 for x in errors), "Resolved cell error invalid")
    median = None
    if len(resolved) >= 18:
        ordered = sorted(errors)
        middle = len(ordered) // 2
        median = (ordered[middle] if len(ordered) % 2 else
                  (ordered[middle - 1] + ordered[middle]) / 2)
    high_error_count = sum(cell["valid"] and cell["primary_max_cycle_error"] > 0.50
                           for cell in cells)
    passed = (len(valid) == 24 and len(resolved) >= 18 and
              median is not None and median <= 0.25 and high_error_count <= 4)
    return {"planned": 24, "valid": len(valid), "resolved": len(resolved),
            "median_resolved_max_cycle_error": median,
            "max_cycle_error_gt_0_50_count": high_error_count, "pass": passed}


def aggregate_campaign(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Four independently audited contexts become the fixed 24 cells/site."""
    need(isinstance(reports, list) and len(reports) == 4,
         "Exactly four audit reports required")
    indexed: dict[int, dict[str, Any]] = {}
    for report in reports:
        need(isinstance(report, dict) and report.get("status") == AUDIT_STATUS and
             report.get("scope") == SCOPE and isinstance(report.get("context"), dict),
             "Context audit status/scope invalid")
        context = report["context"]
        index = context.get("index")
        need(type(index) is int and 0 <= index < 4 and index not in indexed and
             context.get("label") == CONTEXT_LABELS[index],
             "Campaign context index/label missing or duplicate")
        indexed[index] = report
    gate = {}
    for site in SITES:
        cells = []
        for index in range(4):
            rows = indexed[index]["sites"][site]["cells"]
            need(isinstance(rows, list) and len(rows) == 6 and
                 [row.get("alpha") for row in rows] == list(ALPHAS) and
                 all(row.get("context_index") == index and row.get("site") == site
                     for row in rows),
                 "Campaign site/alpha cells differ")
            cells.extend(rows)
        gate[site] = site_gate(cells)
    passed = all(gate[site]["pass"] for site in SITES)
    return {"status": "audited_voxtral_fp32_nonbeaver_campaign_saved_data_only",
            "planned_contexts": 4, "planned_cells": 48,
            "site_gate": gate, "advance_to_fresh_corpus_feasibility_gate": passed,
            "interpretation_limit": "FIT-only numerical method gate; no reusable lens or security result."}


def json_object(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            need(key not in result, f"{path.name}: duplicate JSON key {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(),
                       parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)),
                       object_pairs_hook=unique_pairs)
    need(isinstance(value, dict), f"{path.name}: JSON object required")
    return value


def frozen_rows(engine: str) -> dict[str, list[int]]:
    need(file_sha(MASKS_PATH) == MASKS_SHA256, "Frozen FIT mask bytes differ")
    manifest = json_object(MASKS_PATH)
    matches = [row for row in manifest.get("records", [])
               if row.get("pair_id") == "color_05" and
               row.get("phrasing_id") == "fit_chosen" and
               row.get("engine") == engine]
    need(len(matches) == 1, "Frozen color_05 FIT mask record differs")
    rows = {site: matches[0]["site_maps"][site]["whole_word_rows"]
            for site in SITES}
    production_spec(engine, rows)
    return rows


def _invalidity_set(rows: list[dict[str, Any]]) -> set[tuple]:
    result = set()
    for row in rows:
        need(isinstance(row, dict) and set(row) ==
             {"stage", "descriptor", "failed_checks"} and
             isinstance(row["descriptor"], dict) and
             isinstance(row["failed_checks"], list) and
             bool(row["failed_checks"]) and
             all(isinstance(item, str) for item in row["failed_checks"]),
             "Worker invalidity descriptor malformed")
        descriptor = row["descriptor"]
        stage = row["stage"]
        if stage == "paired_forward":
            need(set(descriptor) ==
                 {"ordinal", "site", "mode", "cycle", "alpha", "label"},
                 "Worker invalidity forward descriptor malformed")
        elif stage == "passive_recipient_to_n0_clean":
            need(set(descriptor) == {"site"} and descriptor["site"] in SITES and
                 row["failed_checks"] == ["full_prepatch_source_bitwise"],
                 "Worker passive-recipient parity descriptor malformed")
        elif stage == "site_clean_zero_identity":
            need(set(descriptor) == {"site", "mode"} and
                 descriptor["site"] in SITES and descriptor["mode"] in MODES and
                 set(row["failed_checks"]).issubset(
                     {"zero_source_bitwise", "zero_effective_all_zero",
                      "zero_metrics_exact", "runtime_selected_edit"}),
                 "Worker clean-zero identity descriptor malformed")
        else:
            need(False, "Worker invalidity stage differs")
        result.add((stage, tuple(sorted(descriptor.items())),
                    tuple(sorted(row["failed_checks"]))))
    need(len(result) == len(rows), "Duplicate worker invalidity descriptor")
    return result


def audit_run(run_dir: Path, output: Path) -> dict[str, Any]:
    """Read-only saved-data verification plus an exclusive CPU audit receipt."""
    need(run_dir.is_dir() and not run_dir.is_symlink() and
         output.parent.resolve() == run_dir.resolve() and not output.exists(),
         "Audit output must be new and directly inside attempt directory")
    need(file_sha(PROTOCOL_V2_PATH) == PROTOCOL_V2_SHA256 and
         file_sha(PROTOCOL_V3_PATH) == PROTOCOL_V3_SHA256 and
         file_sha(PROTOCOL_V4_PATH) == PROTOCOL_V4_SHA256 and
         file_sha(PROTOCOL_V5_PATH) == PROTOCOL_V5_SHA256 and
         file_sha(SELECTION_PATH) == SELECTION_SHA256 and
         file_sha(MASKS_PATH) == MASKS_SHA256 and
         file_sha(PREFLIGHT_PATH) == PREFLIGHT_SHA256 and
         file_sha(BYTE_BUDGET_PATH) == BYTE_BUDGET_SHA256 and
         file_sha(RUNNER_PATH) == RUNNER_SHA256,
         "Reviewed protocol or CPU preflight receipt differs")
    total = 0
    for path in run_dir.rglob("*"):
        need(not path.is_symlink(), "Attempt contains symlink")
        if path.is_file():
            stat = path.stat()
            need(stat.st_nlink == 1, "Attempt contains hard-linked file")
            total += stat.st_size
            need(total < MAX_WORKER_BYTES, "Saved worker output cap reached")
    meta = json_object(run_dir / "meta.json")
    result = json_object(run_dir / "result.json")
    context = result.get("context")
    need(isinstance(context, dict) and type(context.get("index")) is int and
         0 <= context["index"] < 4 and
         context.get("label") == CONTEXT_LABELS[context["index"]] and
         result.get("status") == RESULT_STATUS and result.get("scope") == SCOPE and
         meta.get("schema") == "voxtral_fp32_nonbeaver_linearity_meta_v1" and
         meta.get("status") == "started_source_only_reviewed_context" and
         meta.get("scope") == SCOPE and meta.get("context") == context,
         "Saved result/meta context or status invalid")
    need(result.get("planned_counts") == COUNTS and
         result.get("actual_counts") == COUNTS and
         meta.get("planned_counts") == COUNTS,
         "Saved planned/actual model-work counts differ")
    attempt_id, freeze_sha = result.get("attempt_id"), result.get("freeze_sha256")
    need(isinstance(attempt_id, str) and bool(attempt_id) and
         isinstance(freeze_sha, str) and HEX64.fullmatch(freeze_sha) and
         meta.get("attempt_id") == attempt_id and
         meta.get("freeze_sha256") == freeze_sha,
         "Attempt/freeze binding differs")
    need(meta.get("patched_schedule") == [list(row) for row in patched_schedule(context["engine"])],
         "Saved fixed patched schedule differs")
    for key, expected in (("protocol_v2_sha256", PROTOCOL_V2_SHA256),
                          ("protocol_v3_sha256", PROTOCOL_V3_SHA256),
                          ("protocol_v4_sha256", PROTOCOL_V4_SHA256),
                          ("protocol_v5_sha256", PROTOCOL_V5_SHA256)):
        need(result.get(key) == expected and meta.get(key) == expected,
             f"{key} differs")
    need(result.get("preflight_receipt_sha256") == file_sha(PREFLIGHT_PATH) and
         meta.get("preflight_receipt_sha256") == file_sha(PREFLIGHT_PATH) and
         meta.get("selection_sha256") == SELECTION_SHA256 and
         meta.get("masks_sha256") == MASKS_SHA256,
         "Frozen preflight/selection/mask pin differs")
    need(meta.get("runner_sha256") == file_sha(RUNNER_PATH) and
         meta.get("source_dependencies") ==
         {key: file_sha(path) for key, path in SOURCE_DEPENDENCY_FILES.items()} and
         meta.get("basis_sha256") == BASIS_SHA256 and
         meta.get("wav_sha256") ==
         {word: WAV_SHA256[(context["engine"], word)]
          for word in (context["recipient"], context["donor"])},
         "Runner/dependency/basis/WAV provenance differs")
    for key, name in (("meta_sha256", "meta.json"),
                      ("progress_sha256", "progress.jsonl"),
                      ("model_layout_sha256", "model_layout.json")):
        need(result.get(key) == file_sha(run_dir / name),
             f"{name} receipt hash differs")
    layout = json_object(run_dir / "model_layout.json")
    need(layout.get("valid") is True and layout.get("arm") == "fp32" and
         meta.get("device") == "mps" and
         meta.get("fresh_v2_corpus_opened") is False,
         "Model layout/device/corpus boundary differs")
    need(meta.get("model_seconds_cap") == MAX_SECONDS and
         meta.get("max_mps_driver_bytes") == MAX_MPS_BYTES and
         meta.get("max_worker_bytes") == MAX_WORKER_BYTES and
         meta.get("max_output_bytes") == 2 * 1024**3,
         "Frozen resource caps differ")
    elapsed = result.get("elapsed_seconds")
    peak = result.get("peak_mps_driver_bytes")
    need(type(elapsed) in (float, int) and math.isfinite(elapsed) and
         0 <= elapsed < MAX_SECONDS and type(peak) is int and
         0 <= peak < MAX_MPS_BYTES,
         "Observed process/MPS cap reached")
    need(result.get("output_bytes_before_result") ==
         total - (run_dir / "result.json").stat().st_size,
         "Worker output byte count differs")
    detail_sha, detail_rel = result.get("details_sha256"), result.get("details_path")
    need(isinstance(detail_sha, str) and HEX64.fullmatch(detail_sha) and
         isinstance(detail_rel, str) and detail_rel == f"blobs/{detail_sha}.pt",
         "Content-addressed tensor sidecar name differs")
    detail_path = run_dir / detail_rel
    need(detail_path.is_file() and not detail_path.is_symlink() and
         result.get("details_bytes") == detail_path.stat().st_size and
         detail_path.stat().st_size < MAX_WORKER_BYTES and
         file_sha(detail_path) == detail_sha,
         "Content-addressed tensor sidecar differs")
    rows = frozen_rows(context["engine"])
    spec = production_spec(context["engine"], rows)
    payload = torch.load(detail_path, map_location="cpu",
                         weights_only=True, mmap=True)
    report = audit_payload(payload, spec)
    need(report["context"] == context, "Tensor sidecar context differs")
    need(meta.get("rows") == rows and
         meta.get("batch_sha256") == report["batch_sha256"],
         "Saved mask/batch provenance differs")
    worker_invalidities = _invalidity_set(result.get("invalidities", []))
    auditor_invalidities = _invalidity_set(report["computed_invalidities"])
    need(worker_invalidities == auditor_invalidities,
         "Worker invalidity claims differ from independently retained evidence")
    report.update({"worker_invalidity_disagreement": False,
                   "result_sha256": file_sha(run_dir / "result.json"),
                   "meta_sha256": file_sha(run_dir / "meta.json"),
                   "details_sha256": detail_sha,
                   "preflight_receipt_sha256": file_sha(PREFLIGHT_PATH),
                   "protocol_v2_sha256": PROTOCOL_V2_SHA256,
                   "protocol_v3_sha256": PROTOCOL_V3_SHA256,
                   "protocol_v4_sha256": PROTOCOL_V4_SHA256,
                   "protocol_v5_sha256": PROTOCOL_V5_SHA256,
                   "attempt_id": result.get("attempt_id"),
                   "freeze_sha256": result.get("freeze_sha256"),
                   "actual_attempt_bytes_before_audit": total,
                   "elapsed_seconds": elapsed,
                   "peak_mps_driver_bytes": peak})
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2,
                                allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run_dir, args.output)
    print(json.dumps({"status": report["status"],
                      "audit_sha256": file_sha(args.output)}, allow_nan=False))


if __name__ == "__main__":
    main()
