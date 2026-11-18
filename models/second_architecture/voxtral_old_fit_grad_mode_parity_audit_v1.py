"""CPU-only audit of one saved Voxtral FP32 graph/inference parity trace.

This verifies byte provenance, tensor consistency, and saved-tensor arithmetic.
It cannot attest that a forward call occurred. A separate launcher binds source
review, execution order, and the one-shot model process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
STATUS = "completed_old_fit_fp32_graph_inference_parity_diagnostic_only"
SCOPE = "one_old_fit_case_forward_parity_only_no_vjp_no_security_claim"
SCHEMA = "voxtral_old_fit_grad_mode_parity_tensors_v1"
AUDIT_STATUS = "audited_old_fit_fp32_graph_inference_parity_saved_data_only"
STAGES = ("lm0", "lm7", "lm14", "lm22", "lm29")
FULL_ORDERS = {"encoder31": ("N", "G", "G", "N"),
               "projector": ("G", "N", "N", "G")}
DIRECT_ORDER = ("N", "G", "G", "N")
COUNTS = {"full_forwards": 8, "direct_forwards": 16, "vjps": 0, "generations": 0}
SOURCE_SHA256 = {
    "encoder31": "e0abb561ad91255d1945c81c1dead1a75266f09d0e7f3695a123040fef47b2b6",
    "projector": "a96b5e19ffefe48e537b30ddecb9de11c8c40becf4b4dd3f2813ab5583fce100",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ShapeSpec:
    source: dict[str, tuple[int, ...]]
    projector: tuple[int, ...]
    lm_input: tuple[int, ...]
    lm_width: int
    logits: int
    input_features: tuple[int, ...]
    audio_count: int
    audio_token_id: int
    source_sha256: dict[str, str]
    batch_sha256: dict[str, str]


REAL = ShapeSpec(
    source={"encoder31": (1, 1500, 1280), "projector": (375, 3072)},
    projector=(375, 3072), lm_input=(1, 396, 3072), lm_width=3072,
    logits=131072, input_features=(1, 128, 3000), audio_count=375,
    audio_token_id=24, source_sha256=SOURCE_SHA256,
    # Independently reconstructed from native.preflight_cpu()["batches"]["beaver"]
    # on the pinned old-FIT inputs. The exact field set has no position_ids.
    batch_sha256={
        "input_ids": "915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a",
        "attention_mask": "9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282",
        "input_features": "4f23ae2441eca7587b7acf19d89f717292a4f1efb3067631c7abb7e27236ee61",
    },
)
MAX_ATTEMPT_BYTES = 1024**3
MAX_WORKER_BYTES = MAX_ATTEMPT_BYTES - 24 * 1024**2
MAX_SIDECAR_BYTES = MAX_WORKER_BYTES - 1_048_576


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    """Independent implementation of the frozen tensor-byte hash."""
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    need(isinstance(value, dict), f"{path.name} must be an object")
    return value


def float_tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype,
                 label: str) -> torch.Tensor:
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
         tuple(value.shape) == shape and value.dtype == dtype and
         value.is_contiguous() and bool(torch.isfinite(value).all()),
         f"{label}: shape/dtype/device/contiguity/finiteness invalid")
    return value


def integer_tensor(value: Any, shape: tuple[int, ...], label: str) -> torch.Tensor:
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
         tuple(value.shape) == shape and value.dtype in (torch.int32, torch.int64) and
         value.is_contiguous(), f"{label}: integer tensor invalid")
    return value


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    need(a.shape == b.shape and a.dtype == b.dtype, "Bitwise comparison tensor type differs")
    return bool(torch.equal(a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8)))


def difference(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    need(a.shape == b.shape and a.dtype == b.dtype, "Difference tensor type differs")
    left, right = a.to(torch.float64), b.to(torch.float64)
    delta = left - right
    l2 = float(torch.linalg.vector_norm(delta).item())
    denominator = max(float(torch.linalg.vector_norm(left).item()),
                      float(torch.linalg.vector_norm(right).item()))
    return {"bitwise_equal": bitwise_equal(a, b),
            "l2": l2,
            "relative_l2": l2 / denominator if denominator > 0 else None,
            "max_abs": float(delta.abs().max().item()),
            "changed_coordinates": int(torch.count_nonzero(delta).item())}


def _full_schedule() -> list[tuple[str, str, int]]:
    rows = []
    for site in ("encoder31", "projector"):
        seen = {"N": 0, "G": 0}
        for mode in FULL_ORDERS[site]:
            rows.append((site, mode, seen[mode]))
            seen[mode] += 1
    return rows


def _direct_schedule() -> list[tuple[str, str, str, int]]:
    rows = []
    for site in ("encoder31", "projector"):
        for origin in ("N", "G"):
            seen = {"N": 0, "G": 0}
            for mode in DIRECT_ORDER:
                rows.append((site, origin, mode, seen[mode]))
                seen[mode] += 1
    return rows


def _stages(record: dict[str, Any], spec: ShapeSpec, label: str) -> None:
    terminal = record.get("lm_terminal")
    need(isinstance(terminal, dict) and set(terminal) == set(STAGES),
         f"{label}: LM terminal names invalid")
    for stage in STAGES:
        float_tensor(terminal[stage], (spec.lm_width,), torch.float32,
                     f"{label} {stage}")
    float_tensor(record.get("first_answer_logits"), (spec.logits,), torch.float32,
                 f"{label} first logits")


def _boundaries(record: dict[str, Any], *, kind: str) -> dict[str, torch.Tensor]:
    fields: list[tuple[str, torch.Tensor]] = []
    if kind == "full":
        fields += [("source_post", record["source_post_full"]),
                   ("projector", record["projector_full"])]
    fields += [("lm_input", record["lm_input_full"]),
               ("lm0_input", record["lm0_input_full"])]
    fields += [(stage, record["lm_terminal"][stage]) for stage in STAGES]
    fields += [("first_answer_logits", record["first_answer_logits"])]
    return dict(fields)


def _retained_tensor_hashes(payload: dict[str, Any]) -> dict[str, Any]:
    """Rehash every retained tensor; blob SHA is the independent byte pin."""
    return {
        "sources_bf16": {site: tensor_sha(value)
                         for site, value in payload["sources_bf16"].items()},
        "batch": {name: tensor_sha(value) for name, value in payload["batch"].items()},
        "calls": [{"ordinal": row["ordinal"],
                   "boundaries": {name: tensor_sha(value)
                                  for name, value in _boundaries(row, kind=row["kind"]).items()}}
                  for row in payload["calls"]],
    }


def _pair_summary(records: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    by_mode = {mode: [row for row in records if row["mode"] == mode]
               for mode in ("N", "G")}
    need(all(len(by_mode[mode]) == 2 for mode in ("N", "G")),
         "Mode repeat count invalid")
    first_bitwise = None
    first_resolved = None
    boundaries = {}
    for name in _boundaries(records[0], kind=kind):
        n = [_boundaries(row, kind=kind)[name] for row in by_mode["N"]]
        g = [_boundaries(row, kind=kind)[name] for row in by_mode["G"]]
        within_n = difference(*n)
        within_g = difference(*g)
        crosses = [difference(a, b) for a in n for b in g]
        floor = max(within_n["l2"], within_g["l2"])
        threshold = max(3.0 * floor, 1e-6)
        all_cross_bitwise_equal = all(row["bitwise_equal"] for row in crosses)
        resolved = min(row["l2"] for row in crosses) > threshold
        if first_bitwise is None and not all_cross_bitwise_equal:
            first_bitwise = name
        if first_resolved is None and resolved:
            first_resolved = name
        boundaries[name] = {
            "within_N": within_n, "within_G": within_g,
            "cross_NG": [{"N_repeat": ni, "G_repeat": gi, **crosses[2 * ni + gi]}
                         for ni in range(2) for gi in range(2)],
            "cross_NG_l2_min": min(row["l2"] for row in crosses),
            "cross_NG_l2_max": max(row["l2"] for row in crosses),
            "cross_NG_all_bitwise_equal": all_cross_bitwise_equal,
            "within_mode_bitwise_stable":
                within_n["bitwise_equal"] and within_g["bitwise_equal"],
            "within_mode_l2_floor": floor, "resolved_threshold_l2": threshold,
            "resolved_beyond_observed_repeat_floor": resolved,
        }
    stable = all(row["within_mode_bitwise_stable"] for row in boundaries.values())
    fixed_direct_input = (kind != "direct" or
                          all(boundaries[name]["cross_NG_all_bitwise_equal"] and
                              boundaries[name]["within_mode_bitwise_stable"]
                              for name in ("lm_input", "lm0_input")))
    return {"first_bitwise_difference": first_bitwise,
            "first_resolved_difference": first_resolved,
            "all_within_mode_bitwise_stable": stable,
            "fixed_direct_input_bitwise_equal": fixed_direct_input,
            "deterministic_mode_contrast_supported":
                stable and fixed_direct_input and first_resolved is not None,
            "boundaries": boundaries}


def _lm_input_partition_summary(rows: list[dict[str, Any]],
                                positions: list[int]) -> dict[str, Any]:
    n = [row["lm_input_full"][0, positions] for row in rows if row["mode"] == "N"]
    g = [row["lm_input_full"][0, positions] for row in rows if row["mode"] == "G"]
    within_n, within_g = difference(*n), difference(*g)
    crosses = [difference(a, b) for a in n for b in g]
    floor = max(within_n["l2"], within_g["l2"])
    threshold = max(3.0 * floor, 1e-6)
    return {"within_N": within_n, "within_G": within_g,
            "cross_NG": [{"N_repeat": ni, "G_repeat": gi, **crosses[2 * ni + gi]}
                         for ni in range(2) for gi in range(2)],
            "within_mode_l2_floor": floor,
            "resolved_threshold_l2": threshold,
            "resolved_beyond_observed_repeat_floor":
                min(row["l2"] for row in crosses) > threshold}


def audit_payload(payload: Any, *, spec: ShapeSpec = REAL) -> dict[str, Any]:
    """Check all saved tensor relationships, then summarize paired differences."""
    need(isinstance(payload, dict) and
         set(payload) == {"schema", "sources_bf16", "batch", "audio_token_id", "calls"} and
         payload["schema"] == SCHEMA, "Tensor payload schema invalid")
    sources = payload["sources_bf16"]
    need(isinstance(sources, dict) and set(sources) == set(spec.source),
         "BF16 source map invalid")
    for site, shape in spec.source.items():
        source = float_tensor(sources[site], shape, torch.bfloat16, f"{site} source")
        need(tensor_sha(source) == spec.source_sha256[site],
             f"{site} full BF16 source hash differs")
    need(payload["audio_token_id"] == spec.audio_token_id,
         "Audio token ID differs")
    batch = payload["batch"]
    need(isinstance(batch, dict) and set(batch) == set(spec.batch_sha256),
         "Frozen batch field set invalid")
    prompt_len = spec.lm_input[1]
    ids = integer_tensor(batch["input_ids"], (1, prompt_len), "input_ids")
    mask = integer_tensor(batch["attention_mask"], (1, prompt_len), "attention_mask")
    need(bool(((mask == 0) | (mask == 1)).all()), "attention_mask is not binary")
    float_tensor(batch["input_features"], spec.input_features, torch.float32,
                 "input_features")
    if "position_ids" in batch:
        integer_tensor(batch["position_ids"], (1, prompt_len), "position_ids")
    for name, expected_sha in spec.batch_sha256.items():
        need(tensor_sha(batch[name]) == expected_sha,
             f"Frozen batch tensor hash differs: {name}")
    audio_positions = torch.nonzero(ids[0] == spec.audio_token_id).flatten().tolist()
    need(len(audio_positions) == spec.audio_count and
         len(set(audio_positions)) == spec.audio_count,
         "Audio placeholders/count invalid")
    other_positions = [i for i in range(prompt_len) if i not in set(audio_positions)]
    calls = payload["calls"]
    need(isinstance(calls, list) and len(calls) == 24,
         "Exactly 24 saved calls required")
    full_keys = {"kind", "ordinal", "site", "mode", "repeat", "source_sha256",
                 "source_post_full", "projector_full", "lm_input_full",
                 "lm0_input_full", "lm_terminal", "first_answer_logits",
                 "audio_placeholder_positions"}
    direct_keys = {"kind", "ordinal", "site", "origin_mode",
                   "origin_full_ordinal", "mode", "repeat", "input_sha256",
                   "lm_input_full", "lm0_input_full", "lm_terminal",
                   "first_answer_logits"}
    full_groups = {site: [] for site in spec.source}
    for ordinal, (site, mode, repeat) in enumerate(_full_schedule()):
        row = calls[ordinal]
        label = f"full[{ordinal}]"
        need(isinstance(row, dict) and set(row) == full_keys and
             (row["kind"], row["ordinal"], row["site"], row["mode"], row["repeat"])
             == ("full", ordinal, site, mode, repeat), f"{label}: schedule/schema invalid")
        need(row["source_sha256"] == spec.source_sha256[site] and
             row["audio_placeholder_positions"] == audio_positions,
             f"{label}: source hash or placeholder positions invalid")
        source_post = float_tensor(row["source_post_full"], spec.source[site],
                                   torch.float32, f"{label} source posthook")
        need(bitwise_equal(source_post, sources[site].float()),
             f"{label}: posthook source does not equal frozen BF16 source")
        proj = float_tensor(row["projector_full"], spec.projector,
                            torch.float32, f"{label} projector")
        lm_input = float_tensor(row["lm_input_full"], spec.lm_input,
                                torch.float32, f"{label} LM input")
        lm0_input = float_tensor(row["lm0_input_full"], spec.lm_input,
                                 torch.float32, f"{label} LM0 pre-block")
        need(bitwise_equal(lm0_input, lm_input),
             f"{label}: LM0 pre-block differs from captured LM input")
        need(bitwise_equal(lm_input[0, audio_positions], proj),
             f"{label}: projector differs from LM audio placeholders")
        if site == "projector":
            need(bitwise_equal(source_post, proj),
                 f"{label}: projector posthook differs from projector capture")
        _stages(row, spec, label)
        full_groups[site].append(row)
    full_origins = {}
    for site, rows in full_groups.items():
        for origin in ("N", "G"):
            full_origins[(site, origin)] = next(row for row in rows if row["mode"] == origin)
    direct_groups: dict[tuple[str, str], list[dict[str, Any]]] = {
        (site, origin): [] for site in spec.source for origin in ("N", "G")}
    for ordinal, (site, origin, mode, repeat) in enumerate(_direct_schedule(), start=8):
        row = calls[ordinal]
        label = f"direct[{ordinal}]"
        need(isinstance(row, dict) and set(row) == direct_keys and
             (row["kind"], row["ordinal"], row["site"], row["origin_mode"],
              row["mode"], row["repeat"])
             == ("direct", ordinal, site, origin, mode, repeat),
             f"{label}: schedule/schema invalid")
        origin_row = full_origins[(site, origin)]
        origin_input = origin_row["lm_input_full"]
        need(row["origin_full_ordinal"] == origin_row["ordinal"] and
             row["input_sha256"] == tensor_sha(origin_input),
             f"{label}: origin descriptor/hash invalid")
        direct_input = float_tensor(row["lm_input_full"], spec.lm_input,
                                    torch.float32, f"{label} direct LM input")
        direct_lm0 = float_tensor(row["lm0_input_full"], spec.lm_input,
                                  torch.float32, f"{label} direct LM0 pre-block")
        need(bitwise_equal(direct_input, origin_input) and
             bitwise_equal(direct_lm0, origin_input),
             f"{label}: direct input differs from fixed full-model origin")
        _stages(row, spec, label)
        direct_groups[(site, origin)].append(row)
    full_metrics = {site: _pair_summary(rows, kind="full")
                    for site, rows in full_groups.items()}
    direct_metrics = {f"{site}_{origin}": _pair_summary(rows, kind="direct")
                      for (site, origin), rows in direct_groups.items()}
    # Audio and non-audio subspaces are reported, never treated as equivalent.
    modality = {}
    for site, rows in full_groups.items():
        n = next(row for row in rows if row["mode"] == "N")
        g = next(row for row in rows if row["mode"] == "G")
        modality[site] = {
            "audio_lm_input_N_vs_G": difference(n["lm_input_full"][0, audio_positions],
                                                  g["lm_input_full"][0, audio_positions]),
            "non_audio_lm_input_N_vs_G": difference(n["lm_input_full"][0, other_positions],
                                                      g["lm_input_full"][0, other_positions]),
            "audio_lm_input_all_pairs": _lm_input_partition_summary(rows, audio_positions),
            "non_audio_lm_input_all_pairs": _lm_input_partition_summary(rows, other_positions),
        }
    return {"status": AUDIT_STATUS, "scope": SCOPE, "saved_call_counts": COUNTS,
            "audio_placeholder_count": spec.audio_count,
            "non_audio_token_count": len(other_positions),
            "retained_tensor_sha256": _retained_tensor_hashes(payload),
            "full_model": full_metrics, "direct_lm": direct_metrics,
            "modality_partition": modality,
            "interpretation_limit":
                "Two repeats per mode define only an observed floor; graph mode also changes requires_grad. "
                "Relative L2 uses max(left L2, right L2) as denominator and is null for two zero tensors. "
                "Any within-mode bitwise instability bars a deterministic mode conclusion. "
                "Saved tensors do not independently attest that forwards occurred."}


def audit_run(run_dir: Path, output: Path, *, spec: ShapeSpec = REAL) -> dict[str, Any]:
    """Read a completed immutable attempt and write one exclusive audit receipt."""
    need(output.parent.resolve() == run_dir.resolve() and not output.exists(),
         "Audit output must be a new file directly in the attempt directory")
    need(run_dir.is_dir() and not run_dir.is_symlink(), "Attempt directory invalid")
    # These are actual disk bytes, not result.json's self-reported byte count.
    attempt_bytes = 0
    for path in run_dir.rglob("*"):
        need(not path.is_symlink(), "Attempt contains a symlink")
        if path.is_file():
            attempt_bytes += path.stat().st_size
            need(attempt_bytes < MAX_WORKER_BYTES,
                 "Worker attempt directory exceeds frozen output cap")
    result = json_object(run_dir / "result.json")
    meta = json_object(run_dir / "meta.json")
    need(result.get("status") == STATUS and result.get("scope") == SCOPE and
         meta.get("scope") == SCOPE and
         meta.get("status") == "started_old_fit_fp32_graph_inference_parity_diagnostic_only",
         "Result/meta status or scope invalid")
    need(result.get("planned_counts") == COUNTS and
         result.get("actual_counts") == COUNTS and
         meta.get("planned_counts") == COUNTS,
         "Frozen forward/VJP/generation counts invalid")
    need(result.get("source_sha256") == spec.source_sha256 and
         meta.get("source_sha256") == spec.source_sha256 and
         result.get("audio_token_id") == spec.audio_token_id,
         "Frozen source or audio token IDs invalid")
    need(meta.get("full_schedule") == [list(row) for row in _full_schedule()] and
         meta.get("direct_schedule") == [list(row) for row in _direct_schedule()],
         "Frozen meta schedule differs")
    for key, filename in (("meta_sha256", "meta.json"),
                          ("model_layout_sha256", "model_layout.json"),
                          ("progress_sha256", "progress.jsonl")):
        need(result.get(key) == sha(run_dir / filename), f"{filename} SHA differs")
    layout = json_object(run_dir / "model_layout.json")
    need(layout.get("valid") is True, "Model layout did not pass")
    elapsed = result.get("elapsed_seconds")
    peak = result.get("peak_mps_driver_bytes")
    need(type(elapsed) in (float, int) and 0 <= elapsed < 180 and
         type(peak) is int and 0 <= peak < 30 * 1024**3,
         "Saved process time or MPS peak exceeds diagnostic cap")
    need(result.get("protocol_sha256") == meta.get("protocol_sha256") and
         result.get("runner_sha256") == meta.get("runner_sha256") and
         result.get("failed_stage_b_fp32_manifest_sha256") ==
         meta.get("failed_stage_b_fp32_manifest_sha256"),
         "Protocol/runner/failed-attempt provenance differs")
    details_sha = result.get("details_sha256")
    details_rel = result.get("details_path")
    need(isinstance(details_sha, str) and HEX64.fullmatch(details_sha) is not None and
         details_rel == f"blobs/{details_sha}.pt", "Details content-addressed path invalid")
    details = run_dir / details_rel
    need(not (run_dir / "blobs").is_symlink() and details.is_file() and
         not details.is_symlink(), "Details sidecar path invalid")
    need(details.stat().st_size < MAX_SIDECAR_BYTES,
         "Details sidecar exceeds frozen byte cap")
    need(
         details.stat().st_size == result.get("details_bytes") and
         sha(details) == details_sha, "Details sidecar bytes differ")
    payload = torch.load(details, map_location="cpu", weights_only=True, mmap=True)
    report = audit_payload(payload, spec=spec)
    report.update({"result_sha256": sha(run_dir / "result.json"),
                   "meta_sha256": sha(run_dir / "meta.json"),
                   "details_sha256": details_sha,
                   "actual_attempt_bytes_before_audit": attempt_bytes,
                   "max_attempt_bytes": MAX_ATTEMPT_BYTES,
                   "max_worker_bytes": MAX_WORKER_BYTES,
                   "max_sidecar_bytes": MAX_SIDECAR_BYTES,
                   "protocol_sha256": result["protocol_sha256"],
                   "runner_sha256": result["runner_sha256"],
                   "failed_stage_b_fp32_manifest_sha256":
                       result["failed_stage_b_fp32_manifest_sha256"]})
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.run_dir, args.output)
    print(json.dumps({"status": report["status"], "audit_sha256": sha(args.output)},
                     allow_nan=False))


if __name__ == "__main__":
    main()
