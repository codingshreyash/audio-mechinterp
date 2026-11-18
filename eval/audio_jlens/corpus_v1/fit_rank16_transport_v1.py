"""FIT-only rank-16 corpus-averaged Qwen audio-to-terminal transport.

Stage 1: capture 48 preselected FIT prompt-last states and six source-bin
means, then derive a 16D target basis from 24 pair/voice differences.
Stage 2: reuse one forward graph per FIT prompt for 16 VJPs to both encoder31
and connector, averaging their six-bin gradients at the *pair* level.

No development/test audio or answers are prepared. This fits a map but does
not evaluate its readout, interventions, or security relevance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL_DIR = ROOT / "eval" / "audio_jlens"
sys.path[:0] = [str(ROOT / "models" / "qwen_omni"),
                str(ROOT / "models" / "qwen_omni" / "time_map"),
                str(EVAL_DIR)]
from adapter import MODEL_ID, MODEL_REVISION, TRANSFORMERS_REVISION, prepare_audio_batch, resolve_hook_points  # noqa: E402
from index import build_row_index  # noqa: E402
from fit_corpus_cost_probe_v1 import bin_rows, six_intervals  # noqa: E402
from fit_fp32_diagnostic_v1 import mps_cap_fraction  # noqa: E402
from fit_gate_v1 import (  # noqa: E402
    MAX_MPS_DRIVER_BYTES, _output_tensor, _replace_output,
    load_pinned_thinker, record_mps_memory,
)
from fit_pool_capability_v1 import (  # noqa: E402
    MANIFEST_SHA, SPEC_SHA, digest_json, load_fit_cases, sha, validate_tokenizer,
)
from fit_pool_single_token_score_v1 import selected_cases  # noqa: E402


SITES = ("encoder31", "connector")
SITE_NAMES = {"encoder31": "audio.encoder.31", "connector": "audio.to_language"}
TARGET = "text.residual.35"
RANK = 16
N_FIT = 48
N_PAIRS_X_VOICES = 24
SVD_RELATIVE_TOL = 1e-6
TARGET_MODE_MAX_ABS_TOL = 1e-3
SCORING_V2_SHA = "e7ea18cd0d5e2638bec64b8733049a5a2754ecd674080d93766db8409b46dd9b"
PROPOSAL = EVAL_DIR / "NEXT_PHASE_CORPUS_LENS_PROPOSAL.md"


def tensor_sha(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_row_map(model, processor, batch: dict) -> dict:
    mel_frames = int(batch["feature_attention_mask"].sum())
    positions = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
    return build_row_index(
        mel_frames=mel_frames, audio_token_positions=positions,
        n_window=model.config.audio_config.n_window,
        hop_length=processor.feature_extractor.hop_length,
        sample_rate=processor.feature_extractor.sampling_rate,
    )


def case_batch(model, processor, case: dict, group: dict):
    batch, digest = prepare_audio_batch(
        processor, ROOT / case["audio_path"], system_prompt=group["system_prompt"]
    )
    if digest != case["audio_sha256"]:
        raise ValueError("FIT audio digest changed")
    row_map = source_row_map(model, processor, batch)
    slot = group["groups"][(case["pair_id"], case["template"], case["engine"])]
    intervals = six_intervals(slot["word_slot_start_seconds"],
                              slot["word_slot_end_seconds"])
    batch = batch.to(model.device, dtype=model.dtype)
    return batch, row_map, intervals


def captures_for_forward(model, batch: dict, modules: dict, *, gradient: bool) -> dict:
    """Capture the same post-block hooks as the measured shared-graph probe."""
    captured = {}
    calls = {site: 0 for site in (*SITES, TARGET)}

    def encoder_hook(_module, _inputs, output):
        calls["encoder31"] += 1
        value = _output_tensor(output)
        if gradient:
            value = value.detach().requires_grad_(True)
            captured["encoder31"] = value
            return _replace_output(output, value)
        captured["encoder31"] = value.detach()
        return output

    def connector_hook(_module, _inputs, output):
        calls["connector"] += 1
        value = _output_tensor(output)
        captured["connector"] = value if gradient else value.detach()
        return output

    def target_hook(_module, _inputs, output):
        calls[TARGET] += 1
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError("Terminal target is not [batch,seq,width]")
        captured[TARGET] = value if gradient else value.detach()
        return output

    handles = [
        modules["encoder31"].register_forward_hook(encoder_hook),
        modules["connector"].register_forward_hook(connector_hook),
        modules[TARGET].register_forward_hook(target_hook),
    ]
    try:
        with torch.enable_grad() if gradient else torch.no_grad():
            model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if calls != {"encoder31": 1, "connector": 1, TARGET: 1}:
        raise ValueError(f"Unexpected hook calls {calls}")
    if captured[TARGET].shape[0] != 1 or captured[TARGET].shape[-1] != 2048:
        raise ValueError("Terminal width/batch changed")
    return captured


def rows_for_case(row_map: dict, intervals: list, captured: dict) -> dict:
    return {site: bin_rows(row_map, site, intervals, captured[site].shape[0])
            for site in SITES}


def canonical_basis(deltas: torch.Tensor, rank: int = RANK) -> tuple[torch.Tensor, torch.Tensor]:
    if deltas.ndim != 2 or deltas.shape[0] < rank or not torch.isfinite(deltas).all():
        raise ValueError("Invalid fit-only terminal-difference matrix")
    _, singular, vh = torch.linalg.svd(deltas.float().cpu(), full_matrices=False)
    if singular[rank - 1] <= singular[0] * SVD_RELATIVE_TOL:
        raise ValueError("Fit-only terminal differences have rank below 16")
    basis = vh[:rank].T.contiguous()
    for k in range(rank):
        largest = int(basis[:, k].abs().argmax())
        if basis[largest, k] < 0:
            basis[:, k] *= -1
    if not torch.allclose(basis.T @ basis, torch.eye(rank), atol=1e-4):
        raise ValueError("Target basis is not orthonormal")
    return basis, singular


def basis_from_fit_states(spec: dict, cases: list[dict],
                          states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(cases) != N_FIT or states.shape != (N_FIT, 2048):
        raise ValueError("Fit terminal states are incomplete")
    index = {case["case_id"]: i for i, case in enumerate(cases)}
    deltas = []
    for pair in spec["word_pairs"]["fit_pool"]:
        if not pair["map_fit"]:
            continue
        a, b = pair["words"]
        for engine in spec["render"]["voices"]:
            a_id = f"{pair['pair_id']}/fit_chosen/{engine}/{a}"
            b_id = f"{pair['pair_id']}/fit_chosen/{engine}/{b}"
            deltas.append(states[index[b_id]] - states[index[a_id]])
    if len(deltas) != N_PAIRS_X_VOICES:
        raise ValueError("Expected 24 FIT pair/voice terminal differences")
    return canonical_basis(torch.stack(deltas))


def category_means(cases: list[dict], target_states: torch.Tensor,
                   source_means: dict[str, torch.Tensor]) -> dict:
    categories = sorted({case["category"] for case in cases})
    if len(categories) != 4:
        raise ValueError("Expected four fit categories")
    result = {}
    for category in categories:
        ids = [i for i, case in enumerate(cases) if case["category"] == category]
        if len(ids) != 12:
            raise ValueError("Expected 12 map-fit contexts per category")
        result[category] = {
            "target": target_states[ids].double().mean(dim=0).float(),
            "sources": {
                site: source_means[site][ids].double().mean(dim=0).float()
                for site in SITES
            },
        }
    return result


def fit_metadata(spec: dict, cases: list[dict]) -> dict:
    return {
        "purpose": "FIT-only rank16 corpus-average rectangular audio transport",
        "script_sha256": sha(Path(__file__)),
        "spec_sha256": SPEC_SHA, "manifest_sha256": MANIFEST_SHA,
        "scoring_v2_sha256": SCORING_V2_SHA,
        "proposal_sha256": sha(PROPOSAL),
        "cases_sha256": digest_json(cases),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "target": TARGET, "sites": list(SITES),
        "rank": RANK, "n_fit": N_FIT,
        "basis_definition": "uncentered SVD of 24 undirected natural-audio pair/voice terminal-state differences; first fit template only",
        "svd_relative_rank_tolerance": SVD_RELATIVE_TOL,
        "grad_vs_no_grad_terminal_max_abs_tolerance": TARGET_MODE_MAX_ABS_TOL,
        "site_definition": "post-encoder31 and post-connector output; terminal post-text-block35 prompt-last",
        "bin_definition": "four equal quarters of frozen swapped-word sample interval plus one quarter before and after; nominal row-center tie earlier",
        "averaging": "each of 48 contexts equal weight; four contexts per lexical pair, so pairs equal weight",
        "dtype": "float32", "device": "mps",
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
    }


def stage1_collect(model, processor, modules, spec: dict,
                   cases: list[dict], group: dict, metadata: dict,
                   memory: Path) -> dict:
    target_states = []
    bin_means = {site: [] for site in SITES}
    records = []
    for index, case in enumerate(cases):
        batch, row_map, intervals = case_batch(model, processor, case, group)
        captured = captures_for_forward(model, batch, modules, gradient=False)
        rows = rows_for_case(row_map, intervals, captured)
        target = captured[TARGET][0, -1].float().cpu().contiguous()
        target_states.append(target)
        for site in SITES:
            bin_means[site].append(torch.stack([
                captured[site][r].float().mean(dim=0).cpu() for r in rows[site]
            ]))
        records.append({
            "case_id": case["case_id"], "audio_sha256": case["audio_sha256"],
            "prompt_tokens": int(batch["input_ids"].shape[1]),
            "target_sha256": tensor_sha(target),
            "target_state": target,
            "row_indices": rows,
            "source_shapes": {site: list(captured[site].shape) for site in SITES},
        })
        record_mps_memory(memory, f"stage1_after_case_{index+1}")
    target_states = torch.stack(target_states)
    bin_means = {site: torch.stack(values) for site, values in bin_means.items()}
    basis, singular = basis_from_fit_states(spec, cases, target_states)
    means = category_means(cases, target_states, bin_means)
    return {
        "metadata": metadata, "records": records,
        "target_states": target_states, "source_bin_means": bin_means,
        "target_basis": basis, "singular_values": singular,
        "category_means": means,
        "target_basis_sha256": tensor_sha(basis),
    }


def empty_accumulator(stage1: dict) -> dict:
    basis = stage1["target_basis"]
    if basis.shape != (2048, RANK):
        raise ValueError("Stage1 basis has wrong shape")
    widths = {site: int(stage1["source_bin_means"][site].shape[-1]) for site in SITES}
    return {
        "metadata": stage1["metadata"],
        "stage1_basis_sha256": stage1["target_basis_sha256"],
        "completed_case_ids": [],
        "completed_case_rows": [],
        "jacobian_sums": {
            site: torch.zeros((6, RANK, width), dtype=torch.float64)
            for site, width in widths.items()
        },
    }


def accumulate_one_vjp(accumulator: dict, gradients: dict[str, torch.Tensor],
                       rows: dict[str, list[list[int]]], coordinate: int) -> None:
    if not 0 <= coordinate < RANK:
        raise ValueError("Target basis coordinate outside rank")
    for site in SITES:
        grad = gradients[site]
        if grad.ndim != 2 or not bool(torch.isfinite(grad).all()):
            raise ValueError(f"Invalid {site} VJP")
        if grad.shape[1] != accumulator["jacobian_sums"][site].shape[-1]:
            raise ValueError(f"{site} VJP width mismatch")
        for bin_index, row_indices in enumerate(rows[site]):
            if not row_indices:
                raise ValueError("Empty source bin")
            accumulator["jacobian_sums"][site][bin_index, coordinate] += (
                grad[row_indices].sum(dim=0).detach().double().cpu()
            )


def stage2_case(model, processor, modules, case: dict, group: dict,
                stage1_record: dict, basis: torch.Tensor,
                accumulator: dict, memory: Path, *, max_coordinates: int = RANK) -> dict:
    batch, row_map, intervals = case_batch(model, processor, case, group)
    captured = captures_for_forward(model, batch, modules, gradient=True)
    rows = rows_for_case(row_map, intervals, captured)
    if rows != stage1_record["row_indices"] or {
        site: list(captured[site].shape) for site in SITES
    } != stage1_record["source_shapes"]:
        raise ValueError("Stage2 source rows/shapes differ from stage1")
    terminal_difference = (
        captured[TARGET][0, -1].detach().float().cpu()
        - stage1_record["target_state"]
    )
    max_abs_mode_difference = float(terminal_difference.abs().max())
    if max_abs_mode_difference > TARGET_MODE_MAX_ABS_TOL:
        raise ValueError("Grad/no-grad terminal state gap exceeds frozen tolerance")
    target = captured[TARGET]
    sources = [captured[site] for site in SITES]
    if not all(source.requires_grad for source in sources):
        raise ValueError("Shared audio graph disconnected")
    times = []
    for k in range(max_coordinates):
        cotangent = torch.zeros_like(target)
        cotangent[0, -1] = basis[:, k].to(target.device)
        torch.mps.synchronize()
        tick = time.perf_counter()
        gradients = torch.autograd.grad(
            target, sources, grad_outputs=cotangent,
            retain_graph=k < max_coordinates - 1, allow_unused=False,
        )
        torch.mps.synchronize()
        times.append(time.perf_counter() - tick)
        accumulate_one_vjp(
            accumulator, dict(zip(SITES, gradients, strict=True)), rows, k
        )
        record_mps_memory(memory, f"stage2_{case['case_id']}_coordinate_{k}")
    return {
        "case_id": case["case_id"], "audio_sha256": case["audio_sha256"],
        "vjp_seconds": times,
        "grad_vs_no_grad_terminal_max_abs": max_abs_mode_difference,
        "source_rows_per_bin": {site: list(map(len, rows[site])) for site in SITES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="FIT-only stage1 plus first VJP; separate output and no fitted map")
    parser.add_argument("--output-prefix", type=Path,
                        default=HERE / "fit_rank16_transport_v1")
    args = parser.parse_args()
    spec, manifest, all_cases = load_fit_cases()
    validate_tokenizer(spec)
    cases = selected_cases(spec, all_cases)
    if sha(HERE / "SCORING_V2.json") != SCORING_V2_SHA:
        raise ValueError("V2 scoring artifact changed")
    metadata = fit_metadata(spec, cases)
    if args.validate_only:
        print(json.dumps({"status": "fit_only_inputs_validated_no_model_forward",
                          "cases": len(cases), "metadata": metadata}, indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 corpus fitting requires MPS")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers source revision changed")
    prefix = args.output_prefix.resolve()
    stage1_path = prefix.with_suffix(".stage1.pt")
    checkpoint_path = prefix.with_suffix(".checkpoint.pt")
    progress_path = prefix.with_suffix(".progress.jsonl")
    memory_path = prefix.with_suffix(".memory.jsonl")
    lens_path = prefix.with_suffix(".lens.pt")
    report_path = prefix.with_suffix(".report.json")
    smoke_path = prefix.with_suffix(".smoke.json")
    if lens_path.exists() or report_path.exists() or (args.smoke and smoke_path.exists()):
        raise FileExistsError("Final fit/smoke output already exists")
    if args.smoke and checkpoint_path.exists():
        raise FileExistsError("Smoke must not reuse a full-fit checkpoint")
    if not args.smoke and smoke_path.exists():
        raise FileExistsError("Full fit must not reuse a smoke prefix")
    if not memory_path.exists():
        memory_path.write_text("")
    torch.mps.set_per_process_memory_fraction(
        mps_cap_fraction(int(torch.mps.recommended_max_memory()))
    )
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32 or processor.tokenizer.eos_token_id != spec["tokenizer_eos_id"]:
        raise ValueError("Pinned model dtype/EOS changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record_mps_memory(memory_path, "after_model_load")
    points = {point.name: point.module for point in resolve_hook_points(
        model, audio_layers=(31,), text_layers=(35,)
    )}
    modules = {site: points[SITE_NAMES[site]] for site in SITES}
    modules[TARGET] = points[TARGET]
    group = {
        "system_prompt": spec["system_prompt"],
        "groups": {(p["pair_id"], p["phrasing_id"], p["engine"]): p
                   for p in manifest["pairs"] if p["split"] == "fit_pool"},
    }
    if stage1_path.exists():
        stage1 = torch.load(stage1_path, map_location="cpu", weights_only=True)
        if stage1["metadata"] != metadata or [r["case_id"] for r in stage1["records"]] != [c["case_id"] for c in cases]:
            raise ValueError("Stage1 checkpoint metadata/cases differ")
    else:
        if checkpoint_path.exists() or progress_path.exists():
            raise ValueError("Stage2 state exists without stage1")
        stage1 = stage1_collect(model, processor, modules, spec, cases, group,
                                metadata, memory_path)
        atomic_torch_save(stage1_path, stage1)
    record_mps_memory(memory_path, "after_stage1")
    if args.smoke:
        acc = empty_accumulator(stage1)
        row = stage2_case(model, processor, modules, cases[0], group,
                          stage1["records"][0], stage1["target_basis"], acc,
                          memory_path, max_coordinates=1)
        atomic_json_save(smoke_path, {
            "status": "fit_only_integration_smoke_not_a_fitted_lens",
            "metadata": metadata,
            "stage1_sha256": sha(stage1_path),
            "basis_sha256": stage1["target_basis_sha256"],
            "case": row,
            "first_coordinate_bin_gradient_norms": {
                site: [float(acc["jacobian_sums"][site][b, 0].norm())
                       for b in range(6)] for site in SITES
            },
            "memory_sha256": sha(memory_path),
        })
        return
    if checkpoint_path.exists():
        acc = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if acc["metadata"] != metadata or acc["stage1_basis_sha256"] != stage1["target_basis_sha256"]:
            raise ValueError("Fit accumulator metadata/basis differs")
        if not progress_path.exists():
            progress_path.write_text("")
    else:
        if progress_path.exists():
            raise ValueError("Progress exists without checkpoint")
        acc = empty_accumulator(stage1)
        progress_path.write_text("")
        atomic_torch_save(checkpoint_path, acc)
    completed = acc["completed_case_ids"]
    if completed != [case["case_id"] for case in cases[:len(completed)]]:
        raise ValueError("Checkpoint is not a prefix of frozen FIT cases")
    progress_rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    saved_rows = acc.get("completed_case_rows", [])
    if [r["case_id"] for r in saved_rows] != completed:
        raise ValueError("Checkpoint case IDs/rows disagree")
    if progress_rows != saved_rows[:len(progress_rows)]:
        raise ValueError("Progress is not a prefix of authoritative checkpoint rows")
    if len(progress_rows) != len(saved_rows):
        progress_path.write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in saved_rows
        ))
    for index in range(len(completed), N_FIT):
        case = cases[index]
        row = stage2_case(model, processor, modules, case, group,
                          stage1["records"][index], stage1["target_basis"], acc,
                          memory_path)
        acc["completed_case_ids"].append(case["case_id"])
        acc["completed_case_rows"].append(row)
        # Checkpoint is authoritative; a trailing missing progress line can be
        # reconstructed exactly from its saved row on resume.
        atomic_torch_save(checkpoint_path, acc)
        with progress_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if (index + 1) % 8 == 0:
            print(f"fitted {index+1}/{N_FIT} FIT contexts", flush=True)
    matrices = {site: (acc["jacobian_sums"][site] / N_FIT).float()
                for site in SITES}
    lens = {
        "status": "fit_only_corpus_transport_no_efficacy_evaluation",
        "metadata": metadata,
        "stage1_sha256": sha(stage1_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "basis": stage1["target_basis"],
        "basis_sha256": stage1["target_basis_sha256"],
        "singular_values": stage1["singular_values"],
        "category_means": stage1["category_means"],
        "jacobians": matrices,
        "n_fit": N_FIT,
    }
    atomic_torch_save(lens_path, lens)
    report = {
        "status": lens["status"], "metadata": metadata,
        "stage1_sha256": sha(stage1_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "lens_sha256": sha(lens_path),
        "basis_sha256": stage1["target_basis_sha256"],
        "matrix_shapes": {site: list(matrices[site].shape) for site in SITES},
        "matrix_sha256": {site: tensor_sha(matrices[site]) for site in SITES},
        "completed_fit_contexts": len(acc["completed_case_ids"]),
        "progress_sha256": sha(progress_path),
        "memory_sha256": sha(memory_path),
    }
    atomic_json_save(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
