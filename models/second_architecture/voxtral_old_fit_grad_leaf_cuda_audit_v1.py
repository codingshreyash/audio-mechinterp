"""Independent, CPU-only saved-data audit for the prospective CUDA factorial.

The audit checks serialized evidence, not whether a model forward really ran.
It never imports the runner, checkpoint, or a CUDA module.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SITES = ("encoder31", "projector")
MODES = ("N0", "N1", "G0", "G1")
STAGES = ("lm0", "lm7", "lm14", "lm22", "lm29")
ORDERS = {"encoder31": ("N0", "G1", "G0", "N1", "N1", "G0", "G1", "N0"),
          "projector": ("G1", "N0", "N1", "G0", "G0", "N1", "N0", "G1")}
CASE_IDS = ("animal_08_kokoro_fit_chosen_beaver", "nature_09_piper_fit_chosen_waterfall")
SOURCE_RULES = ("archived_mps_bf16_cast_fp32", "two_passive_cuda_no_grad_first_source")
BEAVER_ARCHIVE_SHA = "0bc13e5831d3c4c325ad903bcbbdadf9f40d2e38eb67346ac68847bb1840dea2"
SCHEMA_SHA256 = "c90f46adcf73a8466232be15ba37d9a1d18b0eca234f2b4648f480bff6b7ad4d"
V1_SHA256 = "188925bc4022aa9841502d310899b5f9b28bf8ec0ba32ab7f7acb3fc45579050"
V2_SHA256 = "c86537552b55f6542c88d2be9a7c71d5e1cecd589c6a11cc89eec514c49419ca"
BF16_BEAVER = {"encoder31": "e0abb561ad91255d1945c81c1dead1a75266f09d0e7f3695a123040fef47b2b6",
               "projector": "a96b5e19ffefe48e537b30ddecb9de11c8c40becf4b4dd3f2813ab5583fce100"}
FP32_BEAVER = {"encoder31": "f0efa8b0692c7cd32f89bb9aff0e6dfcb7a799dc785b09f61ab571eddb2299fd",
               "projector": "36ca0e274eae938d23d32d9924d1ec6d854f2607a7eaee40b9a2b47babb5daee"}
BATCH_HASHES = (
    {"input_ids": "915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a",
     "attention_mask": "9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282",
     "input_features": "4f23ae2441eca7587b7acf19d89f717292a4f1efb3067631c7abb7e27236ee61"},
    {"input_ids": "915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a",
     "attention_mask": "9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282",
     "input_features": "caed5dc4c8328c670c081401c7ddf236f8def0ab3550c9fa62d6aee3a1f8b045"},
)
GIB = 1024**3
MAX_CASE_BYTES = GIB
MAX_CAMPAIGN_BYTES = int(2.5 * GIB)
MAX_LOG_BYTES = 8 * 1024**2
MAX_CUDA_BYTES = 42 * GIB
MAX_RAM_BYTES = 110 * GIB
MAX_WORKER_SECONDS = 600
MAX_AUDIT_SECONDS = 120
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ShapeSpec:
    source: dict[str, tuple[int, ...]]
    projector: tuple[int, ...]
    lm: tuple[int, ...]
    features: tuple[int, ...]
    logits: tuple[int, ...]
    audio_count: int = 375
    audio_token_id: int = 24


REAL = ShapeSpec({"encoder31": (1, 1500, 1280), "projector": (375, 3072)},
                 (375, 3072), (1, 396, 3072), (1, 128, 3000), (131072,))


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def raw_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    need(a.shape == b.shape and a.dtype == b.dtype, "raw comparison shape/dtype mismatch")
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def difference(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    need(a.shape == b.shape and a.dtype == b.dtype, "difference shape/dtype mismatch")
    delta = a.double() - b.double()
    l2 = float(torch.linalg.vector_norm(delta).item())
    maximum = float(delta.abs().max().item())
    # Symmetric relative L2 is bounded by two for nonzero vectors.
    denom = (float(torch.linalg.vector_norm(a.double()).item()) +
             float(torch.linalg.vector_norm(b.double()).item())) / 2
    return {"bitwise_equal": raw_equal(a, b), "changed_coordinates": int(torch.count_nonzero(delta).item()),
            "l2": l2, "max_abs": maximum, "symmetric_relative_l2": l2 / denom if denom else None}


def tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> torch.Tensor:
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
         value.dtype == dtype and tuple(value.shape) == shape and value.is_contiguous(),
         f"{name}: tensor device/dtype/shape/contiguity invalid")
    if dtype.is_floating_point:
        need(bool(torch.isfinite(value).all()), f"{name}: nonfinite tensor")
    return value


def json_file(path: Path) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{path.name}: missing or symlink")
    value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    need(isinstance(value, dict), f"{path.name}: expected object")
    return value


def schedule() -> list[tuple[str, str, int]]:
    rows = []
    for site in SITES:
        seen = dict.fromkeys(MODES, 0)
        for mode in ORDERS[site]:
            rows.append((site, mode, seen[mode]))
            seen[mode] += 1
    return rows


def boundary(row: dict[str, Any], kind: str) -> dict[str, torch.Tensor]:
    values = {}
    if kind == "full":
        values = {"source_post": row["source_post_full"], "projector": row["projector_full"]}
    values.update({"lm_input": row["lm_input_full"], "lm0_pre": row["lm0_input_full"]})
    values.update({stage: row["lm_terminal"][stage] for stage in STAGES})
    values["logits"] = row["first_answer_logits"]
    return values


def _hook_counts(value: Any, required: set[str], name: str) -> None:
    need(isinstance(value, dict) and required <= set(value) and
         all(type(v) is int and v == 1 for v in value.values()), f"{name}: hook counts invalid")


def _flags(row: dict[str, Any], name: str) -> None:
    mode = row["mode"]
    expected = {"grad_enabled": mode.startswith("G"), "leaf_requires_grad": mode.endswith("1")}
    need(row.get("factor_observed") == expected and
         all(type(v) is bool for v in row["factor_observed"].values()), f"{name}: factor flags invalid")
    down = row.get("downstream_grad_flags")
    need(isinstance(down, dict) and set(down) == {"lm_input", "lm0_pre"},
         f"{name}: downstream flags missing")
    for stage in down:
        flag = down[stage]
        need(isinstance(flag, dict) and set(flag) == {"grad_enabled", "requires_grad"} and
             all(type(v) is bool for v in flag.values()) and
             flag["grad_enabled"] == expected["grad_enabled"], f"{name}: downstream flags invalid")
    if row["kind"] == "direct":
        need(down["lm_input"]["requires_grad"] == expected["leaf_requires_grad"],
             f"{name}: direct leaf flag invalid")


def _validate_call(row: Any, ordinal: int, kind: str, site: str, mode: str,
                   repeat: int, spec: ShapeSpec) -> None:
    name = f"{kind}[{ordinal}]"
    need(isinstance(row, dict) and
         (row.get("kind"), row.get("ordinal"), row.get("site"), row.get("mode"), row.get("repeat")) ==
         (kind, ordinal, site, mode, repeat), f"{name}: schedule invalid")
    common = {"kind", "ordinal", "site", "mode", "repeat", "factor_observed",
              "downstream_grad_flags", "hook_counts", "lm_input_full", "lm0_input_full",
              "lm_terminal", "first_answer_logits"}
    extra = ({"source_sha256", "source_post_full", "projector_full", "audio_placeholder_positions"}
             if kind == "full" else {"input_sha256", "origin_mode", "origin_full_ordinal"})
    need(set(row) == common | extra, f"{name}: field set invalid")
    _flags(row, name)
    tensor(row["lm_input_full"], spec.lm, torch.float32, name + " LM input")
    tensor(row["lm0_input_full"], spec.lm, torch.float32, name + " LM0 pre")
    need(isinstance(row["lm_terminal"], dict) and set(row["lm_terminal"]) == set(STAGES),
         f"{name}: terminal stages invalid")
    for stage in STAGES:
        tensor(row["lm_terminal"][stage], (spec.lm[-1],), torch.float32, name + " " + stage)
    tensor(row["first_answer_logits"], spec.logits, torch.float32, name + " logits")
    if kind == "full":
        tensor(row["source_post_full"], spec.source[site], torch.float32, name + " source")
        tensor(row["projector_full"], spec.projector, torch.float32, name + " projector")
        _hook_counts(row["hook_counts"], {"source_override", "source_observer", "projector",
                                         "lm_input", "lm0_pre", *STAGES}, name)
    else:
        _hook_counts(row["hook_counts"], {"lm0_pre", *STAGES}, name)


def _summary(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    grouped = {mode: sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["repeat"])
               for mode in MODES}
    need(all(len(rows) == 2 for rows in grouped.values()), "repeat count invalid")
    names = tuple(boundary(rows[0], kind))
    contrasts = {}
    for left, right in itertools.combinations(MODES, 2):
        first_bitwise = first_above = None
        stages = {}
        for name in names:
            values = {mode: [boundary(r, kind)[name] for r in grouped[mode]] for mode in MODES}
            within = {mode: difference(*values[mode]) for mode in MODES}
            floor = max(1e-6, 3 * max(v["l2"] for v in within.values()))
            crosses = [{"left_repeat": li, "right_repeat": ri, **difference(values[left][li], values[right][ri])}
                       for li in range(2) for ri in range(2)]
            same = all(c["bitwise_equal"] for c in crosses)
            above = min(c["l2"] for c in crosses) > floor
            if not same and first_bitwise is None:
                first_bitwise = name
            if above and first_above is None:
                first_above = name
            stages[name] = {"within": within, "cross_pairs": crosses, "all_cross_bitwise_equal": same,
                            "floor_l2": floor, "above_floor": above}
        contrasts[f"{left}_vs_{right}"] = {"first_saved_bitwise_difference": first_bitwise,
                                          "first_above_floor_difference": first_above,
                                          "boundaries": stages,
                                          "repeats_bitwise_stable": all(v["bitwise_equal"] for stage in stages.values()
                                                                        for v in stage["within"].values())}
    return {"contrasts": contrasts}


def audit_payload(payload: Any, case_index: int, *, spec: ShapeSpec = REAL,
                  expected_batch_hashes: tuple[dict[str, str], dict[str, str]] = BATCH_HASHES,
                  expected_bf16: dict[str, str] = BF16_BEAVER,
                  expected_fp32: dict[str, str] = FP32_BEAVER) -> dict[str, Any]:
    need(case_index in (0, 1), "case index invalid")
    need(isinstance(payload, dict) and set(payload) ==
         {"schema", "case_id", "case_index", "batch", "sources_bf16", "sources_fp32", "passive", "calls"} and
         payload["schema"] == "voxtral_old_fit_grad_leaf_cuda_tensors_v1" and
         (payload["case_id"], payload["case_index"]) == (CASE_IDS[case_index], case_index),
         "sidecar schema/case invalid")
    batch = payload["batch"]
    need(isinstance(batch, dict) and set(batch) == {"input_ids", "attention_mask", "input_features"},
         "batch field set invalid")
    ids = tensor(batch["input_ids"], spec.lm[:2], torch.int64, "input_ids")
    mask = tensor(batch["attention_mask"], spec.lm[:2], torch.int64, "attention_mask")
    need(bool((mask == 1).all()), "attention mask invalid")
    tensor(batch["input_features"], spec.features, torch.float32, "input_features")
    hashes = {k: tensor_sha(v) for k, v in batch.items()}
    need(hashes == expected_batch_hashes[case_index], "batch tensor hash differs")
    positions = torch.nonzero(ids[0] == spec.audio_token_id).flatten().tolist()
    need(len(positions) == spec.audio_count, "audio placeholder count invalid")
    sources = payload["sources_fp32"]
    need(isinstance(sources, dict) and set(sources) == set(SITES), "source map invalid")
    for site in SITES:
        tensor(sources[site], spec.source[site], torch.float32, site + " source")
    source_hashes = {k: tensor_sha(v) for k, v in sources.items()}
    passive = payload["passive"]
    if case_index == 0:
        bf16 = payload["sources_bf16"]
        need(isinstance(bf16, dict) and set(bf16) == set(SITES) and passive == [],
             "beaver source/passive rule invalid")
        for site in SITES:
            tensor(bf16[site], spec.source[site], torch.bfloat16, site + " archived BF16")
            need(tensor_sha(bf16[site]) == expected_bf16[site] and
                 raw_equal(bf16[site].float(), sources[site]), "archived source/cast differs")
        need(source_hashes == expected_fp32, "beaver FP32 source hash differs")
    else:
        need(payload["sources_bf16"] == {} and isinstance(passive, list) and len(passive) == 2,
             "waterfall passive rule invalid")
        for ordinal, record in enumerate(passive):
            need(isinstance(record, dict) and set(record) ==
                 {"ordinal", "sources", "source_sha256", "hook_counts"} and record["ordinal"] == ordinal and
                 set(record["sources"]) == set(SITES), "passive record invalid")
            _hook_counts(record["hook_counts"], set(SITES), "passive")
            for site in SITES:
                t = tensor(record["sources"][site], spec.source[site], torch.float32, "passive source")
                need(record["source_sha256"].get(site) == tensor_sha(t) == source_hashes[site] and
                     raw_equal(t, sources[site]), "passive source differs")
    calls = payload["calls"]
    need(isinstance(calls, list) and len(calls) == 32, "32 factorial calls required")
    full, direct = calls[:16], calls[16:]
    order = schedule()
    origins = {}
    for ordinal, (site, mode, repeat) in enumerate(order):
        row = full[ordinal]
        _validate_call(row, ordinal, "full", site, mode, repeat, spec)
        need(row["source_sha256"] == source_hashes[site] and
             raw_equal(row["source_post_full"], sources[site]), "full source override differs")
        need(row["audio_placeholder_positions"] == positions, "placeholder positions differ")
        need(raw_equal(row["lm_input_full"][0, positions], row["projector_full"]),
             "projector/LM audio handoff differs")
        if site == "projector":
            need(raw_equal(row["source_post_full"], row["projector_full"]), "projector source differs")
        if mode == "N0" and site not in origins:
            origins[site] = row
    for index, (site, mode, repeat) in enumerate(order):
        ordinal = 16 + index
        row = direct[index]
        _validate_call(row, ordinal, "direct", site, mode, repeat, spec)
        fixed = origins[site]["lm_input_full"]
        need(row["origin_mode"] == "N0" and row["origin_full_ordinal"] == origins[site]["ordinal"] and
             row["input_sha256"] == tensor_sha(fixed) and raw_equal(row["lm_input_full"], fixed),
             "direct fixed input/origin differs")
    cells = {}
    required_comparisons = []
    for site in SITES:
        full_rows = [r for r in full if r["site"] == site]
        direct_rows = [r for r in direct if r["site"] == site]
        full_summary, direct_summary = _summary(full_rows, "full"), _summary(direct_rows, "direct")
        joined = {(r["mode"], r["repeat"]): r for r in direct_rows}
        pairs = []
        for f in full_rows:
            d = joined[(f["mode"], f["repeat"])]
            for label, a, b in (("lm0_pre", f["lm0_input_full"], d["lm0_input_full"]),
                                ("lm0", f["lm_terminal"]["lm0"], d["lm_terminal"]["lm0"]),
                                ("logits", f["first_answer_logits"], d["first_answer_logits"])):
                pairs.append({"mode": f["mode"], "repeat": f["repeat"], "boundary": label,
                              **difference(a, b)})
        need(len(pairs) == 24, "direct/full comparison count invalid")
        required_comparisons.extend(pairs)
        contrast = full_summary["contrasts"]["N0_vs_G1"]
        neutral_equal = all(
            full_summary["contrasts"][f"{a}_vs_{b}"]["boundaries"][name]["all_cross_bitwise_equal"]
            for a, b in (("N0", "N1"), ("N0", "G0"), ("N1", "G0"))
            for name in boundary(full_rows[0], "full"))
        stable = all(x["repeats_bitwise_stable"] for x in full_summary["contrasts"].values())
        pre_equal = all(contrast["boundaries"][name]["all_cross_bitwise_equal"]
                        for name in ("source_post", "projector", "lm_input", "lm0_pre"))
        direct_fixed = all(direct_summary["contrasts"][f"{a}_vs_{b}"]["boundaries"][name]["all_cross_bitwise_equal"]
                           for a, b in itertools.combinations(MODES, 2) for name in ("lm_input", "lm0_pre"))
        handoff_equal = all(raw_equal(r["lm_input_full"], r["lm0_input_full"])
                            for r in full_rows + direct_rows)
        same = (all(p["bitwise_equal"] for p in pairs) and neutral_equal and stable and pre_equal and
                direct_fixed and handoff_equal and contrast["first_above_floor_difference"] == "lm0")
        cells[site] = {"label": "same_pattern" if same else "different_pattern",
                       "direct_full_equal_count": sum(p["bitwise_equal"] for p in pairs),
                       "direct_full_comparisons": pairs, "full": full_summary, "direct": direct_summary,
                       "neutral_equal": neutral_equal, "repeats_stable": stable,
                       "pre_equal": pre_equal, "direct_fixed": direct_fixed,
                       "lm_input_to_lm0_pre_equal": handoff_equal}
    need(len(required_comparisons) == 48, "case comparison count invalid")
    return {"case_id": CASE_IDS[case_index], "case_index": case_index, "cells": cells,
            "direct_full_comparison_count": 48, "direct_full_equal_count":
            sum(p["bitwise_equal"] for p in required_comparisons),
            "tensor_hashes": {"batch": hashes, "sources_fp32": source_hashes,
                              "sources_bf16": {k: tensor_sha(v) for k, v in payload["sources_bf16"].items()},
                              "passive": [{k: tensor_sha(v) for k, v in r["sources"].items()} for r in passive],
                              "calls": [{name: tensor_sha(t) for name, t in boundary(r, r["kind"]).items()}
                                        for r in calls]},
            "saved_forwards": 32 + len(passive)}


def audit_case(run_dir: Path, case_index: int, *, spec: ShapeSpec = REAL,
               expected_batch_hashes: tuple[dict[str, str], dict[str, str]] = BATCH_HASHES,
               expected_bf16: dict[str, str] = BF16_BEAVER,
               expected_fp32: dict[str, str] = FP32_BEAVER) -> dict[str, Any]:
    started = time.monotonic()
    need(run_dir.is_dir() and not run_dir.is_symlink(), "attempt directory invalid")
    paths = list(run_dir.rglob("*"))
    need(not any(p.is_symlink() for p in paths), "attempt contains symlink")
    need(not (run_dir / "error.json").exists(), "completed case also has error receipt")
    total = sum(p.stat().st_size for p in paths if p.is_file())
    need(total < MAX_CASE_BYTES, "case output cap exceeded")
    for name in ("stdout", "stderr"):
        for p in paths:
            if p.name.startswith(name) and p.is_file():
                need(p.stat().st_size <= MAX_LOG_BYTES, f"{name} cap exceeded")
    meta, result = json_file(run_dir / "meta.json"), json_file(run_dir / "result.json")
    receipt = json_file(run_dir / "source_receipt.json")
    layout = json_file(run_dir / "model_layout.json")
    need(layout.get("valid") is True and not layout.get("errors"), "model layout invalid")
    need((meta.get("schema"), meta.get("status"), result.get("schema"), result.get("status")) ==
         ("voxtral_old_fit_grad_leaf_cuda_meta_v1", "started",
          "voxtral_old_fit_grad_leaf_cuda_result_v1", "completed"), "meta/result status invalid")
    need(all((obj.get("case_id"), obj.get("case_index")) == (CASE_IDS[case_index], case_index)
             for obj in (meta, result)), "case provenance invalid")
    for key in ("attempt_id", "freeze_sha256", "protocol_v1_sha256", "protocol_v2_sha256",
                "schema_sha256", "runner_sha256"):
        need(meta.get(key) == result.get(key) and isinstance(meta.get(key), str), f"{key} differs")
    for key in ("freeze_sha256", "runner_sha256", "environment_receipt_sha256",
                "checkpoint_tree_sha256"):
        need(isinstance(meta.get(key), str) and HEX64.fullmatch(meta[key]), f"{key} pin invalid")
    need(meta["protocol_v1_sha256"] == V1_SHA256 and meta["protocol_v2_sha256"] == V2_SHA256 and
         meta["schema_sha256"] == SCHEMA_SHA256, "protocol/schema pin differs")
    for key, filename in (("meta_sha256", "meta.json"), ("source_receipt_sha256", "source_receipt.json"),
                          ("model_layout_sha256", "model_layout.json"), ("progress_sha256", "progress.jsonl")):
        need(result.get(key) == sha(run_dir / filename), f"{filename} hash differs")
    progress = (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    need(len(progress) == 37 + (2 if case_index else 0), "progress event count differs")
    events = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
              for line in progress]
    expected_phases = (["preflight_complete_before_model_load", "cuda_configured_before_model_load",
                        "model_loaded"] +
                       [f"after_passive_{i}" for i in range(2 if case_index else 0)] +
                       [f"after_full_{site}_{mode}_{repeat}" for site, mode, repeat in schedule()] +
                       [f"after_direct_{site}_{mode}_{repeat}" for site, mode, repeat in schedule()] +
                       ["source_frozen", "details_saved"])
    # source_frozen precedes factorial calls, while details_saved is last.
    expected_phases.remove("source_frozen")
    expected_phases.insert(3 + (2 if case_index else 0), "source_frozen")
    need([event.get("phase") for event in events] == expected_phases,
         "progress execution schedule differs")
    for event in events:
        for key, cap in (("elapsed_seconds", MAX_WORKER_SECONDS),
                         ("cuda_allocated_bytes", MAX_CUDA_BYTES),
                         ("cuda_reserved_bytes", MAX_CUDA_BYTES),
                         ("process_rss_bytes", MAX_RAM_BYTES)):
            value = event.get(key)
            need(type(value) in (int, float) and math.isfinite(value) and 0 <= value < cap,
                 f"progress {key} invalid")
    need(all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in zip(events, events[1:])),
         "progress elapsed times regress")
    count = {"passive_forwards": 0 if case_index == 0 else 2, "full_forwards": 16,
             "direct_forwards": 16, "vjps": 0, "generations": 0}
    need(meta.get("planned_counts") == result.get("actual_counts") == count,
         "planned/actual count differs")
    need(meta.get("full_schedule") == [list(x) for x in schedule()] and
         meta.get("direct_schedule") == [[site, "N0", mode, repeat]
                                          for site, mode, repeat in schedule()], "schedule pin differs")
    need(meta.get("source_rule") == receipt.get("source_rule") == SOURCE_RULES[case_index] and
         receipt.get("case_id") == CASE_IDS[case_index] and
         receipt.get("schema") == "voxtral_old_fit_grad_leaf_cuda_source_receipt_v1",
         "source receipt provenance invalid")
    need(meta.get("source_sha256") == (expected_fp32 if case_index == 0 else {}),
         "meta pre-model source pin invalid")
    need(meta.get("batch_sha256") == expected_batch_hashes[case_index],
         "meta batch pin invalid")
    need(meta.get("caps") == {"cuda_bytes": MAX_CUDA_BYTES, "rss_bytes": MAX_RAM_BYTES,
                              "worker_seconds": MAX_WORKER_SECONDS,
                              "worker_output_bytes": MAX_CASE_BYTES - 24 * 1024**2,
                              "total_output_bytes": MAX_CASE_BYTES}, "resource caps differ")
    for key, cap in (("elapsed_seconds", MAX_WORKER_SECONDS),
                     ("peak_cuda_allocated_bytes", MAX_CUDA_BYTES),
                     ("peak_cuda_reserved_bytes", MAX_CUDA_BYTES),
                     ("peak_process_rss_bytes", MAX_RAM_BYTES)):
        value = result.get(key)
        need(type(value) in (int, float) and math.isfinite(value) and 0 <= value < cap,
             f"{key} cap/receipt invalid")
    need(result.get("audio_token_id") == spec.audio_token_id, "audio token ID differs")
    need(isinstance(result.get("execution_metadata"), dict) and
         all(key in result["execution_metadata"] for key in
             ("torch_version", "cuda_runtime_version", "cudnn_version", "gpu_name",
              "gpu_capability", "platform", "transformers_version", "accelerate_version",
              "cublas_workspace_config", "flash_sdp_enabled", "mem_efficient_sdp_enabled",
              "math_sdp_enabled", "config_attention_implementation")) and
         result["execution_metadata"].get("cublas_workspace_config") == ":4096:8" and
         result["execution_metadata"].get("deterministic_algorithms") is True and
         result["execution_metadata"].get("deterministic_warn_only") is False and
         result["execution_metadata"].get("matmul_allow_tf32") is False and
         result["execution_metadata"].get("cudnn_allow_tf32") is False and
         result["execution_metadata"].get("matmul_precision") == "highest" and
         result["execution_metadata"].get("requested_attention_implementation") == "sdpa" and
         result["execution_metadata"].get("model_eval") is True and
         result["execution_metadata"].get("language_model_eval") is True,
         "CUDA deterministic/TF32 metadata invalid")
    detail_sha = result.get("details_sha256")
    need(isinstance(detail_sha, str) and HEX64.fullmatch(detail_sha) and
         result.get("details_path") == f"blobs/{detail_sha}.pt", "sidecar path invalid")
    detail = run_dir / "blobs" / f"{detail_sha}.pt"
    need(detail.is_file() and detail.stat().st_size == result.get("details_bytes") and
         sha(detail) == detail_sha, "sidecar bytes/hash differ")
    payload = torch.load(detail, map_location="cpu", weights_only=True, mmap=True)
    report = audit_payload(payload, case_index, spec=spec, expected_batch_hashes=expected_batch_hashes,
                           expected_bf16=expected_bf16, expected_fp32=expected_fp32)
    need(meta.get("batch_sha256") == report["tensor_hashes"]["batch"], "meta batch hash differs")
    need(receipt.get("source_fp32_sha256") == report["tensor_hashes"]["sources_fp32"],
         "source receipt FP32 hash differs")
    if case_index == 0:
        need(receipt.get("source_bf16_sha256") == report["tensor_hashes"]["sources_bf16"],
             "source receipt BF16 hash differs")
        need(receipt.get("archived_sidecar_sha256") == BEAVER_ARCHIVE_SHA and
             receipt.get("passive_source_sha256") == [] and receipt.get("passive_equal") == {},
             "beaver archive/passive receipt differs")
    else:
        need(receipt.get("passive_source_sha256") == report["tensor_hashes"]["passive"] and
             receipt.get("passive_equal") == dict.fromkeys(SITES, True), "passive receipt differs")
    need(time.monotonic() - started < MAX_AUDIT_SECONDS, "CPU audit time cap exceeded")
    return {**report, "status": "audited_saved_data_only", "result_sha256": sha(run_dir / "result.json"),
            "details_sha256": detail_sha, "case_bytes": total, "audit_seconds": time.monotonic() - started,
            "limitation": "Saved bytes cannot attest model execution, kernel identity, or a security finding."}


def audit_campaign(case_dirs: list[Path], output: Path, **kwargs: Any) -> dict[str, Any]:
    need(len(case_dirs) == 2 and len(set(case_dirs)) == 2, "exactly two distinct cases required")
    need(not output.exists(), "audit output already exists")
    cases = []
    for index, run_dir in enumerate(case_dirs):
        try:
            cases.append(audit_case(run_dir, index, **kwargs))
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            cases.append({"case_id": CASE_IDS[index], "case_index": index, "status": "invalid",
                          "cells": {site: {"label": "invalid"} for site in SITES},
                          "error": f"{type(error).__name__}: {error}"})
    total_bytes = sum(x.get("case_bytes", 0) for x in cases)
    if total_bytes >= MAX_CAMPAIGN_BYTES:
        cases = [{**x, "status": "invalid", "cells": {site: {"label": "invalid"} for site in SITES},
                  "error": "campaign output cap exceeded"} for x in cases]
    report = {"schema": "voxtral_old_fit_grad_leaf_cuda_audit_v1", "cases": cases,
              "planned_case_count": 2, "planned_forward_count": 66,
              "saved_forward_count": sum(x.get("saved_forwards", 0) for x in cases),
              "planned_cell_count": 4,
              "direct_full_comparison_count": sum(x.get("direct_full_comparison_count", 0) for x in cases),
              "cells": [{"case_id": x["case_id"], "site": site, **x["cells"][site]}
                        for x in cases for site in SITES],
              "limitation": "Saved-data audit only; no model execution or security effect is proven."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def audit_one_case(run_dir: Path, case_index: int, output: Path, **kwargs: Any) -> dict[str, Any]:
    """Write an exclusive success/invalid receipt for the launcher's case gate."""
    need(case_index in (0, 1) and not output.exists(), "case/output invalid or already exists")
    try:
        report = audit_case(run_dir, case_index, **kwargs)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        report = {"case_id": CASE_IDS[case_index], "case_index": case_index, "status": "invalid",
                  "cells": {site: {"label": "invalid"} for site in SITES},
                  "error": f"{type(error).__name__}: {error}"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beaver-dir", type=Path)
    parser.add_argument("--waterfall-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--case-index", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir is not None and args.case_index is not None and args.beaver_dir is None and args.waterfall_dir is None:
        report = audit_one_case(args.run_dir, args.case_index, args.output)
        print(json.dumps({"status": report["status"], "audit_sha256": sha(args.output)}))
    elif args.beaver_dir is not None and args.waterfall_dir is not None and args.run_dir is None and args.case_index is None:
        report = audit_campaign([args.beaver_dir, args.waterfall_dir], args.output)
        print(json.dumps({"planned_cell_count": 4, "cells": [(x["case_id"], x["site"], x["label"])
                                                               for x in report["cells"]],
                          "audit_sha256": sha(args.output)}))
    else:
        parser.error("Provide either --run-dir with --case-index or both case directories")


if __name__ == "__main__":
    main()
