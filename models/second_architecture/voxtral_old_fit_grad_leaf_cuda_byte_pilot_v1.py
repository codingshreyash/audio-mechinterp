"""CPU-only maximum-shape serialization pilot; no model or checkpoint import."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

import voxtral_old_fit_grad_leaf_cuda_runner_v1 as runner


def tensor(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)


def make_case(case_index: int) -> dict:
    sources = {site: tensor(runner.replay.SOURCE_SHAPES[site]) for site in runner.SITES}
    bf16 = ({site: tensor(runner.replay.SOURCE_SHAPES[site], torch.bfloat16)
             for site in runner.SITES} if case_index == 0 else {})
    batch = {"input_ids": tensor((1, 396), torch.int64),
             "attention_mask": tensor((1, 396), torch.int64),
             "input_features": tensor((1, 128, 3000))}
    passive = []
    if case_index == 1:
        for ordinal in (0, 1):
            passive.append({
                "ordinal": ordinal,
                "sources": {site: sources[site].clone() for site in runner.SITES},
                "source_sha256": {site: "0" * 64 for site in runner.SITES},
                "hook_counts": {site: 1 for site in runner.SITES},
            })
    calls = []
    for kind, schedule in (("full", runner.full_schedule()),
                           ("direct", runner.direct_schedule())):
        for descriptor in schedule:
            site, mode, repeat = (descriptor if kind == "full" else
                                  (descriptor[0], descriptor[2], descriptor[3]))
            row = {
                "kind": kind, "ordinal": len(calls), "site": site,
                "mode": mode, "repeat": repeat,
                "factor_observed": {"grad_enabled": mode.startswith("G"),
                                    "leaf_requires_grad": mode.endswith("1")},
                "downstream_grad_flags": {
                    "lm_input": {"grad_enabled": mode.startswith("G"),
                                 "requires_grad": mode.endswith("1")},
                    "lm0_pre": {"grad_enabled": mode.startswith("G"),
                                "requires_grad": mode.endswith("1")}},
                "lm_input_full": tensor(runner.replay.LM_INPUT_SHAPE),
                "lm0_input_full": tensor(runner.replay.LM_INPUT_SHAPE),
                "lm_terminal": {f"lm{i}": tensor((3072,))
                                for i in (0, 7, 14, 22, 29)},
                "first_answer_logits": tensor((runner.LOGIT_WIDTH,)),
            }
            if kind == "full":
                row.update({
                    "source_sha256": "0" * 64,
                    "source_post_full": tensor(runner.replay.SOURCE_SHAPES[site]),
                    "projector_full": tensor(runner.replay.SOURCE_SHAPES["projector"]),
                    "audio_placeholder_positions": list(range(375)),
                    "hook_counts": {key: 1 for key in (
                        "source_override", "source_observer", "projector", "lm_input",
                        "lm0_pre", "lm0", "lm7", "lm14", "lm22", "lm29")},
                })
            else:
                row.update({
                    "input_sha256": "0" * 64, "origin_mode": "N0",
                    "origin_full_ordinal": 0 if site == "encoder31" else 9,
                    "hook_counts": {key: 1 for key in (
                        "lm0_pre", "lm0", "lm7", "lm14", "lm22", "lm29")},
                })
            calls.append(row)
    return {"schema": "voxtral_old_fit_grad_leaf_cuda_tensors_v1",
            "case_id": runner.CASE_IDS[case_index], "case_index": case_index,
            "batch": batch, "sources_bf16": bf16, "sources_fp32": sources,
            "passive": passive, "calls": calls}


def main() -> None:
    measurements = []
    with tempfile.TemporaryDirectory(prefix="voxtral_cuda_byte_pilot_") as directory:
        for case_index in (0, 1):
            payload = make_case(case_index)
            path = Path(directory) / f"case_{case_index}.pt"
            torch.save(payload, path)
            size = path.stat().st_size
            measurements.append({"case_index": case_index,
                                 "case_id": runner.CASE_IDS[case_index],
                                 "sidecar_bytes": size,
                                 "sidecar_sha256": runner.sha(path),
                                 "within_worker_cap": size + 1024**2 < runner.MAX_WORKER_BYTES})
            del payload
            path.unlink()
    print(json.dumps({"schema": "voxtral_old_fit_grad_leaf_cuda_byte_pilot_v1",
                      "torch_version": torch.__version__,
                      "max_worker_bytes": runner.MAX_WORKER_BYTES,
                      "measurements": measurements}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
