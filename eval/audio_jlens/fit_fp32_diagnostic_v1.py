"""Fit-only FP32 rerun of the audio directional finite-difference diagnostic.

This uses only the original Piper red-to-blue pair and the existing diagnostic
score paths. It does not read development or locked-test audio. It is kept
separate from the BF16 diagnostic and refuses to overwrite either result.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from fit_fd_diagnostic_v1 import (
    ALPHAS,
    DONOR,
    ENGINE,
    RECIPIENT,
    SITES_TO_CHECK,
    ZERO_REPEATS,
    diagnose_site,
)
from fit_gate_v1 import (
    ANSWER_IDS,
    FIT_MANIFEST_SHA,
    MAX_MPS_DRIVER_BYTES,
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
    TRANSFORMERS_REVISION,
    build_inputs,
    load_pinned_thinker,
    record_mps_memory,
    sha,
    verify_fit_files,
    verify_tokenizer,
)


HERE = Path(__file__).resolve().parent


def mps_cap_fraction(recommended_bytes: int) -> float:
    """Convert the fixed 30 GiB ceiling to an MPS allocator fraction."""
    if recommended_bytes <= 0:
        raise ValueError("MPS recommended maximum memory must be positive")
    return min(1.0, MAX_MPS_DRIVER_BYTES / recommended_bytes)


def relabel_fp32_edits(result: dict) -> dict:
    """Remove the BF16 field name emitted by the shared diagnostic helper."""
    if "conditions" not in result:
        raise ValueError("Diagnostic result lacks patch conditions")
    for condition in result["conditions"]:
        if "effective_bf16_delta_norm" not in condition:
            raise ValueError("Shared diagnostic edit-norm field changed")
        condition["effective_fp32_delta_norm"] = condition.pop(
            "effective_bf16_delta_norm"
        )
        condition["effective_edit_dtype"] = "float32"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "fit_fp32_diagnostic_v1_results.json",
    )
    args = parser.parse_args()
    memory_log = args.output.with_suffix(".memory.jsonl")
    if args.output.exists() or memory_log.exists():
        raise FileExistsError("Refusing to overwrite FP32 result or memory log")
    if not torch.backends.mps.is_available():
        raise RuntimeError("FP32 Qwen diagnostic requires MPS; CPU tests do not")
    installed = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not installed or json.loads(installed)["vcs_info"]["commit_id"] != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers revision differs from pin")
    manifest = verify_fit_files([ENGINE])
    recommended = int(torch.mps.recommended_max_memory())
    cap_fraction = mps_cap_fraction(recommended)
    torch.mps.set_per_process_memory_fraction(cap_fraction)
    memory_log.write_text("")
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=torch.float32,
        device_map="mps", low_cpu_mem_usage=True,
    )
    if model.dtype != torch.float32:
        raise ValueError(f"Expected FP32 model, got {model.dtype}")
    record_mps_memory(memory_log, "after_model_load")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    verify_tokenizer(processor)
    inputs, group = build_inputs(model, processor, manifest, ENGINE)

    report = {
        "status": "fit_only_complete",
        "purpose": "FP32 same-path grad versus no-grad directional finite differences",
        "engine": ENGINE, "recipient": RECIPIENT, "donor": DONOR,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "fit_manifest_sha256": FIT_MANIFEST_SHA,
        "split_sha256": sha(HERE / "SPLIT_V1.json"),
        "fit_gate_script_sha256": sha(HERE / "fit_gate_v1.py"),
        "shared_bf16_diagnostic_script_sha256": sha(HERE / "fit_fd_diagnostic_v1.py"),
        "fp32_diagnostic_script_sha256": sha(Path(__file__)),
        "system_prompt": SYSTEM_PROMPT,
        "answer_token_ids": ANSWER_IDS,
        "alpha_grid": list(ALPHAS),
        "zero_repeats": ZERO_REPEATS,
        "memory_log_path": str(memory_log),
        "mps_recommended_max_bytes": recommended,
        "mps_hard_cap_fraction": cap_fraction,
        "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES,
        "device": "mps", "dtype": "float32", "torch_version": torch.__version__,
        "results": [],
    }
    for site in SITES_TO_CHECK:
        result = diagnose_site(model, processor, inputs, group, site, memory_log)
        report["results"].append(relabel_fp32_edits(result))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
