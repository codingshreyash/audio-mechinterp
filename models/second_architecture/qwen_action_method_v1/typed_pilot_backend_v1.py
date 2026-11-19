"""Gated, typed-only Qwen action pilot. Importing this file loads no weights.

The fixed GO is deliberately absent until an independent executable review.
Audio, development, and locked cases are not accepted by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import endpoint_v2, readout_v2, runner_contract_v1, source_audit_v2
from .endpoint import SPEC_PATH, load_spec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTEMPT = HERE / "runs/typed_pilot_v1/attempt_001"
GO = HERE / "TYPED_PILOT_GO_V1.json"
REVIEW = HERE / "TYPED_PILOT_INDEPENDENT_EXECUTABLE_REVIEW_V1.md"
SOURCES = (
    "models/second_architecture/qwen_action_method_v1/typed_pilot_backend_v1.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_parent_v1.py",
    "models/second_architecture/qwen_action_method_v1/typed_pilot_audit_v1.py",
    "models/second_architecture/qwen_action_method_v1/test_typed_pilot_v1.py",
    "models/second_architecture/qwen_action_method_v1/runner_contract_v1.py",
    "models/second_architecture/qwen_action_method_v1/source_audit_v2.py",
    "models/second_architecture/qwen_action_method_v1/endpoint.py",
    "models/second_architecture/qwen_action_method_v1/endpoint_v2.py",
    "models/second_architecture/qwen_action_method_v1/readout_v2.py",
    "models/second_architecture/qwen_action_method_v1/frozen_spec_v1.json",
    "models/second_architecture/qwen_action_method_v1/READOUT_FREEZE_V2.json",
    "models/second_architecture/qwen_action_method_v1/PROTOCOL_AMENDMENT_V2.md",
    "models/second_architecture/qwen_action_method_v1/PROTOCOL_AMENDMENT_V3.md",
    "models/second_architecture/qwen_action_method_v1/TYPED_PILOT_BACKEND_HANDOFF_V1.md",
    "models/qwen_omni/adapter.py",
    "models/qwen_omni/pins.json",
    "models/second_architecture/wav2vec2_fit_noise_norm_causal_host_v1.py",
    "models/second_architecture/nonspeech_probe_v1/causal_parent_v2.py",
    "models/second_architecture/nonspeech_probe_v1/causal_worker_v2.py",
    "models/second_architecture/nonspeech_probe_v1/causal_plan_v1.py",
)
PACKAGES = {"torch": "2.15.0", "transformers": "5.24.0.dev0",
            "accelerate": "1.16.0", "numpy": "2.5.5", "psutil": "7.2.2"}
LABELS = ("LIST", "ASK", "CANCEL")
SITES = runner_contract_v1.SITES
MAX_MPS_BYTES = runner_contract_v1.PILOT_MPS_LIMIT
MAX_EVIDENCE_BYTES = 64 * 1024**2


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def write_once(path: Path, value: Any) -> None:
    data = canonical(value)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def save_array(folder: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    require(array.dtype == np.float32 and np.isfinite(array).all(),
            f"{name} must be finite FP32")
    path = folder / f"{name}.npy"
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": path.name, "sha256": sha(path), "shape": list(array.shape),
            "dtype": str(array.dtype), "bytes": path.stat().st_size}


def source_identity() -> dict[str, Any]:
    """Require every reviewed source to be committed and unchanged."""
    hashes: dict[str, str] = {}
    for relative in SOURCES:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing source: {relative}")
        tracked = subprocess.run(("git", "ls-files", "--error-unmatch", "--", relative),
                                 cwd=ROOT, capture_output=True)
        require(tracked.returncode == 0, f"untracked source: {relative}")
        for flag in ((), ("--cached",)):
            clean = subprocess.run(("git", "diff", *flag, "--quiet", "--", relative),
                                   cwd=ROOT, capture_output=True)
            require(clean.returncode == 0, f"dirty source: {relative}")
        hashes[relative] = sha(path)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == PACKAGES, "pilot package versions changed")
    distribution = importlib.metadata.distribution("transformers")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(direct.get("vcs_info", {}).get("commit_id")
            == load_spec()["transformers_revision"],
            "installed Transformers revision changed")
    source_audit_v2.audit()
    return {"sources": hashes, "packages": versions,
            "transformers_commit": direct["vcs_info"]["commit_id"],
            "spec_sha256": sha(SPEC_PATH),
            "readout_sha256": sha(HERE / "READOUT_FREEZE_V2.json")}


def validate_go() -> dict[str, Any]:
    identity = source_identity()
    require(GO.is_file() and not GO.is_symlink()
            and REVIEW.is_file() and not REVIEW.is_symlink(),
            "independent typed-pilot GO absent")
    receipt = json.loads(GO.read_text())
    require(receipt.get("schema") == "qwen_action_typed_pilot_go_v1"
            and receipt.get("status") == "independent_executable_go"
            and receipt.get("approved_attempt") == str(ATTEMPT)
            and receipt.get("scope") == "one typed F_L1 pilot only"
            and receipt.get("source_identity") == identity
            and receipt.get("review_sha256") == sha(REVIEW),
            "typed-pilot GO binding invalid")
    return receipt


def fixed_case() -> dict[str, str]:
    row = next(row for row in load_spec()["scripts"] if row["id"] == "F_L1")
    require(row["split"] == "fit_text" and row["label"] == "LIST",
            "fixed typed pilot case changed")
    return row


def measure_mps(torch: Any, phase: str, folder: Path) -> int:
    driver = int(torch.mps.driver_allocated_memory())
    data = {"phase": phase, "driver_allocated_bytes": driver,
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "monotonic_seconds": time.monotonic()}
    with (folder / "mps_samples.jsonl").open("ab") as handle:
        handle.write(canonical(data))
        handle.flush()
        os.fsync(handle.fileno())
    require(driver < MAX_MPS_BYTES, "MPS driver allocation limit exceeded")
    return driver


def _first(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_first(output: Any, replacement: Any) -> Any:
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    return replacement


def teacher_forced_triplet(model: Any, batch: dict[str, Any],
                           modules: dict[str, Any], torch: Any,
                           folder: Path) -> list[dict[str, Any]]:
    prompt = batch["input_ids"]
    require(prompt.ndim == 2 and tuple(prompt.shape[:1]) == (1,)
            and prompt.shape[1] > 0, "typed prompt tensor invalid")
    prompt_ids = [int(value) for value in prompt[0].tolist()]
    prompt_hash = hashlib.sha256(json.dumps(prompt_ids,
                             separators=(",", ":")).encode()).hexdigest()
    records = []
    for label in LABELS:
        label_id = readout_v2.LABEL_TOKEN_IDS[LABELS.index(label)]
        extended = dict(batch)
        suffix = torch.tensor([[label_id]], dtype=prompt.dtype, device=prompt.device)
        extended["input_ids"] = torch.cat((prompt, suffix), dim=1)
        if "attention_mask" in batch:
            extended["attention_mask"] = torch.cat((batch["attention_mask"],
                                      torch.ones_like(suffix, dtype=batch["attention_mask"].dtype)), dim=1)
        calls: dict[str, list[Any]] = {site: [] for site in SITES}
        handles = []
        for site in SITES:
            def capture(_module: Any, _inputs: Any, output: Any, *, key: str = site) -> Any:
                value = _first(output)
                require(value.ndim == 3 and value.shape[0] == 1
                        and value.shape[-1] == runner_contract_v1.WIDTH,
                        f"{key} hook state shape invalid")
                if key == SITES[0]:
                    # With frozen weights and integer IDs, no upstream tensor
                    # requires grad. Insert a numerically identical leaf at the
                    # first site; the later site remains connected to it.
                    value = value.detach().requires_grad_(True)
                    calls[key].append(value)
                    return _replace_first(output, value)
                calls[key].append(value)
                return None
            handles.append(modules[site].register_forward_hook(capture))
        try:
            with torch.enable_grad():
                output = model(**extended, use_cache=False, return_dict=True)
                require(all(len(calls[site]) == 1 for site in SITES),
                        "teacher-forced hook call count invalid")
                first = output.logits[0, len(prompt_ids)-1].float()
                after = output.logits[0, len(prompt_ids)].float()
                require(first.ndim == after.ndim == 1
                        and first.shape == after.shape
                        and first.numel() > readout_v2.EOS_TOKEN_ID,
                        "teacher-forced full-vocabulary logits invalid")
                # The model and saved logits remain FP32. Normalize the very
                # large vocabulary in FP64 so the scalar exactly matches the
                # independent NumPy log-sum-exp audit.
                score = (first.double().log_softmax(-1)[label_id]
                         + after.double().log_softmax(-1)[readout_v2.EOS_TOKEN_ID])
                gradients = torch.autograd.grad(score,
                    tuple(calls[site][0] for site in SITES), allow_unused=False)
            entry: dict[str, Any] = {"label": label,
                "target_ids": [label_id, readout_v2.EOS_TOKEN_ID],
                "prompt_ids": prompt_ids, "prompt_ids_sha256": prompt_hash,
                "teacher_forced_input_ids": prompt_ids + [label_id],
                "logit_positions": [len(prompt_ids)-1, len(prompt_ids)],
                "hook_calls": {site: len(calls[site]) for site in SITES},
                "claimed_score": readout_v2.complete_label_log_score(
                    first.detach().cpu().numpy().astype(np.float32),
                    after.detach().cpu().numpy().astype(np.float32), label_id),
                "autograd_score": float(score.detach().cpu())}
            require(abs(entry["claimed_score"]-entry["autograd_score"]) <= 1e-6,
                    "autograd and saved-logit scores disagree")
            entry["sidecars"] = {
                "first_logits": save_array(folder, f"{label}_first_logits",
                                          first.detach().cpu().numpy().astype(np.float32)),
                "after_label_logits": save_array(folder, f"{label}_after_logits",
                                          after.detach().cpu().numpy().astype(np.float32)),
                "states": {}, "gradients": {}}
            for site, grad in zip(SITES, gradients, strict=True):
                key = site.replace(".", "_")
                entry["sidecars"]["states"][site] = save_array(folder,
                    f"{label}_{key}_state",
                    calls[site][0][0, len(prompt_ids)-1].detach().cpu().numpy().astype(np.float32))
                entry["sidecars"]["gradients"][site] = save_array(folder,
                    f"{label}_{key}_vjp",
                    grad[0, len(prompt_ids)-1].detach().cpu().numpy().astype(np.float32))
            records.append(entry)
            measure_mps(torch, f"after_{label}_vjp", folder)
        finally:
            for handle in reversed(handles):
                handle.remove()
        require(sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
                < MAX_EVIDENCE_BYTES, "pilot evidence cap exceeded")
    return records


def generation_pair(model: Any, batch: dict[str, Any], module: Any,
                    processor: Any, torch: Any) -> dict[str, Any]:
    prompt_length = int(batch["input_ids"].shape[1])
    settings = load_spec()["generation"]
    def generate(*, read_only: bool) -> tuple[list[int], int]:
        calls = []
        handle = None
        if read_only:
            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                calls.append(tuple(_first(output).shape))
                return None
            handle = module.register_forward_hook(capture)
        try:
            with torch.no_grad():
                result = model.generate(**batch, **settings,
                                        eos_token_id=readout_v2.EOS_TOKEN_ID,
                                        return_dict_in_generate=True)
            return [int(x) for x in result.sequences[0, prompt_length:].tolist()], len(calls)
        finally:
            if handle is not None:
                handle.remove()
    first, _ = generate(read_only=False)
    second, hook_calls = generate(read_only=True)
    first_result = endpoint_v2.score_complete_answer(first, processor.tokenizer)
    second_result = endpoint_v2.score_complete_answer(second, processor.tokenizer)
    require(first == second, "read-only generation changes complete token path")
    return {"no_hook": first_result, "read_only": second_result,
            "read_only_hook_calls": hook_calls,
            "raw_no_hook_ids": first, "raw_read_only_ids": second}


def run_worker(attempt: Path) -> None:
    validate_go()  # Before importing torch, Transformers, or loading any weights.
    require(attempt.resolve() == ATTEMPT.resolve() and attempt.is_dir()
            and not (attempt / "worker_output").exists(),
            "fixed new pilot output required")
    pre = json.loads((attempt / "pre_host.json").read_text())
    from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host
    host.validate_power(pre)
    import torch
    from models.qwen_omni.adapter import load_pinned_thinker, resolve_hook_points
    require(torch.backends.mps.is_available(), "typed pilot requires MPS")
    fraction = min(1.0, MAX_MPS_BYTES / int(torch.mps.recommended_max_memory()))
    torch.mps.set_per_process_memory_fraction(fraction)
    output = attempt / "worker_output"
    output.mkdir()
    write_once(output / "start.json", {"schema": "qwen_action_typed_pilot_worker_v1",
               "status": "started", "case_id": "F_L1__typed",
               "go_sha256": sha(GO), "pre_host_sha256": sha(attempt / "pre_host.json")})
    model, processor = load_pinned_thinker(allow_download=False,
        dtype=torch.float32, device_map="mps", low_cpu_mem_usage=True)
    require(model.dtype == torch.float32
            and int(processor.tokenizer.eos_token_id) == readout_v2.EOS_TOKEN_ID,
            "pinned model dtype/tokenizer changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    measure_mps(torch, "after_model_load", output)
    row = fixed_case()
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": load_spec()["system_prompt"]}]},
        {"role": "user", "content": [{"type": "text", "text": row["text"]}]},
    ]
    prepared = processor.apply_chat_template(conversation, tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    require(set(prepared).issubset({"input_ids", "attention_mask"}),
            "typed processor unexpectedly added modality inputs")
    batch = {key: value.to("mps") for key, value in prepared.items()}
    points = resolve_hook_points(model, audio_layers=(), text_layers=(17, 34))
    modules = {point.name: point.module for point in points if point.name in SITES}
    require(set(modules) == set(SITES), "pinned residual hook sites absent")
    triplet = teacher_forced_triplet(model, batch, modules, torch, output)
    generated = generation_pair(model, batch, modules["text.residual.34"],
                                processor, torch)
    measure_mps(torch, "after_generation", output)
    # The read-only contract rechecks every saved FP32 array and score.
    write_once(output / "result.json", {"schema": "qwen_action_typed_pilot_worker_v1",
               "status": "candidate_saved_pending_parent_and_audit",
               "case_id": "F_L1__typed", "model_revision": load_spec()["model_revision"],
               "prompt_text": row["text"], "triplet": triplet,
               "generation": generated,
               "generation_config": settings_snapshot(model),
               "mps_log_sha256": sha(output / "mps_samples.jsonl")})


def settings_snapshot(model: Any) -> dict[str, Any]:
    config = model.generation_config.to_dict()
    return {key: config.get(key) for key in
            ("eos_token_id", "pad_token_id", "bos_token_id", "max_new_tokens",
             "do_sample", "temperature", "top_p", "num_beams")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.source_preflight:
        print(json.dumps({"status": "go_required_before_weights",
                          "go_exists": GO.is_file(), "attempt": str(ATTEMPT)}, sort_keys=True))
        return
    require(args.worker and args.attempt is not None,
            "only fixed typed worker or source preflight exists")
    run_worker(args.attempt)


if __name__ == "__main__":
    main()
