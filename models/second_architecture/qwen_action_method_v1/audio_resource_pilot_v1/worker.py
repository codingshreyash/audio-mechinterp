"""One non-study Qwen audio forward/generation parity worker; no causal edits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import signal
import stat
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import gate, plan

SCHEMA = "qwen_audio_resource_worker_v1"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save_array(folder: Path, name: str, value: np.ndarray) -> dict:
    array = np.asarray(value)
    gate.require(array.size > 0 and np.isfinite(array).all(), "invalid array sidecar")
    path = folder / f"{name}.npy"
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "sha256": plan.sha(path),
            "shape": list(array.shape), "dtype": str(array.dtype)}


def _first(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def stop_reason(ids: list[int], eos: int) -> str:
    gate.require(len(ids) <= plan.MAX_NEW_TOKENS
                 and all(type(x) is int and x >= 0 for x in ids),
                 "invalid generated path")
    if not ids:
        return "empty"
    if ids[-1] == eos:
        return "eos"
    if eos in ids[:-1]:
        return "early_eos_malformed"
    if len(ids) == plan.MAX_NEW_TOKENS:
        return "length_censored"
    return "other_stop"


def compare_paths(a: list[int], b: list[int], eos: int) -> dict:
    reason_a, reason_b = stop_reason(a, eos), stop_reason(b, eos)
    gate.require(a == b and reason_a == reason_b,
                 "stock/read-only generated path or stop reason diverged")
    return {"stock_ids": a, "read_only_ids": b,
            "stock_stop_reason": reason_a, "read_only_stop_reason": reason_b,
            "terminal_eos": reason_a == "eos", "exact_path_parity": True}


def _parent_admission(attempt: Path) -> dict:
    """Local-owner liveness guard plus committed GO. Not a cryptographic parent proof."""
    gate.require(attempt.resolve() == plan.ATTEMPT.resolve()
                 and attempt.is_dir() and not (attempt / "worker_output").exists(),
                 "fixed fresh worker attempt required")
    start = gate.read(attempt / "start.json")
    gate.require(start.get("schema") == "qwen_audio_resource_parent_v1"
                 and start.get("status") == "worker_authorized"
                 and start.get("attempt") == str(plan.ATTEMPT.resolve())
                 and start.get("parent_pid") == os.getppid()
                 and start.get("go_sha256") == plan.sha(plan.GO)
                 and 0 < start.get("deadline_monotonic", 0) - time.monotonic()
                    <= plan.MAX_SECONDS,
                 "bounded parent start invalid")
    fd_text = os.environ.get("QWEN_AUDIO_RESOURCE_PARENT_FD", "")
    gate.require(fd_text.isdecimal(), "inherited parent pipe absent")
    fd = int(fd_text)
    gate.require(fd > 2 and stat.S_ISFIFO(os.fstat(fd).st_mode),
                 "parent pipe invalid")
    ready, _, _ = select.select([fd], [], [], 2)
    gate.require(bool(ready), "parent nonce timed out")
    nonce = b""
    while not nonce.endswith(b"\n") and len(nonce) <= 65:
        part = os.read(fd, 1)
        gate.require(bool(part), "parent pipe closed")
        nonce += part
    gate.require(len(nonce) == 65 and nonce.endswith(b"\n"), "nonce malformed")
    try:
        raw = bytes.fromhex(nonce[:-1].decode())
    except (UnicodeDecodeError, ValueError):
        raw = b""
    gate.require(len(raw) == 32
                 and _sha_bytes(raw) == start.get("nonce_sha256"),
                 "parent nonce invalid")
    def alarm(_sig: int, _frame: Any) -> None:
        raise TimeoutError("audio pilot worker deadline")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, start["deadline_monotonic"]-time.monotonic())
    def watch() -> None:
        while True:
            ready, _, _ = select.select([fd], [], [], 3)
            if not ready or os.read(fd, 4096) == b"":
                os._exit(88)
    threading.Thread(target=watch, daemon=True, name="audio-pilot-parent-watch").start()
    go = gate.validate_go(live=False)
    gate.require(start.get("source_identity") == go["source_identity"]
                 and start.get("runtime_identity") == go["runtime_identity"],
                 "parent GO identity mismatch")
    return start


def measure_cuda(torch: Any, folder: Path, phase: str) -> dict:
    torch.cuda.synchronize(0)
    row = {"phase": phase, "allocated_bytes": int(torch.cuda.memory_allocated(0)),
           "reserved_bytes": int(torch.cuda.memory_reserved(0)),
           "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}
    gate.require(0 <= row["allocated_bytes"] <= row["reserved_bytes"]
                 <= row["peak_reserved_bytes"] < plan.GPU_BYTES,
                 "framework GPU cap exceeded")
    with (folder / "cuda_samples.jsonl").open("ab") as stream:
        stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return row


def _observe(modules: dict, call_shapes: dict, states: dict, *, prompt_last: int):
    handles = []
    for site in plan.SITES:
        def hook(_module: Any, _inputs: Any, output: Any, *, key: str = site) -> None:
            state = _first(output)
            gate.require(state.ndim == 3 and state.shape[0] == 1
                         and state.shape[-1] == 2048 and state.shape[1] >= 1,
                         "Thinker read-only state shape invalid")
            call_shapes[key].append(list(state.shape))
            if key not in states and state.shape[1] > prompt_last:
                states[key] = state[0, prompt_last].detach().float().cpu().numpy().astype(np.float32)
            return None
        handles.append(modules[site].register_forward_hook(hook))
    return handles


def require_same_batch(batch: dict, original: dict, torch: Any) -> None:
    gate.require(set(batch) == set(original)
                 and all(torch.equal(batch[key].detach().cpu(), original[key])
                         for key in original),
                 "processor tensor changed between stock/read-only calls")


def run(attempt: Path) -> None:
    start = _parent_admission(attempt)  # Before torch, processor, or weights.
    plan.verify_wav()
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from models.qwen_omni.adapter import prepare_audio_batch, resolve_hook_points
    from ..audio_candidate_v1 import validate_audio_batch
    gate.require(os.getppid() == start["parent_pid"]
                 and gate.runtime_identity(require_idle=False) == start["runtime_identity"],
                 "runtime changed after admission")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    folder = attempt / "worker_output"
    folder.mkdir()
    gate.write_once(folder / "start.json", {"schema": SCHEMA, "status": "started",
        "case_id": plan.CASE_ID, "parent_start_sha256": plan.sha(attempt / "start.json"),
        "go_sha256": plan.sha(plan.GO), "parent_pid": start["parent_pid"]})
    snapshot = Path(gate.read(plan.ADMISSION)["processor_preflight"]["snapshot"]["snapshot_path"])
    from .. import typed_pilot_cuda_admission_v3
    gate.require(typed_pilot_cuda_admission_v3.verify_snapshot(snapshot)
                 == gate.read(plan.ADMISSION)["processor_preflight"]["snapshot"],
                 "worker cached checkpoint bytes differ from admission")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True)
    batch_cpu, digest = prepare_audio_batch(processor, plan.WAV,
                                            system_prompt=plan.SYSTEM_PROMPT)
    gate.require(digest == plan.WAV_SHA256, "processor audio bytes changed")
    record = validate_audio_batch(batch_cpu, processor, wav_sha256=digest,
                                  case_id=plan.CASE_ID)
    plan.verify_geometry(record)
    sidecars = {key: save_array(folder, f"batch_{key}", value.cpu().numpy())
                for key, value in batch_cpu.items()}
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float32,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    gate.require(model.dtype == torch.float32
                 and int(processor.tokenizer.eos_token_id) == 151645,
                 "pinned Thinker dtype/EOS changed")
    points = resolve_hook_points(model, audio_layers=(), text_layers=(17, 34))
    modules = {point.name: point.module for point in points if point.name in plan.SITES}
    gate.require(set(modules) == set(plan.SITES), "shared Thinker hook sites missing")
    measure_cuda(torch, folder, "after_model_load")
    batch = {key: value.to("cuda:0") for key, value in batch_cpu.items()}
    prompt_last = int(batch["input_ids"].shape[1]-1)
    with torch.no_grad():
        native = model(**batch, use_cache=False, return_dict=True)
    require_same_batch(batch, batch_cpu, torch)
    native_logits = native.logits[0, prompt_last].float().detach().cpu().numpy().astype(np.float32)
    del native
    calls = {site: [] for site in plan.SITES}
    states: dict[str, np.ndarray] = {}
    handles = _observe(modules, calls, states, prompt_last=prompt_last)
    try:
        with torch.no_grad():
            observed = model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in reversed(handles):
            handle.remove()
    require_same_batch(batch, batch_cpu, torch)
    gate.require(all(len(calls[site]) == 1 for site in plan.SITES),
                 "read-only forward hook count invalid")
    observed_logits = observed.logits[0, prompt_last].float().detach().cpu().numpy().astype(np.float32)
    del observed
    gate.require(native_logits.shape == observed_logits.shape
                 and native_logits.ndim == 1
                 and native_logits.size > 151645
                 and np.isfinite(native_logits).all()
                 and np.isfinite(observed_logits).all(),
                 "first-step full vocabulary logits invalid")
    delta = float(np.max(np.abs(native_logits - observed_logits)))
    gate.require(delta <= plan.FIRST_LOGIT_TOLERANCE,
                 "stock/read-only first-step logits diverged")
    reference = {"stock_first_logits": save_array(folder, "stock_first_logits", native_logits),
                 "read_only_first_logits": save_array(folder, "read_only_first_logits", observed_logits),
                 "read_only_states": {site: save_array(folder, f"state_{site.replace('.', '_')}", states[site])
                                      for site in plan.SITES},
                 "first_logits_maxabs": delta,
                 "forward_hook_shapes": calls}
    measure_cuda(torch, folder, "after_forward")
    generation_shapes = {site: [] for site in plan.SITES}
    generation_states: dict[str, np.ndarray] = {}
    settings = {"max_new_tokens": plan.MAX_NEW_TOKENS, "do_sample": False,
                "use_cache": True, "eos_token_id": 151645,
                "return_dict_in_generate": True}
    def generate(read_only: bool) -> list[int]:
        hooks = _observe(modules, generation_shapes, generation_states,
                         prompt_last=prompt_last) if read_only else []
        try:
            with torch.no_grad():
                value = model.generate(**batch, **settings)
            raw = value.sequences[0, prompt_last+1:].tolist()
            return [int(x) for x in raw]
        finally:
            for handle in reversed(hooks):
                handle.remove()
    stock_ids = generate(False)
    require_same_batch(batch, batch_cpu, torch)
    read_only_ids = generate(True)
    require_same_batch(batch, batch_cpu, torch)
    gate.require(all(generation_shapes[site] for site in plan.SITES),
                 "read-only generation hook did not fire")
    generation = compare_paths(stock_ids, read_only_ids, 151645)
    generation["read_only_hook_shapes"] = generation_shapes
    measure_cuda(torch, folder, "after_generation")
    gate.require(sum(x.stat().st_size for x in attempt.rglob("*") if x.is_file())
                 < plan.EVIDENCE_BYTES, "worker evidence cap exceeded")
    gate.write_once(folder / "result.json", {"schema": SCHEMA,
        "status": "candidate_saved_pending_parent_and_audit",
        "case_id": plan.CASE_ID, "model_revision": "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        "processor_record": record, "batch_sidecars": sidecars,
        "processor_tensors_unchanged_all_calls": True,
        "reference": reference, "generation": generation,
        "cuda_log_sha256": plan.sha(folder / "cuda_samples.jsonl")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    run(args.attempt)


if __name__ == "__main__":
    main()
