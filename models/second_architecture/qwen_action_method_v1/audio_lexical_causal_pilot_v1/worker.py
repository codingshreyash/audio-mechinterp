"""One-shot Qwen audio red/blue baseline and causal state-swap worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import signal
import stat
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import gate, plan

SCHEMA = "qwen_audio_lexical_worker_v1"
LABEL = re.compile(r"\s*(red|blue)\.?\s*", re.IGNORECASE)


def save_array(folder: Path, name: str, value: np.ndarray) -> dict:
    array = np.asarray(value)
    gate.require(array.size > 0 and np.isfinite(array).all(), "invalid sidecar")
    path = folder / f"{name}.npy"
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "sha256": plan.sha(path),
            "shape": list(array.shape), "dtype": str(array.dtype)}


def stop_reason(ids: list[int]) -> str:
    gate.require(len(ids) <= plan.MAX_NEW_TOKENS and all(type(x) is int and x >= 0 for x in ids),
                 "invalid generated path")
    if not ids:
        return "empty"
    if ids[-1] == plan.EOS_ID and plan.EOS_ID not in ids[:-1]:
        return "eos"
    if plan.EOS_ID in ids[:-1]:
        return "early_eos_malformed"
    if len(ids) == plan.MAX_NEW_TOKENS:
        return "length_censored"
    return "other_stop"


def classify(ids: list[int], tokenizer: Any) -> dict:
    reason = stop_reason(ids)
    decoded_raw = tokenizer.decode(ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
    content = ids[:-1] if reason == "eos" else ids
    decoded = tokenizer.decode(content, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    special = set(int(x) for x in tokenizer.all_special_ids)
    valid_content = reason == "eos" and all(x not in special for x in content)
    match = LABEL.fullmatch(decoded) if valid_content else None
    return {"ids": ids, "stop_reason": reason, "decoded": decoded,
            "decoded_with_special": decoded_raw,
            "label": match.group(1).lower() if match else None,
            "terminal_eos": reason == "eos"}


def _parent_admission(attempt: Path) -> dict:
    gate.require(attempt.resolve() == plan.ATTEMPT.resolve()
                 and attempt.is_dir() and not (attempt / "worker_output").exists(),
                 "fixed fresh worker attempt required")
    start = gate.read(attempt / "start.json")
    gate.require(start.get("schema") == "qwen_audio_lexical_parent_v1"
                 and start.get("status") == "worker_authorized"
                 and start.get("attempt") == str(plan.ATTEMPT.resolve())
                 and start.get("parent_pid") == os.getppid()
                 and start.get("go_sha256") == plan.sha(plan.GO)
                 and 0 < start.get("deadline_monotonic", 0) - time.monotonic()
                    <= plan.MAX_SECONDS,
                 "bounded parent start invalid")
    fd_text = os.environ.get("QWEN_AUDIO_LEXICAL_PARENT_FD", "")
    gate.require(fd_text.isdecimal(), "inherited parent pipe absent")
    fd = int(fd_text)
    gate.require(fd > 2 and stat.S_ISFIFO(os.fstat(fd).st_mode), "parent pipe invalid")
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
    gate.require(len(raw) == 32 and hashlib.sha256(raw).hexdigest() == start.get("nonce_sha256"),
                 "parent nonce invalid")
    def alarm(_sig: int, _frame: Any) -> None:
        raise TimeoutError("lexical worker deadline")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, start["deadline_monotonic"]-time.monotonic())
    def watch() -> None:
        while True:
            ready, _, _ = select.select([fd], [], [], 3)
            if not ready or os.read(fd, 4096) == b"":
                os._exit(88)
    threading.Thread(target=watch, daemon=True, name="lexical-parent-watch").start()
    go = gate.validate_go(live=False)
    gate.require(start.get("source_identity") == go["source_identity"]
                 and start.get("runtime_identity") == go["runtime_identity"],
                 "parent GO identity mismatch")
    return start


def measure_cuda(torch: Any, folder: Path, phase: str) -> None:
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


def progress(folder: Path, row: dict) -> None:
    with (folder / "progress.jsonl").open("ab") as stream:
        stream.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def require_same_batch(batch: dict, original: dict, torch: Any) -> None:
    gate.require(set(batch) == set(original)
                 and all(torch.equal(batch[key].detach().cpu(), original[key]) for key in original),
                 "processor tensor changed between calls")


def _modules(model: Any) -> dict:
    thinker = getattr(model, "thinker", model)
    core = thinker.model
    gate.require(len(core.audio_tower.layers) == 32
                 and core.audio_tower.proj.out_features == 2048,
                 "pinned Qwen audio module layout changed")
    return {"audio.pre_attention.0": core.audio_tower.layers[0],
            "audio.to_language": core.audio_tower.proj}


def _state(value: Any, site: str) -> Any:
    gate.require(hasattr(value, "ndim") and value.ndim == 2
                 and list(value.shape) == [plan.WINDOWS[site]["total"],
                                           plan.WINDOWS[site]["width"]],
                 f"{site} state shape changed")
    return value


def hook_state(modules: dict, callback: Any) -> list:
    """Register the exact pre-attention input and post-projection output sites."""
    def pre(_module: Any, args: tuple, kwargs: dict):
        gate.require(bool(args) and "cu_seqlens" in kwargs,
                     "pre-attention call signature changed")
        hidden = _state(args[0], "audio.pre_attention.0")
        replacement = callback("audio.pre_attention.0", hidden)
        return ((replacement, *args[1:]), kwargs) if replacement is not None else None
    def post(_module: Any, _args: tuple, output: Any):
        hidden = _state(output, "audio.to_language")
        replacement = callback("audio.to_language", hidden)
        return replacement
    return [modules["audio.pre_attention.0"].register_forward_pre_hook(pre, with_kwargs=True),
            modules["audio.to_language"].register_forward_hook(post)]


def hook_one(modules: dict, site: str, callback: Any) -> Any:
    if site == "audio.pre_attention.0":
        def pre(_module: Any, args: tuple, kwargs: dict):
            gate.require(bool(args) and "cu_seqlens" in kwargs,
                         "pre-attention call signature changed")
            hidden = _state(args[0], site)
            return ((callback(hidden), *args[1:]), kwargs)
        return modules[site].register_forward_pre_hook(pre, with_kwargs=True)
    def post(_module: Any, _args: tuple, output: Any):
        return callback(_state(output, site))
    return modules[site].register_forward_hook(post)


def _generate(model: Any, batch: dict, tokenizer: Any, torch: Any,
              hooks: list | None = None) -> dict:
    try:
        with torch.no_grad():
            output = model.generate(**batch, max_new_tokens=plan.MAX_NEW_TOKENS,
                                    do_sample=False, use_cache=True,
                                    eos_token_id=plan.EOS_ID,
                                    return_dict_in_generate=True)
        ids = [int(x) for x in output.sequences[0, batch["input_ids"].shape[1]:].tolist()]
        return classify(ids, tokenizer)
    finally:
        for handle in reversed(hooks or []):
            handle.remove()


def _capture_forward(model: Any, batch: dict, modules: dict, torch: Any,
                     prompt_last: int, *, observe: bool) -> tuple[np.ndarray, dict]:
    captures: dict[str, list] = {site: [] for site in plan.SITES}
    def record(site: str, hidden: Any) -> None:
        captures[site].append(hidden.detach().clone())
        return None
    hooks = hook_state(modules, record) if observe else []
    try:
        with torch.no_grad():
            value = model(**batch, use_cache=False, return_dict=True)
        logits = value.logits[0, prompt_last].float().detach().cpu().numpy().astype(np.float32)
        gate.require(logits.ndim == 1 and logits.size > plan.EOS_ID and np.isfinite(logits).all(),
                     "full first-step logits invalid")
        if observe:
            gate.require(all(len(captures[site]) == 1 for site in plan.SITES),
                         "read-only forward hook census changed")
        return logits, {site: captures[site][0] for site in plan.SITES} if observe else {}
    finally:
        for handle in reversed(hooks):
            handle.remove()


def _maxabs(value: Any, torch: Any) -> float:
    return float(torch.max(torch.abs(value)).item())


def make_edit(site: str, arm: str, live: Any, recipient: Any, donor: Any,
              seed: int, torch: Any) -> tuple[Any, dict]:
    w = plan.WINDOWS[site]
    lo, hi = w["local"]
    x, y = w["relocated"]
    a, b = w["chunk"]
    delta = donor[lo:hi] - recipient[lo:hi]
    local_norm = float(torch.linalg.vector_norm(delta).item())
    gate.require(local_norm >= plan.MIN_LOCAL_DELTA_L2 and np.isfinite(local_norm),
                 "local donor delta below frozen numerical floor")
    patched = live.clone()
    if arm == "local_donor":
        patched[lo:hi] = donor[lo:hi]
    elif arm == "chunk_donor":
        patched[a:b] = donor[a:b]
    elif arm == "full_donor":
        patched[:] = donor
    elif arm == "self":
        patched[:] = recipient
    elif arm == "relocated_delta":
        patched[x:y] += delta
    elif arm == "equal_l2_noise":
        rng = np.random.default_rng(seed)
        noise = torch.from_numpy(rng.standard_normal(tuple(delta.shape)).astype(np.float32)).to(live.device)
        noise = noise * (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(noise))
        patched[lo:hi] += noise
    else:
        raise ValueError(f"Unknown arm {arm}")
    realized = float(torch.linalg.vector_norm(patched-live).item())
    if arm in ("relocated_delta", "equal_l2_noise"):
        gate.require(abs(realized-local_norm) <= max(1e-4, local_norm*1e-4),
                     "equal-L2 control edit norm changed")
    return patched, {"local_delta_l2": local_norm, "realized_edit_l2": realized,
                     "seed": seed if arm == "equal_l2_noise" else None,
                     "window": w["relocated"] if arm == "relocated_delta" else
                               (w["chunk"] if arm == "chunk_donor" else
                                ([0, w["total"]] if arm in ("full_donor", "self") else w["local"]))}


def run(attempt: Path) -> None:
    start = _parent_admission(attempt)  # Before Torch, processor, or weights.
    plan.verify_inputs()
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from models.qwen_omni.adapter import MODEL_REVISION, prepare_audio_batch
    from ..audio_candidate_v1 import validate_audio_batch
    gate.require(os.getppid() == start["parent_pid"]
                 and gate.runtime_identity(require_idle=False) == start["runtime_identity"],
                 "runtime changed after admission")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    folder = attempt / "worker_output"
    folder.mkdir()
    gate.write_once(folder / "start.json", {"schema": SCHEMA, "status": "started",
        "parent_start_sha256": plan.sha(attempt / "start.json"),
        "go_sha256": plan.sha(plan.GO), "parent_pid": start["parent_pid"]})
    admission = gate.read(plan.ADMISSION)
    snapshot = Path(admission["processor_preflight"]["snapshot"]["snapshot_path"])
    from .. import typed_pilot_cuda_admission_v3
    gate.require(typed_pilot_cuda_admission_v3.verify_snapshot(snapshot)
                 == admission["processor_preflight"]["snapshot"],
                 "worker cached checkpoint bytes differ from admission")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True)
    gate.require(int(processor.tokenizer.eos_token_id) == plan.EOS_ID,
                 "pinned tokenizer EOS changed")
    cpu_batches, records, sidecars = {}, {}, {}
    for family, paths in (("calibrated", plan.CALIBRATED), ("original", plan.ORIGINAL)):
        cpu_batches[family], records[family], sidecars[family] = {}, {}, {}
        for word, wav in paths.items():
            batch, digest = prepare_audio_batch(processor, wav,
                                                system_prompt=plan.SYSTEM_PROMPT)
            gate.require(digest == plan.EXPECTED_WAV_SHA256[family][word],
                         "processor WAV bytes changed")
            record = validate_audio_batch(batch, processor, wav_sha256=digest,
                case_id=f"LEXICAL_{family.upper()}_{word.upper()}")
            plan.verify_record(family, word, record,
                               admission["processor_runtime_receipt"])
            cpu_batches[family][word] = batch
            records[family][word] = record
            sidecars[family][word] = {key: save_array(folder,
                f"batch_{family}_{word}_{key}", value.detach().cpu().numpy())
                for key, value in batch.items()}
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float32,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    gate.require(model.dtype == torch.float32
                 and MODEL_REVISION == "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
                 "pinned Thinker dtype/revision changed")
    modules = _modules(model)
    measure_cuda(torch, folder, "after_model_load")
    batches = {family: {word: {key: value.to("cuda:0") for key, value in batch.items()}
                        for word, batch in words.items()}
               for family, words in cpu_batches.items()}
    baselines = {"calibrated": {}, "original": {}}
    for word in plan.WORDS:
        row = _generate(model, batches["calibrated"][word], processor.tokenizer, torch)
        require_same_batch(batches["calibrated"][word], cpu_batches["calibrated"][word], torch)
        baselines["calibrated"][word] = row
        progress(folder, {"kind": "calibrated_baseline", "word": word, "output": row})
        measure_cuda(torch, folder, f"calibrated_baseline_{word}")
    for word in plan.WORDS:
        row = _generate(model, batches["original"][word], processor.tokenizer, torch)
        require_same_batch(batches["original"][word], cpu_batches["original"][word], torch)
        baselines["original"][word] = row
        progress(folder, {"kind": "original_observation", "word": word, "output": row})
        measure_cuda(torch, folder, f"original_observation_{word}")
    baseline_pass = all(baselines["calibrated"][word]["label"] == word
                        and baselines["calibrated"][word]["terminal_eos"]
                        for word in plan.WORDS)
    if not baseline_pass:
        _finish(folder, attempt, records, sidecars, baselines, None, [],
                "baseline_failed_no_interventions")
        return
    sources, references = {}, {}
    for word in plan.WORDS:
        batch = batches["calibrated"][word]
        prompt_last = int(batch["input_ids"].shape[1] - 1)
        stock_logits, _ = _capture_forward(model, batch, modules, torch, prompt_last, observe=False)
        read_logits, captured = _capture_forward(model, batch, modules, torch, prompt_last, observe=True)
        require_same_batch(batch, cpu_batches["calibrated"][word], torch)
        diff = float(np.max(np.abs(stock_logits-read_logits)))
        gate.require(diff <= plan.FIRST_LOGIT_TOLERANCE,
                     "stock/read-only first logits diverged")
        reference_sidecars = {"stock_logits": save_array(folder, f"{word}_stock_logits", stock_logits),
                              "read_only_logits": save_array(folder, f"{word}_read_logits", read_logits),
                              "states": {site: save_array(folder,
                                  f"{word}_{site.replace('.', '_')}_state",
                                  value.detach().float().cpu().numpy().astype(np.float32))
                                  for site, value in captured.items()}}
        calls: dict[str, int] = {site: 0 for site in plan.SITES}
        live_drift: dict[str, float] = {}
        def observe(site: str, hidden: Any) -> None:
            calls[site] += 1
            live_drift[site] = _maxabs(hidden-captured[site], torch)
            return None
        read_only = _generate(model, batch, processor.tokenizer, torch,
                              hook_state(modules, observe))
        gate.require(all(calls[site] == 1 for site in plan.SITES)
                     and all(live_drift[site] <= plan.SOURCE_DRIFT_TOLERANCE
                             for site in plan.SITES)
                     and read_only["ids"] == baselines["calibrated"][word]["ids"],
                     "read-only hook path/census/source differs from stock")
        references[word] = {"first_logits_maxabs": diff, "sidecars": reference_sidecars,
                            "read_only_output": read_only, "read_only_calls": calls,
                            "source_drift_maxabs": live_drift}
        sources[word] = captured
        progress(folder, {"kind": "read_only_reference", "word": word,
                          "output": read_only, "first_logits_maxabs": diff,
                          "source_drift_maxabs": live_drift})
        measure_cuda(torch, folder, f"read_only_{word}")
    outside = {}
    pre = sources["red"]["audio.pre_attention.0"]-sources["blue"]["audio.pre_attention.0"]
    projection = sources["red"]["audio.to_language"]-sources["blue"]["audio.to_language"]
    lo, hi = plan.WINDOWS["audio.pre_attention.0"]["local"]
    outside["audio.pre_attention.0"] = max(_maxabs(pre[:lo], torch), _maxabs(pre[hi:], torch))
    outside["audio.to_language"] = _maxabs(projection[50:], torch)
    gate.require(all(value <= plan.OUTSIDE_SUPPORT_TOLERANCE for value in outside.values()),
                 "unexpected source-state spread outside source chunk/support")
    interventions = []
    for site_index, site in enumerate(plan.SITES):
        for direction_index, recipient_word in enumerate(plan.WORDS):
            donor_word = "blue" if recipient_word == "red" else "red"
            recipient = sources[recipient_word][site]
            donor = sources[donor_word][site]
            batch = batches["calibrated"][recipient_word]
            for arm in plan.ARMS:
                calls = []
                seed = plan.NOISE_SEED + site_index*100 + direction_index*10
                def edit(live: Any) -> Any:
                    gate.require(not calls, "intervention hook fired more than once")
                    drift_tensor = live-recipient
                    drift = _maxabs(drift_tensor, torch)
                    drift_l2 = float(torch.linalg.vector_norm(drift_tensor).item())
                    local_lo, local_hi = plan.WINDOWS[site]["local"]
                    signal_l2 = float(torch.linalg.vector_norm(
                        donor[local_lo:local_hi]-recipient[local_lo:local_hi]).item())
                    gate.require(drift <= plan.SOURCE_DRIFT_TOLERANCE
                                 and signal_l2 >= plan.MIN_LOCAL_DELTA_L2
                                 and drift_l2 <= signal_l2*plan.MAX_PREPATCH_DRIFT_FRACTION,
                                 "intervention live source differs from captured recipient")
                    patched, info = make_edit(site, arm, live, recipient, donor, seed, torch)
                    calls.append({"source_drift_maxabs": drift,
                                  "source_drift_l2": drift_l2, **info})
                    return patched
                output = _generate(model, batch, processor.tokenizer, torch,
                                   [hook_one(modules, site, edit)])
                gate.require(len(calls) == 1, "intervention hook census changed")
                require_same_batch(batch, cpu_batches["calibrated"][recipient_word], torch)
                row = {"site": site, "recipient": recipient_word, "donor": donor_word,
                       "arm": arm, "output": output, "hook_calls": len(calls), **calls[0]}
                interventions.append(row)
                progress(folder, {"kind": "intervention", **row})
                measure_cuda(torch, folder,
                             f"intervention_{site_index}_{direction_index}_{arm}")
                if arm == "self":
                    gate.require(output["ids"] == baselines["calibrated"][recipient_word]["ids"],
                                 "self-patch complete token path differs from stock recipient")
                if arm == "full_donor":
                    gate.require(output["ids"] == baselines["calibrated"][donor_word]["ids"],
                                 "full-donor complete token path differs from stock donor")
    _finish(folder, attempt, records, sidecars, baselines,
            {"references": references, "outside_source_difference_maxabs": outside},
            interventions, "candidate_saved_pending_parent_and_audit")


def _finish(folder: Path, attempt: Path, records: dict, sidecars: dict,
            baselines: dict, reference: dict | None, interventions: list,
            status: str) -> None:
    gate.require(sum(x.stat().st_size for x in attempt.rglob("*") if x.is_file())
                 < plan.EVIDENCE_BYTES, "worker evidence cap exceeded")
    gate.write_once(folder / "result.json", {"schema": SCHEMA, "status": status,
        "model_revision": "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        "processor_records": records, "batch_sidecars": sidecars,
        "baselines": baselines, "reference": reference,
        "interventions": interventions,
        "progress_sha256": plan.sha(folder / "progress.jsonl"),
        "cuda_log_sha256": plan.sha(folder / "cuda_samples.jsonl")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    run(args.attempt)


if __name__ == "__main__":
    main()
