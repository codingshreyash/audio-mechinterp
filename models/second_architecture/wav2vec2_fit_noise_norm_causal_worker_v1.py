"""One-shot 88-forward FIT Wav2Vec2 normalization experiment.

`admit` is no-weight. `run` is locked behind a later independent GO receipt.
The parent launcher owns the wall-clock and host gates; this worker also
checks its own memory, disk, forward count, source bytes and admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_source_v1 as source
from models.second_architecture import wav2vec2_fit_noise_norm_causal_88_plan_v1 as plan88

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots"
            / "22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
WEIGHTS_SHA = "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a"
MODELING_SHA = "0c5027789d81a622f72f062eab1ee16566bb8dfb200cb6da0a9ffef06543b287"
SOURCE_FILES = (
    "WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_PROSPECTIVE_V1.md",
    "WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_88_FORWARD_AMENDMENT_V1.md",
    "wav2vec2_fit_noise_norm_causal_source_v1.py",
    "wav2vec2_fit_noise_norm_causal_88_plan_v1.py",
    "test_wav2vec2_fit_noise_norm_causal_88_plan_v1.py",
    "wav2vec2_fit_noise_norm_causal_worker_v1.py",
    "wav2vec2_fit_noise_norm_causal_launcher_v1.py",
    "wav2vec2_fit_noise_norm_causal_audit_v1.py",
    "wav2vec2_fit_noise_norm_causal_host_v1.py",
    "wav2vec2_fit_baseline_stage_capture_v1.py",
    "test_wav2vec2_fit_noise_norm_causal_worker_v1.py",
    "WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_EXECUTABLE_HANDOFF_V1.md",
)
MAX_FORWARDS = 88
MAX_SECONDS = 1200
MAX_RSS = 4 * 1024**3
MAX_OUTPUT = 2 * 1024**3
RSS_RESERVE = 768 * 1024**2
OUTPUT_RESERVE = 96 * 1024**2
STAGES = ("input_values", "first_conv_raw", "first_conv_norm",
          "feature_projection", "post_block0")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def bytes_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha(value) -> str:
    import numpy as np
    a = np.ascontiguousarray(value)
    return bytes_sha(a.tobytes())


def source_binding() -> dict:
    return {name: source.sha_file(HERE / name) for name in SOURCE_FILES}


def committed_sources() -> dict:
    binding = source_binding()
    for name in binding:
        relative = f"models/second_architecture/{name}"
        for flag in ((), ("--cached",)):
            code = subprocess.run(("git", "diff", *flag, "--quiet", "--", relative),
                                  cwd=ROOT, check=False).returncode
            require(code == 0, f"dirty study source: {name}")
        commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--", relative),
                                         cwd=ROOT, text=True).strip()
        require(len(commit) == 40, f"uncommitted study source: {name}")
    return binding


def no_weight_admission() -> dict:
    import numpy as np
    import torch
    import transformers
    from transformers.models.wav2vec2 import modeling_wav2vec2
    from models.second_architecture import wav2vec2_fit_baseline_stage_capture_v1 as wav_source

    require(np.__version__ == "2.3.5" and torch.__version__ == "2.15.0"
            and transformers.__version__ == "5.20.2", "pinned library version differs")
    require(source.sha_file(Path(modeling_wav2vec2.__file__)) == MODELING_SHA,
            "pinned modeling source differs")
    require(SNAPSHOT.is_dir() and source.sha_file(SNAPSHOT / "model.safetensors") == WEIGHTS_SHA,
            "pinned checkpoint bytes differ")
    config = json.loads((SNAPSHOT / "config.json").read_text())
    require(config["feat_extract_norm"] == "group" and config["conv_kernel"][0] == 10
            and config["conv_stride"][0] == 5 and config["vocab_size"] == 32,
            "pinned architecture differs")
    plan = execution_plan()
    require(plan["total_forward_ceiling"] == MAX_FORWARDS, "forward ceiling differs")
    baseline = wav_source.preflight()  # pins processor/tokenizer assets; no model load
    require(baseline["checkpoint_sha256"] == WEIGHTS_SHA
            and baseline["model_revision"] == SNAPSHOT.name
            and baseline["source_hashes"][modeling_wav2vec2.__name__] == MODELING_SHA,
            "baseline no-weight architecture/asset receipt differs")
    return {"schema": "fit_norm_causal_admission_v1", "status": "no_weight_ready",
            "source_binding": committed_sources(), "checkpoint_sha256": WEIGHTS_SHA,
            "modeling_sha256": MODELING_SHA,
            "snapshot_files_sha256": baseline["snapshot_files_sha256"],
            "processor_source_hashes": baseline["source_hashes"],
            "plan_sha256": bytes_sha(canonical(plan).encode()),
            "max_forwards": MAX_FORWARDS, "max_seconds": MAX_SECONDS,
            "max_rss_bytes": MAX_RSS, "max_output_bytes": MAX_OUTPUT}


def execution_plan() -> dict:
    return plan88.execution_plan()


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_go(admission_path: Path, go_path: Path) -> dict:
    require(admission_path.is_file() and not admission_path.is_symlink()
            and go_path.is_file() and not go_path.is_symlink(), "missing frozen admission or GO")
    admission = json.loads(admission_path.read_text())
    require(admission == no_weight_admission(), "admission no longer binds current sources")
    go = json.loads(go_path.read_text())
    require(go.get("schema") == "fit_norm_causal_executable_go_v1"
            and go.get("status") == "GO"
            and go.get("admission_sha256") == source.sha_file(admission_path)
            and go.get("source_binding") == admission["source_binding"],
            "independent executable GO does not bind admission/source")
    relative = str(go_path.resolve().relative_to(ROOT))
    require(subprocess.run(("git", "diff", "--quiet", "--", relative), cwd=ROOT).returncode == 0
            and subprocess.run(("git", "diff", "--cached", "--quiet", "--", relative), cwd=ROOT).returncode == 0,
            "GO receipt dirty")
    review_commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--", relative),
                                            cwd=ROOT, text=True).strip()
    require(len(review_commit) == 40, "GO receipt uncommitted")
    for name in SOURCE_FILES:
        relative_source = f"models/second_architecture/{name}"
        commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--", relative_source),
                                         cwd=ROOT, text=True).strip()
        require(commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, review_commit),
            cwd=ROOT, check=False).returncode == 0, "GO must postdate source freeze")
    return admission


def output_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class Budget:
    def __init__(self, output: Path, expected_names: list[str] | None = None,
                 progress_path: Path | None = None):
        self.output = output
        self.start = time.monotonic()
        self.calls = 0
        self.expected_names = expected_names
        self.attempted_names: list[str] = []
        self.completed_names: list[str] = []
        self.current_attempt: str | None = None
        self.progress_path = progress_path

    def progress(self, state: str, call_name: str) -> None:
        if self.progress_path is None:
            return
        require(not self.progress_path.is_symlink(), "call progress path is symlink")
        with self.progress_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical({"type": state, "call_index": self.calls,
                                    "call_name": call_name}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def check(self, *, before_forward: bool = False, call_name: str | None = None) -> None:
        require(time.monotonic() - self.start <= MAX_SECONDS, "worker time budget exhausted")
        rss, disk = rss_bytes(), output_bytes(self.output)
        require(rss <= MAX_RSS and disk <= MAX_OUTPUT, "worker RSS/disk cap exceeded")
        if before_forward:
            require(self.calls < MAX_FORWARDS, "forward ceiling exhausted")
            require(rss + RSS_RESERVE <= MAX_RSS
                    and disk + OUTPUT_RESERVE <= MAX_OUTPUT,
                    "insufficient forward RSS/disk reserve")
            require(isinstance(call_name, str) and call_name
                    and self.current_attempt is None, "forward call name/state invalid")
            if self.expected_names is not None:
                require(self.calls < len(self.expected_names)
                        and call_name == self.expected_names[self.calls],
                        "forward name/order differs from frozen 88-call plan")
            self.current_attempt = call_name
            self.attempted_names.append(call_name)
            self.calls += 1  # count attempted forwards, including failures
            self.progress("attempt", call_name)

    def finish(self, call_name: str) -> None:
        require(self.current_attempt == call_name, "forward completion order invalid")
        self.progress("complete", call_name)
        self.completed_names.append(call_name)
        self.current_attempt = None


def ordered_forward_names(plan: dict) -> list[str]:
    names: list[str] = []
    seen: set[tuple[str, str]] = set()
    for pair in plan["pairs"]:
        case, noise_arm = pair["case"], pair["noise_arm"]
        for variant in ("clean", noise_arm):
            if (case, variant) not in seen:
                names += [f"I_{case}_{variant}_stock", f"I_{case}_{variant}_readonly"]
                seen.add((case, variant))
        names += [f"I_{case}_{noise_arm}_{waveform}_{stat_source}"
                  for waveform, stat_source in source.INPUT_CELLS]
    for case in plan["stage_ii_cases"]:
        for arm in source.SOURCE_ARMS:
            names += [f"II_{case}_{arm}_stock", f"II_{case}_{arm}_read_only"]
        names += [f"II_{case}_{waveform}_{stat_source}"
                  for waveform, stat_source in source.GROUPNORM_CELLS]
    require(len(names) == MAX_FORWARDS and len(set(names)) == MAX_FORWARDS,
            "frozen forward name inventory differs")
    return names


def failure_summary(budget: Budget, recorded_forward_names: list[str],
                    error: BaseException) -> dict:
    require(budget.expected_names is not None, "failure census requires frozen plan")
    expected = budget.expected_names
    require(budget.attempted_names == expected[:budget.calls]
            and budget.completed_names == expected[:len(budget.completed_names)]
            and (budget.current_attempt is None
                 or budget.current_attempt == expected[len(budget.completed_names)]),
            "failure call ledger inconsistent")
    return {"type": "failure", "error_type": type(error).__name__,
            "error": str(error), "attempted_forwards": budget.calls,
            "completed_forward_names": budget.completed_names,
            "recorded_forward_names": recorded_forward_names,
            "completed_without_row_names": [name for name in budget.completed_names
                                            if name not in set(recorded_forward_names)],
            "failed_attempted_call_name": budget.current_attempt,
            "censored_call_names": expected[budget.calls:],
            "remaining_censored": len(expected)-budget.calls,
            "elapsed_seconds": time.monotonic()-budget.start}


def save_array(output: Path, relative: str, value, budget: Budget) -> dict:
    import numpy as np
    array = np.asarray(value)
    require(array.dtype in (np.float32, np.int64) and np.isfinite(array).all(),
            "invalid saved array dtype/finite")
    path = output / relative
    require(not path.exists() and not path.is_symlink(), "capture path already exists")
    require(output_bytes(output) + array.nbytes + (1 << 20) <= MAX_OUTPUT,
            "capture would exceed evidence cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    budget.check()
    return {"path": relative, "file_sha256": source.sha_file(path),
            "tensor_sha256": array_sha(array), "shape": list(array.shape),
            "dtype": str(array.dtype), "bytes": path.stat().st_size}


def decode(logits, processor) -> dict:
    import numpy as np
    a = logits.detach().cpu().float().numpy()
    require(a.ndim == 3 and a.shape[0] == 1 and a.shape[2] == 32
            and np.isfinite(a).all(), "CTC logits invalid")
    ids = a.argmax(axis=-1)[0].astype(np.int64)
    transcript = processor.batch_decode(ids[None, :])[0]
    sorted_logits = np.sort(a[0], axis=-1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    return {"argmax_ids": ids.tolist(), "transcript": transcript,
            "min_top2_margin": float(margins.min()), "frame_count": len(ids)}


def forward(model, input_values, budget: Budget, *, capture: bool = False,
            readonly_input: bool = False, readonly_gn: bool = False,
            raw_capture: bool = False, gn_stats=None, call_name: str):
    import torch

    budget.check(before_forward=True, call_name=call_name)
    conv = model.wav2vec2.feature_extractor.conv_layers[0]
    captures, counts, handles = {}, {}, []

    def hook(stage, tensor):
        counts[stage] = counts.get(stage, 0) + 1
        require(counts[stage] == 1 and torch.is_tensor(tensor), f"{stage} hook invalid")
        captures[stage] = tensor.detach().clone()

    if readonly_input:
        handles.append(model.wav2vec2.feature_extractor.register_forward_pre_hook(
            lambda _m, args: hook("read_only_input", args[0])))
    if capture or raw_capture or gn_stats is not None:
        handles.append(conv.conv.register_forward_hook(
            lambda _m, _a, out: hook("first_conv_raw", out)))
    if capture:
        handles.append(conv.layer_norm.register_forward_hook(
            lambda _m, _a, out: hook("first_conv_norm", out)))
        handles.append(model.wav2vec2.feature_projection.register_forward_hook(
            lambda _m, _a, out: hook("feature_projection", out[0])))
        handles.append(model.wav2vec2.encoder.layers[0].register_forward_hook(
            lambda _m, _a, out: hook("post_block0", out)))
    elif readonly_gn:
        handles.append(conv.layer_norm.register_forward_hook(
            lambda _m, _a, out: hook("native_groupnorm", out)))
    elif gn_stats is not None:
        def replace(_m, _a, out):
            hook("native_groupnorm", out)
            raw = captures.get("first_conv_raw")
            require(raw is not None and raw.shape == out.shape, "raw/GN hook order invalid")
            changed = source.groupnorm_with_statistics(raw, gn_stats,
                conv.layer_norm.weight, conv.layer_norm.bias)
            hook("manual_groupnorm", changed)
            return changed
        handles.append(conv.layer_norm.register_forward_hook(replace))
    try:
        with torch.inference_mode():
            logits = model(input_values=input_values, attention_mask=None,
                           output_hidden_states=False, return_dict=True).logits.detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    budget.check()
    expected = set()
    if readonly_input:
        expected.add("read_only_input")
    if capture:
        expected.update(("first_conv_raw", "first_conv_norm", "feature_projection", "post_block0"))
    elif raw_capture or gn_stats is not None:
        expected.add("first_conv_raw")
    if readonly_gn or gn_stats is not None:
        expected.add("native_groupnorm")
    if gn_stats is not None:
        expected.add("manual_groupnorm")
    require(set(counts) == expected and all(n == 1 for n in counts.values()),
            "hook count/inventory differs")
    budget.finish(call_name)
    return logits, captures, counts


def row(output: Path, name: str, logits, processor, budget: Budget, **extra) -> dict:
    evidence = save_array(output, f"logits/{name}.npy",
                          logits.detach().cpu().numpy(), budget)
    return {"name": name, "logits": evidence, "ctc": decode(logits, processor),
            "rss_bytes": rss_bytes(), "elapsed_seconds": time.monotonic() - budget.start,
            **extra}


def time_trace(tensor, baseline) -> dict:
    import torch
    require(tensor.shape == baseline.shape, "stage comparison shape differs")
    d = (tensor - baseline).float()
    # The time dimension is last for first-conv tensors and middle otherwise.
    if d.ndim == 3 and d.shape[1] == 512:
        d = d.transpose(1, 2)
    d = d.reshape(d.shape[0], d.shape[1], -1)
    l2 = torch.linalg.vector_norm(d, dim=-1)[0].cpu().numpy()
    maxabs = d.abs().amax(dim=-1)[0].cpu().numpy()
    return {"shape": list(tensor.shape), "tensor_sha256": array_sha(tensor.cpu().numpy()),
            "l2_by_frame": l2.tolist(), "maxabs_by_frame": maxabs.tolist()}


def stage_i_pair(model, processor, clean_pcm, noisy_pcm, pair: dict,
                 output: Path, budget: Budget, stock_cache: dict,
                 emit) -> None:
    import numpy as np
    import torch
    name = f"{pair['case']}_{pair['noise_arm']}"
    pcm = {"clean": clean_pcm, "noisy": noisy_pcm}
    stats = {arm: source.input_statistics(value) for arm, value in pcm.items()}
    stock, nohook = {}, {}
    for arm in source.SOURCE_ARMS:
        sample = pcm[arm].astype(np.float32) / np.float32(32768)
        batch = processor(sample, sampling_rate=16000, padding=False,
                          return_attention_mask=False, return_tensors="np")
        require(set(batch) == {"input_values"}, "stock processor inventory differs")
        stock[arm] = np.asarray(batch["input_values"])[0]
        own = source.normalize_with_statistics(pcm[arm], *stats[arm])
        require(np.array_equal(stock[arm], own), "stock/manual input diagonal differs")
        cache_key = (pair["case"], "clean" if arm == "clean" else pair["noise_arm"])
        if cache_key in stock_cache:
            nohook[arm] = stock_cache[cache_key]
            continue
        tensor = torch.from_numpy(stock[arm][None, :].copy())
        variant = "clean" if arm == "clean" else pair["noise_arm"]
        stock_name = f"I_{pair['case']}_{variant}_stock"
        readonly_name = f"I_{pair['case']}_{variant}_readonly"
        bare, _, _ = forward(model, tensor, budget, call_name=stock_name)
        viewed, _, count = forward(model, tensor, budget, readonly_input=True,
                                   call_name=readonly_name)
        require(torch.equal(bare, viewed), "input read-only hook changed logits")
        nohook[arm] = bare
        stock_row = row(output, stock_name, bare, processor, budget,
                        stage="I", pair=pair["case"], kind="stock", waveform=arm,
                        variant=variant)
        view_row = row(output, readonly_name, viewed, processor, budget,
                       stage="I", pair=pair["case"], kind="read_only", waveform=arm,
                       variant=variant,
                       hook_counts=count)
        require(stock_row["ctc"] == view_row["ctc"], "read-only CTC differs")
        emit(stock_row)
        emit(view_row)
        stock_cache[cache_key] = bare
    for old_item in pair["prior_stock"]:
        arm = "clean" if old_item["arm"] == "clean" else "noisy"
        output_ctc = decode(nohook[arm], processor)
        ids_sha = source.sha_bytes(json.dumps(output_ctc["argmax_ids"],
                                          separators=(",", ":")).encode())
        require(output_ctc["transcript"] == old_item["transcript"]
                and ids_sha == old_item["argmax_ids_sha256"],
                "stock complete CTC endpoint differs from audited pilot")
    clean_stages = None
    for waveform, stat_source in source.INPUT_CELLS:
        label = f"I_{name}_{waveform}_{stat_source}"
        normalized = source.normalize_with_statistics(pcm[waveform], *stats[stat_source])
        inp = torch.from_numpy(normalized[None, :].copy())
        logits, captures, counts = forward(model, inp, budget, capture=True,
                                           call_name=label)
        if waveform == stat_source:
            require(torch.equal(logits, nohook[waveform]), "Stage I diagonal logits differ")
        stage_tensors = {"input_values": inp, **captures}
        if clean_stages is None:
            clean_stages = {key: value.clone() for key, value in stage_tensors.items()}
        traces = {key: time_trace(stage_tensors[key], clean_stages[key]) for key in STAGES}
        emit(row(output, label, logits, processor, budget,
                        stage="I", pair=name, kind="cell", waveform=waveform,
                        stat_source=stat_source, pcm_sha256=source.sha_bytes(pcm[waveform].tobytes()),
                        input_sha256=array_sha(normalized),
                        statistic={"mean": float(stats[stat_source][0]),
                                   "variance": float(stats[stat_source][1]),
                                   "mean_bits": array_sha(np.asarray(stats[stat_source][0])),
                                   "variance_bits": array_sha(np.asarray(stats[stat_source][1]))},
                        hook_counts=counts, traces=traces))


def stage_ii_pair(model, processor, clean_pcm, noisy_pcm, pair: dict,
                  output: Path, budget: Budget, emit) -> None:
    import numpy as np
    import torch
    name = pair["case"]
    pcm = {"clean": clean_pcm, "noisy": noisy_pcm}
    shared = source.common_outside_statistics(clean_pcm, noisy_pcm, tuple(pair["slot"]))
    inputs = {arm: source.normalize_with_statistics(value, *shared)
              for arm, value in pcm.items()}
    start, end = pair["slot"]
    require(np.array_equal(inputs["clean"][:start], inputs["noisy"][:start])
            and np.array_equal(inputs["clean"][end:], inputs["noisy"][end:]),
            "shared input outside slot differs")
    raw, native, natural_logits, stats = {}, {}, {}, {}
    for arm in source.SOURCE_ARMS:
        inp = torch.from_numpy(inputs[arm][None, :].copy())
        bare, _, _ = forward(model, inp, budget,
                             call_name=f"II_{name}_{arm}_stock")
        viewed, captures, counts = forward(model, inp, budget,
                                           raw_capture=True, readonly_gn=True,
                                           call_name=f"II_{name}_{arm}_read_only")
        require(torch.equal(bare, viewed), "Stage II read-only hook changed logits")
        raw[arm], native[arm], natural_logits[arm] = (
            captures["first_conv_raw"], captures["native_groupnorm"], bare)
        stats[arm] = source.groupnorm_statistics(raw[arm])
        raw_file = save_array(output, f"stage_ii/{name}_{arm}_raw.npy",
                              raw[arm].cpu().numpy(), budget)
        native_file = save_array(output, f"stage_ii/{name}_{arm}_native_gn.npy",
                                 native[arm].cpu().numpy(), budget)
        for kind, logits in (("stock", bare), ("read_only", viewed)):
            emit(row(output, f"II_{name}_{arm}_{kind}", logits, processor, budget,
                            stage="II", pair=name, kind=kind, waveform=arm,
                            shared_input_sha256=array_sha(inputs[arm]),
                            shared_statistic={"mean": float(shared[0]),
                                              "variance": float(shared[1])},
                            raw=raw_file, native_gn=native_file,
                            hook_counts=counts if kind == "read_only" else {}))
    require(raw["clean"].shape == raw["noisy"].shape, "shared raw shapes differ")
    outside = source.first_conv_outside_mask(raw["clean"].shape[-1], tuple(pair["slot"]))
    require(torch.equal(raw["clean"][:, :, outside], raw["noisy"][:, :, outside]),
            "shared-input raw first conv outside differs")
    gamma = model.wav2vec2.feature_extractor.conv_layers[0].layer_norm.weight
    beta = model.wav2vec2.feature_extractor.conv_layers[0].layer_norm.bias
    affine = {"gamma_sha256": array_sha(gamma.detach().cpu().numpy()),
              "beta_sha256": array_sha(beta.detach().cpu().numpy())}
    clean_manual = None
    for waveform, stat_source in source.GROUPNORM_CELLS:
        cell_name = f"II_{name}_{waveform}_{stat_source}"
        inp = torch.from_numpy(inputs[waveform][None, :].copy())
        logits, captures, counts = forward(model, inp, budget,
                                           gn_stats=stats[stat_source], call_name=cell_name)
        require(torch.equal(captures["first_conv_raw"], raw[waveform]),
                "Stage II cell raw differs from frozen shared-path donor")
        manual = captures["manual_groupnorm"]
        if waveform == stat_source:
            gn_error = float((manual - native[waveform]).abs().max())
            logit_error = float((logits - natural_logits[waveform]).abs().max())
            require(gn_error <= 2e-5 and logit_error <= 1e-2,
                    "Stage II natural diagonal numerical error exceeded bound")
            require(decode(logits, processor)["argmax_ids"]
                    == decode(natural_logits[waveform], processor)["argmax_ids"],
                    "Stage II natural diagonal CTC path differs")
        if waveform == "clean" and stat_source == "clean":
            clean_manual = manual.clone()
        if waveform == "noisy" and stat_source == "clean":
            require(torch.equal(clean_manual[:, :, outside], manual[:, :, outside]),
                    "shared clean GN stats did not restore exact outside equality")
        stat_arrays = stats[stat_source]
        emit(row(output, cell_name, logits,
                        processor, budget, stage="II", pair=name, kind="cell",
                        waveform=waveform, stat_source=stat_source,
                        shared_input_sha256=array_sha(inputs[waveform]),
                        raw_source=waveform, raw_tensor_sha256=array_sha(raw[waveform].cpu().numpy()),
                        gn_stat_source=stat_source,
                        gn_mean_sha256=array_sha(stat_arrays[0].cpu().numpy()),
                        gn_variance_sha256=array_sha(stat_arrays[1].cpu().numpy()),
                        manual_gn_sha256=array_sha(manual.cpu().numpy()),
                        native_gn_sha256=array_sha(captures["native_groupnorm"].cpu().numpy()),
                        gn_edit_l2=float(torch.linalg.vector_norm(
                            manual - captures["native_groupnorm"])),
                        affine=affine, hook_counts=counts,
                        diagonal_gn_maxabs=gn_error if waveform == stat_source else None,
                        diagonal_logit_maxabs=logit_error if waveform == stat_source else None))


def run(output: Path, admission_path: Path, go_path: Path) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from models.second_architecture import acoustic_fit_noise_pilot_v2 as old

    validate_go(admission_path, go_path)
    require(not output.exists(), "one-shot output directory already exists")
    output.mkdir(parents=True)
    plan = execution_plan()
    budget = Budget(output, expected_names=ordered_forward_names(plan),
                    progress_path=output / "call_progress.jsonl")
    record_path = output / "records.jsonl"

    def record(value: dict) -> None:
        with record_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        budget.check()

    record({"type": "start", "admission_sha256": source.sha_file(admission_path),
            "go_sha256": source.sha_file(go_path), "plan_sha256": bytes_sha(canonical(plan).encode()),
            "planned_forwards": MAX_FORWARDS, "utc_seconds": time.time()})
    try:
        torch.set_num_threads(4)
        processor = Wav2Vec2Processor.from_pretrained(SNAPSHOT, local_files_only=True)
        model, loading = Wav2Vec2ForCTC.from_pretrained(
            SNAPSHOT, local_files_only=True, output_loading_info=True)
        require(set(loading["missing_keys"]) == {"wav2vec2.masked_spec_embed"}
                and not loading["unexpected_keys"] and not loading["mismatched_keys"]
                and not loading["error_msgs"], "pinned checkpoint load report differs")
        model = model.to("cpu", dtype=torch.float32).eval()
        require(all(p.device.type == "cpu" and p.dtype == torch.float32
                    for p in model.parameters()), "model dtype/device differs")
        clips = old.stimuli()
        stage_i_stock_cache = {}
        for pair in plan["pairs"]:
            name, noise_arm = pair["case"], pair["noise_arm"]
            clean_pcm, noisy_pcm = clips[name]["clean"], clips[name][noise_arm]
            record({"type": "pair_start", "stage": "I", "pair": name,
                    "noise_arm": noise_arm, "forward_count_before": budget.calls})
            stage_i_pair(model, processor, clean_pcm, noisy_pcm, pair,
                         output, budget, stage_i_stock_cache,
                         lambda result: record({"type": "forward", **result}))
            record({"type": "pair_complete", "stage": "I", "pair": name,
                    "noise_arm": noise_arm, "forward_count_after": budget.calls})
        for pair in plan["pairs"]:
            if pair["noise_arm"] != "word_noise":
                continue
            name = pair["case"]
            record({"type": "pair_start", "stage": "II", "pair": name,
                    "noise_arm": "word_noise", "forward_count_before": budget.calls})
            stage_ii_pair(model, processor, clips[name]["clean"],
                          clips[name]["word_noise"], pair, output, budget,
                          lambda result: record({"type": "forward", **result}))
            record({"type": "pair_complete", "stage": "II", "pair": name,
                    "noise_arm": "word_noise", "forward_count_after": budget.calls})
        require(budget.calls == MAX_FORWARDS, "complete run did not make 88 forwards")
        record({"type": "complete", "forward_count": budget.calls,
                "elapsed_seconds": time.monotonic() - budget.start,
                "rss_bytes": rss_bytes(), "output_bytes": output_bytes(output)})
    except BaseException as error:
        # Preserve the failure even when the budget check itself caused it.
        recorded_names = []
        if record_path.exists():
            for line in record_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    break
                if item.get("type") == "forward":
                    recorded_names.append(item["name"])
        failure = failure_summary(budget, recorded_names, error)
        with record_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical(failure) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("admit", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--go", type=Path)
    args = parser.parse_args()
    if args.mode == "admit":
        require(args.admission is None and args.go is None, "admit is no-weight only")
        write_once(args.output, no_weight_admission())
    else:
        require(args.admission is not None and args.go is not None, "run needs frozen receipts")
        run(args.output, args.admission, args.go)


if __name__ == "__main__":
    main()
