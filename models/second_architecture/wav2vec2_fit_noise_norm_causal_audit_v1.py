"""Read-only saved-result audit for the 88-forward FIT normalization study.

This module never runs the candidate model. It checks complete saved logits,
CTC paths, input-stat provenance, GroupNorm donor tensors and call order.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_source_v1 as source
from models.second_architecture import wav2vec2_fit_noise_norm_causal_worker_v1 as worker
from models.second_architecture import wav2vec2_fit_noise_norm_causal_launcher_v1 as launcher
from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host_gate


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_array(attempt: Path, reference: dict, expected_shape: tuple[int, ...]):
    import numpy as np
    relative = reference.get("path")
    require(isinstance(relative, str) and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "unsafe evidence path")
    path = attempt / "worker_output" / relative
    require(path.is_file() and not path.is_symlink()
            and source.sha_file(path) == reference.get("file_sha256")
            and path.stat().st_size == reference.get("bytes"),
            f"evidence file changed: {relative}")
    a = np.load(path, allow_pickle=False)
    require(a.dtype == np.float32 and a.shape == expected_shape
            and np.isfinite(a).all() and list(a.shape) == reference.get("shape")
            and reference.get("dtype") == "float32"
            and worker.array_sha(a) == reference.get("tensor_sha256"),
            f"evidence tensor invalid: {relative}")
    return a


def expected_forward_names(plan: dict) -> list[str]:
    names = []
    seen = set()
    for pair in plan["pairs"]:
        case, noise_arm = pair["case"], pair["noise_arm"]
        for variant in ("clean", noise_arm):
            if (case, variant) not in seen:
                names.extend(f"I_{case}_{variant}_{kind}"
                             for kind in ("stock", "readonly"))
                seen.add((case, variant))
        names.extend(f"I_{case}_{noise_arm}_{raw}_{stats}"
                     for raw, stats in source.INPUT_CELLS)
    for case in plan["stage_ii_cases"]:
        names.extend(f"II_{case}_{arm}_{kind}" for arm in source.SOURCE_ARMS
                     for kind in ("stock", "read_only"))
        names.extend(f"II_{case}_{raw}_{stats}" for raw, stats in source.GROUPNORM_CELLS)
    require(len(names) == 88 and len(set(names)) == 88, "frozen name inventory invalid")
    return names


def check_host(attempt: Path, launch: dict) -> None:
    for name in ("pre", "post"):
        path = attempt / f"{name}_host.json"
        require(path.is_file() and not path.is_symlink(), "host receipt absent")
        evidence = json.loads(path.read_text())
        require(source.sha_file(path) == launch.get(f"{name}_host_sha256"),
                "host receipt hash differs")
        output = evidence["outputs"]
        require("Now drawing from 'AC Power'" in output["power"], "not AC power")
        import plistlib
        require(plistlib.loads(output["clamshell"].encode())[0].get("AppleClamshellState") is False
                and plistlib.loads(output["battery"].encode())[0].get("ExternalConnected") is True,
                "host lid/power differs")
        host_gate.validate_power(evidence)
        log = host_gate.load_archive(attempt, evidence, name)
        collected = datetime.fromisoformat(evidence["collected_utc"])
        if name == "pre":
            host_gate.validate_log(log, collected-timedelta(minutes=10), collected, collected)
        else:
            start = datetime.fromisoformat(launch["start_utc"])
            host_gate.validate_log(log, start, collected, collected)
    require(launch["host_collected_utc"] == [
        json.loads((attempt / f"{name}_host.json").read_text())["collected_utc"]
        for name in ("pre", "post")], "host collection order differs")


def audit(attempt: Path, admission_path: Path, go_path: Path) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from transformers import Wav2Vec2Processor
    from models.second_architecture import acoustic_fit_noise_pilot_v2 as old

    admission = worker.validate_go(admission_path, go_path)  # no model weights loaded
    require(attempt.is_dir() and not attempt.is_symlink(), "attempt directory absent")
    launch = json.loads((attempt / "launch_result.json").read_text())
    started = json.loads((attempt / "launch_start.json").read_text())
    require(launch.get("status") == "completed_pending_saved_audit"
            and started.get("status") == "started"
            and launch.get("schema") == started.get("schema") == "fit_norm_causal_launch_v1"
            and launch.get("worker_returncode") == 0
            and launch.get("worker_timed_out") is False
            and 0 < launch.get("worker_elapsed_seconds", 0) <= worker.MAX_SECONDS
            and launch.get("worker_stop_reason") == "normal_exit"
            and launch.get("worker_rss_limit_hit") is False
            and launch.get("worker_rss_monitor_error") is None
            and launch.get("worker_rss_limit_bytes") == worker.MAX_RSS
            and type(launch.get("worker_peak_tree_rss_bytes")) is int
            and 0 < launch["worker_peak_tree_rss_bytes"] < worker.MAX_RSS
            and type(launch.get("worker_rss_samples")) is int
            and launch["worker_rss_samples"] > 0
            and launch.get("admission_sha256") == started.get("admission_sha256")
            == source.sha_file(admission_path)
            and launch.get("go_sha256") == started.get("go_sha256")
            == source.sha_file(go_path), "parent launch status/binding invalid")
    require(worker.output_bytes(attempt) <= worker.MAX_OUTPUT, "evidence cap exceeded")
    check_host(attempt, launch)
    rss_path = attempt / "worker_rss_samples.jsonl"
    require(rss_path.is_file() and not rss_path.is_symlink()
            and source.sha_file(rss_path) == launch.get("worker_rss_sample_log_sha256"),
            "parent RSS sample evidence absent or altered")
    samples = [json.loads(line) for line in rss_path.read_text().splitlines()]
    require(len(samples) == launch["worker_rss_samples"], "parent RSS sample count differs")
    previous_elapsed = -1.0
    for index, sample in enumerate(samples, start=1):
        elapsed = sample.get("elapsed_seconds")
        rss = sample.get("tree_rss_bytes")
        require(sample.get("sample_index") == index
                and type(sample.get("worker_pid")) is int and sample["worker_pid"] > 0
                and type(elapsed) in (int, float)
                and previous_elapsed <= elapsed <= launch["worker_elapsed_seconds"]
                and type(rss) is int and 0 < rss < worker.MAX_RSS,
                "parent RSS sample invalid or out of order")
        previous_elapsed = elapsed
    require(max(sample["tree_rss_bytes"] for sample in samples)
            == launch["worker_peak_tree_rss_bytes"],
            "parent peak RSS differs from sampled evidence")
    progress_path = attempt / "worker_output/call_progress.jsonl"
    require(progress_path.is_file() and not progress_path.is_symlink()
            and source.sha_file(progress_path) == launch.get("worker_call_progress_sha256"),
            "durable call progress absent or changed")
    plan = worker.execution_plan()
    census = launcher.progress_census(progress_path, expected_forward_names(plan))
    require(census == launch.get("worker_call_census")
            and len(census["completed_call_names"]) == 88
            and census["failed_attempted_call_name"] is None
            and not census["censored_call_names"],
            "parent call census differs from durable worker progress")
    records_path = attempt / "worker_output/records.jsonl"
    require(records_path.is_file() and not records_path.is_symlink(), "worker ledger absent")
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    require(records[0]["type"] == "start" and records[-1]["type"] == "complete"
            and records[0]["planned_forwards"] == records[-1]["forward_count"] == 88
            and records[0]["admission_sha256"] == source.sha_file(admission_path)
            and records[0]["go_sha256"] == source.sha_file(go_path),
            "worker ledger boundary invalid")
    require(records[0]["plan_sha256"] == worker.bytes_sha(worker.canonical(plan).encode()),
            "worker plan changed")
    forwards = [r for r in records if r["type"] == "forward"]
    require(len(forwards) == 88
            and [r["name"] for r in forwards] == expected_forward_names(plan)
            and len([r for r in records if r["type"] == "pair_start"]) == 12
            and len([r for r in records if r["type"] == "pair_complete"]) == 12,
            "forward/pair inventory or order differs")
    require(all(r["type"] in ("start", "pair_start", "forward", "pair_complete", "complete")
                for r in records), "unexpected ledger row")
    # Exact pair boundaries: four clean, four word-noise, four outside-noise controls.
    position, call_count, seen = 1, 0, set()
    for stage, pairs in (("I", plan["pairs"]),
                         ("II", [p for p in plan["pairs"] if p["noise_arm"] == "word_noise"])):
        for pair in pairs:
            case, noise_arm = pair["case"], pair["noise_arm"]
            start = records[position]
            require(start["type"] == "pair_start" and start["stage"] == stage
                    and start["pair"] == case and start["noise_arm"] == noise_arm
                    and start["forward_count_before"] == call_count,
                    "pair start/call count differs")
            position += 1
            n = (6 if stage == "I" and case in seen else 8)
            if stage == "I":
                seen.add(case)
            require(all(r["type"] == "forward" for r in records[position:position+n]),
                    "pair forward span incomplete")
            position += n
            call_count += n
            end = records[position]
            require(end["type"] == "pair_complete" and end["stage"] == stage
                    and end["pair"] == case and end["noise_arm"] == noise_arm
                    and end["forward_count_after"] == call_count,
                    "pair end/call count differs")
            position += 1
    require(position == len(records)-1 and call_count == 88, "ledger terminal position differs")

    processor = Wav2Vec2Processor.from_pretrained(worker.SNAPSHOT, local_files_only=True)
    clips = old.stimuli()
    by_name = {r["name"]: r for r in forwards}
    logits = {}
    referenced_files = set()
    for r in forwards:
        engine = r["pair"].split("_")[0]
        samples = old.EXPECTED[engine]["frames"]
        frames = (samples - 400) // 320 + 1
        a = load_array(attempt, r["logits"], (1, frames, 32))
        referenced_files.add("worker_output/" + r["logits"]["path"])
        decoded = worker.decode(torch.from_numpy(a), processor)
        require(decoded["argmax_ids"] == r["ctc"]["argmax_ids"]
                and decoded["transcript"] == r["ctc"]["transcript"]
                and decoded["frame_count"] == r["ctc"]["frame_count"]
                and abs(decoded["min_top2_margin"] - r["ctc"]["min_top2_margin"]) < 1e-6,
                f"complete CTC/logit evidence differs: {r['name']}")
        require(0 <= r["elapsed_seconds"] <= worker.MAX_SECONDS
                and 0 < r["rss_bytes"] <= worker.MAX_RSS, "worker resource row invalid")
        logits[r["name"]] = a

    with safe_open(worker.SNAPSHOT / "model.safetensors", framework="pt", device="cpu") as weights:
        prefix = "wav2vec2.feature_extractor.conv_layers.0."
        conv_weight = weights.get_tensor(prefix + "conv.weight")
        gamma = weights.get_tensor(prefix + "layer_norm.weight")
        beta = weights.get_tensor(prefix + "layer_norm.bias")
    require(conv_weight.shape == (512, 1, 10) and conv_weight.dtype == torch.float32
            and gamma.shape == beta.shape == (512,)
            and gamma.dtype == beta.dtype == torch.float32,
            "first conv/GN checkpoint tensor shape or dtype invalid")
    torch.set_num_threads(4)

    stage_i_results = []
    for pair in plan["pairs"]:
        case, noise_arm = pair["case"], pair["noise_arm"]
        pcms = {"clean": clips[case]["clean"], "noisy": clips[case][noise_arm]}
        stats = {x: source.input_statistics(p) for x, p in pcms.items()}
        frames = (len(pcms["clean"])-400)//320+1
        algebra_baseline = None
        for arm, variant in (("clean", "clean"), ("noisy", noise_arm)):
            stock = by_name[f"I_{case}_{variant}_stock"]
            viewed = by_name[f"I_{case}_{variant}_readonly"]
            require(np.array_equal(logits[stock["name"]], logits[viewed["name"]])
                    and viewed["hook_counts"] == {"read_only_input": 1}
                    and stock["waveform"] == viewed["waveform"] == arm
                    and stock["variant"] == viewed["variant"] == variant,
                    "Stage I stock/read-only parity failed")
            prior = pair["prior_stock"][0 if arm == "clean" else 1]
            ids = stock["ctc"]["argmax_ids"]
            require(stock["ctc"]["transcript"] == prior["transcript"]
                    and source.sha_bytes(json.dumps(ids, separators=(",", ":")).encode())
                    == prior["argmax_ids_sha256"],
                    "Stage I stock CTC differs from independently audited pilot")
            samples = pcms[arm].astype(np.float32) / np.float32(32768)
            stock_input = processor(samples, sampling_rate=16000, padding=False,
                                    return_attention_mask=False, return_tensors="np")
            require(set(stock_input) == {"input_values"}
                    and np.array_equal(stock_input["input_values"][0],
                                       source.normalize_with_statistics(pcms[arm], *stats[arm])),
                    "Stage I manual/stock processor input differs")
        for waveform, stat_source in source.INPUT_CELLS:
            name = f"I_{case}_{noise_arm}_{waveform}_{stat_source}"
            r = by_name[name]
            x = source.normalize_with_statistics(pcms[waveform], *stats[stat_source])
            inp = torch.from_numpy(x[None, :].copy())
            raw_algebra = functional.conv1d(inp[:, None, :], conv_weight, stride=5)
            gn_algebra = functional.group_norm(raw_algebra, 512, gamma, beta,
                                               eps=source.FIRST_GN_EPSILON)
            algebra = {"input_values": inp, "first_conv_raw": raw_algebra,
                       "first_conv_norm": gn_algebra}
            if algebra_baseline is None:
                algebra_baseline = {key: value.clone() for key, value in algebra.items()}
            require(r["pcm_sha256"] == source.sha_bytes(pcms[waveform].tobytes())
                    and r["input_sha256"] == worker.array_sha(x)
                    and r["statistic"]["mean"] == float(stats[stat_source][0])
                    and r["statistic"]["variance"] == float(stats[stat_source][1])
                    and r["statistic"]["mean_bits"] == worker.array_sha(np.asarray(stats[stat_source][0]))
                    and r["statistic"]["variance_bits"] == worker.array_sha(np.asarray(stats[stat_source][1]))
                    and set(r["hook_counts"]) == set(worker.STAGES) - {"input_values"}
                    and all(v == 1 for v in r["hook_counts"].values()),
                    "Stage I statistic/input/hook provenance invalid")
            if waveform == stat_source:
                variant = "clean" if waveform == "clean" else noise_arm
                require(np.array_equal(logits[name], logits[f"I_{case}_{variant}_stock"]),
                        "Stage I natural diagonal logits differ")
            traces = r["traces"]
            require(set(traces) == set(worker.STAGES), "Stage I trace inventory differs")
            for stage, trace in traces.items():
                if stage == "input_values":
                    shape = [1, len(x)]
                elif stage in ("first_conv_raw", "first_conv_norm"):
                    shape = [1, 512, (len(x)-10)//5+1]
                else:
                    shape = [1, frames, 768]
                require(trace["shape"] == shape and len(trace["tensor_sha256"]) == 64
                        and len(trace["l2_by_frame"]) == shape[-1 if stage.startswith("first_conv") else 1]
                        and len(trace["maxabs_by_frame"]) == len(trace["l2_by_frame"])
                        and all(np.isfinite(v) and v >= 0 for v in trace["l2_by_frame"])
                        and all(np.isfinite(v) and v >= 0 for v in trace["maxabs_by_frame"]),
                        "Stage I time trace malformed")
                if waveform == stat_source == "clean":
                    require(all(v == 0 for v in trace["l2_by_frame"])
                            and all(v == 0 for v in trace["maxabs_by_frame"]),
                            "Stage I baseline trace nonzero")
                if stage == "input_values":
                    require(trace["tensor_sha256"] == worker.array_sha(x[None, :]),
                            "Stage I input trace hash differs")
                if stage in algebra:
                    rebuilt = worker.time_trace(algebra[stage], algebra_baseline[stage])
                    require(trace["tensor_sha256"] == rebuilt["tensor_sha256"]
                            and np.allclose(trace["l2_by_frame"], rebuilt["l2_by_frame"],
                                            atol=1e-5, rtol=1e-6)
                            and np.allclose(trace["maxabs_by_frame"], rebuilt["maxabs_by_frame"],
                                            atol=1e-6, rtol=1e-6),
                            f"Stage I {stage} trace differs from pinned algebra")
        stage_i_results.append({"pair": case, "noise_arm": noise_arm,
             "clean_stat_changes_ctc": (by_name[f"I_{case}_{noise_arm}_clean_noisy"]["ctc"]["argmax_ids"]
                                        != by_name[f"I_{case}_{noise_arm}_clean_clean"]["ctc"]["argmax_ids"]),
             "noisy_stat_changes_ctc": (by_name[f"I_{case}_{noise_arm}_noisy_noisy"]["ctc"]["argmax_ids"]
                                        != by_name[f"I_{case}_{noise_arm}_noisy_clean"]["ctc"]["argmax_ids"])})

    stage_ii_results = []
    for pair in plan["pairs"]:
        if pair["noise_arm"] != "word_noise":
            continue
        case = pair["case"]
        clean_pcm, noisy_pcm = clips[case]["clean"], clips[case]["word_noise"]
        shared = source.common_outside_statistics(clean_pcm, noisy_pcm, tuple(pair["slot"]))
        inputs = {"clean": source.normalize_with_statistics(clean_pcm, *shared),
                  "noisy": source.normalize_with_statistics(noisy_pcm, *shared)}
        raw, native, stats = {}, {}, {}
        for arm in source.SOURCE_ARMS:
            stock = by_name[f"II_{case}_{arm}_stock"]
            viewed = by_name[f"II_{case}_{arm}_read_only"]
            require(np.array_equal(logits[stock["name"]], logits[viewed["name"]])
                    and viewed["hook_counts"] == {"first_conv_raw": 1, "native_groupnorm": 1},
                    "Stage II no-hook/read-only logits differ")
            length = (len(inputs[arm])-10)//5+1
            raw[arm] = torch.from_numpy(load_array(attempt, stock["raw"], (1, 512, length)))
            native[arm] = torch.from_numpy(load_array(attempt, stock["native_gn"], (1, 512, length)))
            referenced_files.add("worker_output/" + stock["raw"]["path"])
            referenced_files.add("worker_output/" + stock["native_gn"]["path"])
            require(stock["raw"] == viewed["raw"] and stock["native_gn"] == viewed["native_gn"]
                    and stock["shared_input_sha256"] == viewed["shared_input_sha256"]
                    == worker.array_sha(inputs[arm])
                    and stock["shared_statistic"] == viewed["shared_statistic"]
                    == {"mean": float(shared[0]), "variance": float(shared[1])},
                    "Stage II shared-input capture provenance differs")
            stats[arm] = source.groupnorm_statistics(raw[arm])
            reconstructed = functional.conv1d(
                torch.from_numpy(inputs[arm][None, None, :].copy()),
                conv_weight, stride=5)
            require(torch.equal(reconstructed, raw[arm]),
                    "Stage II raw donor is not first convolution of shared input")
        outside = source.first_conv_outside_mask(raw["clean"].shape[-1], tuple(pair["slot"]))
        require(torch.equal(raw["clean"][:, :, outside], raw["noisy"][:, :, outside]),
                "Stage II raw convolution outside differs")
        diag_errors = {}
        manuals = {}
        for waveform, stat_source in source.GROUPNORM_CELLS:
            name = f"II_{case}_{waveform}_{stat_source}"
            r = by_name[name]
            manually = source.groupnorm_with_statistics(raw[waveform], stats[stat_source],
                                                        gamma, beta)
            manuals[(waveform, stat_source)] = manually
            require(r["shared_input_sha256"] == worker.array_sha(inputs[waveform])
                    and r["raw_source"] == waveform
                    and r["raw_tensor_sha256"] == worker.array_sha(raw[waveform].numpy())
                    and r["gn_stat_source"] == stat_source
                    and r["gn_mean_sha256"] == worker.array_sha(stats[stat_source][0].numpy())
                    and r["gn_variance_sha256"] == worker.array_sha(stats[stat_source][1].numpy())
                    and r["manual_gn_sha256"] == worker.array_sha(manually.numpy())
                    and r["native_gn_sha256"] == worker.array_sha(native[waveform].numpy())
                    and r["affine"] == {"gamma_sha256": worker.array_sha(gamma.numpy()),
                                         "beta_sha256": worker.array_sha(beta.numpy())}
                    and r["hook_counts"] == {"first_conv_raw": 1,
                                              "native_groupnorm": 1,
                                              "manual_groupnorm": 1},
                    "Stage II donor/hook/affine provenance invalid")
            edit_l2 = float(torch.linalg.vector_norm(manually - native[waveform]))
            require(abs(edit_l2-r["gn_edit_l2"]) < 1e-4,
                    "Stage II edit norm differs")
            if waveform == stat_source:
                gn_err = float((manually-native[waveform]).abs().max())
                logit_err = float(np.max(np.abs(logits[name] - logits[f"II_{case}_{waveform}_stock"])))
                require(gn_err <= 2e-5 and logit_err <= 1e-2
                        and abs(gn_err-r["diagonal_gn_maxabs"]) < 1e-7
                        and abs(logit_err-r["diagonal_logit_maxabs"]) < 1e-7
                        and by_name[name]["ctc"]["argmax_ids"]
                        == by_name[f"II_{case}_{waveform}_stock"]["ctc"]["argmax_ids"],
                        "Stage II natural diagonal parity failed")
                diag_errors[waveform] = logit_err
            else:
                require(r["diagonal_gn_maxabs"] is None
                        and r["diagonal_logit_maxabs"] is None,
                        "crossed arm has diagonal error field")
        require(torch.equal(manuals[("clean", "clean")][:, :, outside],
                            manuals[("noisy", "clean")][:, :, outside]),
                "Stage II clean-stat outside GN equality failed")
        contrasts = []
        threshold = 2 * max(diag_errors.values())
        for left, right in (("clean_noisy", "clean_clean"),
                            ("noisy_noisy", "noisy_clean")):
            a = logits[f"II_{case}_{left}"][0]
            b = logits[f"II_{case}_{right}"][0]
            changed = a.argmax(-1) != b.argmax(-1)
            if changed.any():
                sa, sb = np.sort(a[changed], axis=-1), np.sort(b[changed], axis=-1)
                margin = float(min((sa[:, -1]-sa[:, -2]).min(),
                                   (sb[:, -1]-sb[:, -2]).min()))
            else:
                margin = None
            contrasts.append({"left": left, "right": right,
                              "changed_frames": int(changed.sum()),
                              "min_changed_frame_top2_margin": margin,
                              "diagonal_error_screen_threshold": threshold,
                              "numerically_inconclusive": margin is not None and margin <= threshold,
                              "transcript_changed": by_name[f"II_{case}_{left}"]["ctc"]["transcript"]
                              != by_name[f"II_{case}_{right}"]["ctc"]["transcript"]})
        stage_ii_results.append({"case": case, "contrasts": contrasts,
                                 "diagonal_logit_maxabs": diag_errors})

    standard = {"launch_start.json", "launch_result.json", "pre_host.json", "post_host.json",
                "pre_power_log.txt.gz", "post_power_log.txt.gz", "worker_rss_samples.jsonl",
                "worker_stdout.txt",
                "worker_stderr.txt", "worker_output/records.jsonl",
                "worker_output/call_progress.jsonl"}
    inventory = {str(p.relative_to(attempt)) for p in attempt.rglob("*") if p.is_file()}
    require(all(not p.is_symlink() for p in attempt.rglob("*")), "symlink in evidence tree")
    require(inventory == standard | referenced_files, "evidence inventory incomplete or extra")
    require(records[-1]["output_bytes"] <= worker.MAX_OUTPUT
            and records[-1]["rss_bytes"] <= worker.MAX_RSS
            and records[-1]["elapsed_seconds"] <= worker.MAX_SECONDS,
            "worker terminal resource ceiling differs")
    return {"schema": "fit_norm_causal_saved_audit_v1",
            "status": "saved_result_audit_pass",
            "scope": "Four synthetic FIT English sources; conditional normalization interventions, not security finding",
            "attempt": str(attempt), "records_sha256": source.sha_file(records_path),
            "parent_sampled_peak_tree_rss_bytes": launch["worker_peak_tree_rss_bytes"],
            "parent_rss_limit_bytes": launch["worker_rss_limit_bytes"],
            "parent_rss_samples": launch["worker_rss_samples"],
            "admission_sha256": source.sha_file(admission_path),
            "go_sha256": source.sha_file(go_path),
            "forwards": len(forwards), "stage_i_pairs": stage_i_results,
            "stage_ii_pairs": stage_ii_results,
            "limitations": ["Stage I feature-projection and post-block-0 traces are hashed time summaries, not full saved tensors; the auditor validates their shape, finiteness, and inventory but cannot independently recompute them from model activations.",
                            "The numerical screen uses diagonal reproduction error and does not bound crossed-arm error."]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.attempt, args.admission, args.go)
    worker.write_once(args.output, result)
    print(worker.canonical(result))


if __name__ == "__main__":
    main()
