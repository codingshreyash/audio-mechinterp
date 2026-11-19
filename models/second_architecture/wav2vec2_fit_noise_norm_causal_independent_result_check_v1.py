"""Independent saved-byte check of the FIT normalization result; no model forward.

The implementation intentionally does not call the study worker or its auditor.
It reconstructs inputs and the first convolution/GroupNorm from saved PCM and
checkpoint tensors, then decodes all 88 saved full-logit paths.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import plistlib
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Wav2Vec2Processor

from models.second_architecture import acoustic_fit_noise_pilot_v2 as old

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "runs/wav2vec2_fit_noise_norm_causal_v1/attempt_001"
OUTPUT = ATTEMPT / "worker_output"
AUDIT = ATTEMPT.parent / "attempt_001_audit.json"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h"
            / "snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
CASE_ORDER = ("piper_silver", "piper_teal", "kokoro_silver", "kokoro_teal")
SLOTS = {"piper": (30527, 49847), "kokoro": (33920, 57120)}
REFERENCES: set[str] = set()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(pcm: np.ndarray, stats_pcm: np.ndarray | None = None) -> np.ndarray:
    x = pcm.astype(np.float32) / np.float32(32768)
    y = (pcm if stats_pcm is None else stats_pcm).astype(np.float32) / np.float32(32768)
    return np.asarray((x - y.mean(dtype=np.float32)) /
                      np.sqrt(y.var(dtype=np.float32) + np.float32(1e-7)),
                      dtype=np.float32)


def get(saved: dict, prefix: str, shape: tuple[int, ...]) -> np.ndarray:
    ref = saved[prefix]
    rel = Path(ref["path"])
    require(not rel.is_absolute() and ".." not in rel.parts, "unsafe sidecar path")
    path = OUTPUT / rel
    require(path.is_file() and not path.is_symlink(), "missing sidecar")
    require(file_sha(path) == ref["file_sha256"] and path.stat().st_size == ref["bytes"],
            "sidecar byte hash/size")
    a = np.load(path, allow_pickle=False)
    require(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all()
            and sha(np.ascontiguousarray(a).tobytes()) == ref["tensor_sha256"],
            "sidecar tensor hash/shape")
    REFERENCES.add(str(path.relative_to(ATTEMPT)))
    return a


def host_check(launch: dict) -> None:
    event_re = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+([^\t]+)\t",
                          re.MULTILINE)
    terminal_re = re.compile(
        r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+: Showing all currently held IOKit power assertions\s*$",
        re.MULTILINE)
    stamp = lambda m: datetime.strptime(" ".join(m.group(1, 2, 3)),
                                        "%Y-%m-%d %H:%M:%S %z")
    for phase in ("pre", "post"):
        path = ATTEMPT / f"{phase}_host.json"
        receipt = json.loads(path.read_text())
        require(file_sha(path) == launch[f"{phase}_host_sha256"], "host receipt hash")
        outputs = receipt["outputs"]
        require("Now drawing from 'AC Power'" in outputs["power"]
                and plistlib.loads(outputs["battery"].encode())[0].get("ExternalConnected") is True
                and plistlib.loads(outputs["clamshell"].encode())[0].get("AppleClamshellState") is False,
                "host AC/lid")
        archive = ATTEMPT / receipt["power_log_archive"]["file"]
        require(file_sha(archive) == receipt["power_log_archive"]["compressed_sha256"],
                "host compressed log hash")
        raw = gzip.decompress(archive.read_bytes())
        require(sha(raw) == receipt["power_log_archive"]["raw_sha256"], "host log raw hash")
        log = raw.decode()
        events = list(event_re.finditer(log))
        terminals = list(terminal_re.finditer(log))
        require(bool(events) and bool(terminals), "host log parse")
        collected = datetime.fromisoformat(receipt["collected_utc"])
        start = (collected-timedelta(minutes=10) if phase == "pre"
                 else datetime.fromisoformat(launch["start_utc"]))
        terminal = stamp(terminals[-1])
        require(min(stamp(m) for m in events) <= start
                and terminal >= collected-timedelta(seconds=5)
                and terminal >= collected-timedelta(seconds=1)
                and terminal <= collected+timedelta(seconds=1), "host log time coverage")
        require(not any(start <= stamp(m) <= collected and
                        (m.group(4).strip() in ("Sleep", "DarkWake", "Wake")
                         or "Thermal" in m.group(4)) for m in events),
                "host sleep/thermal event")


def main() -> None:
    torch.set_num_threads(4)
    prior = HERE / "runs/acoustic_fit_noise_parent_v2/wav2vec2_attempt_002/worker_output/records.jsonl"
    require(file_sha(prior) == "c1badaf305dfaebe9beb9567b73f4e4bfdfab84b45ca8194f4c13c943f60038c",
            "old stock result changed")
    require(file_sha(SNAPSHOT / "model.safetensors") ==
            "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a",
            "checkpoint changed")
    audit = json.loads(AUDIT.read_text())
    launch = json.loads((ATTEMPT / "launch_result.json").read_text())
    require(audit["status"] == "saved_result_audit_pass"
            and launch["status"] == "completed_pending_saved_audit"
            and audit["records_sha256"] == file_sha(OUTPUT / "records.jsonl"),
            "saved result/status binding")
    host_check(launch)
    with (OUTPUT / "records.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    forwards = [r for r in rows if r.get("type") == "forward"]
    names = [r["name"] for r in forwards]
    require(len(forwards) == len(set(names)) == 88 and len(rows) == 114
            and rows[0]["type"] == "start" and rows[-1]["type"] == "complete",
            "88-forward inventory")
    progress = [json.loads(line) for line in (OUTPUT / "call_progress.jsonl").read_text().splitlines()]
    require(len(progress) == 176, "durable progress count")
    for i, name in enumerate(names):
        require(progress[2*i] == {"type": "attempt", "call_index": i+1, "call_name": name}
                and progress[2*i+1] == {"type": "complete", "call_index": i+1,
                                        "call_name": name}, "durable progress/order")
    require(launch["worker_call_census"]["completed_call_names"] == names
            and launch["worker_call_census"]["failed_attempted_call_name"] is None
            and launch["worker_call_census"]["censored_call_names"] == [], "parent census")
    rss = [json.loads(x) for x in (ATTEMPT / "worker_rss_samples.jsonl").read_text().splitlines()]
    require(len(rss) == launch["worker_rss_samples"]
            and max(x["tree_rss_bytes"] for x in rss) == launch["worker_peak_tree_rss_bytes"]
            and all(0 < x["tree_rss_bytes"] < 4*1024**3 for x in rss), "RSS log/peak")

    processor = Wav2Vec2Processor.from_pretrained(SNAPSHOT, local_files_only=True)
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt", device="cpu") as weights:
        prefix = "wav2vec2.feature_extractor.conv_layers.0."
        kernel = weights.get_tensor(prefix + "conv.weight")
        scale = weights.get_tensor(prefix + "layer_norm.weight")
        bias = weights.get_tensor(prefix + "layer_norm.bias")
    clips = old.stimuli()
    by_name = dict(zip(names, forwards))
    prior_rows = [json.loads(line) for line in prior.read_text().splitlines()]
    prior_by_case_arm = {(r["case"], r["arm"]): r for r in prior_rows
                         if r.get("type") == "case" and r.get("model") == "wav2vec2"}
    logits: dict[str, np.ndarray] = {}
    for row in forwards:
        case = "_".join(row["pair"].split("_")[:2])
        length = len(clips[case]["clean"])
        frames = (length-400)//320+1
        a = get(row, "logits", (1, frames, 32))
        ids = a.argmax(-1)[0].tolist()
        text = processor.batch_decode(np.asarray([ids], dtype=np.int64))[0]
        require(row["ctc"]["argmax_ids"] == ids
                and row["ctc"]["transcript"] == text
                and row["ctc"]["frame_count"] == len(ids), "full CTC decode")
        logits[row["name"]] = a

    stage_i_paths_changed = stage_i_text_changed = 0
    stage_i_raw_hashes_checked = 0
    for case in CASE_ORDER:
        for noise_arm in ("word_noise", "outside_noise"):
            pcm = {"clean": clips[case]["clean"], "noisy": clips[case][noise_arm]}
            for arm, variant in (("clean", "clean"), ("noisy", noise_arm)):
                stock = f"I_{case}_{variant}_stock"
                readonly = f"I_{case}_{variant}_readonly"
                require(np.array_equal(logits[stock], logits[readonly]), "I read-only parity")
                old_ctc = prior_by_case_arm[(case, variant)]["output"]
                require(by_name[stock]["ctc"]["argmax_ids"] == old_ctc["argmax_ids"]
                        and by_name[stock]["ctc"]["transcript"] == old_ctc["transcript"],
                        "I stock matches prior complete CTC")
                raw = pcm[arm].astype(np.float32) / np.float32(32768)
                model_input = processor(raw, sampling_rate=16000, padding=False,
                                        return_attention_mask=False,
                                        return_tensors="np")["input_values"][0]
                require(np.array_equal(model_input, norm(pcm[arm])), "I stock input normalization")
            base = None
            for waveform in ("clean", "noisy"):
                for stats in ("clean", "noisy"):
                    name = f"I_{case}_{noise_arm}_{waveform}_{stats}"
                    row = by_name[name]
                    x = norm(pcm[waveform], pcm[stats])
                    require(sha(x.tobytes()) == row["input_sha256"]
                            and sha(pcm[waveform].tobytes()) == row["pcm_sha256"],
                            "I input/stats provenance")
                    tensor = torch.from_numpy(x[None, None, :].copy())
                    raw_conv = F.conv1d(tensor, kernel, stride=5)
                    gn = F.group_norm(raw_conv, 512, scale, bias, eps=1e-5)
                    if base is None:
                        base = {"input_values": x[None, :],
                                "first_conv_raw": raw_conv.numpy(),
                                "first_conv_norm": gn.numpy()}
                    for stage, value in (("input_values", x[None, :]),
                                         ("first_conv_raw", raw_conv.numpy()),
                                         ("first_conv_norm", gn.numpy())):
                        require(sha(np.ascontiguousarray(value).tobytes()) ==
                                row["traces"][stage]["tensor_sha256"],
                                f"I {stage} tensor reconstruction")
                        stage_i_raw_hashes_checked += 1
                    if waveform == stats:
                        stock = f"I_{case}_{'clean' if waveform == 'clean' else noise_arm}_stock"
                        require(np.array_equal(logits[name], logits[stock]), "I natural diagonal")
            for waveform in ("clean", "noisy"):
                l = f"I_{case}_{noise_arm}_{waveform}_clean"
                r = f"I_{case}_{noise_arm}_{waveform}_noisy"
                path_change = not np.array_equal(logits[l].argmax(-1), logits[r].argmax(-1))
                text_change = by_name[l]["ctc"]["transcript"] != by_name[r]["ctc"]["transcript"]
                stage_i_paths_changed += int(path_change)
                stage_i_text_changed += int(text_change)

    stage_ii_frame_changes = stage_ii_text_changes = 0
    stage_ii_inconclusive = 0
    stage_ii_raw_contrasts = []
    stage_ii_max_diag_error = 0.0
    outside_raw_exact = outside_gn_common_exact = 0
    native_gn_outside_maxabs = {}
    for case in CASE_ORDER:
        clean, noisy = clips[case]["clean"], clips[case]["word_noise"]
        start, end = SLOTS[case.split("_")[0]]
        require(np.array_equal(clean[:start], noisy[:start])
                and np.array_equal(clean[end:], noisy[end:]), "II original PCM outside slot")
        reference = np.concatenate((clean[:start], clean[end:]))
        inputs = {"clean": norm(clean, reference), "noisy": norm(noisy, reference)}
        raw, native, stats = {}, {}, {}
        for arm in ("clean", "noisy"):
            stock = f"II_{case}_{arm}_stock"
            ro = f"II_{case}_{arm}_read_only"
            require(np.array_equal(logits[stock], logits[ro]), "II read-only parity")
            count = (len(inputs[arm])-10)//5+1
            raw[arm] = get(by_name[stock], "raw", (1, 512, count))
            native[arm] = get(by_name[stock], "native_gn", (1, 512, count))
            require(np.array_equal(F.conv1d(torch.from_numpy(inputs[arm][None, None, :].copy()),
                                               kernel, stride=5).numpy(), raw[arm]),
                    "II raw conv reconstruction")
            stats[arm] = (raw[arm].mean(-1, keepdims=True),
                          raw[arm].var(-1, keepdims=True))
        index = np.arange(raw["clean"].shape[-1])
        outside = ~((5*index < end) & (5*index+9 >= start))
        require(np.array_equal(raw["clean"][:, :, outside], raw["noisy"][:, :, outside]),
                "II outside raw conv equality")
        outside_raw_exact += 1
        native_gn_outside_maxabs[case] = float(np.max(np.abs(
            native["clean"][:, :, outside]-native["noisy"][:, :, outside])))
        manual = {}
        diag_err = {}
        for waveform in ("clean", "noisy"):
            for donor in ("clean", "noisy"):
                name = f"II_{case}_{waveform}_{donor}"
                row = by_name[name]
                x = torch.from_numpy(raw[waveform])
                stat_raw = torch.from_numpy(raw[donor])
                mu = stat_raw.mean(-1, keepdim=True)
                var = stat_raw.var(-1, unbiased=False, keepdim=True)
                manual[(waveform, donor)] = ((x-mu)*torch.rsqrt(var+1e-5)
                                             *scale[None, :, None]+bias[None, :, None]).numpy()
                require(sha(manual[(waveform, donor)].tobytes()) == row["manual_gn_sha256"]
                        and sha(mu.numpy().tobytes()) == row["gn_mean_sha256"]
                        and sha(var.numpy().tobytes()) == row["gn_variance_sha256"],
                        "II GN donor reconstruction")
                if waveform == donor:
                    err = float(np.max(np.abs(manual[(waveform, donor)]-native[waveform])))
                    require(err <= 2e-5, "II diagonal GN error")
                    stock = f"II_{case}_{waveform}_stock"
                    logit_err = float(np.max(np.abs(logits[name]-logits[stock])))
                    require(logit_err <= 1e-2
                            and np.array_equal(logits[name].argmax(-1), logits[stock].argmax(-1)),
                            "II diagonal CTC path")
                    diag_err[waveform] = logit_err
                    stage_ii_max_diag_error = max(stage_ii_max_diag_error, logit_err)
        require(np.array_equal(manual[("clean", "clean")][:, :, outside],
                               manual[("noisy", "clean")][:, :, outside]),
                "II shared donor outside GN equality")
        outside_gn_common_exact += 1
        for waveform in ("clean", "noisy"):
            a = logits[f"II_{case}_{waveform}_clean"][0]
            b = logits[f"II_{case}_{waveform}_noisy"][0]
            changed = a.argmax(-1) != b.argmax(-1)
            stage_ii_frame_changes += int(changed.any())
            stage_ii_text_changes += int(by_name[f"II_{case}_{waveform}_clean"]["ctc"]["transcript"]
                                         != by_name[f"II_{case}_{waveform}_noisy"]["ctc"]["transcript"])
            if changed.any():
                sa = np.sort(a[changed], axis=-1)
                sb = np.sort(b[changed], axis=-1)
                margin = min(float(np.min(sa[:, -1]-sa[:, -2])),
                             float(np.min(sb[:, -1]-sb[:, -2])))
                stage_ii_inconclusive += int(margin <= 2*max(diag_err.values()))
        for donor in ("clean", "noisy"):
            left = f"II_{case}_clean_{donor}"
            right = f"II_{case}_noisy_{donor}"
            a = logits[left][0]
            b = logits[right][0]
            changed = a.argmax(-1) != b.argmax(-1)
            margins = None
            if changed.any():
                aa = np.sort(a[changed], axis=-1)
                bb = np.sort(b[changed], axis=-1)
                margins = min(float(np.min(aa[:, -1]-aa[:, -2])),
                              float(np.min(bb[:, -1]-bb[:, -2])))
            threshold = 2*max(diag_err.values())
            stage_ii_raw_contrasts.append({
                "case": case, "gn_stat_source": donor,
                "changed_ctc_frames": int(np.count_nonzero(changed)),
                "min_changed_frame_top2_margin": margins,
                "diagonal_error_screen_threshold": threshold,
                "numerically_inconclusive": margins is not None and margins <= threshold,
                "clean_transcript": by_name[left]["ctc"]["transcript"],
                "noisy_transcript": by_name[right]["ctc"]["transcript"],
                "transcript_changed": by_name[left]["ctc"]["transcript"] !=
                                      by_name[right]["ctc"]["transcript"],
            })
    standard = {"launch_start.json", "launch_result.json", "pre_host.json",
                "post_host.json", "pre_power_log.txt.gz", "post_power_log.txt.gz",
                "worker_rss_samples.jsonl", "worker_stdout.txt", "worker_stderr.txt",
                "worker_output/records.jsonl", "worker_output/call_progress.jsonl"}
    inventory = {str(p.relative_to(ATTEMPT)) for p in ATTEMPT.rglob("*") if p.is_file()}
    require(inventory == standard | REFERENCES and len(REFERENCES) == 104,
            "exact evidence inventory")
    summary = {"schema": "fit_norm_independent_saved_check_v1",
               "forwards": len(forwards), "rss_samples": len(rss),
               "sampled_peak_tree_rss_bytes": launch["worker_peak_tree_rss_bytes"],
               "stage_i_raw_input_gn_hashes_checked": stage_i_raw_hashes_checked,
               "stage_i_ctc_path_changes": stage_i_paths_changed,
               "stage_i_transcript_changes": stage_i_text_changed,
               "stage_ii_contrasts_with_frame_changes": stage_ii_frame_changes,
               "stage_ii_transcript_changes": stage_ii_text_changes,
               "stage_ii_raw_contrasts": stage_ii_raw_contrasts,
               "stage_ii_numerically_inconclusive_contrasts": stage_ii_inconclusive,
               "stage_ii_max_diagonal_logit_error": stage_ii_max_diag_error,
               "outside_raw_exact_pairs": outside_raw_exact,
               "native_gn_outside_maxabs": native_gn_outside_maxabs,
               "outside_clean_gn_exact_pairs": outside_gn_common_exact}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
