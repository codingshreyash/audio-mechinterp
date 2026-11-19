"""Whisper Tiny time-translation worker; import loads no weights.

The inherited live-parent pipe prevents accidental direct runs under a trusted
local-owner assumption. It is not cryptographic parent authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from pathlib import Path
from typing import Any

from . import natural_tail_runner_v4 as v4
from . import whisper_tail_causal_worker_v3 as old_causal
from . import whisper_time_translation_plan_v1 as plan_source
from . import whisper_time_translation_source_v1 as source

HERE = Path(__file__).resolve().parent
ROOT = v4.ROOT
RUNS = HERE / "runs/whisper_time_translation_v1"
MODEL = "whisper_multilingual"
WALL_SECONDS = 1800
RSS_BYTES = 4 * 1024**3
EVIDENCE_BYTES = 1024**3
POST_RESERVE_SECONDS = 180
STAGES = ("original", "offset_8000", "offset_16000")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


def baselines_for(stage: str, frozen: dict) -> list[dict]:
    require(stage in STAGES, "unknown stage")
    if stage == "original":
        return [{"type": "baseline", "stage": stage, "layout": "original",
                 "variant": kind, "input": frozen["original"][kind]}
                for kind in ("zero", "noise")]
    offset = int(stage.removeprefix("offset_"))
    index = {(row["layout"], row["offset_samples"], row["kind"]): row
             for row in frozen["translated"]}
    return [{"type": "baseline", "stage": stage, "layout": layout,
             "variant": kind, "input": index[(layout, offset, kind)]}
            for layout in ("prepend", "append") for kind in ("zero", "noise")]


def _pcm(item: dict, *, original: bool) -> Any:
    import numpy as np
    name = item["wav"]
    require(type(name) is str and name == Path(name).name, "input name invalid")
    path = (v4.RENDER if original else source.OUTPUT) / name
    require(path.is_file() and not path.is_symlink()
            and sha(path) == item["wav_sha256"], "frozen input WAV changed")
    with wave.open(str(path), "rb") as audio:
        frames = item["frames"]
        require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                 audio.getnframes()) == (1, 2, 16_000, frames),
                "frozen input geometry changed")
        raw = audio.readframes(frames)
    require(len(raw) == frames * 2, "frozen PCM incomplete")
    return np.frombuffer(raw, dtype="<i2")


def execution_arm(arm: dict) -> dict:
    """Map the new exact cross-window relocation to V3's equal-norm hook."""
    if arm["kind"] != "relocated_correct_delta":
        return arm
    mapped = dict(arm)
    mapped["kind"] = "norm_matched_shifted_delta"
    mapped["target_rows"] = arm["delta_source_rows"]
    mapped["source_rows"] = arm["target_rows"]
    require(len(mapped["target_rows"]) == len(mapped["source_rows"])
            and not set(mapped["target_rows"]) & set(mapped["source_rows"]),
            "cross-window relocation geometry invalid")
    return mapped


def _complete(output: dict) -> bool:
    return (output.get("termination") == "eos"
            and output.get("eos_seen") is True
            and output.get("censored") is False
            and type(output.get("sequence_ids")) is list)


def pair_eligibility(zero: dict, noise: dict, sidecars: Path) -> dict:
    import numpy as np
    mask_zero = np.load(sidecars / zero["sidecars"]["mask"]["file"], allow_pickle=False)
    mask_noise = np.load(sidecars / noise["sidecars"]["mask"]["file"], allow_pickle=False)
    mask_equal = bool(np.array_equal(mask_zero, mask_noise))
    out_zero, out_noise = zero["output"], noise["output"]
    result = {"zero_complete": _complete(out_zero),
              "noise_complete": _complete(out_noise),
              "prefix_equal": out_zero.get("prefix_ids") == out_noise.get("prefix_ids"),
              "sequence_differs": out_zero.get("sequence_ids") != out_noise.get("sequence_ids"),
              "mask_equal": mask_equal}
    result["eligible"] = all(result.values())
    return result


def _original_parity(item: dict, outcome: dict, sidecars: Path) -> None:
    import numpy as np
    expected = item["input"]
    actual = outcome["output"]
    require(actual["sequence_ids"] == expected["sequence_ids"]
            and actual["prefix_ids"] == expected["prefix_ids"]
            and actual["termination"] == expected["termination"],
            "original V4 answer replay changed")
    for key in ("mel", "mask", "first_raw_logits", "first_processed_scores"):
        reference = v4.ATTEMPTS[MODEL] / "sidecars" / expected[key]["file"]
        current = sidecars / outcome["sidecars"][key]["file"]
        require(reference.is_file() and not reference.is_symlink()
                and sha(reference) == expected[key]["sha256"],
                "original V4 sidecar changed")
        require(np.array_equal(np.load(reference, allow_pickle=False),
                               np.load(current, allow_pickle=False)),
                "original V4 processed input or first-step vector changed")


def mel_shift_receipt(stage: str, baselines: dict, sidecars: Path,
                      frozen: dict) -> dict:
    """Measure exact interior feature transport; this never selects an arm."""
    import numpy as np
    if stage == "original":
        return {"status": "not_translated"}
    offset = int(stage.removeprefix("offset_"))
    mel_shift = offset // 160
    original_mels = {}
    for kind in ("zero", "noise"):
        ref = frozen["original"][kind]["mel"]
        path = v4.ATTEMPTS[MODEL] / "sidecars" / ref["file"]
        require(path.is_file() and not path.is_symlink()
                and sha(path) == ref["sha256"], "original mel reference changed")
        original_mels[kind] = np.load(path, allow_pickle=False)
    result = {"offset_mel_frames": mel_shift, "windows": [
        {"start": 50, "stop_exclusive": 750},
        {"start": 850, "stop_exclusive": 950}], "measurements": [],
        "layout_mask_equal": True, "active_mask_sum": 1000 + mel_shift}
    for kind in ("zero", "noise"):
        masks = [np.load(sidecars / baselines[(layout, kind)]["sidecars"]["mask"]["file"],
                         allow_pickle=False) for layout in ("prepend", "append")]
        result["layout_mask_equal"] &= bool(np.array_equal(*masks))
        require(all(mask.shape == (3000,) and int(mask.sum()) == 1000 + mel_shift
                    for mask in masks), "translated active mask geometry changed")
    for layout in ("prepend", "append"):
        for kind in ("zero", "noise"):
            outcome = baselines[(layout, kind)]
            path = sidecars / outcome["sidecars"]["mel"]["file"]
            translated = np.load(path, allow_pickle=False)
            original_mel = original_mels[kind]
            require(translated.shape == original_mel.shape == (80, 3000),
                    "Whisper mel geometry changed")
            for start, stop in ((50, 750), (850, 950)):
                moved = mel_shift if layout == "prepend" else 0
                difference = np.max(np.abs(
                    translated[:, start + moved:stop + moved]
                    - original_mel[:, start:stop]))
                result["measurements"].append({"layout": layout, "kind": kind,
                                               "start": start, "stop_exclusive": stop,
                                               "max_abs_difference": float(difference)})
    result["pass_le_1e_5"] = (result["layout_mask_equal"] and
                             all(row["max_abs_difference"] <= 1e-5
                                 for row in result["measurements"]))
    return result


def planned_calls(stage: str, frozen: dict, eligible: dict[str, bool]) -> tuple[list[dict], list[int]]:
    calls = baselines_for(stage, frozen)
    skipped = []
    for ordinal, arm in enumerate(frozen["stages"][stage]["arms"], 1):
        if eligible.get(arm["layout"], False):
            calls.append({"type": "arm", **arm})
        else:
            skipped.append(ordinal)
    require(len(calls) + len(skipped) == frozen["stages"][stage]["max_calls"],
            "conditional arm census changed")
    return calls, skipped


def run(stage: str, attempt: Path, admission: Path, go: Path, heartbeat_fd: int) -> None:
    from . import whisper_time_translation_gate_v1 as gate
    require(stage in STAGES and attempt == gate.ATTEMPTS[stage]
            and attempt.is_absolute() and attempt.is_dir() and not attempt.is_symlink(),
            "fixed expected translation attempt required")
    guard = v4.start_parent_guard(attempt, heartbeat_fd)
    guard.check()
    reviewed = gate.validate_go(stage, admission, go)
    guard.check()
    start = json.loads((attempt / "launch_start.json").read_text())
    require(start.get("parent_pid") == guard.parent_pid
            and start.get("deadline_monotonic") == guard.deadline
            and start.get("admission_sha256") == sha(admission)
            and start.get("go_sha256") == sha(go),
            "parent/admission/GO binding changed")
    import torch
    torch.set_num_threads(4)
    processor, model = v4.old._snapshot_load(MODEL, reviewed["v4_admission"]["assets"])
    guard.check()
    frozen = json.loads(plan_source.PLAN.read_text())
    require(frozen == plan_source.build(), "frozen translation plan changed")
    sidecars = attempt / "sidecars"
    sidecars.mkdir()
    progress, records = attempt / "progress.jsonl", attempt / "records.jsonl"
    baselines: dict[tuple[str, str], dict] = {}
    states: dict[str, dict[str, Any]] = {}
    calls_done = 0
    for item in baselines_for(stage, frozen):
        guard.check()
        index = calls_done + 1
        pcm = _pcm(item["input"], original=stage == "original")
        _write_jsonl(progress, {"type": "attempt", "index": index})
        outcome, captured, patch = old_causal.run_one(
            MODEL, processor, model, pcm, sidecars, f"{index:03d}",
            arm=None, states={})
        require(patch.get("hook_calls") == 0, "baseline unexpectedly patched")
        if stage == "original":
            _original_parity(item, outcome, sidecars)
        layout, kind = item["layout"], item["variant"]
        baselines[(layout, kind)] = outcome
        states.setdefault(layout, {})[kind] = captured
        state_refs = {site: old_causal._save_array(
            sidecars / f"{layout}_{kind}_{site}.npy", value)
            for site, value in captured.items()}
        _write_jsonl(records, {"type": "call", "index": index, "plan": item,
                               "output": outcome, "patch": patch,
                               "state_refs": state_refs, "execution_arm": None})
        _write_jsonl(progress, {"type": "complete", "index": index})
        calls_done = index
        guard.check()
    layouts = ("original",) if stage == "original" else ("prepend", "append")
    eligibility = {layout: pair_eligibility(
        baselines[(layout, "zero")], baselines[(layout, "noise")], sidecars)
        for layout in layouts}
    if stage == "original":
        require(eligibility["original"]["eligible"],
                "original V4 contrast absent; no arms may run")
    mel_shift = mel_shift_receipt(stage, baselines, sidecars, frozen)
    calls, skipped = planned_calls(
        stage, frozen, {layout: item["eligible"] for layout, item in eligibility.items()})
    baseline_count = len(baselines_for(stage, frozen))
    require(len(calls[:baseline_count]) == baseline_count,
            "baseline call order changed")
    for call in calls[baseline_count:]:
        guard.check()
        index = calls_done + 1
        layout = call["layout"]
        pcm_item = next(item["input"] for item in baselines_for(stage, frozen)
                        if item["layout"] == layout and item["variant"] == "noise")
        pcm = _pcm(pcm_item, original=stage == "original")
        mapped = execution_arm(call)
        _write_jsonl(progress, {"type": "attempt", "index": index})
        outcome, _captured, patch = old_causal.run_one(
            MODEL, processor, model, pcm, sidecars, f"{index:03d}",
            arm=mapped, states=states[layout])
        require(outcome["output"]["prefix_ids"]
                == baselines[(layout, "noise")]["output"]["prefix_ids"]
                and outcome["output"]["language_detection"]
                == baselines[(layout, "noise")]["output"]["language_detection"],
                "generation-only patch changed language detection")
        _write_jsonl(records, {"type": "call", "index": index, "plan": call,
                               "output": outcome, "patch": patch,
                               "state_refs": None, "execution_arm": mapped})
        _write_jsonl(progress, {"type": "complete", "index": index})
        calls_done = index
        guard.check()
    _write_jsonl(records, {"type": "complete", "stage": stage,
                           "executed_calls": calls_done,
                           "planned_max_calls": frozen["stages"][stage]["max_calls"],
                           "skipped_arms": skipped, "eligibility": eligibility,
                           "mel_shift": mel_shift})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--heartbeat-fd", type=int, required=True)
    args = parser.parse_args()
    run(args.stage, args.attempt, args.admission, args.go, args.heartbeat_fd)


if __name__ == "__main__":
    main()
