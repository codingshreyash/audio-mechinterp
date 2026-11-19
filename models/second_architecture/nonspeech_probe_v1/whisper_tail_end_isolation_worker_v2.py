"""Gated V2 isolated tail/end worker; import never loads model weights.

The inherited live-parent pipe is an accidental-run guard under a trusted
local-owner assumption, not cryptographic parent authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import natural_tail_runner_v4 as v4
from . import whisper_tail_causal_worker_v3 as causal
from . import whisper_time_translation_worker_v1 as prior
from . import whisper_time_translation_source_v1 as source
from . import whisper_tail_end_isolation_plan_v2 as plan_source

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs/whisper_tail_end_isolation_v2"
STAGES = plan_source.STAGES
MODEL = prior.MODEL
WALL_SECONDS = prior.WALL_SECONDS
RSS_BYTES = prior.RSS_BYTES
EVIDENCE_BYTES = prior.EVIDENCE_BYTES
POST_RESERVE_SECONDS = prior.POST_RESERVE_SECONDS

need = prior.require
sha = prior.sha
write_jsonl = prior._write_jsonl
baselines_for = prior.baselines_for
pair_eligibility = prior.pair_eligibility
mel_shift_receipt = prior.mel_shift_receipt
planned_calls = prior.planned_calls
execution_arm = prior.execution_arm
_complete = prior._complete
_original_parity = prior._original_parity


def source_pcm(item: dict, stage: str):
    return prior._pcm(item, original=stage == "original")


def run(stage: str, attempt: Path, admission: Path, go: Path,
        heartbeat_fd: int) -> None:
    from . import whisper_tail_end_isolation_gate_v2 as gate
    need(stage in STAGES and attempt == gate.ATTEMPTS[stage]
         and attempt.is_absolute() and attempt.is_dir() and not attempt.is_symlink(),
         "fixed expected V2 attempt required")
    guard = v4.start_parent_guard(attempt, heartbeat_fd)
    guard.check()
    reviewed = gate.validate_go(stage, admission, go)
    guard.check()
    start = json.loads((attempt / "launch_start.json").read_text())
    need(start.get("parent_pid") == guard.parent_pid
         and start.get("deadline_monotonic") == guard.deadline
         and start.get("admission_sha256") == sha(admission)
         and start.get("go_sha256") == sha(go),
         "V2 parent/admission/GO binding changed")
    import torch
    torch.set_num_threads(4)
    processor, model = v4.old._snapshot_load(MODEL, reviewed["v4_admission"]["assets"])
    guard.check()
    frozen = json.loads(plan_source.PLAN.read_text())
    need(frozen == plan_source.build(), "frozen V2 plan changed")
    sidecars = attempt / "sidecars"
    sidecars.mkdir()
    progress = attempt / "progress.jsonl"
    records = attempt / "records.jsonl"
    baselines: dict[tuple[str, str], dict] = {}
    states: dict[str, dict] = {}
    calls_done = 0
    for item in baselines_for(stage, frozen):
        guard.check()
        index = calls_done + 1
        pcm = source_pcm(item["input"], stage)
        write_jsonl(progress, {"type": "attempt", "index": index})
        outcome, captured, patch = causal.run_one(
            MODEL, processor, model, pcm, sidecars, f"{index:03d}",
            arm=None, states={})
        need(patch == {"hook_calls": 0}, "V2 baseline unexpectedly patched")
        if stage == "original":
            # V1's original worker already required byte-identical V4 replay.
            prior._original_parity(item, outcome, sidecars)
        layout, kind = item["layout"], item["variant"]
        baselines[(layout, kind)] = outcome
        states.setdefault(layout, {})[kind] = captured
        refs = {site: causal._save_array(
            sidecars / f"{layout}_{kind}_{site}.npy", value)
            for site, value in captured.items()}
        write_jsonl(records, {"type": "call", "index": index, "plan": item,
                              "output": outcome, "patch": patch,
                              "state_refs": refs, "execution_arm": None})
        write_jsonl(progress, {"type": "complete", "index": index})
        calls_done = index
        guard.check()
    layouts = ("original",) if stage == "original" else ("prepend", "append")
    eligibility = {layout: pair_eligibility(
        baselines[(layout, "zero")], baselines[(layout, "noise")], sidecars)
        for layout in layouts}
    if stage == "original":
        need(eligibility["original"]["eligible"],
             "original replay lost complete clean contrast")
    mel_shift = mel_shift_receipt(stage, baselines, sidecars, frozen)
    calls, skipped = planned_calls(stage, frozen,
        {layout: item["eligible"] for layout, item in eligibility.items()})
    need(calls[:len(baselines)] == baselines_for(stage, frozen),
         "V2 baseline schedule changed")
    for call in calls[len(baselines):]:
        guard.check()
        index = calls_done + 1
        layout = call["layout"]
        pcm_item = next(item["input"] for item in baselines_for(stage, frozen)
                        if item["layout"] == layout and item["variant"] == "noise")
        pcm = source_pcm(pcm_item, stage)
        mapped = execution_arm(call)
        write_jsonl(progress, {"type": "attempt", "index": index})
        outcome, _captured, patch = causal.run_one(
            MODEL, processor, model, pcm, sidecars, f"{index:03d}",
            arm=mapped, states=states[layout])
        need(outcome["output"]["prefix_ids"]
             == baselines[(layout, "noise")]["output"]["prefix_ids"]
             and outcome["output"]["language_detection"]
             == baselines[(layout, "noise")]["output"]["language_detection"],
             "V2 generation-only patch changed language detection")
        write_jsonl(records, {"type": "call", "index": index, "plan": call,
                              "output": outcome, "patch": patch,
                              "state_refs": None, "execution_arm": mapped})
        write_jsonl(progress, {"type": "complete", "index": index})
        calls_done = index
        guard.check()
    write_jsonl(records, {"type": "complete", "stage": stage,
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
