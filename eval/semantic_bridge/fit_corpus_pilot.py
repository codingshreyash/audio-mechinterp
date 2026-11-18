"""Sequential, resumable eight-prompt corpus-average Omni Jacobian pilot.

The selected inputs are all in the frozen fit split. Each prompt gets a full
terminal-position Jacobian; the final matrix is their arithmetic mean, matching
the reference J-lens corpus-averaging idea. This is exploratory until the
task-framing validity gate passes and a separate held-out evaluation succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELECTION = HERE / "corpus_pilot_selection.json"
FIT_SCRIPT = ROOT / "models" / "qwen_omni" / "fit_terminal_jacobian.py"
ARTIFACTS = HERE / "artifacts" / "corpus_pilot"
LOGS = HERE / "corpus_fit_logs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    render_path = (HERE / selection["render_manifest"]).resolve()
    if digest(render_path) != selection["render_manifest_sha256"]:
        raise ValueError("Rendered corpus manifest differs from fixed selection")
    render = json.loads(render_path.read_text())
    if render["selection"]["split"] != "fit":
        raise ValueError("Corpus pilot must use fit split only")
    records = {r["item_id"]: r for r in render["records"]}
    if len(records) != len(render["records"]):
        raise ValueError("Duplicate rendered item IDs")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    report_path = HERE / "corpus_pilot_fit_report.json"
    report = {
        "status": "running",
        "selection_sha256": digest(SELECTION),
        "render_manifest_sha256": digest(render_path),
        "model_id": "Qwen/Qwen2.5-Omni-3B",
        "model_revision": "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        "system_prompt": selection["system_prompt"],
        "source_layers": selection["source_layers"],
        "target_layer": selection["target_layer"],
        "items": [],
    }
    write_json(report_path, report)
    try:
        for index, chosen in enumerate(selection["items"]):
            record = records[chosen["id"]]
            if record["source_item"]["split"] != "fit":
                raise ValueError(f"Non-fit item: {chosen['id']}")
            engine = chosen["engine"]
            audio_info = record["outputs"][engine]["processed"]
            audio = (render_path.parent / audio_info["path"]).resolve()
            if digest(audio) != audio_info["file_sha256"]:
                raise ValueError(f"WAV hash mismatch: {audio}")
            stem = f"{index:02d}_{chosen['id'].replace(':', '_')}_{engine}"
            checkpoint = ARTIFACTS / f"{stem}.pt"
            per_prompt_report = HERE / f"corpus_fit_{stem}.json"
            log_path = LOGS / f"{stem}.log"
            command = [
                sys.executable, "-u", str(FIT_SCRIPT), str(audio),
                "--checkpoint", str(checkpoint),
                "--report", str(per_prompt_report),
                "--dim-batch", "8",
                "--max-driver-gib", "30",
                "--system-prompt", selection["system_prompt"],
            ]
            env = os.environ.copy()
            env["HF_HOME"] = str(ROOT / "models" / "qwen_omni" / ".hf_cache")
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            started = time.monotonic()
            with log_path.open("a") as log:
                log.write("\nCOMMAND " + json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            outcome = json.loads(per_prompt_report.read_text())
            if outcome["status"] != "complete" or outcome["completed_coordinates"] != 2048:
                raise ValueError(f"Incomplete fit: {stem}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not bool(state["completed"].all()):
                raise ValueError(f"Incomplete checkpoint: {stem}")
            if state["meta"]["audio_sha256"] != audio_info["file_sha256"]:
                raise ValueError(f"Fitted input differs from manifest: {stem}")
            if state["meta"]["system_prompt"] != selection["system_prompt"]:
                raise ValueError(f"Fitted task frame differs: {stem}")
            for layer in selection["source_layers"]:
                if not torch.isfinite(state["rows"][str(layer)]).all():
                    raise ValueError(f"Nonfinite map: {stem}, layer {layer}")
            item_report = {
                "id": chosen["id"], "engine": engine,
                "audio_sha256": audio_info["file_sha256"],
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": digest(checkpoint),
                "fit_report_path": str(per_prompt_report.relative_to(ROOT)),
                "elapsed_seconds": time.monotonic() - started,
                "backward_seconds": sum(x["backward_seconds"]
                                        for x in outcome["batch_records"]),
                "peak_driver_gib": max(x["memory_bytes"]["driver"]
                                       for x in outcome["batch_records"]) / 2**30,
            }
            report["items"].append(item_report)
            write_json(report_path, report)
            print(f"completed {index + 1}/{len(selection['items'])}: {chosen['id']} {engine}", flush=True)
        sums = {str(layer): None for layer in selection["source_layers"]}
        for item in report["items"]:
            checkpoint = ROOT / item["checkpoint_path"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for layer in selection["source_layers"]:
                key = str(layer)
                matrix = state["rows"][key]
                sums[key] = matrix.clone() if sums[key] is None else sums[key] + matrix
        means = {key: value / len(report["items"]) for key, value in sums.items()}
        mean_path = ARTIFACTS / "corpus_mean.pt"
        temp = mean_path.with_name(mean_path.name + f".tmp.{os.getpid()}")
        torch.save({"rows": means, "n_prompts": len(report["items"]),
                    "selection_sha256": report["selection_sha256"],
                    "model_revision": report["model_revision"],
                    "system_prompt": report["system_prompt"],
                    "source_layers": report["source_layers"],
                    "target_layer": report["target_layer"],
                    "source_checkpoint_sha256": [x["checkpoint_sha256"]
                                                 for x in report["items"]]}, temp)
        os.replace(temp, mean_path)
        report["mean_checkpoint_path"] = str(mean_path.relative_to(ROOT))
        report["mean_checkpoint_sha256"] = digest(mean_path)
        report["status"] = "complete"
        write_json(report_path, report)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
