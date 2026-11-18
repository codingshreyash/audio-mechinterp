"""Pinned child entry point for one externally reserved Stage-B model process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import voxtral_old_fit_handoff_stage_b_launch_gate_v1 as gate
import voxtral_old_fit_handoff_stage_b_runner_v1 as driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=gate.PHASES, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (gate.HERE / ".venv").resolve():
        raise ValueError("Worker requires pinned second_architecture environment")
    gate.verify_worker_start(args.phase, args.attempt_id, args.freeze_sha256)
    output = driver.RUNS / args.phase / args.attempt_id
    gate.exclusive_json(gate.RUNS / args.phase / args.attempt_id / "worker_claim.json", {
        "status": "worker_claimed_no_retry", "phase": args.phase,
        "attempt_id": args.attempt_id, "stage_b_freeze_sha256": args.freeze_sha256,
        "wall_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
    })
    result = driver.run_phase(args.phase, output, model_factory=None, device="mps")
    print(json.dumps({"status": result["status"], "phase": args.phase,
                      "result_sha256": gate.sha(output / "result.json")}, allow_nan=False))


if __name__ == "__main__":
    main()
