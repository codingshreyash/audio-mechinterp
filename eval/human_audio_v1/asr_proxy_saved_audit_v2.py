"""Require V2 executable GO and independently rebuild saved CTC evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import asr_proxy_gate_v2 as gate
from . import asr_proxy_saved_audit_v1 as old_audit
from . import asr_proxy_worker_v1 as worker

RECEIPT = worker.HERE / "runs/asr_proxy_v1/attempt_001_audit_v2.json"


def audit() -> dict:
    go = gate.validate_go()
    attempt = worker.ATTEMPT
    launch = json.loads((attempt / "launch.json").read_text())
    result = json.loads((attempt / "parent_result.json").read_text())
    for record, schema in ((launch, "human_asr_proxy_parent_v2"),
                           (result, "human_asr_proxy_parent_result_v2")):
        gate.need(record.get("schema_v2") == schema
                  and record.get("go_sha256") == go["go_sha256"]
                  and record.get("admission_v2_sha256") == go["admission_v2_sha256"]
                  and record.get("source_identity_v2") == go["source_identity"],
                  "V2 launch/result GO binding invalid")
    gate.need(launch.get("worker_module") == "eval.human_audio_v1.asr_proxy_worker_v2"
              and result.get("worker_module") == launch["worker_module"],
              "V2 worker command binding invalid")
    ready_path = attempt / "worker_ready_v2.json"
    ready = json.loads(ready_path.read_text())
    gate.need(result.get("worker_ready_sha256") == worker.sha(ready_path)
              and ready == {"schema": "human_asr_proxy_worker_ready_v2",
                            "worker_pid": ready.get("worker_pid"),
                            "parent_pid": launch["worker_parent_pid"],
                            "go_sha256": go["go_sha256"],
                            "admission_v2_sha256": go["admission_v2_sha256"],
                            "parent_nonce_sha256": launch["parent_nonce_sha256"]}
              and isinstance(ready["worker_pid"], int),
              "V2 live-channel attestation invalid")
    v1 = old_audit.audit()  # reconstruct complete saved evidence, no model weights
    gate.need(v1["status"] == "PASS" and v1["calls"] == 16,
              "V1 saved audit failed")
    return {"schema": "human_asr_proxy_saved_audit_v2", "status": "PASS",
            "go_sha256": go["go_sha256"],
            "admission_v2_sha256": go["admission_v2_sha256"],
            "source_identity_v2": go["source_identity"],
            "saved_audit_v1": v1,
            "target_model_inference": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RECEIPT)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x") as out:
        out.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": result["status"],
                      "calls": result["saved_audit_v1"]["calls"]}))


if __name__ == "__main__":
    main()
