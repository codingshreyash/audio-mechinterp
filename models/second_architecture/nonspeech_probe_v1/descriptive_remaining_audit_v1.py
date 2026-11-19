"""Saved one-shot audit for a reviewed English Whisper or Wav2Vec2 attempt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import descriptive_audit_v1 as old_audit
from . import descriptive_audit_v2 as timing_audit
from . import descriptive_parent_v2 as timing
from . import descriptive_remaining_parent_v1 as parent
from . import descriptive_worker_v1 as worker


def audit_attempt(model: str) -> dict:
    model_paths = parent.paths(model)
    parent.require_model_go(model)
    attempt = model_paths["attempt"]
    worker.require(attempt.is_dir() and not attempt.is_symlink(),
                   "reserved model-specific attempt missing")
    start_row = json.loads((attempt / "parent_start.json").read_text())
    result = json.loads((attempt / "parent_result.json").read_text())
    worker.require(start_row.get("model") == model and result.get("model") == model
                   and start_row.get("remaining_preflight_sha256") == worker.sha_file(
                       model_paths["receipt"])
                   and start_row.get("remaining_review_sha256") == worker.sha_file(
                       model_paths["review"])
                   and result.get("remaining_parent_source_sha256") == worker.sha_file(
                       Path(parent.__file__))
                   and result.get("remaining_preflight_sha256") == worker.sha_file(
                       model_paths["receipt"])
                   and result.get("remaining_review_sha256") == worker.sha_file(
                       model_paths["review"]),
                   "remaining-arm parent/review binding invalid")
    before, _ = timing_audit.saved_host(attempt, "before")
    after, log = timing_audit.saved_host(attempt, "after")
    started = timing.utc(start_row["utc"])
    exited = timing.utc(result["worker_exit_observed_utc"])
    collected = timing.utc(after["collected_utc"])
    worker.require(timing.utc(before["collected_utc"]) <= started,
                   "before-host probe was collected after launch")
    expected = timing.validate_post_host(log, started, exited, collected)
    worker.require(all(result.get(key) == value for key, value in expected.items())
                   and result.get("host_after_ok") is True,
                   "saved remaining-arm host timing differs")
    model_audit = old_audit.audit_attempt(attempt, parent.V1_RECEIPT, parent.V1_GO)
    worker.require(model_audit["model"] == model and model_audit["cases"] == 20
                   and model_audit["repeats"] == 2,
                   "remaining-arm saved model coverage differs")
    return {"schema": "nonspeech_descriptive_remaining_saved_audit_v1",
            "status": "PASS_STRUCTURAL", "model": model,
            "model_audit": model_audit,
            "host_fence_end_utc": exited.isoformat(),
            "host_probe_collected_utc": collected.isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(parent.MODEL_PATHS), required=True)
    args = parser.parse_args()
    result = audit_attempt(args.model)
    path = parent.paths(args.model)["attempt"] / "saved_audit_remaining_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(worker.canonical(result) + "\n")
    print(worker.canonical(result))


if __name__ == "__main__":
    main()
