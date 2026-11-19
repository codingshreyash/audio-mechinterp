"""Saved V2 host-fence audit plus the unchanged V1 model-record audit."""
from __future__ import annotations

import argparse
import gzip
import json
import plistlib
from pathlib import Path

from . import descriptive_audit_v1 as old_audit
from . import descriptive_parent_v2 as parent
from . import descriptive_worker_v1 as worker


def saved_host(attempt: Path, label: str) -> tuple[dict, str]:
    evidence = json.loads((attempt / f"host_{label}.json").read_text())
    archive = evidence["power_log_archive"]
    path = attempt / archive["file"]
    worker.require(path.is_file() and not path.is_symlink()
                   and path.name == f"host_{label}_power_log.txt.gz"
                   and worker.sha_file(path) == archive["compressed_sha256"],
                   "V2 host archive bytes changed")
    log = gzip.decompress(path.read_bytes())
    worker.require(worker.sha_bytes(log) == archive["raw_sha256"],
                   "V2 host raw log hash changed")
    outputs = evidence["outputs"]
    worker.require("Now drawing from 'AC Power'" in outputs["power"]
                   and plistlib.loads(outputs["clamshell"].encode())[0].get(
                       "AppleClamshellState") is False
                   and plistlib.loads(outputs["battery"].encode())[0].get(
                       "ExternalConnected") is True,
                   "V2 saved host power/lid state invalid")
    return evidence, log.decode()


def audit_attempt(review_path: Path) -> dict:
    parent.require_v2_go(review_path)
    attempt = parent.V2_ATTEMPT
    worker.require(attempt.is_dir() and not attempt.is_symlink(),
                   "reserved V2 attempt missing")
    parent_start = json.loads((attempt / "parent_start.json").read_text())
    result = json.loads((attempt / "parent_result.json").read_text())
    worker.require(parent_start.get("v2_preflight_sha256") == worker.sha_file(parent.V2_RECEIPT)
                   and parent_start.get("v2_review_sha256") == worker.sha_file(review_path)
                   and result.get("v2_parent_source_sha256") == worker.sha_file(
                       Path(parent.__file__))
                   and result.get("v2_preflight_sha256") == worker.sha_file(parent.V2_RECEIPT)
                   and result.get("v2_review_sha256") == worker.sha_file(review_path)
                   and result.get("prior_failed_parent_sha256") == parent.PRIOR_SHA256[
                       "parent_result.json"],
                   "V2 parent/review/prior binding invalid")
    before, _ = saved_host(attempt, "before")
    after, log = saved_host(attempt, "after")
    start = parent.utc(parent_start["utc"])
    exit_observed = parent.utc(result["worker_exit_observed_utc"])
    collected = parent.utc(after["collected_utc"])
    worker.require(parent.utc(before["collected_utc"]) <= start,
                   "before-host probe was collected after launch")
    expected = parent.validate_post_host(log, start, exit_observed, collected)
    worker.require(all(result.get(key) == value for key, value in expected.items())
                   and result.get("host_after_ok") is True,
                   "saved V2 host timing receipt differs")
    # V1 checks all 20 case rows, two silence repeats, 18 differences, and all
    # output/capture geometry. This call is only for a future, new V2 attempt.
    model_audit = old_audit.audit_attempt(attempt, parent.V1_RECEIPT, parent.V1_GO)
    return {"schema": "nonspeech_descriptive_host_timing_saved_audit_v2",
            "status": "PASS_STRUCTURAL", "model_audit": model_audit,
            "host_fence_end_utc": exit_observed.isoformat(),
            "host_probe_collected_utc": collected.isoformat(),
            "prior_attempt_status": "invalid_or_partial_preserved"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    result = audit_attempt(args.review)
    path = parent.V2_ATTEMPT / "saved_audit_v2.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(worker.canonical(result) + "\n")
    print(worker.canonical(result))


if __name__ == "__main__":
    main()
