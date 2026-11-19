"""Read-only fake reproduction of remaining saved-audit gaps; no model load."""

import gzip
import json
import plistlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import acoustic_fit_noise_parent_launch_v1 as parent


def reproduce() -> str:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        attempt = root / "attempt"
        (attempt / "worker_output").mkdir(parents=True)
        (attempt / "worker_output/records.jsonl").write_text("{}\n")
        # Over the 16 MiB parent capture cap, but the saved audit never counts it.
        (attempt / "extra_parent_capture.bin").write_bytes(bytes(17 * 1024**2))
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({"cases": {}}))
        worker_review = root / "worker_go.json"
        worker_review.write_text("{}")
        launcher_review = root / "launcher_go.json"
        launcher_review.write_text("{}")
        binding = {"source": "fake"}
        start_utc = "2026-11-19T00:00:00+00:00"
        end_utc = "2026-11-19T00:00:01+00:00"
        parent.write_once(attempt / "launch_start.json", {
            "schema": "acoustic_fit_noise_launch_start_v1", "model": "whisper",
            "attempt_path": str(attempt.resolve()), "source_binding": binding,
            "receipt_sha256": parent.pilot.sha_file(receipt),
            "worker_review_sha256": parent.pilot.sha_file(worker_review),
            "launcher_review_sha256": parent.pilot.sha_file(launcher_review),
            "start_utc": start_utc, "start_monotonic": 10.0})
        outputs = {
            "power": "Now drawing from 'AC Power'",
            "clamshell": plistlib.dumps([{"AppleClamshellState": False}]).decode(),
            "battery": plistlib.dumps([{"ExternalConnected": True}]).decode(),
        }
        for label in ("pre", "post"):
            (attempt / f"{label}_host.json").write_text(json.dumps({
                # Both samples precede the recorded attempt by one hour.
                "collected_utc": "2026-11-18T23:00:00+00:00", "outputs": outputs}))
            (attempt / f"{label}_power_log.txt.gz").write_bytes(gzip.compress(b"fake"))
        result = {
            "schema": "acoustic_fit_noise_parent_result_v1", "model": "whisper",
            "attempt_path": str(attempt.resolve()), "source_binding": binding,
            "preflight_sha256": parent.pilot.sha_file(receipt),
            "worker_review_sha256": parent.pilot.sha_file(worker_review),
            "launcher_review_sha256": parent.pilot.sha_file(launcher_review),
            "status": "complete", "errors": [], "worker_started": False,
            "worker_returncode": 0, "worker_timed_out": False,
            "worker_elapsed_seconds": 1.0, "worker_complete": True,
            "start_utc": start_utc, "end_utc": end_utc,
            "start_monotonic": 10.0, "end_monotonic": 9.0,
            "wall_seconds": 1.0, "monotonic_seconds": -1.0,
            "capture_bytes_before_result": 0,
            "evidence_sha256": parent.evidence_inventory(attempt),
        }
        parent.write_once(attempt / "launch_result.json", result)
        with (mock.patch.object(parent, "source_binding", return_value=binding),
              mock.patch.object(parent.pilot, "require_review"),
              mock.patch.object(parent, "require_launcher_go"),
              mock.patch.object(parent, "host_sidecars_valid", return_value=True),
              mock.patch.object(parent, "valid_worker_rows", return_value=True),
              mock.patch.object(parent.host_gate, "_power_log_coverage"),
              mock.patch.object(parent.host_gate, "_power_events", return_value=[])):
            return parent.audit_saved_attempt(
                "whisper", receipt, worker_review, launcher_review, attempt)["status"]


if __name__ == "__main__":
    status = reproduce()
    print({"forged_saved_attempt_status": status})
    assert status == "complete_evidence_pass"
