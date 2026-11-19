"""Read-only review reproduction: a forged completed parent result is accepted.

This file documents a source-only HOLD. It does not run a model. Once the
production audit rejects the forged result, this reproduction should fail and
be replaced by a positive regression test in the production fake suite.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from models.second_architecture import voxtral_track_b_v6_launch_freeze_contract_source_v1 as freeze
from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract


def reproduce() -> str:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        output = root / 'pilot'
        output.mkdir()
        measurement = output / 'pilot_measurement.json'
        measurement.write_text('{}')
        claim = root / 'campaign.json'
        claim.write_text('{}')
        watchdog = root / 'campaign.json.watchdog'
        watchdog.write_text(json.dumps({
            'schema': 'voxtral_track_b_v6_parent_deadline_claim_v1',
            'campaign_claim_path': str(claim),
            'output_dir': str(output.resolve()),
            'deadline_seconds': 240.0,
        }))
        # A completed result with a missed deadline and no cap/sample evidence.
        result = root / 'campaign.json.watchdog.result.json'
        result.write_text(json.dumps({
            'status': 'child_completed', 'child_exit_code': 0,
            'elapsed_seconds': 241.0,
        }))
        raw = root / 'pilot.post_host.raw'
        raw.write_bytes(b'fake raw post host log')
        pins = freeze.FutureLauncherPins(
            contract.ExternalPins('a' * 64, 'b' * 64, 'c' * 64), 'd' * 64)
        pre = datetime.now(timezone.utc) - timedelta(seconds=2)
        post = datetime.now(timezone.utc)
        sidecar = root / 'pilot.composition.json'
        sidecar.write_text(json.dumps({
            'schema': 'voxtral_track_b_v6_composition_source_only_v1',
            'status': 'worker_complete_post_host_checked_not_pilot_go',
            'pilot_measurement_sha256': worker.sha_file(measurement),
            'post_host_raw_sha256': worker.sha_file(raw),
            'post_host_raw_bytes': raw.stat().st_size,
            'worker_sha256': pins.external.worker_sha256,
            'admission_sha256': pins.external.admission_sha256,
            'review_sha256': pins.external.review_sha256,
            'campaign_attempt_id': 'fake_attempt',
            'pre_host_collected_utc': pre.isoformat(),
            'post_host_collected_utc': post.isoformat(),
        }))
        audited = freeze.audit_composition_sidecars(
            output_dir=output,
            bindings={'campaign_claim_path': str(claim),
                      'campaign_attempt_id': 'fake_attempt'},
            pins=pins,
            expected_measurement_sha256=worker.sha_file(measurement),
            free_disk_bytes=100 * 1024**3,
        )
        return audited['status']


if __name__ == '__main__':
    print(reproduce())
