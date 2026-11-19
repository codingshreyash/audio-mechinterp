"""Source-only V6 pilot composition shell; public execution is hard HOLD.

The acyclic artifact order is worker W, external admission A, independent
review R, then reviewed launcher L pinning W/A/R. This file is not L: no A/R
exists, and exact old-FIT/synthetic batch construction is not yet reviewed.
"""

from __future__ import annotations

import base64
import gzip
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as admission_launcher
from models.second_architecture import voxtral_track_b_v6_mps_runtime_adapter_source_v1 as runtime
from models.second_architecture import voxtral_track_b_v6_pinned_loader_source_v1 as pinned_loader


FROZEN_PINS: contract.ExternalPins | None = None
LIVE_BATCH_ADAPTER_REVIEWED = False


def _pinned_model_factory(files: worker.AdmissionFiles, review_path: Path,
                          pins: contract.ExternalPins) -> Any:
    """Future child-only worker callback; source-wired but public unreachable."""
    model, _processor, _evidence = pinned_loader._load_pinned_fp32_voxtral(
        files=files, review_path=review_path, pins=pins)
    return model


@dataclass(frozen=True)
class PilotOperation:
    """Top-level pickleable child entry; currently stops before model load."""

    files: worker.AdmissionFiles
    review_path: Path
    pins: contract.ExternalPins
    output_dir: Path

    def __call__(self) -> dict[str, Any]:
        import torch  # child-local; no torch.mps module is pickled by parent
        if not torch.backends.mps.is_available():
            raise RuntimeError('V6 pilot requires MPS')
        admission_launcher.verify_pre_pilot(files=self.files,
                                             review_path=self.review_path, pins=self.pins)
        if not LIVE_BATCH_ADAPTER_REVIEWED:
            raise ValueError('V6 pilot HOLD: bound old-FIT/synthetic batch adapter absent')
        raise ValueError('V6 pilot HOLD: live child composition has no reviewed entry')


def _post_host_evidence() -> dict[str, Any]:
    """Fresh post-run host probe using the no-weight preflight's host parser."""
    source_dir = str(Path(__file__).resolve().parent)
    import sys
    sys.path.insert(0, source_dir)
    try:
        from models.second_architecture import voxtral_fp32_track_b_preflight_v1 as preflight
        result = preflight.host_status()
    finally:
        sys.path.remove(source_dir)
    admission_launcher._verify_fresh_host(result)
    return result


def _execute_fake_testable_composition(*, files: worker.AdmissionFiles,
                                       review_path: Path, pins: contract.ExternalPins,
                                       output_dir: Path,
                                       batch_provider: Callable[[], tuple[Mapping, Mapping, int]],
                                       load_model: Callable[[], Any],
                                       host_probe: Callable[[bool], Mapping[str, Any]],
                                       post_host: Callable[[], dict[str, Any]],
                                       memory_probe: Callable[[], int],
                                       free_bytes: Callable[[Path], int],
                                       sync: Callable[[], None]) -> dict[str, Any]:
    """Fake-only wiring check. Public path never calls this helper."""
    proof = admission_launcher.verify_pre_pilot(files=files,
                                                 review_path=review_path, pins=pins)
    gate = contract.verify_contract(files, review_path, pins)
    bindings = gate['bindings']
    old_fit, synthetic, pad = batch_provider()
    if (not isinstance(pad, int) or pad < 0 or
            worker._batch_sha(old_fit) != bindings['old_fit_batch_sha256'] or
            worker._batch_sha(synthetic) != bindings['synthetic_batch_sha256']):
        raise ValueError('Bound pilot batches differ before model loader')
    result = worker._execute_after_admission(
        load_model=load_model, old_fit_batch=old_fit,
        synthetic_batch=synthetic, pad=pad, output_dir=output_dir,
        bindings=bindings, host_probe=host_probe, memory_probe=memory_probe,
        free_bytes=free_bytes, sync=sync)
    if result.get('status') != 'schedule_complete_resource_unverified':
        raise ValueError('Worker schedule did not finish as resource-unverified')
    after = post_host()
    admission_launcher._verify_fresh_host(after)
    if (datetime.fromisoformat(after['collected_utc']) <=
            datetime.fromisoformat(proof['fresh_host_collected_utc'])):
        raise ValueError('Post-run host probe must be strictly later than pre-run probe')
    raw = gzip.decompress(base64.b64decode(after['power_log_gzip_base64'], validate=True))
    sidecar = output_dir.with_name(output_dir.name + '.composition.json')
    raw_path = output_dir.with_name(output_dir.name + '.post_host.raw')
    if not raw or len(raw) > 128 * 1024**2:
        raise ValueError('Post-run raw host log absent or unbounded')
    with raw_path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    worker._exclusive_json(sidecar, {
        'schema': 'voxtral_track_b_v6_composition_source_only_v1',
        'status': 'worker_complete_post_host_checked_not_pilot_go',
        'pilot_measurement_sha256': result['pilot_measurement_sha256'],
        'post_host_raw_sha256': worker.sha_bytes(raw),
        'post_host_raw_bytes': len(raw),
        'pre_host_collected_utc': proof['fresh_host_collected_utc'],
        'post_host_collected_utc': after['collected_utc'],
        'admission_sha256': pins.admission_sha256,
        'review_sha256': pins.review_sha256,
        'worker_sha256': pins.worker_sha256,
        'campaign_attempt_id': bindings['campaign_attempt_id'],
    })
    claim = Path(bindings['campaign_claim_path'])
    attempt_bytes = sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file())
    attempt_bytes += claim.stat().st_size + sidecar.stat().st_size + raw_path.stat().st_size
    if attempt_bytes > worker.MAX_OUTPUT_BYTES or free_bytes(output_dir) < worker.MIN_FREE_BYTES:
        raise ValueError('Full composition attempt output/free-disk cap exceeded')
    return {'status': 'schedule_complete_resource_unverified',
            'composition_sha256': worker.sha_file(sidecar)}


def _guarded_fake_attempt(*, operation: Callable[[], dict[str, Any]],
                          campaign_claim_path: Path, output_dir: Path,
                          fake_mps: Any, deadline_seconds: float = runtime.DEADLINE_SECONDS) -> dict:
    """Private process wiring for top-level fake operations only."""
    return runtime._guarded_child_attempt(
        campaign_claim_path=campaign_claim_path, output_dir=output_dir,
        operation=operation, mps=fake_mps, deadline_seconds=deadline_seconds)


def launch_v6_pilot(*_args: Any, **_kwargs: Any) -> None:
    """No public load until reviewed exact pins, batch adapter and auditor exist."""
    raise ValueError('V6 composition HOLD: no reviewed live launcher or bound batch adapter')
