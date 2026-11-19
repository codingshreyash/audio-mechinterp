"""Source-only V6 launcher freeze and sidecar audit; public launch is HOLD.

This binds the independently reviewed batch adapter and explicit Piper choice
to an external A/R before any eventual model load. It does not create A/R,
run the pilot, or authorize a model weight load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture import voxtral_track_b_v6_bound_batch_adapter_source_v1 as batches
from models.second_architecture import voxtral_track_b_v6_mps_runtime_adapter_source_v1 as runtime


HERE = Path(__file__).resolve().parent
PINNED_SOURCE_SHAS = {
    'pilot_worker_sha256': '261ff1dc61b96268fd1ba5594c1a95e31dedc7eba5d7daf39bc567c85d0ae30f',
    'batch_adapter_sha256': 'd6954701815285d9895b64e8d30683405b354d897d21b4cdf6a2f58384ef5253',
    'pinned_loader_sha256': 'acb6fe6db514bedd65ec0fb9389ba59c5cc52c29e482b84275666bd660e7d9d4',
    'mps_runtime_sha256': 'db2982f63f803ca26fbf561aec0a101a743146e2bb62cb6c4ac23169349fc635',
}
PINNED_SOURCE_PATHS = {
    'pilot_worker_sha256': HERE / 'voxtral_track_b_v6_pilot_worker_source_v1.py',
    'batch_adapter_sha256': HERE / 'voxtral_track_b_v6_bound_batch_adapter_source_v1.py',
    'pinned_loader_sha256': HERE / 'voxtral_track_b_v6_pinned_loader_source_v1.py',
    'mps_runtime_sha256': HERE / 'voxtral_track_b_v6_mps_runtime_adapter_source_v1.py',
}
PIPER_CHOICE = 'color_05/piper/fit_chosen/silver'


@dataclass(frozen=True)
class FutureLauncherPins:
    """A later reviewed launcher will freeze these values in its source."""

    external: contract.ExternalPins
    composition_sha256: str


def verify_launch_freeze(*, files: worker.AdmissionFiles, review_path: Path,
                         pins: FutureLauncherPins,
                         composition_source: Path) -> dict[str, Any]:
    """Require exact W/B/P/M/C sources and Piper choice in externally reviewed A."""
    gate = contract.verify_contract(files, review_path, pins.external)
    bindings = gate['bindings']
    if pins.external.worker_sha256 != PINNED_SOURCE_SHAS['pilot_worker_sha256']:
        raise ValueError('Worker source pin differs from reviewed freeze')
    for key, path in PINNED_SOURCE_PATHS.items():
        expected = PINNED_SOURCE_SHAS[key]
        if worker.sha_file(path) != expected or bindings.get(key) != expected:
            raise ValueError(f'Reviewed {key} source or admission binding differs')
    worker._sha256_field(pins.composition_sha256, 'composition_sha256')
    if (not composition_source.is_file() or composition_source.is_symlink() or
            composition_source.resolve() != (HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py').resolve() or
            worker.sha_file(composition_source) != pins.composition_sha256 or
            bindings.get('composition_sha256') != pins.composition_sha256):
        raise ValueError('Composition source differs from future launcher/admission pin')
    if (bindings.get('old_fit_source_choice') != PIPER_CHOICE or
            bindings.get('old_fit_wav_sha256') != batches.OLD_WAV_SHA256 or
            bindings.get('old_fit_raw_tensor_sha256') != batches.OLD_FIT_RAW_HASHES or
            bindings.get('old_fit_prompt') != batches.FIT_PROMPT or
            bindings.get('track_b_prompt') != batches.TRACK_PROMPT):
        raise ValueError('Explicit Piper old-FIT input or prompt binding differs')
    if worker.sha_file(batches._resolve_old_fit_silver()) != batches.OLD_WAV_SHA256:
        raise ValueError('Piper silver source WAV bytes changed')
    return {'status': 'source_only_launch_freeze_checked_not_pilot_go',
            'admission_sha256': pins.external.admission_sha256,
            'review_sha256': pins.external.review_sha256,
            'composition_sha256': pins.composition_sha256,
            'campaign_attempt_id': bindings['campaign_attempt_id']}


def checked_transfer(*, bound: batches.BoundPilotBatches,
                     bindings: Mapping[str, Any], device: str) -> tuple[dict, dict]:
    """Check dtype/shape-aware SHA before and after device copy; live uses MPS."""
    if device not in ('mps', 'cpu'):
        raise ValueError('Pilot transfer device invalid')
    originals = ((bound.old_fit, bound.old_fit_sha256, 'old_fit_batch_sha256'),
                 (bound.synthetic, bound.synthetic_sha256, 'synthetic_batch_sha256'))
    moved = []
    for batch, digest, key in originals:
        if worker._batch_sha(batch) != digest or digest != bindings.get(key):
            raise ValueError(f'Bound {key} changed before transfer')
        copy = {name: value.to(device) for name, value in batch.items()}
        if (any(value.device.type != device for value in copy.values()) or
                any(copy[name].shape != batch[name].shape or
                    copy[name].dtype != batch[name].dtype for name in batch) or
                worker._batch_sha(copy) != digest):
            raise ValueError(f'Bound {key} changed during device transfer')
        moved.append(copy)
    return moved[0], moved[1]


def prepare_bound_inputs_for_child(*, files: worker.AdmissionFiles,
                                   review_path: Path, pins: FutureLauncherPins,
                                   composition_source: Path,
                                   device: str = 'mps',
                                   processor_factory: Any | None = None) -> tuple[dict, dict, dict]:
    """Source-only B→device wiring; future child must call with MPS."""
    proof = verify_launch_freeze(files=files, review_path=review_path,
                                 pins=pins, composition_source=composition_source)
    gate = contract.verify_contract(files, review_path, pins.external)
    bound = batches.build_bound_pilot_batches(
        files=files, review_path=review_path, pins=pins.external,
        processor_factory=processor_factory)
    old, zero = checked_transfer(bound=bound, bindings=gate['bindings'], device=device)
    return old, zero, {'source_freeze': proof, 'old_fit_batch_sha256': bound.old_fit_sha256,
                       'synthetic_batch_sha256': bound.synthetic_sha256,
                       'transfer_device': device}


def audit_composition_sidecars(*, output_dir: Path, bindings: Mapping[str, Any],
                               pins: FutureLauncherPins,
                               expected_measurement_sha256: str,
                               free_disk_bytes: int) -> dict[str, Any]:
    """Read-only accounting of external sidecars; never yields a pilot GO."""
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError('Pilot output directory absent or symlinked')
    sidecar = output_dir.with_name(output_dir.name + '.composition.json')
    raw = output_dir.with_name(output_dir.name + '.post_host.raw')
    claim = Path(bindings['campaign_claim_path'])
    watchdog = runtime._claim_path(claim)
    watchdog_result = watchdog.with_name(watchdog.name + '.result.json')
    paths = [sidecar, raw, claim, watchdog, watchdog_result,
             output_dir / 'pilot_measurement.json']
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Required composition/claim evidence missing or symlinked')
    saved = json.loads(sidecar.read_text())
    if (saved.get('schema') != 'voxtral_track_b_v6_composition_source_only_v1' or
            saved.get('status') != 'worker_complete_post_host_checked_not_pilot_go' or
            saved.get('pilot_measurement_sha256') != expected_measurement_sha256 or
            saved.get('pilot_measurement_sha256') != worker.sha_file(paths[-1]) or
            saved.get('post_host_raw_sha256') != worker.sha_file(raw) or
            saved.get('post_host_raw_bytes') != raw.stat().st_size or
            saved.get('worker_sha256') != pins.external.worker_sha256 or
            saved.get('admission_sha256') != pins.external.admission_sha256 or
            saved.get('review_sha256') != pins.external.review_sha256 or
            saved.get('campaign_attempt_id') != bindings['campaign_attempt_id']):
        raise ValueError('Composition sidecar bytes or binding differ')
    pre = datetime.fromisoformat(saved['pre_host_collected_utc'])
    post = datetime.fromisoformat(saved['post_host_collected_utc'])
    if pre.tzinfo is None or post.tzinfo is None or post <= pre:
        raise ValueError('Post-host collection is not strictly later')
    parent_claim = json.loads(watchdog.read_text())
    parent_result = json.loads(watchdog_result.read_text())
    if (parent_claim.get('schema') != 'voxtral_track_b_v6_parent_deadline_claim_v1' or
            parent_claim.get('campaign_claim_path') != str(claim) or
            parent_claim.get('output_dir') != str(output_dir.resolve()) or
            parent_claim.get('deadline_seconds') != runtime.DEADLINE_SECONDS or
            parent_result.get('status') != 'child_completed' or
            parent_result.get('child_exit_code') != 0):
        raise ValueError('Parent watchdog claim/result incomplete or failed')
    worker_files = [p for p in output_dir.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in output_dir.rglob('*')):
        raise ValueError('Symlink in pilot output')
    total = sum(p.stat().st_size for p in worker_files + paths[:-1])
    if total > worker.MAX_OUTPUT_BYTES or free_disk_bytes < worker.MIN_FREE_BYTES:
        raise ValueError('Full pilot attempt output/free-disk cap exceeded')
    return {'status': 'source_only_sidecar_audited_not_pilot_go',
            'composition_sha256': worker.sha_file(sidecar),
            'post_host_raw_sha256': worker.sha_file(raw),
            'total_attempt_bytes': total}


def launch_v6_pilot(*_args: Any, **_kwargs: Any) -> None:
    raise ValueError('V6 launcher HOLD: reviewed A/R, QA and executable composition absent')
