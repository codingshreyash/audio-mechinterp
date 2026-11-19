"""Prospective one-shot V6 pilot launcher. Public launch is HOLD until A/R pins.

This is source wiring for independent review, not a pilot GO. The acyclic
freeze is worker W -> external admission A -> independent review R -> this L.
Only a later reviewed edit may fill the exact A/R SHA-256 constants below.
No candidate weights are loaded by importing this module or checking HOLD.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as pre_pilot
from models.second_architecture import voxtral_track_b_v6_launch_freeze_contract_source_v1 as freeze
from models.second_architecture import voxtral_track_b_v6_pinned_loader_source_v1 as loader
from models.second_architecture import voxtral_track_b_v6_mps_runtime_adapter_source_v1 as runtime
from models.second_architecture import voxtral_track_b_v6_composed_saved_auditor_v1 as saved_auditor


HERE = Path(__file__).resolve().parent
WORKER_SHA256 = freeze.PINNED_SOURCE_SHAS['pilot_worker_sha256']
COMPOSITION_SHA256 = 'da6ee3f5600f7cb9c32af2eb0ce3fe347257bd451032bd5e2136aa9cf57ea02a'
ADMISSION_SHA256: str | None = None  # no external A exists
REVIEW_SHA256: str | None = None  # no independent R exists
PARENT_RESULT_RESERVE_BYTES = 1024**2
LAUNCHER_FILE = 'voxtral_track_b_v6_one_shot_launcher_source_v1.py'
COMPOSED_AUDITOR_SHA256 = 'f12c64f2f64045b2b8aba1efc24a7bec8a37bb32ad7423baa7c79d8895524218'


def _frozen_pins() -> freeze.FutureLauncherPins:
    """Reject before model/processor construction while A/R are unreviewed."""
    if ADMISSION_SHA256 is None or REVIEW_SHA256 is None:
        raise ValueError('V6 pilot HOLD: exact external A/R source pins absent')
    worker._sha256_field(ADMISSION_SHA256, 'frozen admission SHA')
    worker._sha256_field(REVIEW_SHA256, 'frozen review SHA')
    return freeze.FutureLauncherPins(
        external=contract.ExternalPins(WORKER_SHA256, ADMISSION_SHA256, REVIEW_SHA256),
        composition_sha256=COMPOSITION_SHA256)


def _verify_launcher_source(expected_launcher_sha256: str | None) -> str:
    """Compare L with a later externally reviewed digest; never self-pin L."""
    if expected_launcher_sha256 is None:
        raise ValueError('V6 pilot HOLD: independent final L source pin absent')
    worker._sha256_field(expected_launcher_sha256, 'expected_launcher_sha256')
    path = HERE / LAUNCHER_FILE
    if (not path.is_file() or path.is_symlink() or
            worker.sha_file(path) != expected_launcher_sha256):
        raise ValueError('V6 pilot launcher differs from external L source pin')
    auditor = HERE / 'voxtral_track_b_v6_composed_saved_auditor_v1.py'
    if (not auditor.is_file() or auditor.is_symlink() or
            worker.sha_file(auditor) != COMPOSED_AUDITOR_SHA256):
        raise ValueError('V6 composed saved auditor source differs')
    return expected_launcher_sha256


def _check_attempt_evidence_budget(output_dir: Path, claim: Path,
                                   external_bytes: int, *, already_written: bool = False) -> int:
    """Count run files and external sidecars, reserving the parent result."""
    if (not output_dir.is_dir() or output_dir.is_symlink() or
            not claim.is_file() or claim.is_symlink() or
            type(external_bytes) is not int or external_bytes < 0):
        raise ValueError('Pilot evidence path or pending byte count invalid')
    watchdog = runtime._claim_path(claim)
    result_path = watchdog.with_name(watchdog.name + '.result.json')
    if (not watchdog.is_file() or watchdog.is_symlink() or
            result_path.exists() or result_path.is_symlink()):
        raise ValueError('Pilot watchdog claim/result state invalid')
    total = claim.stat().st_size + watchdog.stat().st_size
    for path in output_dir.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in pilot evidence')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('Irregular pilot evidence path')
        total += path.stat().st_size
        if total + external_bytes + PARENT_RESULT_RESERVE_BYTES > worker.MAX_OUTPUT_BYTES:
            raise ValueError('Full child output cap would be exceeded before post-host write')
    total += external_bytes + PARENT_RESULT_RESERVE_BYTES
    if total > worker.MAX_OUTPUT_BYTES:
        raise ValueError('Full child output cap would be exceeded before post-host write')
    to_write = 0 if already_written else external_bytes
    if shutil.disk_usage(output_dir).free < worker.MIN_FREE_BYTES + to_write + PARENT_RESULT_RESERVE_BYTES:
        raise ValueError('Free-disk bound would be exceeded before post-host write')
    return total


def _pinned_pad_before_load(files: worker.AdmissionFiles,
                            bindings: Mapping[str, Any]) -> int:
    """Read only the manifest-bound generation config; no weights or factory."""
    frozen = json.loads(files.preflight.read_text())
    manifest = loader._snapshot_manifest()
    if (manifest != frozen.get('snapshot') or
            manifest.get('snapshot_tree', {}).get('sha256') !=
            bindings.get('checkpoint_tree_sha256')):
        raise ValueError('Pinned snapshot changed before pad read')
    path = loader._pinned_snapshot_path().resolve(strict=True) / 'generation_config.json'
    if not path.is_file() or path.is_symlink():
        raise ValueError('Pinned generation config absent or symlinked')
    config = json.loads(path.read_text())
    pad = config.get('pad_token_id')
    if (type(pad) is not int or pad < 0 or
            config.get('eos_token_id') != worker.core.EOS):
        raise ValueError('Pinned generation config token IDs invalid')
    return pad


def _fresh_worker_host(*, post: bool) -> tuple[dict[str, Any], str]:
    """Reuse the reviewed Mac host parser and preserve exact raw pmset bytes."""
    source_dir = str(HERE)
    sys.path.insert(0, source_dir)
    try:
        from models.second_architecture import voxtral_fp32_track_b_preflight_v1 as preflight
        evidence = preflight.host_status()
    finally:
        sys.path.remove(source_dir)
    raw = pre_pilot._verify_fresh_host(evidence)
    return ({'ac': True, 'lid_open': True, 'ten_minute_clean': True,
             'during_run_clean': True if post else None,
             'raw_power_log': raw}, evidence['collected_utc'])


def _write_post_host_sidecar(*, files: worker.AdmissionFiles, review_path: Path,
                             pins: freeze.FutureLauncherPins, output_dir: Path,
                             pre_collected: str,
                             result: Mapping[str, Any],
                             expected_launcher_sha256: str) -> dict[str, Any]:
    producer_sha = _verify_launcher_source(expected_launcher_sha256)
    after, post_collected = _fresh_worker_host(post=True)
    if (not isinstance(pre_collected, str) or len(pre_collected) > 64 or
            not isinstance(post_collected, str) or len(post_collected) > 64):
        raise ValueError('Post-host collection time unbounded')
    if datetime.fromisoformat(post_collected) <= datetime.fromisoformat(pre_collected):
        raise ValueError('Post-host probe is not later than pre-host probe')
    raw = after['raw_power_log']
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 128 * 1024**2:
        raise ValueError('Post-host raw log absent or unbounded')
    raw_path = output_dir.with_name(output_dir.name + '.post_host.raw')
    sidecar = output_dir.with_name(output_dir.name + '.composition.json')
    bindings = contract.verify_contract(files, review_path, pins.external)['bindings']
    worker._sha256_field(result.get('pilot_measurement_sha256'),
                         'pilot_measurement_sha256')
    sidecar_value = {
        'schema': 'voxtral_track_b_v6_composition_source_only_v1',
        'status': 'worker_complete_post_host_checked_not_pilot_go',
        'pilot_measurement_sha256': result['pilot_measurement_sha256'],
        'post_host_raw_sha256': worker.sha_bytes(raw),
        'post_host_raw_bytes': len(raw),
        'pre_host_collected_utc': pre_collected,
        'post_host_collected_utc': post_collected,
        'admission_sha256': pins.external.admission_sha256,
        'review_sha256': pins.external.review_sha256,
        'worker_sha256': pins.external.worker_sha256,
        'campaign_attempt_id': bindings['campaign_attempt_id'],
        'producer_source_file': LAUNCHER_FILE,
        'producer_source_sha256': producer_sha,
        'composition_schema_source_sha256': COMPOSITION_SHA256,
    }
    sidecar_raw = (json.dumps(sidecar_value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False) + '\n').encode()
    claim = Path(bindings['campaign_claim_path'])
    if raw_path.exists() or raw_path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise ValueError('Post-host sidecar already exists; no retry')
    _check_attempt_evidence_budget(output_dir, claim, len(raw) + len(sidecar_raw))
    for path, payload in ((raw_path, raw), (sidecar, sidecar_raw)):
        with path.open('xb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    if (raw_path.stat().st_size != len(raw) or worker.sha_file(raw_path) != worker.sha_bytes(raw) or
            sidecar.stat().st_size != len(sidecar_raw) or
            worker.sha_file(sidecar) != worker.sha_bytes(sidecar_raw)):
        raise ValueError('Post-host sidecar changed during write')
    _check_attempt_evidence_budget(output_dir, claim, len(raw) + len(sidecar_raw),
                                   already_written=True)
    return {'status': 'schedule_complete_resource_unverified',
            'composition_sha256': worker.sha_file(sidecar),
            'sidecar_producer_source_sha256': producer_sha}


@dataclass(frozen=True)
class PilotChild:
    """Top-level spawn-pickleable operation; MPS cap is set by runtime first."""

    files: worker.AdmissionFiles
    review_path: Path
    pins: freeze.FutureLauncherPins
    output_dir: Path
    expected_launcher_sha256: str | None = None

    def __call__(self) -> dict[str, Any]:
        _verify_launcher_source(self.expected_launcher_sha256)
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('V6 pilot requires MPS')
        proof = pre_pilot.verify_pre_pilot(
            files=self.files, review_path=self.review_path, pins=self.pins.external)
        old, zero, transfer = freeze.prepare_bound_inputs_for_child(
            files=self.files, review_path=self.review_path, pins=self.pins,
            composition_source=HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py',
            device='mps')
        if transfer['transfer_device'] != 'mps':
            raise ValueError('V6 pilot input transfer was not MPS')
        bindings = contract.verify_contract(
            self.files, self.review_path, self.pins.external)['bindings']
        pad = _pinned_pad_before_load(self.files, bindings)
        pre_collected = proof['fresh_host_collected_utc']

        def host_probe(post: bool) -> Mapping[str, Any]:
            nonlocal pre_collected
            evidence, collected = _fresh_worker_host(post=post)
            if not post:
                pre_collected = collected
            return evidence

        def load_model() -> Any:
            model, _processor, load_evidence = loader._load_pinned_fp32_voxtral(
                files=self.files, review_path=self.review_path, pins=self.pins.external)
            if load_evidence['pad_token_id'] != pad:
                raise ValueError('Loaded model pad token differs from pinned config')
            return model

        result = worker._execute_after_admission(
            load_model=load_model, old_fit_batch=old, synthetic_batch=zero,
            pad=pad, output_dir=self.output_dir, bindings=bindings,
            host_probe=host_probe,
            memory_probe=torch.mps.driver_allocated_memory,
            free_bytes=lambda path: shutil.disk_usage(path).free,
            sync=torch.mps.synchronize)
        if result.get('status') != 'schedule_complete_resource_unverified':
            raise ValueError('V6 pilot schedule did not meet resource projection')
        return _write_post_host_sidecar(
            files=self.files, review_path=self.review_path, pins=self.pins,
            output_dir=self.output_dir, pre_collected=pre_collected, result=result,
            expected_launcher_sha256=self.expected_launcher_sha256)


def launch_pilot(*, files: worker.AdmissionFiles, review_path: Path,
                 output_dir: Path,
                 expected_launcher_sha256: str | None = None) -> dict[str, Any]:
    """Exactly one bounded attempt; read-only audit never issues post-pilot GO."""
    pins = _frozen_pins()  # first: no A/R means no child and no model load
    producer_sha = _verify_launcher_source(expected_launcher_sha256)
    freeze.verify_launch_freeze(
        files=files, review_path=review_path, pins=pins,
        composition_source=HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py')
    pre_pilot.verify_pre_pilot(files=files, review_path=review_path,
                               pins=pins.external)
    bindings = contract.verify_contract(files, review_path, pins.external)['bindings']
    claim = Path(bindings['campaign_claim_path'])
    if (not output_dir.is_absolute() or not output_dir.parent.is_dir() or
            output_dir.parent.is_symlink() or output_dir.exists() or
            shutil.disk_usage(output_dir.parent).free < worker.MIN_FREE_BYTES):
        raise ValueError('V6 pilot output path or free-disk gate invalid')
    parent = runtime._guarded_child_attempt(
        campaign_claim_path=claim, output_dir=output_dir,
        operation=PilotChild(files, review_path, pins, output_dir, producer_sha), mps=None,
        deadline_seconds=runtime.DEADLINE_SECONDS)
    freeze._audit_parent_result(parent)
    measurement = output_dir / 'pilot_measurement.json'
    digest = worker.sha_file(measurement)
    sidecar_audit = freeze.audit_composition_sidecars(
        output_dir=output_dir, bindings=bindings, pins=pins,
        expected_measurement_sha256=digest,
        free_disk_bytes=shutil.disk_usage(output_dir).free)
    saved = saved_auditor.audit_composed_saved(
        output_dir, admission=files.admission, qa=files.qa,
        qa_review=files.qa_review, preflight=files.preflight,
        study_runner=files.study_runner,
        worker=HERE / 'voxtral_track_b_v6_pilot_worker_source_v1.py',
        review=review_path,
        composition_source=HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py',
        launcher_source=HERE / LAUNCHER_FILE,
        expected_launcher_sha256=producer_sha,
        expected_measurement_sha256=digest,
        expected_admission_sha256=pins.external.admission_sha256,
        expected_review_sha256=pins.external.review_sha256,
        expected_composition_sha256=sidecar_audit['composition_sha256'],
        free_disk_bytes=shutil.disk_usage(output_dir).free)
    if saved['full_attempt_output_bytes'] != sidecar_audit['total_attempt_bytes']:
        raise ValueError('Independent full-attempt byte inventories disagree')
    return {'status': 'HOLD_no_post_pilot_go', 'go_issued': False,
            'sidecar_producer_source_sha256': producer_sha,
            'saved_audit': saved, 'runtime': parent}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('admission', 'qa', 'qa-review', 'preflight', 'review', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-launcher-sha256', required=True)
    args = parser.parse_args()
    files = worker.AdmissionFiles(
        admission=args.admission, qa=args.qa, qa_review=args.qa_review,
        preflight=args.preflight,
        study_runner=HERE / 'voxtral_fp32_track_b_study_runner_v1.py',
        pilot_auditor=HERE / 'voxtral_track_b_v6_pilot_saved_auditor_v1.py')
    print(json.dumps(launch_pilot(files=files, review_path=args.review,
                                  output_dir=args.output_dir,
                                  expected_launcher_sha256=args.expected_launcher_sha256), sort_keys=True,
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
