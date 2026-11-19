"""Read-only composed V6 saved-data audit; never emits pilot or Track B GO.

This composes the reviewed worker saved-audit with the prospective post-host
composition sidecar. It does not parse raw pmset semantics or load weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from models.second_architecture import voxtral_track_b_v6_pilot_saved_auditor_v1 as base


HERE = Path(__file__).resolve().parent
COMPOSITION_SOURCE_SHA = '487385e17dd629a045f0ef5eb2bf796afd16a2a1e059bf94848f0c8834c10c80'
SCHEMA = 'voxtral_track_b_v6_composition_source_only_v1'
STATUS = 'worker_complete_post_host_checked_not_pilot_go'


def audit_composed_saved(pilot_dir: Path, *, admission: Path, qa: Path,
                         qa_review: Path, preflight: Path, study_runner: Path,
                         worker: Path, review: Path, composition_source: Path,
                         expected_measurement_sha256: str,
                         expected_admission_sha256: str,
                         expected_review_sha256: str,
                         expected_composition_sha256: str,
                         free_disk_bytes: int) -> dict:
    """Bind post-host evidence to the exact worker pilot and full-attempt cap."""
    base.require(base.sha(composition_source) == COMPOSITION_SOURCE_SHA,
                 'Frozen composition source changed')
    worker_audit = base.audit_saved(
        pilot_dir, admission=admission, qa=qa, qa_review=qa_review,
        preflight=preflight, study_runner=study_runner, worker=worker,
        expected_measurement_sha256=expected_measurement_sha256,
        expected_admission_sha256=expected_admission_sha256,
        free_disk_bytes=free_disk_bytes)
    sidecar_path = pilot_dir.with_name(pilot_dir.name + '.composition.json')
    raw_path = pilot_dir.with_name(pilot_dir.name + '.post_host.raw')
    sidecar = base.read_json(sidecar_path)
    base.require(base.sha(sidecar_path) == expected_composition_sha256,
                 'Externally pinned composition sidecar SHA differs')
    base.require(raw_path.is_file() and not raw_path.is_symlink(),
                 'Missing or symlinked composition post-host raw log')
    base.require(review.is_file() and not review.is_symlink() and
                 base.sha(review) == expected_review_sha256,
                 'Independent pre-pilot review bytes differ')
    measurement = base.read_json(pilot_dir / 'pilot_measurement.json')
    binding = measurement['bindings']
    base.require(sidecar.get('schema') == SCHEMA and sidecar.get('status') == STATUS and
                 sidecar.get('pilot_measurement_sha256') == expected_measurement_sha256 and
                 sidecar.get('admission_sha256') == expected_admission_sha256 and
                 sidecar.get('review_sha256') == expected_review_sha256 and
                 sidecar.get('worker_sha256') == base.WORKER_SHA and
                 sidecar.get('campaign_attempt_id') == binding.get('campaign_attempt_id'),
                 'Composition sidecar measurement/admission/review/worker binding differs')
    raw_bytes = raw_path.stat().st_size
    base.require(0 < raw_bytes <= 128 * 1024**2 and
                 sidecar.get('post_host_raw_bytes') == raw_bytes and
                 sidecar.get('post_host_raw_sha256') == base.sha(raw_path),
                 'Composition post-host raw size/hash differs')
    try:
        pre = datetime.fromisoformat(sidecar['pre_host_collected_utc'])
        post = datetime.fromisoformat(sidecar['post_host_collected_utc'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Composition pre/post host collection timestamp invalid') from exc
    base.require(pre.tzinfo is not None and post.tzinfo is not None and post > pre,
                 'Composition post-host collection must be strictly later than pre-host')
    full_bytes = worker_audit['output_bytes'] + sidecar_path.stat().st_size + raw_bytes
    base.require(full_bytes <= base.MAX_OUTPUT_BYTES and
                 free_disk_bytes >= base.MIN_FREE_BYTES,
                 'Full composed attempt output/free-disk cap exceeded')
    return {
        'schema': 'voxtral_track_b_v6_composed_saved_audit_v1',
        'status': 'HOLD_no_post_pilot_go', 'go_issued': False,
        'hold_reasons': worker_audit['hold_reasons'] + [
            'composition_post_host_raw_requires_independent_semantic_review',
            'sidecar_timestamps_cannot_reconstruct_worker_monotonic_endpoints'],
        'pilot_measurement_sha256': expected_measurement_sha256,
        'admission_sha256': expected_admission_sha256,
        'review_sha256': expected_review_sha256,
        'composition_sha256': expected_composition_sha256,
        'post_host_raw_sha256': base.sha(raw_path),
        'pre_host_collected_utc': pre.isoformat(),
        'post_host_collected_utc': post.isoformat(),
        'pilot_output_and_claim_bytes': worker_audit['output_bytes'],
        'composition_sidecar_bytes': sidecar_path.stat().st_size,
        'composition_post_host_raw_bytes': raw_bytes,
        'full_attempt_output_bytes': full_bytes,
        'timing_recomputed': worker_audit['timing_recomputed'],
        'timing_feasible': worker_audit['timing_feasible'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('pilot-dir', 'admission', 'qa', 'qa-review', 'preflight', 'review'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('expected-measurement-sha256', 'expected-admission-sha256',
                 'expected-review-sha256', 'expected-composition-sha256'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    result = audit_composed_saved(
        args.pilot_dir, admission=args.admission, qa=args.qa,
        qa_review=args.qa_review, preflight=args.preflight,
        study_runner=HERE / 'voxtral_fp32_track_b_study_runner_v1.py',
        worker=HERE / 'voxtral_track_b_v6_pilot_worker_source_v1.py',
        review=args.review,
        composition_source=HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py',
        expected_measurement_sha256=args.expected_measurement_sha256,
        expected_admission_sha256=args.expected_admission_sha256,
        expected_review_sha256=args.expected_review_sha256,
        expected_composition_sha256=args.expected_composition_sha256,
        free_disk_bytes=shutil.disk_usage(args.pilot_dir).free)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
