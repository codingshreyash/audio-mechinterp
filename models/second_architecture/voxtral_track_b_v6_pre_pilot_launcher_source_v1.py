"""Source-only V6 launcher gate. Public model execution is deliberately HOLD.

There is no checkpoint loader or CLI here. A later reviewed launcher must pin
worker/admission/review hashes in its own bytes, add a hard per-call watchdog
and measured MPS high-water, then delegate to the worker private schedule.
"""

from __future__ import annotations

import base64
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract


FROZEN_PINS: contract.ExternalPins | None = None  # no externally reviewed A/R exists
FRESH_HOST_MAX_SECONDS = 60
STABLE_PREFLIGHT_FIELDS = (
    'source_sha256', 'anchors', 'installed_source_sha256', 'snapshot',
    'fixture', 'selected_wavs', 'processor', 'v6_synthetic_zero_geometry',
)


def _actual_qa() -> dict[str, Any]:
    from models.second_architecture.word_disjoint_v2 import candidate_common as common
    cases, _fixture = common.fixture_cases()
    result = common.qa_gate(cases)
    return {'qa_go_receipt_sha256': result['qa_receipt_sha256'],
            'qa_review_sha256': result['qa_review_sha256'],
            'human_qa_audit_sha256': common.sha(common.HUMAN_QA_AUDIT)}


def _fresh_no_weight_preflight() -> dict[str, Any]:
    # The frozen preflight imports its sibling modules by top-level names.
    source_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, source_dir)
    try:
        from models.second_architecture import voxtral_fp32_track_b_preflight_v1 as preflight
        return preflight.run(check_host=True)
    finally:
        sys.path.remove(source_dir)


def _verify_fresh_host(host: Any) -> None:
    if not isinstance(host, dict) or host.get('eligible_at_collection') is not True:
        raise ValueError('Fresh host eligibility absent')
    try:
        collected = datetime.fromisoformat(host['collected_utc'])
        age = (datetime.now(timezone.utc) - collected).total_seconds()
        packed = base64.b64decode(host['power_log_gzip_base64'], validate=True)
        raw = gzip.decompress(packed)
        hashes = host['output_sha256']
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError('Fresh host evidence malformed') from exc
    if (not isinstance(hashes, dict) or collected.tzinfo is None or
            not 0 <= age <= FRESH_HOST_MAX_SECONDS or
            worker.sha_bytes(packed) != host.get('power_log_gzip_sha256') or
            worker.sha_bytes(raw) != hashes.get('power_log') or
            len(raw) != host.get('power_log_raw_bytes')):
        raise ValueError('Fresh host timestamp or raw log binding differs')


def _verify_admission_fingerprints(receipt: dict[str, Any],
                                   bindings: dict[str, Any]) -> None:
    snapshot = receipt.get('snapshot')
    installed = receipt.get('installed_source_sha256')
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('snapshot_tree'), dict):
        raise ValueError('Preflight checkpoint tree fingerprint absent')
    tree_sha = snapshot['snapshot_tree'].get('sha256')
    worker._sha256_field(tree_sha, 'preflight snapshot tree')
    if (not isinstance(installed, dict) or not installed or
            any(not isinstance(key, str) or not key for key in installed)):
        raise ValueError('Preflight installed source fingerprint map absent')
    for key, value in installed.items():
        worker._sha256_field(value, f'installed source {key}')
    installed_digest = worker.sha_bytes(json.dumps(
        installed, sort_keys=True, separators=(',', ':')).encode())
    if (bindings.get('checkpoint_tree_sha256') != tree_sha or
            bindings.get('transformers_source_sha256') != installed_digest):
        raise ValueError('Admission checkpoint or Transformers source binding differs')


def verify_pre_pilot(*, files: worker.AdmissionFiles, review_path: Path,
                     pins: contract.ExternalPins) -> dict[str, Any]:
    """Recheck actual 32/32 QA, source/processor bytes, and fresh host before load."""
    gate = contract.verify_contract(files, review_path, pins)
    bindings = gate['bindings']
    qa = _actual_qa()  # candidate_common.qa_gate re-audits human exports
    for key in ('qa_go_receipt_sha256', 'qa_review_sha256', 'human_qa_audit_sha256'):
        if qa.get(key) != bindings.get(key):
            raise ValueError(f'Actual human QA {key} differs from admission')
    if qa['human_qa_audit_sha256'] != bindings['qa_sha256']:
        raise ValueError('Worker human QA binding differs from actual audit')
    frozen = json.loads(files.preflight.read_text())
    fresh = _fresh_no_weight_preflight()  # no candidate weights or inference
    if (fresh.get('schema') != 'voxtral_fp32_track_b_no_weight_preflight_v1' or
            fresh.get('candidate_weights_loaded') is not False or
            fresh.get('candidate_forward_count') != 0 or
            fresh.get('human_qa', {}).get('passed') is not True):
        raise ValueError('Fresh no-weight preflight or QA failed')
    for key in STABLE_PREFLIGHT_FIELDS:
        if fresh.get(key) != frozen.get(key):
            raise ValueError(f'Fresh source/processor fingerprint changed: {key}')
    _verify_admission_fingerprints(frozen, bindings)
    _verify_admission_fingerprints(fresh, bindings)
    _verify_fresh_host(fresh.get('host'))
    return {'admission_sha256': pins.admission_sha256,
            'worker_sha256': pins.worker_sha256,
            'review_sha256': pins.review_sha256,
            'campaign_attempt_id': bindings['campaign_attempt_id'],
            'campaign_claim_path': bindings['campaign_claim_path'],
            'fresh_host_collected_utc': fresh['host']['collected_utc']}


def launch_pilot(*, files: worker.AdmissionFiles, review_path: Path,
                 load_model: Any, **_unused: Any) -> None:
    """No public delegation until independently reviewed runtime bounds exist."""
    if FROZEN_PINS is None:
        raise ValueError('V6 pilot HOLD: worker/admission/review hashes not pinned')
    verify_pre_pilot(files=files, review_path=review_path, pins=FROZEN_PINS)
    raise ValueError('V6 pilot HOLD: hard watchdog and MPS peak sampler absent')
