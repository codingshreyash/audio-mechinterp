"""Source-only external admission/review contract for the V6 resource pilot.

This module can validate artifacts; it cannot issue a pilot GO or load a model.
The dependency graph is worker W -> admission A -> independent review R ->
later launcher L, where L alone pins W, A and R byte hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker


@dataclass(frozen=True)
class ExternalPins:
    worker_sha256: str
    admission_sha256: str
    review_sha256: str


def verify_contract(files: worker.AdmissionFiles, review_path: Path,
                    pins: ExternalPins) -> dict[str, Any]:
    """Check externally pinned immutable bytes and independent review chain."""
    for name, digest in vars(pins).items():
        worker._sha256_field(digest, name)
    gate = worker.verify_admission(files, pins.admission_sha256, pins.worker_sha256)
    review = worker._read_bound(review_path, pins.review_sha256,
                                'independent pre-pilot review')
    bindings = gate['bindings']
    if (review.get('schema') != 'voxtral_track_b_v6_pre_pilot_independent_review_v1' or
            review.get('decision') != 'GO_for_pilot_only' or
            review.get('admission_sha256') != pins.admission_sha256 or
            review.get('worker_sha256') != pins.worker_sha256 or
            review.get('campaign_attempt_id') != bindings['campaign_attempt_id'] or
            review.get('campaign_claim_path') != bindings['campaign_claim_path']):
        raise ValueError('Independent review does not bind exact admission and campaign')
    return gate
