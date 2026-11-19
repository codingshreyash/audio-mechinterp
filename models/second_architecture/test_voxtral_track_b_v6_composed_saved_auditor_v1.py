"""Synthetic composition saved-data tamper tests; no model load."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture import voxtral_track_b_v6_composed_saved_auditor_v1 as composed
from models.second_architecture import voxtral_track_b_v6_pilot_saved_auditor_v1 as base
from models.second_architecture.test_voxtral_track_b_v6_pilot_saved_auditor_v1 import (
    make_fixture, write_json)


def composed_fixture(root: Path) -> tuple[Path, dict]:
    pilot, args = make_fixture(root)
    review = root / 'review.md'
    review.write_text('Fake independent review; synthetic test only\n')
    post_raw = pilot.with_name(pilot.name + '.post_host.raw')
    post_raw.write_bytes(b'fake post-run raw host log')
    sidecar = pilot.with_name(pilot.name + '.composition.json')
    write_json(sidecar, {
        'schema': composed.SCHEMA, 'status': composed.STATUS,
        'pilot_measurement_sha256': args['expected_measurement_sha256'],
        'admission_sha256': args['expected_admission_sha256'],
        'review_sha256': base.sha(review), 'worker_sha256': base.WORKER_SHA,
        'campaign_attempt_id': 'fake_1',
        'post_host_raw_sha256': base.sha(post_raw),
        'post_host_raw_bytes': post_raw.stat().st_size,
        'pre_host_collected_utc': '2026-11-18T12:00:00+00:00',
        'post_host_collected_utc': '2026-11-18T12:00:01+00:00',
    })
    args.update(review=review,
                composition_source=composed.HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py',
                expected_review_sha256=base.sha(review),
                expected_composition_sha256=base.sha(sidecar))
    return pilot, args


class ComposedSavedTests(unittest.TestCase):
    def test_complete_fake_composition_stays_hold_and_counts_all_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            result = composed.audit_composed_saved(pilot, **args)
            self.assertEqual(result['status'], 'HOLD_no_post_pilot_go')
            self.assertFalse(result['go_issued'])
            self.assertEqual(result['full_attempt_output_bytes'],
                             result['pilot_output_and_claim_bytes'] +
                             result['composition_sidecar_bytes'] +
                             result['composition_post_host_raw_bytes'])
            self.assertGreater(result['post_host_collected_utc'],
                               result['pre_host_collected_utc'])

    def test_rejects_post_raw_and_rehashed_late_time_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            raw = pilot.with_name(pilot.name + '.post_host.raw')
            raw.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'raw size/hash'):
                composed.audit_composed_saved(pilot, **args)
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            sidecar_path = pilot.with_name(pilot.name + '.composition.json')
            sidecar = base.read_json(sidecar_path)
            sidecar['post_host_collected_utc'] = sidecar['pre_host_collected_utc']
            write_json(sidecar_path, sidecar)
            args['expected_composition_sha256'] = base.sha(sidecar_path)
            with self.assertRaisesRegex(ValueError, 'strictly later'):
                composed.audit_composed_saved(pilot, **args)

    def test_rejects_measurement_admission_or_sidecar_binding_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            sidecar_path = pilot.with_name(pilot.name + '.composition.json')
            row = base.read_json(sidecar_path)
            row['pilot_measurement_sha256'] = '0' * 64
            write_json(sidecar_path, row)
            args['expected_composition_sha256'] = base.sha(sidecar_path)
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                composed.audit_composed_saved(pilot, **args)
            row['pilot_measurement_sha256'] = args['expected_measurement_sha256']
            row['admission_sha256'] = '0' * 64
            write_json(sidecar_path, row)
            args['expected_composition_sha256'] = base.sha(sidecar_path)
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                composed.audit_composed_saved(pilot, **args)

    def test_full_attempt_cap_includes_both_external_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            worker_and_claim = base.audit_saved(pilot, **{
                key: value for key, value in args.items() if key not in
                ('review', 'composition_source', 'expected_review_sha256',
                 'expected_composition_sha256')})['output_bytes']
            with patch.object(base, 'MAX_OUTPUT_BYTES', worker_and_claim):
                with self.assertRaisesRegex(ValueError, 'Full composed attempt'):
                    composed.audit_composed_saved(pilot, **args)


if __name__ == '__main__':
    unittest.main()
