"""Synthetic composition saved-data tamper tests; no model load."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture import voxtral_track_b_v6_composed_saved_auditor_v1 as composed
from models.second_architecture import voxtral_track_b_v6_pilot_saved_auditor_v1 as base
from models.second_architecture import voxtral_track_b_v6_mps_runtime_adapter_source_v1 as runtime
from models.second_architecture import voxtral_track_b_v6_launch_freeze_contract_source_v1 as freeze
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
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
        'producer_source_file': composed.LAUNCHER_FILE,
        'producer_source_sha256': base.sha(composed.HERE / composed.LAUNCHER_FILE),
        'composition_schema_source_sha256': composed.COMPOSITION_SOURCE_SHA,
    })
    args.update(review=review,
                composition_source=composed.HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py',
                launcher_source=composed.HERE / composed.LAUNCHER_FILE,
                expected_launcher_sha256=base.sha(composed.HERE / composed.LAUNCHER_FILE),
                expected_review_sha256=base.sha(review),
                expected_composition_sha256=base.sha(sidecar))
    claim = Path(base.read_json(args['admission'])['bindings']['campaign_claim_path'])
    watchdog = runtime._claim_path(claim)
    write_json(watchdog, {
        'schema': 'voxtral_track_b_v6_parent_deadline_claim_v1',
        'campaign_claim_path': str(claim),
        'output_dir': str(pilot.resolve()),
        'deadline_seconds': runtime.DEADLINE_SECONDS,
    })
    write_json(watchdog.with_name(watchdog.name + '.result.json'), {
        'status': 'child_completed', 'child_exit_code': 0,
        'operation_status': 'schedule_complete_resource_unverified',
        'elapsed_seconds': 120.0,
        'cap': {'recommended_max_memory_bytes': 40 * 1024**3,
                'configured_allocator_cap_bytes': runtime.CAP_BYTES,
                'configured_fraction': runtime.CAP_BYTES / (40 * 1024**3),
                'allocator_cap_kind': 'PyTorch MPS allocation limit; not exact device peak'},
        'samples': {'sampled_driver_highwater_lower_bound_bytes': 1024,
                    'sample_count': 10,
                    'sample_interval_seconds': runtime.SAMPLE_INTERVAL_SECONDS,
                    'sampled_lower_bound_by_call_ordinal': {'1': 1024},
                    'exact_transient_peak_measured': False},
    })
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
                             result['composition_post_host_raw_bytes'] +
                             result['watchdog_claim_bytes'] +
                             result['watchdog_result_bytes'])
            self.assertGreater(result['post_host_collected_utc'],
                               result['pre_host_collected_utc'])
            self.assertEqual(result['sidecar_producer_source_sha256'],
                             args['expected_launcher_sha256'])

    def test_watchdog_bytes_enter_full_total_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            before = composed.audit_composed_saved(pilot, **args)
            claim = Path(base.read_json(args['admission'])['bindings']['campaign_claim_path'])
            watchdog = runtime._claim_path(claim)
            result_path = watchdog.with_name(watchdog.name + '.result.json')
            watchdog.write_bytes(watchdog.read_bytes() + b' ' * 73)
            result_path.write_bytes(result_path.read_bytes() + b' ' * 91)
            after = composed.audit_composed_saved(pilot, **args)
            self.assertEqual(after['full_attempt_output_bytes'] -
                             before['full_attempt_output_bytes'], 164)
            self.assertEqual(after['watchdog_claim_bytes'] -
                             before['watchdog_claim_bytes'], 73)
            self.assertEqual(after['watchdog_result_bytes'] -
                             before['watchdog_result_bytes'], 91)
            self.assertEqual(after['watchdog_claim_sha256'], base.sha(watchdog))
            self.assertEqual(after['watchdog_result_sha256'], base.sha(result_path))

    def test_full_total_agrees_with_separate_freeze_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            binding = base.read_json(pilot / 'pilot_measurement.json')['bindings']
            pins = freeze.FutureLauncherPins(contract.ExternalPins(
                base.WORKER_SHA, args['expected_admission_sha256'],
                args['expected_review_sha256']),
                base.sha(args['composition_source']))
            separate = freeze.audit_composition_sidecars(
                output_dir=pilot, bindings=binding, pins=pins,
                expected_measurement_sha256=args['expected_measurement_sha256'],
                free_disk_bytes=100 * 1024**3)
            composed_result = composed.audit_composed_saved(pilot, **args)
            self.assertEqual(composed_result['full_attempt_output_bytes'],
                             separate['total_attempt_bytes'])

    def test_rejects_missing_or_tampered_watchdog_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            claim = Path(base.read_json(args['admission'])['bindings']['campaign_claim_path'])
            watchdog = runtime._claim_path(claim)
            result_path = watchdog.with_name(watchdog.name + '.result.json')
            result_path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing or symlinked'):
                composed.audit_composed_saved(pilot, **args)
            write_json(result_path, {'status': 'child_completed'})
            with self.assertRaisesRegex(ValueError, 'Parent watchdog'):
                composed.audit_composed_saved(pilot, **args)
            parent_claim = base.read_json(watchdog)
            parent_claim['output_dir'] = str(Path(tmp) / 'other')
            write_json(watchdog, parent_claim)
            with self.assertRaisesRegex(ValueError, 'Parent watchdog claim path or identity'):
                composed.audit_composed_saved(pilot, **args)

    def test_rejects_missing_or_tampered_actual_launcher_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            sidecar_path = pilot.with_name(pilot.name + '.composition.json')
            row = base.read_json(sidecar_path)
            row.pop('producer_source_sha256')
            write_json(sidecar_path, row)
            args['expected_composition_sha256'] = base.sha(sidecar_path)
            with self.assertRaisesRegex(ValueError, 'producer/schema source provenance'):
                composed.audit_composed_saved(pilot, **args)
            row['producer_source_sha256'] = '0' * 64
            write_json(sidecar_path, row)
            args['expected_composition_sha256'] = base.sha(sidecar_path)
            with self.assertRaisesRegex(ValueError, 'producer/schema source provenance'):
                composed.audit_composed_saved(pilot, **args)
        with tempfile.TemporaryDirectory() as tmp:
            pilot, args = composed_fixture(Path(tmp))
            args['expected_launcher_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Actual launcher source differs'):
                composed.audit_composed_saved(pilot, **args)

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
                ('review', 'composition_source', 'launcher_source',
                 'expected_launcher_sha256', 'expected_review_sha256',
                 'expected_composition_sha256')})['output_bytes']
            with patch.object(base, 'MAX_OUTPUT_BYTES', worker_and_claim):
                with self.assertRaisesRegex(ValueError, 'Full composed attempt'):
                    composed.audit_composed_saved(pilot, **args)


if __name__ == '__main__':
    unittest.main()
