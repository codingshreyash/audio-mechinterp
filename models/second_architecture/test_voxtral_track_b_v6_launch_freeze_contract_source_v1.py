"""Fake/no-model V6 launch-freeze contract tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from models.second_architecture import voxtral_track_b_v6_launch_freeze_contract_source_v1 as freeze
from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_bound_batch_adapter_source_v1 as adapter
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture.test_voxtral_track_b_v6_pre_pilot_contract_v1 import Fixture
from models.second_architecture.test_voxtral_track_b_answer_link_core_v1 import fixture as fake_tensor_fixture


class LaunchFreezeSourceTests(unittest.TestCase):
    def test_external_admission_binds_reviewed_adapter_and_piper_choice(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            composition = freeze.HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py'
            comp_sha = worker.sha_file(composition)
            f.bindings.update(freeze.PINNED_SOURCE_SHAS)
            f.bindings.update({
                'composition_sha256': comp_sha,
                'old_fit_source_choice': freeze.PIPER_CHOICE,
                'old_fit_wav_sha256': adapter.OLD_WAV_SHA256,
                'old_fit_raw_tensor_sha256': adapter.OLD_FIT_RAW_HASHES,
                'old_fit_prompt': adapter.FIT_PROMPT,
                'track_b_prompt': adapter.TRACK_PROMPT})
            f.repin_admission_and_review()
            pins = freeze.FutureLauncherPins(f.pins, comp_sha)
            self.assertEqual(freeze.verify_launch_freeze(
                files=f.files, review_path=f.review_path,
                pins=pins, composition_source=composition)['status'],
                'source_only_launch_freeze_checked_not_pilot_go')
            f.bindings['old_fit_source_choice'] = 'color_05/kokoro/fit_chosen/silver'
            f.repin_admission_and_review()
            with self.assertRaisesRegex(ValueError, 'Explicit Piper'):
                freeze.verify_launch_freeze(files=f.files, review_path=f.review_path,
                    pins=freeze.FutureLauncherPins(f.pins, comp_sha),
                    composition_source=composition)

    def test_repinned_wrong_adapter_or_composition_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            composition = freeze.HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py'
            comp_sha = worker.sha_file(composition)
            f.bindings.update(freeze.PINNED_SOURCE_SHAS)
            f.bindings.update({
                'composition_sha256': comp_sha,
                'old_fit_source_choice': freeze.PIPER_CHOICE,
                'old_fit_wav_sha256': adapter.OLD_WAV_SHA256,
                'old_fit_raw_tensor_sha256': adapter.OLD_FIT_RAW_HASHES,
                'old_fit_prompt': adapter.FIT_PROMPT,
                'track_b_prompt': adapter.TRACK_PROMPT})
            f.bindings['batch_adapter_sha256'] = 'f' * 64
            f.repin_admission_and_review()
            with self.assertRaisesRegex(ValueError, 'batch_adapter_sha256'):
                freeze.verify_launch_freeze(files=f.files, review_path=f.review_path,
                    pins=freeze.FutureLauncherPins(f.pins, comp_sha),
                    composition_source=composition)
            f.bindings['batch_adapter_sha256'] = freeze.PINNED_SOURCE_SHAS['batch_adapter_sha256']
            f.bindings['composition_sha256'] = 'e' * 64
            f.repin_admission_and_review()
            with self.assertRaisesRegex(ValueError, 'Composition source'):
                freeze.verify_launch_freeze(files=f.files, review_path=f.review_path,
                    pins=freeze.FutureLauncherPins(f.pins, comp_sha),
                    composition_source=composition)

    def test_device_transfer_rechecks_worker_hashes(self):
        _model, batch, _source, _donor = fake_tensor_fixture()
        digest = worker._batch_sha(batch)
        bound = adapter.BoundPilotBatches(batch, batch, digest, digest,
                                          'a' * 64, 'b' * 64, 'c' * 64)
        bindings = {'old_fit_batch_sha256': digest, 'synthetic_batch_sha256': digest}
        old, zero = freeze.checked_transfer(bound=bound, bindings=bindings, device='cpu')
        self.assertEqual(worker._batch_sha(old), digest)
        self.assertEqual(worker._batch_sha(zero), digest)
        with self.assertRaisesRegex(ValueError, 'before transfer'):
            freeze.checked_transfer(bound=bound,
                bindings={**bindings, 'old_fit_batch_sha256': '0' * 64}, device='cpu')

    def test_source_freeze_precedes_bound_adapter_and_transfer(self):
        _model, batch, _source, _donor = fake_tensor_fixture()
        digest = worker._batch_sha(batch)
        bound = adapter.BoundPilotBatches(batch, batch, digest, digest,
                                          adapter.OLD_WAV_SHA256, 'b' * 64, 'c' * 64)
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            composition = freeze.HERE / 'voxtral_track_b_v6_pilot_composition_source_v1.py'
            comp_sha = worker.sha_file(composition)
            f.bindings.update(freeze.PINNED_SOURCE_SHAS)
            f.bindings.update({
                'composition_sha256': comp_sha,
                'old_fit_source_choice': freeze.PIPER_CHOICE,
                'old_fit_wav_sha256': adapter.OLD_WAV_SHA256,
                'old_fit_raw_tensor_sha256': adapter.OLD_FIT_RAW_HASHES,
                'old_fit_prompt': adapter.FIT_PROMPT,
                'track_b_prompt': adapter.TRACK_PROMPT,
                'old_fit_batch_sha256': digest,
                'synthetic_batch_sha256': digest})
            f.repin_admission_and_review()
            with patch.object(adapter, 'build_bound_pilot_batches', return_value=bound) as build:
                old, zero, evidence = freeze.prepare_bound_inputs_for_child(
                    files=f.files, review_path=f.review_path,
                    pins=freeze.FutureLauncherPins(f.pins, comp_sha),
                    composition_source=composition, device='cpu')
            build.assert_called_once()
            self.assertEqual(worker._batch_sha(old), digest)
            self.assertEqual(worker._batch_sha(zero), digest)
            self.assertEqual(evidence['source_freeze']['status'],
                             'source_only_launch_freeze_checked_not_pilot_go')

    def test_sidecars_and_watchdog_counted_but_never_go(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'pilot'
            output.mkdir()
            measurement = output / 'pilot_measurement.json'
            measurement.write_text('{}')
            claim = root / 'campaign.json'
            watchdog = root / 'campaign.json.watchdog'
            watchdog.write_text(json.dumps({
                'schema': 'voxtral_track_b_v6_parent_deadline_claim_v1',
                'campaign_claim_path': str(claim), 'output_dir': str(output.resolve()),
                'deadline_seconds': 240.0}))
            watchdog_result = root / 'campaign.json.watchdog.result.json'
            completed = {'status': 'child_completed', 'child_exit_code': 0,
                         'operation_status': 'schedule_complete_resource_unverified',
                         'elapsed_seconds': 120.0,
                         'cap': {'recommended_max_memory_bytes': 40 * 1024**3,
                                 'configured_allocator_cap_bytes': 30 * 1024**3,
                                 'configured_fraction': 0.75,
                                 'allocator_cap_kind': 'PyTorch MPS allocation limit; not exact device peak'},
                         'samples': {'sampled_driver_highwater_lower_bound_bytes': 1024,
                                     'sample_count': 10,
                                     'sample_interval_seconds': 0.02,
                                     'sampled_lower_bound_by_call_ordinal': {'1': 1024},
                                     'exact_transient_peak_measured': False}}
            watchdog_result.write_text(json.dumps(completed))
            raw = root / 'pilot.post_host.raw'
            raw.write_bytes(b'fake raw post host log')
            pre = datetime.now(timezone.utc) - timedelta(seconds=2)
            post = datetime.now(timezone.utc)
            pins = freeze.FutureLauncherPins(contract.ExternalPins(
                'a' * 64, 'b' * 64, 'c' * 64), 'd' * 64)
            saved = {
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
                'post_host_collected_utc': post.isoformat()}
            sidecar = root / 'pilot.composition.json'
            sidecar.write_text(json.dumps(saved))
            binding = {'campaign_claim_path': str(claim),
                       'campaign_attempt_id': 'fake_attempt'}
            claim.write_text(json.dumps({
                'schema': 'voxtral_track_b_v6_campaign_claim_v1',
                'campaign_attempt_id': 'fake_attempt',
                'output_dir': str(output.resolve()),
                'bindings_sha256': worker.sha_bytes(json.dumps(
                    binding, sort_keys=True, separators=(',', ':')).encode())}))
            result = freeze.audit_composition_sidecars(
                output_dir=output, bindings=binding, pins=pins,
                expected_measurement_sha256=worker.sha_file(measurement),
                free_disk_bytes=100 * 1024**3)
            self.assertEqual(result['status'], 'source_only_sidecar_audited_not_pilot_go')
            for changed in (
                    {**completed, 'elapsed_seconds': 241.0},
                    {**completed, 'elapsed_seconds': float('nan')},
                    {**completed, 'elapsed_seconds': float('inf')},
                    {**completed, 'elapsed_seconds': 0},
                    {key: value for key, value in completed.items() if key != 'elapsed_seconds'},
                    {key: value for key, value in completed.items() if key != 'cap'},
                    {key: value for key, value in completed.items() if key != 'samples'},
                    {**completed, 'operation_status': 'fake_complete'},
                    {**completed, 'cap': {**completed['cap'], 'configured_fraction': 1.0}},
                    {**completed, 'cap': {**completed['cap'],
                        'configured_allocator_cap_bytes': 31 * 1024**3}},
                    {**completed, 'samples': {**completed['samples'], 'sample_count': 0}},
                    {**completed, 'samples': {**completed['samples'],
                        'sampled_driver_highwater_lower_bound_bytes': 0}},
                    {**completed, 'samples': {**completed['samples'],
                        'sampled_driver_highwater_lower_bound_bytes': 31 * 1024**3}},
                    {**completed, 'samples': {**completed['samples'],
                        'sample_interval_seconds': 0.5}},
                    {**completed, 'samples': {**completed['samples'],
                        'sampled_lower_bound_by_call_ordinal': {}}},
            ):
                with self.subTest(changed=changed):
                    watchdog_result.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, 'Parent'):
                        freeze.audit_composition_sidecars(
                            output_dir=output, bindings=binding, pins=pins,
                            expected_measurement_sha256=worker.sha_file(measurement),
                            free_disk_bytes=100 * 1024**3)
            watchdog_result.write_text(json.dumps(completed))
            claim.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'claim incomplete'):
                freeze.audit_composition_sidecars(
                    output_dir=output, bindings=binding, pins=pins,
                    expected_measurement_sha256=worker.sha_file(measurement),
                    free_disk_bytes=100 * 1024**3)
            claim.write_text(json.dumps({
                'schema': 'voxtral_track_b_v6_campaign_claim_v1',
                'campaign_attempt_id': 'fake_attempt',
                'output_dir': str(output.resolve()),
                'bindings_sha256': worker.sha_bytes(json.dumps(
                    binding, sort_keys=True, separators=(',', ':')).encode())}))
            saved['post_host_collected_utc'] = pre.isoformat()
            sidecar.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'strictly later'):
                freeze.audit_composition_sidecars(
                    output_dir=output, bindings=binding, pins=pins,
                    expected_measurement_sha256=worker.sha_file(measurement),
                    free_disk_bytes=100 * 1024**3)

    def test_public_entry_is_hard_hold(self):
        with self.assertRaisesRegex(ValueError, 'HOLD'):
            freeze.launch_v6_pilot()


if __name__ == '__main__':
    unittest.main()
