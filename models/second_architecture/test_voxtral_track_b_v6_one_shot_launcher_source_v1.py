"""Fake-only tests for the prospective V6 one-shot launcher; no weights."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from models.second_architecture import voxtral_track_b_v6_one_shot_launcher_source_v1 as launch
from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture import voxtral_track_b_v6_launch_freeze_contract_source_v1 as freeze


def fake_files(tmp_path: Path) -> worker.AdmissionFiles:
    return worker.AdmissionFiles(*(tmp_path / name for name in (
        'A.json', 'qa.json', 'qa_review.md', 'preflight.json',
        'study.py', 'auditor.py')))


def fake_pins() -> freeze.FutureLauncherPins:
    return freeze.FutureLauncherPins(
        contract.ExternalPins(launch.WORKER_SHA256, 'a' * 64, 'b' * 64),
        launch.COMPOSITION_SHA256)


def test_public_launch_holds_before_process_or_model(tmp_path: Path) -> None:
    with (mock.patch.object(launch.runtime, '_guarded_child_attempt') as child,
          mock.patch.object(launch.loader, '_load_pinned_fp32_voxtral') as model):
        with pytest.raises(ValueError, match='exact external A/R source pins absent'):
            launch.launch_pilot(files=fake_files(tmp_path),
                                review_path=tmp_path / 'R.json',
                                output_dir=tmp_path / 'attempt')
        child.assert_not_called()
        model.assert_not_called()


def test_pinned_pad_reads_manifest_bound_config_without_model(tmp_path: Path) -> None:
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'generation_config.json').write_text(
        json.dumps({'pad_token_id': 11, 'eos_token_id': worker.core.EOS}))
    files = fake_files(tmp_path)
    files.preflight.write_text(json.dumps({'snapshot': {
        'snapshot_tree': {'sha256': 'c' * 64}}}))
    with (mock.patch.object(launch.loader, '_snapshot_manifest',
                            return_value={'snapshot_tree': {'sha256': 'c' * 64}}),
          mock.patch.object(launch.loader, '_pinned_snapshot_path',
                            return_value=snapshot),
          mock.patch.object(launch.loader, '_load_pinned_fp32_voxtral') as model):
        assert launch._pinned_pad_before_load(
            files, {'checkpoint_tree_sha256': 'c' * 64}) == 11
        model.assert_not_called()
        (snapshot / 'generation_config.json').write_text(
            json.dumps({'pad_token_id': 11, 'eos_token_id': 999}))
        with pytest.raises(ValueError, match='token IDs invalid'):
            launch._pinned_pad_before_load(files, {'checkpoint_tree_sha256': 'c' * 64})


def test_child_orders_gate_batch_transfer_then_model(tmp_path: Path) -> None:
    events: list[str] = []
    files = fake_files(tmp_path)
    pins = fake_pins()
    child = launch.PilotChild(files, tmp_path / 'R.json', pins, tmp_path / 'attempt')

    def pre(**_kwargs):
        events.append('gate')
        return {'fresh_host_collected_utc': '2026-01-01T00:00:00+00:00'}

    def batch(**kwargs):
        assert kwargs['device'] == 'mps'
        assert events == ['gate']
        events.append('batch_transfer')
        return {}, {}, {'transfer_device': 'mps'}

    def load(**_kwargs):
        assert events == ['gate', 'batch_transfer', 'pad', 'worker']
        events.append('model')
        return object(), object(), {'pad_token_id': 11}

    def schedule(**kwargs):
        events.append('worker')
        assert kwargs['pad'] == 11
        kwargs['load_model']()
        return {'status': 'schedule_complete_resource_unverified',
                'pilot_measurement_sha256': 'd' * 64}

    def sidecar(**_kwargs):
        assert events[-1] == 'model'
        events.append('sidecar')
        return {'status': 'schedule_complete_resource_unverified'}

    with (mock.patch('torch.backends.mps.is_available', return_value=True),
          mock.patch.object(launch.pre_pilot, 'verify_pre_pilot', side_effect=pre),
          mock.patch.object(launch.freeze, 'prepare_bound_inputs_for_child', side_effect=batch),
          mock.patch.object(launch.contract, 'verify_contract',
                            return_value={'bindings': {}}),
          mock.patch.object(launch, '_pinned_pad_before_load',
                            side_effect=lambda *_args: events.append('pad') or 11),
          mock.patch.object(launch.loader, '_load_pinned_fp32_voxtral', side_effect=load),
          mock.patch.object(launch.worker, '_execute_after_admission', side_effect=schedule),
          mock.patch.object(launch, '_write_post_host_sidecar', side_effect=sidecar)):
        assert child()['status'] == 'schedule_complete_resource_unverified'
    assert events == ['gate', 'batch_transfer', 'pad', 'worker', 'model', 'sidecar']


def test_child_rejects_loader_pad_mismatch(tmp_path: Path) -> None:
    files = fake_files(tmp_path)
    child = launch.PilotChild(files, tmp_path / 'R.json', fake_pins(), tmp_path / 'attempt')

    def schedule(**kwargs):
        kwargs['load_model']()
        raise AssertionError('unreachable')

    with (mock.patch('torch.backends.mps.is_available', return_value=True),
          mock.patch.object(launch.pre_pilot, 'verify_pre_pilot',
                            return_value={'fresh_host_collected_utc': '2026-01-01T00:00:00+00:00'}),
          mock.patch.object(launch.freeze, 'prepare_bound_inputs_for_child',
                            return_value=({}, {}, {'transfer_device': 'mps'})),
          mock.patch.object(launch.contract, 'verify_contract',
                            return_value={'bindings': {}}),
          mock.patch.object(launch, '_pinned_pad_before_load', return_value=11),
          mock.patch.object(launch.loader, '_load_pinned_fp32_voxtral',
                            return_value=(object(), object(), {'pad_token_id': 12})),
          mock.patch.object(launch.worker, '_execute_after_admission', side_effect=schedule),
          mock.patch.object(launch, '_write_post_host_sidecar') as sidecar):
        with pytest.raises(ValueError, match='pad token differs'):
            child()
        sidecar.assert_not_called()
