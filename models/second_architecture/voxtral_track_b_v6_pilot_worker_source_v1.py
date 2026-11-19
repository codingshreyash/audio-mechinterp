"""V6 measured resource-pilot source; hard HOLD until external admission freeze.

No CLI, model factory, checkpoint path, or candidate-audio reader is provided.
`run_pilot` cannot invoke a loader until an independently reviewed admission
SHA is inserted in a later source freeze. Fake tests exercise only the private
post-admission schedule with an in-memory toy model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from models.second_architecture import voxtral_track_b_answer_link_core_v1 as core


SCHEMA = 'voxtral_track_b_v6_pilot_measurement_v1'
MAX_MODEL_SECONDS = 240.0
MAX_DRIVER_BYTES = 30 * 1024**3
MAX_OUTPUT_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 12 * 1024**3
NO_GRAD_CALLS = 72
GRAD_CALLS = 12
LONGEST_ALIAS = (1353, 11135, 3313, 1079, core.EOS)
FROZEN_ADMISSION_SHA256: str | None = None  # reviewed future freeze must pin this
PROTOCOL_SHAS = {
    'v3': 'f3716897cfff30b26a4bb66d882222fc2137b0911266d76ea0a0c4319854ce0b',
    'v4': '87ef41b3c854ed23fc4cfb3f67a23f2b6cd595958a1f941bc9726288ea0d923c',
    'v5': '70d24e5d4ba341e1aaa918608d57f1ed237fd56742b0d956816f4ce636ac3115',
    'v6': '6a341f316e4056c6bc81c6e191deea3f90dcddcd81050a39c734a257d4f02b9c',
    'v6_go': '28341f87e9eb7ba8a7827460f2872d01ea17adcab3f3a36b649d4542eacae0b1',
}
HERE = Path(__file__).resolve().parent
PROTOCOL_FILES = {
    'v3': HERE / 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V3.md',
    'v4': HERE / 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V4_ERRATUM.md',
    'v5': HERE / 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V5_SEQUENTIAL_ALIAS_AMENDMENT.md',
    'v6': HERE / 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V6_RESOURCE_PILOT_AMENDMENT.md',
    'v6_go': HERE / 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V6_INDEPENDENT_REVIEW_GO.md',
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _sha256_field(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(f'{label} must be a SHA-256 hex digest')
    return value


def _read_bound(path: Path, expected: str, label: str) -> dict[str, Any]:
    _sha256_field(expected, label)
    if not path.is_file() or sha_file(path) != expected:
        raise ValueError(f'{label} bytes differ from admission')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


@dataclass(frozen=True)
class AdmissionFiles:
    admission: Path
    qa: Path
    qa_review: Path
    preflight: Path
    study_runner: Path
    pilot_auditor: Path


def verify_admission(files: AdmissionFiles) -> dict[str, Any]:
    """Fail closed before *calling* a model loader; no self-issued GO accepted."""
    if FROZEN_ADMISSION_SHA256 is None:
        raise ValueError('No independently reviewed pre-pilot admission SHA pinned in source')
    for name, path in PROTOCOL_FILES.items():
        if not path.is_file() or sha_file(path) != PROTOCOL_SHAS[name]:
            raise ValueError(f'Pinned {name} protocol bytes differ')
    gate = _read_bound(files.admission, FROZEN_ADMISSION_SHA256, 'independent pilot admission')
    if gate.get('schema') != 'voxtral_track_b_v6_external_pilot_admission_v1' or gate.get('status') != 'pilot_go':
        raise ValueError('External admission is not pilot_go')
    bindings = gate.get('bindings')
    if not isinstance(bindings, dict) or bindings.get('protocol_sha256') != PROTOCOL_SHAS:
        raise ValueError('Pilot admission protocol bindings differ')
    fixed = {'qa_sha256': files.qa, 'qa_review_sha256': files.qa_review,
             'preflight_sha256': files.preflight,
             'study_runner_sha256': files.study_runner,
             'pilot_auditor_sha256': files.pilot_auditor}
    for key, path in fixed.items():
        if not path.is_file() or sha_file(path) != _sha256_field(bindings.get(key), key):
            raise ValueError(f'{key} binding differs')
    if sha_file(Path(__file__)) != _sha256_field(bindings.get('pilot_worker_sha256'), 'pilot_worker_sha256'):
        raise ValueError('Pilot worker source differs from independent admission')
    qa = _read_bound(files.qa, bindings['qa_sha256'], '32/32 QA')
    if (qa.get('schema') != 'word_disjoint_human_qa_audit_v1' or
            qa.get('status') != 'human_qa_32_of_32_pass' or
            qa.get('human_quality_gate_passed') is not True or
            qa.get('planned_clips') != 32 or qa.get('passed_clips') != 32):
        raise ValueError('Bound QA receipt is not independently audited 32/32 GO')
    review = files.qa_review.read_text()
    if 'Decision: GO' not in review or bindings['qa_sha256'] not in review:
        raise ValueError('Independent QA evidence review does not bind exact receipt')
    preflight = _read_bound(files.preflight, bindings['preflight_sha256'], 'preflight')
    geometry = preflight.get('v6_synthetic_zero_geometry')
    geometry_digest = (sha_bytes(json.dumps(geometry, sort_keys=True,
                       separators=(',', ':')).encode()) if isinstance(geometry, dict) else None)
    if (not isinstance(geometry, dict) or
            geometry.get('longest_alias', {}).get('token_ids_with_eos') != list(LONGEST_ALIAS) or
            geometry_digest != bindings.get('synthetic_geometry_sha256')):
        raise ValueError('Bound maximum geometry or longest alias differs')
    actual_geometry = {
        'input_features_shape': geometry.get('input_features_shape'),
        'audio_placeholder_count': geometry.get('audio_placeholder_count'),
        'attention_mask_length': geometry.get('attention_mask_length'),
    }
    if (bindings.get('synthetic_processor_geometry') != actual_geometry or
            geometry.get('projector_rows_implied') != actual_geometry['audio_placeholder_count']):
        raise ValueError('Bound synthetic processor geometry differs')
    for key in ('checkpoint_tree_sha256', 'transformers_source_sha256',
                'synthetic_batch_sha256', 'old_fit_batch_sha256'):
        _sha256_field(bindings.get(key), key)
    return gate


def _suffix_hash(ids: list[int]) -> str:
    if not ids or len(ids) > core.MAX_NEW_TOKENS or any(not isinstance(x, int) or x < 0 for x in ids):
        raise ValueError('Generated suffix length/token invalid')
    return sha_bytes(struct.pack('<I', len(ids)) + b''.join(struct.pack('<q', x) for x in ids))


def _batch_sha(batch: Mapping[str, torch.Tensor]) -> str:
    fields = ('input_ids', 'attention_mask', 'input_features')
    if set(batch) != set(fields):
        raise ValueError('Pilot batch fields differ')
    raw = json.dumps({key: core.tensor_sha(batch[key]) for key in fields},
                     sort_keys=True, separators=(',', ':')).encode()
    return sha_bytes(raw)


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open('x') as out:
        json.dump(value, out, sort_keys=True, separators=(',', ':'), allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())


def _append_json(path: Path, value: dict[str, Any]) -> None:
    with path.open('a') as out:
        json.dump(value, out, sort_keys=True, separators=(',', ':'), allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())


def _clock_pair() -> tuple[float, float]:
    return time.time(), time.monotonic()


def _host_evidence(value: Mapping[str, Any], *, post: bool) -> dict[str, Any]:
    required = ('ac', 'lid_open', 'ten_minute_clean', 'raw_power_log')
    if any(key not in value for key in required) or any(value[key] is not True for key in required[:3]):
        raise ValueError('Host AC/lid/ten-minute admission absent')
    if post and value.get('during_run_clean') is not True:
        raise ValueError('Host transitioned during pilot')
    raw = value['raw_power_log']
    if not isinstance(raw, bytes) or not raw or len(raw) > 128 * 1024**2:
        raise ValueError('Raw host power log absent or unbounded')
    return {'raw_power_log_sha256': sha_bytes(raw), 'raw_power_log_bytes': len(raw),
            'ac': True, 'lid_open': True, 'ten_minute_clean': True,
            'during_run_clean': value.get('during_run_clean') if post else None,
            'raw': raw}


def _record_host(path: Path, probe: dict[str, Any]) -> dict[str, Any]:
    raw = probe.pop('raw')
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return probe


def _projector(model: Any):
    return model.model.multi_modal_projector


def _capture_generation(model: Any, batch: Mapping[str, torch.Tensor], *, pad: int,
                        force_32: bool, source: torch.Tensor | None,
                        sync: Callable[[], None]) -> tuple[str, int, torch.Tensor | None, dict[str, Any]]:
    """Natural clean captures source in the same generation; zero edits it."""
    t = int((batch['input_ids'] == 24).sum())
    if batch['input_ids'].shape[0] != 1 or t < 1:
        raise ValueError('Pilot prompt/audio placeholder geometry invalid')
    captures: list[torch.Tensor] = []
    hook_evidence = None
    kwargs = dict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                  input_features=batch['input_features'], do_sample=False, use_cache=True,
                  max_new_tokens=32, eos_token_id=core.EOS, pad_token_id=pad)
    if force_32:
        kwargs['min_new_tokens'] = 32
    sync()
    if source is None:
        def capture(_module, _args, output):
            if (not isinstance(output, torch.Tensor) or
                    output.shape != (t, core.PROJECTOR_WIDTH) or
                    output.dtype != torch.float32 or not bool(torch.isfinite(output).all())):
                raise ValueError('Natural projector capture shape/dtype invalid')
            captures.append(output.detach().clone())
            return None
        handle = _projector(model).register_forward_hook(capture)
        try:
            with torch.no_grad():
                generated = model.generate(**kwargs)
        finally:
            handle.remove()
        if len(captures) != 1:
            raise ValueError('Natural generation did not capture projector exactly once')
        source = captures[0]
        hook_evidence = {'hook_calls': 1, 'pre_hashes': [core.tensor_sha(source)],
                         'post_hashes': [core.tensor_sha(source)]}
    else:
        with torch.no_grad():
            with core.ProjectorHook(model, source, 1, t, core.Edit('zero')) as hook:
                generated = model.generate(**kwargs)
                hook_evidence = hook.evidence()
    sync()
    if (generated.ndim != 2 or generated.shape[0] != 1 or
            generated.shape[1] <= batch['input_ids'].shape[1]):
        raise ValueError('Pilot generation output invalid')
    ids = generated[0, batch['input_ids'].shape[1]:].tolist()
    if force_32 and len(ids) != 32:
        raise ValueError('Forced pilot generation did not emit 32 tokens')
    if hook_evidence['hook_calls'] != 1 or hook_evidence['pre_hashes'] != hook_evidence['post_hashes']:
        raise ValueError('Pilot source hook/prepost parity failed')
    return _suffix_hash(ids), len(ids), source, hook_evidence


def _check_output_budget(directory: Path, *, free_bytes: Callable[[Path], int]) -> None:
    total = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if total > MAX_OUTPUT_BYTES or free_bytes(directory) < MIN_FREE_BYTES:
        raise ValueError('Pilot output/free-disk cap exceeded')


def _execute_after_admission(*, load_model: Callable[[], Any],
                             old_fit_batch: Mapping[str, torch.Tensor],
                             synthetic_batch: Mapping[str, torch.Tensor],
                             pad: int, output_dir: Path, bindings: Mapping[str, Any],
                             host_probe: Callable[[bool], Mapping[str, Any]],
                             memory_probe: Callable[[], int],
                             free_bytes: Callable[[Path], int],
                             sync: Callable[[], None]) -> dict[str, Any]:
    """Private schedule for fake tests; public wrapper must verify admission first."""
    expected_geometry = bindings.get('synthetic_processor_geometry')
    actual_geometry = {
        'input_features_shape': list(synthetic_batch['input_features'].shape),
        'audio_placeholder_count': int((synthetic_batch['input_ids'] == 24).sum()),
        'attention_mask_length': int(synthetic_batch['attention_mask'].shape[-1]),
    }
    if (not isinstance(expected_geometry, dict) or actual_geometry != expected_geometry or
            pad < 0 or _batch_sha(old_fit_batch) != bindings['old_fit_batch_sha256'] or
            _batch_sha(synthetic_batch) != bindings['synthetic_batch_sha256'] or
            tuple(bindings['longest_alias_ids_with_eos']) != LONGEST_ALIAS):
        raise ValueError('Frozen pilot geometry, batch, or alias binding differs')
    if output_dir.exists():
        raise FileExistsError('V6 pilot attempt already exists; no retry')
    output_dir.mkdir(parents=True, exist_ok=False)
    calls = output_dir / 'calls.jsonl'
    with calls.open('x') as f:
        f.flush(); os.fsync(f.fileno())
    expected = {'old_fit_clean': 1, 'old_fit_zero': 1,
                'forced_clean': 1, 'forced_zero': 1,
                'score_nograd': NO_GRAD_CALLS, 'score_grad': GRAD_CALLS}
    counts = {key: 0 for key in expected}
    completed = {key: 0 for key in expected}
    try:
        pre = _record_host(output_dir / 'host_pre.raw', _host_evidence(host_probe(False), post=False))
        _check_output_budget(output_dir, free_bytes=free_bytes)
    except BaseException as exc:
        _exclusive_json(output_dir / 'pilot_failure.json',
                        {'schema': SCHEMA, 'status': 'failed_no_retry',
                         'error_type': type(exc).__name__, 'error': str(exc),
                         'planned_calls': expected, 'attempted_calls': counts,
                         'completed_calls': completed,
                         'missing_calls': expected, 'calls_sha256': sha_file(calls)})
        raise
    wall_start, mono_start = _clock_pair()
    timings: dict[str, list[float]] = {key: [] for key in counts}
    peaks: list[int] = []
    ordinal = 0

    def timed(label: str, fn: Callable[[], Any]) -> Any:
        nonlocal ordinal
        ordinal += 1
        counts[label] += 1
        _append_json(calls, {'ordinal': ordinal, 'class': label, 'phase': 'started'})
        before = time.monotonic()
        try:
            result = fn()
            sync()
            elapsed = time.monotonic() - before
            observed = int(memory_probe())
            if not (math.isfinite(elapsed) and elapsed > 0 and 0 <= observed <= MAX_DRIVER_BYTES):
                raise ValueError('Pilot call time or observed MPS driver cap invalid')
            row: dict[str, Any] = {'ordinal': ordinal, 'class': label,
                                   'phase': 'completed', 'elapsed_seconds': elapsed,
                                   'observed_postcall_mps_driver_bytes': observed}
            if label.startswith('score_'):
                score = result
                q = score['alias_logprobs'][0]
                if not math.isfinite(q) or not all(math.isfinite(v) for v in score['token_logprobs_including_eos'][0]):
                    raise ValueError('Pilot score nonfinite')
                row['score_sha256'] = sha_bytes(struct.pack('<d', q))
                row['score_finite'] = True
                row['projector_pre_sha256'] = score['hooks'][0]['pre_hashes'][0]
                row['hook_calls'] = score['hooks'][0]['hook_calls']
                if label == 'score_grad':
                    gradient = score['gradient']
                    if gradient is None or not bool(torch.isfinite(gradient).all()):
                        raise ValueError('Pilot gradient absent/nonfinite')
                    row['gradient_sha256'] = core.tensor_sha(gradient)
                    row['gradient_finite'] = True
            else:
                suffix_hash, length, _source, hook = result
                row.update({'suffix_sha256': suffix_hash, 'suffix_length': length,
                            'hook_calls': hook['hook_calls'],
                            'projector_pre_sha256': hook['pre_hashes'][0],
                            'projector_post_sha256': hook['post_hashes'][0]})
            if row['hook_calls'] != 1:
                raise ValueError('Pilot projector hook count differs')
            timings[label].append(elapsed)
            peaks.append(observed)
            completed[label] += 1
            _append_json(calls, row)
            _check_output_budget(output_dir, free_bytes=free_bytes)
            if time.monotonic() - mono_start > MAX_MODEL_SECONDS:
                raise TimeoutError('V6 pilot exceeded 240-second model-work cap')
            return result
        except BaseException as exc:
            _append_json(calls, {'ordinal': ordinal, 'class': label, 'phase': 'failed',
                                 'error_type': type(exc).__name__, 'error': str(exc)})
            raise

    try:
        model = load_model()  # only after public verify_admission succeeds
        if int(memory_probe()) > MAX_DRIVER_BYTES:
            raise MemoryError('Pilot model load exceeded 30-GiB MPS driver cap')
        old_clean = timed('old_fit_clean', lambda: _capture_generation(
            model, old_fit_batch, pad=pad, force_32=False, source=None, sync=sync))
        old_zero = timed('old_fit_zero', lambda: _capture_generation(
            model, old_fit_batch, pad=pad, force_32=False, source=old_clean[2], sync=sync))
        if old_clean[0] != old_zero[0] or old_clean[3]['pre_hashes'] != old_zero[3]['pre_hashes']:
            raise ValueError('Old-FIT natural clean/zero parity failed')
        synthetic_clean = timed('forced_clean', lambda: _capture_generation(
            model, synthetic_batch, pad=pad, force_32=True, source=None, sync=sync))
        synthetic_zero = timed('forced_zero', lambda: _capture_generation(
            model, synthetic_batch, pad=pad, force_32=True, source=synthetic_clean[2], sync=sync))
        if (synthetic_clean[0] != synthetic_zero[0] or
                synthetic_clean[3]['pre_hashes'] != synthetic_zero[3]['pre_hashes']):
            raise ValueError('Forced 32-token clean/zero parity failed')
        synthetic_source = synthetic_clean[2]
        if synthetic_source is None:
            raise ValueError('Synthetic source not captured inside forced generation')
        ref_q: float | None = None
        for _ in range(NO_GRAD_CALLS):
            result = timed('score_nograd', lambda: core.score_alias_class(
                model, synthetic_batch, (LONGEST_ALIAS,), synthetic_source,
                core.Edit('zero'), pad=pad, grad=False))
            if ref_q is None:
                ref_q = result['alias_logprobs'][0]
            elif abs(result['alias_logprobs'][0] - ref_q) > 1e-4:
                raise ValueError('Pilot no-grad score repeat parity failed')
        for _ in range(GRAD_CALLS):
            result = timed('score_grad', lambda: core.score_alias_class(
                model, synthetic_batch, (LONGEST_ALIAS,), synthetic_source,
                core.Edit('zero'), pad=pad, grad=True, gradient_rows=(0,)))
            if ref_q is None or abs(result['alias_logprobs'][0] - ref_q) > 1e-4:
                raise ValueError('Pilot grad/no-grad score parity failed')
        post = _record_host(output_dir / 'host_post.raw', _host_evidence(host_probe(True), post=True))
        wall_end, mono_end = _clock_pair()
        gap = abs((wall_end - wall_start) - (mono_end - mono_start))
        if gap > 5 or not math.isfinite(gap):
            raise ValueError('Pilot UTC/monotonic elapsed gap exceeded 5 seconds')
        if counts != expected or completed != expected:
            raise ValueError('Pilot call ledger incomplete')
        tg, ts, tb = (max(timings[key]) for key in ('forced_clean', 'score_nograd', 'score_grad'))
        tg = max(tg, max(timings['forced_zero']))
        if min(tg, ts, tb) <= 0:
            raise ValueError('Pilot class maximum missing')
        p = mono_end - mono_start
        w = 7 * tg + 72 * ts + 12 * tb
        projections = {'t_g': tg, 't_s': ts, 't_b': tb, 'P': p,
                       'per_context_1p2_W': 1.2 * w,
                       'campaign_1p2_P_plus_16W': 1.2 * (p + 16 * w)}
        feasible = projections['per_context_1p2_W'] <= 240 and projections['campaign_1p2_P_plus_16W'] <= 3840
        receipt = {
            'schema': SCHEMA,
            'status': ('schedule_complete_resource_unverified' if feasible
                       else 'complete_infeasible'),
            'bindings': dict(bindings), 'planned_calls': expected,
            'attempted_calls': counts, 'completed_calls': completed,
            'missing_calls': {k: expected[k]-counts[k] for k in expected},
            'calls_sha256': sha_file(calls), 'attempted_call_count': ordinal,
            'host_pre_sha256': sha_file(output_dir / 'host_pre.raw'),
            'host_post_sha256': sha_file(output_dir / 'host_post.raw'),
            'host_pre': pre, 'host_post': post,
            'clock_gap_seconds': gap,
            'max_observed_postcall_mps_driver_bytes': max(peaks),
            'measured_mps_high_water_present': False,
            'hard_per_call_watchdog_present': False,
            'synthetic_suffix_length': 32,
            'synthetic_input_sha256': bindings['synthetic_batch_sha256'],
            'old_fit_input_sha256': bindings['old_fit_batch_sha256'],
            'longest_alias_ids_with_eos': list(LONGEST_ALIAS),
            'timing': projections,
            'model_work_seconds': p,
            'external_post_pilot_review_required': True,
        }
        _exclusive_json(output_dir / 'pilot_measurement.json', receipt)
        _check_output_budget(output_dir, free_bytes=free_bytes)
        return {'measurement_path': str(output_dir / 'pilot_measurement.json'),
                'pilot_measurement_sha256': sha_file(output_dir / 'pilot_measurement.json'),
                'status': receipt['status']}
    except BaseException as exc:
        failure = {'schema': SCHEMA, 'status': 'failed_no_retry',
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'planned_calls': expected, 'attempted_calls': counts,
                   'completed_calls': completed,
                   'missing_calls': {k: expected[k]-counts[k] for k in expected},
                   'calls_sha256': sha_file(calls),
                   'external_post_pilot_review_required': True}
        _exclusive_json(output_dir / 'pilot_failure.json', failure)
        raise


def run_pilot(*, files: AdmissionFiles, load_model: Callable[[], Any],
              old_fit_batch: Mapping[str, torch.Tensor],
              synthetic_batch: Mapping[str, torch.Tensor], pad: int,
              output_dir: Path, host_probe: Callable[[bool], Mapping[str, Any]],
              memory_probe: Callable[[], int], free_bytes: Callable[[Path], int],
              sync: Callable[[], None]) -> dict[str, Any]:
    """Public entry remains a hard HOLD until a later reviewed admission pin."""
    gate = verify_admission(files)  # must precede load_model()
    bindings = gate['bindings']
    return _execute_after_admission(load_model=load_model,
                                    old_fit_batch=old_fit_batch,
                                    synthetic_batch=synthetic_batch, pad=pad,
                                    output_dir=output_dir, bindings=bindings,
                                    host_probe=host_probe, memory_probe=memory_probe,
                                    free_bytes=free_bytes, sync=sync)
