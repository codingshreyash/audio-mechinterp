"""Source-only, full-size serialization probe for the fixed V4 sidecar schema.

Every synthetic tensor owns distinct zero-filled storage. No checkpoint,
processor, model forward, audio corpus, or experiment output is opened.
"""
from __future__ import annotations

import gc
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

import voxtral_fp32_nonbeaver_linearity_runner_v1 as runner


def _tensor(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)


def _terminals():
    return {name: _tensor((runner.TARGET_WIDTH,)) for name in ('lm0', 'lm14', 'lm29')}


def _passive(kind):
    return {'kind': kind, 'mode': 'N0',
            'encoder31_full': _tensor(runner.EXPECTED_SHAPES['encoder31']),
            'projector_full': _tensor(runner.EXPECTED_SHAPES['projector']),
            'lm_input_full': _tensor(runner.EXPECTED_SHAPES['lm_input']),
            'lm_terminal': _terminals(),
            'first_answer_logits': _tensor((runner.LOGIT_WIDTH,)),
            'audio_positions': list(range(375)),
            'runtime_flags': {'grad_enabled': False, 'lm0_grad_enabled': False,
                              'lm_input_requires_grad': False}}


def _trace(site, mode, alpha, row_count, full):
    shape = runner.EXPECTED_SHAPES[site]
    row_shape = (row_count, shape[-1])
    result = {'site': site, 'mode': mode, 'alpha': alpha,
              'source_pre_rows': _tensor(row_shape),
              'source_post_rows': _tensor(row_shape),
              'effective_rows': _tensor(row_shape),
              'source_pre_sha256': '0'*64, 'source_post_sha256': '0'*64,
              'projector_sha256': '0'*64, 'lm_input_sha256': '0'*64,
              'lm0_pre_sha256': '0'*64,
              'lm_terminal': _terminals(),
              'first_answer_logits': _tensor((runner.LOGIT_WIDTH,)),
              'audio_positions': list(range(375)),
              'runtime_flags': {'grad_enabled': mode == 'G1',
                                'leaf_requires_grad': mode == 'G1',
                                'lm0_grad_enabled': mode == 'G1',
                                'lm0_input_requires_grad': mode == 'G1',
                                'lm_input_requires_grad': mode == 'G1'},
              'runtime_checks': {'selected_edit_bitwise': True,
                                 'zero_source_bitwise': True,
                                 'projector_lm_input_bitwise': True,
                                 'lm_input_lm0_pre_bitwise': True},
              'edit_metrics': {'effective_norm': 0.0,
                               'mean_fidelity_ratio': 0.0,
                               'row_nonuniformity_ratio': 0.0,
                               'unchanged_fraction': 1.0}}
    if full:
        result.update({'source_pre_full': _tensor(shape),
                       'source_post_full': _tensor(shape),
                       'projector_full': _tensor(runner.EXPECTED_SHAPES['projector']),
                       'lm_input_full': _tensor(runner.EXPECTED_SHAPES['lm_input'])})
    return result


def _batch():
    return {'input_ids': _tensor((1,396), torch.int64),
            'attention_mask': _tensor((1,396), torch.int64),
            'input_features': _tensor((1,128,3000))}


def synthetic_payload(index):
    engine, recipient, donor = runner.CONTEXTS[index]
    row_counts = {'encoder31': 60 if engine == 'piper' else 73,
                  'projector': 15 if engine == 'piper' else 18}
    rows = {site: list(range(count)) for site, count in row_counts.items()}
    sites = {site: {'n0_clean': _trace(site, 'N0', 0.0, row_counts[site], True),
                    'g1_clean': _trace(site, 'G1', 0.0, row_counts[site], True),
                    'gradients': _tensor((4,row_counts[site],runner.EXPECTED_SHAPES[site][-1]))}
             for site in runner.SITES}
    forwards = []
    for ordinal,(site,mode,cycle,alpha,label) in enumerate(runner.patched_schedule(engine)):
        forwards.append({'ordinal': ordinal, 'site': site, 'mode': mode,
                         'cycle': cycle, 'alpha': alpha, 'label': label,
                         'trace': _trace(site,mode,0.0 if label == 'zero' else alpha,
                                         row_counts[site],label == 'zero')})
    return {'schema': 'voxtral_fp32_nonbeaver_linearity_tensors_v1',
            'context': {'index': index, 'label': runner.CONTEXT_LABELS[index],
                        'engine': engine, 'recipient': recipient, 'donor': donor},
            'basis': _tensor((runner.TARGET_WIDTH,4), torch.float64),
            'batches': {'recipient': _batch(), 'donor': _batch()},
            'rows': rows,
            'delta': {site: _tensor((runner.EXPECTED_SHAPES[site][-1],))
                      for site in runner.SITES},
            'donor': _passive('donor'),
            'recipient_passive': _passive('recipient'),
            'sites': sites, 'forwards': forwards,
            'generations': {'recipient': {'suffix_ids': list(range(32))},
                            'donor': {'suffix_ids': list(range(32))}}}


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def probe(*, keep_piper_path: Path | None = None):
    if shutil.disk_usage(runner.HERE).free < 4*1024**3:
        raise OSError('Full-size synthetic serialization needs 4 GiB free')
    results = []
    with tempfile.TemporaryDirectory(prefix='voxtral_fp32_nonbeaver_budget_') as temporary:
        for index in (0,2):
            payload = synthetic_payload(index)
            actual_tensor_bytes = tensor_bytes(payload)
            estimate = runner.exact_tensor_payload_estimate(runner.CONTEXTS[index][0])
            if actual_tensor_bytes > estimate['estimated_tensor_payload_bytes']:
                raise ValueError('Shape estimate undercounts synthetic retained tensors')
            path = Path(temporary)/f'{runner.CONTEXT_LABELS[index]}.pt'
            with path.open('xb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            serialized = path.stat().st_size
            if serialized >= runner.MAX_WORKER_BYTES:
                raise OSError('Full-size synthetic sidecar exceeds worker output cap')
            results.append({'engine': runner.CONTEXTS[index][0],
                            'context_index': index,
                            'tensor_bytes': actual_tensor_bytes,
                            'estimated_tensor_payload_bytes': estimate['estimated_tensor_payload_bytes'],
                            'serialized_bytes': serialized,
                            'serialization_overhead_bytes': serialized-actual_tensor_bytes,
                            'worker_headroom_after_sidecar_bytes': runner.MAX_WORKER_BYTES-serialized})
            del payload
            gc.collect()
            if index == 0 and keep_piper_path is not None:
                keep_piper_path.parent.mkdir(parents=True,exist_ok=True)
                if keep_piper_path.exists():
                    raise FileExistsError('Synthetic sidecar destination already exists')
                os.replace(path,keep_piper_path)
            else:
                path.unlink()
    campaign_sidecar_bytes = 2*sum(row['serialized_bytes'] for row in results)
    if campaign_sidecar_bytes >= runner.MAX_CAMPAIGN_BYTES:
        raise OSError('Synthetic four-context sidecars exceed campaign cap')
    return {'schema': 'voxtral_fp32_nonbeaver_byte_budget_v1',
            'status': 'verified_source_only_serialized_byte_budget',
            'protocol_v4_sha256': runner.PROTOCOL_V4_SHA,
            'protocol_v5_sha256': runner.PROTOCOL_V5_SHA,
            'runner_sha256': runner.sha(Path(runner.__file__)),
            'torch_version': str(torch.__version__),
            'max_output_bytes_per_context': runner.MAX_OUTPUT_BYTES,
            'max_worker_bytes_per_context': runner.MAX_WORKER_BYTES,
            'max_campaign_bytes': runner.MAX_CAMPAIGN_BYTES,
            'contexts_per_engine': 2,
            'measurements': results,
            'max_serialized_bytes': max(row['serialized_bytes'] for row in results),
            'four_context_sidecar_bytes': campaign_sidecar_bytes,
            'campaign_headroom_after_sidecars_bytes': runner.MAX_CAMPAIGN_BYTES-campaign_sidecar_bytes}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-piper-path',type=Path)
    args=parser.parse_args()
    print(json.dumps(probe(keep_piper_path=args.keep_piper_path),indent=2,allow_nan=False))
