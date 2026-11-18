"""Source-only until external freeze: four fixed FP32 FIT local-linearity contexts.

The CLI refuses model construction without the separate reviewed campaign gate.
No fresh-v2 corpus, development, or locked-test item is opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import wave
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import voxtral_fit_pilot_capability_v1 as base
import voxtral_fit_v2_causal_first_record as first
import voxtral_fit_v2_causal_masks as mask_builder
import voxtral_v1_fit_backward_numerical_followup_v2 as prior
import voxtral_old_fit_handoff_stage_a_runner_v1 as sidecars

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL_V2 = HERE / 'VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V2.md'
PROTOCOL_V3 = HERE / 'VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V3.md'
PROTOCOL_V2_SHA = '04f99648f5483fa7f32da80515c31e3f8472511c954cd95711d8eba09bb5b42d'
PROTOCOL_V3_SHA = 'a222ce0f06a0e82b2c22bf6231e605983d726610b2f51dfbe7299de3ea8d79bd'
SELECTION_SHA = 'f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b'
MASKS_SHA = 'a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e'
BASIS_SEED = 3693039655029897049
BASIS_SHA = '10c9132454a01e034886a3e14bf9b9670e75bb3befd58300b9b45f8d9ba7751f'
CONTEXT_LABELS = ('piper_silver', 'piper_teal', 'kokoro_silver', 'kokoro_teal')
CONTEXTS = (('piper', 'silver', 'teal'), ('piper', 'teal', 'silver'),
            ('kokoro', 'silver', 'teal'), ('kokoro', 'teal', 'silver'))
WAV_SHAS = {
    ('piper', 'silver'): 'fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9',
    ('piper', 'teal'): '112ed36abf5fb2be1ce77dcfffa9d1bb6deba10b4ecb2d14fc5960b8106626e9',
    ('kokoro', 'silver'): 'bcad702fa8b55c453286cb0e83ee3336a4bb6ade5e5ac9d4a42dc82c1a97c26e',
    ('kokoro', 'teal'): '5038dbe167ae2ba6c9317f74c17bd9bb8ef7d07bbdee9da49f6dc0731e119365',
}
BATCH_SHAS = {
    'input_ids': '915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a',
    'attention_mask': '9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282',
    ('piper', 'silver'): '36ec6b38ba8202821e6cabf8e4047ff493a58dce826279385eecb86f6b325d48',
    ('piper', 'teal'): '6879a7f915bc01a7d9c117f96dffe119d259e5d947e00f309b9aafdf93b66c4c',
    ('kokoro', 'silver'): '5b8fd75b50812f0f6866470fecddcb96d67819da26473dbd7ea78e93bc5f74bc',
    ('kokoro', 'teal'): '8e3aaa1a3857682797cae8931c20093abd7fc1e2e9ac2e2647c3aa7df4888221',
}
SOURCE_DEPENDENCIES = {
    'fixture_sha256': Path(base.__file__).resolve(),
    'fit_first_record_sha256': Path(first.__file__).resolve(),
    'mask_builder_sha256': Path(mask_builder.__file__).resolve(),
    'numerical_followup_sha256': Path(prior.__file__).resolve(),
    'sidecar_writer_sha256': Path(sidecars.__file__).resolve(),
}
RUNS = HERE / 'runs/voxtral_fp32_nonbeaver_linearity_v1'
PREFLIGHT_RECEIPT = HERE / 'VOXTRAL_FP32_NONBEAVER_CPU_PREFLIGHT_V1.json'
SITES = ('encoder31', 'projector')
MODES = ('G1', 'N0')
ALPHAS = (0.00625, -0.00625, 0.025, -0.025, 0.1, -0.1)
CYCLES = (ALPHAS, tuple(reversed(ALPHAS)))
MODEL_SECONDS = 300
MAX_DRIVER_BYTES = 30 * 1024**3
MAX_OUTPUT_BYTES = 2 * 1024**3
MAX_WORKER_BYTES = MAX_OUTPUT_BYTES - 24 * 1024**2
MAX_CAMPAIGN_BYTES = 8 * 1024**3
MIN_FREE_BYTES = 12 * 1024**3
LOGIT_WIDTH = 131072
TARGET_WIDTH = 3072
EXPECTED_SHAPES = {'encoder31': (1, 1500, 1280), 'projector': (375, 3072),
                   'lm_input': (1, 396, 3072)}


def sha(path: Path) -> str:
    return base.sha(path)


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and
            tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype and
            first.tensor_sha(left) == first.tensor_sha(right))


def checked_tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype, label: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
            value.dtype != dtype or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{label} shape, dtype, or finiteness differs')
    return value


def basis_cpu() -> torch.Tensor:
    payload = hashlib.sha256(bytes.fromhex(SELECTION_SHA) + b'color05_fp32_four_probe_v2').digest()
    seed = int.from_bytes(payload[:8], 'big') % 2**63
    if seed != BASIS_SEED:
        raise ValueError('Frozen basis seed differs')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    signs = torch.randint(0, 2, (TARGET_WIDTH, 4), generator=generator, dtype=torch.int8)
    matrix = signs.to(torch.float64).mul_(2).sub_(1)
    q, r = torch.linalg.qr(matrix, mode='reduced')
    diagonal = torch.diag(r)
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(diagonal).all()) or bool((diagonal == 0).any()):
        raise ValueError('Fixed basis QR is degenerate')
    q *= torch.sign(diagonal).unsqueeze(0)
    q = q.contiguous()
    if not torch.allclose(q.T @ q, torch.eye(4, dtype=torch.float64), atol=1e-12, rtol=0):
        raise ValueError('Fixed basis orthonormality differs')
    digest = hashlib.sha256(b'float64[3072,4]\0' +
                            q.numpy().astype('<f8', copy=False).tobytes(order='C')).hexdigest()
    if digest != BASIS_SHA:
        raise ValueError('Frozen basis bytes differ')
    return q


def patched_schedule(engine: str) -> tuple[tuple[str, str, int, float, str], ...]:
    if engine not in ('piper', 'kokoro'):
        raise ValueError('Unknown engine')
    sites = SITES if engine == 'piper' else tuple(reversed(SITES))
    plan = tuple((site, mode, cycle, alpha, label)
                 for site in sites for mode in MODES
                 for cycle, order in enumerate(CYCLES)
                 for alpha in order for label in ('zero', 'edit'))
    if len(plan) != 96:
        raise AssertionError('Patched forward schedule changed')
    return plan


def _pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), 'rb') as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(),
                stream.getcomptype()) != (1, 16000, 2, 'NONE'):
            raise ValueError('Frozen FIT WAV must be 16k mono PCM16')
        nframes = stream.getnframes()
        return stream.readframes(nframes), nframes


def _selected_records() -> dict[str, dict]:
    if sha(PROTOCOL_V2) != PROTOCOL_V2_SHA or sha(PROTOCOL_V3) != PROTOCOL_V3_SHA:
        raise ValueError('Independently reviewed prospective protocol bytes differ')
    if sha(first.SELECTION) != SELECTION_SHA or sha(first.MASKS) != MASKS_SHA:
        raise ValueError('Frozen FIT selection or mask bytes differ')
    first.verify_frozen()
    masks = json.loads(first.MASKS.read_text())
    selection = json.loads(first.SELECTION.read_text())
    index = json.loads(first.INDEX.read_text())
    if mask_builder.make_manifest(index, selection) != masks:
        raise ValueError('Independent FIT mask recomputation differs')
    records = {r['engine']: r for r in masks['records'] if r['pair_id'] == 'color_05'
               and r['phrasing_id'] == 'fit_chosen'}
    if set(records) != {'piper', 'kokoro'}:
        raise ValueError('Exact color_05 FIT records missing')
    for engine, record in records.items():
        if record['words'] != ['silver', 'teal'] or len(record['clips']) != 2:
            raise ValueError('FIT words or clips changed')
        raw = {}
        for clip in record['clips']:
            word = clip['word']
            path = (ROOT / clip['wav_path']).resolve()
            expected_root = (ROOT / 'eval/audio_jlens/corpus_v1/rendered/clips').resolve()
            if not path.is_relative_to(expected_root) or sha(path) != WAV_SHAS[(engine, word)] or clip['wav_sha256'] != WAV_SHAS[(engine, word)]:
                raise ValueError('Selected FIT WAV path or bytes differ')
            raw[word], frames = _pcm(path)
            if frames != clip['frames']:
                raise ValueError('Selected FIT WAV length differs')
        start, stop = record['word_slot_start_sample'], record['word_slot_end_sample_exclusive']
        if (start <= 0 or stop <= start or stop * 2 > len(raw['silver']) or
                raw['silver'][:start*2] != raw['teal'][:start*2] or
                raw['silver'][stop*2:] != raw['teal'][stop*2:] or
                raw['silver'][start*2:stop*2] == raw['teal'][start*2:stop*2]):
            raise ValueError('Frozen matched PCM identity or word difference changed')
        expected_counts = (60, 15) if engine == 'piper' else (73, 18)
        for site, nrows in zip(SITES, expected_counts):
            rows = record['site_maps'][site]['whole_word_rows']
            if (len(rows) != nrows or rows != sorted(set(rows)) or
                    any(row < 0 or row >= EXPECTED_SHAPES[site][-2] for row in rows)):
                raise ValueError('Selected FIT whole-word row mask changed')
    return records


def preflight_cpu() -> tuple[dict, dict]:
    """Recompute exact four FIT batches, masks, basis, source and checkpoint pins."""
    from transformers import AutoProcessor
    import transformers.models.llama.modeling_llama as modeling_llama
    records = _selected_records()
    if not base.validate_transformers_source():
        raise ValueError('Installed Voxtral source validation absent')
    if sha(Path(modeling_llama.__file__)) != prior.SOURCE_SHAS['modeling_llama']:
        raise ValueError('Installed Llama source bytes differ')
    snapshot = base.validate_snapshot(json.loads(first.INDEX.read_text()))
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    if processor.tokenizer.eos_token_id != 2 or int(processor.audio_token_id) != 24:
        raise ValueError('Pinned processor token IDs changed')
    basis = basis_cpu()
    batches: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    receipt_batches = []
    for engine in ('piper', 'kokoro'):
        built = first.make_batches(processor, records[engine])
        for word in ('silver', 'teal'):
            value = built[word]
            if not isinstance(value, Mapping) or set(value) != {'input_ids', 'attention_mask', 'input_features'}:
                raise ValueError('Processor batch field set changed')
            specs = {'input_ids': ((1,396), torch.int64, BATCH_SHAS['input_ids']),
                     'attention_mask': ((1,396), torch.int64, BATCH_SHAS['attention_mask']),
                     'input_features': ((1,128,3000), torch.float32, BATCH_SHAS[(engine,word)])}
            batch = {}
            hashes = {}
            for key, (shape,dtype,expected) in specs.items():
                tensor = _cpu(checked_tensor(value[key], shape, dtype, key))
                digest = first.tensor_sha(tensor)
                if digest != expected:
                    raise ValueError(f'{engine}/{word} {key} hash differs')
                batch[key], hashes[key] = tensor, digest
            if not torch.equal(batch['attention_mask'], torch.ones_like(batch['attention_mask'])) or int((batch['input_ids'] == 24).sum()) != 375:
                raise ValueError('Frozen audio placeholder/mask structure differs')
            batches[(engine,word)] = batch
            receipt_batches.append({'engine':engine, 'word':word, 'tensor_sha256':hashes,
                                    'shape':{k:list(t.shape) for k,t in batch.items()},
                                    'dtype':{k:str(t.dtype) for k,t in batch.items()},
                                    'audio_placeholders':375})
        if not bitwise_equal(batches[(engine,'silver')]['input_ids'], batches[(engine,'teal')]['input_ids']):
            raise ValueError('Matched prompt token IDs differ')
    receipt = {'schema':'voxtral_fp32_nonbeaver_cpu_preflight_v1',
               'status':'verified_source_only_before_model_load',
               'protocol_v2_sha256':PROTOCOL_V2_SHA, 'protocol_v3_sha256':PROTOCOL_V3_SHA,
               'selection_sha256':SELECTION_SHA, 'masks_sha256':MASKS_SHA,
               'snapshot':snapshot,
               'installed_sources':{name:sha(path) for name,path in SOURCE_DEPENDENCIES.items()},
               'basis_seed':BASIS_SEED, 'basis_sha256':BASIS_SHA,
               'torch_version':torch.__version__, 'numpy_version':np.__version__,
               'prompt':base.PROMPT, 'prompt_input_ids':batches[('piper','silver')]['input_ids'][0].tolist(),
               'batch_tensors':receipt_batches,
               'contexts':[{'index':i,'label':CONTEXT_LABELS[i], 'engine':engine,
                            'recipient':recipient,'donor':donor,
                            'wav_sha256':WAV_SHAS[(engine,recipient)],
                            'word_rows':{site:records[engine]['site_maps'][site]['whole_word_rows']
                                         for site in SITES}}
                           for i,(engine,recipient,donor) in enumerate(CONTEXTS)]}
    prepared = {'processor':processor, 'records':records, 'batches':batches,
                'basis':basis, 'receipt':receipt}
    return prepared, receipt


def verify_cpu_preflight_receipt(receipt: dict) -> None:
    """Fail closed by recomputing CPU evidence, not trusting a receipt's claims."""
    _, actual = preflight_cpu()
    if receipt != actual:
        raise ValueError('Committed CPU preflight receipt differs from recomputation')


def _json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _line(path: Path, value: dict) -> None:
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _source_module(model: Any, site: str):
    return (model.model.audio_tower.layers[31] if site == 'encoder31'
            else model.model.multi_modal_projector)


def trace_forward(model: Any, batch: dict[str, torch.Tensor], *, site: str | None,
                  rows: list[int] | None, direction: torch.Tensor | None,
                  alpha: float | None, mode: str, archive_full: bool,
                  device: str = 'mps') -> tuple[dict, torch.Tensor | None, torch.Tensor | None]:
    """Return CPU evidence and optional live site leaf/LM29 terminal for VJP.

    `site=None` is the one passive N0 donor capture. Every site patch,
    including a clean/zero patch, uses the same detach/leaf/clone topology.
    """
    if site not in (None, *SITES) or mode not in MODES:
        raise ValueError('Trace site, mode, or edit descriptor invalid')
    if site is None and (rows is not None or direction is not None or alpha is not None or
                         mode != 'N0' or not archive_full):
        raise ValueError('Donor trace must be a full passive N0 capture')
    if site is not None and (not rows or direction is None or alpha is None or
                             not math.isfinite(float(alpha))):
        raise ValueError('Patch trace needs frozen rows, direction and alpha')
    source_pre: list[torch.Tensor] = []
    source_post: list[torch.Tensor] = []
    passive_encoder: list[torch.Tensor] = []
    projector: list[torch.Tensor] = []
    lm_input: list[torch.Tensor] = []
    lm0_pre: list[torch.Tensor] = []
    hits: dict[str, list[torch.Tensor]] = {'lm0':[], 'lm14':[], 'lm29':[]}
    source_leaf: list[torch.Tensor] = []
    edit_info: list[dict] = []
    flags: list[dict] = []
    handles = []
    audio_token = int(model.model.config.audio_token_id)
    positions = torch.nonzero(batch['input_ids'][0] == audio_token, as_tuple=False).flatten().tolist()
    if audio_token != 24 or len(positions) != 375:
        raise ValueError('Audio placeholder mapping changed')
    if site is None:
        def encoder_hook(_module, _args, output):
            passive_encoder.append(checked_tensor(output, EXPECTED_SHAPES['encoder31'],
                                                  torch.float32, 'donor encoder31'))
        handles.append(_source_module(model,'encoder31').register_forward_hook(encoder_hook))
    else:
        shape = EXPECTED_SHAPES[site]
        if (rows != sorted(set(rows)) or any(i < 0 or i >= shape[-2] for i in rows) or
                not isinstance(direction, torch.Tensor) or direction.dtype != torch.float32 or
                tuple(direction.shape) != (shape[-1],) or not bool(torch.isfinite(direction).all())):
            raise ValueError('Frozen site rows or direction changed')
        direction_device = direction.to(device)
        def source_hook(_module, _args, output):
            value = checked_tensor(output, shape, torch.float32, 'site prepatch source')
            source_pre.append(value)
            leaf = value.detach().clone().requires_grad_(mode == 'G1')
            source_leaf.append(leaf)
            selected = leaf[0, rows] if site == 'encoder31' else leaf[rows]
            edit = prior.realized_edit(selected, direction_device, float(alpha), torch.float32)
            patched = leaf.clone()
            if site == 'encoder31':
                patched[0, rows] = edit['edited']
            else:
                patched[rows] = edit['edited']
            source_post.append(patched)
            edit_info.append(edit)
            flags.append({'grad_enabled':bool(torch.is_grad_enabled()),
                          'leaf_requires_grad':bool(leaf.requires_grad)})
            return patched
        handles.append(_source_module(model,site).register_forward_hook(source_hook))
    def projector_hook(_module, _args, output):
        projector.append(checked_tensor(output, EXPECTED_SHAPES['projector'],
                                        torch.float32, 'post-projector'))
    handles.append(_source_module(model,'projector').register_forward_hook(projector_hook))
    def lm_input_hook(_module, _args, kwargs):
        value = checked_tensor(kwargs.get('inputs_embeds'), EXPECTED_SHAPES['lm_input'],
                               torch.float32, 'LM input')
        lm_input.append(value)
    handles.append(model.model.language_model.register_forward_pre_hook(lm_input_hook,with_kwargs=True))
    def lm0_pre_hook(_module,args):
        value = checked_tensor(args[0], EXPECTED_SHAPES['lm_input'],
                               torch.float32, 'LM0 pre-input')
        lm0_pre.append(value)
        flags.append({'lm0_grad_enabled':bool(torch.is_grad_enabled()),
                      'lm0_input_requires_grad':bool(value.requires_grad),
                      'lm_input_requires_grad':bool(lm_input[0].requires_grad)})
    handles.append(model.model.language_model.layers[0].register_forward_pre_hook(lm0_pre_hook))
    for index in (0,14,29):
        def terminal_hook(_module,_args,output, *, layer=index):
            tensor=checked_tensor(output, EXPECTED_SHAPES['lm_input'], torch.float32,
                                  f'LM{layer} output')
            hits[f'lm{layer}'].append(tensor[0,-1])
        handles.append(model.model.language_model.layers[index].register_forward_hook(terminal_hook))
    try:
        gpu_batch = {key:value.to(device) for key,value in batch.items()}
        with torch.enable_grad() if mode == 'G1' else torch.no_grad():
            output = model(**gpu_batch, use_cache=False, logits_to_keep=1)
        if device == 'mps':
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    expected_one = [projector,lm_input,lm0_pre,*hits.values()]
    if site is None:
        expected_one.append(passive_encoder)
    else:
        expected_one.extend((source_pre,source_post,source_leaf,edit_info))
    if any(len(values)!=1 for values in expected_one):
        raise ValueError('Passive/patch hook invocation count changed')
    if not bitwise_equal(lm_input[0],lm0_pre[0]) or not bitwise_equal(lm_input[0][0,positions],projector[0]):
        raise ValueError('Projector/LM-input/LM0 handoff differs')
    logits=checked_tensor(output.logits[0,-1],(LOGIT_WIDTH,),torch.float32,'first-answer logits')
    if site is None:
        save={'kind':'donor','mode':'N0','encoder31_full':_cpu(passive_encoder[0]),
              'projector_full':_cpu(projector[0]),'lm_input_full':_cpu(lm_input[0]),
              'lm_terminal':{key:_cpu(values[0]) for key,values in hits.items()},
              'first_answer_logits':_cpu(logits), 'audio_positions':positions,
              'runtime_flags':{'grad_enabled':False,'lm0_grad_enabled':False,
                               'lm_input_requires_grad':bool(lm_input[0].requires_grad)}}
        return save,None,None
    if len(flags)!=2 or flags[0] != {'grad_enabled':mode=='G1','leaf_requires_grad':mode=='G1'} or flags[1]['lm0_grad_enabled']!=(mode=='G1'):
        raise ValueError('Runtime grad/leaf flags differ from mode')
    pre=source_pre[0]; post=source_post[0]; edit=edit_info[0]
    selected_pre=pre[0,rows] if site=='encoder31' else pre[rows]
    selected_post=post[0,rows] if site=='encoder31' else post[rows]
    selected_edit_bitwise = bitwise_equal(_cpu(selected_post-selected_pre),_cpu(edit['effective']))
    zero_source_bitwise = bool(alpha != 0 or bitwise_equal(pre,post))
    save={'site':site,'mode':mode,'alpha':float(alpha),
          'source_pre_rows':_cpu(selected_pre),'source_post_rows':_cpu(selected_post),
          'effective_rows':_cpu(edit['effective']),
          'source_pre_sha256':first.tensor_sha(pre),
          'source_post_sha256':first.tensor_sha(post),
          'projector_sha256':first.tensor_sha(projector[0]),
          'lm_input_sha256':first.tensor_sha(lm_input[0]),
          'lm0_pre_sha256':first.tensor_sha(lm0_pre[0]),
          'lm_terminal':{key:_cpu(values[0]) for key,values in hits.items()},
          'first_answer_logits':_cpu(logits),'audio_positions':positions,
          'runtime_flags':{**flags[0],**flags[1]},
          'runtime_checks':{'selected_edit_bitwise':selected_edit_bitwise,
                            'zero_source_bitwise':zero_source_bitwise,
                            'projector_lm_input_bitwise':True,
                            'lm_input_lm0_pre_bitwise':True},
          'edit_metrics':{'effective_norm':edit['effective_norm'],
                          'mean_fidelity_ratio':edit['mean_fidelity_ratio'],
                          'row_nonuniformity_ratio':edit['row_nonuniformity_ratio'],
                          'unchanged_fraction':edit['unchanged_fraction']}}
    if archive_full:
        save.update({'source_pre_full':_cpu(pre),'source_post_full':_cpu(post),
                     'projector_full':_cpu(projector[0]),
                     'lm_input_full':_cpu(lm_input[0])})
    return save,(source_leaf[0] if mode=='G1' else None),(hits['lm29'][0] if mode=='G1' else None)


def exact_tensor_payload_estimate(engine: str) -> dict[str,int]:
    """Conservative retained FP32 bytes before serialization overhead."""
    if engine not in ('piper','kokoro'):
        raise ValueError('Unknown engine')
    nenc,nproj=(60,15) if engine=='piper' else (73,18)
    enc=1500*1280*4; proj=375*3072*4; lm=396*3072*4; logits=LOGIT_WIDTH*4
    zero=24*(2*enc+proj+lm+logits)+24*(3*proj+lm+logits)
    references=2*(2*enc+proj+lm+logits)+2*(3*proj+lm+logits)
    donor=enc+2*proj+lm+logits
    edited=24*3*(nenc*1280*4+nproj*3072*4)+48*logits
    terminal=96*3*3072*4
    gradients=4*(nenc*1280+nproj*3072)*4
    batches=2*(396*8*2+128*3000*4)
    total=zero+references+donor+edited+terminal+gradients+batches
    if total>=MAX_WORKER_BYTES:
        raise ValueError('Pre-run tensor payload estimate exceeds worker cap')
    return {'zero_full':zero,'clean_references':references,'donor':donor,
            'edited_rows_logits':edited,'lm_terminal':terminal,'gradients':gradients,
            'batch_tensors':batches,'estimated_tensor_payload_bytes':total,
            'worker_cap_bytes':MAX_WORKER_BYTES,'headroom_bytes':MAX_WORKER_BYTES-total}


STATUS = 'completed_fp32_nonbeaver_local_linearity_raw_only'
SCOPE = 'old_fit_color05_fp32_local_linearity_numerical_only'
COUNTS = {'patch_forwards':96,'vjps':8,'g1_clean_graphs':2,
          'n0_clean_references':2,'donor_captures':1,'natural_generations':2}


def _full_parity(left: dict, right: dict) -> dict[str,bool]:
    """Compare retained bytes at all preregistered clean/zero boundaries."""
    keys=('source_pre_full','source_post_full','projector_full','lm_input_full',
          'first_answer_logits')
    result={key:bitwise_equal(left[key],right[key]) for key in keys}
    result['lm29']=bitwise_equal(left['lm_terminal']['lm29'],right['lm_terminal']['lm29'])
    return result


def _record_problem(problems: list[dict], *, stage: str, descriptor: dict, checks: dict) -> None:
    failed=[name for name,ok in checks.items() if not ok]
    if failed:
        problems.append({'stage':stage,'descriptor':descriptor,'failed_checks':failed})


def _selected_state(record: dict, site: str, rows: list[int], *, full_key: str='source_pre_full') -> torch.Tensor:
    value=record[full_key]
    return value[0,rows] if site=='encoder31' else value[rows]


def _load_model_fp32(*, device: str='mps') -> Any:
    from transformers import VoxtralForConditionalGeneration
    if device!='mps':
        raise ValueError('Real checkpoint launch is MPS-only; fake CPU tests use injected model')
    model=VoxtralForConditionalGeneration.from_pretrained(
        str(base.SNAPSHOT),local_files_only=True,dtype=torch.float32,
        low_cpu_mem_usage=True,device_map='mps',attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    if model.training or model.model.language_model.training:
        raise ValueError('FP32 study requires eval mode')
    return model


def run_context(index: int, attempt_id: str, freeze_sha256: str, *,
                admitted: dict, model_factory=None, device: str='mps') -> dict:
    """One context only; parent launcher owns campaign order and hard watchdog."""
    if index not in range(4) or not attempt_id or len(freeze_sha256)!=64:
        raise ValueError('Worker context/attempt/freeze invalid')
    if not isinstance(admitted, dict) or 'prior_model_work_seconds' not in admitted:
        raise ValueError('Reviewed campaign admission missing')
    label=CONTEXT_LABELS[index]
    output=RUNS/label/attempt_id
    if output.exists():
        raise FileExistsError('Context output already exists; no retry')
    started=time.monotonic()
    prepared,receipt=preflight_cpu()
    verify_cpu_preflight_receipt(json.loads(PREFLIGHT_RECEIPT.read_text()))
    if receipt != json.loads(PREFLIGHT_RECEIPT.read_text()):
        raise ValueError('Committed CPU preflight receipt differs')
    engine,recipient,donor_word=CONTEXTS[index]
    record=prepared['records'][engine]
    rows={site:record['site_maps'][site]['whole_word_rows'] for site in SITES}
    batch=prepared['batches'][(engine,recipient)]
    donor_batch=prepared['batches'][(engine,donor_word)]
    basis=prepared['basis']
    estimate=exact_tensor_payload_estimate(engine)
    output.mkdir(parents=True,exist_ok=False)
    progress=output/'progress.jsonl'
    progress.touch(exist_ok=False)
    stack=ExitStack()
    memory=None
    model=None
    counts={key:0 for key in COUNTS}

    def mark(phase: str) -> None:
        elapsed=time.monotonic()-started
        peak=int(torch.mps.driver_allocated_memory()) if device=='mps' else 0
        if memory is not None:
            peak=max(peak,memory.driver_max)
        if elapsed>=MODEL_SECONDS or peak>=MAX_DRIVER_BYTES or sidecars.directory_bytes(output)>=MAX_WORKER_BYTES:
            raise ValueError('Model time, MPS memory, or worker output cap reached')
        _line(progress,{'phase':phase,'elapsed_seconds':elapsed,'mps_driver_bytes':peak,
                        'counts':counts.copy()})

    try:
        meta={'schema':'voxtral_fp32_nonbeaver_linearity_meta_v1',
              'status':'started_source_only_reviewed_context',
              'scope':SCOPE,'context':{'index':index,'label':label,'engine':engine,
                                      'recipient':recipient,'donor':donor_word},
              'attempt_id':attempt_id,'freeze_sha256':freeze_sha256,
              'protocol_v2_sha256':PROTOCOL_V2_SHA,'protocol_v3_sha256':PROTOCOL_V3_SHA,
              'runner_sha256':sha(Path(__file__)),
              'preflight_receipt_sha256':sha(PREFLIGHT_RECEIPT),
              'source_dependencies':{key:sha(path) for key,path in SOURCE_DEPENDENCIES.items()},
              'selection_sha256':SELECTION_SHA,'masks_sha256':MASKS_SHA,
              'basis_sha256':BASIS_SHA,'wav_sha256':{
                  recipient:WAV_SHAS[(engine,recipient)],donor_word:WAV_SHAS[(engine,donor_word)]},
              'batch_sha256':{name:{key:first.tensor_sha(value) for key,value in data.items()}
                              for name,data in (('recipient',batch),('donor',donor_batch))},
              'rows':rows,'patched_schedule':patched_schedule(engine),
              'planned_counts':COUNTS,'tensor_payload_estimate':estimate,
              'model_seconds_cap':MODEL_SECONDS,'max_mps_driver_bytes':MAX_DRIVER_BYTES,
              'max_worker_bytes':MAX_WORKER_BYTES,'max_output_bytes':MAX_OUTPUT_BYTES,
              'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),
              'transformers_version':__import__('transformers').__version__,
              'platform':platform.platform(),'device':device,'fresh_v2_corpus_opened':False}
        _json_exclusive(output/'meta.json',meta)
        mark('preflight_complete_before_model_load')
        if device=='mps':
            if not torch.backends.mps.is_available():
                raise RuntimeError('MPS unavailable')
            recommended=int(torch.mps.recommended_max_memory())
            if recommended<=0:
                raise RuntimeError('MPS recommended memory unavailable')
            torch.mps.set_per_process_memory_fraction(min(1.0,MAX_DRIVER_BYTES/recommended))
            memory=stack.enter_context(base.MemorySampler(torch))
        if model_factory is None:
            model=_load_model_fp32(device=device)
        else:
            if device!='cpu':
                raise ValueError('Injected model factory is CPU-test-only')
            model=model_factory()
        model.requires_grad_(False)
        if device=='mps':
            layout=prior.parameter_layout(model,'fp32')
        else:
            layout={'arm':'fp32','valid':True,'test_only_cpu_factory':True}
        _json_exclusive(output/'model_layout.json',layout)
        if not layout['valid']:
            raise ValueError('FP32 model parameter layout differs')
        mark('model_loaded')
        generations={}
        for word,task_batch in ((recipient,batch),(donor_word,donor_batch)):
            if device=='mps':
                natural=prior.generate(model,task_batch,torch.float32,passive=False)
                suffix=natural['suffix_ids']
            else:
                suffix=[]
            if not isinstance(suffix,list) or len(suffix)>32 or any(type(i) is not int or i<0 for i in suffix):
                raise ValueError('Natural generation suffix invalid')
            generations[word]={'suffix_ids':suffix}
            counts['natural_generations']+=1
            mark(f'natural_{word}')
        donor_trace,_,_=trace_forward(model,donor_batch,site=None,rows=None,
                                      direction=None,alpha=None,mode='N0',
                                      archive_full=True,device=device)
        counts['donor_captures']+=1
        mark('donor_capture')
        sites={}
        delta={}
        problems=[]
        for site in (SITES if engine=='piper' else tuple(reversed(SITES))):
            selected_rows=rows[site]
            width=EXPECTED_SHAPES[site][-1]
            zero_direction=torch.zeros(width,dtype=torch.float32)
            n0_clean,_,_=trace_forward(model,batch,site=site,rows=selected_rows,
                                       direction=zero_direction,alpha=0.0,mode='N0',
                                       archive_full=True,device=device)
            counts['n0_clean_references']+=1
            mark(f'n0_clean_{site}')
            donor_source=(donor_trace['encoder31_full'] if site=='encoder31'
                          else donor_trace['projector_full'])
            donor_rows=(donor_source[0,selected_rows] if site=='encoder31'
                        else donor_source[selected_rows])
            recipient_rows=_selected_state(n0_clean,site,selected_rows)
            direction=(donor_rows.float().mean(0)-recipient_rows.float().mean(0)).contiguous()
            if not bool(torch.isfinite(direction).all()) or float(torch.linalg.vector_norm(direction))<=0:
                raise ValueError(f'{site} donor-recipient direction is zero or nonfinite')
            delta[site]=direction
            g1_clean,source_leaf,target=trace_forward(model,batch,site=site,
                    rows=selected_rows,direction=direction,alpha=0.0,mode='G1',
                    archive_full=True,device=device)
            counts['g1_clean_graphs']+=1
            mark(f'g1_clean_{site}_before_vjp')
            if source_leaf is None or target is None or not target.requires_grad:
                raise ValueError('G1 clean source-to-LM29 graph disconnected')
            gradients=[]
            for k in range(4):
                scalar=(target.float()*basis[:,k].to(device=device,dtype=torch.float32)).sum()
                grad,=torch.autograd.grad(scalar,source_leaf,retain_graph=k<3,
                                          allow_unused=False)
                selected=grad[0,selected_rows] if site=='encoder31' else grad[selected_rows]
                checked_tensor(selected,(len(selected_rows),width),torch.float32,'source VJP')
                gradients.append(_cpu(selected))
                counts['vjps']+=1
                mark(f'vjp_{site}_{k}')
            del source_leaf,target,grad,scalar
            sites[site]={'n0_clean':n0_clean,'g1_clean':g1_clean,
                         'gradients':torch.stack(gradients)}
        forwards=[]
        for ordinal,(site,mode,cycle,alpha,label_kind) in enumerate(patched_schedule(engine)):
            edit_alpha=0.0 if label_kind=='zero' else alpha
            trace,_,_=trace_forward(model,batch,site=site,rows=rows[site],
                                    direction=delta[site],alpha=edit_alpha,mode=mode,
                                    archive_full=label_kind=='zero',device=device)
            descriptor={'ordinal':ordinal,'site':site,'mode':mode,'cycle':cycle,
                        'alpha':alpha,'label':label_kind}
            forwards.append({**descriptor,'trace':trace})
            counts['patch_forwards']+=1
            ref=sites[site]['g1_clean' if mode=='G1' else 'n0_clean']
            checks={'prepatch_full_hash':trace['source_pre_sha256']==ref['source_pre_sha256'],
                    'prepatch_selected_rows':bitwise_equal(trace['source_pre_rows'],ref['source_pre_rows']),
                    'runtime_selected_edit':trace['runtime_checks']['selected_edit_bitwise']}
            if label_kind=='zero':
                checks.update(_full_parity(ref,trace))
                checks['zero_source_bitwise']=trace['runtime_checks']['zero_source_bitwise']
            else:
                checks.update({'realized_norm':math.isfinite(trace['edit_metrics']['effective_norm']) and
                               trace['edit_metrics']['effective_norm']>0,
                               'mean_fidelity':trace['edit_metrics']['mean_fidelity_ratio']<=.1,
                               'row_uniformity':trace['edit_metrics']['row_nonuniformity_ratio']<=.1})
            _record_problem(problems,stage='paired_forward',descriptor=descriptor,checks=checks)
            mark(f'patched_{ordinal:02d}_{site}_{mode}_{cycle}_{label_kind}_{alpha}')
        if counts!=COUNTS or len(forwards)!=96:
            raise ValueError('Fixed model-work counts differ')
        mark('all_model_work_complete')
        payload={'schema':'voxtral_fp32_nonbeaver_linearity_tensors_v1',
                 'context':meta['context'],'basis':basis,
                 'batches':{'recipient':batch,'donor':donor_batch},
                 'rows':rows,'delta':delta,'donor':donor_trace,'sites':sites,
                 'forwards':forwards,'generations':{
                     'recipient':generations[recipient], 'donor':generations[donor_word]}}
        remaining=MAX_WORKER_BYTES-sidecars.directory_bytes(output)-1_048_576
        if remaining<=0:
            raise OSError('Tensor sidecar lacks frozen output headroom')
        with (output/'details.pt').open('xb') as raw:
            limited=sidecars.LimitedWriter(raw,remaining)
            torch.save(payload,limited)
            limited.flush()
            os.fsync(limited.fileno())
        details_sha,details_path=sidecars.freeze_content_addressed(output/'details.pt')
        mark('details_saved')
        result={'status':STATUS,'scope':SCOPE,'context':meta['context'],
                'attempt_id':attempt_id,'freeze_sha256':freeze_sha256,
                'protocol_v2_sha256':PROTOCOL_V2_SHA,'protocol_v3_sha256':PROTOCOL_V3_SHA,
                'preflight_receipt_sha256':sha(PREFLIGHT_RECEIPT),
                'meta_sha256':sha(output/'meta.json'),
                'progress_sha256':sha(progress),
                'model_layout_sha256':sha(output/'model_layout.json'),
                'details_sha256':details_sha,'details_path':details_path,
                'details_bytes':(output/details_path).stat().st_size,
                'planned_counts':COUNTS,'actual_counts':counts,
                'invalidities':problems,'elapsed_seconds':time.monotonic()-started,
                'peak_mps_driver_bytes':max(memory.driver_max,int(torch.mps.driver_allocated_memory()))
                    if device=='mps' else 0,
                'output_bytes_before_result':sidecars.directory_bytes(output)}
        _json_exclusive(output/'result.json',result)
        if sidecars.directory_bytes(output)>=MAX_WORKER_BYTES:
            raise OSError('Final worker directory exceeded cap')
        return result
    except Exception as exc:
        error={'status':'invalid_fp32_nonbeaver_context_no_retry','context_index':index,
               'context_label':label,'attempt_id':attempt_id,
               'freeze_sha256':freeze_sha256,'error_type':type(exc).__name__,
               'error':str(exc),'elapsed_seconds':time.monotonic()-started,
               'progress_sha256':sha(progress) if progress.is_file() else None}
        if not (output/'result.json').exists() and not (output/'error.json').exists():
            _json_exclusive(output/'error.json',error)
        raise
    finally:
        stack.close()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--run-context',type=int)
    parser.add_argument('--attempt-id')
    parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.preflight:
        if args.attempt_id or args.freeze_sha256:
            parser.error('--preflight takes no live attempt arguments')
        _,receipt=preflight_cpu()
        print(json.dumps(receipt,indent=2,allow_nan=False))
        return
    if args.attempt_id is None or args.freeze_sha256 is None:
        parser.error('Live context requires gate-provided attempt ID and freeze SHA')
    import voxtral_fp32_nonbeaver_linearity_launch_gate_v1 as gate
    admission=gate.verify_worker_start(args.run_context,args.attempt_id,args.freeze_sha256)
    result=run_context(args.run_context,args.attempt_id,args.freeze_sha256,
                       admitted=admission)
    print(json.dumps({'status':result['status'],'context':result['context'],
                      'details_sha256':result['details_sha256']}))


if __name__=='__main__':
    main()
