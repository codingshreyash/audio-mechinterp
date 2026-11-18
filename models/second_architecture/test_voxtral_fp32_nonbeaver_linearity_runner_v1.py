"""CPU-only tests for the prospective non-beaver runner; no checkpoint weights."""
from __future__ import annotations

import json
import gc
import math
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
import voxtral_fp32_nonbeaver_linearity_runner_v1 as runner


class FakeTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(nn.Identity() for _ in range(32))
    def forward(self,features):
        value=torch.arange(1500*1280,dtype=torch.float32).reshape(1,1500,1280)/1000000
        for layer in self.layers:
            value=layer(value)
        return value

class FakeProjector(nn.Module):
    def forward(self,value):
        rows=value[0,::4,:]
        return torch.cat((rows,torch.full((375,3072-1280),-0.0,dtype=torch.float32)),dim=-1)

class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(nn.Identity() for _ in range(30))
        self.layers[0]=FakeMix()
    def forward(self,*,inputs_embeds,use_cache=False,attention_mask=None,position_ids=None):
        value=inputs_embeds
        for layer in self.layers:
            value=layer(value)
        return SimpleNamespace(last_hidden_state=value)

class FakeMix(nn.Module):
    def forward(self,value):
        return value + value.mean(dim=1,keepdim=True)

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        core=nn.Module()
        core.audio_tower=FakeTower()
        core.multi_modal_projector=FakeProjector()
        core.language_model=FakeLanguage()
        core.config=SimpleNamespace(audio_token_id=24)
        self.model=core
    def forward(self,input_ids,attention_mask,input_features,use_cache=False,logits_to_keep=1):
        encoded=self.model.audio_tower(input_features)
        project=self.model.multi_modal_projector(encoded)
        positions=torch.nonzero(input_ids[0]==24).flatten()
        embeds=torch.zeros((1,396,3072),dtype=torch.float32)
        embeds[0,positions]=project
        hidden=self.model.language_model(inputs_embeds=embeds,attention_mask=attention_mask,
                                         use_cache=use_cache).last_hidden_state
        logits=torch.cat((hidden[:,:,:1],torch.zeros((1,396,131071))),dim=-1)
        return SimpleNamespace(logits=logits)


def fake_batch():
    ids=torch.zeros((1,396),dtype=torch.int64)
    ids[0,:375]=24
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
            'input_features':torch.zeros((1,128,3000),dtype=torch.float32)}


class RunnerCPUChecks(unittest.TestCase):
    def test_fixed_basis_schedule_and_budget(self):
        self.assertEqual(runner.basis_cpu().shape,(3072,4))
        self.assertEqual(len(runner.patched_schedule('piper')),96)
        self.assertEqual(len(runner.patched_schedule('kokoro')),96)
        self.assertEqual(runner.patched_schedule('piper')[0],
                         ('encoder31','G1',0,.00625,'zero'))
        self.assertEqual(runner.patched_schedule('kokoro')[0],
                         ('projector','G1',0,.00625,'zero'))
        self.assertEqual(runner.patched_schedule('piper')[-1],
                         ('projector','N0',1,.00625,'edit'))
        for engine,expected in [('piper',1306674560),('kokoro',1322601856)]:
            estimate=runner.exact_tensor_payload_estimate(engine)
            self.assertEqual(estimate['estimated_tensor_payload_bytes'],expected)
            self.assertEqual(estimate['zero_full'],1069744128)
            self.assertEqual(estimate['recipient_passive'],17715200)
            self.assertEqual(estimate['basis_and_deltas'],115712)
            self.assertGreater(estimate['headroom_bytes'],0)
        self.assertEqual(runner.COUNTS['recipient_passive_captures'],1)

    def test_signed_zero_is_not_bitwise_equal(self):
        self.assertFalse(runner.bitwise_equal(torch.tensor([0.0]),torch.tensor([-0.0])))
        self.assertTrue(runner.bitwise_equal(torch.tensor([-0.0]),torch.tensor([-0.0])))

    def test_zero_identity_preserves_negative_zero_with_either_signed_direction(self):
        model=FakeModel().eval()
        batch=fake_batch()
        rows=[0]
        signed_direction=torch.full((3072,),-0.01,dtype=torch.float32)
        arbitrary_zero=torch.zeros_like(signed_direction)
        clean,_,_=runner.trace_forward(model,batch,site='projector',rows=rows,
                    direction=signed_direction,alpha=0.0,mode='N0',
                    archive_full=True,device='cpu')
        zero,_,_=runner.trace_forward(model,batch,site='projector',rows=rows,
                    direction=signed_direction,alpha=0.0,mode='N0',
                    archive_full=True,device='cpu')
        wrong_clean,_,_=runner.trace_forward(model,batch,site='projector',rows=rows,
                    direction=arbitrary_zero,alpha=0.0,mode='N0',
                    archive_full=True,device='cpu')
        self.assertTrue(all(runner._full_parity(clean,zero).values()))
        self.assertTrue(all(runner._zero_identity_checks(clean).values()))
        self.assertTrue(all(runner._zero_identity_checks(zero).values()))
        self.assertTrue(all(runner._zero_identity_checks(wrong_clean).values()))
        self.assertTrue(runner.bitwise_equal(clean['source_post_full'],
                                              wrong_clean['source_post_full']))
        self.assertTrue(runner.bitwise_equal(clean['source_pre_full'],
                                              clean['source_post_full']))

    def test_real_cpu_preflight_exact_four_batches(self):
        prepared,receipt=runner.preflight_cpu()
        self.assertEqual([row['label'] for row in receipt['contexts']],list(runner.CONTEXT_LABELS))
        self.assertEqual(len(receipt['batch_tensors']),4)
        self.assertEqual(receipt['basis_sha256'],runner.BASIS_SHA)
        self.assertEqual(receipt['protocol_v5_sha256'],runner.PROTOCOL_V5_SHA)
        self.assertEqual(receipt,json.loads(runner.PREFLIGHT_RECEIPT.read_text()))
        self.assertEqual(receipt['modeling_llama_sha256'],
                         runner.prior.SOURCE_SHAS['modeling_llama'])
        self.assertEqual(set(prepared['batches']),set((e,w) for e in ('piper','kokoro') for w in ('silver','teal')))
        self.assertTrue(all(row['audio_placeholders']==375 for row in receipt['batch_tensors']))
        mutated=json.loads(json.dumps(receipt))
        mutated['batch_tensors'][0]['tensor_sha256']['input_features']='0'*64
        with mock.patch.object(runner,'preflight_cpu',return_value=(prepared,receipt)):
            with self.assertRaises(ValueError):
                runner.verify_cpu_preflight_receipt(mutated)

    def test_full_size_serialization_receipt_is_under_both_caps(self):
        path=runner.HERE/'VOXTRAL_FP32_NONBEAVER_BYTE_BUDGET_V1.json'
        receipt=json.loads(path.read_text())
        self.assertEqual(receipt['status'],'verified_source_only_serialized_byte_budget')
        self.assertEqual(receipt['protocol_v5_sha256'],runner.PROTOCOL_V5_SHA)
        self.assertEqual(receipt['runner_sha256'],runner.sha(Path(runner.__file__)))
        self.assertEqual(receipt['max_output_bytes_per_context'],runner.MAX_OUTPUT_BYTES)
        self.assertEqual(receipt['max_campaign_bytes'],runner.MAX_CAMPAIGN_BYTES)
        self.assertEqual([row['engine'] for row in receipt['measurements']],['piper','kokoro'])
        self.assertEqual(receipt['max_serialized_bytes'],
                         max(row['serialized_bytes'] for row in receipt['measurements']))
        self.assertLess(receipt['max_serialized_bytes'],runner.MAX_WORKER_BYTES)
        self.assertLess(receipt['four_context_sidecar_bytes'],runner.MAX_CAMPAIGN_BYTES)

    def test_fake_hook_paths_and_g1_source_vjp(self):
        model=FakeModel().eval()
        batch=fake_batch()
        donor,leaf,target=runner.trace_forward(model,batch,site=None,rows=None,
                  direction=None,alpha=None,mode='N0',archive_full=True,device='cpu')
        self.assertIsNone(leaf)
        self.assertIsNone(target)
        self.assertEqual(tuple(donor['encoder31_full'].shape),(1,1500,1280))
        rows=[0,1,2]
        direction=torch.ones(3072,dtype=torch.float32)*-.01
        clean,leaf,target=runner.trace_forward(model,batch,site='projector',rows=rows,
                  direction=direction,alpha=0.0,mode='G1',archive_full=True,device='cpu')
        self.assertTrue(target.requires_grad)
        self.assertTrue(clean['runtime_flags']['leaf_requires_grad'])
        self.assertTrue(clean['runtime_flags']['lm_input_requires_grad'])
        self.assertTrue(clean['runtime_flags']['lm0_input_requires_grad'])
        self.assertTrue(clean['runtime_checks']['zero_source_bitwise'])
        grad,=torch.autograd.grad(target[0],leaf)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().sum()),0)
        zero,_,_=runner.trace_forward(model,batch,site='projector',rows=rows,
                  direction=direction,alpha=0.0,mode='G1',archive_full=True,device='cpu')
        self.assertTrue(all(runner._full_parity(clean,zero).values()))
        edited,_,_=runner.trace_forward(model,batch,site='projector',rows=rows,
                  direction=direction,alpha=.1,mode='N0',archive_full=False,device='cpu')
        self.assertNotIn('source_pre_full',edited)
        self.assertGreater(float(edited['effective_rows'].abs().sum()),0)
        self.assertFalse(edited['runtime_flags']['grad_enabled'])
        self.assertFalse(edited['runtime_flags']['lm_input_requires_grad'])
        self.assertFalse(edited['runtime_flags']['lm0_input_requires_grad'])

    def test_g1_paired_graph_can_be_released_before_next_forward(self):
        model=FakeModel().eval()
        batch=fake_batch()
        trace,leaf,target=runner.trace_forward(model,batch,site='projector',rows=[0],
                  direction=torch.ones(3072,dtype=torch.float32)*.01,
                  alpha=.00625,mode='G1',archive_full=False,device='cpu')
        self.assertTrue(target.requires_grad)
        self.assertNotIn('source_pre_full',trace)
        weak_target=weakref.ref(target)
        del leaf,target
        gc.collect()
        self.assertIsNone(weak_target())

    def test_trace_rejects_missing_patch_descriptor(self):
        with self.assertRaises(ValueError):
            runner.trace_forward(FakeModel(),fake_batch(),site='encoder31',rows=[0],
                direction=None,alpha=0.0,mode='G1',archive_full=True,device='cpu')
        with self.assertRaises(ValueError):
            runner.trace_forward(FakeModel(),fake_batch(),site=None,rows=[0],
                direction=None,alpha=None,mode='N0',archive_full=True,device='cpu')

if __name__=='__main__':
    unittest.main()
